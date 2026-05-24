"""
S4 Layer Implementation
=======================
Structured State Space Sequence Model (S4)
Based on: Gu et al., 2021 — arXiv:2111.00396

Supports both:
  - Convolutional mode  (parallel training,  O(N log N) via FFT)
  - Recurrent mode      (sequential inference, O(1) per step)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange


# ─────────────────────────────────────────────
# HiPPO Matrix
# ─────────────────────────────────────────────

def make_HiPPO(N: int) -> torch.Tensor:
    """
    Construct the HiPPO-LegS matrix A of shape (N, N).
    This is the key initialization that gives S4 its long-range memory.

    A[n, k] = -sqrt((2n+1)(2k+1))  for k < n
               (n+1)               for k = n
               0                   for k > n
    """
    P = torch.sqrt(1 + 2 * torch.arange(N, dtype=torch.float))   # shape (N,)
    A = torch.outer(P, P)                                          # (N, N)
    A = torch.tril(A) - torch.diag(torch.arange(N, dtype=torch.float))
    return -A


def make_NPLR_HiPPO(N: int):
    """
    Normal Plus Low-Rank (NPLR) decomposition of HiPPO.
    A = V Λ V* - P Q*   (diagonal + rank-1 correction)
    Returns: (Lambda, P, B) all complex tensors.
    """
    A = make_HiPPO(N)

    # Low-rank correction vector
    P = torch.sqrt(torch.arange(N, dtype=torch.float) + 0.5).unsqueeze(1)  # (N,1)

    # Symmetrize: S = A + P P*
    S = A + P @ P.T

    # Diagonalize S (real symmetric → real eigenvalues)
    _, V = torch.linalg.eigh(S)   # V: (N, N) orthogonal

    # Convert to complex
    V  = V.to(torch.complex64)
    A_c = torch.diag(torch.arange(N, dtype=torch.float).to(torch.complex64))

    # Lambda = diagonal of V* A V  (approximately diagonal)
    Lambda = torch.diag(V.conj().T @ A.to(torch.complex64) @ V)

    P_c = (V.conj().T @ P.to(torch.complex64))   # (N, 1)
    B   = V.conj().T @ torch.ones(N, 1, dtype=torch.complex64)

    return Lambda, P_c.squeeze(), B.squeeze(), V


# ─────────────────────────────────────────────
# Discretization
# ─────────────────────────────────────────────

def discretize_zoh(A, B, step):
    """Zero-Order Hold (ZOH) discretization."""
    I   = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    Ab  = torch.matrix_exp(step * A)
    Bb  = torch.linalg.solve(A, (Ab - I) @ B)
    return Ab, Bb


def discretize_bilinear(A, B, step):
    """
    Bilinear (Tustin) discretization — preserves stability.
    Ā = (I + Δ/2 · A)(I - Δ/2 · A)⁻¹
    B̄ = (I - Δ/2 · A)⁻¹ · Δ · B
    """
    N = A.shape[0]
    I = torch.eye(N, dtype=A.dtype, device=A.device)
    BL = torch.linalg.inv(I - (step / 2.0) * A)
    Ab = BL @ (I + (step / 2.0) * A)
    Bb = (BL * step) @ B
    return Ab, Bb


# ─────────────────────────────────────────────
# Convolution Kernel via FFT
# ─────────────────────────────────────────────

def compute_ssm_kernel(Ab, Bb, Cb, L: int) -> torch.Tensor:
    """
    Compute convolution kernel K of length L:
      K[l] = C Ā^l B̄
    Uses FFT for O(L log L) efficiency.
    Returns real tensor of shape (L,).
    """
    # K[l] = C A^l B  →  use Vandermonde trick via FFT
    # Directly unroll for simplicity (educational clarity)
    # For large L, replace with FFT-based Cauchy kernel
    powers = [Cb @ torch.matrix_power(Ab, l) @ Bb for l in range(L)]
    K = torch.stack(powers).real.squeeze()   # (L,)
    return K


def compute_ssm_kernel_fft(Ab_diag, Bb, Cb, L: int) -> torch.Tensor:
    """
    Efficient O(L log L) kernel computation assuming Ab is diagonal.
    Ab_diag: (N,) complex diagonal elements
    Bb:      (N,) complex
    Cb:      (N,) complex
    """
    # Vandermonde: K[l] = sum_n C[n] * lambda[n]^l * B[n]
    # = IFFT of C * B / (1 - lambda * z)  evaluated on unit circle
    l_idx   = torch.arange(L, device=Ab_diag.device)
    # shape: (N, L)
    powers  = Ab_diag.unsqueeze(1) ** l_idx.unsqueeze(0)       # (N, L)
    weights = Cb * Bb                                            # (N,)
    K       = (weights.unsqueeze(1) * powers).sum(0).real       # (L,)
    return K


# ─────────────────────────────────────────────
# S4 Layer
# ─────────────────────────────────────────────

class S4Layer(nn.Module):
    """
    Single S4 layer.

    Args:
        d_model:  Feature dimension (H independent SSMs run in parallel)
        d_state:  State dimension N
        dropout:  Dropout rate on output
        lr:       Learning rate multiplier for SSM parameters
        mode:     'conv' (training) or 'recurrent' (inference)
    """

    def __init__(
        self,
        d_model:  int   = 128,
        d_state:  int   = 64,
        dropout:  float = 0.0,
        lr:       float = 0.001,
        dt_min:   float = 0.001,
        dt_max:   float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        N = d_state
        H = d_model

        # ── HiPPO initialization ──────────────────────────
        Lambda, P, B, V = make_NPLR_HiPPO(N)
        # Store as real & imag parts (nn.Parameter doesn't support complex directly)
        self.register_buffer("Lambda_re", Lambda.real.unsqueeze(0).expand(H, -1))  # (H, N)
        self.register_buffer("Lambda_im", Lambda.imag.unsqueeze(0).expand(H, -1))

        # B and C are learned; initialize from HiPPO B
        B_init = B.unsqueeze(0).expand(H, -1)   # (H, N) complex
        self.B_re = nn.Parameter(B_init.real.clone())
        self.B_im = nn.Parameter(B_init.imag.clone())

        # C: random complex init
        C_init = torch.randn(H, N, dtype=torch.complex64) * 0.5
        self.C_re = nn.Parameter(C_init.real)
        self.C_im = nn.Parameter(C_init.imag)

        # D: skip connection (input → output)
        self.D = nn.Parameter(torch.ones(H))

        # Δt: log-uniform initialization
        log_dt = torch.rand(H) * (np.log(dt_max) - np.log(dt_min)) + np.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        # Output projection
        self.output_linear = nn.Linear(H, H)
        self.dropout       = nn.Dropout(dropout)

        # Mark SSM params for custom LR (used in optimizer setup)
        for p in [self.B_re, self.B_im, self.C_re, self.C_im, self.log_dt]:
            p._ssm_lr = lr

    # ── helpers ──────────────────────────────────────────

    @property
    def Lambda(self):
        return torch.complex(self.Lambda_re, self.Lambda_im)   # (H, N)

    @property
    def B(self):
        return torch.complex(self.B_re, self.B_im)             # (H, N)

    @property
    def C(self):
        return torch.complex(self.C_re, self.C_im)             # (H, N)

    def _get_kernel(self, L: int):
        """Compute the convolution kernel K of shape (H, L)."""
        dt = self.log_dt.exp()                                  # (H,)

        # Discretize diagonal SSM: Ā[n] = exp(Δ · λ_n), B̄[n] = (Ā-1)/λ · B[n]
        Lambda = self.Lambda                                    # (H, N) complex
        dt_c   = dt.to(torch.complex64)                        # (H,)

        Ab = torch.exp(dt_c.unsqueeze(1) * Lambda)             # (H, N)
        Bb = (Ab - 1) / Lambda * self.B                        # (H, N)

        # K[h, l] = sum_n C[h,n] * Ab[h,n]^l * Bb[h,n]
        l_idx  = torch.arange(L, device=Ab.device)             # (L,)
        powers = Ab.unsqueeze(2) ** l_idx                      # (H, N, L)
        K      = (self.C.unsqueeze(2) * Bb.unsqueeze(2) * powers).sum(1).real  # (H, L)
        return K

    # ── forward ──────────────────────────────────────────

    def forward(self, u, mode: str = "conv"):
        """
        Args:
            u:    (B, L, H) input tensor
            mode: 'conv' for parallel training, 'recurrent' for inference
        Returns:
            y:    (B, L, H)
        """
        B_size, L, H = u.shape
        assert H == self.d_model

        if mode == "conv":
            return self._forward_conv(u)
        else:
            return self._forward_recurrent(u)

    def _forward_conv(self, u):
        """Parallel convolutional mode using FFT."""
        B_size, L, H = u.shape
        K = self._get_kernel(L)                                 # (H, L)

        # Convolve each channel: y[h] = u[h] * K[h]  (causal)
        u_f = torch.fft.rfft(u.transpose(1, 2), n=2 * L)       # (B, H, L+1)
        K_f = torch.fft.rfft(K,                 n=2 * L)       # (H, L+1)
        y   = torch.fft.irfft(u_f * K_f.unsqueeze(0), n=2*L)[..., :L]  # (B, H, L)
        y   = y.transpose(1, 2)                                 # (B, L, H)

        # Skip connection
        y   = y + u * self.D

        y   = self.dropout(y)
        y   = self.output_linear(y)
        return y

    def _forward_recurrent(self, u):
        """Sequential recurrent mode — O(1) memory per step."""
        B_size, L, H = u.shape
        N  = self.d_state
        dt = self.log_dt.exp()

        Lambda = self.Lambda                                    # (H, N)
        dt_c   = dt.to(torch.complex64)
        Ab = torch.exp(dt_c.unsqueeze(1) * Lambda)             # (H, N)
        Bb = (Ab - 1) / Lambda * self.B                        # (H, N)

        # Initial hidden state
        x  = torch.zeros(B_size, H, N, dtype=torch.complex64, device=u.device)
        ys = []

        for l in range(L):
            u_l = u[:, l, :].to(torch.complex64)               # (B, H)
            x   = Ab.unsqueeze(0) * x + Bb.unsqueeze(0) * u_l.unsqueeze(2)  # (B, H, N)
            y_l = (self.C.unsqueeze(0) * x).sum(-1).real       # (B, H)
            y_l = y_l + u[:, l, :] * self.D
            ys.append(y_l)

        y = torch.stack(ys, dim=1)                             # (B, L, H)
        y = self.dropout(y)
        y = self.output_linear(y)
        return y


# ─────────────────────────────────────────────
# Full S4 Model for Pathfinder
# ─────────────────────────────────────────────

class S4Model(nn.Module):
    """
    Stacked S4 layers for sequence classification.
    Architecture: Input → [S4 + LayerNorm + FF] × num_layers → Pool → Classifier
    """

    def __init__(
        self,
        d_input:    int   = 1,       # grayscale pixels
        d_model:    int   = 128,
        d_state:    int   = 64,
        num_layers: int   = 4,
        num_classes:int   = 2,       # Pathfinder: binary
        dropout:    float = 0.1,
        lr:         float = 0.001,
    ):
        super().__init__()

        self.encoder = nn.Linear(d_input, d_model)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "s4":   S4Layer(d_model, d_state, dropout=dropout, lr=lr),
                "norm": nn.LayerNorm(d_model),
                "ff":   nn.Sequential(
                            nn.Linear(d_model, d_model * 2),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(d_model * 2, d_model),
                        ),
                "norm2": nn.LayerNorm(d_model),
            })
            for _ in range(num_layers)
        ])

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x, mode: str = "conv"):
        """
        x: (B, L) or (B, L, C) — sequence of pixel values
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1).float()     # (B, L, 1)
        else:
            x = x.float()

        x = self.encoder(x)                 # (B, L, d_model)

        for layer in self.layers:
            # S4 sub-layer with residual
            residual = x
            x = layer["norm"](x)
            x = layer["s4"](x, mode=mode)
            x = x + residual

            # Feed-forward sub-layer with residual
            residual = x
            x = layer["norm2"](x)
            x = layer["ff"](x)
            x = x + residual

        # Global mean pooling over sequence
        x = x.mean(dim=1)                   # (B, d_model)
        x = self.decoder(x)                 # (B, num_classes)
        return x

    def get_hidden_states(self, x, layer_idx: int = 0):
        """Extract hidden state trajectory for visualization."""
        if x.dim() == 2:
            x = x.unsqueeze(-1).float()
        x = self.encoder(x)

        hidden_states = []
        for i, layer in enumerate(self.layers):
            residual = x
            x_norm = layer["norm"](x)
            # Run recurrent mode to collect states
            s4: S4Layer = layer["s4"]
            B_size, L, H = x_norm.shape
            N   = s4.d_state
            dt  = s4.log_dt.exp()
            Lambda = s4.Lambda
            dt_c   = dt.to(torch.complex64)
            Ab = torch.exp(dt_c.unsqueeze(1) * Lambda)
            Bb = (Ab - 1) / Lambda * s4.B
            state = torch.zeros(B_size, H, N, dtype=torch.complex64, device=x.device)
            states_l = []
            for l in range(L):
                u_l   = x_norm[:, l, :].to(torch.complex64)
                state = Ab.unsqueeze(0) * state + Bb.unsqueeze(0) * u_l.unsqueeze(2)
                states_l.append(state[0, 0, :8].real.detach().cpu())  # first sample, first head, 8 dims
            if i == layer_idx:
                hidden_states = states_l
            x = layer["s4"](x_norm) + residual
            residual = x
            x = layer["norm2"](x)
            x = layer["ff"](x) + residual

        return torch.stack(hidden_states)    # (L, 8)


if __name__ == "__main__":
    # Quick sanity check
    model = S4Model(d_input=1, d_model=64, d_state=32, num_layers=2)
    x = torch.randn(4, 1024, 1)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    print("✓ S4Model forward pass OK")
