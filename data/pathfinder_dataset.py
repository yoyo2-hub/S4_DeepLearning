"""
Pathfinder Dataset Loader
=========================
Loads the Pathfinder task from the Long Range Arena (LRA) benchmark.

Dataset: https://github.com/google-research/long-range-arena
Paper:   Tay et al., 2020 — arXiv:2011.04006

Two versions:
  - Pathfinder-32:  32×32  images → 1024-step sequences
  - Pathfinder-128: 128×128 images → 16384-step sequences  (the hard one S4 solves)

The task: given a 2D grid with a dotted path between two circles and
visual distractors, predict whether the path connects the endpoints (1) or not (0).
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import urllib.request
import tarfile


# ─────────────────────────────────────────────
# Synthetic Pathfinder (for quick local testing)
# ─────────────────────────────────────────────

class SyntheticPathfinderDataset(Dataset):
    """
    Synthetic Pathfinder dataset for development/testing.
    Generates simple paths on a grid without downloading LRA data.

    This is a simplified version — for real benchmarks download LRA.
    """

    def __init__(
        self,
        n_samples:   int  = 2000,
        img_size:    int  = 32,
        seed:        int  = 42,
        difficulty:  str  = "easy",   # 'easy' | 'hard'
    ):
        super().__init__()
        self.n_samples  = n_samples
        self.img_size   = img_size
        self.seq_len    = img_size * img_size
        self.difficulty = difficulty

        rng = np.random.default_rng(seed)
        self.data, self.labels = self._generate(rng)

    def _draw_circle(self, img, cx, cy, r=2):
        """Draw a filled circle marker."""
        H, W = img.shape
        for y in range(max(0, cy - r), min(H, cy + r + 1)):
            for x in range(max(0, cx - r), min(W, cx + r + 1)):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                    img[y, x] = 1.0

    def _draw_path(self, img, points, width=1):
        """Draw a dotted path through waypoints."""
        H, W = img.shape
        for i in range(len(points) - 1):
            x0, y0 = points[i]
            x1, y1 = points[i + 1]
            steps   = max(abs(x1 - x0), abs(y1 - y0)) * 2
            if steps == 0:
                continue
            for t in range(steps):
                frac = t / steps
                x    = int(round(x0 + frac * (x1 - x0)))
                y    = int(round(y0 + frac * (y1 - y0)))
                # Dotted: only draw every other pixel
                if t % 2 == 0:
                    for dy in range(-width + 1, width):
                        for dx in range(-width + 1, width):
                            if 0 <= y + dy < H and 0 <= x + dx < W:
                                img[y + dy, x + dx] = 0.8

    def _generate_one(self, rng, connected: bool):
        """Generate a single Pathfinder image."""
        S   = self.img_size
        img = np.zeros((S, S), dtype=np.float32)

        margin = 4
        # Two endpoint circles
        c1 = (rng.integers(margin, S - margin), rng.integers(margin, S // 2))
        c2 = (rng.integers(margin, S - margin), rng.integers(S // 2, S - margin))

        if connected:
            # Generate a winding path from c1 to c2
            n_waypoints = 4 if self.difficulty == "easy" else 8
            waypoints   = [c1]
            for i in range(n_waypoints):
                frac = (i + 1) / (n_waypoints + 1)
                wx   = int(c1[0] + frac * (c2[0] - c1[0]) + rng.integers(-S // 5, S // 5))
                wy   = int(c1[1] + frac * (c2[1] - c1[1]) + rng.integers(-S // 5, S // 5))
                wx   = np.clip(wx, margin, S - margin)
                wy   = np.clip(wy, margin, S - margin)
                waypoints.append((wx, wy))
            waypoints.append(c2)
            self._draw_path(img, waypoints)
        else:
            # Draw a distractor path that doesn't connect c1 to c2
            mid_x = rng.integers(margin, S - margin)
            mid_y = rng.integers(margin, S - margin)
            d1    = (rng.integers(margin, S // 2), rng.integers(margin, S - margin))
            d2    = (rng.integers(S // 2, S - margin), rng.integers(margin, S - margin))
            self._draw_path(img, [d1, (mid_x, mid_y), d2])

        # Add random distractor segments
        n_distractors = 3 if self.difficulty == "easy" else 6
        for _ in range(n_distractors):
            dx1 = rng.integers(margin, S - margin)
            dy1 = rng.integers(margin, S - margin)
            dx2 = rng.integers(margin, S - margin)
            dy2 = rng.integers(margin, S - margin)
            self._draw_path(img, [(dx1, dy1), (dx2, dy2)])

        # Draw endpoint circles on top
        self._draw_circle(img, c1[0], c1[1])
        self._draw_circle(img, c2[0], c2[1])

        return img

    def _generate(self, rng):
        data, labels = [], []
        for i in range(self.n_samples):
            connected = (i % 2 == 0)          # balanced classes
            img       = self._generate_one(rng, connected)
            data.append(img.flatten())         # (seq_len,)
            labels.append(int(connected))
        return np.stack(data), np.array(labels)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx]).float()    # (seq_len,)
        y = torch.tensor(self.labels[idx]).long()
        return x, y


# ─────────────────────────────────────────────
# Real LRA Pathfinder Loader
# ─────────────────────────────────────────────

class PathfinderLRADataset(Dataset):
    """
    Loads the real Pathfinder dataset from LRA.
    Data must be downloaded from:
      https://storage.googleapis.com/long-range-arena/lra_release.gz

    After extraction, set data_dir to the pathfinder subdirectory.
    """

    SPLITS = {"train": "train", "val": "val", "test": "test"}

    def __init__(
        self,
        data_dir: str,
        split:    str  = "train",
        img_size: int  = 32,
        normalize: bool = True,
    ):
        super().__init__()
        self.data_dir  = Path(data_dir)
        self.split     = split
        self.img_size  = img_size
        self.normalize = normalize

        self.samples = self._load_index()

    def _load_index(self):
        """Load the list of (image_path, label) pairs."""
        split_dir = self.data_dir / self.SPLITS[self.split]
        samples   = []

        for label_dir in sorted(split_dir.iterdir()):
            label = int(label_dir.name)
            for img_path in sorted(label_dir.glob("*.png")):
                samples.append((img_path, label))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        img_path, label = self.samples[idx]

        img = Image.open(img_path).convert("L")                       # grayscale
        img = np.array(img, dtype=np.float32) / 255.0                 # [0, 1]

        if self.normalize:
            img = (img - 0.5) / 0.5                                   # [-1, 1]

        x = torch.from_numpy(img.flatten())                           # (H*W,)
        y = torch.tensor(label).long()
        return x, y


# ─────────────────────────────────────────────
# DataLoader Factory
# ─────────────────────────────────────────────

def get_pathfinder_loaders(
    mode:        str   = "synthetic",   # 'synthetic' | 'lra'
    data_dir:    str   = None,
    img_size:    int   = 32,
    batch_size:  int   = 32,
    num_workers: int   = 2,
    n_samples:   int   = 4000,          # for synthetic mode
    val_frac:    float = 0.15,
    test_frac:   float = 0.15,
    seed:        int   = 42,
):
    """
    Returns (train_loader, val_loader, test_loader).

    Args:
        mode:       'synthetic' — generate locally  (no download needed)
                    'lra'       — load real LRA data (requires data_dir)
    """

    if mode == "synthetic":
        full_dataset = SyntheticPathfinderDataset(
            n_samples=n_samples,
            img_size=img_size,
            seed=seed,
        )
        n_total = len(full_dataset)
        n_val   = int(n_total * val_frac)
        n_test  = int(n_total * test_frac)
        n_train = n_total - n_val - n_test

        generator = torch.Generator().manual_seed(seed)
        train_ds, val_ds, test_ds = random_split(
            full_dataset,
            [n_train, n_val, n_test],
            generator=generator,
        )

    elif mode == "lra":
        assert data_dir is not None, "data_dir must be provided for LRA mode"
        train_ds = PathfinderLRADataset(data_dir, split="train", img_size=img_size)
        val_ds   = PathfinderLRADataset(data_dir, split="val",   img_size=img_size)
        test_ds  = PathfinderLRADataset(data_dir, split="test",  img_size=img_size)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    _loader = lambda ds, shuffle: DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return (
        _loader(train_ds, shuffle=True),
        _loader(val_ds,   shuffle=False),
        _loader(test_ds,  shuffle=False),
    )


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_pathfinder_loaders(
        mode="synthetic",
        img_size=32,
        n_samples=200,
        batch_size=16,
    )
    x, y = next(iter(train_loader))
    print(f"Batch x: {x.shape}  (batch × seq_len)")
    print(f"Batch y: {y.shape}  labels: {y[:8].tolist()}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val   batches: {len(val_loader)}")
    print(f"Test  batches: {len(test_loader)}")
    print("✓ DataLoader OK")
