"""Datasets shared by the teacher pass and the student training loop."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def scan_images(root: Path) -> list[Path]:
    """Every image file under `root`, sorted for a stable manifest order.

    Order matters more than it looks: the teacher shards are written in
    manifest order and joined back by row index, so a reshuffled manifest
    would silently mispair images with embeddings.
    """
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "path" not in df.columns:
        raise ValueError(f"{path} has no 'path' column; rebuild it with s01.")
    return df


class ImageDataset(Dataset):
    """Decodes images and applies a torchvision-style transform.

    Undecodable files yield a zero image rather than raising: a handful of
    corrupt JPEGs is normal in scraped biology corpora, and one bad file
    should not kill a 20-hour teacher pass. Their indices are reported by
    `failed_indices` so s02 can mask them out of the shard.
    """

    def __init__(self, paths: list[str], transform):
        self.paths = paths
        self.transform = transform
        self.failed: set[int] = set()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        try:
            with Image.open(self.paths[idx]) as im:
                img = self.transform(im.convert("RGB"))
            ok = True
        except Exception:
            # Shape must match the transform's output so collation still works.
            img = torch.zeros(3, 224, 224)
            ok = False
        return img, idx, ok


class DistillDataset(Dataset):
    """Pairs an image with its cached teacher embedding.

    The teacher saw a deterministic resize + center crop. The student is fed a
    lightly augmented view of the same image, which teaches it to map nearby
    crops to the same point, a cheap stand-in for MobileCLIP's much more
    expensive multi-augmentation "reinforced" caching. Keep the augmentation
    mild: aggressive crops show the student content the teacher never saw and
    the target embedding becomes actively wrong.
    """

    def __init__(self, paths: list[str], embeddings: np.ndarray, transform):
        if len(paths) != len(embeddings):
            raise ValueError(
                f"{len(paths)} paths vs {len(embeddings)} embeddings. "
                "the manifest and teacher cache are out of sync."
            )
        self.paths = paths
        self.embeddings = embeddings
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        try:
            with Image.open(self.paths[idx]) as im:
                img = self.transform(im.convert("RGB"))
        except Exception:
            img = torch.zeros(3, 224, 224)
        target = torch.from_numpy(self.embeddings[idx].astype(np.float32))
        return img, target
