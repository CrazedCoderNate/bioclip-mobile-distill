"""Step 1: inventory the training images into a stable, ordered manifest.

Cheap and fast; run it any time the image folder changes. Everything
downstream joins on the manifest's ROW INDEX, so rebuilding the manifest
after caching teacher embeddings invalidates that cache.

    python s01_build_manifest.py
    python s01_build_manifest.py --limit 200000   # subset for a first run
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

import config
from floradistill.data import scan_images


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="Keep only the first N images (evenly strided).")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing manifest.")
    args = ap.parse_args()

    config.ensure_dirs()

    if config.MANIFEST_PATH.exists() and not args.force:
        existing = pd.read_parquet(config.MANIFEST_PATH)
        print(f"Manifest already exists with {len(existing):,} rows: "
              f"{config.MANIFEST_PATH}")
        print("Pass --force to rebuild (this invalidates the teacher cache).")
        return 0

    if not config.IMAGE_ROOT.exists():
        print(f"Image root does not exist: {config.IMAGE_ROOT}", file=sys.stderr)
        print("Point FLORA_IMAGE_ROOT at your image folder, or drop images "
              "into that path. See README step 1.", file=sys.stderr)
        return 1

    print(f"Scanning {config.IMAGE_ROOT} ...")
    paths = scan_images(config.IMAGE_ROOT)
    if not paths:
        print("No images found.", file=sys.stderr)
        return 1
    print(f"Found {len(paths):,} images.")

    if args.limit and args.limit < len(paths):
        # Stride rather than truncate so a subset still spans the whole corpus
        # (truncating an alphabetically sorted tree would hand you only the
        # species starting with 'A').
        stride = len(paths) / args.limit
        paths = [paths[int(i * stride)] for i in range(args.limit)]
        print(f"Strided down to {len(paths):,} images.")

    df = pd.DataFrame({
        "path": [str(p) for p in paths],
        # Parent folder name is the usual label in biology image dumps. Not
        # used for distillation (the teacher's embedding is the target), but
        # invaluable for the eval in s06.
        "folder": [p.parent.name for p in paths],
    })
    df.to_parquet(config.MANIFEST_PATH, index=False)

    print(f"\nWrote {config.MANIFEST_PATH}")
    print(f"  rows          : {len(df):,}")
    print(f"  distinct folders: {df['folder'].nunique():,}")
    est_gb = len(df) * config.EMBED_DIM * 2 / 1e9
    print(f"  teacher cache will be ~{est_gb:.1f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
