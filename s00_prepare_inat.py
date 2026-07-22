"""Step 0b: turn an extracted iNaturalist 2021 download into a ready corpus.

Use this INSTEAD OF s01 if your images come from iNat21. It does the same job
(writes manifest.parquet) plus two things s01 cannot: it filters to a single
kingdom, and it derives data/taxa.txt from the folder names, so you never
hand-write a species list.

iNat21 folder names carry the full taxonomy in 8 underscore-separated parts:

    04567_Plantae_Tracheophyta_Magnoliopsida_Fagales_Fagaceae_Quercus_alba
    |     |       |           |             |       |        |      |
    id    kingdom phylum      class         order   family   genus  species

so `Quercus alba` falls straight out of parts 6 and 7.

    python s00_prepare_inat.py --inat-root D:/datasets/inat2021/train_mini
    python s00_prepare_inat.py --inat-root ... --kingdom Fungi
    python s00_prepare_inat.py --inat-root ... --max-per-species 25
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

import config
from floradistill.data import IMAGE_EXTS

RANKS = ("kingdom", "phylum", "class", "order", "family", "genus", "species")


def parse_folder(name: str) -> dict | None:
    """Split an iNat21 category folder into its taxonomic ranks.

    Returns None for anything that does not match the 8-part convention, so a
    stray README or .DS_Store in the tree is skipped rather than fatal.
    """
    parts = name.split("_")
    if len(parts) != 8:
        return None
    cat_id, *ranks = parts
    if not cat_id.isdigit():
        return None
    return dict(zip(RANKS, ranks)) | {"category_id": cat_id}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inat-root", type=Path, required=True,
                    help="The extracted train_mini (or train) directory.")
    ap.add_argument("--kingdom", default="Plantae",
                    help="Kingdom to keep. Use 'all' for everything.")
    ap.add_argument("--max-per-species", type=int, default=None,
                    help="Cap images per species, useful for a fast first run.")
    ap.add_argument("--min-per-species", type=int, default=5,
                    help="Drop species with fewer images than this.")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing manifest.")
    args = ap.parse_args()

    config.ensure_dirs()

    if not args.inat_root.exists():
        print(f"Not found: {args.inat_root}", file=sys.stderr)
        print("Point --inat-root at the directory that CONTAINS the "
              "0xxxx_Kingdom_... folders.", file=sys.stderr)
        return 1

    if config.MANIFEST_PATH.exists() and not args.force:
        print(f"Manifest already exists: {config.MANIFEST_PATH}")
        print("Pass --force to rebuild (this invalidates the teacher cache).")
        return 0

    all_dirs = [d for d in args.inat_root.iterdir() if d.is_dir()]
    if not all_dirs:
        print(f"No subdirectories in {args.inat_root}.", file=sys.stderr)
        return 1
    print(f"{len(all_dirs):,} category folders found")

    parsed = [(d, parse_folder(d.name)) for d in all_dirs]
    bad = [d.name for d, p in parsed if p is None]
    if bad:
        print(f"  skipping {len(bad)} unparseable folders (e.g. {bad[0]!r})")
    parsed = [(d, p) for d, p in parsed if p is not None]

    if args.kingdom.lower() != "all":
        before = len(parsed)
        parsed = [(d, p) for d, p in parsed
                  if p["kingdom"].lower() == args.kingdom.lower()]
        print(f"  {len(parsed):,} of {before:,} are kingdom {args.kingdom}")
        if not parsed:
            kingdoms = Counter(p["kingdom"] for _, p in
                               [(d, parse_folder(d.name)) for d in all_dirs]
                               if p)
            print(f"No folders matched. Present kingdoms: "
                  f"{dict(kingdoms)}", file=sys.stderr)
            return 1

    # --- Collect image paths ---------------------------------------------
    rows = []
    skipped_small = 0
    for d, taxa in sorted(parsed, key=lambda t: t[0].name):
        imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if len(imgs) < args.min_per_species:
            skipped_small += 1
            continue
        if args.max_per_species:
            imgs = imgs[: args.max_per_species]
        species = f"{taxa['genus']} {taxa['species']}"
        for p in imgs:
            rows.append({
                "path": str(p),
                "folder": d.name,
                "species": species,
                "genus": taxa["genus"],
                "family": taxa["family"],
            })

    if not rows:
        print("No images collected.", file=sys.stderr)
        return 1
    if skipped_small:
        print(f"  dropped {skipped_small} species under "
              f"{args.min_per_species} images")

    df = pd.DataFrame(rows)
    df.to_parquet(config.MANIFEST_PATH, index=False)

    # --- taxa.txt ---------------------------------------------------------
    # One line per species, sorted. These are exactly the classes the phone
    # will be able to predict, so this file and the manifest must be built
    # from the same run, a species in the table with no training images is a
    # confident wrong answer waiting to happen.
    species = sorted(df["species"].unique())
    config.TAXA_LIST_PATH.write_text(
        "# Generated by s00_prepare_inat.py, one species per line.\n"
        "# Add '|Common name' after any entry to improve the text prompts.\n"
        + "\n".join(species) + "\n",
        encoding="utf-8",
    )

    print(f"\nWrote {config.MANIFEST_PATH}")
    print(f"  images  : {len(df):,}")
    print(f"  species : {len(species):,}")
    print(f"  families: {df['family'].nunique():,}")
    print(f"Wrote {config.TAXA_LIST_PATH}")

    per = df.groupby("species").size()
    print(f"\nImages per species: min {per.min()}, median "
          f"{int(per.median())}, max {per.max()}")
    est_h = len(df) / 115 / 3600   # ~115 img/s midpoint on a 12 GB Blackwell card
    print(f"Teacher cache will be ~{len(df) * config.EMBED_DIM * 2 / 1e9:.1f} GB "
          f"and take roughly {est_h:.1f} h on your GPU.")
    print("\nNext: python s02_cache_teacher.py --limit-shards 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
