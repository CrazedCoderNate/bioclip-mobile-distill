"""Step 2: run BioCLIP 2.5 over every image and cache its embeddings.

This is the expensive step and the one that actually needs the GPU. It is
also a one-time cost: once the embeddings are on disk you can retrain the
student as many times as you like without ever loading the teacher again.

Resumable by design. Embeddings are written in fixed-size shards; on restart
any complete shard is skipped. Killing this with Ctrl-C costs you at most one
shard's worth of work.

    python s02_cache_teacher.py
    python s02_cache_teacher.py --batch-size 32     # if you hit OOM

Throughput on a 12 GB RTX 5070 at fp16 lands around 80-150 img/s, so roughly
2-4 hours per million images. Run it overnight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from floradistill.data import ImageDataset, load_manifest


def shard_path(idx: int) -> Path:
    return config.TEACHER_DIR / f"emb_{idx:05d}.npy"


def valid_path(idx: int) -> Path:
    """Boolean mask of which rows in this shard decoded successfully."""
    return config.TEACHER_DIR / f"valid_{idx:05d}.npy"


META_PATH = lambda: config.TEACHER_DIR / "cache_meta.json"  # noqa: E731


def fingerprint(paths: list[str]) -> dict:
    """Identity of the manifest this cache was built against.

    Shards are joined back to the manifest by ROW INDEX, so a rebuilt
    manifest silently mispairs images with embeddings. Resume logic skips any
    shard whose file exists, which turns that into a trap: rerunning s00 and
    then s02 keeps the stale shards and looks like it worked.

    Hashing every path (not a sample) is cheap, a few hundred ms for 200k
    rows, and catches reordering as well as insertion or deletion.
    """
    h = hashlib.sha256()
    for p in paths:
        h.update(p.encode("utf-8"))
        h.update(b"\0")
    return {"n_rows": len(paths), "sha256": h.hexdigest(),
            "shard_size": config.SHARD_SIZE}


def check_cache_identity(paths: list[str], trust_existing: bool = False) -> bool:
    """False if existing shards belong to a different manifest."""
    current = fingerprint(paths)
    meta_file = META_PATH()

    if trust_existing:
        # Escape hatch for the case where you have already deleted the stale
        # shards by hand and know the survivors match. Adopts the current
        # manifest as ground truth, so a wrong call here is silent.
        meta_file.write_text(json.dumps(current, indent=2))
        print("Accepted existing shards as matching this manifest "
              "(--trust-existing).")
        return True

    if meta_file.exists():
        stored = json.loads(meta_file.read_text())
        if stored == current:
            return True
        print("\nThe cached embeddings belong to a different manifest.",
              file=sys.stderr)
        print(f"  cached : {stored['n_rows']:,} rows, shard size "
              f"{stored.get('shard_size')}", file=sys.stderr)
        print(f"  current: {current['n_rows']:,} rows, shard size "
              f"{current['shard_size']}", file=sys.stderr)
    elif any(config.TEACHER_DIR.glob("emb_*.npy")):
        # Shards from before this check existed. Cannot prove they match.
        print("\nFound cached embeddings with no fingerprint, so they cannot "
              "be verified against the current manifest.", file=sys.stderr)
    else:
        meta_file.write_text(json.dumps(current, indent=2))
        return True

    print("\nDelete the stale cache and re-run:", file=sys.stderr)
    print(f"  rm -rf {config.TEACHER_DIR}", file=sys.stderr)
    print("\nOr, if only some shards are stale, delete just those "
          "emb_*/valid_* pairs and the cache_meta.json.", file=sys.stderr)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-size", type=int, default=config.TEACHER_BATCH_SIZE)
    ap.add_argument("--workers", type=int, default=config.TEACHER_WORKERS)
    ap.add_argument("--limit-shards", type=int, default=None,
                    help="Stop after N shards, useful for a timing smoke test.")
    ap.add_argument("--trust-existing", action="store_true",
                    help="Adopt already-present shards as matching the current "
                         "manifest. Only after deleting the stale ones by hand.")
    args = ap.parse_args()

    config.ensure_dirs()

    if not config.MANIFEST_PATH.exists():
        print("No manifest. Run s01_build_manifest.py first.", file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print("CUDA is not available. This step is impractical on CPU, a "
              "ViT-H/14 forward pass is ~350 GFLOPs per image.", file=sys.stderr)
        return 1

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    df = load_manifest(config.MANIFEST_PATH)
    paths = df["path"].tolist()
    n_shards = (len(paths) + config.SHARD_SIZE - 1) // config.SHARD_SIZE
    print(f"Manifest: {len(paths):,} images across {n_shards} shards")

    # Refuse to resume onto a cache built from a different manifest. Doing so
    # produces embeddings paired with the wrong images, which no downstream
    # metric would obviously flag: the student would just train badly.
    if not check_cache_identity(paths, args.trust_existing):
        return 1

    todo = [i for i in range(n_shards) if not shard_path(i).exists()]
    if not todo:
        print("Every shard is already cached. Nothing to do.")
        return 0
    print(f"Remaining: {len(todo)} shards ({n_shards - len(todo)} already done)")
    if args.limit_shards:
        todo = todo[: args.limit_shards]

    # --- Load the teacher -------------------------------------------------
    print(f"\nLoading {config.TEACHER_MODEL} ...")
    print("(first run downloads ~4 GB from HuggingFace)")
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        config.TEACHER_MODEL
    )
    model = model.to(device).eval()

    # Verify the embedding dimension rather than trusting config. A silent
    # mismatch here would poison the student and the taxa table together, and
    # would not surface until on-device accuracy was inexplicably at chance.
    with torch.no_grad():
        probe = torch.zeros(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE,
                            device=device)
        dim = model.encode_image(probe).shape[-1]
    if dim != config.EMBED_DIM:
        print(f"\nEMBED_DIM mismatch: config says {config.EMBED_DIM}, the "
              f"model produces {dim}.", file=sys.stderr)
        print("Set EMBED_DIM in config.py to "
              f"{dim} and rebuild any existing cache.", file=sys.stderr)
        return 1
    print(f"Embedding dimension confirmed: {dim}")

    # --- Shard loop -------------------------------------------------------
    total_imgs = 0
    started = time.time()

    for shard_idx in todo:
        lo = shard_idx * config.SHARD_SIZE
        hi = min(lo + config.SHARD_SIZE, len(paths))
        shard_paths = paths[lo:hi]

        ds = ImageDataset(shard_paths, preprocess)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            # Workers are expensive to spawn on Windows; keep them alive
            # across the many batches within a shard.
            persistent_workers=args.workers > 0,
        )

        out = np.zeros((len(shard_paths), config.EMBED_DIM), dtype=np.float16)
        valid = np.zeros(len(shard_paths), dtype=bool)

        desc = f"shard {shard_idx + 1}/{n_shards}"
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for imgs, idxs, oks in tqdm(loader, desc=desc, unit="batch"):
                feats = model.encode_image(imgs.to(device, non_blocking=True))
                # L2-normalize now so every consumer (student loss, taxa
                # lookup, eval) works with unit vectors and cosine similarity
                # collapses to a plain dot product.
                feats = feats / feats.norm(dim=-1, keepdim=True)
                rows = idxs.numpy()
                out[rows] = feats.cpu().numpy().astype(np.float16)
                valid[rows] = oks.numpy()

        # Write the mask first: if the process dies between the two writes,
        # the missing emb_*.npy means the shard is correctly treated as
        # incomplete and redone, rather than half-trusted.
        np.save(valid_path(shard_idx), valid)
        np.save(shard_path(shard_idx), out)

        n_bad = int((~valid).sum())
        total_imgs += len(shard_paths)
        rate = total_imgs / (time.time() - started)
        print(f"  wrote {shard_path(shard_idx).name} "
              f"({len(shard_paths):,} rows, {n_bad} undecodable) "
              f"| {rate:.0f} img/s")

    elapsed = (time.time() - started) / 60
    print(f"\nDone. {total_imgs:,} images in {elapsed:.1f} min.")
    remaining = [i for i in range(n_shards) if not shard_path(i).exists()]
    if remaining:
        print(f"{len(remaining)} shards still pending, rerun to continue.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
