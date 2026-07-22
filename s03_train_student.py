"""Step 3: train the small student to reproduce the teacher's embeddings.

No labels are involved. The cached teacher embeddings ARE the supervision,
which is why this works on any pile of biology images regardless of whether
anyone annotated them.

    python s03_train_student.py
    python s03_train_student.py --epochs 10 --student fastvit_t12
    python s03_train_student.py --resume out/checkpoints/last.pt

Watch `val_cos`. That is mean cosine similarity to the teacher on held-out
images, and it is the honest progress number:
    < 0.80  the student is not learning the space; check your data pipeline
    ~ 0.90  usable for genus-level and common species
    > 0.95  close to the practical ceiling for a model this small
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import config
from floradistill.data import DistillDataset, load_manifest
from floradistill.student import StudentEncoder, distillation_loss

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_teacher_cache(n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate the shards written by s02 back into one array."""
    shards = sorted(config.TEACHER_DIR.glob("emb_*.npy"))
    if not shards:
        raise FileNotFoundError(
            f"No teacher embeddings in {config.TEACHER_DIR}. Run s02 first."
        )
    embs, valids = [], []
    for s in shards:
        embs.append(np.load(s))
        v = config.TEACHER_DIR / s.name.replace("emb_", "valid_")
        valids.append(np.load(v) if v.exists()
                      else np.ones(len(embs[-1]), dtype=bool))
    emb = np.concatenate(embs)
    valid = np.concatenate(valids)

    if len(emb) != n_rows:
        raise ValueError(
            f"Teacher cache has {len(emb):,} rows but the manifest has "
            f"{n_rows:,}. The manifest changed after caching: either restore "
            f"it or delete {config.TEACHER_DIR} and re-run s02."
        )
    return emb, valid


def build_transforms(image_size: int):
    """Mild train augmentation; deterministic val.

    scale=(0.7, 1.0) is deliberately conservative. The teacher labelled a
    center crop of the full image, so a student crop that keeps at least 70%
    of the area still mostly contains what the teacher described. Drop that
    lower bound and you start training the student to predict the teacher's
    description of content its crop no longer contains.
    """
    train = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train, val


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--student", default=config.STUDENT_MODEL)
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.TRAIN_BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    ap.add_argument("--workers", type=int, default=config.TRAIN_WORKERS)
    ap.add_argument("--resume", type=Path, default=None)
    args = ap.parse_args()

    config.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: no CUDA. Training will be very slow.", file=sys.stderr)

    # --- Data -------------------------------------------------------------
    df = load_manifest(config.MANIFEST_PATH)
    paths = df["path"].tolist()
    emb, valid = load_teacher_cache(len(paths))

    keep = np.flatnonzero(valid)
    print(f"{len(keep):,} usable pairs ({len(paths) - len(keep):,} dropped)")

    rng = np.random.default_rng(0)  # fixed seed: same split across reruns
    rng.shuffle(keep)
    n_val = max(1, int(len(keep) * config.VAL_FRACTION))
    val_idx, train_idx = keep[:n_val], keep[n_val:]

    train_tf, val_tf = build_transforms(config.IMAGE_SIZE)
    train_ds = DistillDataset([paths[i] for i in train_idx], emb[train_idx], train_tf)
    val_ds = DistillDataset([paths[i] for i in val_idx], emb[val_idx], val_tf)

    common = dict(num_workers=args.workers, pin_memory=True,
                  persistent_workers=args.workers > 0)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          drop_last=True, **common)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **common)
    print(f"train {len(train_ds):,} | val {len(val_ds):,}")

    # --- Model ------------------------------------------------------------
    model = StudentEncoder(args.student, config.EMBED_DIM,
                           pretrained=config.STUDENT_PRETRAINED).to(device)
    mb = model.num_params() * 4 / 1e6
    print(f"Student {args.student}: {model.num_params() / 1e6:.1f}M params "
          f"({mb:.0f} MB fp32, ~{mb / 4:.0f} MB after int8)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=config.WEIGHT_DECAY)
    steps_per_epoch = len(train_ld)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=config.WARMUP_EPOCHS / max(args.epochs, 1),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch, best_cos = 0, -1.0
    if args.resume and args.resume.exists():
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch, best_cos = ck["epoch"] + 1, ck.get("best_cos", -1.0)
        print(f"Resumed from epoch {start_epoch} (best_cos {best_cos:.4f})")

    # --- Train ------------------------------------------------------------
    history = []
    for epoch in range(start_epoch, args.epochs):
        model.train()
        run_loss = run_cos = seen = 0
        t0 = time.time()
        bar = tqdm(train_ld, desc=f"epoch {epoch + 1}/{args.epochs}")
        for imgs, targets in bar:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast("cuda", enabled=device.type == "cuda"):
                out = model(imgs)
                loss, cos = distillation_loss(
                    out, targets, config.COSINE_WEIGHT, config.L1_WEIGHT
                )

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

            run_loss += loss.item()
            run_cos += cos
            seen += 1
            bar.set_postfix(loss=f"{run_loss / seen:.4f}",
                            cos=f"{run_cos / seen:.4f}")

        # --- Validate -----------------------------------------------------
        model.eval()
        val_cos, n_batches = 0.0, 0
        with torch.no_grad(), torch.autocast("cuda", enabled=device.type == "cuda"):
            for imgs, targets in val_ld:
                out = model(imgs.to(device, non_blocking=True))
                _, cos = distillation_loss(
                    out, targets.to(device, non_blocking=True),
                    config.COSINE_WEIGHT, config.L1_WEIGHT
                )
                val_cos += cos
                n_batches += 1
        val_cos /= max(n_batches, 1)

        mins = (time.time() - t0) / 60
        print(f"  epoch {epoch + 1}: train_cos {run_cos / seen:.4f} | "
              f"val_cos {val_cos:.4f} | {mins:.1f} min")
        history.append({"epoch": epoch + 1, "train_cos": run_cos / seen,
                        "val_cos": val_cos})

        ck = {"model": model.state_dict(), "opt": opt.state_dict(),
              "sched": sched.state_dict(), "epoch": epoch,
              "best_cos": max(best_cos, val_cos),
              "student": args.student, "embed_dim": config.EMBED_DIM}
        torch.save(ck, config.CHECKPOINT_DIR / "last.pt")
        if val_cos > best_cos:
            best_cos = val_cos
            torch.save(ck, config.CHECKPOINT_DIR / "best.pt")
            print(f"  new best ({best_cos:.4f}) -> best.pt")

    (config.CHECKPOINT_DIR / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\nBest val_cos: {best_cos:.4f}")
    print(f"Checkpoint: {config.CHECKPOINT_DIR / 'best.pt'}")
    print("Next: s04_build_taxa_table.py, then s06_eval.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
