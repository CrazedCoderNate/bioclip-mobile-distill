"""Collect the publishable artifacts into one directory for HuggingFace.

Run this after s05 and s06. It strips the training checkpoint down to weights
only (the saved optimizer and scheduler state triple its size and are useless
to anyone downloading it), copies the exports, and writes the model card as
README.md, which is what HuggingFace renders on the model page.

    python prepare_release.py --ml-root ../BotanicalBuddy/ml
    python prepare_release.py --ml-root ../BotanicalBuddy/ml --out release
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ml-root", type=Path, required=True,
                    help="Working directory holding out/ from the pipeline.")
    ap.add_argument("--out", type=Path, default=Path("release"))
    args = ap.parse_args()

    export = args.ml_root / "out" / "export"
    ckpt = args.ml_root / "out" / "checkpoints" / "best.pt"
    report = args.ml_root / "out" / "eval_report.json"

    if not export.exists():
        print(f"No exports at {export}. Run s05 first.", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)

    # --- Straight copies --------------------------------------------------
    wanted = [
        (export / "flora_student_fp16.onnx", "flora_student_fp16.onnx"),
        (export / "flora_student_fp32.onnx", "flora_student_fp32.onnx"),
        (export / "taxa_table.npy", "taxa_table.npy"),
        (export / "taxa_labels.json", "taxa_labels.json"),
        (report, "eval_report.json"),
    ]
    for src, name in wanted:
        if src.exists():
            shutil.copy2(src, args.out / name)
            print(f"  {name:<28} {(args.out / name).stat().st_size / 1e6:>6.1f} MB")
        else:
            print(f"  {name:<28} MISSING at {src}", file=sys.stderr)

    # --- Slim checkpoint --------------------------------------------------
    # best.pt carries optimizer and scheduler state so training can resume.
    # A downloader wants the weights and the two fields needed to rebuild the
    # architecture, nothing else.
    if ckpt.exists():
        import torch

        full = torch.load(ckpt, map_location="cpu")
        slim = {
            "model": full["model"],
            "student": full.get("student"),
            "embed_dim": full.get("embed_dim"),
            "val_cos": full.get("best_cos"),
        }
        out_ckpt = args.out / "student_weights.pt"
        torch.save(slim, out_ckpt)
        before = ckpt.stat().st_size / 1e6
        after = out_ckpt.stat().st_size / 1e6
        print(f"  {'student_weights.pt':<28} {after:>6.1f} MB "
              f"(from {before:.1f} MB)")
    else:
        print(f"  student_weights.pt           MISSING at {ckpt}",
              file=sys.stderr)

    # --- Model card -------------------------------------------------------
    # HuggingFace renders README.md on the model page, so the card ships
    # under that name regardless of what it is called in the repo.
    card = Path(__file__).parent / "MODEL_CARD.md"
    if card.exists():
        shutil.copy2(card, args.out / "README.md")
        print(f"  {'README.md (model card)':<28}")

    total = sum(p.stat().st_size for p in args.out.iterdir()) / 1e6
    print(f"\n{args.out} ready, {total:.1f} MB total")
    print("\nUpload with:")
    print(f"  hf upload <your-username>/bioclip-2.5-mobile-fastvit {args.out} . --repo-type=model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
