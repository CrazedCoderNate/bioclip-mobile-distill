"""Step 7: package the model and taxa table as Android assets.

The taxa table ships as .npy, which Kotlin cannot read. This converts it to a
flat little-endian float16 blob that maps directly onto a Java FloatArray, and
emits a manifest so the app can verify what it loaded instead of trusting
hardcoded constants.

    python s07_android_assets.py --ml-root ../BotanicalBuddy/ml \
                                 --app-assets ../BotanicalBuddy/app/src/main/assets

Output:
    flora_student.onnx      the fp16 model, renamed for the app
    taxa_table.f16          [n_taxa, 1024] row-major little-endian float16
    taxa_labels.json        parallel names
    model_manifest.json     dims and checksums, verified at load time
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ml-root", type=Path, required=True)
    ap.add_argument("--app-assets", type=Path, required=True)
    args = ap.parse_args()

    export = args.ml_root / "out" / "export"
    model_src = export / "flora_student_fp16.onnx"
    table_src = export / "taxa_table.npy"
    labels_src = export / "taxa_labels.json"

    for p in (model_src, table_src, labels_src):
        if not p.exists():
            print(f"Missing {p}. Run s05 and s04 first.", file=sys.stderr)
            return 1

    args.app_assets.mkdir(parents=True, exist_ok=True)

    # --- Model ------------------------------------------------------------
    model_dst = args.app_assets / "flora_student.onnx"
    shutil.copy2(model_src, model_dst)
    print(f"  flora_student.onnx   {model_dst.stat().st_size / 1e6:>6.1f} MB")

    # --- Taxa table -------------------------------------------------------
    table = np.load(table_src)
    if table.ndim != 2:
        print(f"Expected a 2-D table, got shape {table.shape}", file=sys.stderr)
        return 1

    # Re-normalize before the cast. The rows are unit length in float32, but
    # rounding to float16 perturbs them slightly, and the app treats the dot
    # product as cosine similarity with no normalization of its own.
    table = table.astype(np.float32)
    table /= np.linalg.norm(table, axis=1, keepdims=True)
    table_f16 = table.astype("<f2")   # explicit little-endian

    table_dst = args.app_assets / "taxa_table.f16"
    table_dst.write_bytes(table_f16.tobytes(order="C"))
    print(f"  taxa_table.f16       {table_dst.stat().st_size / 1e6:>6.1f} MB "
          f"{table.shape}")

    # How much did the fp16 cast cost? Worst-case row cosine against the
    # float32 original. Anything below ~0.999 would mean the table itself is
    # degrading matches before the model even runs.
    back = table_f16.astype(np.float32)
    back /= np.linalg.norm(back, axis=1, keepdims=True)
    row_cos = (back * table).sum(axis=1)
    print(f"  fp16 table cosine    {row_cos.mean():.6f} "
          f"(min {row_cos.min():.6f})")
    if row_cos.min() < 0.999:
        print("  WARNING: fp16 cast is degrading table rows.", file=sys.stderr)

    # --- Labels -----------------------------------------------------------
    labels = json.loads(labels_src.read_text(encoding="utf-8"))
    if len(labels) != table.shape[0]:
        print(f"Label count {len(labels)} does not match table rows "
              f"{table.shape[0]}.", file=sys.stderr)
        return 1
    labels_dst = args.app_assets / "taxa_labels.json"
    labels_dst.write_text(json.dumps(labels, ensure_ascii=False),
                          encoding="utf-8")
    print(f"  taxa_labels.json     {labels_dst.stat().st_size / 1e6:>6.1f} MB "
          f"({len(labels):,} taxa)")

    # --- Manifest ---------------------------------------------------------
    # The app asserts these at load time. A silent dimension mismatch between
    # the model and the table would otherwise produce plausible-looking
    # nonsense rather than an error.
    report_path = args.ml_root / "out" / "eval_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    manifest = {
        "embedDim": int(table.shape[1]),
        "taxaCount": int(table.shape[0]),
        "inputSize": 224,
        "modelSha256": sha256_of(model_dst),
        "tableSha256": sha256_of(table_dst),
        "top1Agreement": report.get("top1_agree"),
        "top5Contain": report.get("top5_contain"),
    }
    (args.app_assets / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  model_manifest.json  embedDim={manifest['embedDim']} "
          f"taxa={manifest['taxaCount']}")

    total = sum(p.stat().st_size for p in args.app_assets.iterdir()) / 1e6
    print(f"\n{args.app_assets} ready, {total:.1f} MB of assets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
