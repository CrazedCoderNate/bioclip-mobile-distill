"""Step 5: export the trained student for the phone, and quantize it.

Produces ONNX by default, which works on Windows and runs on Android via
ONNX Runtime Mobile. That is the recommended path for this project: the
TFLite converter (ai-edge-torch) is Linux-only, and going Windows -> WSL2
just to change container format buys you nothing here.

    python s05_export.py                      # fp32 + int8 ONNX
    python s05_export.py --no-quantize        # fp32 only
    python s05_export.py --checkpoint out/checkpoints/last.pt

Static int8 quantization needs real images to calibrate the activation
ranges; it pulls a few hundred from the manifest automatically. Dynamic
quantization is offered as a fallback but is a poor fit for a mostly-conv
backbone: it leaves the convolutions in float and saves much less.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import config
from floradistill.data import load_manifest
from floradistill.student import ExportWrapper, StudentEncoder

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CalibrationReader:
    """Feeds real images to onnxruntime's static quantizer.

    Calibration images must look like what the app will actually see. Random
    noise would give the quantizer activation ranges that no real photo
    produces, and int8 accuracy would collapse for reasons that are miserable
    to debug later.
    """

    def __init__(self, paths: list[str], input_name: str, size: int):
        self.input_name = input_name
        self.tf = transforms.Compose([
            transforms.Resize(int(size * 1.14)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
        ])
        self.paths = paths
        self._it = None

    def _gen(self):
        for p in self.paths:
            try:
                with Image.open(p) as im:
                    x = self.tf(im.convert("RGB"))
            except Exception:
                continue
            yield {self.input_name: x.unsqueeze(0).numpy()}

    def get_next(self):
        if self._it is None:
            self._it = self._gen()
        return next(self._it, None)


def compare_to_fp32(ref_sess, model_path: Path, paths: list[str]
                    ) -> tuple[float, float]:
    """Mean and worst-case cosine between fp32 and a converted model.

    Cosine on real images is the metric that matters, not file size or a
    per-tensor error bound: the app compares embeddings by direction, so an
    error that preserves direction is free and one that rotates the vector is
    fatal regardless of how small it looks elementwise.
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(str(model_path),
                                providers=["CPUExecutionProvider"])
    tf = transforms.Compose([
        transforms.Resize(int(config.IMAGE_SIZE * 1.14)),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    sims = []
    for p in paths:
        try:
            with Image.open(p) as im:
                x = tf(im.convert("RGB")).unsqueeze(0).numpy()
        except Exception:
            continue
        a = ref_sess.run(None, {"image": x})[0][0]
        b = sess.run(None, {"image": x})[0][0]
        sims.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
    if not sims:
        return float("nan"), float("nan")
    return float(np.mean(sims)), float(min(sims))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path,
                    default=config.CHECKPOINT_DIR / "best.pt")
    ap.add_argument("--no-quantize", action="store_true")
    ap.add_argument("--no-reparam", action="store_true",
                    help="Skip inference fusion. Diagnostic only: the "
                         "unfused model is slower and quantizes badly.")
    ap.add_argument("--calib-images", type=int, default=300)
    ap.add_argument("--calib-method", default="sweep",
                    choices=["sweep", "percentile", "entropy", "minmax"],
                    help="Activation calibration. 'sweep' tries all three and "
                         "reports which holds up best (a few minutes).")
    # 18 is the floor the current exporter implements. Requesting 17 makes it
    # export at 18 and then fail an automatic down-conversion, which is noisy
    # and buys nothing. Android's ONNX Runtime handles 18 fine.
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()

    config.ensure_dirs()

    if not args.checkpoint.exists():
        print(f"No checkpoint at {args.checkpoint}. Run s03 first.", file=sys.stderr)
        return 1

    ck = torch.load(args.checkpoint, map_location="cpu")
    backbone = ck.get("student", config.STUDENT_MODEL)
    embed_dim = ck.get("embed_dim", config.EMBED_DIM)
    print(f"Checkpoint: {args.checkpoint.name} | {backbone} | {embed_dim}-d "
          f"| epoch {ck.get('epoch', '?')} | best_cos {ck.get('best_cos', float('nan')):.4f}")

    # pretrained=False: the checkpoint supplies every weight, and fetching
    # ImageNet weights just to overwrite them wastes a download.
    student = StudentEncoder(backbone, embed_dim, pretrained=False)
    student.load_state_dict(ck["model"])
    student.eval()

    # --- Reparameterize --------------------------------------------------
    # FastViT/MobileOne train as multi-branch blocks (conv + BN + identity in
    # parallel) and are designed to be algebraically fused into a single conv
    # for inference. Skipping this ships the training skeleton: slower on
    # device, and the leftover rank-1 identity/BN weights make per-channel
    # int8 quantization collapse (cosine ~0.57 instead of ~0.99).
    #
    # Mathematically exact. The fused model is verified against the
    # unfused one below rather than assumed.
    if not args.no_reparam:
        with torch.no_grad():
            ref = student(probe_ref := torch.rand(1, 3, config.IMAGE_SIZE,
                                                  config.IMAGE_SIZE))
        try:
            from timm.utils.model import reparameterize_model
            student = reparameterize_model(student).eval()
        except Exception as e:
            print(f"  reparameterization unavailable ({e}); exporting "
                  f"unfused. int8 accuracy will suffer.", file=sys.stderr)
        else:
            with torch.no_grad():
                after = student(probe_ref)
            drift = float((after - ref).abs().max())
            n_after = sum(p.numel() for p in student.parameters())
            print(f"Reparameterized: {n_after / 1e6:.1f}M params "
                  f"(fusion drift {drift:.2e})")
            if drift > 1e-4:
                print(f"  Fusion changed the output by {drift:.2e}, which is "
                      f"more than numerical noise. Rerun with --no-reparam "
                      f"and investigate before shipping.", file=sys.stderr)
                return 1

    model = ExportWrapper(student, IMAGENET_MEAN, IMAGENET_STD).eval()

    # Sanity: the exported graph must still emit unit vectors, because the
    # Android side treats cosine similarity as a bare dot product.
    with torch.no_grad():
        probe = torch.rand(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
        out = model(probe)
    norm = out.norm().item()
    if abs(norm - 1.0) > 1e-3:
        print(f"Output norm is {norm:.4f}, expected 1.0. The normalization "
              f"was lost somewhere in the wrapper.", file=sys.stderr)
        return 1
    print(f"Output: {tuple(out.shape)}, L2 norm {norm:.4f}")

    fp32_path = config.EXPORT_DIR / "flora_student_fp32.onnx"
    print(f"\nExporting ONNX (opset {args.opset}) ...")
    # Fixed batch of 1, deliberately NOT a dynamic axis. The phone runs one
    # frame at a time and s06 evaluates from the .pt checkpoint, so nothing
    # needs batching here. A symbolic batch dimension also breaks
    # onnxruntime's shape inference during quantization below, which fails
    # with an unhelpful bare AssertionError.
    torch.onnx.export(
        model,
        probe,
        str(fp32_path),
        input_names=["image"],
        output_names=["embedding"],
        opset_version=args.opset,
        do_constant_folding=True,
    )

    import onnx

    # The exporter offloads weights to a .onnx.data sidecar. Re-save with
    # everything inline so the app ships ONE asset. A missing sidecar fails
    # at model-load time on device, which is a miserable way to find out.
    model_proto = onnx.load(str(fp32_path))   # pulls in external data
    onnx.save_model(model_proto, str(fp32_path), save_as_external_data=False)
    sidecar = fp32_path.with_suffix(".onnx.data")
    if sidecar.exists():
        sidecar.unlink()
        print("  folded external weights back into the .onnx")

    print(f"  {fp32_path.name}  {fp32_path.stat().st_size / 1e6:.1f} MB")
    onnx.checker.check_model(onnx.load(str(fp32_path)))
    print("  graph validates")

    # Numerical parity between torch and onnxruntime. A mismatch here means
    # an op was exported with different semantics, and it is far cheaper to
    # catch now than after it ships inside an APK.
    import onnxruntime as ort
    sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"image": probe.numpy()})[0]
    max_diff = float(np.abs(onnx_out - out.numpy()).max())
    print(f"  torch vs onnxruntime max abs diff: {max_diff:.2e}")
    if max_diff > 1e-3:
        print("  WARNING: larger than expected, inspect before shipping.")

    # --- Calibration / verification images --------------------------------
    if not config.MANIFEST_PATH.exists():
        print("No manifest, so no calibration images. Run s01 or s00.",
              file=sys.stderr)
        return 1
    df = load_manifest(config.MANIFEST_PATH)
    # Stride across the whole corpus rather than taking the first N, so
    # calibration sees a spread of species and lighting.
    stride = max(1, len(df) // args.calib_images)
    calib_paths = df["path"].tolist()[::stride][: args.calib_images]
    verify_paths = calib_paths[:50]

    results: list[tuple[str, Path, float, float]] = []

    # --- fp16 -------------------------------------------------------------
    # Half the size for near-zero accuracy cost. Unlike int8 it needs no
    # calibration, because it rescales rather than discretizing.
    print("\nConverting to fp16 ...")
    fp16_path = config.EXPORT_DIR / "flora_student_fp16.onnx"
    try:
        from onnxconverter_common import float16
        # The converter warns once per constant it clamps into fp16 range.
        # Every one of them is a normalization epsilon, values like 1e-12
        # inside sqrt(var + eps), where the change is far below anything that
        # affects the result. Silenced because the cosine check below is the
        # real verdict, and hundreds of warnings bury it.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            fp16_model = float16.convert_float_to_float16(
                onnx.load(str(fp32_path)), keep_io_types=True
            )
        onnx.save_model(fp16_model, str(fp16_path), save_as_external_data=False)
        mean_c, min_c = compare_to_fp32(sess, fp16_path, verify_paths)
        print(f"  {fp16_path.name}  {fp16_path.stat().st_size / 1e6:.1f} MB "
              f"| cosine {mean_c:.4f} (min {min_c:.4f})")
        results.append(("fp16", fp16_path, mean_c, min_c))
    except ImportError:
        print("  skipped (pip install onnxconverter-common)")

    if args.no_quantize:
        print("\nSkipping int8 (--no-quantize).")
        return 0

    # --- int8 -------------------------------------------------------------
    try:
        from onnxruntime.quantization import (
            CalibrationMethod, QuantFormat, QuantType, quantize_static,
        )
        from onnxruntime.quantization.shape_inference import quant_pre_process
    except ImportError:
        print("onnxruntime.quantization unavailable; install onnxruntime.",
              file=sys.stderr)
        return 1

    # Calibration method decides how activation ranges are chosen, and for
    # attention-bearing backbones it matters far more than any other knob.
    # MinMax takes the single most extreme activation seen, so one outlier in
    # an attention block stretches the range and crushes every ordinary value
    # into a handful of levels. Percentile and Entropy clip that tail.
    #
    # Which one wins is model-specific and cheap to measure, so measure it
    # rather than guessing.
    methods = {
        "percentile": CalibrationMethod.Percentile,
        "entropy": CalibrationMethod.Entropy,
        "minmax": CalibrationMethod.MinMax,
    }
    chosen = list(methods) if args.calib_method == "sweep" else [args.calib_method]

    prepped = config.EXPORT_DIR / "_prepped.onnx"
    # skip_symbolic_shape: the graph has fully static shapes now, so symbolic
    # inference has nothing to solve and only adds a failure mode.
    quant_pre_process(str(fp32_path), str(prepped), skip_symbolic_shape=True)

    print(f"\nQuantizing to int8 on {len(calib_paths)} calibration images")
    for name in chosen:
        out_path = config.EXPORT_DIR / f"flora_student_int8_{name}.onnx"
        try:
            quantize_static(
                str(prepped), str(out_path),
                CalibrationReader(calib_paths, "image", config.IMAGE_SIZE),
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
                calibrate_method=methods[name],
                per_channel=True,   # materially better for conv-heavy backbones
            )
        except Exception as e:
            print(f"  {name:<11} failed: {e}")
            continue
        mean_c, min_c = compare_to_fp32(sess, out_path, verify_paths)
        print(f"  {name:<11} {out_path.stat().st_size / 1e6:>5.1f} MB "
              f"| cosine {mean_c:.4f} (min {min_c:.4f})")
        results.append((f"int8/{name}", out_path, mean_c, min_c))
    prepped.unlink(missing_ok=True)

    # --- Recommendation ---------------------------------------------------
    # Below ~0.99 the converted model drifts far enough to flip top-1 on close
    # calls, which for look-alike species is most of the interesting cases.
    print("\n" + "=" * 58)
    int8_ok = [r for r in results if r[0].startswith("int8") and r[2] >= 0.99]
    if int8_ok:
        best = max(int8_ok, key=lambda r: r[2])
        print(f"  Ship {best[1].name}")
        print(f"  {best[0]} holds cosine {best[2]:.4f} at "
              f"{best[1].stat().st_size / 1e6:.1f} MB")
    else:
        fp16 = next((r for r in results if r[0] == "fp16"), None)
        best_int8 = max((r for r in results if r[0].startswith("int8")),
                        key=lambda r: r[2], default=None)
        if best_int8:
            print(f"  int8 tops out at cosine {best_int8[2]:.4f} "
                  f"({best_int8[0]}), too lossy to ship.")
        if fp16:
            print(f"  Ship {fp16[1].name} instead: cosine {fp16[2]:.4f} at "
                  f"{fp16[1].stat().st_size / 1e6:.1f} MB.")
            print("  The extra megabytes are worth far more than the accuracy.")
        print("  To pursue int8 anyway, quantization-aware training is the "
              "next lever.")
    print("=" * 58)

    print(f"\nExports in {config.EXPORT_DIR}")
    print("Next: s06_eval.py, then copy the chosen .onnx + taxa table into "
          "app/src/main/assets/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
