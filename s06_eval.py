"""Step 6: measure what the student actually lost relative to the teacher.

val_cos from training tells you the student lands near the teacher in
embedding space. It does not tell you whether the remaining gap changes the
ANSWER. This does: both models classify the same held-out images against the
taxa table, and the report is how often they disagree.

    python s06_eval.py --n 2000

Read the output as:
    top1_agree   student picks the teacher's top species. The headline number.
    top5_contain teacher's top species is somewhere in the student's top 5.
                 A two-stage app can exploit this even when top-1 misses.
    Per-taxon disagreements show WHICH species the student cannot hold onto,
    usually visually similar congeners, and exactly the cases worth routing
    to the cloud.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import config
from floradistill.data import ImageDataset, load_manifest
from floradistill.student import StudentEncoder

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path,
                    default=config.CHECKPOINT_DIR / "best.pt")
    ap.add_argument("--n", type=int, default=2000,
                    help="How many held-out images to evaluate.")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    for required in (config.TAXA_TABLE_PATH, args.checkpoint):
        if not required.exists():
            print(f"Missing {required}. Run s03 and s04 first.", file=sys.stderr)
            return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table = torch.from_numpy(np.load(config.TAXA_TABLE_PATH)).to(device)
    labels = json.loads(config.TAXA_LABELS_PATH.read_text(encoding="utf-8"))
    print(f"Taxa table: {tuple(table.shape)}")

    # Same seed and split as s03, so these images were never trained on.
    df = load_manifest(config.MANIFEST_PATH)
    paths = df["path"].tolist()
    rng = np.random.default_rng(0)
    order = np.arange(len(paths))
    rng.shuffle(order)
    n_val = max(1, int(len(order) * config.VAL_FRACTION))
    val_idx = order[:n_val][: args.n]
    val_paths = [paths[i] for i in val_idx]
    print(f"Evaluating {len(val_paths):,} held-out images")

    val_tf = transforms.Compose([
        transforms.Resize(int(config.IMAGE_SIZE * 1.14)),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    loader = DataLoader(ImageDataset(val_paths, val_tf),
                        batch_size=args.batch_size, num_workers=4)

    # --- Student ----------------------------------------------------------
    ck = torch.load(args.checkpoint, map_location="cpu")
    student = StudentEncoder(ck.get("student", config.STUDENT_MODEL),
                             ck.get("embed_dim", config.EMBED_DIM),
                             pretrained=False)
    student.load_state_dict(ck["model"])
    student = student.to(device).eval()

    print("Student pass ...")
    s_emb = []
    with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
        for imgs, _, _ in tqdm(loader, unit="batch"):
            s_emb.append(student(imgs.to(device)).float())
    s_emb = torch.cat(s_emb)

    # --- Teacher ----------------------------------------------------------
    print(f"Teacher pass ({config.TEACHER_MODEL}) ...")
    import open_clip
    teacher, _, _ = open_clip.create_model_and_transforms(config.TEACHER_MODEL)
    teacher = teacher.to(device).eval()

    t_emb = []
    with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
        for imgs, _, _ in tqdm(loader, unit="batch"):
            f = teacher.encode_image(imgs.to(device)).float()
            t_emb.append(f / f.norm(dim=-1, keepdim=True))
    t_emb = torch.cat(t_emb)

    # --- Compare ----------------------------------------------------------
    # Both sides are unit vectors, so a matmul against the (unit) taxa table
    # is exactly cosine similarity.
    s_scores, t_scores = s_emb @ table.T, t_emb @ table.T
    s_top5 = s_scores.topk(min(5, table.shape[0]), dim=-1).indices
    t_top1 = t_scores.argmax(dim=-1)

    top1 = (s_top5[:, 0] == t_top1).float().mean().item()
    top5 = (s_top5 == t_top1.unsqueeze(1)).any(dim=1).float().mean().item()
    cos = torch.nn.functional.cosine_similarity(s_emb, t_emb, dim=-1).mean().item()

    print("\n" + "=" * 58)
    print(f"  embedding cosine (student vs teacher) : {cos:.4f}")
    print(f"  top1_agree                            : {top1:.1%}")
    print(f"  top5_contain                          : {top5:.1%}")
    print("=" * 58)

    disagree = (s_top5[:, 0] != t_top1).nonzero().flatten().tolist()
    if disagree:
        pairs = Counter(
            (labels[t_top1[i].item()]["scientific"],
             labels[s_top5[i, 0].item()]["scientific"])
            for i in disagree
        )
        print(f"\nMost common confusions ({len(disagree)} total):")
        print("  teacher said -> student said   (count)")
        for (truth, pred), count in pairs.most_common(12):
            print(f"  {truth} -> {pred}   ({count})")
        print("\nRoute these to the cloud rather than trusting the on-device "
              "answer; that is what the two-stage design is for.")

    out = config.OUT_DIR / "eval_report.json"
    out.write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "n_images": len(val_paths),
        "embedding_cosine": cos,
        "top1_agree": top1,
        "top5_contain": top5,
    }, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
