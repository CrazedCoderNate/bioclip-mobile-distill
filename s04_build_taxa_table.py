"""Step 4: precompute the species lookup table with the teacher's text tower.

This is the trick that keeps the text encoder off the phone entirely. Every
species name is encoded once, here, on the desktop. The app ships the
resulting matrix and at inference time does nothing but one matrix multiply
against it.

Input is a newline-delimited list of scientific names at config.TAXA_LIST_PATH:

    Acer rubrum
    Quercus alba
    Toxicodendron radicans

Optionally `Scientific name|Common name`. The common name is carried into the
labels file for display and is also encoded as an extra prompt, which helps
for species whose vernacular name is more distinctive than the Latin.

    python s04_build_taxa_table.py

Output:
    out/export/taxa_table.npy    float32 [n_taxa, 1024], L2-normalized
    out/export/taxa_labels.json  parallel list of {scientific, common}
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch
from tqdm import tqdm

import config


def parse_taxa(path) -> list[dict]:
    taxa = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        scientific, _, common = line.partition("|")
        taxa.append({
            "scientific": scientific.strip(),
            "common": common.strip() or None,
        })
    return taxa


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    config.ensure_dirs()

    if not config.TAXA_LIST_PATH.exists():
        print(f"No taxa list at {config.TAXA_LIST_PATH}", file=sys.stderr)
        print("Create it: one scientific name per line, optionally "
              "'Scientific|Common'.", file=sys.stderr)
        return 1

    taxa = parse_taxa(config.TAXA_LIST_PATH)
    if not taxa:
        print("Taxa list is empty.", file=sys.stderr)
        return 1
    print(f"{len(taxa):,} taxa")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {config.TEACHER_MODEL} on {device} ...")

    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(config.TEACHER_MODEL)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(config.TEACHER_MODEL)

    # Build every prompt for every taxon up front, then encode in flat
    # batches, one pass over the text tower rather than one per taxon.
    prompts: list[str] = []
    spans: list[tuple[int, int]] = []
    for t in taxa:
        start = len(prompts)
        names = [t["scientific"]]
        if t["common"]:
            names.append(t["common"])
        for name in names:
            prompts.extend(tpl.format(name=name) for tpl in config.PROMPT_TEMPLATES)
        spans.append((start, len(prompts)))

    print(f"{len(prompts):,} prompts "
          f"({len(config.PROMPT_TEMPLATES)} templates x name variants)")

    feats = []
    with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
        for i in tqdm(range(0, len(prompts), args.batch_size), desc="encoding"):
            tokens = tokenizer(prompts[i:i + args.batch_size]).to(device)
            f = model.encode_text(tokens)
            feats.append((f / f.norm(dim=-1, keepdim=True)).float().cpu())
    feats = torch.cat(feats)

    if feats.shape[-1] != config.EMBED_DIM:
        print(f"Text embeddings are {feats.shape[-1]}-d but EMBED_DIM is "
              f"{config.EMBED_DIM}.", file=sys.stderr)
        return 1

    # Average each taxon's prompts, then re-normalize. Averaging unit vectors
    # does not produce a unit vector, and skipping the second normalization
    # would let prompt-count differences (taxa with vs without a common name)
    # act as a similarity bias at lookup time.
    table = torch.stack([feats[a:b].mean(dim=0) for a, b in spans])
    table = torch.nn.functional.normalize(table, dim=-1).numpy().astype(np.float32)

    np.save(config.TAXA_TABLE_PATH, table)
    config.TAXA_LABELS_PATH.write_text(json.dumps(taxa, indent=2, ensure_ascii=False),
                                       encoding="utf-8")

    # Sanity check: the most similar OTHER taxon for a few entries. Near-1.0
    # values mean those two species are nearly indistinguishable to the text
    # tower, and no image encoder downstream will separate them either.
    sims = table @ table.T
    np.fill_diagonal(sims, -1.0)
    print("\nClosest confusable pairs (high = hard to tell apart):")
    for i in np.argsort(-sims.max(axis=1))[:5]:
        j = int(sims[i].argmax())
        print(f"  {sims[i, j]:.3f}  {taxa[i]['scientific']} <-> "
              f"{taxa[j]['scientific']}")

    print(f"\nWrote {config.TAXA_TABLE_PATH}  {table.shape}")
    print(f"Wrote {config.TAXA_LABELS_PATH}")
    print(f"Ships as ~{table.nbytes / 1e6:.1f} MB fp32 "
          f"(~{table.nbytes / 2e6:.1f} MB if cast to fp16 for the app)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
