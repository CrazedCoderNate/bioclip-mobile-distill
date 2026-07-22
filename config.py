"""Central configuration for the BioCLIP 2.5 -> mobile student distillation.

Every path and tuning knob lives here so the numbered scripts stay readable.
Override any value with an environment variable of the same name, e.g.

    set FLORA_IMAGE_ROOT=D:\\datasets\\treeoflife
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths ----------------------------------------------------------------
ML_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("FLORA_DATA_DIR", ML_ROOT / "data"))
OUT_DIR = Path(os.getenv("FLORA_OUT_DIR", ML_ROOT / "out"))

# Folder tree of source images. Any nesting; scanned recursively.
IMAGE_ROOT = Path(os.getenv("FLORA_IMAGE_ROOT", DATA_DIR / "images"))

MANIFEST_PATH = DATA_DIR / "manifest.parquet"
TEACHER_DIR = DATA_DIR / "teacher"          # embedding shards land here
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
EXPORT_DIR = OUT_DIR / "export"

# --- Teacher --------------------------------------------------------------
# Verified against the model card at
# https://huggingface.co/imageomics/bioclip-2.5-vith14
TEACHER_MODEL = "hf-hub:imageomics/bioclip-2.5-vith14"

# ViT-H/14 CLIP lineage -> 1024-d joint image/text space. Asserted at runtime
# in s02 rather than trusted blindly; if this ever changes, the taxa table and
# the student projection head must both change with it.
EMBED_DIM = 1024
IMAGE_SIZE = 224

# Batch size for the teacher pass. 64 fits comfortably in 12 GB at fp16;
# drop to 32 if you hit OOM while something else is using the GPU.
TEACHER_BATCH_SIZE = int(os.getenv("FLORA_TEACHER_BATCH", 64))
TEACHER_WORKERS = int(os.getenv("FLORA_TEACHER_WORKERS", 8))

# Images per shard file. 50k x 1024 x 2 bytes ~= 100 MB per shard, which is a
# convenient resume granularity: a crash costs you at most one shard.
SHARD_SIZE = 50_000

# --- Student --------------------------------------------------------------
# timm model name for the student backbone. Good options, cheapest first:
#   fastvit_t8    ~3.6M params  - fastest, weakest
#   fastvit_t12   ~6.8M params
#   fastvit_sa12  ~10.9M params - recommended starting point
#   fastvit_sa24  ~20.6M params - if sa12 loses too much accuracy
STUDENT_MODEL = os.getenv("FLORA_STUDENT", "fastvit_sa12")
STUDENT_PRETRAINED = True   # ImageNet warm start; far better than from scratch

# Sized for a 12 GB card. Training holds activations for the backward pass,
# so it needs far more memory per image than the teacher's inference-only
# pass in s02. Do not copy TEACHER_BATCH_SIZE here.
#
# On Windows this matters more than it looks: WDDM does not raise a clean OOM
# when VRAM runs out, it spills to system RAM over PCIe and everything gets
# ~50x slower before something eventually aborts inside a destructor. If you
# see seconds-per-iteration instead of iterations-per-second, this number is
# too high, whatever the error message says.
TRAIN_BATCH_SIZE = int(os.getenv("FLORA_TRAIN_BATCH", 64))

# Each Windows DataLoader worker is a full process that re-imports torch, and
# pinned prefetch buffers scale with workers x batch size. 4 keeps the GPU fed
# without pinning gigabytes of host RAM.
TRAIN_WORKERS = int(os.getenv("FLORA_TRAIN_WORKERS", 4))
EPOCHS = int(os.getenv("FLORA_EPOCHS", 30))
LEARNING_RATE = float(os.getenv("FLORA_LR", 1e-3))
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 2

# Loss mix. Cosine drives direction (all that matters after L2 normalization);
# a small smooth-L1 term stabilizes early training. See s03 for the rationale.
COSINE_WEIGHT = 1.0
L1_WEIGHT = 0.1

VAL_FRACTION = 0.02   # held out from the manifest for honest eval

# --- Taxa table -----------------------------------------------------------
# Newline-delimited scientific names, one per line. See s04 docstring.
TAXA_LIST_PATH = DATA_DIR / "taxa.txt"
TAXA_TABLE_PATH = EXPORT_DIR / "taxa_table.npy"
TAXA_LABELS_PATH = EXPORT_DIR / "taxa_labels.json"

# BioCLIP was trained with taxonomic-hierarchy captions, so prompting it the
# same way at lookup time measurably beats a bare species name. Embeddings for
# all templates are averaged, then re-normalized.
PROMPT_TEMPLATES = [
    "a photo of {name}.",
    "a photo of {name}, a type of plant.",
    "a close-up photo of the leaves of {name}.",
    "a photo of {name} in its natural habitat.",
]


def ensure_dirs() -> None:
    """Create every output directory. Safe to call repeatedly."""
    for d in (DATA_DIR, OUT_DIR, TEACHER_DIR, CHECKPOINT_DIR, EXPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)
