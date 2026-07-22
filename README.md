# bioclip-mobile-distill

Distills [BioCLIP 2.5 Huge](https://huggingface.co/imageomics/bioclip-2.5-vith14)
(ViT-H/14, roughly 632M parameters) into an 11.6M parameter FastViT student
that runs on a mid-range Android phone.

_See Hugging Face Model Here:
https://huggingface.co/crazedcodernate/bioclip-2.5-mobile-fastvit_

The student produces embeddings in the teacher's own coordinate space, so
species lookup on device is a single matrix multiply against a table of text
embeddings computed once on the desktop. The text encoder never ships to the
phone.

## Results

Trained on the plant subset of iNaturalist 2021 `train_mini`.

| | value |
|---|---|
| Teacher | BioCLIP 2.5 ViT-H/14, 1024-d embeddings |
| Student | `fastvit_sa12`, 11.6M params |
| Training images | 213,550 |
| Species | 4,271 (kingdom Plantae) |
| Epochs | 30 |
| Held-out embedding cosine vs teacher | 0.8383 |
| **top-1 agreement with teacher** | **71.7%** |
| top-5 contains teacher's top-1 | 88.8% |
| Shipped model | 23.8 MB fp16 ONNX |
| Taxa lookup table | 8.7 MB fp16 |

Random guessing across 4,271 species would score 0.023%.

Total training cost was about 3.5 hours on a single RTX 5070 (12 GB), with no
cloud compute. The teacher pass over 213,550 images ran at 120 images/second
and took 30 minutes. Student training took 6 minutes per epoch.

### Precision

fp16 is the recommended build. int8 was measured and rejected.

| build | size | cosine vs fp32 |
|---|---|---|
| fp32 | 47.0 MB | 1.0000 (reference) |
| **fp16** | **23.8 MB** | **1.0000** |
| int8, percentile calibration | 12.9 MB | 0.7897 |
| int8, minmax calibration | 12.9 MB | 0.3781 |

FastViT's stage 3 uses self-attention with LayerNorm, which does not survive
int8 quantization here regardless of calibration method. Quantization-aware
training would be the next lever if the extra 11 MB ever matters.

### Error characteristics

Most errors are same-genus. The model learned botanical structure and fails
at the final species split:

```
Xanthium orientale      -> Xanthium strumarium
Toxicodendron rydbergii -> Toxicodendron radicans
Abies sibirica          -> Abies alba
Ulmus thomasii          -> Ulmus americana
```

A minority of errors cross families, which matters if you intend to display
toxicity or edibility based on the predicted name. See Limitations.

## Pipeline

```
s00_prepare_inat.py     iNat21 to manifest + taxa list    minutes   \ pick
s01_build_manifest.py   any folder tree to manifest       seconds   / one
s02_cache_teacher.py    teacher embeds every image        ~30 min per 200k images
s03_train_student.py    student learns to copy them       ~6 min per epoch
s04_build_taxa_table.py species names to lookup matrix    minutes
s05_export.py           ONNX, fp16, int8 sweep            minutes
s06_eval.py             student vs teacher agreement      minutes
```

No labels are used anywhere. The teacher's cached embeddings are the
supervision, so any pile of biology images works whether or not anyone
annotated it.

## Setup

Requires Python 3.12 or 3.13. PyTorch has no 3.14 wheels.

```bash
py -3.13 -m venv .venv
source .venv/Scripts/activate     # Git Bash. PowerShell: .\.venv\Scripts\Activate.ps1

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python -c "import torch; print(torch.cuda.is_available())"
```

The cu128 index matters on Blackwell cards (RTX 50 series, compute capability
sm_120). Default PyPI wheels lack sm_120 kernels and fail at runtime with
`no kernel image is available for execution`.

## Data

```bash
mkdir -p /c/datasets/inat2021 && cd /c/datasets/inat2021
curl -L -O https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz
md5sum train_mini.tar.gz     # db6ed8330e634445efc8fec83ae81442
tar -xzf train_mini.tar.gz --wildcards '*_Plantae_*'
```

42 GB download, of which the plant subset is roughly 13 GB. iNat21 folder
names encode the full taxonomy, so `s00_prepare_inat.py` derives the species
list automatically:

```
04178_Plantae_Tracheophyta_Magnoliopsida_Asterales_Asteraceae_Achillea_millefolium
  id   kingdom    phylum        class       order     family    genus   species
```

## Running it

```bash
python s00_prepare_inat.py --inat-root /c/datasets/inat2021/train_mini
python s02_cache_teacher.py
python s03_train_student.py --epochs 30
python s04_build_taxa_table.py
python s05_export.py
python s06_eval.py --n 2000
```

Start with `--max-per-species 10` on `s00` for a fast end to end smoke test
before committing to the full run.

Both `s02` and `s03` are resumable. `s02` fingerprints the manifest and
refuses to resume onto a cache built from a different one, which is the
failure mode that silently mispairs images with embeddings.

## Interpreting val_cos

Do not read the training cosine as an accuracy proxy. Measured on this
project:

| val_cos | top-1 agreement | top-5 |
|---|---|---|
| 0.718 (10 img/species, 5 epochs) | 33.0% | 62.6% |
| 0.838 (50 img/species, 30 epochs) | 71.7% | 88.8% |

A 0.12 gain in cosine more than doubled top-1 accuracy. What matters is not
absolute closeness to the teacher but whether the student lands nearer the
correct species than the runner up, and that flips well before the cosine
looks impressive.

## Limitations

**Fine-grained species separation is where the loss lands.** BioCLIP 2.5 uses
a ViT-H trained on 200M images precisely because separating visually similar
species rewards scale. An 11.6M parameter student cannot hold that, and 28% of
top-1 predictions are wrong.

**Same-genus confusions usually preserve toxicity and edibility. Cross-family
confusions do not.** If your application displays safety information derived
from the predicted species, an on-device-only identification should show a
name but not a toxicity or edibility claim. Confirm those against a larger
model or a curated database.

**The taxa table sets its own ceiling.** Some species names sit almost on top
of each other in the teacher's text space (`Viola tricolor` and `Viola bicolor`
score 0.937 similarity). No image encoder can separate entries the lookup
table cannot distinguish. `s04` reports the worst pairs so you can see them
before shipping.

**Capacity is binding.** Training cosine plateaued at 0.9014 while validation
sat at 0.8412, so the student cannot fully match the teacher even on data it
trained on. `fastvit_sa24` (roughly 20.6M params) is the next step up if you
want more accuracy at double the on-device size.

## Attribution and licensing

The pipeline code in this repository is MIT licensed. See `LICENSE`.

**Teacher model.** BioCLIP 2.5 is MIT licensed. If you use this work, cite the
BioCLIP 2.5 paper, and consider citing OpenCLIP and the original BioCLIP as
the model card requests.

**Training data.** iNaturalist 2021 images are individually licensed by their
photographers under various Creative Commons licenses or CC0. Photographers
retain copyright. Attribution requirements vary per image, and the iNat
metadata carries the license, observer name, and observer login needed to
build an attribution statement.

This repository distributes **no images and no dataset**, only code that reads
a dataset you download yourself. If you redistribute trained weights, satisfy
yourself about the licensing position for your jurisdiction and use case,
since whether model weights constitute a derivative work of their training
images is not settled law.

## Acknowledgements

- [BioCLIP 2.5](https://huggingface.co/imageomics/bioclip-2.5-vith14) by the Imageomics Institute
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [timm](https://github.com/huggingface/pytorch-image-models) for the FastViT implementation
- [iNaturalist 2021 competition dataset](https://github.com/visipedia/inat_comp/tree/master/2021)
- Apple's MobileCLIP work, which established the mobile CLIP distillation recipe this follows
