---
license: mit
library_name: onnx
base_model: imageomics/bioclip-2.5-vith14
pipeline_tag: zero-shot-image-classification
tags:
  - biology
  - botany
  - plants
  - clip
  - knowledge-distillation
  - mobile
  - on-device
  - onnx
  - fastvit
  - species-identification
---

# BioCLIP 2.5 Mobile (FastViT student)

A 23.8 MB on-device plant identification model, distilled from
[BioCLIP 2.5 Huge](https://huggingface.co/imageomics/bioclip-2.5-vith14)
(ViT-H/14, roughly 632M parameters).

Author: **Nate Hamilton** ([CrazedCoderNate](https://github.com/CrazedCoderNate))
Training code: https://github.com/CrazedCoderNate/bioclip-mobile-distill

The student produces embeddings in the teacher's own 1024-dimensional joint
image and text space. Species lookup is therefore a single matrix multiply
against a precomputed table of text embeddings, which means the text encoder
never needs to run on the device. A phone runs the image encoder only.

## What it is for

Fast, offline, first-pass plant identification on mobile hardware, in front of
a larger model that arbitrates. It is not a replacement for a full-size
classifier, and the Limitations section below is not boilerplate.

## Results

Evaluated against the teacher on 2,000 held-out images, classifying across all
4,271 species in the lookup table.

| metric | value |
|---|---|
| top-1 agreement with teacher | **71.7%** |
| top-5 contains teacher's top-1 | 88.8% |
| embedding cosine vs teacher | 0.8383 |
| random baseline | 0.023% |

Model size and precision, measured by embedding cosine against the fp32 export
on real images:

| build | size | cosine vs fp32 |
|---|---|---|
| fp32 ONNX | 47.0 MB | 1.0000 (reference) |
| **fp16 ONNX (recommended)** | **23.8 MB** | **1.0000** |
| int8, percentile calibration | 12.9 MB | 0.7897 |
| int8, entropy calibration | 12.9 MB | 0.3781 |
| int8, minmax calibration | 12.9 MB | 0.3781 |

**int8 is not published because it does not work here.** FastViT's stage 3 uses
self-attention with LayerNorm, and no calibration method tested kept the
embedding usable. Ship fp16.

## Training

| | |
|---|---|
| Teacher | BioCLIP 2.5 ViT-H/14, 1024-d |
| Student | `fastvit_sa12` (timm), 11.6M params, ImageNet warm start |
| Objective | cosine distillation to cached teacher embeddings, plus 0.1 weight smooth L1 |
| Data | iNaturalist 2021 `train_mini`, kingdom Plantae only |
| Images | 213,550 |
| Species | 4,271 |
| Epochs | 30, batch 64, AdamW, OneCycleLR |
| Hardware | one RTX 5070 (12 GB) |
| Wall clock | 30 min teacher embedding pass, 3 hours student training |

No labels were used. The teacher's cached embeddings are the entire
supervision signal.

Final training cosine was 0.9014 against 0.8412 validation, so the student
does not fully match the teacher even on data it trained on. Capacity is
binding, and `fastvit_sa24` is the obvious next step for anyone wanting more.

## Limitations

### Do not use this for toxicity or edibility decisions

**This is the important one.** 28% of top-1 predictions are wrong. Most errors
are same-genus, which usually preserves toxic and edible properties, but a
meaningful minority cross families entirely:

```
Carya glabra (pignut hickory)  ->  Euonymus atropurpureus (burning bush)
Heuchera cylindrica            ->  Bowlesia incana
```

If your application displays whether a plant is poisonous or safe to eat, an
identification from this model alone must not drive that claim. Show a name,
and confirm against a larger model or a curated database before making any
safety statement. Foraging or medicinal decisions based on this model could
cause serious harm.

### Error pattern

The model learned genus-level botanical structure well and fails at the final
species split. The most frequent confusions:

```
Xanthium orientale       ->  Xanthium strumarium
Toxicodendron rydbergii  ->  Toxicodendron radicans
Abies sibirica           ->  Abies alba
Ulmus thomasii           ->  Ulmus americana
Oenothera rubricaulis    ->  Oenothera biennis
Fallopia scandens        ->  Fallopia convolvulus
```

This is the expected failure mode for a 55x parameter reduction. BioCLIP 2.5
uses a ViT-H trained on 200M images precisely because separating visually
similar species rewards scale, and an 11.6M parameter student cannot hold
that. Design around it with a two-stage system rather than trying to tune it
away.

### The lookup table has its own ceiling

Some species names sit almost on top of each other in the teacher's text
space. `Viola tricolor` and `Viola bicolor` score 0.937 cosine similarity as
text alone. No image encoder can separate entries that the lookup table cannot
distinguish, so these are a property of the approach, not of this student.

### Coverage and bias

The 4,271 species are those present in iNaturalist 2021 `train_mini` under
kingdom Plantae, at 50 images each. Coverage reflects where iNaturalist users
photograph plants, which skews toward North America and Europe. Species absent
from the table cannot be predicted at all, and will instead return whichever
listed species is nearest, with no signal that the true answer was missing.

### Evaluation caveat

`top1_agree` measures agreement with the teacher, not correctness against
ground-truth labels. Where BioCLIP 2.5 itself is wrong, this metric rewards
the student for reproducing that error.

## Files

| file | what it is |
|---|---|
| `flora_student_fp16.onnx` | the model to ship, 23.8 MB |
| `flora_student_fp32.onnx` | full precision reference, 47.0 MB |
| `student_weights.pt` | PyTorch weights, for further fine-tuning |
| `taxa_table.npy` | `[4271, 1024]` float32 L2-normalized text embeddings |
| `taxa_labels.json` | parallel list of species names |
| `eval_report.json` | the numbers above, as produced by the eval script |

## Usage

Input is a `[1, 3, 224, 224]` float32 tensor with RGB values in `0..1`.
ImageNet normalization is folded into the graph, so do not apply it yourself.
Output is a `[1, 1024]` L2-normalized embedding.

```python
import json
import numpy as np
import onnxruntime as ort
from PIL import Image

sess = ort.InferenceSession("flora_student_fp16.onnx")
table = np.load("taxa_table.npy")                      # [4271, 1024]
labels = json.load(open("taxa_labels.json"))

img = Image.open("plant.jpg").convert("RGB").resize((224, 224))
x = (np.asarray(img, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]

emb = sess.run(None, {"image": x})[0][0]               # already unit length
scores = table @ emb                                   # cosine similarity
top5 = np.argsort(-scores)[:5]

for i in top5:
    print(f"{scores[i]:.3f}  {labels[i]['scientific']}")
```

Because both the embedding and the table rows are unit vectors, the matrix
multiply is exactly cosine similarity. On Android, ONNX Runtime Mobile runs
the same graph.

## License and attribution

This model is released under the MIT License, matching its teacher.

**Teacher.** BioCLIP 2.5 by the Imageomics Institute, MIT licensed. If you use
this work, please cite the BioCLIP 2.5 paper, and consider citing OpenCLIP and
the original BioCLIP as that model card requests.

**Training data.** iNaturalist 2021 competition images, individually licensed
by their photographers under various Creative Commons licenses or CC0.
Photographers retain copyright. No images are redistributed here, only weights
derived from them. Whether model weights constitute a derivative work of their
training images is not settled law, and users with strict compliance needs
should form their own view.

**Built on.** [OpenCLIP](https://github.com/mlfoundations/open_clip) (MIT),
[timm](https://github.com/huggingface/pytorch-image-models) (Apache 2.0) for
the FastViT implementation, and Apple's MobileCLIP work, which established the
mobile CLIP distillation recipe this follows.

## Citation

```bibtex
@misc{hamilton2026bioclipmobile,
  author = {Hamilton, Nate},
  title  = {BioCLIP 2.5 Mobile: a FastViT student for on-device plant identification},
  year   = {2026},
  url    = {https://github.com/CrazedCoderNate/bioclip-mobile-distill}
}
```
