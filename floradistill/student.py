"""The mobile student: a small vision backbone plus a projection head.

The head exists because the backbone's native feature width (e.g. 1024 for
fastvit_sa12 after pooling) has no reason to match the teacher's 1024-d joint
space, and the two spaces are unrelated even when the numbers coincide. The
projection is what actually lands the student in the teacher's coordinate
system, and it is the only part that must change if you ever swap teachers.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class StudentEncoder(nn.Module):
    """Image -> L2-normalized embedding in the teacher's space."""

    def __init__(
        self,
        backbone_name: str,
        embed_dim: int,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        # num_classes=0 strips the classifier and returns pooled features.
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0
        )
        feat_dim = self.backbone.num_features

        # A single linear layer is enough and stays cheap on-device. An MLP
        # head trains marginally better but costs latency at every inference
        # forever, which is the wrong trade for the thing that runs on a phone.
        self.proj = nn.Linear(feat_dim, embed_dim)
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        emb = self.proj(feats)
        return F.normalize(emb, dim=-1)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ExportWrapper(nn.Module):
    """Inference-only wrapper used at export time.

    Folds the ImageNet mean/std normalization into the graph so the Android
    side can hand over a plain 0..1 RGB tensor and not have to keep two sets
    of magic constants in sync across a language boundary.
    """

    def __init__(self, student: StudentEncoder, mean, std) -> None:
        super().__init__()
        self.student = student
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.student((x - self.mean) / self.std)


def distillation_loss(
    student_emb: torch.Tensor,
    teacher_emb: torch.Tensor,
    cosine_weight: float,
    l1_weight: float,
) -> tuple[torch.Tensor, float]:
    """Match the student's embedding to the teacher's.

    Both inputs are unit vectors, so cosine similarity is a dot product and
    `1 - cos` is the real objective: only direction matters after
    normalization. The smooth-L1 term is a small stabilizer for early
    training, when a randomly initialized projection head can otherwise sit at
    near-zero cosine and give a very flat gradient.

    Returns (loss, mean_cosine_similarity). The cosine is the number to watch;
    it is directly interpretable as "how close is the student to the teacher".
    """
    cos = F.cosine_similarity(student_emb, teacher_emb, dim=-1)
    loss = cosine_weight * (1.0 - cos).mean()
    if l1_weight > 0:
        loss = loss + l1_weight * F.smooth_l1_loss(student_emb, teacher_emb)
    return loss, cos.mean().item()
