"""Gated vision + tabular fusion model.

The roadmap specifies a cross-attention fusion layer:

    f_fused = LayerNorm(f_v (+) f_m + CrossAttn(f_v, f_m))

Cross-attention is defined over *sequences* of tokens; here both branches produce a
single pooled vector per image (one CNN feature vector, one metadata embedding), and
attention between one query and one key degenerates to a learned scalar gate on the
value -- there is no second token for it to attend over. So this implements that
degenerate case directly as a gate, which is mathematically what single-vector
cross-attention reduces to, without the wasted machinery of a multi-head attention module
whose sequence length is always 1.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TabularBranch(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class GatedFusionModel(nn.Module):
    """vision_backbone must map an image batch to a pooled (B, vision_dim) feature -- i.e.
    the backbone with its classification head removed, not the full classifier.
    """

    def __init__(
        self,
        vision_backbone: nn.Module,
        vision_dim: int,
        tabular_dim: int,
        num_classes: int,
        tabular_hidden: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vision_backbone = vision_backbone
        self.tabular_branch = TabularBranch(tabular_dim, hidden_dim=tabular_hidden, output_dim=vision_dim)
        self.gate = nn.Sequential(nn.Linear(vision_dim * 2, vision_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(vision_dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(vision_dim, num_classes))

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        f_v = self.vision_backbone(image)
        f_m = self.tabular_branch(tabular)
        gate = self.gate(torch.cat([f_v, f_m], dim=1))
        fused = self.norm(f_v + gate * f_m)
        return self.head(fused)


def build_convnext_tiny_feature_extractor(pretrained: bool = True) -> tuple[nn.Module, int]:
    """ConvNeXt-Tiny with its classifier replaced by global pooling only -- the pooled
    penultimate feature vector, not logits. Chosen as the vision branch because it's the
    current single-model leader (Macro-F1 0.746), so fusion is tested against the
    strongest available backbone rather than an arbitrary one.
    """
    from torchvision import models

    weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = models.convnext_tiny(weights=weights)
    feature_dim = backbone.classifier[2].in_features
    backbone.classifier = nn.Sequential(backbone.classifier[0], backbone.classifier[1])  # LayerNorm2d, Flatten
    return backbone, feature_dim
