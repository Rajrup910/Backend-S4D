"""Epoch-level weight averaging for every V6 run (docs/V6_RUNSHEET.md section 0.4, V6-0).

This reuses `research.v5.m7.SwadAverager` unchanged: one snapshot per epoch, so it is SWA, **not** iteration-dense SWAD.
Two things V5's use did not need:
  * the window scales with the step budget: the last two thirds of the epochs, so a run shortened to hold optimiser steps
    constant (section 0.4) averages the same share of its trajectory (30 epochs -> 11..30, 20 snapshots);
  * BatchNorm members. `SwadAverager` averages running mean/var as if they were weights, which is wrong, so after loading
    the average a BN model gets `reestimate_bn`: reset the statistics and recompute them as a cumulative average over
    training images. Models with no BatchNorm (ConvNeXt, every ViT) need nothing.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from research.v5.m7 import SwadAverager

WINDOW_FRACTION = 2 / 3


def window_for(epochs: int) -> tuple[int, int]:
    """1-indexed inclusive (start, end): the last two thirds of `epochs`."""
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    length = max(1, round(epochs * WINDOW_FRACTION))
    return epochs - length + 1, epochs


class EpochAverager(SwadAverager):
    def __init__(self, epochs: int) -> None:
        super().__init__(*window_for(epochs))
        self.epochs = epochs


def _batch_norms(model: nn.Module) -> list[nn.modules.batchnorm._BatchNorm]:
    return [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]


def has_batchnorm(model: nn.Module) -> bool:
    return bool(_batch_norms(model))


@torch.no_grad()
def reestimate_bn(model: nn.Module, batches: Iterable[torch.Tensor], device: torch.device | str = "cpu",
                  max_batches: int | None = None) -> int:
    """Recompute BatchNorm running statistics from `batches` (image tensors of training images, no labels).

    Statistics are reset and accumulated as a cumulative average (momentum=None). Dropout and stochastic depth stay in eval
    mode so only the statistics change. Returns the number of batches used; a model without BatchNorm is untouched.
    """
    norms = _batch_norms(model)
    if not norms:
        return 0
    saved = [m.momentum for m in norms]
    model.eval()
    for m in norms:
        m.reset_running_stats()
        m.momentum = None
        m.train()
    used = 0
    try:
        for batch in batches:
            if max_batches is not None and used >= max_batches:
                break
            model(batch.to(device))
            used += 1
    finally:
        for m, momentum in zip(norms, saved):
            m.momentum = momentum
        model.eval()
    if used == 0:
        raise ValueError("reestimate_bn received no batches; the statistics were reset and are now meaningless")
    return used
