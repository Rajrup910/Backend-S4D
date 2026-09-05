"""Class-imbalanced margin losses: LDAM-DRW and ASL, as alternatives to weighted CE.

Both exist because `effective_number` class weighting (the current default, see
`compute_class_weights`) reweights the *loss magnitude* per class but leaves the decision
boundary itself untouched. These two instead reshape the boundary or the gradient:

- **LDAM-DRW** (Cao et al., NeurIPS 2019) enforces a larger margin for rare classes at
  train time, then defers class reweighting to the last few epochs (DRW) so the network
  first learns good features before the reweighting perturbs the boundary -- reweighting
  from epoch 1 tends to hurt representation learning on a backbone this small.
- **ASL** (Ridnik et al., ICCV 2021) was built for multi-label classification; the
  single-label adaptation here applies its asymmetric focusing (down-weight easy negatives
  harder than easy positives, shift positives away from the decision boundary) to a
  one-hot target on top of softmax rather than independent sigmoids per label.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LDAMLoss(nn.Module):
    """Label-Distribution-Aware Margin loss. Class reweighting is applied externally
    (pass `weight` once the DRW schedule activates it) -- this module only owns the margin.
    """

    def __init__(self, class_counts: list[int], max_margin: float = 0.5, scale: float = 30.0):
        super().__init__()
        counts = np.asarray(class_counts, dtype=np.float64)
        counts = np.maximum(counts, 1.0)  # guard against a class with zero training samples
        margins = 1.0 / np.power(counts, 0.25)
        margins = margins * (max_margin / margins.max())
        self.register_buffer("margins", torch.tensor(margins, dtype=torch.float32))
        self.scale = scale
        # Settable by the training loop each epoch (the DRW schedule): None until the
        # deferred-reweighting stage begins, then the effective-number weight tensor.
        # A forward-call kwarg would require threading a schedule through `run_epoch`,
        # which every other loss in this file also has to share -- an attribute keeps
        # `criterion(logits, labels)` the same call for every loss variant.
        self.weight: torch.Tensor | None = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        margin_per_sample = self.margins[target]
        adjusted = logits - F.one_hot(target, logits.size(1)).float() * margin_per_sample.unsqueeze(1)
        return F.cross_entropy(self.scale * adjusted, target, weight=self.weight)


class AsymmetricLoss(nn.Module):
    """Single-label adaptation of ASL: asymmetric focusing on softmax probabilities.

    gamma_pos focuses the loss on hard positives (the true class); gamma_neg focuses much
    harder on hard negatives, i.e. it barely penalizes negative classes the model is
    already confidently rejecting -- which is what lets it stop the vast `nv` majority
    from dominating every gradient step. `clip` hard-zeros negative probabilities below a
    floor before they contribute anything, the same "easy negative" suppression the paper
    uses for multi-label tagging.
    """

    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 4.0, clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.weight: torch.Tensor | None = None  # see LDAMLoss.weight docstring

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(target, logits.size(1)).float()

        pos_probs = probs
        neg_probs = (1.0 - probs + self.clip).clamp(max=1.0)

        pos_loss = one_hot * torch.log(pos_probs.clamp_min(self.eps)) * (1.0 - pos_probs).pow(self.gamma_pos)
        neg_loss = (1.0 - one_hot) * torch.log(neg_probs.clamp_min(self.eps)) * (probs).pow(self.gamma_neg)

        loss = -(pos_loss + neg_loss)
        if self.weight is not None:
            loss = loss * self.weight.unsqueeze(0)
        return loss.sum(dim=1).mean()


def build_drw_weights(class_counts: list[int], beta: float = 0.999) -> torch.Tensor:
    """Effective-number weights (Cui et al., 2019), reused as the DRW reweighting stage."""
    counts = np.asarray(class_counts, dtype=np.float64)
    present = counts > 0
    weights = np.zeros_like(counts)
    effective = (1.0 - np.power(beta, counts[present])) / (1.0 - beta)
    weights[present] = 1.0 / effective
    weights = weights / weights[present].mean()
    return torch.tensor(weights, dtype=torch.float32)
