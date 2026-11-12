"""Build B3's objective: one-way partial AUC (FPR <= BETA) on the escalation score, added to V6's cross-entropy.

The owner's report names LibAUC `pAUCLoss('1w', margin=0.6, gamma=0.1)` with SOPA-s and DualSampler. LibAUC is not installed
in this venv (`import libauc` fails, checked 10 Nov 2026), so V7 declares an in-repo surrogate of the same family instead:
a squared hinge on positive-negative gaps, restricted in each batch to the top ceil(BETA * n_negative) negatives, i.e. the
negatives that decide the low-FPR region. SOPA-s weights negatives softly (a KL-regularised version of the same top set);
the hard top-k form is its limit as the temperature goes to zero. Equivalence of results is NOT claimed. If the owner
prefers LibAUC itself, install it and swap `OneWayPAUCLoss` for it; the build entry records which ran.

The escalation score is the logit of the escalating classes' probability mass: logsumexp(escalating logits) minus
logsumexp(other logits). The cross-entropy term keeps the 7-class head, so the member can still join the soft vote.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

ESCALATING_7 = (0, 1, 4)  # akiec, bcc, mel in head8.CLASSES_7 order
BETA = 0.20
MARGIN = 0.6
PAUC_WEIGHT = 1.0


def escalation_logit(logits: torch.Tensor, escalating: tuple[int, ...] = ESCALATING_7) -> torch.Tensor:
    mask = torch.zeros(logits.shape[1], dtype=torch.bool, device=logits.device)
    mask[list(escalating)] = True
    return torch.logsumexp(logits[:, mask], dim=1) - torch.logsumexp(logits[:, ~mask], dim=1)


class OneWayPAUCLoss(nn.Module):
    def __init__(self, beta: float = BETA, margin: float = MARGIN) -> None:
        super().__init__()
        if not 0 < beta <= 1:
            raise ValueError("beta is a false-positive-rate cap in (0, 1]")
        self.beta, self.margin = beta, margin

    def forward(self, score: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
        pos, neg = score[positive], score[~positive]
        if pos.numel() == 0 or neg.numel() == 0:
            return score.sum() * 0.0  # a batch without both classes contributes nothing, and keeps the graph
        k = max(1, math.ceil(self.beta * neg.numel()))
        hard = torch.topk(neg, k).values
        return torch.clamp(self.margin - (pos[:, None] - hard[None, :]), min=0).pow(2).mean()


class CrossEntropyPlusPAUC(nn.Module):
    """criterion(logits, labels) with the same call signature as V6's CrossEntropyLoss, so the trainer is unchanged."""

    def __init__(self, cross_entropy: nn.Module, weight: float = PAUC_WEIGHT, beta: float = BETA, margin: float = MARGIN,
                 escalating: tuple[int, ...] = ESCALATING_7) -> None:
        super().__init__()
        self.cross_entropy, self.weight, self.escalating = cross_entropy, weight, escalating
        self.pauc = OneWayPAUCLoss(beta, margin)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        positive = torch.isin(labels, torch.tensor(self.escalating, device=labels.device))
        return self.cross_entropy(logits, labels) + self.weight * self.pauc(escalation_logit(logits.float(), self.escalating), positive)
