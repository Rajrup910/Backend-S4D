"""Clinical cost matrix C_ij: penalty of predicting class j when the truth is class i.

Built from `class_mapping.json`'s malignancy tiers rather than hardcoded indices, so it
stays correct if classes are ever added/reordered. The asymmetry follows straightforward
clinical reasoning, not a fitted quantity -- there is no ground truth for "how many QALYs
does a missed melanoma cost", so these are illustrative weights, documented here so a
domain expert can substitute better ones without touching the optimizer:

  - Missing a malignant lesion (predicting it benign) is the worst outcome: melanoma can
    metastasize, so it is weighted above basal cell carcinoma, which grows slowly.
  - Missing the premalignant class (predicting it benign) is bad but recoverable --
    actinic keratosis progresses slowly and is easy to re-catch at a follow-up.
  - Confusing two escalating classes (e.g. calling a melanoma an actinic keratosis) still
    routes the patient to a biopsy, so the cost is low even though the label is wrong.
  - A false alarm (predicting escalation for a true benign lesion) costs an unnecessary
    referral -- real, but far cheaper than a missed cancer.
  - Confusing two benign classes has no clinical consequence beyond a wrong label.
"""

from __future__ import annotations

import numpy as np

from ml.paths import ClassMapping, load_class_mapping

MISSED_MALIGNANT_COST = {"mel": 10.0, "bcc": 8.0}  # per-code override; others fall back below
DEFAULT_MISSED_MALIGNANT_COST = 9.0
MISSED_PREMALIGNANT_COST = 5.0
ESCALATION_CONFUSION_COST = 1.0  # both true and predicted classes escalate, just the wrong one
FALSE_ALARM_COST = 1.0  # true benign, predicted escalating
BENIGN_CONFUSION_COST = 0.5  # both benign, different class


def build_cost_matrix(mapping: ClassMapping | None = None) -> np.ndarray:
    mapping = mapping or load_class_mapping()
    c = mapping.num_classes
    cost = np.zeros((c, c))

    for true_class in mapping.classes:
        for pred_class in mapping.classes:
            if true_class.index == pred_class.index:
                continue
            true_escalates = true_class.needs_escalation
            pred_escalates = pred_class.needs_escalation

            if true_escalates and not pred_escalates:
                if true_class.malignancy == "malignant":
                    value = MISSED_MALIGNANT_COST.get(true_class.code, DEFAULT_MISSED_MALIGNANT_COST)
                else:
                    value = MISSED_PREMALIGNANT_COST
            elif true_escalates and pred_escalates:
                value = ESCALATION_CONFUSION_COST
            elif not true_escalates and pred_escalates:
                value = FALSE_ALARM_COST
            else:
                value = BENIGN_CONFUSION_COST

            cost[true_class.index, pred_class.index] = value

    return cost


def expected_cost(y_true: np.ndarray, y_pred: np.ndarray, cost_matrix: np.ndarray) -> float:
    return float(cost_matrix[y_true, y_pred].mean())
