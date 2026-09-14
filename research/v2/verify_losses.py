"""S36 -- prove the three Track B objectives do what their docstrings claim, before any GPU
time is spent on them.

Same discipline as S31's synthetic suite, and for the same reason: a loss whose gradient
points somewhere subtly different from its stated population objective will train perfectly
happily, produce a checkpoint, and quietly invalidate every downstream comparison. Each
scenario below plants a situation where the *claimed* mechanism makes a specific prediction
that a plausible wrong implementation would not satisfy.

Every check runs on CPU in seconds and reads no data from disk.

    $py -m research.v2.verify_losses
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from research.v2.frontier import partial_auc
from research.v2.losses_escalation import (
    BandConditionalLogitAdjustment,
    BudgetConstrainedRankingLoss,
    WorstBandRankingLoss,
    band_conditional_log_priors,
    induced_escalation_logit,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED = 42
CHECKS: list[tuple[str, bool, str]] = []


def record(name: str, passed: bool, detail: str) -> None:
    CHECKS.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")


# ------------------------------------------------------------------ 1. the identity
def check_induced_identity() -> None:
    """`sigmoid(lambda)` must equal the deployed escalation mass exactly, or every arm is
    optimizing a different object from the one the evaluation scores."""
    torch.manual_seed(SEED)
    logits = torch.randn(500, 7) * 3.0
    esc = [0, 1, 4]
    lam = induced_escalation_logit(logits, esc)
    mass = torch.softmax(logits, dim=1)[:, esc].sum(dim=1)
    err = (torch.sigmoid(lam) - mass).abs().max().item()
    record("1. sigmoid(lambda) == escalation mass", err < 1e-6, f"max abs error {err:.2e}")


# ---------------------------------------------------- 2. the budget restriction is active
def check_tail_restriction() -> None:
    """N3 must be blind to reordering below the tail and sensitive to reordering inside it.

    A full-AUC objective would respond to both. This is the check that the `alpha`
    truncation is load-bearing rather than decorative.
    """
    esc = [0]
    n_neg = 100
    loss_a = BudgetConstrainedRankingLoss(esc, alpha=0.10, tau=1.0)
    loss_full = BudgetConstrainedRankingLoss(esc, alpha=1.00, tau=1.0)

    def batch(easy_neg_score: float) -> tuple[torch.Tensor, torch.Tensor]:
        # 10 positives at 1.0; 10 "tail" negatives at 2.0 (they outrank the positives);
        # 90 easy negatives whose score we vary well below the tail.
        lam = torch.cat([
            torch.full((10,), 1.0),
            torch.full((10,), 2.0),
            torch.full((n_neg - 10,), easy_neg_score),
        ])
        labels = torch.cat([torch.zeros(10, dtype=torch.long), torch.ones(n_neg, dtype=torch.long)])
        # build 2-class logits whose induced lambda equals `lam`
        logits = torch.stack([lam, torch.zeros_like(lam)], dim=1)
        return logits, labels

    low = loss_a(*batch(-5.0)).item()
    high = loss_a(*batch(-1.0)).item()          # easy negatives moved up, still below the tail
    low_full = loss_full(*batch(-5.0)).item()
    high_full = loss_full(*batch(-1.0)).item()

    tail_blind = abs(low - high) < 1e-6
    full_sensitive = abs(low_full - high_full) > 1e-3
    record(
        "2. alpha-tail restriction is load-bearing",
        tail_blind and full_sensitive,
        f"N3(alpha=0.10) unchanged by easy-negative reordering ({low:.6f} vs {high:.6f}); "
        f"full-AUC control does change ({low_full:.4f} vs {high_full:.4f})",
    )


# ------------------------------------------------- 3. N3 actually increases pAUC at alpha
def check_n3_optimizes_pauc() -> None:
    """Gradient descent on N3 must raise the quantity N3 claims to maximize."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    n = 800
    x = torch.tensor(rng.normal(size=(n, 4)), dtype=torch.float32)
    w_true = torch.tensor([1.5, -1.0, 0.0, 0.0])
    p = torch.sigmoid(x @ w_true - 1.2)
    y_esc = torch.bernoulli(p).bool()
    labels = torch.where(y_esc, 0, 1).long()  # class 0 escalating, class 1 benign

    head = torch.nn.Linear(4, 2)
    opt = torch.optim.Adam(head.parameters(), lr=0.05)
    loss_fn = BudgetConstrainedRankingLoss([0], alpha=0.20, tau=1.0)

    def pauc_now() -> float:
        with torch.no_grad():
            lam = induced_escalation_logit(head(x), [0]).numpy()
        return partial_auc(y_esc.numpy(), lam, 0.20)["partial_auc_mcclish"]

    before = pauc_now()
    for _ in range(300):
        opt.zero_grad()
        loss_fn(head(x), labels).backward()
        opt.step()
    after = pauc_now()
    record("3. N3 descent raises pAUC@0.20", after > before + 0.02,
           f"McClish pAUC {before:.4f} -> {after:.4f}")


# ------------------------------------------------------ 4. N4's adversary finds the worst band
def check_n4_adversary_concentrates() -> None:
    """`q` must migrate onto the band with the largest risk. A wrong-signed update would
    concentrate on the *best* band and silently optimize the wrong group."""
    torch.manual_seed(SEED)
    esc = [0]
    loss = WorstBandRankingLoss(esc, n_bands=3, alpha=0.5, tau=1.0, eta_q=0.5)

    # band 0 ranked badly (positives below negatives), bands 1 and 2 ranked well.
    lam = torch.cat([
        torch.tensor([-2.0] * 8 + [2.0] * 8),   # band 0: positives low, negatives high
        torch.tensor([2.0] * 8 + [-2.0] * 8),   # band 1: correct
        torch.tensor([2.0] * 8 + [-2.0] * 8),   # band 2: correct
    ])
    labels = torch.cat([torch.zeros(8, dtype=torch.long), torch.ones(8, dtype=torch.long)]).repeat(3)
    bands = torch.arange(3).repeat_interleave(16)
    logits = torch.stack([lam, torch.zeros_like(lam)], dim=1)

    for _ in range(50):
        loss(logits, labels, bands)
    q = loss.q.detach().numpy()
    record("4. N4 adversary concentrates on the worst band", q[0] > 0.8,
           f"q = [{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}], worst band is 0")


# ------------------------------------- 5. N4 reduces to band-averaged N3 when bands are alike
def check_n4_reduces_to_n3() -> None:
    """With identical bands the adversary has nothing to find, so the minimax must collapse
    to the uniform average -- otherwise N4 is adding an effect where none exists."""
    torch.manual_seed(SEED)
    esc = [0]
    lam = torch.randn(180)
    labels = torch.cat([torch.zeros(30, dtype=torch.long), torch.ones(30, dtype=torch.long)]).repeat(3)
    bands = torch.arange(3).repeat_interleave(60)
    logits = torch.stack([lam, torch.zeros_like(lam)], dim=1)

    dro = WorstBandRankingLoss(esc, n_bands=3, alpha=0.3, tau=1.0, eta_q=0.01)
    uniform = WorstBandRankingLoss(esc, n_bands=3, alpha=0.3, tau=1.0,
                                    fixed_q=torch.ones(3) / 3)
    for _ in range(5):
        dro_value = dro(logits, labels, bands).item()
    uniform_value = uniform(logits, labels, bands).item()
    q = dro.q.detach().numpy()
    close = abs(dro_value - uniform_value) < 0.05 and q.std() < 0.05
    record("5. N4 collapses to uniform when bands are alike", close,
           f"DRO {dro_value:.4f} vs uniform {uniform_value:.4f}, q sd {q.std():.4f}")


# ---------------------------------------- 6. N5 recovers a prior-free likelihood ratio
def _two_band_prior_shift_data(rng: np.random.Generator, n: int = 4000):
    """Two bands, identical class-conditional densities, very different priors.

    band 0: P(escalating) = 0.05      band 1: P(escalating) = 0.50
    p(x | benign) = N(0, 1), p(x | escalating) = N(2, 1) in both bands.
    True prior-free log-LR is therefore `2x - 2`, identical in both bands.
    """
    bands = rng.integers(0, 2, size=n)
    priors = np.array([0.05, 0.50])
    y_esc = rng.random(n) < priors[bands]
    x = rng.normal(loc=np.where(y_esc, 2.0, 0.0), scale=1.0)
    feats = np.stack([x, bands == 0, bands == 1], axis=1).astype(np.float32)
    labels = np.where(y_esc, 0, 1).astype(np.int64)  # class 0 escalating
    return (torch.tensor(feats), torch.tensor(labels), torch.tensor(bands, dtype=torch.long),
            torch.tensor(y_esc), torch.tensor((2.0 * x - 2.0), dtype=torch.float32))


def _fit(loss_fn, feats, labels, bands, steps: int = 600) -> torch.nn.Linear:
    torch.manual_seed(SEED)
    head = torch.nn.Linear(3, 2)
    opt = torch.optim.Adam(head.parameters(), lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        loss_fn(head(feats), labels, bands).backward()
        opt.step()
    return head


def check_n5_recovers_likelihood_ratio() -> None:
    """Plain CE's induced score must sit a *band-dependent* distance from the true log-LR;
    N5's must sit a *band-independent* distance from it.

    This is the sharpest available test of the derivation, and it is the one that decides
    what N5 can and cannot be expected to improve downstream.
    """
    rng = np.random.default_rng(SEED)
    feats, labels, bands, y_esc, true_llr = _two_band_prior_shift_data(rng)

    ce = torch.nn.CrossEntropyLoss()
    head_ce = _fit(lambda lo, la, b: ce(lo, la), feats, labels, bands)

    log_prior = band_conditional_log_priors(labels.numpy(), bands.numpy(), n_bands=2, n_classes=2)
    n5 = BandConditionalLogitAdjustment(log_prior, t=1.0)
    head_n5 = _fit(n5, feats, labels, bands)

    def band_offsets(head) -> np.ndarray:
        with torch.no_grad():
            lam = induced_escalation_logit(head(feats), [0])
        residual = (lam - true_llr).numpy()
        return np.array([residual[bands.numpy() == g].mean() for g in (0, 1)])

    off_ce, off_n5 = band_offsets(head_ce), band_offsets(head_n5)
    spread_ce = abs(off_ce[0] - off_ce[1])
    spread_n5 = abs(off_n5[0] - off_n5[1])
    expected_ce = abs(np.log(0.05 / 0.95) - np.log(0.50 / 0.50))
    record(
        "6. N5 removes the band-dependent prior offset",
        spread_n5 < 0.35 and spread_ce > 1.5,
        f"cross-band offset spread: CE {spread_ce:.3f} (theory {expected_ce:.3f}), N5 {spread_n5:.3f}",
    )


def check_n5_effect_is_invisible_within_band() -> None:
    """The consequence that governs how N5 must be evaluated.

    A band-constant offset cannot change a *within-band* ranking, so within-band partial AUC
    is invariant to it. If N5's benefit were expected to show up in band-conditional pAUC
    the evaluation would be pointed at a metric that is blind to the mechanism by
    construction. Assert the invariance and the corresponding shift in TPR at a single
    global threshold, which is where the effect does appear.
    """
    rng = np.random.default_rng(SEED)
    feats, labels, bands, y_esc, _ = _two_band_prior_shift_data(rng)
    band_np, y_np = bands.numpy(), y_esc.numpy()

    ce = torch.nn.CrossEntropyLoss()
    head_ce = _fit(lambda lo, la, b: ce(lo, la), feats, labels, bands)
    log_prior = band_conditional_log_priors(labels.numpy(), band_np, n_bands=2, n_classes=2)
    head_n5 = _fit(BandConditionalLogitAdjustment(log_prior, t=1.0), feats, labels, bands)

    def scores(head) -> np.ndarray:
        with torch.no_grad():
            return induced_escalation_logit(head(feats), [0]).numpy()

    s_ce, s_n5 = scores(head_ce), scores(head_n5)

    within = []
    for g in (0, 1):
        rows = band_np == g
        within.append((
            partial_auc(y_np[rows], s_ce[rows], 0.20)["partial_auc_mcclish"],
            partial_auc(y_np[rows], s_n5[rows], 0.20)["partial_auc_mcclish"],
        ))
    within_gap = max(abs(a - b) for a, b in within)

    def tpr_gap_at_global_threshold(s: np.ndarray) -> float:
        tau = np.quantile(s[~y_np], 0.80)  # one global threshold at 20% overall FPR
        refer = s >= tau
        tprs = [refer[(band_np == g) & y_np].mean() for g in (0, 1)]
        return abs(tprs[0] - tprs[1])

    gap_ce = tpr_gap_at_global_threshold(s_ce)
    gap_n5 = tpr_gap_at_global_threshold(s_n5)
    record(
        "7. N5 is invisible within band, visible at a global threshold",
        within_gap < 0.05 and gap_n5 < gap_ce - 0.05,
        f"max within-band pAUC difference {within_gap:.4f} (expected ~0); "
        f"cross-band TPR gap at one global threshold: CE {gap_ce:.3f} -> N5 {gap_n5:.3f}",
    )


def _append_ledger(passed: int, total: int) -> None:
    """One row, pruning this runner's own prior rows -- the repository has twice been bitten
    by runners that append a duplicate set on every re-run."""
    import pandas as pd

    path = REPO_ROOT / "research" / "experiments.csv"
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session": "v2_s36", "method": "loss_validation[N3,N4,N5]", "split": "synthetic",
        "macro_f1": "", "accuracy": "", "balanced_accuracy": "", "weighted_f1": "",
        "macro_roc_auc": "", "ece": "", "escalation_sens": "", "missed_serious": "",
        "p_value_vs_baseline": "",
        "notes": (
            f"S36 loss validation {passed}/{total} checks passed on synthetic data, no disk read; "
            f"confirms sigmoid(lambda)==escalation mass, N3's alpha-tail truncation is "
            f"load-bearing, N3 descent raises pAUC@0.20, N4's adversary finds the worst band and "
            f"collapses to uniform when bands are alike, N5 removes the band-dependent prior "
            f"offset and is provably invisible to within-band partial AUC"
        ),
    }
    frame = pd.DataFrame([row])
    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[~((existing["session"] == "v2_s36") & (existing["method"] == row["method"]))]
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(path, index=False)


def main() -> int:
    print("S36 -- Track B loss validation (CPU, no data read)\n")
    check_induced_identity()
    check_tail_restriction()
    check_n3_optimizes_pauc()
    check_n4_adversary_concentrates()
    check_n4_reduces_to_n3()
    check_n5_recovers_likelihood_ratio()
    check_n5_effect_is_invisible_within_band()

    passed = sum(1 for _, ok, _ in CHECKS if ok)
    total = len(CHECKS)
    print(f"\nS36 loss validation: {passed}/{total} checks passed")
    _append_ledger(passed, total)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
