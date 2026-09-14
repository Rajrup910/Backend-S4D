"""S36 -- the three Track B training objectives (N3, N4, N5), and the induced escalation
logit all three of them shape.

Why these three, and why not more class-imbalance losses. S34 ruled out the decision rule
(`C` ~ 0 with tight intervals) and failed to certify a score-choice gap (`B` = 0), leaving
the ranking as the only place the under-40 deficit can live; S35's N2 probe then showed a
plain linear head on the *existing* ConvNeXt-Tiny features reaches 0.993 under-40 partial
AUC against the deployed posterior's 0.810, i.e. the representation already carries
escalation signal the 7-way head does not expose. Both results point at the same target:
the *induced escalation score*, not generic class imbalance. The repository has already
tried generic class imbalance twice -- LDAM-DRW (test Macro-F1 0.7256) and ASL (0.7305),
both below the plain-CE ConvNeXt-Tiny baseline (0.7459) -- so a third reweighting variant
would be a fourth negative result, not an experiment.

**The object every arm shapes.** For logits `z(x)` and the escalating class set
`E = {akiec, bcc, mel}`:

    lambda(x) = logsumexp_{c in E} z_c(x) - logsumexp_{c not in E} z_c(x)

`sigmoid(lambda(x))` equals `sum_{c in E} softmax(z(x))_c` **exactly** -- the deployed
escalation mass `s(x)`. So `lambda` is a strictly monotone transform of the deployed score,
every ranking claim about one is a ranking claim about the other, and an arm that improves
`lambda` improves the score the deployed pipeline already uses without changing that
pipeline. This is checked numerically in `--selftest`, not asserted.

**The prohibition this module respects.** The blueprint forbids calling a weighted sum of
existing terms a new loss. None of the three is one: N3 is a *pairwise* objective over a
*data-dependent* subset of negatives (not decomposable into per-example terms at all), N4
is a *saddle point* of a minimax program whose weights are the adversary's variable rather
than hyperparameters, and N5 is an additive shift *inside* the softmax that changes the
estimand from the posterior to a likelihood ratio (Menon et al. are explicit that this is
not loss reweighting, and Byrd & Lipton 2019 is the reason it matters that it is not).

**No auxiliary cross-entropy term.** N3 and N4 do not supervise discrimination *within* the
escalating and benign groups at all, so a naive from-scratch run would destroy the 7-class
head. The usual fix -- `L = L_rank + beta * L_CE` -- is exactly the forbidden weighted sum.
Instead both arms are specified as *fine-tunes* of the frozen HAM-only checkpoint at low
learning rate with val early stopping, so within-group structure is preserved by the
initialization rather than by an added term. The CE-augmented variant is in the ablation
plan, labelled as an ablation, and is not the headline arm.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-12


# --------------------------------------------------------------------- the induced score
def induced_escalation_logit(logits: torch.Tensor, esc_idx: list[int]) -> torch.Tensor:
    """`lambda(x)`, the log-odds of the escalating super-class under the model's own softmax.

    Derivation: with `A = logsumexp_{c in E} z_c` and `B = logsumexp_{c not in E} z_c`,
    `sum_{c in E} softmax(z)_c = e^A / (e^A + e^B) = sigmoid(A - B)`. No new parameters, no
    second head: the escalation score is a deterministic function of the same 7 logits the
    deployed system already produces.
    """
    non_esc = [c for c in range(logits.size(1)) if c not in esc_idx]
    return torch.logsumexp(logits[:, esc_idx], dim=1) - torch.logsumexp(logits[:, non_esc], dim=1)


def escalating_mask(labels: torch.Tensor, esc_idx: list[int]) -> torch.Tensor:
    out = torch.zeros_like(labels, dtype=torch.bool)
    for c in esc_idx:
        out |= labels == c
    return out


def _zero_like(logits: torch.Tensor) -> torch.Tensor:
    """A differentiable zero: a batch with no positives (or no negatives) contributes no
    signal, but returning a detached constant would break `.backward()`."""
    return logits.sum() * 0.0


# ------------------------------------------------------------------------------- N3
def tail_ranking_risk(
    lam: torch.Tensor, y_esc: torch.Tensor, alpha: float, tau: float,
) -> torch.Tensor:
    """The empirical tail-conditional pairwise risk shared by N3 and N4.

    Separated out so that N4 is *provably* N3's risk under a different aggregation rather
    than a second, subtly different risk that would make the pooled-vs-worst-band contrast
    uninterpretable.
    """
    pos, neg = lam[y_esc], lam[~y_esc]
    if pos.numel() == 0 or neg.numel() == 0:
        return _zero_like(lam)
    k = max(1, math.ceil(alpha * neg.numel()))
    tail = torch.topk(neg, k).values                       # the negatives that set the threshold
    return F.softplus(-(pos.unsqueeze(1) - tail.unsqueeze(0)) / tau).mean()


class BudgetConstrainedRankingLoss(nn.Module):
    """**N3 -- budget-constrained escalation ranking (partial AUC over FPR in [0, alpha]).**

    *Population objective.* With `P = law(x | y in E)` and `N = law(x | y not in E)`, and
    `t_alpha` the `(1 - alpha)`-quantile of `lambda` under `N`,

        maximize  (1/alpha) * pAUC_alpha(lambda)
                = E_{x ~ P} [ P_{x' ~ N}( lambda(x) > lambda(x') | lambda(x') >= t_alpha ) ]

    the probability that an escalating lesion outranks a benign lesion **drawn from the top
    `alpha` tail of the benign distribution**. That conditioning is the whole point: at a
    referral budget only the highest-scoring benign lesions compete for capacity, and
    whether an escalating lesion outranks an obviously-benign one is clinically irrelevant.
    Full AUC weights those irrelevant pairs equally, which is why two scores with equal full
    AUC can differ sharply at a fixed budget.

    *Empirical form.* Per minibatch, take the top `k = ceil(alpha * n_neg)` negatives by
    score, pair every positive against every one of them, and apply a smooth decreasing
    surrogate to the margin:

        L = mean_{i in pos, j in tail} softplus( -(lambda_i - lambda_j) / tau )

    *Derivation.* The inner set `{top k negatives}` is the empirical `(1-alpha)`-quantile
    level set of `N`, so it converges to `{lambda >= t_alpha}`; replacing the 0-1 indicator
    `1[lambda_i > lambda_j]` in the population expression by the convex surrogate
    `softplus(-u/tau)` (an upper bound on `1[u < 0]` up to the scale `log 2` at `u = 0`)
    gives a consistent, differentiable estimator of the negated tail-conditional
    probability. The objective is **not decomposable** into per-example terms -- the top-k
    selection makes each example's contribution depend on the rest of the batch -- which is
    precisely why it cannot be written as a weighted sum of existing losses.

    *Prior art.* Narasimhan & Agarwal (2013), `SVM^pAUC_tight`, the structural-SVM
    formulation of this same tail-conditional risk; Rudin (2009)'s p-norm push, the nearest
    ancestor, which emphasises the top of the ranking by raising the loss to a power rather
    than truncating at a budget; Yang et al. (ICML 2021) for the stochastic minibatch
    version and its DRO reading; Scott & Nowak (2005) and Tong, Feng & Li (2018) for the
    Neyman-Pearson framing (maximize TPR subject to FPR <= alpha), of which the pAUC
    objective is the budget-averaged relaxation. **What none of the repository's existing
    losses do:** LDAM-DRW reshapes per-class margins and ASL reshapes per-example focusing;
    neither has any notion of a referral budget, and neither touches the ordering between
    escalating lesions and the top-scoring benign ones.

    *Ablation plan.* (a) `alpha` in {0.05, 0.10, 0.20} -- does matching the training budget
    to the evaluation budget matter, or is any tail emphasis enough; (b) `tau` in
    {0.5, 1.0, 2.0} -- surrogate sharpness against gradient vanishing; (c) full-AUC control
    (`alpha = 1.0`), which reduces this to ordinary pairwise ranking and isolates the
    budget restriction as the active ingredient; (d) the CE-augmented variant
    `L_rank + beta * L_CE`, labelled an ablation, to test whether the fine-tune-from-frozen
    design is actually necessary to protect Macro-F1; (e) the same objective with the
    band-balanced sampler off, to separate the sampler's effect from the objective's.

    *Failure cases.* (i) **Minibatch quantile bias** -- the empirical `(1-alpha)`-quantile of
    a small negative sample is a biased estimate of the population quantile, so the risk is
    estimated on the wrong tail at small batch sizes; mitigated by large batches, and the
    bias shrinks as `alpha * n_neg` grows. (ii) **No within-group supervision** -- nothing in
    this objective distinguishes `mel` from `bcc` or `nv` from `bkl`, so 7-class Macro-F1
    can degrade; this is why the arm fine-tunes from the frozen checkpoint and why Macro-F1
    is monitored as a guard, and criterion 3 of the go-criterion cuts both ways here.
    (iii) **Surrogate saturation** -- `tau` too small makes `softplus` flat wherever the
    margin is already large and the gradient vanishes. (iv) **Degenerate collapse** -- pushing
    all escalating logits to `+inf` maximises the objective while destroying the class
    structure; detectable as Macro-F1 collapse, which is exactly why it is recorded every
    epoch rather than only at the end.
    """

    def __init__(self, esc_idx: list[int], alpha: float = 0.20, tau: float = 1.0):
        super().__init__()
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.esc_idx = list(esc_idx)
        self.alpha = float(alpha)
        self.tau = float(tau)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, bands: torch.Tensor | None = None) -> torch.Tensor:
        logits = logits.float()
        lam = induced_escalation_logit(logits, self.esc_idx)
        return tail_ranking_risk(lam, escalating_mask(labels, self.esc_idx), self.alpha, self.tau)


# ------------------------------------------------------------------------------- N4
class WorstBandRankingLoss(nn.Module):
    """**N4 -- worst-band distributionally robust escalation ranking (Group DRO).**

    *Population objective.* With age bands `G = {<40, 40-59, 60+}` and `R_g(theta)` the N3
    tail-conditional ranking risk computed **within** band `g`,

        min_theta max_{g in G} R_g(theta)
        ==  min_theta max_{q in simplex(G)} sum_g q_g R_g(theta)

    *Why this is a different arm and not a knob on N3.* It deliberately uses N3's risk
    unchanged and varies only the aggregation, so the pair forms a controlled contrast:
    pooled expectation versus worst-band minimax, holding the risk fixed. The motivation is
    structural -- the under-40 band contributes 64 of the 1,354 escalating training images
    (`results/age_band_prior.csv`: 4.9% escalating under 40 against 35.5% at 60+), so under a
    pooled risk its ranking errors are close to invisible and the optimizer is nearly
    indifferent to them. Minimax aggregation makes the pooled objective's blind spot the
    binding constraint. This is the exact quantity S34's efficiency analysis measured, where
    under-40 was the least efficient band in 16 of 20 cohort x budget cells.

    *Empirical form.* Online Group DRO (Sagawa et al., Algorithm 1): per batch, compute the
    per-band risks `R_g`, take a no-grad multiplicative (exponentiated-gradient) step on the
    adversary `q_g <- q_g * exp(eta_q * R_g)`, renormalize over the bands actually present in
    the batch, then take the model gradient step on `sum_g q_g R_g`.

    *Derivation.* `max_{q in simplex} sum_g q_g R_g = max_g R_g` because a linear functional
    on the simplex attains its maximum at a vertex, so the minimax program is exactly
    worst-band minimization. The multiplicative update is mirror ascent on `q` with the
    negative-entropy potential, giving the standard `O(1/sqrt(T))` saddle-point rate; the
    stochastic version with per-batch group risks is Sagawa et al.'s online algorithm, whose
    practical finding -- that worst-group generalization requires strong regularization or
    early stopping, not merely the objective -- is the reason this arm inherits the
    repository's weight decay and is early-stopped on a validation metric.

    *Prior art.* Sagawa et al. (ICLR 2020) Group DRO; Hashimoto et al. (ICML 2018) repeated
    loss minimization; Duchi & Namkoong (2021) for the general DRO treatment;
    Hebert-Johnson et al. (2018) multicalibration, already cited in this project's
    manuscript for the calibration thread -- N4 is where the calibration and fairness
    threads the manuscript currently keeps apart become one objective. Distinct from the
    project's own frozen lambda age-rule (S5), which is a *test-time* additive bonus fitted
    per band and needs age at inference; N4 changes what is optimized at train time and
    needs age only during training.

    *Ablation plan.* (a) **fixed-`q` control** -- the same risk with `q` frozen uniform (which
    reduces to band-averaged N3) and with `q` frozen at the inverse band prior; this is the
    decisive ablation, because it isolates whether the minimax matters or whether any
    up-weighting of under-40 would do. (b) `eta_q` in {0.001, 0.01, 0.1} -- adversary
    responsiveness against instability. (c) bands as 3 groups versus band x escalation as 6
    groups. (d) sampler on/off, shared with N3.

    *Failure cases.* (i) **The worst band is the smallest band** -- under-40 has 64 escalating
    training images, so the adversary can chase label noise; a single mislabeled lesion can
    dominate `q`. (ii) **Oversampling 64 images memorizes them** -- the band-balanced sampler
    N4 needs in order to see under-40 positives at all draws those 64 images roughly nine
    times per epoch; the train/val gap on the under-40 metric is the diagnostic, and it is
    recorded every epoch. (iii) **Worst-group overfitting** is Sagawa et al.'s own headline
    caveat and is why early stopping is not optional here. (iv) **Missing age** -- 38 of 6,981
    training rows have no age; they are dropped from this objective rather than pooled into
    a fourth "unknown" band, because letting the adversary chase a data-completeness
    artifact would be perverse (the same reasoning S8b applied to the Fitzpatrick
    `unknown` group, where the raw pooled gap was a missing-data artifact and not a
    skin-tone finding).
    """

    def __init__(
        self, esc_idx: list[int], n_bands: int = 3, alpha: float = 0.20, tau: float = 1.0,
        eta_q: float = 0.01, fixed_q: torch.Tensor | None = None,
    ):
        super().__init__()
        self.esc_idx = list(esc_idx)
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.eta_q = float(eta_q)
        self.n_bands = int(n_bands)
        self.fixed_q = None if fixed_q is None else (fixed_q / fixed_q.sum())
        self.register_buffer("q", torch.full((n_bands,), 1.0 / n_bands))

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, bands: torch.Tensor) -> torch.Tensor:
        if bands is None:
            raise ValueError("N4 requires per-sample band indices; none were passed")
        logits = logits.float()
        lam = induced_escalation_logit(logits, self.esc_idx)
        y_esc = escalating_mask(labels, self.esc_idx)

        risks, present = [], []
        for g in range(self.n_bands):
            rows = bands == g
            if rows.sum() == 0:
                continue
            risk = tail_ranking_risk(lam[rows], y_esc[rows], self.alpha, self.tau)
            risks.append(risk)
            present.append(g)
        if not risks:
            return _zero_like(logits)

        stacked = torch.stack(risks)
        idx = torch.tensor(present, device=logits.device)

        if self.fixed_q is not None:
            weights = self.fixed_q.to(logits.device)[idx]
        else:
            with torch.no_grad():  # adversary step: q is not differentiated through
                self.q[idx] = self.q[idx] * torch.exp(self.eta_q * stacked.detach())
                self.q.clamp_(min=1e-8)
                self.q /= self.q.sum()
            weights = self.q[idx]
        weights = weights / weights.sum()
        return (weights * stacked).sum()


# ------------------------------------------------------------------------------- N5
class BandConditionalLogitAdjustment(nn.Module):
    """**N5 -- band-conditional logit adjustment on the escalation partition.**

    *The diagnosis this encodes.* Within a band `g`, the Neyman-Pearson optimal ranker is the
    likelihood ratio, and by Bayes

        p(x | E, g) / p(x | B, g)  =  [p(E | x, g) / p(B | x, g)] * [pi_g(B) / pi_g(E)]

    so the posterior log-odds differ from the likelihood-ratio ranking by `log(pi_g(E)/pi_g(B))`
    -- a constant *within* a band, and therefore harmless if one ranked each band separately.
    The deployed system does not: it fits one posterior to the **pooled** prior and applies one
    global rule. With `pi(E)` at 4.9% under 40 and 35.5% at 60+ (`results/age_band_prior.csv`),
    a score trained to match the pooled posterior systematically under-scores escalating
    lesions in the low-prevalence band to the exact extent the image carries age signal. That
    is a mechanism for the measured failure, and it is testable.

    *Population objective.* Train with the log-prior added *inside* the softmax:

        min_theta  E[ -log softmax( z_theta(x) + t * log pi_{g(x)} )_y ]

    *Derivation.* The population minimizer satisfies
    `softmax(z + log pi_g)_c = p(c | x, g)`, hence
    `z_c = log p(c | x, g) - log pi_{g,c} + const = log p(x | c, g) - log p(x | g) + const`,
    so differences of the *unadjusted* logits recover the prior-free class-conditional
    likelihood ratio: `z_c - z_c' = log[ p(x | c, g) / p(x | c', g) ]`. The induced escalation
    logit read off those unadjusted logits is then

        lambda(x) = log[ sum_{c in E} p(x | c, g) ] - log[ sum_{c not in E} p(x | c, g) ]

    the **equal-weight** class-mixture likelihood ratio. Stated precisely, because the
    distinction matters: this is the NP-optimal ranker for a balanced within-band class
    mixture, not for the band's own mixture `p(x | E, g)` (which would weight the three
    escalating classes by their within-band relative frequencies). The equal-weight version
    is the intended target -- it is the one that does not inherit the prior being removed.

    *The deployment property.* The adjustment is applied at training time and removed at
    inference: the served score is the ordinary induced escalation logit. **N5 therefore
    yields an age-corrected score that does not need age at inference**, unlike this
    project's frozen lambda age-rule (S5), which requires the patient's age to select a
    band-specific bonus. Same diagnosis, opposite end of the pipeline.

    *Prior art.* Menon et al. (ICLR 2021) logit adjustment, including the explicit argument
    that logit adjustment and loss reweighting are different and that the latter is
    ineffective for overparameterized networks; Byrd & Lipton (2019) for the empirical
    version of that claim; Ren et al. (2020) balanced meta-softmax; Cao et al. (2019) LDAM,
    where the intervention is an additive *margin* rather than an additive log-prior -- the
    sharpest available contrast, and one this repository can run directly, since LDAM-DRW is
    already implemented and already trained. **Relevance to the incumbent:** the deployed
    models train with `effective_number` class weighting, i.e. precisely the loss reweighting
    Menon et al. and Byrd & Lipton argue against, so N5 is the principled alternative to
    what the incumbent already does rather than a new idea bolted on beside it.

    *Ablation plan.* (a) `t` in {0.5, 1.0, 1.5} -- theory says 1.0, practice often prefers
    less. (b) **pooled logit adjustment** (one global class prior, no bands) -- the decisive
    ablation separating "prior removal helps" from "*band-conditional* prior removal helps",
    which is the actual claim. (c) adjustment on the 7 classes versus on the 2-way escalation
    partition only. (d) against LDAM-DRW at equal budget: additive margin versus additive
    log-prior on the same architecture and schedule.

    *Failure cases.* (i) **Band-dependent constant** -- `const_g` above depends on `g`, so
    within-band ranking is corrected while *cross-band comparability* of the score is not
    guaranteed; the prediction is therefore that N5 improves band-conditional frontiers more
    reliably than the pooled frontier, and the evaluation must report both. This is stated in
    advance because it is the result most likely to be mistaken for failure. (ii) **Noisy
    band priors** -- 64 escalating under-40 images make `pi_{<40, mel}` a small-count estimate;
    Laplace smoothing is applied and the smoothed priors are recorded in the checkpoint.
    (iii) **Age carries real signal** -- melanoma genuinely is rarer in younger patients, so
    removing the prior discards epidemiologically valid information and may cost aggregate
    sensitivity to buy within-band fairness. That is a value judgment, not an optimization
    result, and it should be reported as one. (iv) **Fine-tuning only partially undoes a prior
    already baked into the initialization**, so a short fine-tune understates what this
    objective would do trained from scratch.
    """

    def __init__(self, log_prior_by_band: torch.Tensor, t: float = 1.0, weight: torch.Tensor | None = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        if log_prior_by_band.ndim != 2:
            raise ValueError(f"log_prior_by_band must be (n_bands, n_classes), got {tuple(log_prior_by_band.shape)}")
        self.register_buffer("log_prior", log_prior_by_band.float())
        self.t = float(t)
        self.weight = weight
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, bands: torch.Tensor) -> torch.Tensor:
        if bands is None:
            raise ValueError("N5 requires per-sample band indices; none were passed")
        adjusted = logits.float() + self.t * self.log_prior[bands]
        return F.cross_entropy(adjusted, labels, weight=self.weight, label_smoothing=self.label_smoothing)


def band_conditional_log_priors(
    labels: np.ndarray, bands: np.ndarray, n_bands: int, n_classes: int, smoothing: float = 1.0,
) -> torch.Tensor:
    """Laplace-smoothed `log pi_{g,c}` from the training split only.

    Smoothing is not cosmetic here: several band x class cells are small-count (under-40 has
    64 escalating images spread over three escalating classes), and an unsmoothed zero cell
    would send a logit to `-inf`.
    """
    counts = np.full((n_bands, n_classes), smoothing, dtype=np.float64)
    for g in range(n_bands):
        rows = bands == g
        for c in range(n_classes):
            counts[g, c] += int(((labels == c) & rows).sum())
    priors = counts / counts.sum(axis=1, keepdims=True)
    return torch.tensor(np.log(priors), dtype=torch.float32)


LOSS_REGISTRY = {
    "n3_pauc": BudgetConstrainedRankingLoss,
    "n4_groupdro": WorstBandRankingLoss,
    "n5_logitadj": BandConditionalLogitAdjustment,
}
