"""S45 / Phase D arm **D2** -- age-invariant representation via gradient reversal.

Provides the mechanism primitives that `research/v3/train_age_invariant.py` imports:
`AgeAdversarialHead`, `AgeInvariantWrapper`, `dann_lambda_schedule`.

## Why this arm, and why it has to touch the backbone

Three measured facts compose into the only intervention this project has not tried.

1. **The escalation score rides on age at fixed true class.** `age_residual` = **+0.1226**
   [+0.0924, +0.1513], certified above zero (S42, `results/v3/probe_battery.csv`), and the
   representation decodes age band at AUC **0.6922** [0.6638, 0.7187].
2. **No logit-level fix can repair that.** S36's `research/v2/verify_losses.py` **check 7**
   proved a band-constant offset of the escalation score cannot change any within-band ranking.
   Logit adjustment, the frozen per-band lambda rule, class priors and per-band thresholds are
   *all* such offsets.
3. **There is no head-recoverable headroom left.** `Delta_head` is negative in every band and
   NOT CERTIFIED anywhere (S42 §1), over a probe family including the deployed head's own
   functional form, an MLP and a GBM.

Together the defect is **in the representation, upstream of the head**, and every lever pulled so
far acts downstream of it.

## Why D2 rather than D1

Both fired at the S43 checkpoint. S43's addendum wrote the tie-breaker **before** Phase C
produced any result: *"if pooling archives moves nothing (outcome 4), a dual-view fix to the same
entanglement is less likely to help than the raw AUC suggests."* Phase C pooled three archives
and moved nothing -- `all_three` vs control **-0.0088** [-0.0554, +0.0354] on HAM val, and **0 of
2** zero-shot cross-archive transfer gains with a CI excluding zero
(`results/v3/external_by_cohort.json`). D1's premise was tested and failed; D2's was never
touched by Phase C, which varied archive composition, not age.

## The mechanism, and the failure mode that shapes it

An age-band head reads the pooled 768-d ConvNeXt feature through a **gradient reversal layer**
(Ganin & Lempitsky 2015). The head minimises age-band cross-entropy; the gradient is negated on
the way back, so the encoder maximises it.

    loss = CE(class_logits, y) + age_weight * CE(age_logits, band)      [GRL on the age path]

**The textbook single-step joint update does not work, and `--selftest` proves it.** Check 5
sweeps `max_lambda` from 0 to 25 under a 1:1 simultaneous update on a synthetic problem whose
answer is known by construction: post-hoc age decodability never falls below ~0.93 against a
chance of 0.333 at *any* strength, while class accuracy collapses once lambda passes ~2. A
simultaneous min-max step chases the discriminator's most recent direction instead of finding an
equilibrium.

Check 6 shows the standard fix working on the same data: train the age head to near-convergence
against **frozen, detached** pooled features before each combined encoder step
(`k_inner` discriminator steps per generator step). `train_age_invariant.py` implements exactly
that alternating pattern for the real CNN.

This is why the trainer is structured the way it is, and why `k_inner` is not an optional
tuning knob -- at `k_inner = 0` the method provably does nothing to the representation.

Rows whose age is missing carry band `"unknown"` and are masked out of the age loss via
`ignore_index` -- never imputed, matching `research.external.frozen_params.age_bands`.

## Pre-registration

    primary    HAM-val Macro-F1 vs the Phase C condition the run warm-starts from, lesion-grouped
               paired bootstrap. MCID 0.03, CI must exclude zero.
    mechanism  age_band probe AUC on the D2 representation must fall below the S42 interval
               lower bound of 0.6638. If it does not, the adversary did not move the
               representation and the primary endpoint is uninterpretable -- report that, do not
               tune until it moves.
    secondary  under-40 escalation sensitivity. REPORTED, NEVER DECISIVE: S44 measured a
               same-data retraining spread of 0.2273 (5 of 22 cases) on this endpoint, larger
               than the entire between-condition spread of 0.1818
               (`results/v3/s44_control_and_multiplicity.json`). At 22 positives it cannot
               referee anything.

Two-sided. A negative result is kept and written up.

    $py -m research.v3.age_invariant --selftest
    $py -m research.v3.train_age_invariant --help      # the real trainer
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from research.external import frozen_params as fp

BANDS = ("<40", "40-59", "60+")
IGNORE_INDEX = -100

# S42 measurements this arm is aimed at
S42_AGE_BAND_AUC = 0.6922
S42_AGE_BAND_AUC_CI_LO = 0.6638
S42_AGE_RESIDUAL = 0.1226


# ------------------------------------------------------------------ primitives
class _GradientReversal(torch.autograd.Function):
    """Identity forward, negated-and-scaled gradient backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float) -> torch.Tensor:
    return _GradientReversal.apply(x, lambda_)


def dann_lambda_schedule(progress: float, gamma: float = 10.0,
                         max_lambda: float = 1.0) -> float:
    """Ganin & Lempitsky's ramp: 0 at progress 0, approaching `max_lambda` at progress 1.

    Starting at zero matters -- adversarial pressure applied before the encoder has a class
    signal makes it discard information it has not yet learned to use.
    """
    p = float(min(1.0, max(0.0, progress)))
    return float(max_lambda) * (2.0 / (1.0 + float(np.exp(-gamma * p))) - 1.0)


class AgeAdversarialHead(nn.Module):
    """Age-band classifier behind a gradient reversal layer.

    `self.net` is the bare MLP with **no** GRL, so a trainer can run discriminator-only inner
    steps against frozen features (`head.net(feat)`) and adversarial outer steps through the
    wrapper (`head(feat)`) using the same parameters.
    """

    def __init__(self, in_features: int, num_age_bands: int = len(BANDS),
                 hidden: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden, num_age_bands),
        )
        self.lambda_ = 0.0

    def set_lambda(self, lambda_: float) -> None:
        self.lambda_ = float(lambda_)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(grad_reverse(features, self.lambda_))


class AgeInvariantWrapper(nn.Module):
    """Backbone + adversarial age head, sharing the backbone's own pooled feature.

    The class path is the **deployed forward, unmodified**: the pooled feature is captured by a
    forward hook on the `Flatten` that `ml.training.common.build_model` inserts before the final
    `Linear`, so no surgery is done on the backbone and `backbone_state_dict()` stays
    byte-compatible with every other checkpoint loader in this repository.
    """

    SUPPORTED_ARCHS = ("convnext_tiny", "convnext_small")

    def __init__(self, backbone: nn.Module, arch: str, num_age_bands: int = len(BANDS),
                 adv_hidden: int = 256, adv_dropout: float = 0.3) -> None:
        super().__init__()
        if arch not in self.SUPPORTED_ARCHS:
            raise ValueError(f"arch must be one of {self.SUPPORTED_ARCHS}, got {arch!r}")
        self.backbone = backbone
        self.arch = arch
        self._pooled: torch.Tensor | None = None

        # classifier = [LayerNorm2d, Flatten, Dropout, Linear]; hook the Flatten
        flatten, final = backbone.classifier[1], backbone.classifier[3]
        flatten.register_forward_hook(self._capture)
        self.age_head = AgeAdversarialHead(final.in_features, num_age_bands,
                                           hidden=adv_hidden, dropout=adv_dropout)

    def _capture(self, _module, _inputs, output: torch.Tensor) -> None:
        self._pooled = output

    def set_lambda(self, lambda_: float) -> None:
        self.age_head.set_lambda(lambda_)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        class_logits = self.backbone(images)
        return class_logits, self.age_head(self._pooled)

    def backbone_state_dict(self) -> dict:
        """Plain backbone weights; the age head is training scaffolding and is discarded."""
        return self.backbone.state_dict()


def band_indices(ages: np.ndarray) -> list[int]:
    """Band index per row; missing age -> `IGNORE_INDEX`, never imputed."""
    lookup = {b: i for i, b in enumerate(BANDS)}
    return [lookup.get(b, IGNORE_INDEX) for b in fp.age_bands(np.asarray(ages, dtype=float))]


# ------------------------------------------------------------------ self-test
def _synthetic(n: int, rng: np.random.Generator, shift: bool, dim: int = 24):
    """Class and age directions orthogonal by construction; correlated in train, broken in test.

    In train, class 1 is 90% band 2 -- an age shortcut that predicts class well. Under `shift`
    the two are independent, so a model leaning on the age direction must degrade.
    """
    c = rng.integers(0, 2, size=n)
    a = (rng.integers(0, 3, size=n) if shift
         else np.where(rng.random(n) < 0.9, c * 2, rng.integers(0, 3, size=n)))
    x = rng.normal(0, 0.35, size=(n, dim))
    x[:, 0] += 1.4 * (2 * c - 1)          # class direction
    x[:, 1] += 1.4 * (a - 1)              # age direction, orthogonal to it
    return (torch.tensor(x, dtype=torch.float32),
            torch.tensor(c, dtype=torch.long), torch.tensor(a, dtype=torch.long))


def _fit_toy(xtr, ytr, atr, max_lambda: float, seed: int, k_inner: int = 0, epochs: int = 150):
    """Encoder + class head + age adversary.

    `k_inner = 0` is the textbook single-step joint update. `k_inner > 0` trains the age head to
    near-convergence against frozen, detached features first -- the alternating pattern
    `train_age_invariant.py` uses.
    """
    torch.manual_seed(seed)
    enc = nn.Sequential(nn.Linear(xtr.shape[1], 32), nn.ReLU(), nn.Linear(32, 16))
    head = nn.Linear(16, 2)
    adv = AgeAdversarialHead(16, hidden=32, dropout=0.0)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters())
                           + list(adv.parameters()), lr=0.01)
    opt_adv = torch.optim.Adam(adv.parameters(), lr=0.01)

    for e in range(epochs):
        adv.set_lambda(dann_lambda_schedule(e / max(1, epochs - 1), max_lambda=max_lambda))
        if k_inner:
            with torch.no_grad():
                frozen = enc(xtr).detach()
            for _ in range(k_inner):
                opt_adv.zero_grad()
                F.cross_entropy(adv.net(frozen), atr).backward()
                opt_adv.step()
        opt.zero_grad()
        z = enc(xtr)
        (F.cross_entropy(head(z), ytr) + F.cross_entropy(adv(z), atr)).backward()
        opt.step()
    return enc, head


def _age_decodability(enc, x, a, seed: int) -> float:
    """Post-hoc: how well a FRESH probe reads age band off the frozen representation."""
    torch.manual_seed(seed)
    with torch.no_grad():
        z = enc(x)
    probe = nn.Linear(z.shape[1], 3)
    opt = torch.optim.Adam(probe.parameters(), lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        F.cross_entropy(probe(z), a).backward()
        opt.step()
    with torch.no_grad():
        return float((probe(z).argmax(1) == a).float().mean())


def selftest() -> int:
    print("age_invariant.py self-test -- mechanism proof on synthetic data\n")
    ok = True

    # 1. gradient reversal negates and scales, exactly
    x = torch.tensor([2.0], requires_grad=True)
    grad_reverse(x, 3.0).backward(torch.tensor([1.0]))
    good = torch.allclose(x.grad, torch.tensor([-3.0]))
    print(f"  1. GRL backward is -lambda*grad (got {x.grad.item():+.1f}, expect -3.0) "
          f"-> {'PASS' if good else 'FAIL'}")
    ok &= good

    # 2. GRL is the identity forward
    y = torch.randn(5, 4)
    good = torch.equal(grad_reverse(y, 7.0), y)
    print(f"  2. GRL forward is the identity -> {'PASS' if good else 'FAIL'}")
    ok &= good

    # 3. DANN schedule: 0 at start, monotone, bounded
    sched = [dann_lambda_schedule(p / 29, max_lambda=0.5) for p in range(30)]
    good = (abs(sched[0]) < 1e-12 and all(b >= a - 1e-12 for a, b in zip(sched, sched[1:]))
            and max(sched) <= 0.5 + 1e-9)
    print(f"  3. lambda ramp 0 -> {max(sched):.3f}, monotone, bounded "
          f"-> {'PASS' if good else 'FAIL'}")
    ok &= good

    rng = np.random.default_rng(0)
    xtr, ytr, atr = _synthetic(1500, rng, shift=False)
    xid, yid, aid = _synthetic(600, rng, shift=False)
    xsh, ysh, _ = _synthetic(600, rng, shift=True)

    def acc(enc, head, x, y):
        with torch.no_grad():
            return float((head(enc(x)).argmax(1) == y).float().mean())

    base_enc, base_head = _fit_toy(xtr, ytr, atr, max_lambda=0.0, seed=1)
    base_id, base_sh = acc(base_enc, base_head, xid, yid), acc(base_enc, base_head, xsh, ysh)
    base_dec = _age_decodability(base_enc, xid, aid, seed=2)

    # 4. the shortcut is really there -- without this the rest is vacuous
    good = base_sh < base_id - 0.02
    print(f"  4. baseline exploits the age shortcut (in-dist {base_id:.3f} -> "
          f"shifted {base_sh:.3f}), age decodable {base_dec:.3f} "
          f"-> {'PASS' if good else 'FAIL'}")
    ok &= good

    # 5. single-step joint GRL does NOT remove age, at ANY strength. Asserted, not hoped for:
    #    this is what forces the alternating trainer, and it must not silently change.
    sweep = {lam: _age_decodability(_fit_toy(xtr, ytr, atr, max_lambda=lam, seed=1)[0],
                                    xid, aid, seed=2)
             for lam in (0.5, 2.0, 10.0, 25.0)}
    good = min(sweep.values()) > 0.7      # chance is 0.333
    print(f"  5. k_inner=0 leaves age decodable at every lambda "
          f"({', '.join(f'{k:g}:{v:.3f}' for k, v in sweep.items())}, chance 0.333) "
          f"-> {'PASS' if good else 'FAIL'}")
    ok &= good

    # 6. the alternating fix DOES move the representation, where k_inner=0 never does
    strong_enc, strong_head = _fit_toy(xtr, ytr, atr, max_lambda=10.0, seed=1, k_inner=20)
    strong_dec = _age_decodability(strong_enc, xid, aid, seed=2)
    good = strong_dec < 0.70
    print(f"  6. k_inner=20 at lambda=10 DOES drop age decodability "
          f"({base_dec:.3f} -> {strong_dec:.3f}) -> {'PASS' if good else 'FAIL'}")
    ok &= good

    # 7. ...but only by destroying the class signal, and the cheap setting does not move age.
    #    Both halves asserted: in this toy there is NO operating point that buys invariance
    #    cheaply. This is the finding the handoff rests on, so it must not silently change.
    strong_id = acc(strong_enc, strong_head, xid, yid)
    cheap_enc, cheap_head = _fit_toy(xtr, ytr, atr, max_lambda=1.0, seed=1, k_inner=20)
    cheap_dec = _age_decodability(cheap_enc, xid, aid, seed=2)
    cheap_id = acc(cheap_enc, cheap_head, xid, yid)
    good = strong_id < 0.70 and cheap_id > 0.90 and cheap_dec > 0.85
    print(f"  7. the trade is UNFAVOURABLE: lambda=10 costs accuracy "
          f"({base_id:.3f} -> {strong_id:.3f}); lambda=1 keeps it ({cheap_id:.3f}) but leaves "
          f"age decodable ({cheap_dec:.3f}) -> {'PASS' if good else 'FAIL'}")
    ok &= good

    # 8. the wrapper's class path is byte-identical to the plain backbone
    from ml.training.common import build_model
    torch.manual_seed(0)
    backbone = build_model("convnext_tiny", 7, pretrained=False)
    model = AgeInvariantWrapper(backbone, "convnext_tiny", adv_hidden=32).eval()
    img = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        class_logits, age_logits = model(img)
        direct = model.backbone(img)
    good = (torch.allclose(class_logits, direct, atol=1e-5)
            and model._pooled.shape == (2, 768) and age_logits.shape == (2, 3))
    print(f"  8. wrapper forward == backbone forward, pooled {tuple(model._pooled.shape)} "
          f"-> {'PASS' if good else 'FAIL'}")
    ok &= good

    # 9. the saved state dict is the plain backbone -- no age-head keys leak into checkpoints
    keys = set(model.backbone_state_dict())
    good = keys == set(backbone.state_dict()) and not any("age_head" in k for k in keys)
    print(f"  9. backbone_state_dict() is checkpoint-compatible ({len(keys)} keys, no age_head) "
          f"-> {'PASS' if good else 'FAIL'}")
    ok &= good

    # 10. missing age is masked, never imputed
    got = band_indices(np.array([30.0, 50.0, 70.0, np.nan]))
    good = got == [0, 1, 2, IGNORE_INDEX]
    print(f" 10. bands {got} -- unknown age masked to ignore_index "
          f"-> {'PASS' if good else 'FAIL'}")
    ok &= good

    print("\n" + ("ALL CHECKS PASS -- single-step GRL is inert; the alternating fix works"
                  if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


def sweep() -> int:
    """Persist the (k_inner x max_lambda) grid the handoff's tuning advice rests on.

    Hard Rule 4: the numbers quoted in `results/v3/S45_STATUS.md` come from this file, not from
    a transcript.
    """
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    rng = np.random.default_rng(0)
    xtr, ytr, atr = _synthetic(1500, rng, shift=False)
    xid, yid, aid = _synthetic(600, rng, shift=False)
    xsh, ysh, _ = _synthetic(600, rng, shift=True)

    def acc(enc, head, x, y):
        with torch.no_grad():
            return float((head(enc(x)).argmax(1) == y).float().mean())

    rows = []
    print(f"{'k_inner':>8} {'lambda':>7} {'in-dist':>8} {'shifted':>8} {'age-dec':>8}"
          f"  (chance 0.333)")
    for k in (0, 5, 20, 50):
        for lam in (1.0, 3.0, 10.0, 30.0):
            enc, head = _fit_toy(xtr, ytr, atr, max_lambda=lam, seed=1, k_inner=k)
            row = {"k_inner": k, "max_lambda": lam,
                   "acc_in_dist": acc(enc, head, xid, yid),
                   "acc_shifted": acc(enc, head, xsh, ysh),
                   "age_decodability": _age_decodability(enc, xid, aid, seed=2)}
            rows.append(row)
            print(f"{k:>8} {lam:>7.1f} {row['acc_in_dist']:>8.3f} "
                  f"{row['acc_shifted']:>8.3f} {row['age_decodability']:>8.3f}")

    out = Path(__file__).resolve().parents[2] / "results" / "v3" / "d2_mechanism_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "session": "S45", "arm": "D2_age_invariant",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "does the alternating fix buy invariance at an acceptable class-accuracy cost?",
        "chance_age_decodability": 1 / 3, "baseline_is_k_inner_0": True,
        "grid": rows,
        "finding": ("k_inner=0 leaves age decodable at every lambda; k_inner>=5 moves it only "
                    "at lambda where class accuracy collapses; no operating point in this toy "
                    "buys invariance cheaply"),
        "note": "synthetic data only; no split of any real cohort is read",
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="S45 Phase D arm D2 -- mechanism primitives")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--sweep", action="store_true",
                   help="write the k_inner x lambda grid to results/v3/d2_mechanism_sweep.json")
    args = p.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.sweep:
        return sweep()
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
