"""S58 -- the three-stage front end: admissibility gate, domain router, domain-matched head.

S42's transport experiment measured the largest unused gain in the repository: on PAD, a head
fitted on the target domain reached 0.887 against the HAM-fitted head's 0.640 (recovery +0.2463
[+0.2151, +0.2752]) -- "the head is mis-aimed, not the features blind". That was cross-modality.
S58 asks the deployable version of the question, within dermoscopy, on the one cohort V4 has not
selected on:

    1. gate    class-conditional Mahalanobis (research.selective.mahalanobis, unchanged) fitted on
               in-scope training features; score above the in-scope 95th percentile ->
               OUT_OF_SCOPE, no prediction issued. Scored on reserved (in scope) and PAD
               (smartphone clinical, out of scope).
    2. router  multinomial logistic over {ham, bcn20000, mskcc} on the same features. S43/S49
               measured archive AUC 0.990 *within* dermoscopy, so the regime is identifiable.
    3. head    route each image to a head fitted for its archive.

Everything sits on ONE frozen trunk -- `convnext_tiny_best.HAM-only.pt`, the extractor behind every
feature cache in the repo, 224 px, 1 view.

**TRANSFORM note.** Features use the *deployed* eval transform
(`ml.preprocessing.transforms.build_eval_transform`, bilinear resize), NOT S51's
`extract_backbone_features._eval_transform` (bicubic). The first S58 extraction used S51's and the
control failed to reproduce the checkpoint: 95.5% argmax agreement on HAM val, features 22% apart
(relative L2), Macro-F1 0.748 -> 0.727, with the layer itself exact to 1e-6 and the ISIC-2019 HAM
JPEGs byte-identical to HAM10000's. A head refit on bicubic features would recover that
interpolation penalty and report it as head gain, so every S58 cache -- reserved included -- is
re-extracted here, and `--select` refuses unless H0 reproduces the checkpoint on >= 99.9% of HAM val. No trunk weight moves; the heads are L2 logistic
regressions (CPU, seconds). The only GPU step is feature extraction, which the user runs.

Arms (all scored on every reserved row, so the gate is a common factor and not a confound):

    H0  ham_head        the checkpoint's own final linear layer -- the control
    H1  pooled_head     one head on all V4 train rows (HAM + BCN + MSKCC)
    H2  routed          router -> {ham: H0's layer, bcn20000: BCN head, mskcc: pooled head}
    H3  oracle_routed   H2's heads, true archive -- the router's cost is H3 - H2
    D1  routed_mskcc3   H2 with a 3-class MSKCC head -- descriptive only, NOT deployable

**Why MSKCC falls back to the pooled head.** Every MSKCC row in manifest_v4 (train, val and
reserved) is `bkl`, `mel` or `nv`. A head fitted on those rows can never say `bcc`, and on
reserved MSKCC that is free accuracy that comes from the archive's *curation*, not from aiming the
head. A real MSKCC clinic sees BCC. The declared rule: a domain gets its own head only when its
train rows hold every class with >= MIN_CLASS_IMAGES images; otherwise it routes to the pooled
head. D1 reports what the label-space restriction would have bought, so the number is visible
and cannot be quietly adopted.

**Why HAM keeps the checkpoint's layer.** S42's H2 falsified head refits on HAM (every matched
Delta_head negative), and HAM train features are *in-sample* for this trunk, so an LR head fitted
on them would be fitted to over-separated features. The pooled head carries the same in-sample
HAM rows -- declared, not hidden.

Endpoints (reserved, frozen in `results/v4/s58_plan.json` before the read):

    primary      Macro-F1 (7-class), H2 - H0, lesion-grouped paired bootstrap, MCID 0.020
    family       {primary, under-40 escalation pAUC@0.20 H2 - H0}, Holm over 2
    design rule  H2 is stage 3 only if H2 - H1 is non-inferior (CI lower > -0.005) with a
                 point >= 0; otherwise the pooled head is stage 3 and the router is kept as a
                 drift / provenance signal only

Stages, in order:

    $py -m research.v4.s58_front_end --selftest
    $py -m research.v4.s58_front_end --extract --smoke        # GPU, ~1 min, user runs
    $py -m research.v4.s58_front_end --extract                # GPU, user runs
    $py -m research.v4.s58_front_end --select                 # CPU: fit on train, select on val
    $py -m research.v4.s58_front_end --freeze-plan
    $py -m research.v4.s58_front_end --reserved               # read once; receipt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "ml" / "data" / "manifest_v4.csv"
PAD_MANIFEST = REPO_ROOT / "ml" / "data" / "manifest_pad.csv"
CHECKPOINT = REPO_ROOT / "ml" / "checkpoints" / "convnext_tiny_best.HAM-only.pt"
FEAT_DIR = REPO_ROOT / "research" / "v4" / "features"
FEATURES_FIT = FEAT_DIR / "convnext_tiny_s58_fit.npz"
FEATURES_PAD = FEAT_DIR / "convnext_tiny_s58_pad.npz"
FEATURES_RESERVED = FEAT_DIR / "convnext_tiny_s58_reserved.npz"  # NOT S51's: see TRANSFORM note
SMOKE_FEAT_DIR = FEAT_DIR / "smoke_s58"
OUT_DIR = REPO_ROOT / "results" / "v4" / "s58"
SMOKE_DIR = OUT_DIR / "smoke"
PLAN_PATH = REPO_ROOT / "results" / "v4" / "s58_plan.json"
SELECTION_PATH = OUT_DIR / "selection_val.json"
STATE_PATH = OUT_DIR / "fit_state.npz"
RECEIPT_PATH = OUT_DIR / "reserved_receipt.json"
LEDGER_PATH = REPO_ROOT / "research" / "experiments.csv"
SESSION = "v4_s58"

FIT_SPLITS = ("train", "val", "ham_val")      # ham_test and reserved are never extracted here
DOMAINS = ("ham", "bcn20000", "mskcc")
K = 7
C_GRID = (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
CONTROL_AGREEMENT_MIN = 0.999
MIN_CLASS_IMAGES = 20
GATE_QUANTILE = 0.95                            # nominal in-scope false-rejection 5%
PRIMARY_BAND = "<40"
FPR_MAX = 0.20
MCID = 0.020                                    # S52's Macro-F1 MCID (~7x the seed spread)
NONINFERIORITY = 0.005                          # S52's margin
STAGE_TARGETS = {"gate_reserved_false_reject_max": 0.10,
                 "gate_pad_reject_min": 0.80,
                 "router_reserved_accuracy_min": 0.95}
N_BOOT = 2000
BOOT_SEED = 42
SMOKE_ROWS = 256


# ============================================================================ small utilities
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def macro_f1(y: np.ndarray, pred: np.ndarray, k: int = K) -> float:
    from research.v4.s54_gate import macro_f1 as _f1

    return _f1(y, pred, k)


def escalation_mass(probs: np.ndarray) -> np.ndarray:
    from research.v4.s54_gate import escalation_mass as _mass

    return _mass(probs)


def pauc(y_esc: np.ndarray, score: np.ndarray) -> float:
    from research.v4.s54_gate import pauc as _pauc

    return _pauc(y_esc, score)


def manifest() -> pd.DataFrame:
    frame = pd.read_csv(MANIFEST, low_memory=False)
    return frame.assign(image_id=frame["image_id"].astype(str),
                        group_id=frame["group_id"].astype(str),
                        age_band=frame["age_band"].astype(str))


def extraction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame[frame["split"].isin(FIT_SPLITS)].sort_values("image_id").reset_index(drop=True)
    if rows["split"].isin(["ham_test", "reserved"]).any():
        raise AssertionError("extraction frame reaches a sealed split")
    return rows


def eligible_domains(frame: pd.DataFrame) -> dict[str, bool]:
    """A domain gets its own head only if its train rows hold every class >= MIN_CLASS_IMAGES."""
    train = frame[frame["split"] == "train"]
    out = {}
    for domain in DOMAINS:
        counts = np.bincount(train.loc[train["archive"] == domain, "class_index_7"].astype(int),
                             minlength=K)
        out[domain] = bool((counts >= MIN_CLASS_IMAGES).all())
    return out


# ============================================================================ heads
@dataclass
class LinearHead:
    """Standardise -> affine -> softmax, over `classes`; absent classes get probability 0."""

    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray            # (len(classes), D)
    intercept: np.ndarray       # (len(classes),)
    classes: np.ndarray         # indices into the 7-class output

    def logits(self, x: np.ndarray) -> np.ndarray:
        z = ((x - self.mean) / self.scale) @ self.coef.T + self.intercept
        return z

    def predict_proba(self, x: np.ndarray, width: int = K) -> np.ndarray:
        out = np.zeros((len(x), width))
        out[:, self.classes] = softmax(self.logits(x))
        return out

    def to_arrays(self, prefix: str) -> dict[str, np.ndarray]:
        return {f"{prefix}__{k}": v for k, v in self.__dict__.items()}

    @classmethod
    def from_arrays(cls, store, prefix: str) -> "LinearHead":
        return cls(**{k: store[f"{prefix}__{k}"] for k in
                      ("mean", "scale", "coef", "intercept", "classes")})


def fit_lr(x: np.ndarray, y: np.ndarray, c: float) -> LinearHead:
    from sklearn.linear_model import LogisticRegression

    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    model = LogisticRegression(C=c, class_weight="balanced", max_iter=5000, tol=1e-5)
    model.fit((x - mean) / scale, y)
    coef, intercept = model.coef_, model.intercept_
    if coef.shape[0] == 1:                       # sklearn collapses a binary problem
        coef = np.vstack([-coef[0] / 2, coef[0] / 2])
        intercept = np.array([-intercept[0] / 2, intercept[0] / 2])
    return LinearHead(mean, scale, coef, intercept, model.classes_.astype(int))


def checkpoint_head() -> LinearHead:
    """The checkpoint's own final nn.Linear (input = the cached penultimate features)."""
    import torch  # noqa: F401  (build_model_from_checkpoint needs it)

    from ml.training.common import build_model_from_checkpoint
    from research.selective.features import _final_linear

    model, _ = build_model_from_checkpoint(CHECKPOINT, "cpu")
    layer = _final_linear(model)
    w = layer.weight.detach().double().numpy()
    b = layer.bias.detach().double().numpy()
    d = w.shape[1]
    return LinearHead(np.zeros(d), np.ones(d), w, b, np.arange(K))


def route(router: LinearHead, x: np.ndarray) -> np.ndarray:
    """Predicted domain name per row."""
    return np.asarray(DOMAINS)[router.predict_proba(x, width=len(DOMAINS)).argmax(1)]


def assemble(domains: np.ndarray, x: np.ndarray, heads: dict[str, LinearHead]) -> np.ndarray:
    out = np.full((len(x), K), np.nan)
    for name in np.unique(domains):
        m = domains == name
        out[m] = heads[str(name)].predict_proba(x[m])
    if np.isnan(out).any():
        raise AssertionError("a row was not routed to any head")
    return out


# ============================================================================ features
def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    return z["features"].astype(np.float64), z["image_ids"].astype(str)


def align(features: np.ndarray, ids: np.ndarray, wanted: pd.Series) -> np.ndarray:
    order = pd.Index(ids).get_indexer(wanted.astype(str))
    if (order < 0).any():
        raise KeyError(f"{int((order < 0).sum())} rows have no cached features")
    return features[order]


def fit_panel(feature_path: Path = FEATURES_FIT) -> tuple[pd.DataFrame, np.ndarray]:
    feats, ids = _load_npz(feature_path)
    frame = manifest().set_index("image_id").loc[ids].reset_index()
    if not set(frame["split"]).issubset(FIT_SPLITS):
        raise AssertionError(f"fit features reach splits {sorted(set(frame['split']))}")
    return frame, feats


class PadDataset:
    """PAD-UFES-20 rows by manifest path. Module level: Windows spawn must pickle it."""

    def __init__(self, frame: pd.DataFrame, transform) -> None:
        self.paths = frame["path"].tolist()
        self.ids = frame["image_id"].astype(str).tolist()
        self.labels = frame["class_index"].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        from PIL import Image

        with Image.open(REPO_ROOT / self.paths[i]) as image:
            image = image.convert("RGB")
        return self.transform(image), self.labels[i], self.ids[i]


# ============================================================================ extraction (GPU)
def run_extract(smoke: bool, workers: int, batch_size: int) -> int:
    import torch
    from torch.utils.data import DataLoader

    from research import testguard
    from ml.preprocessing.transforms import build_eval_transform
    from research.v4 import extract_backbone_features as ebf
    from research.v4 import s54_guard as guard

    testguard.block_test_reads("S58 extraction: train/val/ham_val + PAD only")
    guard.preflight(commit_gb=guard.torch_commit_gb(workers))

    fit_rows = extraction_frame(manifest())
    pad_rows = pd.read_csv(PAD_MANIFEST)
    if smoke:
        fit_rows = (fit_rows.groupby(["archive", "split"]).head(SMOKE_ROWS // 4)
                    .sort_values("image_id").reset_index(drop=True))
        pad_rows = pad_rows.head(64)
    out_dir = SMOKE_FEAT_DIR if smoke else FEAT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, embed, meta = ebf._build("convnext_tiny", device)
    transform = build_eval_transform(224)          # the deployed transform, not S51's bicubic
    reserved_rows = manifest().query("split == 'reserved'").sort_values("image_id")
    if smoke:
        reserved_rows = reserved_rows.head(64)
    jobs = (("fit", ebf._ManifestDataset(fit_rows, transform)),
            ("pad", PadDataset(pad_rows, transform)),
            ("reserved", ebf._ManifestDataset(reserved_rows, transform)))
    for name, dataset in jobs:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                            pin_memory=device.type == "cuda",
                            persistent_workers=workers > 0)
        chunks, labels, ids = [], [], []
        started = time.time()
        with torch.no_grad():
            for step, (images, y, batch_ids) in enumerate(loader):
                images = images.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=device.type == "cuda"):
                    primary, _ = embed(model, images)
                chunks.append(primary.float().cpu().numpy())
                labels.append(np.asarray(y))
                ids.extend(batch_ids)
                if step % 25 == 0:
                    print(f"  [{name}] {len(ids):>6}/{len(dataset)}  "
                          f"{len(ids) / max(time.time() - started, 1e-9):6.1f} img/s", flush=True)
        path = out_dir / {"fit": FEATURES_FIT, "pad": FEATURES_PAD,
                          "reserved": FEATURES_RESERVED}[name].name
        feats = np.concatenate(chunks)
        np.savez_compressed(path, features=feats, labels=np.concatenate(labels),
                            image_ids=np.asarray([str(i) for i in ids]))
        info = meta | {"session": "S58", "part": name, "n": len(ids), "dim": int(feats.shape[1]),
                       "splits": {"fit": list(FIT_SPLITS), "pad": ["pad_ufes_20 (all)"],
                                  "reserved": ["reserved (features only, no labels read)"]}[name],
                       "transform": "ml.preprocessing.transforms.build_eval_transform(224) "
                                    "(bilinear 256 -> centre crop 224), the deployed transform",
                       "checkpoint_sha256": sha256(CHECKPOINT), "smoke": smoke,
                       "seconds": round(time.time() - started, 1)}
        path.with_suffix(".meta.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        print(f"  wrote {rel(path)}  {feats.shape}  in {(time.time() - started) / 60:.1f} min")
    return 0


# ============================================================================ selection (CPU)
def fit_components(frame: pd.DataFrame, feats: np.ndarray, verbose: bool = True,
                   strict: bool = True) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Fit every component on train, select each C on held-out in-scope rows (val + ham_val)."""
    from sklearn.metrics import log_loss

    from research.selective import mahalanobis

    train = (frame["split"] == "train").to_numpy()
    held = ~train
    y = frame["class_index_7"].to_numpy(int)
    arch = frame["archive"].to_numpy(str)
    unknown = sorted(set(arch) - set(DOMAINS))
    if unknown:
        raise ValueError(f"unknown archive(s) {unknown}")
    dom_idx = np.array([DOMAINS.index(a) for a in arch])     # DOMAINS order = router columns
    eligible = eligible_domains(frame)
    arrays: dict[str, np.ndarray] = {}
    selection: dict[str, Any] = {"eligible_domains": eligible, "c_grid": list(C_GRID)}

    def grid(label: str, xs: np.ndarray, ys: np.ndarray, xv: np.ndarray, yv: np.ndarray,
             score: Callable[[LinearHead], float], higher_is_better: bool) -> LinearHead:
        trail, best, best_head = {}, None, None
        for c in C_GRID:
            head = fit_lr(xs, ys, c)
            value = score(head)
            trail[str(c)] = value
            better = best is None or (value > best if higher_is_better else value < best)
            if better:
                best, best_head, best_c = value, head, c
        if strict and best_c in (C_GRID[0], C_GRID[-1]):
            raise SystemExit(f"{label}: selected C={best_c} sits on the grid boundary; widen C_GRID")
        selection[label] = {"selected_c": best_c, "value": best, "trail": trail,
                            "n_fit": int(len(xs)), "n_select": int(len(xv))}
        if verbose:
            print(f"  {label:<16} C={best_c:<5} value={best:.4f}  (fit {len(xs)}, select {len(xv)})")
        return best_head

    # gate: tied-covariance class-conditional Mahalanobis on in-scope train, threshold on held-out
    state = mahalanobis.fit(feats[train], y[train], num_classes=K)
    held_scores = mahalanobis.score(state, feats[held])
    threshold = float(np.quantile(held_scores, GATE_QUANTILE))
    arrays |= {"gate__means": state.means, "gate__precision": state.precision,
               "gate__threshold": np.array(threshold)}
    selection["gate"] = {"threshold": threshold, "quantile": GATE_QUANTILE,
                         "shrinkage": state.shrinkage, "n_fit": int(train.sum()),
                         "held_out_reject_rate": float((held_scores > threshold).mean())}

    # router: 3 archives, log loss on held-out
    router = grid("router", feats[train], dom_idx[train], feats[held], dom_idx[held],
                  lambda h: log_loss(dom_idx[held], h.predict_proba(feats[held], len(DOMAINS)),
                                     labels=list(range(len(DOMAINS)))),
                  higher_is_better=False)
    arrays |= router.to_arrays("router")

    def f1_on(mask: np.ndarray) -> Callable[[LinearHead], float]:
        return lambda h: macro_f1(y[mask], h.predict_proba(feats[mask]).argmax(1))

    pooled = grid("pooled", feats[train], y[train], feats[held], y[held], f1_on(held), True)
    arrays |= pooled.to_arrays("pooled")
    heads = {"ham": checkpoint_head(), "pooled": pooled}
    arrays |= heads["ham"].to_arrays("ham")

    for domain in ("bcn20000", "mskcc"):
        fit_m = train & (arch == domain)
        sel_m = held & (arch == domain)
        head = grid(f"head_{domain}", feats[fit_m], y[fit_m], feats[sel_m], y[sel_m],
                    f1_on(sel_m), True)
        selection[f"head_{domain}"]["classes"] = head.classes.tolist()
        arrays |= head.to_arrays(f"head_{domain}")
    return selection, arrays


def load_state(path: Path = STATE_PATH) -> dict[str, Any]:
    store = np.load(path, allow_pickle=False)
    heads = {"ham": LinearHead.from_arrays(store, "ham"),
             "pooled": LinearHead.from_arrays(store, "pooled"),
             "head_bcn20000": LinearHead.from_arrays(store, "head_bcn20000"),
             "head_mskcc": LinearHead.from_arrays(store, "head_mskcc")}
    return {"router": LinearHead.from_arrays(store, "router"), "heads": heads,
            "gate_means": store["gate__means"], "gate_precision": store["gate__precision"],
            "gate_threshold": float(store["gate__threshold"])}


def head_maps(state: dict[str, Any], eligible: dict[str, bool]) -> dict[str, dict[str, LinearHead]]:
    h = state["heads"]
    routed = {"ham": h["ham"],
              "bcn20000": h["head_bcn20000"] if eligible["bcn20000"] else h["pooled"],
              "mskcc": h["head_mskcc"] if eligible["mskcc"] else h["pooled"]}
    return {"routed": routed, "routed_mskcc3": routed | {"mskcc": h["head_mskcc"]}}


def arm_probs(state: dict[str, Any], x: np.ndarray, true_domain: np.ndarray,
              eligible: dict[str, bool]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    maps = head_maps(state, eligible)
    routed_to = route(state["router"], x)
    arms = {"H0_ham_head": state["heads"]["ham"].predict_proba(x),
            "H1_pooled_head": state["heads"]["pooled"].predict_proba(x),
            "H2_routed": assemble(routed_to, x, maps["routed"]),
            "H3_oracle_routed": assemble(true_domain, x, maps["routed"]),
            "D1_routed_mskcc3": assemble(routed_to, x, maps["routed_mskcc3"])}
    return arms, routed_to


def gate_scores(state: dict[str, Any], x: np.ndarray) -> np.ndarray:
    from research.selective import mahalanobis

    ms = mahalanobis.MahalanobisState(state["gate_means"], state["gate_precision"],
                                      tuple(range(K)), 0.0, 0)
    return mahalanobis.score(ms, x)


def run_select() -> int:
    from research import testguard

    testguard.block_test_reads("S58 selection: fit on V4 train, select on val + ham_val")
    frame, feats = fit_panel()
    print(f"fit panel: {len(frame)} rows  " +
          "  ".join(f"{s}={int((frame['split'] == s).sum())}" for s in FIT_SPLITS))
    selection, arrays = fit_components(frame, feats)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(STATE_PATH, **arrays)
    state = load_state()
    held = (frame["split"] != "train").to_numpy()
    arms, routed_to = arm_probs(state, feats[held], frame.loc[held, "archive"].to_numpy(str),
                                selection["eligible_domains"])
    y = frame.loc[held, "class_index_7"].to_numpy(int)
    selection["val_macro_f1"] = {k: macro_f1(y, p.argmax(1)) for k, p in arms.items()}
    selection["val_router_accuracy"] = float((routed_to == frame.loc[held, "archive"]).mean())

    # the control must be the checkpoint: compare with its published 1-view HAM val predictions
    ref = pd.read_csv(REPO_ROOT / "research" / "predictions" / "convnext_tiny_val.csv")
    ham = frame[held & (frame["split"] == "ham_val").to_numpy()]
    ref = ref.set_index(ref["image_id"].astype(str)).loc[ham["image_id"]]
    pred_cols = [c for c in ref.columns if c.startswith("p_")]
    h0 = state["heads"]["ham"].predict_proba(feats[ham.index.to_numpy()])
    agreement = float((h0.argmax(1) == ref[pred_cols].to_numpy().argmax(1)).mean())
    selection["control_check"] = {
        "reference": "research/predictions/convnext_tiny_val.csv (bilinear resize)",
        "argmax_agreement_ham_val": agreement,
        "note": "same deployed transform as the reference; must reproduce it"}
    if agreement < CONTROL_AGREEMENT_MIN:
        raise SystemExit(f"H0 does not reproduce the checkpoint (agreement {agreement:.3f})")

    selection["inputs"] = {rel(FEATURES_FIT): sha256(FEATURES_FIT),
                           rel(CHECKPOINT): sha256(CHECKPOINT),
                           rel(MANIFEST): sha256(MANIFEST)}
    selection["state"] = {rel(STATE_PATH): sha256(STATE_PATH)}
    with SELECTION_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(selection, indent=2, sort_keys=True, default=float))
    print(f"\ncontrol argmax agreement on HAM val: {agreement:.4f}")
    print(f"val router accuracy: {selection['val_router_accuracy']:.4f}")
    for k, v in selection["val_macro_f1"].items():
        print(f"  val Macro-F1 {k:<18} {v:.4f}")
    print(f"wrote {rel(SELECTION_PATH)}, {rel(STATE_PATH)}")
    write_ledger([{"method": "S58_select", "split": "val",
                   "macro_f1": selection["val_macro_f1"]["H2_routed"],
                   "notes": "S58 front end fit on V4 train, C selected on val+ham_val; "
                            f"val Macro-F1 H0 {selection['val_macro_f1']['H0_ham_head']:.4f} / "
                            f"H1 {selection['val_macro_f1']['H1_pooled_head']:.4f} / "
                            f"H2 {selection['val_macro_f1']['H2_routed']:.4f}; router acc "
                            f"{selection['val_router_accuracy']:.4f}; eligible "
                            f"{selection['eligible_domains']}"}], prune=["S58_select"])
    return 0


# ============================================================================ plan + receipt
def plan_payload(selection: dict[str, Any]) -> dict[str, Any]:
    return {
        "session": "S58",
        "title": "Domain router + domain-matched head (three-stage front end)",
        "trunk": f"{rel(CHECKPOINT)} frozen, penultimate features, 224 px, 1 view, "
                 "deployed eval transform (bilinear 256 -> crop 224)",
        "fit_split": "manifest_v4 split=train (HAM 6,981 in-sample for the trunk; BCN 6,704 "
                     "and MSKCC 1,609 out-of-sample)",
        "selection_split": "manifest_v4 split in {val, ham_val} (BCN 1,853, MSKCC 417, HAM 1,532)",
        "evaluation": "manifest_v4 split=reserved (4,733 images, 0 HAM, 104 under-40 escalating "
                      "lesions); features re-extracted with the deployed transform; "
                      "read once, receipt results/v4/s58/reserved_receipt.json",
        "ood_probe": "PAD-UFES-20, all 2,106 images, same transform (not sealed; S8b scored it)",
        "components": {
            "gate": f"research.selective.mahalanobis (tied, Ledoit-Wolf), 7 classes, fit on train; "
                    f"threshold = held-out in-scope {GATE_QUANTILE} quantile",
            "router": "L2 multinomial logistic over (ham, bcn20000, mskcc), balanced, "
                      "C by held-out log loss",
            "heads": "L2 multinomial logistic, standardised, balanced, C by held-out Macro-F1 of "
                     "the head's own domain (pooled: all held-out rows); HAM = checkpoint layer",
            "eligibility": f"own head only if every class has >= {MIN_CLASS_IMAGES} train images "
                           "in the domain, else the pooled head",
            "c_grid": list(C_GRID)},
        "arms": {"H0_ham_head": "checkpoint final layer (control)",
                 "H1_pooled_head": "one head on all train rows",
                 "H2_routed": "router -> domain head (the system)",
                 "H3_oracle_routed": "H2 heads, true archive (diagnostic: router cost H3 - H2)",
                 "D1_routed_mskcc3": "H2 with the 3-class MSKCC head -- descriptive, not "
                                     "deployable (label space is an archive-curation artifact)"},
        "references_descriptive": ["deployed V1 ensemble (S54 frozen reserved probabilities)",
                                   "V4 pooled control R0 seeds 42/43/44 _last (S54 predictions)"],
        "primary_endpoint": {
            "quantity": "reserved 7-class Macro-F1, H2 - H0",
            "interval": f"lesion-grouped (group_id) paired percentile bootstrap, {N_BOOT} draws, "
                        f"seed {BOOT_SEED}",
            "mcid": MCID,
            "outcomes": {"SUPPORTED": "delta >= MCID and CI lower > 0",
                         "POSITIVE_BELOW_MCID": "CI lower > 0 and delta < MCID",
                         "NOT_RESOLVED": "CI contains 0",
                         "HARM": "CI upper < 0"}},
        "holm_family": ["macro_f1 H2 - H0 (all reserved)",
                        f"escalation-mass pAUC@{FPR_MAX} McClish H2 - H0 ({PRIMARY_BAND} rows, "
                        "bootstrap over that band's groups -- S51/S54 convention)"],
        "p_value": "two-sided bootstrap: min(1, 2 * min((#d<=0)+1, (#d>=0)+1) / (B+1))",
        "design_rule": f"stage 3 = H2 if (H2 - H1 Macro-F1) >= 0 and its CI lower > "
                       f"-{NONINFERIORITY}; otherwise stage 3 = H1 and the router is a "
                       "provenance / drift signal only",
        "stage_targets": STAGE_TARGETS,
        "stage_reports": ["gate: reserved false-reject rate (overall, escalating, per band), PAD "
                          "reject rate, AUROC reserved-vs-PAD and ham_val-vs-PAD",
                          "router: reserved accuracy overall / per archive / within nv and mel",
                          "head: marginals per arm, per archive, per band; H2 - H0 on the "
                          "gate-accepted subset (descriptive)"],
        "selection_file": {rel(SELECTION_PATH): sha256(SELECTION_PATH)},
        "fit_state": selection["state"],
        "inputs": selection["inputs"] | {rel(FEATURES_RESERVED): sha256(FEATURES_RESERVED),
                                         rel(FEATURES_PAD): sha256(FEATURES_PAD)},
        "selected": {k: selection[k]["selected_c"] for k in
                     ("router", "pooled", "head_bcn20000", "head_mskcc")},
        "eligible_domains": selection["eligible_domains"],
        "deviations": [
            "runbook lists head training as a GPU step; the heads are logistic regressions on "
            "cached features and are fit on CPU -- same model class, exact solver",
            "HAM train features are in-sample for the trunk; they enter the pooled head and the "
            "gate fit",
            "MSKCC routes to the pooled head (eligibility rule); its own 3-class head is D1 only"],
        "not_done": ["no HAM test read", "no trunk weight updated",
                     "no calibration, lambda or abstention: S59 composes those"],
    }


def freeze_plan() -> int:
    if RECEIPT_PATH.is_file():
        raise SystemExit("reserved already read under a frozen S58 plan; refusing to rewrite it")
    if not SELECTION_PATH.is_file():
        raise SystemExit("run --select first: the plan pins the selection and the fit state")
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    for path, digest in (selection["state"] | selection["inputs"]).items():
        if sha256(REPO_ROOT / path) != digest:
            raise SystemExit(f"{path} changed since --select")
    with PLAN_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(plan_payload(selection), indent=2, sort_keys=True))
    digest = sha256(PLAN_PATH)
    print(f"wrote {rel(PLAN_PATH)}\nsha256 {digest}")
    write_ledger([{"method": "S58_plan", "split": "none",
                   "notes": f"S58 plan frozen before the reserved read; primary Macro-F1 H2-H0 "
                            f"MCID {MCID}; Holm over Macro-F1 + <40 pAUC; sha256 {digest}"}],
                 prune=["S58_plan"])
    return 0


def receipt_begin(rerun_reason: str | None, resume: bool = False,
                  receipt_path: Path = RECEIPT_PATH, plan_path: Path = PLAN_PATH) -> dict[str, Any]:
    from research.v4.s54_guard import git_head, now

    if not plan_path.is_file():
        raise SystemExit("plan not frozen: run --freeze-plan")
    digest = sha256(plan_path)
    receipt = (json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.is_file()
               else {"cohort": "manifest_v4 split=reserved (S58)", "plan_sha256": digest,
                     "executions": []})
    if receipt["plan_sha256"] != digest:
        raise SystemExit("s58_plan.json changed after the reserved read; refusing")
    last = receipt["executions"][-1] if receipt["executions"] else None
    if resume:
        if last is None or last["status"] != "started":
            raise SystemExit("--resume needs an execution that started and did not complete")
        last.setdefault("resumed_at", []).append(now())
    else:
        if last is not None and last["status"] == "started":
            raise SystemExit("the last S58 reserved execution did not complete: use --resume")
        if receipt["executions"] and not rerun_reason:
            raise SystemExit("S58 has already read the reserved cohort; a repeat needs "
                             "--rerun-reason, which the receipt keeps permanently")
        receipt["executions"].append({"execution": len(receipt["executions"]) + 1,
                                      "status": "started", "started_at": now(),
                                      "rerun_reason": rerun_reason, "git_head": git_head()})
    _write_json(receipt_path, receipt)
    return receipt


def receipt_complete(receipt: dict[str, Any], items: list[Path],
                     receipt_path: Path = RECEIPT_PATH) -> None:
    from research.v4.s54_guard import now

    receipt["executions"][-1].update(status="completed", completed_at=now(),
                                     items={rel(p): sha256(p) for p in items})
    _write_json(receipt_path, receipt)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, default=float))


def write_ledger(rows: list[dict[str, Any]], prune: list[str]) -> None:
    """Prune this session's rows for these methods, then append (the S20 hazard)."""
    stamp = pd.Timestamp.now(tz="UTC").isoformat()
    frame = pd.DataFrame([{"timestamp": stamp, "session": SESSION, **r} for r in rows])
    old = pd.read_csv(LEDGER_PATH, low_memory=False)
    kept = old[~((old["session"] == SESSION) & (old["method"].isin(prune)))]
    print(f"ledger: pruned {len(old) - len(kept)} prior {SESSION} row(s), appended {len(frame)}")
    pd.concat([kept, frame], ignore_index=True).to_csv(LEDGER_PATH, index=False)


# ============================================================================ statistics
def grouped_draws(statistic: Callable[[np.ndarray], float], groups: np.ndarray,
                  n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> np.ndarray:
    """Lesion-grouped bootstrap draws (non-finite draws dropped)."""
    rng = np.random.default_rng(seed)
    codes, inverse = np.unique(groups, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    starts = np.searchsorted(inverse[order], np.arange(len(codes)))
    ends = np.append(starts[1:], len(order))
    draws = []
    for _ in range(n_boot):
        picked = rng.integers(0, len(codes), len(codes))
        idx = np.concatenate([order[starts[g]:ends[g]] for g in picked])
        value = statistic(idx)
        if np.isfinite(value):
            draws.append(value)
    return np.asarray(draws)


def boot_p(draws: np.ndarray) -> float:
    b = len(draws)
    lo = (np.sum(draws <= 0) + 1) / (b + 1)
    hi = (np.sum(draws >= 0) + 1) / (b + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted.tolist()


def classify(delta: float, lo: float, hi: float) -> str:
    if hi < 0:
        return "HARM"
    if lo <= 0:
        return "NOT_RESOLVED"
    return "SUPPORTED" if delta >= MCID else "POSITIVE_BELOW_MCID"


def paired(label: str, a: np.ndarray, b: np.ndarray, panel: pd.DataFrame,
           mask: np.ndarray | None = None, n_boot: int = N_BOOT) -> list[dict[str, Any]]:
    keep = np.ones(len(panel), bool) if mask is None else mask
    y7 = panel["y7"].to_numpy()[keep]
    groups = panel["group_id"].to_numpy()[keep]
    pa, pb = a[keep].argmax(1), b[keep].argmax(1)
    band = (panel["age_band"].to_numpy()[keep] == PRIMARY_BAND)
    y_esc = panel["y_esc"].to_numpy()[keep][band]
    sa, sb = escalation_mass(a[keep])[band], escalation_mass(b[keep])[band]
    rows = []

    def f1d(idx: np.ndarray) -> float:
        return macro_f1(y7[idx], pa[idx]) - macro_f1(y7[idx], pb[idx])

    def pad(idx: np.ndarray) -> float:
        yy = y_esc[idx]
        if yy.all() or not yy.any():
            return float("nan")
        return pauc(yy, sa[idx]) - pauc(yy, sb[idx])

    for endpoint, stat, g, n in (("macro_f1", f1d, groups, len(y7)),
                                 (f"pauc_{PRIMARY_BAND}", pad, groups[band], int(band.sum()))):
        point = stat(np.arange(len(g))) if n else np.nan
        draws = grouped_draws(stat, g, n_boot) if n else np.array([])
        lo, hi = np.percentile(draws, [2.5, 97.5]) if len(draws) else (np.nan, np.nan)
        rows.append({"contrast": label, "endpoint": endpoint, "n": n, "delta": float(point),
                     "ci_lo": float(lo), "ci_hi": float(hi),
                     "p_boot": boot_p(draws) if len(draws) else np.nan,
                     "n_draws": len(draws)})
    return rows


def marginal(name: str, probs: np.ndarray, panel: pd.DataFrame, subset: str,
             mask: np.ndarray, n_boot: int = N_BOOT) -> dict[str, Any]:
    from research.v2 import frontier as fr

    y7 = panel["y7"].to_numpy()[mask]
    groups = panel["group_id"].to_numpy()[mask]
    pred = probs[mask].argmax(1)
    f1 = macro_f1(y7, pred)
    lo, hi = np.percentile(grouped_draws(lambda i: macro_f1(y7[i], pred[i]), groups, n_boot),
                           [2.5, 97.5])
    row = {"model": name, "subset": subset, "n": int(mask.sum()), "macro_f1": f1,
           "macro_f1_ci_lo": float(lo), "macro_f1_ci_hi": float(hi),
           "balanced_accuracy": float(np.mean([(pred[y7 == c] == c).mean()
                                               for c in np.unique(y7)]))}
    y_esc = panel["y_esc"].to_numpy()[mask]
    esc_pred = np.isin(pred, esc_idx())
    row["escalation_sensitivity"] = float(esc_pred[y_esc].mean()) if y_esc.any() else np.nan
    band = panel["age_band"].to_numpy()[mask] == PRIMARY_BAND
    if y_esc[band].any() and (~y_esc[band]).any():
        pa = fr.partial_auc_ci(y_esc[band], escalation_mass(probs[mask])[band], groups[band],
                               fpr_max=FPR_MAX, seed=BOOT_SEED, n_boot=n_boot)
        row |= {"pauc_u40": pa["partial_auc_mcclish"], "pauc_u40_ci_lo": pa["ci_lo"],
                "pauc_u40_ci_hi": pa["ci_hi"],
                "u40_escalation_sensitivity": float(esc_pred[band & y_esc].mean())}
    return row


def esc_idx() -> list[int]:
    from research.v4.s54_gate import escalating_indices

    return escalating_indices()


def auroc(inside: np.ndarray, outside: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(np.r_[np.zeros(len(inside)), np.ones(len(outside))],
                               np.r_[inside, outside]))


# ============================================================================ evaluation
def evaluation_panel(smoke: bool) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    from research.v4 import s54_gate as g

    if smoke:
        # V4 val stands in for reserved: wiring only, no receipt, nothing reported
        frame, feats = fit_panel(SMOKE_FEAT_DIR / FEATURES_FIT.name)
        keep = (frame["split"] == "val").to_numpy()
        panel = frame[keep].reset_index(drop=True)
        panel = panel.assign(y7=panel["class_index_7"].astype(int),
                             y_esc=panel["escalating_7"].astype(bool))
        return panel, feats[keep], {"smoke": True}
    panel = g.build_panel("reserved")
    cohort = g.check_cohort(panel)
    feats, ids = _load_npz(FEATURES_RESERVED)
    return panel, align(feats, ids, panel["image_id"]), {"cohort": cohort}


def references(panel: pd.DataFrame) -> dict[str, np.ndarray]:
    from research.v4 import s54_gate as g

    probs, covered, _ = g.load_v1_probs(panel, smoke=False)
    assert covered.all()
    out = {"REF_V1_deployed_ensemble": probs}
    for seed in (42, 43, 44):
        p, _ = g.load_v4_probs(g.prediction_path("control", seed, "last", False), panel)
        out[f"REF_V4_pooled_control_s{seed}_last"] = p
    return out


def run_evaluate(smoke: bool, rerun_reason: str | None, resume: bool, n_boot: int) -> int:
    from research import testguard

    testguard.block_test_reads("S58 evaluation: reserved (or V4 val in smoke) only")
    out_dir = SMOKE_DIR if smoke else OUT_DIR
    if not smoke:
        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8")) if PLAN_PATH.is_file() else None
        if plan is None:
            raise SystemExit("plan not frozen: run --freeze-plan")
        for path, digest in (plan["fit_state"] | plan["inputs"]).items():
            if sha256(REPO_ROOT / path) != digest:
                raise SystemExit(f"{path} differs from the frozen plan")
        receipt = receipt_begin(rerun_reason, resume)
        state = load_state()
        eligible = plan["eligible_domains"]
    else:
        frame, feats = fit_panel(SMOKE_FEAT_DIR / FEATURES_FIT.name)
        _, arrays = fit_components(frame, feats, verbose=False, strict=False)
        tmp = Path(tempfile.mkdtemp()) / "state.npz"
        np.savez_compressed(tmp, **arrays)
        state = load_state(tmp)
        eligible = eligible_domains(manifest())

    panel, x, meta = evaluation_panel(smoke)
    true_domain = panel["archive"].to_numpy(str)
    arms, routed_to = arm_probs(state, x, true_domain, eligible)
    models = dict(arms) if smoke else arms | references(panel)

    # ---- stage 1: gate
    from research.stats.intervals import proportion

    s_in = gate_scores(state, x)
    thr = state["gate_threshold"]
    rejected = s_in > thr
    pad_x, _ = _load_npz((SMOKE_FEAT_DIR if smoke else FEAT_DIR) / FEATURES_PAD.name)
    s_pad = gate_scores(state, pad_x)
    groups = panel["group_id"].to_numpy()
    gate_rows = []
    for label, m in (("all", np.ones(len(panel), bool)),
                     ("escalating", panel["y_esc"].to_numpy()),
                     *[(f"band {b}", panel["age_band"].to_numpy() == b)
                       for b in ("<40", "40-59", "60+", "unknown")],
                     *[(f"archive {a}", true_domain == a) for a in ("bcn20000", "mskcc")]):
        if m.sum() == 0:
            continue
        p = proportion(rejected[m], groups[m], label=label, n_boot=n_boot)
        gate_rows.append({"subset": label, "n": int(m.sum()), "false_reject": p.point,
                          "ci_lo": p.interval[0], "ci_hi": p.interval[1]})
    gate = {"threshold": thr, "reserved_false_reject": float(rejected.mean()),
            "pad_reject": float((s_pad > thr).mean()), "pad_n": int(len(s_pad)),
            "auroc_reserved_vs_pad": auroc(s_in, s_pad)}
    if not smoke:
        frame, feats = fit_panel()
        hv = (frame["split"] == "ham_val").to_numpy()
        gate["auroc_ham_val_vs_pad"] = auroc(gate_scores(state, feats[hv]), s_pad)
        gate["ham_val_false_reject_in_sample_threshold"] = float(
            (gate_scores(state, feats[hv]) > thr).mean())

    # ---- stage 2: router
    confusion = pd.crosstab(pd.Series(true_domain, name="true"),
                            pd.Series(routed_to, name="routed")).reindex(columns=list(DOMAINS),
                                                                         fill_value=0)
    correct = routed_to == true_domain
    router = {"accuracy": float(correct.mean()),
              "per_archive": {a: float(correct[true_domain == a].mean())
                              for a in np.unique(true_domain)},
              "within_class": {c: float(correct[(panel["class_7"] == c).to_numpy()].mean())
                               for c in ("nv", "mel") if (panel["class_7"] == c).any()},
              "confusion": confusion.to_dict()}

    # ---- stage 3: heads
    everyone = np.ones(len(panel), bool)
    subsets = [("all", everyone), ("gate_accepted", ~rejected)]
    subsets += [(f"archive {a}", true_domain == a) for a in np.unique(true_domain)]
    marg = [marginal(name, p, panel, s, m, n_boot) for name, p in models.items()
            for s, m in subsets if name in arms or s == "all"]
    contrasts = []
    for label, a, b, mask in (("H2-H0", "H2_routed", "H0_ham_head", None),
                              ("H2-H1", "H2_routed", "H1_pooled_head", None),
                              ("H1-H0", "H1_pooled_head", "H0_ham_head", None),
                              ("H3-H2", "H3_oracle_routed", "H2_routed", None),
                              ("D1-H2", "D1_routed_mskcc3", "H2_routed", None),
                              ("H2-H0 gate_accepted", "H2_routed", "H0_ham_head", ~rejected)):
        contrasts += paired(label, models[a], models[b], panel, mask, n_boot)
    cframe = pd.DataFrame(contrasts)

    fam = cframe[(cframe["contrast"] == "H2-H0")].reset_index(drop=True)
    adjusted = holm(fam["p_boot"].tolist())
    cframe["p_holm"] = np.nan
    for (i, row), p in zip(fam.iterrows(), adjusted):
        cframe.loc[(cframe["contrast"] == "H2-H0") & (cframe["endpoint"] == row["endpoint"]),
                   "p_holm"] = p
    prim = fam[fam["endpoint"] == "macro_f1"].iloc[0]
    design = cframe[(cframe["contrast"] == "H2-H1") & (cframe["endpoint"] == "macro_f1")].iloc[0]
    routed_wins = bool(design["delta"] >= 0 and design["ci_lo"] > -NONINFERIORITY)
    verdict = {
        "primary": {"delta": prim["delta"], "ci": [prim["ci_lo"], prim["ci_hi"]],
                    "p_holm": float(adjusted[0]), "outcome": classify(prim["delta"],
                                                                      prim["ci_lo"],
                                                                      prim["ci_hi"])},
        "stage3": "H2_routed" if routed_wins else "H1_pooled_head",
        "router_role": "selects the head" if routed_wins else "provenance / drift signal only",
        "stage_checks": {
            "gate_reserved_false_reject": gate["reserved_false_reject"]
            <= STAGE_TARGETS["gate_reserved_false_reject_max"],
            "gate_pad_reject": gate["pad_reject"] >= STAGE_TARGETS["gate_pad_reject_min"],
            "router_accuracy": router["accuracy"]
            >= STAGE_TARGETS["router_reserved_accuracy_min"]}}

    out_dir.mkdir(parents=True, exist_ok=True)
    mpath, cpath, gpath, rpath = (out_dir / "stage3_marginals.csv", out_dir / "stage3_contrasts.csv",
                                  out_dir / "stage1_gate.csv", out_dir / "s58_report.json")
    pd.DataFrame(marg).to_csv(mpath, index=False)
    cframe.to_csv(cpath, index=False)
    pd.DataFrame(gate_rows).to_csv(gpath, index=False)
    report = {"session": "S58", "smoke": smoke, "test_read": False, "meta": meta,
              "plan_sha256": None if smoke else sha256(PLAN_PATH), "n_boot": n_boot,
              "gate": gate, "router": router, "verdict": verdict,
              "generated_at": pd.Timestamp.now(tz="UTC").isoformat()}
    _write_json(rpath, report)
    print_report(report, pd.DataFrame(marg), cframe)
    if smoke:
        print(f"\nSMOKE ONLY -> {rel(out_dir)}; nothing here is a result")
        return 0

    receipt_complete(receipt, [mpath, cpath, gpath, rpath])
    rows = [{"method": f"S58_{r['model']}", "split": "reserved", "macro_f1": r["macro_f1"],
             "balanced_accuracy": r["balanced_accuracy"],
             "escalation_sensitivity": r["escalation_sensitivity"],
             "notes": f"S58 front end, subset=all; <40 pAUC@{FPR_MAX} {r.get('pauc_u40', np.nan):.4f}"}
            for r in marg if r["subset"] == "all"]
    rows.append({"method": "S58_verdict", "split": "reserved", "macro_f1": prim["delta"],
                 "notes": f"H2-H0 Macro-F1 {prim['delta']:+.4f} [{prim['ci_lo']:+.4f}, "
                          f"{prim['ci_hi']:+.4f}] Holm p {adjusted[0]:.3g} -> "
                          f"{verdict['primary']['outcome']}; stage3 {verdict['stage3']}; gate FRR "
                          f"{gate['reserved_false_reject']:.3f}, PAD reject {gate['pad_reject']:.3f}; "
                          f"router acc {router['accuracy']:.3f}"})
    write_ledger(rows, prune=[r["method"] for r in rows])
    return 0


def print_report(report: dict[str, Any], marg: pd.DataFrame, cframe: pd.DataFrame) -> None:
    g, r, v = report["gate"], report["router"], report["verdict"]
    print(f"\n[gate]   threshold {g['threshold']:.1f}  in-scope false-reject "
          f"{g['reserved_false_reject']:.3f}  PAD reject {g['pad_reject']:.3f}  "
          f"AUROC in-vs-PAD {g['auroc_reserved_vs_pad']:.3f}")
    print(f"[router] accuracy {r['accuracy']:.4f}  per archive {r['per_archive']}")
    allm = marg[marg["subset"] == "all"]
    print("[heads]  reserved, all rows")
    for _, row in allm.iterrows():
        print(f"  {row['model']:<36} F1 {row['macro_f1']:.4f} "
              f"[{row['macro_f1_ci_lo']:.4f}, {row['macro_f1_ci_hi']:.4f}]  "
              f"<40 pAUC {row.get('pauc_u40', np.nan):.4f}")
    print("[contrasts]")
    for _, row in cframe.iterrows():
        print(f"  {row['contrast']:<20} {row['endpoint']:<10} {row['delta']:+.4f} "
              f"[{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}]  p {row['p_boot']:.3g}")
    print(f"\nprimary: {v['primary']['outcome']}  stage 3: {v['stage3']}  "
          f"checks: {v['stage_checks']}")


# ============================================================================ self-test
def selftest() -> int:
    from sklearn.linear_model import LogisticRegression

    print("s58_front_end.py self-test\n")
    results = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append(ok)
        print(f"  {len(results):>2}. {name}: {detail} -> {'PASS' if ok else 'FAIL'}")

    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, 12))
    y = rng.choice([0, 2, 5], size=600)
    x[:, 0] += y
    head = fit_lr(x, y, 1.0)
    ref = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, tol=1e-5).fit(
        (x - head.mean) / head.scale, y).predict_proba((x - head.mean) / head.scale)
    p = head.predict_proba(x)
    check("stored head reproduces sklearn", np.allclose(p[:, [0, 2, 5]], ref, atol=1e-8),
          f"max|d|={np.abs(p[:, [0, 2, 5]] - ref).max():.1e}")
    check("absent classes get zero, rows sum to 1",
          bool(np.all(p[:, [1, 3, 4, 6]] == 0) and np.allclose(p.sum(1), 1)))

    yb = (x[:, 0] > 2).astype(int) * 4
    hb = fit_lr(x, yb, 1.0)
    check("binary head keeps both classes", hb.coef.shape[0] == 2 and
          np.array_equal(hb.predict_proba(x).argmax(1) == 4,
                         LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000,
                                            tol=1e-5).fit((x - hb.mean) / hb.scale, yb)
                         .predict((x - hb.mean) / hb.scale) == 4))

    heads = {"ham": LinearHead(np.zeros(12), np.ones(12), np.eye(K, 12), np.zeros(K), np.arange(K)),
             "bcn20000": head, "mskcc": head}
    doms = np.array(["ham", "bcn20000"] * 300)
    out = assemble(doms, x, heads)
    check("assemble routes each row to its head",
          np.allclose(out[doms == "bcn20000"], head.predict_proba(x[doms == "bcn20000"])) and
          np.allclose(out[doms == "ham"], heads["ham"].predict_proba(x[doms == "ham"])))

    frame = manifest()
    eligible = eligible_domains(frame)
    check("eligibility: BCN own head, MSKCC falls back, HAM eligible",
          eligible == {"ham": True, "bcn20000": True, "mskcc": False}, str(eligible))

    ext = extraction_frame(frame)
    check("extraction frame = train + val + ham_val only",
          set(ext["split"]) == set(FIT_SPLITS) and len(ext) == int(frame["split"]
                                                                    .isin(FIT_SPLITS).sum()),
          f"{len(ext)} rows")
    res = frame[frame["split"] == "reserved"]
    check("fit rows and reserved share no image and no group",
          not set(ext["image_id"]) & set(res["image_id"]) and
          not set(ext["group_id"]) & set(res["group_id"]))

    from research.v4.extract_backbone_features import IMAGE_DIR
    sample = ext.groupby("archive").sample(30, random_state=0)
    pad = pd.read_csv(PAD_MANIFEST).sample(30, random_state=0)
    found = sum((IMAGE_DIR / f"{i}.jpg").is_file() for i in sample["image_id"])
    found_pad = sum((REPO_ROOT / p).is_file() for p in pad["path"])
    check("images resolve (90 manifest, 30 PAD)", found == 90 and found_pad == 30,
          f"{found}/90, {found_pad}/30")

    check("bootstrap p: centred draws ~1, one-sided draws small",
          boot_p(rng.normal(size=2000)) > 0.5 and boot_p(np.abs(rng.normal(size=2000)) + .1)
          < 0.001)
    check("holm step-down", np.allclose(holm([0.01, 0.04]), [0.02, 0.04]) and
          np.allclose(holm([0.04, 0.01]), [0.04, 0.02]))
    groups = np.repeat(np.arange(50), 3)
    d = grouped_draws(lambda i: float(len(np.unique(groups[i]))), groups, n_boot=50)
    check("grouped draws resample whole groups", bool(np.all(d <= 50) and d.min() > 20))

    frame_s = pd.DataFrame({"split": ["train"] * 400 + ["val"] * 200,
                            "class_index_7": np.tile(np.arange(K), 86)[:600],
                            "archive": np.tile(DOMAINS, 200)})
    xs = rng.normal(size=(600, 8)) + frame_s["class_index_7"].to_numpy()[:, None] * .5
    from research.selective import mahalanobis
    st = mahalanobis.fit(xs[:400], frame_s["class_index_7"].to_numpy()[:400], K)
    sc = mahalanobis.score(st, xs[400:])
    rate = float((sc > np.quantile(sc, GATE_QUANTILE)).mean())
    check("gate threshold realises its nominal rate on its own held-out rows",
          abs(rate - (1 - GATE_QUANTILE)) < 0.01, f"{rate:.3f}")

    with tempfile.TemporaryDirectory() as tmp:
        plan, rec = Path(tmp) / "plan.json", Path(tmp) / "receipt.json"
        plan.write_text("{}", encoding="utf-8")
        r = receipt_begin(None, receipt_path=rec, plan_path=plan)
        blocked = False
        try:
            receipt_begin(None, receipt_path=rec, plan_path=plan)
        except SystemExit:
            blocked = True                                    # started, not completed
        receipt_complete(r, [], receipt_path=rec)
        try:
            receipt_begin(None, receipt_path=rec, plan_path=plan)
        except SystemExit:
            blocked &= True
        else:
            blocked = False
        plan.write_text('{"x": 1}', encoding="utf-8")
        try:
            receipt_begin("edited", receipt_path=rec, plan_path=plan)
        except SystemExit:
            pass
        else:
            blocked = False
    check("receipt refuses repeat, incomplete and edited-plan reads", blocked)

    from research import testguard
    testguard.block_test_reads("selftest")
    try:
        testguard.check_split("test", "selftest")
        armed = False
    except Exception:  # noqa: BLE001
        armed = True
    finally:
        testguard.allow_test_reads()
    check("test lock refuses a test read", armed)

    ok = all(results)
    print(f"\n{'ALL CHECKS PASS' if ok else 'SELF-TEST FAILED'} ({sum(results)}/{len(results)})")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S58 -- domain router + domain-matched head")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--selftest", action="store_true")
    mode.add_argument("--extract", action="store_true", help="GPU feature extraction (user runs)")
    mode.add_argument("--select", action="store_true", help="fit on train, select on val")
    mode.add_argument("--freeze-plan", action="store_true")
    mode.add_argument("--reserved", action="store_true", help="the one reserved read")
    parser.add_argument("--smoke", action="store_true",
                        help="--extract: a few hundred rows; --reserved: V4 val, no receipt")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--rerun-reason")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.extract:
        return run_extract(args.smoke, args.workers, args.batch_size)
    if args.select:
        return run_select()
    if args.freeze_plan:
        return freeze_plan()
    if args.n_boot != N_BOOT and not args.smoke:
        raise SystemExit(f"the plan fixes n_boot={N_BOOT}")
    return run_evaluate(args.smoke, args.rerun_reason, args.resume, args.n_boot)


if __name__ == "__main__":
    raise SystemExit(main())
