"""M1 protocol amendment 01: density-normalised scores, written BEFORE the held-out benchmark is read for them.

    python -m research.v7.m1.amend write       # writes results/v7/m1/protocol_amendment_01.json once; refuses to rewrite it
    python -m research.v7.m1.amend score       # the one held-out read: both candidates against the kept baseline

Why an amendment: the kept matcher (SSCD, raw cosine, one threshold) finds the right neighbour for about 90% of images but
links about a quarter of same-lesion pairs, because dense look-alike regions set the threshold (`density.py`). Everything
below was chosen on the TRAINING clusters only; the held-out clusters of `protocol.json` have not been scored with it.

Fixed here:
* Candidates, both with the score of `density.py` (ranks LO..HI-1 over every master row, weight 1):
  `sscd_sn`        SSCD, untrained;
  `sscd_femto_sn`  SSCD fused with the trained ConvNeXt-Femto (the follow-up declared when Femto was not adopted): each is
                   normalised by its own density and the two normalised scores are averaged.
* Benchmark, primary endpoint and adoption rule: those of `protocol.json`, unchanged (pair TPR at FMR 1e-5 on the same
  9,000 held-out images and the same clusters; paired lesion-bootstrap 95% interval of the difference above 0 and not lower
  on any archive with >= 100 held-out pairs). The baseline is the kept matcher, SSCD raw.
* Order: each candidate is judged against the baseline. If both pass, the fused one is adopted only if it also passes the
  same rule against `sscd_sn`; otherwise `sscd_sn` (no trained part). If neither passes, SSCD raw stays.
* Secondary, reported, never deciding: TPR at FMR 1e-6 and 1e-4; the share of images with a partner that are linked to at
  least one of them; same-lesion pairs joined after transitive closure; different-cluster pairs joined by closure; the
  largest component.
* The tier rule applied by the scan: `tiers.RULE`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch

from research import testguard
from research.v7.m1 import bench, data, density, protocol, tiers

AMENDMENT = data.OUT / "protocol_amendment_01.json"
RESULT = data.OUT / "amendment_01.json"
DEVELOPMENT = data.OUT / "amendment_01_development.json"   # `develop.py`: the training-cluster evidence behind every choice
FEMTO = "ml/checkpoints/m1/m1_convnext_femto_ms_s42.pt"
CANDIDATES = {"sscd_sn": ("sscd",), "sscd_femto_sn": ("sscd", FEMTO)}
BASELINE = "sscd"
FMR_STRICT = 1e-6
FMRS = (protocol.FMR_PRIMARY, protocol.FMR_SECONDARY, FMR_STRICT)
BENCH_SOURCE = data.REPO / "ml" / "data" / "v7" / "_pre_m1_2026-11-12"   # the links the benchmark's clusters were drawn from


def spec() -> dict:
    return {"what": "M1 amendment 01: density-normalised scores, fixed before the held-out benchmark is read for them",
            "amends_protocol_sha256": protocol.require()["sha256"],
            "density": {"ranks": [density.LO, density.HI], "weight": 1.0, "gallery": "every master-table row, pixels only",
                        "fusion": "each matcher normalised by its own density; fused score = mean of the normalised scores"},
            "candidates": {k: list(v) for k, v in CANDIDATES.items()}, "femto_checkpoint_sha256": bench.file_sha(data.REPO / FEMTO),
            "baseline": "sscd, raw cosine (the kept matcher)", "benchmark": "the held-out images and clusters of protocol.json, unchanged",
            "primary": f"pair TPR at FMR {protocol.FMR_PRIMARY}", "fmr_strict": FMR_STRICT,
            "adoption": "protocol.json's rule against the baseline; the fused candidate must also pass it against sscd_sn",
            "chosen_on": "training clusters only (pre-M1 links); evidence in amendment_01_development.json",
            "development_receipt_sha256": bench.file_sha(DEVELOPMENT),
            "tier_rule": tiers.RULE}


def _sha(record: dict) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def write() -> dict:
    if AMENDMENT.is_file():
        raise SystemExit(f"{AMENDMENT} exists; the amendment is written once")
    if RESULT.is_file():
        raise SystemExit("the amendment's result exists; it cannot be written after the read")
    record = {**spec(), "written": time.strftime("%Y-%m-%dT%H:%M:%S")}
    record["sha256"] = _sha(spec())
    AMENDMENT.write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


def require() -> dict:
    if not AMENDMENT.is_file():
        raise SystemExit("write the amendment first: python -m research.v7.m1.amend write")
    record = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    if record["sha256"] != _sha(spec()):
        raise SystemExit("amendment 01 (or the tier rule, the density ranks or the Femto checkpoint) changed after it was written")
    return record


def result() -> dict:
    """The amendment's decision, for the scan and the verification; refuses a result from another amendment text."""
    if not RESULT.is_file():
        raise SystemExit("amendment 01 has not been scored: python -m research.v7.m1.amend score")
    record = json.loads(RESULT.read_text(encoding="utf-8"))
    if record["amendment_sha256"] != require()["sha256"]:
        raise SystemExit("amendment_01.json was scored under a different amendment text")
    return record


# ------------------------------------------------------------------ embeddings
def scored(method: str, master: pd.DataFrame, device) -> tuple[np.ndarray, np.ndarray]:
    """A candidate's embedding of every master row and each row's offset (`density.pair_scores`).

    Each matcher is normalised by its OWN local density; a fused score is the mean of the parts' normalised scores (fused
    embedding = concatenation / sqrt(n), offset = mean of the parts' half densities). Normalising the concatenation instead
    was worse on the inner-validation clusters Femto never trained on: pair TPR at FMR 1e-6 0.478 against 0.564.
    """
    from research.v7.m1 import scan
    embs = [scan.embed_all(p, master, device) for p in CANDIDATES[method]]
    halves = [density.local_density(e, device) / 2.0 for e in embs]
    if len(embs) == 1:
        return embs[0], halves[0]
    return density.fuse(*embs), (np.mean(halves, axis=0)).astype(np.float32)


# ------------------------------------------------------------------ decisions
def passes(base: np.ndarray, cand: np.ndarray, pairs: bench.Pairs) -> dict:
    """protocol.json's adoption rule for `cand` against `base` (hits per positive pair), paired by lesion."""
    boot = bench.lesion_bootstrap({"base": base, "cand": cand}, pairs.pos_lesion, protocol.N_BOOT, protocol.SEED)
    diff = boot["boot"]["cand"] - boot["boot"]["base"]
    ci = [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]
    per_archive, not_lower = {}, True
    for a in sorted(set(pairs.pos_archive)):
        sel = pairs.pos_archive == a
        if sel.sum() >= protocol.MIN_ARCHIVE_PAIRS:
            per_archive[a] = {"base": float(base[sel].mean()), "candidate": float(cand[sel].mean()), "positive_pairs": int(sel.sum())}
            not_lower &= per_archive[a]["candidate"] >= per_archive[a]["base"]
    return {"base_tpr": float(base.mean()), "candidate_tpr": float(cand.mean()), "candidate_tpr_ci95": boot["ci"]["cand"],
            "difference": float(cand.mean() - base.mean()), "difference_ci95": ci, "per_archive": per_archive,
            "not_lower_on_any_powered_archive": bool(not_lower), "passes": bool(ci[0] > 0 and not_lower)}


def choose(vs_base: dict[str, dict], fused_vs_plain: dict) -> str:
    """The amendment's order: the fused candidate only if it passes against the baseline AND against `sscd_sn`."""
    plain, fused = vs_base["sscd_sn"]["passes"], vs_base["sscd_femto_sn"]["passes"]
    if fused and (not plain or fused_vs_plain["passes"]):
        return "sscd_femto_sn"
    return "sscd_sn" if plain else BASELINE


def linkage(emb: np.ndarray, offset: np.ndarray | None, thr: float, pairs: bench.Pairs) -> dict:
    """What the threshold does to whole lesions: image-level recall, closure, and the false merges closure would make."""
    import scipy.sparse as sp
    import scipy.sparse.csgraph as cg
    s = emb @ emb.T
    if offset is not None:
        s = s - offset[:, None] - offset[None, :]
    np.fill_diagonal(s, -4.0)
    link = s >= thr
    _, comp = cg.connected_components(sp.csr_matrix(link), directed=False)
    p = pairs.pos
    with_partner = np.unique(p.ravel())
    sibling = (pairs.lesion[:, None] == pairs.lesion[None, :]) & (pairs.lesion[:, None] >= 0)
    np.fill_diagonal(sibling, False)
    sizes = np.bincount(comp)
    by_cluster = pd.DataFrame({"c": comp, "k": pairs.cluster}).groupby(["c", "k"]).size()
    return {"images_linked_to_a_partner": float((link & sibling).any(1)[with_partner].mean()),
            "pairs_joined_after_closure": float((comp[p[:, 0]] == comp[p[:, 1]]).mean()),
            "different_cluster_pairs_joined_by_closure": int((sizes * (sizes - 1) // 2).sum() - (by_cluster * (by_cluster - 1) // 2).sum()),
            "largest_component": int(sizes.max())}


def benchmark_frame() -> pd.DataFrame:
    """The benchmark of record: the clusters before M1's own links were added, the same 9,000 images the baselines were scored on."""
    frame = data.benchmark_set(data.rows(master=BENCH_SOURCE / "master_v7.csv", links=BENCH_SOURCE / "links_v7.csv"))
    saved = np.load(bench.EMB_DIR / f"{BASELINE}.npz", allow_pickle=False)["image_id"]
    if not np.array_equal(saved, frame["image_id"].astype(str).to_numpy()):
        raise SystemExit("the benchmark rebuilt from the pre-M1 links is not the one the baselines were scored on")
    return frame


def score(device) -> dict:
    testguard.block_test_reads("M1 amendment 01: development rows only, by lesion and patient ids; no diagnosis label is read")
    amendment = require()
    if RESULT.is_file():
        raise SystemExit(f"{RESULT} exists: the held-out benchmark is read once for this amendment")
    frame = benchmark_frame()
    pairs = bench.Pairs(frame)
    master = pd.read_csv(data.MASTER, dtype=str, low_memory=False)
    row = pd.Series(np.arange(len(master)), index=master["image_id"]).loc[frame["image_id"]].to_numpy()
    base_emb = np.load(bench.EMB_DIR / f"{BASELINE}.npz", allow_pickle=False)["emb"]
    runs = {BASELINE: (base_emb, None)}
    for name in CANDIDATES:
        emb, half = scored(name, master, device)
        runs[name] = (emb[row], half[row])
    out = {k: bench.score_embedding(e, pairs, device, offset=o, fmrs=FMRS) for k, (e, o) in runs.items()}
    fmr = str(protocol.FMR_PRIMARY)
    recorded = json.loads((data.OUT / "baselines.json").read_text(encoding="utf-8"))
    recorded_tpr = next(m["tpr"] for m in recorded["methods"] if m["method"] == f"{BASELINE} embedding, FMR {fmr}")
    if abs(out[BASELINE]["hits"][fmr].mean() - recorded_tpr) > 1e-9:
        raise SystemExit("the baseline does not reproduce baselines.json; nothing is written")
    vs_base = {k: passes(out[BASELINE]["hits"][fmr], out[k]["hits"][fmr], pairs) for k in CANDIDATES}
    fused_vs_plain = passes(out["sscd_sn"]["hits"][fmr], out["sscd_femto_sn"]["hits"][fmr], pairs)
    adopted = choose(vs_base, fused_vs_plain)
    report = {"amendment_sha256": amendment["sha256"], "protocol_sha256": protocol.require()["sha256"], "images": pairs.n,
              "positive_pairs": int(len(pairs.pos)), "negative_pairs": pairs.n_neg, "adopted": adopted,
              "against_baseline": vs_base, "fused_against_sscd_sn": fused_vs_plain, "methods": {}}
    for k, (e, o) in runs.items():
        report["methods"][k] = {"thresholds": out[k]["thresholds"], "tpr": {f: float(h.mean()) for f, h in out[k]["hits"].items()},
                                "recall_at_1": out[k]["recall_at_1"],
                                "linkage_at_primary_fmr": linkage(e, o, out[k]["thresholds"][fmr], pairs)}
    RESULT.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("write", "score"))
    args = parser.parse_args(argv)
    if args.step == "write":
        print(json.dumps(write(), indent=1))
        return 0
    report = score(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    for k, m in report["methods"].items():
        print(f"{k:<16} TPR@1e-5 {m['tpr'][str(protocol.FMR_PRIMARY)]:.3f}  @1e-6 {m['tpr'][str(FMR_STRICT)]:.3f}  "
              f"images linked {m['linkage_at_primary_fmr']['images_linked_to_a_partner']:.3f}")
    print(f"adopted: {report['adopted']} -> {RESULT.relative_to(data.REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
