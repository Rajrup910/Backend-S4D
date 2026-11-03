"""End-to-end rehearsal on real data with a tiny budget. Real trainer, real labels, real pool rows, real workers, GPU, real TTA, real decision tools, down both T paths.
Every output is redirected here, and the freeze check is stubbed, so nothing in the repo is written and nothing is a run of record."""
import json, shutil, sys, time, traceback
from pathlib import Path
import numpy as np, pandas as pd

SP = Path(sys.argv[1]); ROOT = SP / "rehearsal"
LOG = []


def say(*args):
    line = " ".join(str(a) for a in args); LOG.append(line); print("##", line, flush=True)


if __name__ == "__main__":
    import torch
    from research.v5 import train_v5
    from research.v6 import gate, plan_freeze, predict_tta as pt, stage_rules as st, system_reads as sr, train_v6
    shutil.rmtree(ROOT, ignore_errors=True)
    RUNS, PREDS, TTA, CK, SCREENS = (ROOT / n for n in ("runs", "preds", "preds_tta", "ck", "screens"))
    for d in (RUNS, PREDS, TTA, CK, SCREENS): d.mkdir(parents=True)
    real_frames, real_extra, real_added = train_v5.fold_frames, train_v5.load_extra_train, train_v6.load_added_train

    def tiny_frames(manifest, fold):
        train, val = real_frames(manifest, fold)
        hard = val[(val["age_band"] == "<40") & val["escalating_7"].astype(bool)].head(8)
        rest = val.drop(hard.index).sample(56, random_state=1)
        return train.sample(48, random_state=1).reset_index(drop=True), pd.concat([hard, rest]).sort_index().reset_index(drop=True)

    def tiny_extra(path, train, val):
        return real_extra(path, train, val).sample(16, random_state=1).reset_index(drop=True)

    def tiny_added(path, manifest, pool):
        frame = real_added(path, manifest, pool).sample(170, random_state=1).reset_index(drop=True)
        return frame.assign(image_path=[str(Path.cwd() / q) for q in frame["image_path"]])  # the harness moves REPO_ROOT, so make these absolute

    train_v6.fold_frames, train_v6.load_extra_train, train_v6.load_added_train = tiny_frames, tiny_extra, tiny_added
    pt.fold_frames = tiny_frames
    train_v6.RUN_DIR, train_v6.PRED_DIR, train_v6.CHECKPOINT_DIR = RUNS, PREDS, CK
    train_v6.REPO_ROOT = Path("C:/")
    pt.PRED_DIR, pt.TTA_DIR, pt.CHECKPOINT_DIR = PREDS, TTA, CK
    pt.REPO_ROOT = Path("C:/")
    train_v5.LEDGER_PATH = ROOT / "ledger.csv"
    plan_freeze.require = lambda *a, **k: "rehearsal"
    gate.V6_PRED_DIR, gate.OUT_DIR = PREDS, SCREENS
    sr.V6_TTA_DIR, sr.RUN_DIR = TTA, RUNS
    st.RUN_DIRS = (RUNS, Path("results/v5/runs"))
    st.REPO = Path("C:/")
    ADDED = "results/v6/added_pool/added_train.csv"
    done, failed = [], []

    def train(*a):
        t = time.time(); args = [*a, "--fold", "0", "--device", "cuda", "--num-workers", "2"]
        code = train_v6.main(list(args)); torch.cuda.empty_cache()
        say(f"train {' '.join(a)} -> exit {code} in {time.time()-t:.0f}s")
        assert code == 0

    def step(name, fn):
        t = time.time()
        try:
            fn(); done.append(name); say(f"PASS {name} ({time.time()-t:.0f}s)")
        except BaseException as e:
            failed.append(name); say(f"FAIL {name}: {type(e).__name__}: {str(e)[:300]}"); traceback.print_exc()

    def recheck_run(label_tag):
        if not (RUNS / label_tag).exists():
            say("NO RUN JSON", label_tag); failed.append(label_tag); return
        j = json.loads((RUNS / label_tag).read_text(encoding="utf-8"))
        say(label_tag, {k: j[k] for k in ("head", "epochs", "epochs_run", "optimizer_steps", "step_budget", "head_steps", "warmup_steps", "sampler", "train_images", "train_class_counts", "young_share_trained_on")}, "stages", "".join(h["stage"][0] for h in j["history"]), "vram", round(j.get("vram_peak_gb", 0), 2))

    # ---------- path A: T adopted, pool carried
    for s in (42, 43, 44):
        step(f"A train anchor s{s} v61t", lambda s=s: train("--member", "anchor", "--seed", str(s), "--run-tag", "v61t"))
    recheck_run("anchor_f0_s42_in22k_v61t.json")
    step("A stage v61", lambda: st.main(["v61", "--arm-tag", "v61t"]))
    for s in (42, 43, 44):
        step(f"A train P s{s}", lambda s=s: train("--member", "anchor", "--seed", str(s), "--run-tag", "v62p", "--added-train", ADDED))
        step(f"A train Y s{s}", lambda s=s: train("--member", "anchor", "--seed", str(s), "--run-tag", "v62y", "--added-train", ADDED, "--sampler", "young"))
    recheck_run("anchor_f0_s42_in22k_v62p.json"); recheck_run("anchor_f0_s42_in22k_v62y.json")
    step("A stage v62", lambda: st.main(["v62", "--p-tag", "v62p", "--y-tag", "v62y", "--comparator", "anchor", "--comparator-tag", "v61t"]))
    for m in ("panderm_b", "dinov2_b", "siglip2_b"):
        step(f"A train {m}", lambda m=m: train("--member", m, "--seed", "42", "--run-tag", "v63", "--added-train", ADDED))
        recheck_run(f"{m}_f0_s42_v63.json")
    step("A train anchor384", lambda: train("--member", "anchor", "--image-size", "384", "--batch-size", "16", "--grad-accum", "2", "--seed", "42", "--run-tag", "v63", "--added-train", ADDED))
    recheck_run("anchor384_f0_s42_in22k_v63.json")
    cands = ["panderm_b=v63", "dinov2_b=v63", "siglip2_b=v63", "anchor384=v63"]
    step("A stage v63", lambda: st.main(["v63", "--anchor-tag", "v62p", "--candidates", *cands]))
    if (SCREENS / "decision_v63.json").exists():
        d = json.loads((SCREENS / "decision_v63.json").read_text()); say("v63 routes", d["routes"], "promoted", d["promoted"], "dropped", d["dropped_by_cut_order"], "cost", d["cost_ratio_to_anchor"])
    ids = ["anchor_f0_s42_in22k_v62p", "panderm_b_f0_s42_v63", "dinov2_b_f0_s42_v63", "anchor384_f0_s42_in22k_v63"]
    step("A TTA last", lambda: pt.main(["--runs", *ids, "--device", "cuda"]))
    step("A TTA swad", lambda: pt.main(["--runs", ids[1], "--checkpoint", "swad", "--device", "cuda"]))
    step("A go-no-go", lambda: st.main(["go-no-go", "--members", "anchor=v62p", "panderm_b=v63", "dinov2_b=v63", "anchor384=v63"]))
    if (SCREENS / "decision_go_no_go.json").exists():
        d = json.loads((SCREENS / "decision_go_no_go.json").read_text()); say("go-no-go", {k: d[k] for k in ("v6_beats_v5", "combined_beats_v5", "continue_to_v6_4")})

    # ---------- path B: T rejected, then V5 pool carried
    for s in (42, 43, 44):
        step(f"B train P h7 s{s}", lambda s=s: train("--member", "anchor", "--head", "7", "--seed", str(s), "--run-tag", "v62p", "--added-train", ADDED))
        step(f"B train Y h7 s{s}", lambda s=s: train("--member", "anchor", "--head", "7", "--seed", str(s), "--run-tag", "v62y", "--added-train", ADDED, "--sampler", "young"))
    step("B stage v62", lambda: st.main(["v62", "--arm", "anchor_h7", "--p-tag", "v62p", "--y-tag", "v62y", "--comparator", "youngdata", "--comparator-tag", "v5scr"]))
    step("B train anchor h7 v5 pool", lambda: train("--member", "anchor", "--head", "7", "--seed", "42", "--run-tag", "v63b"))
    step("B train panderm h7", lambda: train("--member", "panderm_b", "--head", "7", "--seed", "42", "--run-tag", "v63b"))
    step("B train 384 h7", lambda: train("--member", "anchor", "--image-size", "384", "--batch-size", "16", "--grad-accum", "2", "--head", "7", "--seed", "42", "--run-tag", "v63b"))
    step("B stage v63", lambda: st.main(["v63", "--anchor", "anchor_h7", "--anchor-tag", "v63b", "--candidates", "panderm_b_h7=v63b", "anchor384_h7=v63b"]))
    if (SCREENS / "decision_v63.json").exists():
        d = json.loads((SCREENS / "decision_v63.json").read_text()); say("v63 B promoted", d["promoted"], "claim3", d["claim3_testable"])
    step("B TTA", lambda: pt.main(["--runs", "anchor_h7_f0_s42_in22k_v63b", "panderm_b_h7_f0_s42_v63b", "anchor384_h7_f0_s42_in22k_v63b", "--device", "cuda"]))
    step("B go-no-go", lambda: st.main(["go-no-go", "--members", "anchor_h7=v63b", "panderm_b_h7=v63b", "anchor384_h7=v63b"]))
    # a mixed recipe must be refused by the real tool
    def refuses():
        try:
            st.main(["go-no-go", "--members", "anchor=v61t", "panderm_b=v63"])
        except SystemExit as e:
            assert "different recipes" in str(e) or "missing" in str(e); say("mixed recipe refused:", str(e)[:120]); return
        raise AssertionError("a mixed recipe was accepted")
    step("guard: mixed recipe refused", refuses)

    # ---------- system reads on the real full rows, through V5's per-seed files standing in as members
    V5T, V5P = Path("results/v5/preds_tta"), Path("results/v5/preds")
    real_path = gate.pred_path

    def stand_in(label, seed, tag, pred_dir=None, fold=0, checkpoint=""):
        s = {"anchor": 42, "anchor_h7": 42, "panderm_b": 43, "panderm_b_h7": 43, "dinov2_b": 44}[label]
        kind = "v5stack" if fold == 0 else "v5conf"
        return V5T / f"composite_f{fold}_s{s}_in22k_{kind}_tta.csv" if "tta" in checkpoint else V5P / f"composite_f{fold}_s{s}_in22k_{kind}.csv"

    sr.paired_bootstrap.__defaults__ = (40, sr.SEED)
    sr.claim3_read.__defaults__ = (40,)
    sr.nested_selection.__defaults__ = (2, sr.SEED, 40)
    real_check = sr.check_runs
    sr.check_runs = lambda tags, **k: {"recipe": "stand-in", "runs_checked": sum(len(v) for v in tags.values())}
    gate.pred_path = stand_in
    gate.lock_stage0.require = lambda: None
    sr.E_PRIMARY, sr.CLAIM3, sr.NESTED, sr.LEDGER = ROOT / "e_primary.json", ROOT / "claim3.json", ROOT / "nested.json", ROOT / "exp.csv"

    def reads(finalists, tags):
        return sr.main(["--finalists", *finalists, "--tag", "v64", "--fold0-tags", *tags])

    step("reads: full run", lambda: reads(["anchor", "panderm_b", "dinov2_b"], ["anchor=x", "panderm_b=x", "dinov2_b=x"]))
    if sr.E_PRIMARY.exists():
        r = json.loads(sr.E_PRIMARY.read_text()); say("e_primary fields", sorted(r), "deployed", r["deployed_system"], "dMF1", round(r["e_primary"]["read"]["macro_f1"]["delta"], 4), "combined vs V5", round(r["combined_vote"]["read"]["macro_f1"]["delta"], 4))
        say("claim3", json.loads(sr.CLAIM3.read_text()).get("contrast"), "nested replaces", json.loads(sr.NESTED.read_text())["replaces_uniform"], "ledger rows", len(pd.read_csv(sr.LEDGER)))
    def second_read_refused():
        try:
            reads(["anchor", "panderm_b"], ["anchor=x", "panderm_b=x"])
        except SystemExit as e:
            say("second read refused:", str(e)[:90]); return
        raise AssertionError("second read accepted")
    step("guard: the read runs once", second_read_refused)
    step("reads: rerun with reason, 7-class labels", lambda: sr.main(["--finalists", "anchor_h7", "panderm_b_h7", "--tag", "v64", "--fold0-tags", "anchor_h7=x", "panderm_b_h7=x", "--rerun-reason", "rehearsal"]))
    if sr.CLAIM3.exists():
        say("claim3 h7", json.loads(sr.CLAIM3.read_text()).get("added"))
    gate.pred_path, sr.check_runs = real_path, real_check
    say(f"REHEARSAL: {len(done)} passed, {len(failed)} failed: {failed}")
    (SP / "rehearsal_summary.txt").write_text("\n".join(LOG), encoding="utf-8")
