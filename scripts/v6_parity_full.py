"""Full 30-epoch schedule parity, CPU, deterministic: train_v5 --arm youngdata --trunk in22k vs train_v6 --member anchor --head 7.

Both run the REAL `run()` on the same tiny row subset, including the frozen stage and the switch to fine-tune. Every output path is
redirected to this scratch folder and the freeze check is stubbed, so nothing in the repo is written.
"""
import json, sys
from pathlib import Path
import numpy as np

SCRATCH = Path(sys.argv[1]); which = sys.argv[2]
OUT = SCRATCH / f"parity_{which}"; OUT.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    from research.v5 import train_v5
    real_frames, real_extra = train_v5.fold_frames, train_v5.load_extra_train

    def tiny_frames(manifest, fold):
        train, val = real_frames(manifest, fold)
        return train.head(48).reset_index(drop=True), val.head(32).reset_index(drop=True)

    def tiny_extra(path, train, val):
        return real_extra(path, train, val).head(16).reset_index(drop=True)

    train_v5.fold_frames, train_v5.load_extra_train = tiny_frames, tiny_extra
    train_v5.RUN_DIR, train_v5.PRED_DIR, train_v5.CHECKPOINT_DIR = OUT, OUT, OUT
    train_v5.LEDGER_PATH = OUT / "ledger.csv"
    train_v5.REPO_ROOT = Path("C:/")  # relative_to() in the summary; paths stay absolute under the scratch drive
    if which == "v5":
        code = train_v5.main(["--arm", "youngdata", "--fold", "0", "--seed", "42", "--trunk", "in22k", "--extra-train", "results/v5/young_data/extra_train.csv",
                              "--deterministic", "--log-steps", "--device", "cpu", "--num-workers", "0", "--run-tag", "par"])
    else:
        from research.v6 import plan_freeze, train_v6
        plan_freeze.require = lambda *a, **k: "stub"
        train_v6.fold_frames, train_v6.load_extra_train = tiny_frames, tiny_extra
        train_v6.RUN_DIR, train_v6.PRED_DIR, train_v6.CHECKPOINT_DIR = OUT, OUT, OUT
        train_v6.REPO_ROOT = Path("C:/")
        code = train_v6.main(["--member", "anchor", "--head", "7", "--fold", "0", "--seed", "42", "--deterministic", "--log-steps", "--device", "cpu", "--num-workers", "0", "--run-tag", "par"])
    print("exit", code)
