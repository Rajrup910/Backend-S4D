#!/usr/bin/env bash
# V7 GPU smokes: two fine-tune batches per declared build, disposable outputs (results/v7/smoke), behind the data gate.
# Runs one after another (never two GPU jobs at once); a failure is logged and the chain continues.
cd "$(dirname "$0")/.."
PY="/c/Users/RAJ/Downloads/Capstone/.venv/Scripts/python.exe"
LOG=results/v7/logs/smokes; mkdir -p "$LOG"
ADDED=results/v6/added_pool/added_train.csv
run() {  # name, args...
  name=$1; shift
  start=$(date +%s)
  PYTHONIOENCODING=utf-8 "$PY" -m research.v7.train_v7 "$@" --smoke --num-workers 2 --added-train "$ADDED" --sampler young > "$LOG/$name.log" 2>&1
  code=$?
  echo "$name exit=$code seconds=$(( $(date +%s) - start ))" | tee -a "$LOG/summary.txt"
}
: > "$LOG/summary.txt"
run B3 --build B3 --member anchor --fold 0 --image-size 384 --batch-size 16 --grad-accum 2
run B5 --build B5 --member anchor --fold 0
run R1 --build R1 --member anchor --fold 0
run R2 --build R2 --member dinov2_b --fold 0
run R3 --build R3 --member panderm_b --fold 0
run L0 --build L0 --member anchor --loao-holdout bcn20000
echo "ALL DONE" >> "$LOG/summary.txt"
