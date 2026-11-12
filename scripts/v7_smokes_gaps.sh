#!/usr/bin/env bash
# V7 GPU smokes for the builds made ready on 10 Nov (B1's two members, B2's student, B5 on the V7 basis), on the full V7 pool.
# Two fine-tune batches each, disposable outputs (results/v7/smoke), behind the data gate; one GPU job at a time.
# B2 needs results/v7/concept_teacher/fold0_targets.csv (python -m research.v7.concept_student teacher --fold 0).
cd "$(dirname "$0")/.."
PY="/c/Users/RAJ/Downloads/Capstone/.venv/Scripts/python.exe"
LOG=results/v7/logs/smokes; mkdir -p "$LOG"
ADDED=results/v7/added_pool_v7/added_train_v7.csv
run() {  # name, args...
  name=$1; shift
  start=$(date +%s)
  PYTHONIOENCODING=utf-8 "$PY" -m research.v7.train_v7 "$@" --smoke --num-workers 2 --added-train "$ADDED" --sampler young > "$LOG/$name.log" 2>&1
  code=$?
  echo "$name exit=$code seconds=$(( $(date +%s) - start ))" | tee -a "$LOG/summary_gaps.txt"
}
: > "$LOG/summary_gaps.txt"
run B1_dinov2_392 --build B1 --member dinov2_b --fold 0 --image-size 392 --batch-size 16 --grad-accum 2
run B1_panderm_384 --build B1 --member panderm_b --fold 0 --image-size 384 --batch-size 16 --grad-accum 2
run B2 --build B2 --member anchor --fold 0
run B5_v7basis --build B5 --member anchor --fold 0
echo "ALL DONE" >> "$LOG/summary_gaps.txt"
