"""Live progress dashboard for a V5 queue (read-only: it only reads logs and result files).

    python scripts/v5_progress.py              # Q3, refreshes every 3 s, Ctrl+C to quit
    python scripts/v5_progress.py --queue Q4
    python scripts/v5_progress.py --once       # print one snapshot and exit

Shows: runs complete / total (a run is complete when results/v5/runs/<run_id>.json exists, the same
rule the queue runner uses), the current run's epoch and batch bars from its tqdm log, and an ETA.
The ETA uses only run times measured in this queue and is shown only once a run has finished
(a projection from the early epochs would include start-up and the cheaper frozen-trunk stage).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT / "results" / "v5"
LOGS = V5 / "logs"
RUNS = V5 / "runs"
WIDTH = 34
TQDM = re.compile(r"epoch (\d+)/(\d+) \[(\w+)\]:\s+(\d+)%\|.*?\|\s*(\d+)/(\d+) \[([^<\]]*)<([^,\]]*)")


def bar(done: float, total: float, width: int = WIDTH) -> str:
    frac = 0.0 if total <= 0 else max(0.0, min(1.0, done / total))
    full = int(frac * width)
    return "[" + "#" * full + "-" * (width - full) + f"] {frac * 100:5.1f}%"


def run_ids(queue: str) -> list[str]:
    """Run ids exactly as the queue names them: <arm>_f<fold>_s<seed>[_<trunk>][_<tag>]."""
    ids = []
    for line in (V5 / f"queue_{queue}.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "--arm" not in line:
            continue
        tok = shlex.split(line)
        arg = {tok[i]: tok[i + 1] for i in range(len(tok) - 1) if tok[i].startswith("--")}
        split = f"loao-{arg['--loao-holdout']}" if "--loao-holdout" in arg else f"f{arg.get('--fold', '0')}"
        rid = f"{arg['--arm']}_{split}_s{arg['--seed']}"  # matches train_v5 / run_v5_queue (LOAO fix 3 Oct)
        if arg.get("--trunk", "in1k") != "in1k":
            rid += f"_{arg['--trunk']}"
        if arg.get("--run-tag"):
            rid += f"_{arg['--run-tag']}"
        ids.append(rid)
    return ids


def run_minutes(rid: str) -> float | None:
    path = RUNS / f"{rid}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        seconds = data.get("train_time_seconds") or data.get("elapsed_seconds")
        return float(seconds) / 60 if seconds else None
    except (OSError, ValueError):
        return None


def tail_text(path: Path, size: int = 6000) -> str:
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        fh.seek(max(0, fh.tell() - size))
        return fh.read().decode("utf-8", "replace").replace("\r", "\n")


def current_run(ids: list[str]) -> dict | None:
    logs = sorted(LOGS.glob("*.err.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    for log in logs:
        rid = log.name.split(".")[0]
        if rid in ids and not (RUNS / f"{rid}.json").exists():
            age = time.time() - log.stat().st_mtime
            lines = [m for m in TQDM.finditer(tail_text(log))]
            info = {"id": rid, "stale_s": age, "started": log.stat().st_ctime}
            if lines:
                m = lines[-1]
                info.update(epoch=int(m.group(1)), epochs=int(m.group(2)), stage=m.group(3),
                            batch=int(m.group(5)), batches=int(m.group(6)),
                            elapsed=m.group(7).strip(), remaining=m.group(8).strip())
            return info
    return None


def last_epoch_line(rid: str) -> str:
    outs = sorted(LOGS.glob(f"{rid}.*.out.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not outs:
        return ""
    found = [l.strip() for l in tail_text(outs[0], 20000).splitlines() if l.strip().startswith("epoch ")]
    return found[-1] if found else ""


def snapshot(queue: str) -> str:
    ids = run_ids(queue)
    done = [r for r in ids if (RUNS / f"{r}.json").exists()]
    out = [f"V5 queue {queue}   {time.strftime('%H:%M:%S')}", "",
           f"runs      {bar(len(done), len(ids))}  {len(done)}/{len(ids)}"]
    cur = current_run(ids)
    measured = [m for m in (run_minutes(r) for r in done) if m]
    if cur:
        out.append(f"current   {cur['id']}")
        if "epoch" in cur:
            ep_done = cur["epoch"] - 1 + cur["batch"] / max(cur["batches"], 1)
            out.append(f"epochs    {bar(ep_done, cur['epochs'])}  {cur['epoch']}/{cur['epochs']} [{cur['stage']}]")
            out.append(f"batches   {bar(cur['batch'], cur['batches'])}  {cur['batch']}/{cur['batches']}"
                       f"  ({cur['elapsed']} elapsed, {cur['remaining']} left in epoch)")
            if not measured:
                out.append("ETA       shown after the first run finishes (measured, never guessed)")
        if cur["stale_s"] > 300:
            out.append(f"WARNING   current log not updated for {cur['stale_s'] / 60:.0f} min "
                       f"(queue stopped, paused or stuck?)")
        line = last_epoch_line(cur["id"])
        if line:
            out.append(f"last      {line[:110]}")
    else:
        out.append("current   (no run in progress)")
    if measured:
        mean = sum(measured) / len(measured)
        out.append(f"ETA       ~{mean * (len(ids) - len(done)) / 60:.1f} h left "
                   f"(measured mean {mean:.1f} min/run over {len(measured)} finished run(s))")
    if (V5 / "PAUSE").exists():
        out.append("NOTE      results/v5/PAUSE exists: the queue stops after the current run")
    young = ROOT / "data" / "external" / "isic_young_v5"
    if young.is_dir():
        out.append(f"young     {sum(1 for _ in young.glob('*.jpg'))} images kept so far")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--queue", default="Q3")
    parser.add_argument("--every", type=float, default=3.0, help="refresh seconds")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.once:
        print(snapshot(args.queue))
        return 0
    try:
        while True:
            text = snapshot(args.queue)
            os.system("cls" if os.name == "nt" else "clear")
            print(text + "\n\n(Ctrl+C to close this window; the queue keeps running)")
            time.sleep(args.every)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
