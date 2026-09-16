# S52, morning of 2026-09-16, unattended.
#
#   1. read the Block 1 screen and report the deltas
#   2. run ml/training/train.py -- the ORIGINAL trainer -- on the identical split (~15 min)
#   3. decide, by a rule fixed before the run, whether the control gate's failure is a stale
#      reference or a divergence in research/v4/train_v4.py
#   4. run Block 2 only if it is the former
#
# Why step 2 exists. R0 scored 0.7773 where the published ConvNeXt-Tiny figure is 0.7482 and
# S44's retrain 0.7509, so the control gate fired. The data was eliminated as a cause before this
# script was written: the HAM images in the ISIC-2019 directory are byte-identical to
# data/ham10000/, and the V4 splits are the same 6,981 train / 1,532 val images with the same
# per-class counts. What remains is the trainer. Running the reference trainer on the same data
# separates "the band was set against a stale number" from "train_v4.py diverges".
#
# The decision rule, fixed here and not at 10am: the two trainers agree if they land within the
# MCID (0.02) of each other. Anything the screen cannot resolve is not a difference the screen can
# act on. Agreement -> the band is stale, Block 2 runs. Disagreement -> Block 2 does NOT run.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_morning.ps1

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$py   = "C:\Users\RAJ\Downloads\Capstone\.venv\Scripts\python.exe"
$log  = Join-Path $repo "results\v4\morning_run.log"

Set-Location $repo
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null

function Log($msg) {
    $line = "[{0:yyyy-MM-dd HH:mm:ss}] {1}" -f (Get-Date), $msg
    Write-Output $line
    Add-Content -Path $log -Value $line -Encoding utf8
}

Log "=== morning run starting ==="

# --- 1. the screen as it stands -------------------------------------------------------------
Log "--- Block 1 screen ---"
& $py -m research.v4.screen 2>&1 | ForEach-Object { Log $_ }

# --- 2. the reference trainer on the identical split ----------------------------------------
$refHistory = Join-Path $repo "ml\results\convnext_tiny-v4_control_ref_training_history.json"
if (Test-Path $refHistory) {
    Log "control reference already present; not retraining"
} else {
    Log "--- control reference: ml/training/train.py, convnext_tiny, split_v1 (~15 min) ---"
    $started = Get-Date
    & $py -m ml.training.train --arch convnext_tiny --checkpoint-tag v4_control_ref `
        --run-name v4_control_ref --num-workers 2 `
        --notes "S52 control reference: reference trainer on the identical split, to decide whether the failed control gate is a stale band or a divergence in train_v4.py" 2>&1 |
        ForEach-Object { Log $_ }
    if ($LASTEXITCODE -ne 0) {
        Log "control reference FAILED with exit $LASTEXITCODE. Block 2 NOT started."
        exit $LASTEXITCODE
    }
    Log ("control reference finished in {0:N1} min" -f ((Get-Date) - $started).TotalMinutes)
}

# --- 3. record it where the gate can read it ------------------------------------------------
# Deliberately a module call, not Python embedded in a here-string: PowerShell strips double
# quotes when passing one to a native executable, and the first version of this reached the
# interpreter as `print(fcontrol`.
& $py -m research.v4.screen --record-control-reference $refHistory 2>&1 | ForEach-Object { Log $_ }
if ($LASTEXITCODE -ne 0) {
    Log "could not record the control reference (exit $LASTEXITCODE). Block 2 NOT started."
    exit $LASTEXITCODE
}

# --- 4. re-screen, now that the gate has a reference to weigh -------------------------------
Log "--- re-screening with the control reference ---"
& $py -m research.v4.screen --emit-commands 2>&1 | ForEach-Object { Log $_ }
$gate = $LASTEXITCODE

if ($gate -ne 0) {
    Log "GATE STILL DOES NOT PASS (exit $gate). Block 2 NOT started -- this is the correct outcome"
    Log "when the two trainers disagree. See results/v4/screen_verdict.json."
    exit $gate
}

# --- 5. Block 2 -----------------------------------------------------------------------------
$verdict   = Get-Content (Join-Path $repo "results\v4\screen_verdict.json") -Raw | ConvertFrom-Json
$composite = @($verdict.composite)

# Defence in depth. screen.py already refuses an empty composite, but a zero-exit plus an empty
# rung list would otherwise reach the trainer as `--rungs` with no value, and the failure would
# look like a training bug rather than a screening one.
if ($composite.Count -eq 0) {
    Log "screen returned success with an EMPTY composite. Refusing to run Block 2 -- this is a"
    Log "screening inconsistency, not a training problem. See results/v4/screen_verdict.json."
    exit 3
}
Log ("gate cleared ({0}); composite = {1}" -f $verdict.control_gate, ($composite -join " "))

foreach ($arm in @(@{ Name = "pooled control"; Rungs = @("R0") },
                   @{ Name = "composite";      Rungs = $composite })) {
    Log ("--- {0}: --rungs {1} ---" -f $arm.Name, ($arm.Rungs -join " "))
    $started = Get-Date
    & $py -m research.v4.train_v4 --rungs $arm.Rungs --corpus pooled --batch-size 32 --num-workers 2 2>&1 |
        ForEach-Object { Log $_ }
    if ($LASTEXITCODE -ne 0) {
        Log ("{0} FAILED with exit {1}. Stopping -- the second arm is not comparable without the first." -f $arm.Name, $LASTEXITCODE)
        exit $LASTEXITCODE
    }
    Log ("{0} finished in {1:N1} min" -f $arm.Name, ((Get-Date) - $started).TotalMinutes)
}

Log "=== complete. Block 3 (seeds 43, 44) was NOT started by design. ==="
exit 0
