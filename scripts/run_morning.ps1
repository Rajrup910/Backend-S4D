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

# "Continue", not "Stop": under Stop, PS 5.1 wraps every stderr line a native command writes
# (tqdm progress, warnings) in a terminating NativeCommandError, killing the script on the first
# such line regardless of the process's real exit code. Every python call below already checks
# $LASTEXITCODE explicitly, so that -- not this preference -- is what enforces failure.
$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
$py   = "C:\Users\RAJ\Downloads\Capstone\.venv\Scripts\python.exe"
$log  = Join-Path $repo "results\v4\morning_run.log"

Set-Location $repo
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null

# One instance at a time. On 2026-09-16 the script was launched two or three times within ten
# minutes; the copies trained the same rung onto the same checkpoint names and fought over the log
# ("Add-Content : Stream was not readable"). A named mutex dies with its process, so a crashed run
# never leaves a stale lock behind.
$mutex = New-Object System.Threading.Mutex($false, "Global\capstone_v4_run_morning")
# An abandoned mutex (the previous holder crashed) counts as acquired.
try { $owned = $mutex.WaitOne(0) } catch [System.Threading.AbandonedMutexException] { $owned = $true }
if (-not $owned) {
    Write-Output "Another run_morning.ps1 is already running. Not starting a second copy."
    Write-Output "Watch it with:  Get-Content '$log' -Tail 5 -Wait"
    exit 4
}

# Shared-access append: tolerates a reader tailing the log, and a failed write never stops a run.
function Log($msg) {
    $line = "[{0:yyyy-MM-dd HH:mm:ss}] {1}" -f (Get-Date), $msg
    Write-Output $line
    for ($try = 0; $try -lt 5; $try++) {
        try {
            $fs = [System.IO.File]::Open($log, 'Append', 'Write', 'ReadWrite')
            try {
                $bytes = [System.Text.Encoding]::UTF8.GetBytes($line + "`r`n")
                $fs.Write($bytes, 0, $bytes.Length)
            } finally { $fs.Dispose() }
            return
        } catch { Start-Sleep -Milliseconds 50 }
    }
}

Log "=== morning run starting ==="

# --- 0. re-run R1 if the overnight attempt died ---------------------------------------------
# R1 failed at epoch 3 with Windows error 1455 (ERROR_COMMITMENT_LIMIT) in a dataloader worker.
# Root cause: the disk filled, so the auto-managed pagefile could not grow, so the system commit
# limit was pinned and shared-memory allocation failed. At 384 px a batch of 32 needs 54 MB of
# shared mappings against 18 MB at 224 px, which is why R1 died where the 224 px arms did not.
#
# The settings below are a hardware accommodation, not a recipe change: batch 16 with grad-accum
# 2 keeps the EFFECTIVE batch at 32, identical to every other arm, so R1 remains exactly one
# declared change (resolution) away from the control. One worker halves the mappings again, to
# 54 MB in flight -- below the 74 MB that the 224 px arms are demonstrably surviving on.
$r1Json = Join-Path $repo "results\v4\recipe_runs\R1_ham_only_s42.json"
if (Test-Path $r1Json) {
    Log "R1 already complete; not re-running"
} else {
    Log "--- R1 re-run (384 px, batch 16 x accum 2 = effective 32, 1 worker, ~50 min) ---"
    $started = Get-Date
    & $py -m research.v4.train_v4 --rungs R1 --corpus ham_only `
        --batch-size 16 --grad-accum 2 --num-workers 1 2>&1 | ForEach-Object { Log $_ }
    if ($LASTEXITCODE -ne 0) {
        Log "R1 re-run FAILED with exit $LASTEXITCODE."
        Log "The screen cannot rank a partial field, so Block 2 will not run. Stopping here."
        exit $LASTEXITCODE
    }
    Log ("R1 finished in {0:N1} min" -f ((Get-Date) - $started).TotalMinutes)
}

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

# Project the finish against the user's 13:30 deadline and say so plainly in the log, so the
# 13:20 readout can report a known number instead of guessing. Measured 224 px rates on the HAM
# split were 31-40 s/epoch; the pooled corpus is 2.19x larger and 384 px costs 2.26x more.
# The pooled control is always R0, i.e. 224 px; only the composite can be 384 px. The first
# version priced both arms at the composite's resolution and overstated the finish by ~2 h.
$at384   = $composite -contains "R1"
$ctrlSec = 31 * (15294/6981)
$compSec = if ($at384) { 106 * (15294/6981) } else { $ctrlSec }
$compEp  = if ($composite -contains "R6") { 60 } else { 30 }
$projMin = ($ctrlSec * 30 + $compSec * $compEp) / 60
$projEnd = (Get-Date).AddMinutes($projMin)
$deadline = (Get-Date).Date.AddHours(13).AddMinutes(30)
Log ("Block 2 projection: {0} px, {1} composite epochs, ~{2:N0} min at full length -> {3:HH:mm}" -f `
     $(if ($at384) {384} else {224}), $compEp, $projMin, $projEnd)
if ($projEnd -gt $deadline) {
    Log ("*** PROJECTED FINISH {0:HH:mm} IS PAST THE 13:30 DEADLINE by {1:N0} min at full length." -f $projEnd, ($projEnd - $deadline).TotalMinutes)
    Log "*** Running anyway: early stopping has shortened every arm so far (R0 23/30, R2 15/30),"
    Log "*** so the real finish is likely earlier. The 13:20 readout will report actual progress."
} else {
    Log ("Block 2 projected to finish by {0:HH:mm}, inside the 13:30 deadline." -f $projEnd)
}

foreach ($arm in @(@{ Name = "pooled control"; Rungs = @("R0") },
                   @{ Name = "composite";      Rungs = $composite })) {
    # Skip an arm that already banked a result, so a relaunch after a failure resumes rather than
    # retraining the control it already has. train_v4 names runs '+'.join(rungs)_corpus_sSEED.
    $runId = "{0}_pooled_s42" -f ($arm.Rungs -join "+")
    if (Test-Path (Join-Path $repo "results\v4\recipe_runs\$runId.json")) {
        Log ("{0} ({1}) already complete; skipping" -f $arm.Name, $runId)
        continue
    }

    # 384 px arms get the same memory accommodation R1 needed in Block 1: batch 16 x grad-accum 2
    # (effective batch 32, unchanged) on one worker. At batch 32 / 2 workers a 384 px arm needs
    # 216 MB of shared mappings -- the configuration that died with error 1455 overnight. It also
    # keeps the composite's R1 component under the same conditions R1 was screened under.
    $memArgs = if ($arm.Rungs -contains "R1") {
        @("--batch-size", "16", "--grad-accum", "2", "--num-workers", "1")
    } else {
        @("--batch-size", "32", "--num-workers", "2")
    }

    Log ("--- {0}: --rungs {1}  {2} ---" -f $arm.Name, ($arm.Rungs -join " "), ($memArgs -join " "))
    $started = Get-Date
    & $py -m research.v4.train_v4 --rungs $arm.Rungs --corpus pooled @memArgs 2>&1 |
        ForEach-Object { Log $_ }
    if ($LASTEXITCODE -ne 0) {
        Log ("{0} FAILED with exit {1}. Stopping -- the second arm is not comparable without the first." -f $arm.Name, $LASTEXITCODE)
        exit $LASTEXITCODE
    }
    Log ("{0} finished in {1:N1} min" -f $arm.Name, ((Get-Date) - $started).TotalMinutes)
}

Log "=== complete. Block 3 (seeds 43, 44) was NOT started by design. ==="
exit 0
