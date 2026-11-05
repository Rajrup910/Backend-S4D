# V6-4 chain: anchor and the three promoted members on folds 1-4, seed 42, then 24-view TTA on averaged weights (docs/V6_RUNSHEET.md 0.4). 16 runs.
# No judgement happens here, and the V6-5 read is never started from here: its read happens once, on the owner's word.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_v6_4.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_v6_4.ps1 -Folds 1,2     # one night: these folds, then stops. TTA runs only once all 16 runs exist
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_v6_4.ps1 -DryRun
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_v6_4.ps1 -BranchTest -RunsFolder <folder>   # only prints what it would do
#
# Re-running the same command resumes: finished runs and finished TTA files are skipped. To stop between runs, create results\v5\PAUSE; delete it before resuming.
# It stops, keeping every finished run, if:
#   * scripts\preflight_overnight.ps1 fails, or the plan freeze does not hold, or a weight or input file is missing, or the promoted list differs from decision_v63.json;
#   * after a fold, a run is missing or incomplete, or its last-epoch Macro-F1 is below 0.60. That is a sanity check, not a scientific gate: it means a broken run. The value is not logged.
#   * TTA fails.

param([switch]$DryRun, [switch]$BranchTest, [string]$RunsFolder = "", [string]$Folds = "1,2,3,4")

$ErrorActionPreference = "Continue"
$repo  = Split-Path -Parent $PSScriptRoot
$py    = "C:\Users\RAJ\Downloads\Capstone\.venv\Scripts\python.exe"
$logs  = Join-Path $repo "results\v5\logs"
$log   = Join-Path $logs "overnight_v64.log"
$queue = Join-Path $repo "scripts\run_v6_queue.ps1"
$pause = Join-Path $repo "results\v5\PAUSE"
$runs  = Join-Path $repo "results\v6\runs"
$decision = Join-Path $repo "results\v6\screens\decision_v63.json"
$PROMOTED = @("panderm_b_h7", "dinov2_b_h7", "anchor384_h7")
$LABELS = @("anchor_h7") + $PROMOTED
$ALL_FOLDS = 1, 2, 3, 4
$THIS_RUN = @($Folds -split "," | ForEach-Object { [int]$_.Trim() })
if (@($THIS_RUN | Where-Object { $ALL_FOLDS -notcontains $_ }).Count) { Write-Host "-Folds takes values from 1,2,3,4"; exit 2 }
$TAG = "v64"
$MIN_SANE_MACRO_F1 = 0.60
Set-Location $repo
$env:HF_HUB_OFFLINE = "1"

function Log($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Host $line
    if (-not $BranchTest) { Add-Content -Path $log -Value $line -Encoding utf8 }
}
function Run-Id([string]$label, [int]$fold) {
    if ($label.StartsWith("anchor")) { return "{0}_f{1}_s42_in22k_{2}" -f $label, $fold, $TAG }
    return "{0}_f{1}_s42_{2}" -f $label, $fold, $TAG
}
# Returns what is wrong with a fold's runs; empty means every run is complete and sane.
function Fold-Problems([string]$folder, [int]$fold) {
    $problems = @()
    foreach ($label in $LABELS) {
        $id = Run-Id $label $fold
        $path = Join-Path $folder "$id.json"
        if (-not (Test-Path $path)) { $problems += "$id is missing"; continue }
        $j = Get-Content $path -Raw | ConvertFrom-Json
        if ($j.smoke -or [int]$j.epochs_run -ne [int]$j.epochs -or [int]$j.optimizer_steps -ne [int]$j.step_budget) { $problems += "$id is incomplete" }
        elseif ([double]$j.final_val_macro_f1 -lt $MIN_SANE_MACRO_F1) { $problems += "$id is below the sanity floor" }
    }
    return $problems
}
function Run-Python([string]$name, [string[]]$arguments) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $out = Join-Path $logs ("{0}.{1}.out.log" -f $name, $stamp)
    $err = Join-Path $logs ("{0}.{1}.err.log" -f $name, $stamp)
    $proc = Start-Process -FilePath $py -ArgumentList ($arguments -join " ") -WorkingDirectory $repo -RedirectStandardOutput $out -RedirectStandardError $err -NoNewWindow -PassThru
    $null = $proc.Handle  # PS 5.1: cache the handle or ExitCode stays null
    $proc.WaitForExit()
    return $proc.ExitCode
}
$ids = @(foreach ($fold in $ALL_FOLDS) { foreach ($label in $LABELS) { Run-Id $label $fold } })
$tta = @("-m", "research.v6.predict_tta", "--runs") + $ids + @("--checkpoint", "swad", "--num-workers", "2", "--device", "cuda")

if ($BranchTest) {
    foreach ($fold in $ALL_FOLDS) { Write-Host ("fold {0}: {1}" -f $fold, ((Fold-Problems $RunsFolder $fold) -join "; ")) }
    Write-Host ("TTA: python {0}" -f ($tta -join " "))
    exit 0
}
if ($DryRun) {
    foreach ($fold in $ALL_FOLDS) { & powershell -NoProfile -ExecutionPolicy Bypass -File $queue -Queue ("V64f{0}" -f $fold) -DryRun }
    Write-Host ("TTA: python {0}" -f ($tta -join " "))
    exit 0
}

# ----- before anything starts
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $repo "scripts\preflight_overnight.ps1") -StopHogs
if ($LASTEXITCODE -ne 0) { Log "Preflight failed; not starting. See the newest results\v5\logs\preflight_*.log."; exit 1 }
$code = Run-Python "v64_freeze" @("-m", "research.v6.plan_freeze", "--verify")
if ($code -ne 0) { Log "The plan freeze does not hold; not starting."; exit 1 }
$code = Run-Python "v64_inputs" @("scripts\check_v63_inputs.py")
if ($code -ne 0) { Log "A file or weight the runs need is missing; not starting. See the v64_inputs log."; exit 1 }
$decided = @((Get-Content $decision -Raw | ConvertFrom-Json).promoted)
if (($decided -join ",") -ne ($PROMOTED -join ",")) { Log ("decision_v63.json promotes {0}, this chain runs {1}; not starting." -f ($decided -join ", "), ($PROMOTED -join ", ")); exit 1 }

Log ("=== V6-4 start: folds {0} ===" -f ($THIS_RUN -join ", "))
foreach ($fold in $THIS_RUN) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File $queue -Queue ("V64f{0}" -f $fold)
    if (Test-Path $pause) { Log "PAUSE file present; stopping. Delete it and run the same command to resume."; exit 0 }
    $problems = @(Fold-Problems $runs $fold)
    if ($problems.Count) { Log ("STOP after fold {0}: {1}. Finished runs are safe." -f $fold, ($problems -join "; ")); exit 1 }
    Log ("Fold {0} finished: 4 runs complete." -f $fold)
}
$left = @($ALL_FOLDS | Where-Object { @(Fold-Problems $runs $_).Count })
if ($left.Count) { Log ("Folds {0} finished. Still to run: {1}. TTA waits for all 16 runs; this chain stops here." -f ($THIS_RUN -join ", "), ($left -join ", ")); exit 0 }
Log "All 16 runs complete. Next: 24-view TTA on averaged weights."
$code = Run-Python "v64_tta" $tta
if ($code -ne 0) { Log "STOP: TTA failed (exit $code). Finished files are kept and skipped on a re-run; see the v64_tta log."; exit 1 }
Log "=== V6-4 finished. The V6-5 read is not started from here; it happens once, on the owner's word. ==="
