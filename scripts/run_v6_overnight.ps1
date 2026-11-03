# V6 overnight chain: V6-1, its decision by rule, then the V6-2 queue that decision selects, then its decision (docs/V6_RUNSHEET.md 0.4).
# No judgement happens here: research.v6.stage_rules decides, and this only stops when something is wrong. It stops, without starting V6-2, if:
#   * a V6-1 run is missing or incomplete, or its last-epoch Macro-F1 is below 0.60. That is a sanity check, not a scientific gate: the six banked anchor
#     seeds are 0.6507-0.6777, so a value below it means a broken run, and spending the rest of the night on top of it would be wasted;
#   * the V6-1 decision tool fails, or the PAUSE file exists (results\v5\PAUSE).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_v6_overnight.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_v6_overnight.ps1 -DryRun
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_v6_overnight.ps1 -BranchTest <decision.json> <runs folder>   # only prints what it would do

param([switch]$DryRun, [string]$BranchTest = "", [string]$RunsFolder = "")

$ErrorActionPreference = "Continue"
$repo  = Split-Path -Parent $PSScriptRoot
$py    = "C:\Users\RAJ\Downloads\Capstone\.venv\Scripts\python.exe"
$log   = Join-Path $repo "results\v5\logs\overnight_v6.log"
$queue = Join-Path $repo "scripts\run_v6_queue.ps1"
$pause = Join-Path $repo "results\v5\PAUSE"
$MIN_SANE_MACRO_F1 = 0.60
$SEEDS = @(42, 43, 44)
Set-Location $repo

function Log($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Host $line
    if (-not $BranchTest) { Add-Content -Path $log -Value $line -Encoding utf8 }
}
function Run-Queue([string]$name, [switch]$dry) {
    $child = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $queue, "-Queue", $name)
    if ($dry) { $child += "-DryRun" }
    & powershell @child
}
# Empty when all three V6-1 runs are complete and sane; otherwise the reason.
function Check-V61([string]$folder) {
    foreach ($seed in $SEEDS) {
        $path = Join-Path $folder ("anchor_f0_s{0}_in22k_v61t.json" -f $seed)
        if (-not (Test-Path $path)) { return "missing $path" }
        $j = Get-Content $path -Raw | ConvertFrom-Json
        if ($j.smoke -or [int]$j.epochs_run -ne [int]$j.epochs -or [int]$j.optimizer_steps -ne [int]$j.step_budget) { return "incomplete run: $path" }
        if ([double]$j.final_val_macro_f1 -lt $MIN_SANE_MACRO_F1) { return ("seed {0} Macro-F1 {1:N4} is below the sanity floor {2}" -f $seed, [double]$j.final_val_macro_f1, $MIN_SANE_MACRO_F1) }
    }
    return ""
}
# The V6-2 queue and the decision command, chosen only by V6-1's written decision.
function Plan-V62([string]$decisionPath) {
    $d = Get-Content $decisionPath -Raw | ConvertFrom-Json
    if ($d.t_adopted) {
        return @{ queue = "V62T"; stage = @("-m", "research.v6.stage_rules", "v62", "--p-tag", "v62p", "--y-tag", "v62y", "--comparator", "anchor", "--comparator-tag", "v61t") }
    }
    return @{ queue = "V62H"; stage = @("-m", "research.v6.stage_rules", "v62", "--arm", "anchor_h7", "--p-tag", "v62p", "--y-tag", "v62y", "--comparator", "youngdata", "--comparator-tag", "v5scr") }
}

if ($BranchTest) {
    $problem = Check-V61 $RunsFolder
    if ($problem) { Write-Host "STOP: $problem"; exit 0 }
    $plan = Plan-V62 $BranchTest
    Write-Host ("CONTINUE: queue {0}; then python {1}" -f $plan.queue, ($plan.stage -join " "))
    exit 0
}
if ($DryRun) {
    foreach ($name in "V61", "V62T", "V62H") { Run-Queue $name -dry }
    exit 0
}

Log "=== overnight start: V6-1 ==="
Run-Queue "V61"
if (Test-Path $pause) { Log "PAUSE file present; stopping before V6-2."; exit 0 }
$problem = Check-V61 (Join-Path $repo "results\v6\runs")
if ($problem) { Log "STOP before V6-2: $problem"; exit 1 }
& $py -m research.v6.stage_rules v61 --arm-tag v61t
if ($LASTEXITCODE -ne 0) { Log "STOP before V6-2: the V6-1 decision tool failed (exit $LASTEXITCODE)."; exit 1 }
$plan = Plan-V62 (Join-Path $repo "results\v6\screens\decision_v61.json")
Log ("V6-1 decided by rule. Next: queue {0}." -f $plan.queue)
Run-Queue $plan.queue
if (Test-Path $pause) { Log "PAUSE file present; V6-2 not finished."; exit 0 }
& $py @($plan.stage)
if ($LASTEXITCODE -ne 0) { Log "The V6-2 decision did not complete (exit $LASTEXITCODE): check the queue log."; exit 1 }
Log "=== overnight finished: see results\v6\screens\decision_v61.json and decision_v62.json ==="
