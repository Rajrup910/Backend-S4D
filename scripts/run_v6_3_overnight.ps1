# V6-3 overnight chain: the four member runs, the V6-3 decision by rule, 24-view TTA on the anchor and the promoted members, then the go/no-go check (docs/V6_RUNSHEET.md 0.4).
# No judgement happens here: research.v6.stage_rules decides, and this only stops when something is wrong. V6-4 is never started from here.
# Carried from the earlier decisions: head 7 (decision_v61.json), pool Y (decision_v62.json), averaged weights as the read-out (decision_v61.json).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_v6_3_overnight.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_v6_3_overnight.ps1 -DryRun
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_v6_3_overnight.ps1 -BranchTest <decision.json> -RunsFolder <folder>   # only prints what it would do
#
# It stops, keeping every finished run, if:
#   * scripts\preflight_overnight.ps1 fails (battery, sleep settings, pending restart, disk, memory, GPU, PAUSE file), or the plan freeze does not hold, or a weight or input file is missing;
#   * no member run finished, or a decision tool or TTA fails;
#   * a member failed the 2.5x cost limit on a night when the machine slept or lost mains. train_time_seconds includes that gap, so the limit could drop a member for the wrong reason.
# A member that fails or diverges is dropped, as the plan says (0.8g): the decision reads the runs that finished, and the log names the others.

param([switch]$DryRun, [string]$BranchTest = "", [string]$RunsFolder = "")

$ErrorActionPreference = "Continue"
$repo  = Split-Path -Parent $PSScriptRoot
$py    = "C:\Users\RAJ\Downloads\Capstone\.venv\Scripts\python.exe"
$logs  = Join-Path $repo "results\v5\logs"
$log   = Join-Path $logs "overnight_v63.log"
$queue = Join-Path $repo "scripts\run_v6_queue.ps1"
$pause = Join-Path $repo "results\v5\PAUSE"
$runs  = Join-Path $repo "results\v6\runs"
$decisionFile = Join-Path $repo "results\v6\screens\decision_v63.json"
$ANCHOR = "anchor_h7"; $ANCHOR_TAG = "v62y"; $TAG = "v63"
$CANDIDATES = @("panderm_b_h7", "siglip2_b_h7", "dinov2_b_h7", "anchor384_h7")
Set-Location $repo
# The weights are pinned and already downloaded, so a network failure during the night cannot stop a build.
$env:HF_HUB_OFFLINE = "1"

function Log($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Host $line
    if (-not $BranchTest) { Add-Content -Path $log -Value $line -Encoding utf8 }
}
function Run-Id([string]$label, [string]$tag) {
    if ($label.StartsWith("anchor")) { return "{0}_f0_s42_in22k_{1}" -f $label, $tag }
    return "{0}_f0_s42_{1}" -f $label, $tag
}
function Is-Complete([string]$folder, [string]$label) {
    $path = Join-Path $folder ("{0}.json" -f (Run-Id $label $TAG))
    if (-not (Test-Path $path)) { return $false }
    $j = Get-Content $path -Raw | ConvertFrom-Json
    return (-not $j.smoke) -and ([int]$j.epochs_run -eq [int]$j.epochs) -and ([int]$j.optimizer_steps -eq [int]$j.step_budget)
}
# Runs python with its output in its own log files and returns the exit code; a child's stderr cannot end the chain.
function Run-Python([string]$name, [string[]]$arguments) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $out = Join-Path $logs ("{0}.{1}.out.log" -f $name, $stamp)
    $err = Join-Path $logs ("{0}.{1}.err.log" -f $name, $stamp)
    $proc = Start-Process -FilePath $py -ArgumentList ($arguments -join " ") -WorkingDirectory $repo -RedirectStandardOutput $out -RedirectStandardError $err -NoNewWindow -PassThru
    $null = $proc.Handle  # PS 5.1: cache the handle or ExitCode stays null
    $proc.WaitForExit()
    return $proc.ExitCode
}
function Plan-Check([string]$path) {
    $d = Get-Content $path -Raw | ConvertFrom-Json
    $promoted = @($d.promoted | Where-Object { $_ })  # none promoted means the anchor alone, never an empty name
    $ids = @(Run-Id $ANCHOR $ANCHOR_TAG) + @($promoted | ForEach-Object { Run-Id $_ $TAG })
    $members = @("$ANCHOR=$ANCHOR_TAG") + @($promoted | ForEach-Object { "$_=$TAG" })
    return @{ promoted = $promoted; tta = @("-m", "research.v6.predict_tta", "--runs") + $ids + @("--checkpoint", "swad", "--num-workers", "2", "--device", "cuda")
              check = @("-m", "research.v6.stage_rules", "go-no-go", "--members") + $members + @("--checkpoint", "swad") }
}
function Over-Limit([string]$path) {
    $d = Get-Content $path -Raw | ConvertFrom-Json
    $over = @()
    foreach ($p in $d.cost_ratio_to_anchor.PSObject.Properties) {
        if ($null -ne $p.Value -and [double]$p.Value -gt [double]$d.cost_limit) { $over += ("{0} {1:N2}x" -f $p.Name, [double]$p.Value) }
    }
    return $over
}

if ($BranchTest) {
    $finished = @($CANDIDATES | Where-Object { Is-Complete $RunsFolder $_ })
    Write-Host ("finished: {0}" -f ($finished -join " "))
    $plan = Plan-Check $BranchTest
    Write-Host ("promoted: {0}" -f ($plan.promoted -join " "))
    Write-Host ("over cost limit: {0}" -f ((Over-Limit $BranchTest) -join ", "))
    Write-Host ("TTA: python {0}" -f ($plan.tta -join " "))
    Write-Host ("check: python {0}" -f ($plan.check -join " "))
    exit 0
}
if ($DryRun) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File $queue -Queue V63 -DryRun
    Write-Host ("decision: python -m research.v6.stage_rules v63 --anchor {0} --anchor-tag {1} --candidates {2} --checkpoint swad" -f $ANCHOR, $ANCHOR_TAG, (($CANDIDATES | ForEach-Object { "$_=$TAG" }) -join " "))
    exit 0
}

# ----- before anything starts
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $repo "scripts\preflight_overnight.ps1") -StopHogs
if ($LASTEXITCODE -ne 0) { Log "Preflight failed; not starting. See the newest results\v5\logs\preflight_*.log."; exit 1 }
$code = Run-Python "v63_freeze" @("-m", "research.v6.plan_freeze", "--verify")
if ($code -ne 0) { Log "The plan freeze does not hold; not starting."; exit 1 }
$code = Run-Python "v63_inputs" @("scripts\check_v63_inputs.py")
if ($code -ne 0) { Log "A file or weight the runs need is missing; not starting. See the v63_inputs log."; exit 1 }

$started = Get-Date
Log "=== V6-3 start: four member runs ==="
& powershell -NoProfile -ExecutionPolicy Bypass -File $queue -Queue V63
if (Test-Path $pause) { Log "PAUSE file present; stopping before the decision."; exit 0 }

# ----- the runs that finished go to the decision; the plan drops the others
$finished = @($CANDIDATES | Where-Object { Is-Complete $runs $_ })
$missing = @($CANDIDATES | Where-Object { $finished -notcontains $_ })
if ($missing.Count) { Log ("Not finished, so dropped by the plan (0.8g): {0}" -f ($missing -join ", ")) }
if (-not $finished.Count) { Log "STOP: no member run finished."; exit 1 }

$candidates = @($finished | ForEach-Object { "$_=$TAG" })
$code = Run-Python "v63_decision" (@("-m", "research.v6.stage_rules", "v63", "--anchor", $ANCHOR, "--anchor-tag", $ANCHOR_TAG, "--candidates") + $candidates + @("--checkpoint", "swad"))
if ($code -ne 0) { Log "STOP: the V6-3 decision tool failed (exit $code). The runs are safe; see the v63_decision log."; exit 1 }

$disturbed = @(Get-WinEvent -FilterHashtable @{ LogName = "System"; ProviderName = "Microsoft-Windows-Kernel-Power"; Id = 42, 105; StartTime = $started } -ErrorAction SilentlyContinue)
$over = @(Over-Limit $decisionFile)
if ($disturbed.Count -and $over.Count) {
    Log ("STOP: {0} over the cost limit, and the machine slept or changed power source {1} time(s) during the runs, so the limit may be wrong. The runs are safe; read the times first." -f ($over -join ", "), $disturbed.Count)
    exit 1
}
if ($disturbed.Count) { Log ("Note: {0} sleep or power-source event(s) during the runs. No member is over the cost limit, so the decision stands." -f $disturbed.Count) }

$plan = Plan-Check $decisionFile
Log ("V6-3 decided by rule. Promoted: {0}. Next: 24-view TTA." -f $(if ($plan.promoted.Count) { $plan.promoted -join ", " } else { "none, anchor only" }))
$code = Run-Python "v63_tta" $plan.tta
if ($code -ne 0) { Log "STOP: TTA failed (exit $code). Finished files are kept and skipped on a re-run; see the v63_tta log."; exit 1 }
$code = Run-Python "v63_check" $plan.check
if ($code -ne 0) { Log "STOP: the go/no-go check failed (exit $code); see the v63_check log."; exit 1 }
Log "=== V6-3 finished: see results\v6\screens\decision_v63.json and decision_go_no_go.json. V6-4 is not started from here. ==="
