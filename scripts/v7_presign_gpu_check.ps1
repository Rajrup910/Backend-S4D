# V7, before Gate 0: prove on the GPU that tonight's six run forms start and that the R4 range check runs end to end (plan 14.1 F8, F13).
# Nothing it writes is a V7 output of record, so it cannot block signing: smokes go to results\v7\smoke\ (overwritten, as smokes always are),
# range-check rehearsals to results\v7\lr_range_rehearsal\ (unsigned; `lr_range.require` refuses them). It reads fold-0 training rows only.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\v7_presign_gpu_check.ps1            # about 20 min, UNMEASURED (see the plan's 14.8 rows 2-3)
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\v7_presign_gpu_check.ps1 -DryRun    # prints the commands only
#
# Stops at the first smoke that fails (that form would fail tonight too). A rehearsal FAIL is reported, not a stop: it is the answer we want before signing.

param([switch]$DryRun)

$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
$py   = "C:\Users\RAJ\Downloads\Capstone\.venv\Scripts\python.exe"
$pool = "results/v7/added_pool_v7/added_train_v7.csv"
$logDir = Join-Path $repo "results\v7\logs\presign"
$log  = Join-Path $logDir "summary.log"     # UTF-8; each step's own output goes to its own file (PowerShell 5.1 writes it as UTF-16)
Set-Location $repo
New-Item -ItemType Directory -Force $logDir | Out-Null

function Log($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Host $line
    if (-not $DryRun) { Add-Content -Path $log -Value $line -Encoding utf8 }
}

# The exact forms of tonight's six screens (AUDIT_REPORT.md section J), as smokes.
$smokes = @(
    @("R1",  "anchor",    @()),
    @("R2",  "dinov2_b",  @()),
    @("R3",  "panderm_b", @()),
    @("R3b", "panderm_b", @()),
    @("B3",  "anchor",    @("--image-size", "384", "--batch-size", "16", "--grad-accum", "2")),
    @("B5",  "anchor",    @())
)
$ranges = @(@("R2", "dinov2_b"), @("R3", "panderm_b"), @("R3b", "panderm_b"))

if (-not (Test-Path (Join-Path $repo $pool))) { Log "STOP: the added pool $pool is missing"; exit 1 }
Log "=== V7 pre-sign GPU check ==="
foreach ($s in $smokes) {
    $cmd = @("-m", "research.v7.train_v7", "--build", $s[0], "--member", $s[1], "--fold", "0", "--added-train", $pool, "--num-workers", "2", "--smoke") + $s[2]
    if ($DryRun) { Write-Host ("python " + ($cmd -join " ")); continue }
    $t = Get-Date
    & $py @cmd *> (Join-Path $logDir ("smoke_{0}.log" -f $s[0]))
    $code = $LASTEXITCODE
    Log ("smoke {0} {1}: exit {2} in {3:N1} min" -f $s[0], $s[1], $code, ((Get-Date) - $t).TotalMinutes)
    if ($code -ne 0) { Log "STOP: a smoke failed; tonight's chain would fail on the same form. See $log."; exit 1 }
}
foreach ($r in $ranges) {
    $cmd = @("-m", "research.v7.lr_range", "--build", $r[0], "--member", $r[1], "--added-train", $pool, "--num-workers", "2", "--rehearsal")
    if ($DryRun) { Write-Host ("python " + ($cmd -join " ")); continue }
    $t = Get-Date
    & $py @cmd *> (Join-Path $logDir ("range_{0}.log" -f $r[0]))
    Log ("range-check rehearsal {0} {1}: exit {2} (0 = PASS or PASS_AT_EDGE, 1 = FAIL) in {3:N1} min" -f $r[0], $r[1], $LASTEXITCODE, ((Get-Date) - $t).TotalMinutes)
}
if (-not $DryRun) {
    & $py -c "import json,glob; [print(r['build'], r['member'], r['verdict'], '-', r['reason'], '-', r['seconds'], 's') for r in (json.load(open(p)) for p in sorted(glob.glob('results/v7/lr_range_rehearsal/*.json')))]" *> (Join-Path $logDir "verdicts.log")
    Get-Content (Join-Path $logDir "verdicts.log") | ForEach-Object { Log $_ }
    Log "=== done: read the verdicts above, then decide on signing ==="
}
