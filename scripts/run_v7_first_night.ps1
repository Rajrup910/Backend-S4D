# V7 first night (plan 14.8 steps 0-4): Gate 0 check, verifier, preflight, six smokes, three R4 range checks, six fold-0 screens.
# Runs only AFTER the owner has signed Gate 0. No judgement happens here. Whether a screen may run is decided by the trainer itself
# (`train_v7.refuse_unready` -> `lr_range.require`): a FAILed range check makes that screen refuse in seconds, and the chain moves on.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_v7_first_night.ps1                      # the night (about 7.3 h, see times below)
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_v7_first_night.ps1 -DryRun              # prints every command, runs nothing
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_v7_first_night.ps1 -Simulate "smoke:B3=1"   # rehearses a branch with
#       scripted exit codes (keys gate, verifier, preflight, smoke:<build>, range:<build>, screen:<build>, screen1455:<build>); runs nothing
#
# Stops (nothing later starts): Gate 0 not signed or changed; verifier non-zero; preflight FAIL; any smoke non-zero; the PAUSE file
# (results\v7\PAUSE) before any step; C: under 5 GB before a screen. Moves on: a range check that fails (its screen then refuses); a screen
# that exits non-zero (logged); a screen that would end after -Deadline (skipped, not started). Error 1455 (commit limit): ONE retry of
# that screen with one worker, never a loop. A finished run is skipped, never re-run or overwritten.

param([switch]$DryRun, [string]$Simulate = "", [string]$Deadline = "09:00", [string]$SimStart = "22:00")

$ErrorActionPreference = "Continue"
$repo   = Split-Path -Parent $PSScriptRoot
$py     = "C:\Users\RAJ\Downloads\Capstone\.venv\Scripts\python.exe"
$pool   = "results/v7/added_pool_v7/added_train_v7.csv"
$logDir = Join-Path $repo "results\v7\logs\first_night"
$log    = Join-Path $logDir "chain.log"
$pause  = Join-Path $repo "results\v7\PAUSE"
$MIN_FREE_GB = 5
Set-Location $repo
$fake = @{}
foreach ($pair in ($Simulate -split "," | Where-Object { $_ })) { $k, $v = $pair -split "="; $fake[$k.Trim()] = [int]$v }
$rehearse = $DryRun -or $Simulate
if (-not $rehearse) { New-Item -ItemType Directory -Force $logDir | Out-Null }

# The deadline is the next occurrence of the clock time (tonight's run ends tomorrow morning).
$end = [datetime]::Today.Add([timespan]::Parse($Deadline))
# A rehearsal runs on a simulated clock that starts at -SimStart and advances by each screen's planned minutes.
$clock = if ($rehearse) { [datetime]::Today.Add([timespan]::Parse($SimStart)) } else { Get-Date }
if ($end -le $clock) { $end = $end.AddDays(1) }
function Now { if ($rehearse) { return $script:clock } ; return (Get-Date) }

function Log($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Host $line
    if (-not $rehearse) { Add-Content -Path $log -Value $line -Encoding utf8 }
}
# One step: its own log file; returns the exit code. In a rehearsal, prints the command and returns the scripted code (default 0).
function Step([string]$key, [string]$name, [string[]]$cmd, [string]$exe = $py) {
    if ($rehearse) {
        $code = if ($fake.ContainsKey($key)) { $fake[$key] } else { 0 }
        Write-Host ("  [{0} -> exit {1}] {2} {3}" -f $key, $code, (Split-Path $exe -Leaf), ($cmd -join " "))
        return $code
    }
    $out = Join-Path $logDir "$name.log"
    & $exe @cmd *>> $out
    return $LASTEXITCODE
}
function Stop-Night([string]$why) { Log "STOP: $why"; exit 1 }
function Check-Pause { if ((-not $rehearse) -and (Test-Path $pause)) { Log "PAUSE file present; stopping cleanly."; exit 0 } }
function FreeGB { if ($rehearse) { return 99 } ; return [math]::Round((Get-PSDrive C).Free / 1GB, 1) }

# Fold-0 screens in the plan's order; minutes MEASURED per member on V6 folds 1-4 (B5 EXTRAPOLATED), plan 14.8 step 4.
$screens = @(
    @{ b = "R1";  m = "anchor";    x = @();                                                       min = 55.6; id = "R1_anchor_h7_f0_s42_v70" },
    @{ b = "B3";  m = "anchor";    x = @("--image-size", "384", "--batch-size", "16", "--grad-accum", "2"); min = 88.8; id = "B3_anchor384_h7_f0_s42_v70" },
    @{ b = "B5";  m = "anchor";    x = @();                                                       min = 66.0; id = "B5_anchor_h7_f0_s42_v70" },
    @{ b = "R2";  m = "dinov2_b";  x = @();                                                       min = 78.4; id = "R2_dinov2_b_h7_f0_s42_v70" },
    @{ b = "R3";  m = "panderm_b"; x = @();                                                       min = 64.5; id = "R3_panderm_b_h7_f0_s42_v70" },
    @{ b = "R3b"; m = "panderm_b"; x = @();                                                       min = 64.5; id = "R3b_panderm_b_h7_f0_s42_v70" }
)

Log ("=== V7 first night: deadline {0:yyyy-MM-dd HH:mm}; {1} ===" -f $end, $(if ($rehearse) { "REHEARSAL, nothing runs" } else { "live" }))
Check-Pause
# Step 0: signed and unchanged, and the verifier passes.
if ((Step "gate" "00_gate0" @("-c", "from research.v7 import plan_freeze; print(plan_freeze.require())")) -ne 0) { Stop-Night "V7 Gate 0 is not signed, or a frozen file changed since signing" }
if ((Step "verifier" "00_verifier" @("scripts/audit_v7_final_verification.py")) -ne 0) { Stop-Night "the V7 verifier did not exit 0" }
# Step 1: machine preflight (stops only its approved idle list).
if ((Step "preflight" "01_preflight" @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "scripts\preflight_overnight.ps1", "-StopHogs") "powershell") -ne 0) { Stop-Night "machine preflight failed; see 01_preflight.log" }
# Step 2: smokes of the exact forms. One failure stops the night: that form would fail too.
foreach ($s in $screens) {
    Check-Pause
    $code = Step "smoke:$($s.b)" "02_smoke_$($s.b)" (@("-m", "research.v7.train_v7", "--build", $s.b, "--member", $s.m, "--fold", "0", "--added-train", $pool, "--num-workers", "2", "--smoke") + $s.x)
    Log ("smoke {0}: exit {1}" -f $s.b, $code)
    if ($code -ne 0) { Stop-Night "smoke $($s.b) failed" }
}
# Step 3: R4 range checks. Not a stop: a FAIL (exit 1) leaves no passing receipt, and that screen refuses itself below.
foreach ($r in @(@("R2", "dinov2_b"), @("R3", "panderm_b"), @("R3b", "panderm_b"))) {
    Check-Pause
    $code = Step "range:$($r[0])" "03_range_$($r[0])" @("-m", "research.v7.lr_range", "--build", $r[0], "--member", $r[1], "--added-train", $pool, "--num-workers", "2")
    Log ("range check {0}: exit {1} ({2})" -f $r[0], $code, $(if ($code -eq 0) { "PASS or PASS_AT_EDGE" } else { "FAIL or refused: screen $($r[0]) will refuse" }))
}
# Step 4: the screens, one at a time.
$done = @()
foreach ($s in $screens) {
    Check-Pause
    if ((-not $rehearse) -and (Test-Path (Join-Path $repo "results\v7\runs\$($s.id).json"))) { Log "screen $($s.b): finished earlier; skipped"; continue }
    $finish = (Now).AddMinutes($s.min * 1.1)
    if ($finish -gt $end) { Log ("screen {0}: would end about {1:HH:mm}, after the deadline; not started" -f $s.b, $finish); continue }
    if ((FreeGB) -lt $MIN_FREE_GB) { Stop-Night ("C: has {0} GB free, under {1} GB" -f (FreeGB), $MIN_FREE_GB) }
    $cmd = @("-m", "research.v7.train_v7", "--build", $s.b, "--member", $s.m, "--fold", "0", "--added-train", $pool, "--run-tag", "v70") + $s.x
    $t = Now
    $code = Step "screen:$($s.b)" "04_screen_$($s.b)" ($cmd + @("--num-workers", "2"))
    $hit1455 = if ($rehearse) { $fake.ContainsKey("screen1455:$($s.b)") } else { (Test-Path (Join-Path $logDir "04_screen_$($s.b).log")) -and (Select-String -Path (Join-Path $logDir "04_screen_$($s.b).log") -Pattern "1455|paging file is too small" -Quiet) }
    if ($code -ne 0 -and $hit1455) {
        Log "screen $($s.b): error 1455 (commit limit); one retry with one worker"
        $code = Step "screen1455:$($s.b)" "04_screen_$($s.b)_retry_w1" ($cmd + @("--num-workers", "1"))
    }
    if ($rehearse) { $script:clock = $script:clock.AddMinutes($s.min) }
    Log ("screen {0}: exit {1} after {2:N0} min (planned {3})" -f $s.b, $code, ((Now) - $t).TotalMinutes, $s.min)
    if ($code -eq 0) { $done += $s.b }
}
Log ("=== night finished: screens completed {0} of 6 ({1}). Read fold 0 and apply each drop rule in the morning. ===" -f $done.Count, ($done -join ", "))
exit 0
