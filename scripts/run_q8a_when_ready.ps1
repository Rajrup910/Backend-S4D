# Waits for the overnight chain (Q7 + Q7b) to finish cleanly, then starts Q8a (seed-43 confirmation) and scores the
# new control checkpoints. Start it ONCE, in its own PowerShell window, before you sleep:
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_q8a_when_ready.ps1
#
# DETERMINISTIC GATES (nothing here depends on a prompt or on Claude being awake):
#   * not before -StartAt (default 04:30 local, next occurrence) -- the buffer for your own work;
#   * all 8 Q7 / Q7b run JSONs complete: smoke=false, epochs_run=30, final Macro-F1 finite and > 0.30;
#   * no train_v4 / train_v5 / infer_last process alive, V5 queue mutex free;
#   * results\v5\PAUSE absent (an advisory check may create it; delete the file to release);
#   * C: >= 10 GB free (the queue's own C2 gate re-checks memory + disk before every run).
# If a gate is unmet it logs why and polls every 60 s until -Deadline (default 09:00), then exits WITHOUT starting.
# It also holds the laptop awake (SetThreadExecutionState) for as long as it runs. Log: results\v5\logs\q8a_launcher.log
# Dry run (check the gates now, start nothing):  ... -DryRun

param([string]$StartAt = "04:30", [string]$Deadline = "09:00", [switch]$DryRun)

$repo = Split-Path -Parent $PSScriptRoot
$py   = "C:\Users\RAJ\Downloads\Capstone\.venv\Scripts\python.exe"
$v5   = Join-Path $repo "results\v5"
$log  = Join-Path $v5 "logs\q8a_launcher.log"
function Log($m) { $l = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH.mm.ss"), $m; Write-Host $l; Add-Content $log $l }

Add-Type -Namespace Win -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
$null = [Win.Power]::SetThreadExecutionState([uint32]2147483649)   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

function Next-Time([string]$hhmm) {
  $t = [datetime]::ParseExact($hhmm, "HH:mm", $null)
  if ($t -lt (Get-Date)) { $t = $t.AddDays(1) }
  return $t
}
$tStart = Next-Time $StartAt
$tEnd = Next-Time $Deadline
if ($tEnd -le $tStart) { $tEnd = $tEnd.AddDays(1) }

$required = @(1..4 | ForEach-Object { "composite_f{0}_s42_in22k_v5conf" -f $_ }) +
            @(1..4 | ForEach-Object { "m4_f{0}_s42_in22k_v5conf" -f $_ })

function Gate-Problems {
  $p = New-Object System.Collections.Generic.List[string]
  foreach ($id in $required) {
    $j = Join-Path $v5 ("runs\{0}.json" -f $id)
    if (-not (Test-Path $j)) { $p.Add("$id missing"); continue }
    try {
      $d = Get-Content $j -Raw | ConvertFrom-Json
      $f = [double]$d.final_val_macro_f1
      if ($d.smoke -or [int]$d.epochs_run -ne 30) { $p.Add("$id incomplete (epochs_run=$($d.epochs_run), smoke=$($d.smoke))") }
      elseif ([double]::IsNaN($f) -or [double]::IsInfinity($f) -or $f -le 0.30) { $p.Add("$id final Macro-F1 $f not sane") }
    } catch { $p.Add("$id unreadable") }
  }
  $busy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
            Where-Object { $_.CommandLine -match 'research\.v[45]\.train_v[45]|research\.v5\.infer_last' })
  if ($busy.Count) { $p.Add("$($busy.Count) training/inference process(es) still alive") }
  $m = New-Object System.Threading.Mutex($false, "Global\capstone_v5_queue")
  $got = $false
  try { $got = $m.WaitOne(0) } catch [System.Threading.AbandonedMutexException] { $got = $true }
  if ($got) { $m.ReleaseMutex() } else { $p.Add("V5 queue mutex held") }
  if (Test-Path (Join-Path $v5 "PAUSE")) { $p.Add("results\v5\PAUSE present") }
  if ((Get-PSDrive C).Free / 1GB -lt 10) { $p.Add("C: below 10 GB free") }
  return $p
}

$sTxt = $tStart.ToString("ddd HH\:mm"); $dTxt = $tEnd.ToString("ddd HH\:mm"); $dry = if ($DryRun) { "  [DRY RUN]" } else { "" }
Log ("=== Q8a launcher: not before $sTxt, gives up $dTxt$dry ===")
if ($DryRun) {
  $p = Gate-Problems; Log ("gates now: " + $(if ($p.Count) { $p -join "; " } else { "ALL CLEAR" })); exit 0
}
$lastMsg = ""
while ($true) {
  $now = Get-Date
  if ($now -ge $tEnd) { Log "deadline reached with gates unmet; NOT starting Q8a. Start it by hand when ready."; exit 4 }
  if ($now -ge $tStart) {
    $p = Gate-Problems
    if ($p.Count -eq 0) { break }
    $msg = $p -join "; "
    if ($msg -ne $lastMsg) { Log "waiting: $msg"; $lastMsg = $msg }
  }
  Start-Sleep -Seconds 60
}
Log "all gates clear -> starting Q8a"
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "run_v5_queue.ps1") -Queue Q8a
Log "Q8a queue returned (exit $LASTEXITCODE); scoring the new control checkpoints"
$ckpts = @(1..4 | ForEach-Object { "ml\checkpoints\convnext_tiny-v4_R0_kfold_f{0}_s43_v5ctl_last.pt" -f $_ } |
           Where-Object { Test-Path (Join-Path $repo $_) })
if ($ckpts.Count) {
  Push-Location $repo
  & $py -m research.v5.infer_last --checkpoint @ckpts --device cuda
  Pop-Location
  Log "infer_last exit $LASTEXITCODE for $($ckpts.Count) checkpoint(s)"
} else { Log "no control _last checkpoints found to score" }
Log "=== Q8a launcher done ==="
[void][Win.Power]::SetThreadExecutionState([uint32]2147483648)
