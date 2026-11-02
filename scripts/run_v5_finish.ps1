# Unattended V5 finish: waits for the running Q9 queue, then oof_table -> refit_dirichlet -> refit_s56 ->
# S84 --power -> S84 --dry-run -> S84 --execute (the single MILK10k read). Written 2 Nov 2026 for an overnight run.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_v5_finish.ps1 -RunnerPid <pid of q9_runner>
#
# Deterministic gates (no step starts unless every earlier one succeeded; nothing is retried):
#   * Q9: the runner process has exited AND `q9_runner --dry-run` lists 0 `todo` jobs (a failed job stays `todo`).
#   * Every step must exit 0.
#   * S84 starts only if its --dry-run shows no MISSING / NOT READY line and n_executions = 0, and C: has >= 5 GB free.
#   * S84 itself refuses a second execution (receipt), so this script can never read MILK10k twice.
# Keeps Windows awake while it runs. Log: results/v5/logs/v5_finish.log

param([Parameter(Mandatory = $true)][int]$RunnerPid, [int]$MaxWaitHours = 5)

$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
$py   = "C:\Users\RAJ\Downloads\Capstone\.venv\Scripts\python.exe"
$log  = Join-Path $repo "results\v5\logs\v5_finish.log"
Set-Location $repo
function Log($m) { $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m; $line; Add-Content $log $line }
function Stop-Chain($m) { Log "STOPPED: $m"; [Win32.Power]::SetThreadExecutionState([uint32]2147483648) | Out-Null; exit 1 }

Add-Type -Name Power -Namespace Win32 -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[Win32.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: no sleep while running

function Run-Step($name, [string[]]$pyArgs) {
  Log "start: $name"
  & $py @pyArgs 2>&1 | ForEach-Object { $s = "$_"; Add-Content $log $s; Write-Host $s }
  if ($LASTEXITCODE -ne 0) { Stop-Chain "$name exited $LASTEXITCODE" }
  Log "done: $name"
}

# ---- 1. wait for Q9
Log "=== V5 finish chain; waiting for q9_runner (pid $RunnerPid) ==="
$deadline = (Get-Date).AddHours($MaxWaitHours)
while (Get-Process -Id $RunnerPid -ErrorAction SilentlyContinue) {
  if ((Get-Date) -gt $deadline) { Stop-Chain "Q9 still running after $MaxWaitHours h" }
  Start-Sleep -Seconds 120
}
Log "q9_runner has exited"
$todo = @(& $py -m research.v5.q9_runner --dry-run 2>&1 | Where-Object { "$_" -match '^todo' })
if ($todo.Count -gt 0) { Stop-Chain "Q9 left $($todo.Count) job(s) undone: $($todo -join ' | ')" }
Log "Q9 complete: 0 todo"

# ---- 2-6. CPU steps (GPU hidden)
$env:CUDA_VISIBLE_DEVICES = ""
Run-Step "oof_table" @("-m", "research.v5.oof_table")
Run-Step "refit_dirichlet" @("-m", "research.v5.refit_dirichlet")
Run-Step "refit_s56" @("-m", "research.v5.refit_s56")
Run-Step "s84 --power" @("-m", "research.v5.s84_milk10k", "--power")
Log "start: s84 --dry-run"
$dry = @(& $py -m research.v5.s84_milk10k --dry-run 2>&1 | ForEach-Object { "$_" })
$dry | ForEach-Object { Add-Content $log "    $_"; $_ }
if ($LASTEXITCODE -ne 0) { Stop-Chain "s84 --dry-run exited $LASTEXITCODE" }
$bad = @($dry | Where-Object { $_ -match 'MISSING|NOT READY' })
if ($bad.Count -gt 0) { Stop-Chain "s84 preflight not clean: $($bad -join ' | ')" }
if (-not ($dry | Where-Object { $_ -match 'n_executions = 0' })) { Stop-Chain "receipt does not show n_executions = 0" }
Remove-Item Env:CUDA_VISIBLE_DEVICES

# ---- 7. the single S84 read
# Commit headroom is the binding constraint on this machine (error 1455): the pagefile is system-managed and grows into
# free disk, so the floor here is 10 GB of disk (not the usual 5), and S84 runs with 1 data worker (each worker is a
# full spawned torch import, ~1.8 GB of commit).
$free = (Get-PSDrive C).Free / 1GB
$c = Get-Counter '\Memory\Committed Bytes','\Memory\Commit Limit'
Log ("before S84: C: free {0:N1} GB; commit {1:N1} / {2:N1} GB" -f $free, ($c.CounterSamples[0].CookedValue/1GB), ($c.CounterSamples[1].CookedValue/1GB))
if ($free -lt 10) { Stop-Chain ("C: has {0:N1} GB free (< 10; the pagefile needs room)" -f $free) }
Log "start: s84 --execute (single MILK10k read)"
& $py -m research.v5.s84_milk10k --execute --num-workers 1 2>&1 | ForEach-Object { $s = "$_"; Add-Content $log $s; Write-Host $s }
if ($LASTEXITCODE -ne 0) {
  # One automatic resume, only if no label was read: S84 itself refuses --resume once label_read_count > 0.
  $r = Get-Content (Join-Path $repo "results\v5\s84_receipt.json") -Raw | ConvertFrom-Json
  if ($r.n_executions -eq 1 -and $r.label_read_count -eq 0) {
    Log "s84 --execute failed before any label read; one resume with --num-workers 0"
    & $py -m research.v5.s84_milk10k --execute --resume --num-workers 0 2>&1 | ForEach-Object { $s = "$_"; Add-Content $log $s; Write-Host $s }
    if ($LASTEXITCODE -ne 0) { Stop-Chain "s84 resume exited $LASTEXITCODE (labels unread; resume again by hand)" }
  } else { Stop-Chain "s84 --execute exited $LASTEXITCODE (label_read_count=$($r.label_read_count); do not rerun, inspect)" }
}
Log "done: s84 --execute"

Log "=== V5 finish chain COMPLETE: results/v5/s84/s84_report.json ==="
[Win32.Power]::SetThreadExecutionState([uint32]2147483648) | Out-Null
