# S53 Blocks 2+3 completion -- make sure all six pooled runs exist: seeds 42, 43, 44 for the pooled
# control (R0) and the composite (from results/v4/screen_verdict.json). Unattended and self-healing.
#
# Every rule below exists because the same thing failed on 2026-09-16 while it lived in a prompt
# or a person's memory:
#
#   WAITING    Takes the same named lock as run_morning.ps1 and waits for it, then for three
#              consecutive minutes with no training process. While waiting it watches the other
#              job: a training process whose checkpoint has not advanced for STALL_MIN is treated
#              as hung and killed, so a deadlocked Block 2 cannot block Block 3 forever.
#   SEED 42    If Block 2's composite never banked (it crashed or was killed), this script trains
#              it like any other seed instead of giving up.
#   CONFIG     Batch and grad-accum per arm are fixed here and identical across seeds: any arm
#              containing R1 (384 px) runs batch 16 x accum 2 (effective 32); everything else
#              batch 32 x 1. They are cross-checked against a banked seed-42 result.
#   RETRY      Each run gets up to three attempts, stepping down the worker count 2 -> 1 -> 0.
#              Workers change only which RNG stream augments which image -- seed-level noise,
#              never the optimisation -- and every run records num_workers. Error 1455 was the
#              overnight crash; fewer workers is its direct remedy.
#   WATCHDOG   A run whose _last.pt has not advanced for STALL_MIN is killed and retried. This
#              catches Windows dataloader deadlocks and VRAM spilling into system RAM.
#   DISK       Refuses to launch below MIN_DISK_GB (train_v4 also enforces 5 GB / 2 GB per epoch).
#   VALIDATE   A run counts only if it exited 0 AND its run JSON exists with the right rungs and
#              seed and is not a smoke run.
#   CONTINUE   A permanently failed arm is recorded and the block moves on; complete pairs are
#              worth more than stopping at the first failure. Exit code is non-zero if any failed.
#   STATUS     results/v4/block3_status.json is rewritten every minute, readable from a phone.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_block3.ps1            # run
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_block3.ps1 -DryRun    # plan only
#
# Launch unattended runs via WMI so they live outside the Claude app's container (see memory).

param([switch]$DryRun)

$ErrorActionPreference = "Continue"
$repo    = Split-Path -Parent $PSScriptRoot
$py      = "C:\Users\RAJ\Downloads\Capstone\.venv\Scripts\python.exe"
$runsDir = Join-Path $repo "results\v4\recipe_runs"
$ckptDir = Join-Path $repo "ml\checkpoints"
$logDir  = Join-Path $repo "results\v4\block3_logs"
$log     = Join-Path $repo "results\v4\block3_run.log"
$status  = Join-Path $repo "results\v4\block3_status.json"
$SEEDS         = @(42, 43, 44)
$WORKER_LADDER = @(2, 1, 0)
$STALL_MIN     = 30      # an epoch at 384 px on the pooled corpus takes ~5 min on 1 worker
$MIN_DISK_GB   = 6
Set-Location $repo
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Log($msg) {
    $line = "[{0:yyyy-MM-dd HH:mm:ss}] {1}" -f (Get-Date), $msg
    Write-Output $line
    for ($t = 0; $t -lt 5; $t++) {
        try {
            $fs = [System.IO.File]::Open($log, 'Append', 'Write', 'ReadWrite')
            try { $b = [System.Text.Encoding]::UTF8.GetBytes($line + "`r`n"); $fs.Write($b, 0, $b.Length) }
            finally { $fs.Dispose() }
            return
        } catch { Start-Sleep -Milliseconds 50 }
    }
}

function Write-Status($state, $extra = @{}) {
    $o = [ordered]@{ updated = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss"); state = $state }
    foreach ($k in $extra.Keys) { $o[$k] = $extra[$k] }
    try { ($o | ConvertTo-Json -Depth 4) | Set-Content -Path $status -Encoding utf8 } catch { }
}

function Get-TrainingProcs {
    @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
      Where-Object { $_.CommandLine -match 'research\.v4\.train_v4|ml\.training\.train' })
}

function Stop-Tree([int]$rootId) {
    $all = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
    $ids = New-Object System.Collections.ArrayList
    $queue = New-Object System.Collections.Queue; $queue.Enqueue($rootId)
    while ($queue.Count) {
        $id = $queue.Dequeue(); [void]$ids.Add($id)
        foreach ($c in $all | Where-Object { $_.ParentProcessId -eq $id }) { $queue.Enqueue([int]$c.ProcessId) }
    }
    foreach ($id in ($ids | Sort-Object -Descending)) { Stop-Process -Id $id -Force -ErrorAction SilentlyContinue }
    return ($ids -join ", ")
}

function Get-NewestPooledCheckpointAge {
    $f = Get-ChildItem $ckptDir -Filter "convnext_tiny-v4_*_pooled_*_last.pt" -ErrorAction SilentlyContinue |
         Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $f) { return $null }
    return ((Get-Date) - $f.LastWriteTime).TotalMinutes
}

function Get-DiskFreeGB { [math]::Round((Get-PSDrive -Name (Split-Path $repo -Qualifier).TrimEnd(':')).Free / 1GB, 2) }

function Test-RunBanked($rungs, $seed) {
    $path = Join-Path $runsDir ("{0}_pooled_s{1}.json" -f ($rungs -join "+"), $seed)
    if (-not (Test-Path $path)) { return $false }
    try {
        $j = Get-Content $path -Raw | ConvertFrom-Json
        return (-not $j.smoke) -and ([int]$j.seed -eq $seed) -and ((@($j.rungs) -join "+") -eq ($rungs -join "+"))
    } catch { return $false }
}

# ------------------------------------------------------------------------------ plan
$verdict = Get-Content (Join-Path $repo "results\v4\screen_verdict.json") -Raw | ConvertFrom-Json
if (-not $verdict.proceed) { Log "screen_verdict.json says proceed=false. Nothing to run."; Write-Status "refused: screen not cleared"; exit 2 }
$composite = @($verdict.composite)
if ($composite.Count -eq 0) { Log "empty composite in screen_verdict.json. Nothing to run."; Write-Status "refused: empty composite"; exit 2 }

$arms = @()
foreach ($rungs in @(@("R0"), $composite)) {
    $cfg = if ($rungs -contains "R1") { @{ Batch = 16; Accum = 2 } } else { @{ Batch = 32; Accum = 1 } }
    $s42 = Join-Path $runsDir ("{0}_pooled_s42.json" -f ($rungs -join "+"))
    if (Test-Path $s42) {
        $j = Get-Content $s42 -Raw | ConvertFrom-Json
        $jAccum = if ($j.grad_accum) { [int]$j.grad_accum } else { 1 }
        if (([int]$j.batch_size -ne $cfg.Batch) -or ($jAccum -ne $cfg.Accum)) {
            Log ("WARNING: banked seed 42 for {0} used batch {1} x accum {2}; later seeds use {3} x {4}. Effective batch is 32 in both." -f `
                 ($rungs -join "+"), $j.batch_size, $jAccum, $cfg.Batch, $cfg.Accum)
        }
    }
    $arms += @{ Rungs = $rungs; Batch = $cfg.Batch; Accum = $cfg.Accum }
}

Log ("=== Block 3 runner started{0}. Arms: {1}. Seeds: {2}. ===" -f `
     $(if ($DryRun) { " (DRY RUN)" } else { "" }), (($arms | ForEach-Object { $_.Rungs -join "+" }) -join " | "), ($SEEDS -join ", "))
foreach ($seed in $SEEDS) { foreach ($arm in $arms) {
    Log ("  plan: {0}_pooled_s{1}  batch {2} x accum {3}  banked={4}" -f ($arm.Rungs -join "+"), $seed, $arm.Batch, $arm.Accum, (Test-RunBanked $arm.Rungs $seed))
} }

if ($DryRun) {
    $busy = Get-TrainingProcs
    Log ("  dry run: {0} training process(es) running; disk {1} GB free; newest pooled checkpoint {2} min old" -f `
         $busy.Count, (Get-DiskFreeGB), $(if ($null -ne ($a = Get-NewestPooledCheckpointAge)) { [math]::Round($a, 1) } else { "n/a" }))
    exit 0
}

# ------------------------------------------------------------------------------ wait
$mutex = New-Object System.Threading.Mutex($false, "Global\capstone_v4_run_morning")
Log "waiting for the GPU lock (run_morning.ps1 holds it while Block 2 runs)"
$owned = $false
while (-not $owned) {
    try { $owned = $mutex.WaitOne(60000) } catch [System.Threading.AbandonedMutexException] { $owned = $true }
    if ($owned) { break }
    # Watch the job we are waiting on: alive but not progressing means hung.
    $busy = Get-TrainingProcs
    $age = Get-NewestPooledCheckpointAge
    Write-Status "waiting for lock" @{ training_processes = $busy.Count; newest_checkpoint_age_min = $age }
    if ($busy.Count -gt 0 -and $null -ne $age -and $age -gt $STALL_MIN) {
        Log ("the job holding the lock looks HUNG: training alive, newest checkpoint {0:N0} min old (> {1}). Killing it." -f $age, $STALL_MIN)
        foreach ($p in @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.CommandLine -match 'run_morning' })) {
            Log ("  stopped run_morning tree: " + (Stop-Tree ([int]$p.ProcessId)))
        }
        foreach ($p in Get-TrainingProcs) { Log ("  stopped training tree: " + (Stop-Tree ([int]$p.ProcessId))) }
    }
}
Log "lock acquired"

$idle = 0
while ($idle -lt 3) {
    $busy = Get-TrainingProcs
    if ($busy.Count -gt 0) {
        $age = Get-NewestPooledCheckpointAge
        if ($null -ne $age -and $age -gt $STALL_MIN) {
            Log ("an unlocked training process looks HUNG (checkpoint {0:N0} min old). Killing it." -f $age)
            foreach ($p in $busy) { Log ("  stopped: " + (Stop-Tree ([int]$p.ProcessId))) }
        } else { Log "a training process is still running; waiting" }
        $idle = 0
    } else { $idle++ }
    Write-Status "waiting for idle GPU" @{ idle_minutes = $idle }
    if ($idle -lt 3) { Start-Sleep -Seconds 60 }
}
Log "GPU idle for 3 consecutive minutes; starting"

# ------------------------------------------------------------------------------ run
$failed = @()
$done = 0
foreach ($seed in $SEEDS) {
    foreach ($arm in $arms) {
        $runId = "{0}_pooled_s{1}" -f ($arm.Rungs -join "+"), $seed
        if (Test-RunBanked $arm.Rungs $seed) { Log "$runId already banked; skipping"; $done++; continue }
        $lastCkpt = Join-Path $ckptDir ("convnext_tiny-v4_{0}_last.pt" -f $runId)
        $ok = $false

        foreach ($workers in $WORKER_LADDER) {
            $disk = Get-DiskFreeGB
            if ($disk -lt $MIN_DISK_GB) {
                Log ("DISK LOW ({0} GB < {1} GB). Not launching {2}. Free space and relaunch this script." -f $disk, $MIN_DISK_GB, $runId)
                Write-Status "stopped: disk low" @{ run = $runId; disk_free_gb = $disk }
                exit 5
            }
            $stamp = Get-Date -Format "HHmmss"
            $out = Join-Path $logDir "$runId.w$workers.$stamp.out.log"
            $err = Join-Path $logDir "$runId.w$workers.$stamp.err.log"
            $argLine = "-m research.v4.train_v4 --rungs {0} --corpus pooled --seed {1} --batch-size {2} --grad-accum {3} --num-workers {4}" -f `
                       ($arm.Rungs -join " "), $seed, $arm.Batch, $arm.Accum, $workers
            Log ("--- {0}: attempt workers={1}  ({2}) ---" -f $runId, $workers, $argLine)
            $started = Get-Date
            $proc = Start-Process -FilePath $py -ArgumentList $argLine -WorkingDirectory $repo `
                        -RedirectStandardOutput $out -RedirectStandardError $err -NoNewWindow -PassThru
            $null = $proc.Handle     # PS 5.1 quirk: cache the handle or ExitCode stays null
            $stalled = $false

            while (-not $proc.HasExited) {
                Start-Sleep -Seconds 60
                if ($proc.HasExited) { break }
                $since = if (Test-Path $lastCkpt) {
                    $w = (Get-Item $lastCkpt).LastWriteTime
                    if ($w -gt $started) { ((Get-Date) - $w).TotalMinutes } else { ((Get-Date) - $started).TotalMinutes }
                } else { ((Get-Date) - $started).TotalMinutes }
                $epoch = $null
                $tail = Get-Content $out -Tail 40 -ErrorAction SilentlyContinue | Select-String "^epoch\s+(\d+)/(\d+)" | Select-Object -Last 1
                if ($tail) { $epoch = $tail.Matches[0].Groups[1].Value + "/" + $tail.Matches[0].Groups[2].Value }
                Write-Status "training" @{ run = $runId; seed = $seed; workers = $workers; epoch = $epoch;
                    minutes_since_progress = [math]::Round($since, 1); disk_free_gb = (Get-DiskFreeGB);
                    runs_banked = $done; runs_total = $SEEDS.Count * $arms.Count }
                if ($since -gt $STALL_MIN) {
                    Log ("{0} STALLED: no checkpoint progress for {1:N0} min. Killing and retrying." -f $runId, $since)
                    Log ("  stopped: " + (Stop-Tree $proc.Id))
                    $stalled = $true
                    break
                }
            }
            if (-not $stalled) { $proc.WaitForExit() }
            $code = if ($stalled) { "stalled" } else { $proc.ExitCode }
            $minutes = ((Get-Date) - $started).TotalMinutes

            if ($code -eq 0 -and (Test-RunBanked $arm.Rungs $seed)) {
                $j = Get-Content (Join-Path $runsDir "$runId.json") -Raw | ConvertFrom-Json
                Log ("{0} DONE in {1:N1} min: best val Macro-F1 {2:N4} at epoch {3}/{4} (workers={5})" -f `
                     $runId, $minutes, [double]$j.best_val_macro_f1, $j.best_epoch, $j.epochs_run, $workers)
                $ok = $true; $done++
                break
            }
            $hint = (Get-Content $err -Tail 400 -ErrorAction SilentlyContinue | Select-String "1455|out of memory|OutOfMemory|DISK LOW|Error|error" | Select-Object -Last 2) -join " | "
            Log ("{0} attempt workers={1} FAILED after {2:N1} min (exit {3}). {4}" -f $runId, $workers, $minutes, $code, $hint)
            Start-Sleep -Seconds 60     # let the OS reclaim memory and the pagefile settle
        }

        if (-not $ok) {
            Log "$runId FAILED on every attempt (workers 2, 1, 0). Recorded; moving on to the next arm."
            $failed += $runId
        }
    }
}

$final = if ($failed.Count) { "finished with failures: " + ($failed -join ", ") } else { "complete: all {0} pooled runs banked" -f ($SEEDS.Count * $arms.Count) }
Log "=== Block 3 $final ==="
Write-Status $final @{ runs_banked = $done; runs_total = $SEEDS.Count * $arms.Count; failed = $failed }
$mutex.ReleaseMutex()
if ($failed.Count) { exit 1 } else { exit 0 }

