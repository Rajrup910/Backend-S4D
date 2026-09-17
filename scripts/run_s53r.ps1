# S53r -- overnight re-run of everything the V4 audit invalidated. Unattended, self-guarding and
# self-restarting: whatever happens in the night, the morning finds banked runs and a report.
#
#   1. S49 archive probe with the fixed colour step           (GPU feature pass, short)
#   2. smoke every rung once (one epoch, two batches)         (catches a broken arm in minutes)
#   3. the S53r schedule from research.v4.ladder_rerun, in priority order:
#        tier A  R0 + R2          x seeds 42/43/44            ~4.2 h (measured min/epoch)
#        tier B  R6, R5, R7       x seeds 42/43/44            ~4.6 h
#        tier C  R1, R4           x seeds 43/44 (s42 reused)  ~2.5 h
#      every run: HAM-only, --patience 0 (full schedule), --run-tag rerun, --device cuda
#   4. a second pass over anything that failed
#   5. python -m research.v4.ladder_rerun --report            (always runs, partial results included)
#
# Crash resistance -- every rule lives here, not in a prompt (memory: unattended runs need gates):
#   ONE COPY    an instance mutex; a second copy exits at once
#   RESTART     on its first start the script registers a Task Scheduler watchdog that re-launches
#               it with -Resume every 15 minutes. -Resume does nothing unless this run was ARMED,
#               is not DONE, and no copy is alive. The task removes itself when the run is DONE.
#               Task Scheduler processes live outside the host terminal / IDE, so closing it kills nothing.
#   RESUME      banked runs (plan-matching JSON) and passed smokes are skipped on every restart
#   ORPHANS     a training process left behind by a dead runner is allowed to finish (it banks
#               its own JSON); if its checkpoint stops advancing it is killed
#   AWAKE       asks Windows not to sleep while the script runs (does not survive a closed lid)
#   GPU         runs are pinned to --device cuda, so a lost GPU fails fast instead of training on
#               CPU for hours; the script waits up to 30 min for nvidia-smi to come back
#   RETRY       workers 2 -> 1 -> 0, then half batch x double accumulation (same effective batch)
#               if the error was out-of-memory
#   WATCHDOG    a run whose _last.pt has not advanced for STALL_MIN is killed and retried
#   DISK        stops below MIN_DISK_GB; the watchdog resumes once space exists
#   CONTINUE    a failed run is recorded, the schedule moves on, and failures get a second pass
#   REPORT      the report runs in a finally block, so even a crashed night leaves a readout
#   STATUS      results/v4/s53r/status.json every minute (readable from a phone), with an ETA
#   STOP        create results\v4\s53r\STOP to finish the current run and stop for good
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_s53r.ps1 -DryRun
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_s53r.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_s53r.ps1 -Disarm   # remove the watchdog

param([switch]$DryRun, [switch]$Resume, [switch]$Disarm, [switch]$SkipS49)

$ErrorActionPreference = "Continue"
$repo     = Split-Path -Parent $PSScriptRoot
$self     = $PSCommandPath
$py       = "C:\Users\RAJ\Downloads\Capstone\.venv\Scripts\python.exe"
$runsDir  = Join-Path $repo "results\v4\recipe_runs"
$ckptDir  = Join-Path $repo "ml\checkpoints"
$outDir   = Join-Path $repo "results\v4\s53r"
$logDir   = Join-Path $outDir "logs"
$log      = Join-Path $outDir "run.log"
$status   = Join-Path $outDir "status.json"
$stopFile = Join-Path $outDir "STOP"
$armed    = Join-Path $outDir "ARMED"
$doneFile = Join-Path $outDir "DONE"
$taskName = "Capstone_S53r_Watchdog"
$WORKER_LADDER = @(2, 1, 0)
$STALL_MIN     = 20
$GPU_WAIT_MIN  = 30
$MIN_DISK_GB   = 6
$EPOCHS = @{ R0 = 30; R1 = 30; R2 = 30; R4 = 30; R5 = 30; R6 = 60; R7 = 30 }
Set-Location $repo
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Log($msg) {
    $line = "[{0:yyyy-MM-dd HH:mm:ss}] [pid {1}] {2}" -f (Get-Date), $PID, $msg
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
    $o = [ordered]@{ updated = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss"); state = $state; runner_pid = $PID }
    foreach ($k in $extra.Keys) { $o[$k] = $extra[$k] }
    $tmp = "$status.tmp"
    try { ($o | ConvertTo-Json -Depth 4) | Set-Content -Path $tmp -Encoding utf8; Move-Item -Force $tmp $status } catch { }
}

function Remove-Watchdog {
    try { Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction Stop; Log "watchdog task removed" } catch { }
}

if ($Disarm) {
    Remove-Watchdog
    Remove-Item $armed -ErrorAction SilentlyContinue
    Write-Output "disarmed: watchdog removed, ARMED marker cleared"
    exit 0
}

# ------------------------------------------------------------------------------ one copy only
$instance = New-Object System.Threading.Mutex($false, "Global\capstone_s53r_instance")
$haveInstance = $false
try { $haveInstance = $instance.WaitOne(0) } catch [System.Threading.AbandonedMutexException] { $haveInstance = $true }
if (-not $haveInstance) {
    if (-not $Resume) { Write-Output "another run_s53r.ps1 is already running; nothing to do" }
    exit 0
}
if ($Resume) {
    $quit = $false
    if (Test-Path $doneFile) { Remove-Watchdog; $quit = $true }
    elseif (-not (Test-Path $armed)) { $quit = $true }     # the owner never launched it: start nothing
    elseif (Test-Path $stopFile) { $quit = $true }
    if ($quit) { $instance.ReleaseMutex(); exit 0 }
}

# ------------------------------------------------------------------------------ helpers
Add-Type -Namespace S53r -Name Power -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint esFlags);
'@ -ErrorAction SilentlyContinue
function Set-KeepAwake([bool]$on) {
    # ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED to hold, ES_CONTINUOUS alone to release
    $flags = if ($on) { [uint32]2147483713 } else { [uint32]2147483648 }
    try { [void][S53r.Power]::SetThreadExecutionState($flags) } catch { }
}

function Get-TrainingProcs {
    @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
      Where-Object { $_.CommandLine -match 'research\.v4\.train_v4|ml\.training\.train|research\.v4\.archive_probe' })
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

function Get-DiskFreeGB { [math]::Round((Get-PSDrive -Name (Split-Path $repo -Qualifier).TrimEnd(':')).Free / 1GB, 2) }

function Get-NewestRerunCheckpointAge {
    $f = Get-ChildItem $ckptDir -Filter "convnext_tiny-v4_*_rerun_last.pt" -ErrorAction SilentlyContinue |
         Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $f) { return $null }
    return ((Get-Date) - $f.LastWriteTime).TotalMinutes
}

function Test-Gpu {
    try { $null = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null; return ($LASTEXITCODE -eq 0) } catch { return $false }
}

function Wait-Gpu {
    for ($m = 0; $m -lt $GPU_WAIT_MIN; $m++) {
        if (Test-Gpu) { return $true }
        if ($m -eq 0) { Log "GPU not visible to nvidia-smi; waiting up to $GPU_WAIT_MIN min" }
        Write-Status "waiting for GPU" @{ minutes = $m }
        Start-Sleep -Seconds 60
    }
    return (Test-Gpu)
}

function Test-Banked($item) {
    $path = Join-Path $runsDir ("{0}.json" -f $item.run_id)
    if (-not (Test-Path $path)) { return $false }
    try {
        $j = Get-Content $path -Raw | ConvertFrom-Json
        return (-not $j.smoke) -and ([int]$j.seed -eq [int]$item.seed) -and ($j.run_tag -eq "rerun") -and
               ($null -eq $j.early_stopping_patience) -and ([int]$j.epochs_run -eq $EPOCHS[$item.rung])
    } catch { return $false }
}

function Get-EtaHours {
    $left = 0.0
    foreach ($i in $script:schedule) { if (-not (Test-Banked $i)) { $left += [double]$i.est_minutes } }
    return [math]::Round($left / 60, 1)
}

# Runs one python command under the watchdog. Returns a hashtable with result ok|failed|stalled.
function Invoke-Watched($argLine, $name, $progressFile, $stallMin = $STALL_MIN) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $out = Join-Path $logDir "$name.$stamp.out.log"
    $err = Join-Path $logDir "$name.$stamp.err.log"
    $started = Get-Date
    try {
        $proc = Start-Process -FilePath $py -ArgumentList $argLine -WorkingDirectory $repo `
                    -RedirectStandardOutput $out -RedirectStandardError $err -NoNewWindow -PassThru -ErrorAction Stop
    } catch {
        Log ("could not start {0}: {1}" -f $name, $_.Exception.Message)
        return @{ result = "failed"; err = $err; minutes = 0 }
    }
    $null = $proc.Handle     # PS 5.1 quirk: cache the handle or ExitCode stays null
    while (-not $proc.HasExited) {
        Start-Sleep -Seconds 60
        if ($proc.HasExited) { break }
        $since = ((Get-Date) - $started).TotalMinutes
        if ($progressFile -and (Test-Path $progressFile)) {
            $w = (Get-Item $progressFile).LastWriteTime
            if ($w -gt $started) { $since = ((Get-Date) - $w).TotalMinutes }
        }
        $epoch = $null
        $tail = Get-Content $out -Tail 40 -ErrorAction SilentlyContinue | Select-String "^epoch\s+(\d+)/(\d+)" | Select-Object -Last 1
        if ($tail) { $epoch = $tail.Matches[0].Groups[1].Value + "/" + $tail.Matches[0].Groups[2].Value }
        Write-Status "running" @{ step = $name; epoch = $epoch; minutes_since_progress = [math]::Round($since, 1);
                                  disk_free_gb = (Get-DiskFreeGB); banked = $script:banked; planned = $script:planned;
                                  failed = @($script:failed); eta_hours_left = (Get-EtaHours) }
        if ($since -gt $stallMin) {
            Log ("{0} STALLED: no progress for {1:N0} min. Killing." -f $name, $since)
            Log ("  stopped: " + (Stop-Tree $proc.Id))
            return @{ result = "stalled"; err = $err; minutes = ((Get-Date) - $started).TotalMinutes }
        }
    }
    $proc.WaitForExit()
    $res = if ($proc.ExitCode -eq 0) { "ok" } else { "failed" }
    return @{ result = $res; err = $err; code = $proc.ExitCode; minutes = ((Get-Date) - $started).TotalMinutes }
}

function Invoke-Run($item) {
    $lastCkpt = Join-Path $ckptDir ("convnext_tiny-v4_{0}_last.pt" -f $item.run_id)
    $attempts = @()
    foreach ($w in $WORKER_LADDER) { $attempts += @{ Workers = $w; Batch = [int]$item.batch; Accum = [int]$item.accum; Oom = $false } }
    $attempts += @{ Workers = 0; Batch = [int][math]::Max(1, [int]$item.batch / 2); Accum = [int]$item.accum * 2; Oom = $true }
    $lastWasOom = $false
    foreach ($a in $attempts) {
        if ($a.Oom -and -not $lastWasOom) { continue }     # the half-batch attempt is only for OOM
        if (Test-Path $stopFile) { return $false }
        if (Test-Banked $item) { return $true }             # an orphan may have finished it meanwhile
        $disk = Get-DiskFreeGB
        if ($disk -lt $MIN_DISK_GB) {
            Log ("DISK LOW ({0} GB < {1} GB). Stopping; the watchdog resumes once space is freed." -f $disk, $MIN_DISK_GB)
            Write-Status "stopped: disk low" @{ run = $item.run_id; disk_free_gb = $disk }
            throw "disk low"
        }
        if (-not (Wait-Gpu)) { Log "GPU still missing after $GPU_WAIT_MIN min; skipping $($item.run_id) for now"; return $false }
        $argLine = "-m research.v4.train_v4 --rungs {0} --corpus ham_only --seed {1} --batch-size {2} --grad-accum {3} --num-workers {4} --patience 0 --run-tag rerun --device cuda" -f `
                   $item.rung, $item.seed, $a.Batch, $a.Accum, $a.Workers
        Log ("--- tier {0} {1}: workers={2} batch={3}x{4} (~{5} min, ETA {6} h left) ---" -f `
             $item.tier, $item.run_id, $a.Workers, $a.Batch, $a.Accum, $item.est_minutes, (Get-EtaHours))
        $r = Invoke-Watched $argLine ("{0}.w{1}.b{2}" -f $item.run_id, $a.Workers, $a.Batch) $lastCkpt
        if ($r.result -eq "ok" -and (Test-Banked $item)) {
            $j = Get-Content (Join-Path $runsDir "$($item.run_id).json") -Raw | ConvertFrom-Json
            Log ("{0} DONE in {1:N1} min: final val Macro-F1 {2:N4}, best {3:N4} at epoch {4}/{5}" -f `
                 $item.run_id, $r.minutes, [double]$j.final_val_macro_f1, [double]$j.best_val_macro_f1, $j.best_epoch, $j.epochs_run)
            return $true
        }
        $errText = (Get-Content $r.err -Tail 400 -ErrorAction SilentlyContinue) -join "`n"
        $lastWasOom = $errText -match "out of memory|OutOfMemory"
        $hint = ($errText -split "`n" | Select-String "1455|memory|DISK|Error|error" | Select-Object -Last 2) -join " | "
        Log ("{0} workers={1} batch={2} FAILED after {3:N1} min ({4}). {5}" -f $item.run_id, $a.Workers, $a.Batch, $r.minutes, $r.result, $hint)
        Start-Sleep -Seconds 60     # let the OS reclaim memory and the driver settle
    }
    return $false
}

# ------------------------------------------------------------------------------ plan
$schedJson = & $py -m research.v4.ladder_rerun --schedule
if ($LASTEXITCODE -ne 0) { Log "could not read the S53r schedule"; $instance.ReleaseMutex(); exit 2 }
$script:schedule = @((($schedJson -join "") | ConvertFrom-Json) | ForEach-Object { $_ })   # PS 5.1 emits a JSON array as one object
$script:planned = $script:schedule.Count
$script:banked = @($script:schedule | Where-Object { Test-Banked $_ }).Count
$script:failed = New-Object System.Collections.ArrayList
$rungs = @($script:schedule | ForEach-Object { $_.rung } | Select-Object -Unique)

$mode = if ($DryRun) { "DRY RUN" } elseif ($Resume) { "RESUMED by watchdog" } else { "started" }
Log ("=== S53r runner {0}: {1} runs ({2} banked), rungs {3}, S49 {4} ===" -f `
     $mode, $script:planned, $script:banked, ($rungs -join ","), $(if ($SkipS49) { "skipped" } else { "first" }))
if ($DryRun) {
    foreach ($item in $script:schedule) {
        Log ("  plan: tier {0}  {1}  batch {2} x accum {3}  ~{4} min  banked={5}" -f $item.tier, $item.run_id, $item.batch, $item.accum, $item.est_minutes, (Test-Banked $item))
    }
    $hasTask = [bool](Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue)
    Log ("  dry run: {0} training process(es); disk {1} GB free; GPU visible {2}; ETA {3} h; watchdog task present {4}" -f `
         (Get-TrainingProcs).Count, (Get-DiskFreeGB), (Test-Gpu), (Get-EtaHours), $hasTask)
    $instance.ReleaseMutex()
    exit 0
}

# ------------------------------------------------------------------------------ arm + watchdog
if (-not $Resume) {
    Remove-Item $stopFile -ErrorAction SilentlyContinue      # a fresh manual launch clears an old STOP
    Remove-Item $doneFile -ErrorAction SilentlyContinue
    Set-Content -Path $armed -Value (Get-Date).ToString("s") -Encoding utf8
    try {
        $action = New-ScheduledTaskAction -Execute "powershell.exe" `
                    -Argument ("-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"{0}`" -Resume" -f $self) `
                    -WorkingDirectory $repo
        $trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(15) `
                    -RepetitionInterval (New-TimeSpan -Minutes 15) -RepetitionDuration (New-TimeSpan -Days 2)
        $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
                    -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Days 2) -MultipleInstances IgnoreNew
        Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Force -ErrorAction Stop | Out-Null
        Log "watchdog task '$taskName' registered: re-launches with -Resume every 15 min until DONE"
    } catch {
        Log ("WARNING: could not register the watchdog task ({0}). The run proceeds; a crash needs a manual relaunch." -f $_.Exception.Message)
    }
}

Set-KeepAwake $true
$gpuLock = New-Object System.Threading.Mutex($false, "Global\capstone_v4_run_morning")
$haveGpuLock = $false
try {
    # -------------------------------------------------------------------------- wait for the GPU
    while (-not $haveGpuLock) {
        try { $haveGpuLock = $gpuLock.WaitOne(60000) } catch [System.Threading.AbandonedMutexException] { $haveGpuLock = $true }
        if (-not $haveGpuLock) { Write-Status "waiting for GPU lock"; Log "waiting for the GPU lock" }
    }
    $idle = 0
    while ($idle -lt 3) {
        $busy = Get-TrainingProcs
        if ($busy.Count -gt 0) {
            $idle = 0
            $age = Get-NewestRerunCheckpointAge
            if ($null -ne $age -and $age -gt $STALL_MIN) {
                Log ("an orphaned training process looks HUNG (newest rerun checkpoint {0:N0} min old); killing it" -f $age)
                foreach ($p in $busy) { Log ("  stopped: " + (Stop-Tree ([int]$p.ProcessId))) }
            } else {
                Write-Status "waiting for a running training process to finish"
                Log "a training process is running (possibly an orphan from a crashed runner); letting it finish"
            }
        } else { $idle++ }
        if ($idle -lt 3) { Start-Sleep -Seconds 60 }
    }
    Log ("lock held, GPU idle; disk {0} GB; ETA {1} h" -f (Get-DiskFreeGB), (Get-EtaHours))

    # -------------------------------------------------------------------------- 1. S49
    $s49json = Join-Path $repo "results\v4\archive_probe_pre_post.json"
    if (-not $SkipS49 -and -not (Test-Path $s49json) -and -not (Test-Path $stopFile)) {
        Log "--- S49 archive probe (fixed colour) ---"
        $r = Invoke-Watched "-m research.v4.archive_probe" "S49_archive_probe" $null 60
        if ($r.result -eq "ok" -and (Test-Path $s49json)) { Log ("S49 DONE in {0:N1} min" -f $r.minutes) }
        else { Log ("S49 FAILED ({0}); see {1}. Continuing with the ladder." -f $r.result, $r.err); [void]$script:failed.Add("S49") }
    }

    # -------------------------------------------------------------------------- 2. smoke
    $broken = @()
    foreach ($rung in $rungs) {
        $marker = Join-Path $outDir "smoke_ok_$rung"
        if (Test-Path $marker) { continue }
        $first = $script:schedule | Where-Object { $_.rung -eq $rung } | Select-Object -First 1
        $argLine = "-m research.v4.train_v4 --rungs {0} --corpus ham_only --seed {1} --batch-size {2} --grad-accum {3} --num-workers 0 --patience 0 --run-tag rerun --device cuda --smoke" -f `
                   $rung, $first.seed, $first.batch, $first.accum
        $r = Invoke-Watched $argLine "smoke_$rung" $null
        if ($r.result -eq "ok") { Log ("smoke {0} OK ({1:N1} min)" -f $rung, $r.minutes); Set-Content $marker "ok" }
        else {
            $hint = (Get-Content $r.err -Tail 200 -ErrorAction SilentlyContinue | Select-String "Error|error" | Select-Object -Last 2) -join " | "
            Log ("smoke {0} FAILED ({1}): {2}. Its runs are still attempted once in pass 1." -f $rung, $r.result, $hint)
            $broken += $rung
        }
    }

    # -------------------------------------------------------------------------- 3-4. ladder, two passes
    foreach ($pass in 1, 2) {
        $todo = @($script:schedule | Where-Object { -not (Test-Banked $_) })
        if ($todo.Count -eq 0) { break }
        if ($pass -eq 2) { Log ("--- second pass over {0} unfinished run(s) ---" -f $todo.Count) }
        $script:failed.Clear()
        foreach ($item in $todo) {
            if (Test-Path $stopFile) { Log "STOP file present; stopping before $($item.run_id)"; break }
            if (Test-Banked $item) { continue }
            if ($broken -contains $item.rung -and $pass -eq 2) { [void]$script:failed.Add($item.run_id); continue }
            if (Invoke-Run $item) { $script:banked = @($script:schedule | Where-Object { Test-Banked $_ }).Count }
            else { Log "$($item.run_id) not banked in pass $pass; moving on"; [void]$script:failed.Add($item.run_id) }
        }
        if (Test-Path $stopFile) { break }
    }
}
catch {
    Log ("runner error: {0}" -f $_.Exception.Message)
}
finally {
    # -------------------------------------------------------------------------- 5. report, always
    try {
        Log "--- report ---"
        $rep = & $py -m research.v4.ladder_rerun --report 2>&1
        $rep | ForEach-Object { Log "  $_" }
    } catch { Log ("report failed: {0}" -f $_.Exception.Message) }
    $script:banked = @($script:schedule | Where-Object { Test-Banked $_ }).Count
    $complete = ($script:banked -eq $script:planned)
    $final = if ($complete) { "complete" } elseif (Test-Path $stopFile) { "stopped by STOP file" } else { "incomplete: " + (@($script:failed) -join ", ") }
    Log "=== S53r $final ($($script:banked)/$($script:planned) runs banked) ==="
    Write-Status $final @{ banked = $script:banked; planned = $script:planned; failed = @($script:failed); eta_hours_left = (Get-EtaHours) }
    if ($complete -or (Test-Path $stopFile)) {
        Set-Content -Path $doneFile -Value (Get-Date).ToString("s") -Encoding utf8
        Remove-Watchdog
    }
    Set-KeepAwake $false
    if ($haveGpuLock) { try { $gpuLock.ReleaseMutex() } catch { } }
    try { $instance.ReleaseMutex() } catch { }
}
if ($script:banked -eq $script:planned) { exit 0 } else { exit 1 }
