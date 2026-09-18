# S72 -- the V4 K-fold training programme. Unattended, self-guarding and self-restarting: whatever
# happens in the night, the morning finds five banked folds, a cross-fitted OOF matrix and a report.
#
#   1. verify the frozen S71 partition (hash + three leak checks)   -- refuses to train otherwise
#   2. smoke fold 0 once (one epoch, two batches)                   -- catches a broken path in a minute
#   3. train folds 0..4: R0 control, pooled corpus, 224 px, 30 epochs, --patience 0, --num-workers 2
#      each fold writes ml/checkpoints/convnext_tiny-v4_R0_kfold_f{N}_s42_{best,last}.pt
#      and results/v4/kfold/predictions/fold{N}.csv (the held-out rows, best epoch)
#   4. a second pass over anything that failed
#   5. python -m research.v4.s71_kfold --assemble    (always runs, in a finally block)
#
# Crash resistance -- every rule lives here, not in a prompt (memory: unattended runs need gates):
#   ONE COPY    an instance mutex; a second copy exits at once
#   RESTART     registers a Task Scheduler watchdog that re-launches with -Resume every 15 min.
#               -Resume does nothing unless this run was ARMED, is not DONE, and no copy is alive.
#               The task removes itself when the run is DONE. Task Scheduler processes live outside
#               the host terminal, so closing the window kills nothing.
#   RESUME      banked folds (plan-matching JSON + prediction CSV) are skipped on every restart
#   ORPHANS     a training process left by a dead runner is allowed to finish; if its checkpoint
#               stops advancing it is killed
#   AWAKE       asks Windows not to sleep while the script runs (does not survive a closed lid)
#   GPU         pinned to --device cuda, so a lost GPU fails fast instead of training on CPU for
#               hours; waits up to 30 min for nvidia-smi to come back
#   RETRY       workers 2 -> 1 -> 0, then half batch x double accumulation (same effective batch)
#               if the error was out-of-memory. NEVER above 2: workers=4 dies with Windows 1455.
#   WATCHDOG    a fold whose _last.pt has not advanced for STALL_MIN is killed and retried
#   DISK        stops below MIN_DISK_GB; the watchdog resumes once space exists. Ten checkpoints at
#               111 MB is ~1.1 GB, but the pagefile must also be free to grow -- that is what the
#               floor protects, and it is why error 1455 was misdiagnosed three times in September.
#   CONTINUE    a failed fold is recorded, the schedule moves on, and failures get a second pass
#   STATUS      results/v4/kfold/status.json every minute (readable from a phone), with an ETA
#   STOP        create results\v4\kfold\STOP to finish the current fold and stop for good
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_s72.ps1 -DryRun
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_s72.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_s72.ps1 -Disarm   # remove watchdog

param([switch]$DryRun, [switch]$Resume, [switch]$Disarm, [switch]$SkipSmoke)

$ErrorActionPreference = "Continue"
$repo     = Split-Path -Parent $PSScriptRoot
$self     = $PSCommandPath
$py       = "C:\Users\RAJ\Downloads\Capstone\.venv\Scripts\python.exe"
$ckptDir  = Join-Path $repo "ml\checkpoints"
$outDir   = Join-Path $repo "results\v4\kfold"
$runsDir  = Join-Path $outDir "runs"
$predDir  = Join-Path $outDir "predictions"
$logDir   = Join-Path $outDir "logs"
$log      = Join-Path $outDir "run.log"
$status   = Join-Path $outDir "status.json"
$stopFile = Join-Path $outDir "STOP"
$armed    = Join-Path $outDir "ARMED"
$doneFile = Join-Path $outDir "DONE"
$taskName = "Capstone_S72_Watchdog"
$WORKER_LADDER = @(2, 1, 0)
$STALL_MIN     = 25
$GPU_WAIT_MIN  = 30
$MIN_DISK_GB   = 6
$SEED          = 42
$EPOCHS        = 30
$EST_MIN_PER_FOLD = 52      # measured: R0_pooled_s43 at 117 s/epoch finetune, scaled to a fold
Set-Location $repo
New-Item -ItemType Directory -Force -Path $outDir, $runsDir, $predDir, $logDir | Out-Null

function Log($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path $log -Value $line -Encoding utf8
}

function Write-Status($state, $extra = @{}) {
    $payload = @{ state = $state; updated = (Get-Date -Format "o"); session = "S72" }
    foreach ($k in $extra.Keys) { $payload[$k] = $extra[$k] }
    try { $payload | ConvertTo-Json -Depth 5 | Out-File $status -Encoding utf8 } catch { }
}

function Remove-Watchdog {
    try { schtasks /Delete /TN $taskName /F 2>$null | Out-Null } catch { }
}

# ------------------------------------------------------------------------------ one copy only
$instance = New-Object System.Threading.Mutex($false, "Global\capstone_s72_instance")
$haveInstance = $false
try { $haveInstance = $instance.WaitOne(0) } catch [System.Threading.AbandonedMutexException] { $haveInstance = $true }
if (-not $haveInstance) {
    if (-not $Resume) { Write-Host "another copy of run_s72.ps1 is running; exiting." }
    exit 0
}
if ($Disarm) {
    Remove-Watchdog; Remove-Item $armed -ErrorAction SilentlyContinue
    Log "watchdog removed and run disarmed."; $instance.ReleaseMutex(); exit 0
}
if ($Resume) {
    if ((-not (Test-Path $armed)) -or (Test-Path $doneFile)) { $instance.ReleaseMutex(); exit 0 }
    Log "watchdog resume: no live copy was running, continuing the schedule."
}

# ------------------------------------------------------------------------------ helpers
Add-Type -Namespace S72 -Name Power -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint esFlags);
'@ -ErrorAction SilentlyContinue
function Set-KeepAwake([bool]$on) {
    $flags = if ($on) { [uint32]2147483713 } else { [uint32]2147483648 }
    try { [void][S72.Power]::SetThreadExecutionState($flags) } catch { }
}

function Get-TrainingProcs {
    @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
      Where-Object { $_.CommandLine -match 'research\.v4\.train_v4' })
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

# A fold counts as banked only when BOTH artefacts exist: the run JSON at the full epoch count and
# the held-out prediction CSV. A checkpoint without predictions cannot be assembled into the OOF
# matrix, so treating it as done would leave a hole nothing downstream could fill.
function Test-Banked([int]$fold) {
    $path = Join-Path $runsDir ("R0_kfold_f{0}_s{1}.json" -f $fold, $SEED)
    $pred = Join-Path $predDir ("fold{0}.csv" -f $fold)
    if (-not (Test-Path $path)) { return $false }
    if (-not (Test-Path $pred)) { return $false }
    try {
        $j = Get-Content $path -Raw | ConvertFrom-Json
        return (-not $j.smoke) -and ([int]$j.seed -eq $SEED) -and ([int]$j.fold -eq $fold) -and
               ([int]$j.epochs_run -eq $EPOCHS) -and ($null -eq $j.early_stopping_patience)
    } catch { return $false }
}

function Get-EtaHours {
    $left = 0
    foreach ($f in 0..4) { if (-not (Test-Banked $f)) { $left += $EST_MIN_PER_FOLD } }
    return [math]::Round($left / 60, 1)
}

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
                                  disk_free_gb = (Get-DiskFreeGB); banked = $script:banked;
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

function Invoke-Fold([int]$fold) {
    $lastCkpt = Join-Path $ckptDir ("convnext_tiny-v4_R0_kfold_f{0}_s{1}_last.pt" -f $fold, $SEED)
    $attempts = @()
    foreach ($w in $WORKER_LADDER) { $attempts += @{ Workers = $w; Batch = 32; Accum = 1; Oom = $false } }
    $attempts += @{ Workers = 0; Batch = 16; Accum = 2; Oom = $true }
    $lastWasOom = $false
    foreach ($a in $attempts) {
        if ($a.Oom -and -not $lastWasOom) { continue }
        if (Test-Path $stopFile) { return $false }
        if (Test-Banked $fold) { return $true }
        $disk = Get-DiskFreeGB
        if ($disk -lt $MIN_DISK_GB) {
            Log ("DISK LOW ({0} GB < {1} GB). Stopping; the watchdog resumes once space is freed." -f $disk, $MIN_DISK_GB)
            Write-Status "stopped: disk low" @{ fold = $fold; disk_free_gb = $disk }
            throw "disk low"
        }
        if (-not (Wait-Gpu)) { Log "GPU still missing after $GPU_WAIT_MIN min; skipping fold $fold for now"; return $false }
        $argLine = "-m research.v4.train_v4 --rungs R0 --corpus pooled --fold {0} --seed {1} --batch-size {2} --grad-accum {3} --num-workers {4} --patience 0 --device cuda" -f `
                   $fold, $SEED, $a.Batch, $a.Accum, $a.Workers
        Log ("--- fold {0}/4: workers={1} batch={2}x{3} (~{4} min, ETA {5} h left, disk {6} GB) ---" -f `
             $fold, $a.Workers, $a.Batch, $a.Accum, $EST_MIN_PER_FOLD, (Get-EtaHours), $disk)
        $r = Invoke-Watched $argLine ("f{0}.w{1}.b{2}" -f $fold, $a.Workers, $a.Batch) $lastCkpt
        if ($r.result -eq "ok" -and (Test-Banked $fold)) {
            $j = Get-Content (Join-Path $runsDir ("R0_kfold_f{0}_s{1}.json" -f $fold, $SEED)) -Raw | ConvertFrom-Json
            Log ("fold {0} DONE in {1:N1} min: best val Macro-F1 {2:N4} at epoch {3}/{4}, VRAM peak {5:N2} GB" -f `
                 $fold, $r.minutes, [double]$j.best_val_macro_f1, $j.best_epoch, $j.epochs_run, [double]$j.vram_peak_gb)
            return $true
        }
        $errText = (Get-Content $r.err -Tail 400 -ErrorAction SilentlyContinue) -join "`n"
        $lastWasOom = $errText -match "out of memory|OutOfMemory"
        $hint = ($errText -split "`n" | Select-String "1455|memory|DISK|Error|error" | Select-Object -Last 2) -join " | "
        Log ("fold {0} workers={1} batch={2} FAILED after {3:N1} min ({4}). {5}" -f $fold, $a.Workers, $a.Batch, $r.minutes, $r.result, $hint)
        Start-Sleep -Seconds 60     # let the OS reclaim memory and the driver settle
    }
    return $false
}

# ------------------------------------------------------------------------------ preflight
Log "=================== S72 V4 K-fold training ==================="
& $py -m research.v4.s71_kfold --verify 2>&1 | ForEach-Object { Log "  $_" }
if ($LASTEXITCODE -ne 0) {
    Log "S71 partition does not verify -- refusing to train. Fix the partition first."
    $instance.ReleaseMutex(); exit 2
}
$disk = Get-DiskFreeGB
Log ("preflight: disk {0} GB free (floor {1}), 5 folds x 2 checkpoints x 111 MB = ~1.1 GB expected" -f $disk, $MIN_DISK_GB)
if ($disk -lt $MIN_DISK_GB) { Log "not enough disk to start."; $instance.ReleaseMutex(); exit 2 }
if (-not (Test-Gpu)) { Log "nvidia-smi does not answer -- is the GPU present?"; }

if ($DryRun) {
    Log "DRY RUN. Would train folds 0..4, R0 pooled 224 px, 30 epochs, batch 32, workers 2, patience 0."
    foreach ($f in 0..4) { Log ("  fold {0}: banked={1}" -f $f, (Test-Banked $f)) }
    Log ("ETA {0} h" -f (Get-EtaHours))
    $instance.ReleaseMutex(); exit 0
}

# ------------------------------------------------------------------------------ arm + watchdog
New-Item -ItemType File -Force -Path $armed | Out-Null
Remove-Item $doneFile -ErrorAction SilentlyContinue
if (-not $Resume) {
    $cmd = "powershell -NoProfile -ExecutionPolicy Bypass -File `"$self`" -Resume"
    try {
        schtasks /Create /TN $taskName /TR $cmd /SC MINUTE /MO 15 /F 2>$null | Out-Null
        Log "watchdog registered: $taskName re-launches this script every 15 min until DONE."
    } catch { Log "could not register the watchdog task: $($_.Exception.Message)" }
}
Set-KeepAwake $true
$script:banked = @()
$script:failed = @()

try {
    # smoke once, unless every fold is already banked
    $allBanked = $true
    foreach ($f in 0..4) { if (-not (Test-Banked $f)) { $allBanked = $false } }
    if ((-not $SkipSmoke) -and (-not $allBanked) -and (-not (Test-Path (Join-Path $outDir "smoke_ok")))) {
        Log "smoking fold 0 (one epoch, two batches) before committing the night"
        $r = Invoke-Watched "-m research.v4.train_v4 --rungs R0 --corpus pooled --fold 0 --batch-size 32 --num-workers 2 --device cuda --smoke" "smoke_f0" $null 15
        if ($r.result -ne "ok") {
            Log "SMOKE FAILED -- not starting the schedule. See $($r.err)"
            Write-Status "smoke failed" @{ err = $r.err }
            throw "smoke failed"
        }
        New-Item -ItemType File -Force -Path (Join-Path $outDir "smoke_ok") | Out-Null
        Remove-Item (Join-Path $predDir "smoke") -Recurse -Force -ErrorAction SilentlyContinue
        Log "smoke OK"
    }

    foreach ($pass in 1..2) {
        foreach ($f in 0..4) {
            if (Test-Path $stopFile) { Log "STOP file present; stopping."; break }
            if (Test-Banked $f) {
                if ($pass -eq 1) { Log "fold $f already banked; skipping." }
                if ($script:banked -notcontains $f) { $script:banked += $f }
                continue
            }
            if ($pass -eq 2) { Log "second pass: retrying fold $f" }
            $ok = Invoke-Fold $f
            if ($ok) {
                if ($script:banked -notcontains $f) { $script:banked += $f }
                $script:failed = @($script:failed | Where-Object { $_ -ne $f })
            } elseif ($script:failed -notcontains $f) { $script:failed += $f }
        }
        if ($script:banked.Count -eq 5) { break }
    }
} catch {
    Log "runner stopped: $($_.Exception.Message)"
} finally {
    Set-KeepAwake $false
    Log ("banked {0}/5 folds{1}" -f $script:banked.Count,
         $(if ($script:failed.Count) { "; failed: " + ($script:failed -join ", ") } else { "" }))
    if ($script:banked.Count -eq 5) {
        Log "assembling the cross-fitted OOF matrix"
        & $py -m research.v4.s71_kfold --assemble 2>&1 | ForEach-Object { Log "  $_" }
        # S68 fits target-side thresholds on the OOF matrix and develops on V4 val, so val needs a
        # V4 panel too. No fold model saw a val row, which is why the plain ensemble is leak-free
        # here. ~10 min; done now so the morning has everything S68 needs.
        if (-not (Test-Path (Join-Path $outDir "val_predictions.csv"))) {
            Log "scoring V4 val with the 5-fold ensemble (for S68)"
            $r = Invoke-Watched "-m research.v4.s72_infer --split val --batch-size 32 --num-workers 2 --device cuda" "val_infer" $null 20
            if ($r.result -ne "ok") { Log "val inference FAILED ($($r.result)); S68 can be run after a manual retry. See $($r.err)" }
        }
        New-Item -ItemType File -Force -Path $doneFile | Out-Null
        Remove-Watchdog
        Write-Status "done" @{ banked = $script:banked; failed = @($script:failed) }
        Log "S72 COMPLETE. Watchdog removed. Next: S73."
    } else {
        Write-Status "incomplete" @{ banked = $script:banked; failed = @($script:failed);
                                     eta_hours_left = (Get-EtaHours) }
        Log "S72 INCOMPLETE -- the watchdog will resume in <=15 min. Use -Disarm to stop for good."
    }
    try { $instance.ReleaseMutex() } catch { }
}
