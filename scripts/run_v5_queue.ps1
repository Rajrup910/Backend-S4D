# V5 queue runner -- runs the lines of results\v5\queue_<Queue>.txt one at a time (docs/V5_RUNSHEET.md
# section 11.3, E4). Deliberately simpler than run_s72.ps1: it installs NO scheduled task and never
# restarts itself. An interrupted night is resumed by running the same command again.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_v5_queue.ps1 -Queue N2 -DryRun
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_v5_queue.ps1 -Queue N2
#
# Queue file: one run per line, `#` comments and blank lines ignored. A line is the argument list of
#   research.v5.train_v5   (default), or, with the prefix `v4 `, of research.v4.train_v4
# e.g.
#   --arm look --fold 0 --seed 42 --image-size 224 --batch-size 32 --run-tag v5scr --no-save-best
#   v4 --rungs R1 --corpus pooled --fold 1 --seed 42 --batch-size 16 --grad-accum 2 --run-tag v5ctl
#
# Rules it enforces (runsheet 11.1):
#   C1  exactly one GPU job: a mutex, and it refuses to start if another train_v4/train_v5 is alive
#   C2  before EVERY run: commit headroom >= 6 GB and C: >= 10 GB, else it waits up to 10 min, then stops
#   C6  GPU > 87 C after a run: pause the queue 20 min (the run itself is never interrupted)
#   C7  results\v5\PAUSE stops the queue between runs, never mid-run
#   --num-workers 2 and --patience 0 are appended if the line does not set them
# A run is COMPLETE when its run JSON exists with smoke=false and epochs_run == 30; completed runs are
# skipped. A failed run is retried once (workers 1), recorded, and the queue moves on.

param([Parameter(Mandatory = $true)][string]$Queue, [switch]$DryRun, [switch]$SkipC2)

$ErrorActionPreference = "Continue"
$repo    = Split-Path -Parent $PSScriptRoot
$py      = "C:\Users\RAJ\Downloads\Capstone\.venv\Scripts\python.exe"
$v5Dir   = Join-Path $repo "results\v5"
$qFile   = Join-Path $v5Dir ("queue_{0}.txt" -f $Queue)
$logDir  = Join-Path $v5Dir "logs"
$log     = Join-Path $logDir ("queue_{0}.log" -f $Queue)
$pause   = Join-Path $v5Dir "PAUSE"
$MIN_HEADROOM_GB = 6.0
$MIN_DISK_GB     = 10.0
$HOT_C           = 87
$EPOCHS          = 30
Set-Location $repo
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Log($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path $log -Value $line -Encoding utf8
}

function Get-Headroom {
    $cl = (Get-Counter '\Memory\Commit Limit').CounterSamples[0].CookedValue / 1GB
    $cb = (Get-Counter '\Memory\Committed Bytes').CounterSamples[0].CookedValue / 1GB
    return [math]::Round($cl - $cb, 1)
}
function Get-DiskFreeGB { [math]::Round((Get-PSDrive -Name "C").Free / 1GB, 1) }
function Get-GpuTemp {
    try { return [int](& nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>$null) } catch { return 0 }
}
function Get-TrainingProcs {
    @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
      Where-Object { $_.CommandLine -match 'research\.v[45]\.train_v[45]' })
}

# ----- one copy only (C1)
$instance = New-Object System.Threading.Mutex($false, "Global\capstone_v5_queue")
$have = $false
try { $have = $instance.WaitOne(0) } catch [System.Threading.AbandonedMutexException] { $have = $true }
if (-not $have) { Write-Host "another run_v5_queue.ps1 is already running; exiting."; exit 0 }

# ----- parse one queue line into its module, args and expected run-JSON path
function Get-Arg([string[]]$tokens, [string]$name) {
    $i = [array]::IndexOf($tokens, $name)
    if ($i -ge 0 -and $i + 1 -lt $tokens.Count) { return $tokens[$i + 1] }
    return $null
}
function Parse-Line([string]$line) {
    $module = "research.v5.train_v5"
    if ($line -match '^\s*v4\s+') { $module = "research.v4.train_v4"; $line = $line -replace '^\s*v4\s+', '' }
    $t = @($line -split '\s+' | Where-Object { $_ })
    $seed = Get-Arg $t "--seed"; if (-not $seed) { $seed = "42" }
    $tag  = Get-Arg $t "--run-tag"
    if ($module -eq "research.v5.train_v5") {
        $arm = Get-Arg $t "--arm"; $fold = Get-Arg $t "--fold"; $loao = Get-Arg $t "--loao-holdout"
        $split = if ($loao) { "loao-$loao" } else { "f$fold" }
        $id = "{0}_{1}_s{2}" -f $arm, $split, $seed
        $trunk = Get-Arg $t "--trunk"
        if ($trunk -and $trunk -ne "in1k") { $id = "${id}_$trunk" }  # matches train_v5's run id
        if ($tag) { $id = "${id}_$tag" }
        $json = Join-Path $v5Dir ("runs\{0}.json" -f $id)
    } else {
        $rungs = Get-Arg $t "--rungs"; $fold = Get-Arg $t "--fold"
        $id = "{0}_kfold_f{1}_s{2}" -f $rungs, $fold, $seed
        if ($tag) { $id = "${id}_$tag" }
        $json = Join-Path $repo ("results\v4\kfold\runs\{0}.json" -f $id)
    }
    $extra = @()
    if ($t -notcontains "--num-workers") { $extra += @("--num-workers", "2") }
    if ($t -notcontains "--patience")    { $extra += @("--patience", "0") }
    if ($t -notcontains "--device")      { $extra += @("--device", "cuda") }
    return @{ module = $module; tokens = ($t + $extra); id = $id; json = $json }
}
function Test-Complete($json) {
    if (-not (Test-Path $json)) { return $false }
    try {
        $j = Get-Content $json -Raw | ConvertFrom-Json
        return (-not $j.smoke) -and ([int]$j.epochs_run -eq $EPOCHS)
    } catch { return $false }
}

# ----- C2 gate
function Wait-C2 {
    if ($SkipC2) { return $true }
    for ($m = 0; $m -le 10; $m++) {
        $h = Get-Headroom; $d = Get-DiskFreeGB
        if ($h -ge $MIN_HEADROOM_GB -and $d -ge $MIN_DISK_GB) { return $true }
        if ($m -eq 0) { Log ("C2 not met: commit headroom {0} GB (need {1}), C: free {2} GB (need {3}); waiting up to 10 min" -f $h, $MIN_HEADROOM_GB, $d, $MIN_DISK_GB) }
        Start-Sleep -Seconds 60
    }
    return $false
}

# ----- one attempt
function Invoke-Run($p, [string[]]$tokens) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $out = Join-Path $logDir ("{0}.{1}.out.log" -f $p.id, $stamp)
    $err = Join-Path $logDir ("{0}.{1}.err.log" -f $p.id, $stamp)
    $argLine = "-m {0} {1}" -f $p.module, ($tokens -join " ")
    $proc = Start-Process -FilePath $py -ArgumentList $argLine -WorkingDirectory $repo `
                -RedirectStandardOutput $out -RedirectStandardError $err -NoNewWindow -PassThru
    $null = $proc.Handle       # PS 5.1: cache the handle or ExitCode stays null
    $proc.WaitForExit()
    return @{ code = $proc.ExitCode; err = $err; out = $out }
}

# ----- main
if (-not (Test-Path $qFile)) { Log "queue file not found: $qFile"; $instance.ReleaseMutex(); exit 2 }
$lines = @(Get-Content $qFile | Where-Object { $_.Trim() -and -not $_.Trim().StartsWith("#") })
$plan = @($lines | ForEach-Object { Parse-Line $_ })
Log ("=== queue {0}: {1} runs, {2} already complete ===" -f $Queue, $plan.Count, @($plan | Where-Object { Test-Complete $_.json }).Count)

if ($DryRun) {
    foreach ($p in $plan) { Log ("  {0,-46} complete={1}  -> {2} {3}" -f $p.id, (Test-Complete $p.json), $p.module, ($p.tokens -join " ")) }
    Log ("C2 now: headroom {0} GB, C: free {1} GB, GPU {2} C" -f (Get-Headroom), (Get-DiskFreeGB), (Get-GpuTemp))
    $instance.ReleaseMutex(); exit 0
}

if ((Get-TrainingProcs).Count -gt 0) { Log "a train_v4/train_v5 process is already running (C1); refusing to start."; $instance.ReleaseMutex(); exit 3 }

$done = 0; $failed = @()
try {
    foreach ($p in $plan) {
        if (Test-Path $pause) { Log "PAUSE file present; stopping between runs."; break }
        if (Test-Complete $p.json) { Log ("skip {0} (complete)" -f $p.id); continue }
        if (-not (Wait-C2)) { Log "C2 still not met after 10 min; stopping the queue. Free memory/disk and re-run the same command."; break }
        Log ("--- {0}  (headroom {1} GB, C: {2} GB, GPU {3} C) ---" -f $p.id, (Get-Headroom), (Get-DiskFreeGB), (Get-GpuTemp))
        $started = Get-Date
        $r = Invoke-Run $p $p.tokens
        if ($r.code -ne 0 -or -not (Test-Complete $p.json)) {
            $errText = (Get-Content $r.err -Tail 300 -ErrorAction SilentlyContinue) -join "`n"
            $hint = ($errText -split "`n" | Select-String "1455|out of memory|DISK|Error" | Select-Object -Last 2) -join " | "
            Log ("{0} FAILED (exit {1}). {2}" -f $p.id, $r.code, $hint)
            Start-Sleep -Seconds 60
            $retry = @($p.tokens | ForEach-Object { $_ })
            $i = [array]::IndexOf($retry, "--num-workers"); if ($i -ge 0) { $retry[$i + 1] = "1" }
            Log ("retrying {0} with --num-workers 1" -f $p.id)
            $r = Invoke-Run $p $retry
        }
        if ($r.code -eq 0 -and (Test-Complete $p.json)) {
            $done++
            Log ("{0} DONE in {1:N1} min" -f $p.id, ((Get-Date) - $started).TotalMinutes)
        } else {
            $failed += $p.id
            Log ("{0} FAILED again; recorded, moving on. See {1}" -f $p.id, $r.err)
        }
        $temp = Get-GpuTemp
        if ($temp -gt $HOT_C) { Log ("GPU {0} C > {1}: pausing the queue 20 min (C6)" -f $temp, $HOT_C); Start-Sleep -Seconds 1200 }
    }
} finally {
    Log ("=== queue {0} finished: {1} run(s) completed this session; failed: {2} ===" -f $Queue, $done, $(if ($failed.Count) { $failed -join ", " } else { "none" }))
    try { $instance.ReleaseMutex() } catch { }
}
