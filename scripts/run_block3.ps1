# S53 Block 3 -- seeds 43 and 44 for both Block 2 arms, paired and seed-matched.
#
# Why this is a script and not a list of commands: every rule it applies is one that failed on
# 2026-09-16 when it lived in a prompt or a person's memory.
#
#   * It WAITS for Block 2 rather than assuming it is done. It takes the same named lock as
#     run_morning.ps1, so it cannot share the GPU with it -- queue it now and it starts the moment
#     Block 2 releases the lock.
#   * It refuses to start unless both seed-42 arms banked a result. A seed comparison against a
#     missing seed 42 is not paired.
#   * Each seed reuses the batch size, grad-accum and worker count its seed-42 run actually
#     recorded, read from the run JSON -- never typed here -- so the three seeds of an arm differ in
#     the seed and nothing else.
#   * Arms that already banked a result are skipped, so a relaunch resumes.
#   * Disk space is enforced inside train_v4 (5 GB at launch, 2 GB per epoch).
#
# S48's rule binds both arms unconditionally: every V4 under-40 condition runs >= 3 seeds. Report
# the seed-averaged estimate with its range, and never the best seed.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_block3.ps1

$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
$py   = "C:\Users\RAJ\Downloads\Capstone\.venv\Scripts\python.exe"
$runs = Join-Path $repo "results\v4\recipe_runs"
$log  = Join-Path $repo "results\v4\block3_run.log"
Set-Location $repo

function Log($msg) {
    $line = "[{0:yyyy-MM-dd HH:mm:ss}] {1}" -f (Get-Date), $msg
    Write-Output $line
    for ($try = 0; $try -lt 5; $try++) {
        try {
            $fs = [System.IO.File]::Open($log, 'Append', 'Write', 'ReadWrite')
            try { $b = [System.Text.Encoding]::UTF8.GetBytes($line + "`r`n"); $fs.Write($b, 0, $b.Length) }
            finally { $fs.Dispose() }
            return
        } catch { Start-Sleep -Milliseconds 50 }
    }
}

# Same lock as run_morning.ps1: the two can never train at once. Block here until Block 2 is done.
$mutex = New-Object System.Threading.Mutex($false, "Global\capstone_v4_run_morning")
Log "=== Block 3 queued; waiting for the GPU lock (Block 2 holds it while it runs) ==="
try { [void]$mutex.WaitOne(-1) } catch [System.Threading.AbandonedMutexException] { }
Log "lock acquired"

# The lock only excludes copies that take it. A run_morning.ps1 started before the mutex existed
# (as on 2026-09-16) holds no lock, so also wait until no training process exists at all.
# Three consecutive idle minutes are required: between Block 2's two arms there is a gap of a few
# seconds with no python process, and a single idle poll could mistake it for "done".
function Test-TrainingRunning {
    @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
      Where-Object { $_.CommandLine -match 'research\.v4\.train_v4|ml\.training\.train' }).Count -gt 0
}
$idle = 0; $waited = 0
while ($idle -lt 3) {
    if (Test-TrainingRunning) {
        if ($idle -gt 0 -or $waited % 15 -eq 0) { Log "training still running; waiting" }
        $idle = 0
    } else { $idle++ }
    if ($idle -lt 3) { Start-Sleep -Seconds 60; $waited++ }
}
Log "no training process for 3 consecutive minutes; proceeding"

# --- which arms, and what seed 42 actually used ---------------------------------------------
$verdictPath = Join-Path $repo "results\v4\screen_verdict.json"
$verdict = Get-Content $verdictPath -Raw | ConvertFrom-Json
if (-not $verdict.proceed) { Log "screen_verdict.json says proceed=false. Block 3 NOT started."; exit 2 }
$composite = @($verdict.composite)

$arms = @()
foreach ($rungs in @(@("R0"), $composite)) {
    $base = Join-Path $runs ("{0}_pooled_s42.json" -f ($rungs -join "+"))
    if (-not (Test-Path $base)) {
        Log ("seed 42 result missing for --rungs {0} ({1}). A seed comparison needs seed 42. Block 3 NOT started." -f ($rungs -join " "), $base)
        exit 3
    }
    $s42 = Get-Content $base -Raw | ConvertFrom-Json
    $accum = if ($s42.grad_accum) { [int]$s42.grad_accum } else { 1 }
    # Seed 42's worker count is not stored; it is implied by the memory accommodation it used.
    $workers = if ($accum -gt 1) { 1 } else { 2 }
    $arms += @{ Rungs = $rungs; Batch = [int]$s42.batch_size; Accum = $accum; Workers = $workers }
    Log ("arm --rungs {0}: seed 42 used batch {1} x accum {2} (effective {3}), {4} worker(s)" -f `
         ($rungs -join " "), $s42.batch_size, $accum, ($s42.batch_size * $accum), $workers)
}

# --- the runs, interleaved by seed so a stop leaves complete pairs ---------------------------
foreach ($seed in 43, 44) {
    foreach ($arm in $arms) {
        $runId = "{0}_pooled_s{1}" -f ($arm.Rungs -join "+"), $seed
        if (Test-Path (Join-Path $runs "$runId.json")) { Log "$runId already complete; skipping"; continue }
        Log ("--- {0}: batch {1} x accum {2}, {3} worker(s) ---" -f $runId, $arm.Batch, $arm.Accum, $arm.Workers)
        $started = Get-Date
        & $py -m research.v4.train_v4 --rungs $arm.Rungs --corpus pooled --seed $seed `
            --batch-size $arm.Batch --grad-accum $arm.Accum --num-workers $arm.Workers 2>&1 |
            ForEach-Object { Log $_ }
        if ($LASTEXITCODE -ne 0) {
            Log ("{0} FAILED with exit {1}. Stopping; relaunch this script to resume." -f $runId, $LASTEXITCODE)
            exit $LASTEXITCODE
        }
        Log ("{0} finished in {1:N1} min" -f $runId, ((Get-Date) - $started).TotalMinutes)
    }
}

Log "=== Block 3 complete: seeds 42/43/44 for both arms. Next: S54 on the reserved cohort. ==="
exit 0
