# One-screen status for S53 Blocks 2 and 3. Read-only: starts, stops and changes nothing.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\status.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\status.ps1 -Watch     # refresh every 60 s

param([switch]$Watch)

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

function Show {
    Clear-Host
    Write-Host ("=== S53 status  {0:HH:mm:ss} ===" -f (Get-Date)) -ForegroundColor Cyan

    Write-Host "`n-- banked pooled runs (Blocks 2+3) --" -ForegroundColor Yellow
    $banked = Get-ChildItem "results\v4\recipe_runs" -Filter "*_pooled_s*.json" -ErrorAction SilentlyContinue
    foreach ($seed in 42, 43, 44) {
        foreach ($arm in "R0", "R1+R4") {
            $f = "results\v4\recipe_runs\${arm}_pooled_s$seed.json"
            if (Test-Path $f) {
                $j = Get-Content $f -Raw | ConvertFrom-Json
                "  {0,-18} DONE  best {1:N4} @ epoch {2}/{3}  {4,5:N1} min  workers={5}" -f `
                    "${arm}_s$seed", [double]$j.best_val_macro_f1, $j.best_epoch, $j.epochs_run, ($j.train_time_seconds / 60), $(if ($j.num_workers -ne $null) { $j.num_workers } else { "-" })
            } else {
                "  {0,-18} pending" -f "${arm}_s$seed"
            }
        }
    }
    "  {0} of 6 banked" -f @($banked).Count

    Write-Host "`n-- run in progress --" -ForegroundColor Yellow
    $procs = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
               Where-Object { $_.CommandLine -match 'research\.v4\.train_v4' })
    if ($procs.Count -eq 0) { "  no training process running" }
    else {
        $cmd = $procs[0].CommandLine
        $rungs = ([regex]::Match($cmd, '--rungs (.+?) --corpus').Groups[1].Value).Trim() -replace ' ', '+'
        $seed = [regex]::Match($cmd, '--seed (\d+)').Groups[1].Value; if (-not $seed) { $seed = "42" }
        $workers = [regex]::Match($cmd, '--num-workers (\d+)').Groups[1].Value
        "  training {0}_pooled_s{1}  (workers={2})" -f $rungs, $seed, $workers
        $ck = "ml\checkpoints\convnext_tiny-v4_${rungs}_pooled_s${seed}_last.pt"
        if (Test-Path $ck) { "  last checkpoint written {0:N1} min ago" -f ((Get-Date) - (Get-Item $ck).LastWriteTime).TotalMinutes }
        # epoch lines come from Block 2's log or from this run's Block 3 log, whichever is newer
        $sources = @("results\v4\morning_run.log") + @(Get-ChildItem "results\v4\block3_logs" -Filter "${rungs}_pooled_s$seed.*.out.log" -ErrorAction SilentlyContinue |
                    Sort-Object LastWriteTime | Select-Object -Last 1 -ExpandProperty FullName)
        $src = $sources | Where-Object { $_ -and (Test-Path $_) } | Sort-Object { (Get-Item $_).LastWriteTime } | Select-Object -Last 1
        # Whole-file search: progress bars write hundreds of lines per epoch, so a tail misses the summaries.
        Select-String -Path $src -Pattern "val_macroF1" -ErrorAction SilentlyContinue | Select-Object -Last 3 |
            ForEach-Object { "  " + ($_.Line -replace '^\[[^\]]+\]\s*', '').Trim() }
    }

    Write-Host "`n-- Block 3 runner --" -ForegroundColor Yellow
    if (Test-Path "results\v4\block3_status.json") {
        $s = Get-Content "results\v4\block3_status.json" -Raw | ConvertFrom-Json
        $s.PSObject.Properties | ForEach-Object { "  {0,-26} {1}" -f $_.Name, ($_.Value -join ', ') }
    } else { "  no status file yet" }
    $runners = @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" -ErrorAction SilentlyContinue |
                 Where-Object { $_.CommandLine -match 'run_block3|run_morning' } |
                 ForEach-Object { [regex]::Match($_.CommandLine, 'run_\w+').Value })
    "  runner scripts alive: " + $(if ($runners) { $runners -join ", " } else { "none" })

    Write-Host "`n-- recent log lines --" -ForegroundColor Yellow
    foreach ($l in "results\v4\morning_run.log", "results\v4\block3_run.log") {
        if (Test-Path $l) {
            "  " + (Split-Path $l -Leaf) + ":"
            Select-String -Path $l -Pattern "\] (=== |--- |.*DONE|.*FAILED|.*STALLED|.*HUNG|.*finished in|.*DISK|.*skipping|.*lock acquired|.*idle)" |
                Where-Object { $_.Line -notmatch 'RemoteException|stage (head|finetune)' } | Select-Object -Last 3 |
                ForEach-Object { "    " + $_.Line }
        }
    }

    Write-Host "`n-- machine --" -ForegroundColor Yellow
    $gpu = (nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits) -split ',\s*'
    "  GPU {0}% busy, {1}/{2} MiB, {3} C" -f $gpu[0], $gpu[1], $gpu[2], $gpu[3]
    $os = Get-CimInstance Win32_OperatingSystem
    "  RAM free {0:N1} GB   disk C: free {1:N1} GB" -f ($os.FreePhysicalMemory / 1MB), ((Get-PSDrive C).Free / 1GB)
    $ac = (Get-CimInstance -Namespace root\wmi -ClassName BatteryStatus -ErrorAction SilentlyContinue | Select-Object -First 1).PowerOnline
    "  on AC power: $ac"
}

if ($Watch) { while ($true) { Show; Write-Host "`n(refreshing every 60 s -- Ctrl+C to stop; training is unaffected)"; Start-Sleep -Seconds 60 } }
else { Show }
