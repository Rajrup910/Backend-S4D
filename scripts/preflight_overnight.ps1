# Preflight for any unattended overnight GPU run. Every overnight chain calls this before its first run and stops if it exits non-zero, so the check cannot be forgotten.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\preflight_overnight.ps1              # check only, changes nothing
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\preflight_overnight.ps1 -StopHogs    # also stops the approved idle list below, then checks
#
# FAIL means the night would be at risk, so the chain does not start: PAUSE file, on battery, idle sleep or hibernate on mains, restart pending, a training job already running,
# C: or commit headroom below the floor, GPU memory taken or GPU too hot, CUDA not available.
# WARN never stops a run. It reports what only the owner can decide, including anything heavy that is not on the approved list: this script never stops a process it does not name.
#
# Stopped: only what training never needs and that holds no unsaved work. Not touched, whatever they use: this session, drivers, fan and power control (Armoury Crate), security software, browsers.
# Services are stopped, never disabled, so they return at the next restart. A service that needs administrator rights is reported and left running.

param([switch]$StopHogs, [double]$MinDiskGB = 20, [double]$MinHeadroomGB = 6, [int]$MaxGpuMiB = 1500, [int]$MaxGpuC = 80)

$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
$py   = "C:\Users\RAJ\Downloads\Capstone\.venv\Scripts\python.exe"
$logs = Join-Path $repo "results\v5\logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$log  = Join-Path $logs ("preflight_{0}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
$APPROVED_PROCESSES = @("NVIDIA Overlay", "Taskmgr", "nearby_share")
# In stopping order: each database's telemetry service goes before the database.
$APPROVED_SERVICES  = @('SQLTELEMETRY$SQLEXPRESS', 'SQLTELEMETRY$SQLEXPRESS01', 'MSSQL$SQLEXPRESS', 'MSSQL$SQLEXPRESS01', 'SQLWriter', 'MySQL80')
$NEVER_REPORT = @("claude", "Memory Compression", "System", "Idle", "_total", "idle", "system")
$script:fails = 0; $script:warns = 0

function Say($level, $msg) {
    $line = "{0}  {1,-4} {2}" -f (Get-Date -Format "HH:mm:ss"), $level, $msg
    Write-Host $line
    Add-Content -Path $log -Value $line -Encoding utf8
    if ($level -eq "FAIL") { $script:fails++ }
    if ($level -eq "WARN") { $script:warns++ }
}
function Headroom { [math]::Round(((Get-Counter '\Memory\Commit Limit').CounterSamples[0].CookedValue - (Get-Counter '\Memory\Committed Bytes').CounterSamples[0].CookedValue) / 1GB, 1) }
function Idle-Index([string]$setting) {
    $line = powercfg /query SCHEME_CURRENT SUB_SLEEP $setting | Select-String 'Current AC'
    if ($line -match '0x([0-9a-fA-F]+)') { return [Convert]::ToInt32($Matches[1], 16) }
    return -1
}

Say "INFO" ("preflight on {0}" -f $env:COMPUTERNAME)

# ----- stop the approved idle list first, so the checks below measure the machine the run will get
if ($StopHogs) {
    $before = Headroom
    foreach ($name in $APPROVED_PROCESSES) {
        $found = @(Get-Process -Name $name -ErrorAction SilentlyContinue)
        if ($found.Count) {
            $found | Stop-Process -Force -ErrorAction SilentlyContinue
            Say "INFO" ("stopped {0} ({1} process(es))" -f $name, $found.Count)
        }
    }
    foreach ($name in $APPROVED_SERVICES) {
        $service = Get-Service -Name $name -ErrorAction SilentlyContinue
        if ($service -and $service.Status -eq "Running") {
            try {
                Stop-Service -Name $name -Force -ErrorAction Stop
                Say "INFO" ("stopped service {0}; it returns at the next restart, or: Start-Service '{0}'" -f $name)
            } catch {
                Say "WARN" ("could not stop service {0}: needs administrator rights, so it keeps running" -f $name)
            }
        }
    }
    Start-Sleep -Seconds 3
    Say "INFO" ("commit headroom {0} GB before stopping, {1} GB after" -f $before, (Headroom))
}

# ----- anything that would end or block the night
if (Test-Path (Join-Path $repo "results\v5\PAUSE")) { Say "FAIL" "the PAUSE file exists: results\v5\PAUSE" } else { Say "OK" "no PAUSE file" }

$battery = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue | Select-Object -First 1
$mains = (Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus -ErrorAction SilentlyContinue | Select-Object -First 1).PowerOnline
if ($mains -eq $false) { Say "FAIL" "on battery: plug the charger in" } else { Say "OK" ("on mains, battery {0}%" -f $battery.EstimatedChargeRemaining) }
if ($battery -and $battery.EstimatedChargeRemaining -lt 50) { Say "WARN" ("battery {0}%: little reserve if mains drops; on 4 Nov it lasted 55 min from 80% and then hibernated" -f $battery.EstimatedChargeRemaining) }

foreach ($setting in "STANDBYIDLE", "HIBERNATEIDLE") {
    $index = Idle-Index $setting
    if ($index -eq 0) { Say "OK" "$setting on mains is off" } else { Say "FAIL" "$setting on mains is $index seconds: the machine would sleep during the run" }
}
Say "WARN" "closing the lid hibernates this laptop, on mains or battery: keep it open"

$pending = $false
try { $pending = (New-Object -ComObject Microsoft.Update.SystemInfo).RebootRequired } catch { Say "WARN" "could not read Windows Update's restart state" }
if ($pending) { Say "FAIL" "a Windows restart is pending and could happen outside active hours" } else { Say "OK" "no restart pending" }
$pausedUntil = (Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings' -ErrorAction SilentlyContinue).PauseUpdatesExpiryTime
if ($pausedUntil) { Say "OK" "Windows updates paused until $pausedUntil" } else { Say "WARN" "Windows updates are not paused: an update could still need a restart overnight; only the owner should change this" }

$training = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -match 'research\.v\d+\.(train_|predict_)' })
if ($training.Count) { Say "FAIL" ("{0} training or TTA process(es) already running" -f $training.Count) } else { Say "OK" "no other training or TTA process" }

$disk = [math]::Round((Get-PSDrive -Name "C").Free / 1GB, 1)
if ($disk -lt $MinDiskGB) { Say "FAIL" ("C: has {0} GB free, need {1}: a run takes about 7.5 GB more while going, and the pagefile grows into free disk" -f $disk, $MinDiskGB) } else { Say "OK" ("C: {0} GB free (need {1})" -f $disk, $MinDiskGB) }

$room = Headroom
if ($room -lt $MinHeadroomGB) { Say "FAIL" ("commit headroom {0} GB, need {1}" -f $room, $MinHeadroomGB) } else { Say "OK" ("commit headroom {0} GB (need {1}); about 1.2 GB during a run is normal" -f $room, $MinHeadroomGB) }

$gpu = & nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader,nounits 2>$null
if ($gpu) {
    $used, $temp = ($gpu -split ",") | ForEach-Object { [int]$_.Trim() }
    if ($used -gt $MaxGpuMiB) { Say "FAIL" ("GPU memory {0} MiB in use, limit {1}" -f $used, $MaxGpuMiB) } else { Say "OK" ("GPU memory {0} MiB in use" -f $used) }
    if ($temp -gt $MaxGpuC) { Say "FAIL" ("GPU at {0} C, limit {1}" -f $temp, $MaxGpuC) } else { Say "OK" ("GPU at {0} C" -f $temp) }
} else { Say "FAIL" "nvidia-smi gave no answer" }

& $py -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" 2>$null
if ($LASTEXITCODE -ne 0) { Say "FAIL" "the venv's torch cannot see CUDA" } else { Say "OK" "torch sees CUDA" }

# ----- reported, never stopped: anything else heavy is the owner's to decide
Get-Process -ErrorAction SilentlyContinue | Group-Object ProcessName | Where-Object { $NEVER_REPORT -notcontains $_.Name } | ForEach-Object {
    $gb = ($_.Group | Measure-Object PagedMemorySize64 -Sum).Sum / 1GB
    if ($gb -ge 1.5) { Say "WARN" ("{0} holds {1:N1} GB of commit and is not on the approved list; left running" -f $_.Name, $gb) }
}
$samples = (Get-Counter '\Process(*)\% Processor Time' -SampleInterval 2 -MaxSamples 1 -ErrorAction SilentlyContinue).CounterSamples
foreach ($sample in $samples) {
    if ($sample.CookedValue -ge 100 -and $NEVER_REPORT -notcontains $sample.InstanceName) { Say "WARN" ("{0} is using {1:N0}% of one core; left running" -f $sample.InstanceName, $sample.CookedValue) }
}

Say "INFO" ("result: {0} failure(s), {1} warning(s). Log: {2}" -f $script:fails, $script:warns, $log)
if ($script:fails) { exit 1 }
exit 0
