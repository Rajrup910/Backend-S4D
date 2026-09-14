# S36 -- unattended Track B pipeline: train N3/N4/N5, then score them against the frozen
# criterion. Every component here was tested in S36; nothing in this script is new logic.
#
#   powershell -ExecutionPolicy Bypass -File research\v2\run_track_b_overnight.ps1
#
# Roughly one hour on the RTX 5050 (25 min training + ~30 min evaluation). Writes a
# timestamped log to results\v2\ and prints a summary table at the end.
#
# Deliberately does NOT stop on the first failure: if one arm dies, the other two are still
# worth having, and eval_arms skips checkpoints that do not exist rather than crashing. The
# summary at the end reports each step's exit code so a partial night is legible in the
# morning.

$ErrorActionPreference = "Continue"

$py   = "C:\Users\RAJ\Downloads\Capstone\.venv\Scripts\python.exe"
$repo = "C:\Users\RAJ\Downloads\Capstone review 1"
Set-Location $repo

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log   = Join-Path $repo "results\v2\track_b_run_$stamp.log"
$steps = New-Object System.Collections.ArrayList

function Invoke-Step {
    param([string]$Name, [string[]]$Arguments)

    $started = Get-Date
    $banner = "`n=== $Name  [started $($started.ToString('HH:mm:ss'))] ==="
    Write-Host $banner
    Add-Content -Path $log -Value $banner -Encoding utf8

    # NO `2>&1` here. In PowerShell 5.1 redirecting a native executable's stderr wraps every
    # line in a NativeCommandError ErrorRecord -- and tqdm writes its progress bars to stderr,
    # so the first progress update kills the pipeline even though python is perfectly healthy.
    # stdout (the per-epoch summary lines) is what belongs in the log; the progress bars stay
    # on the console where they are useful and out of the log where they are noise.
    & $py @Arguments | Tee-Object -FilePath $log -Append
    $code = $LASTEXITCODE
    $elapsed = [math]::Round(((Get-Date) - $started).TotalMinutes, 1)

    $line = "--- $Name finished: exit $code after $elapsed min ---"
    Write-Host $line
    Add-Content -Path $log -Value $line -Encoding utf8
    [void]$steps.Add([pscustomobject]@{ Step = $Name; Exit = $code; Minutes = $elapsed })
}

Add-Content -Path $log -Value "Track B overnight run started $(Get-Date)" -Encoding utf8

# Keep Windows awake for the duration. Without this a sleep timer will kill the run
# mid-epoch and the morning's result is a truncated log.
try {
    powercfg /change standby-timeout-ac 0
    powercfg /change monitor-timeout-ac 0
    Write-Host "Sleep disabled on AC power for the duration of this run."
} catch {
    Write-Host "WARNING: could not change power settings; make sure the machine will not sleep."
}

$frozen = "ml/checkpoints/convnext_tiny_best.HAM-only.pt"

Invoke-Step "gate: frozen checkpoints (before)" @("-m", "research.v2.frozen_checkpoints", "--check")

Invoke-Step "N3 budget-constrained ranking" @(
    "-m", "research.v2.train_escalation", "--loss", "n3_pauc", "--arch", "convnext_tiny",
    "--init-weights", $frozen, "--epochs", "12", "--batch-size", "64",
    "--alpha", "0.20", "--finetune-lr", "2e-5", "--num-workers", "4")

Invoke-Step "N4 worst-band group DRO" @(
    "-m", "research.v2.train_escalation", "--loss", "n4_groupdro", "--arch", "convnext_tiny",
    "--init-weights", $frozen, "--epochs", "12", "--batch-size", "64",
    "--alpha", "0.20", "--eta-q", "0.01", "--finetune-lr", "2e-5", "--num-workers", "4")

Invoke-Step "N5 band-conditional logit adjustment" @(
    "-m", "research.v2.train_escalation", "--loss", "n5_logitadj", "--arch", "convnext_tiny",
    "--init-weights", $frozen, "--epochs", "12", "--batch-size", "64",
    "--t-adjust", "1.0", "--finetune-lr", "2e-5", "--num-workers", "4")

Invoke-Step "gate: frozen checkpoints (after training)" @("-m", "research.v2.frozen_checkpoints", "--check")

# val MUST precede bcn: the BCN stage reads the val deltas back to judge criterion 5.
Invoke-Step "eval: HAM val" @("-m", "research.v2.eval_arms", "--stage", "val", "--num-workers", "4")
Invoke-Step "eval: BCN-20000" @("-m", "research.v2.eval_arms", "--stage", "bcn", "--num-workers", "4")

Invoke-Step "gate: frozen checkpoints (final)" @("-m", "research.v2.frozen_checkpoints", "--check")

$summary = $steps | Format-Table -AutoSize | Out-String
Write-Host "`n================ SUMMARY ================"
Write-Host $summary
Add-Content -Path $log -Value "`n================ SUMMARY ================" -Encoding utf8
Add-Content -Path $log -Value $summary -Encoding utf8

$failed = @($steps | Where-Object { $_.Exit -ne 0 })
if ($failed.Count -eq 0) {
    $verdict = "ALL STEPS OK. Read results\v2\arm_criterion.json and results\v2\arm_results.csv."
} else {
    $verdict = "$($failed.Count) STEP(S) FAILED: $(($failed | ForEach-Object { $_.Step }) -join '; '). See $log."
}
Write-Host $verdict
Add-Content -Path $log -Value $verdict -Encoding utf8
Write-Host "Full log: $log"
