# Progress counter for the overnight chain (control scoring -> Q7 -> Q7b by default). READ-ONLY: it reads the
# queue logs and the live tqdm lines in each run's .err.log; it never touches the GPU, the queue or a process.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\progress.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\progress.ps1 -Watch 60      # refresh every 60 s
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\progress.ps1 -Queues Q8 -Minutes 71.5 -NoScoring
#   (-Queues / -Minutes take comma-separated lists, e.g. -Queues Q7,Q7b -Minutes 71.5,39.8)
#
# ETAs are MEASURED wherever possible: the current run's ETA comes from its own elapsed time and the fraction of
# batches done; a queue's remaining runs use the mean of that queue's DONE durations. Only before a queue has
# finished any run does it fall back to -Minutes (defaults: Q7 71.5 = Q6's measured composite runs; Q7b 39.8 =
# m4's measured fold-0 runs), and the line says "est".

param(
  [string]$Queues = "Q7,Q7b",
  [string]$Minutes = "71.5,39.8",
  [switch]$NoScoring,
  [int]$Watch = 0
)
# Comma-separated so they survive `powershell -File` (which passes arrays as one string).
$QueueList  = @($Queues -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$MinuteList = @($Minutes -split "," | ForEach-Object { [double]::Parse($_.Trim(), [System.Globalization.CultureInfo]::InvariantCulture) })

$repo   = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $repo "results\v5\logs"
$preds  = Join-Path $repo "results\v5\preds"
$scoringFiles = 1..4 | ForEach-Object { Join-Path $preds ("R0_kfold_f{0}_s42_last.csv" -f $_) }

function Read-Tail([string]$path, [int]$bytes = 8192) {
  try {
    $fs = [System.IO.File]::Open($path, 'Open', 'Read', 'ReadWrite')
    try {
      $len = $fs.Length; $start = [Math]::Max(0, $len - $bytes)
      $null = $fs.Seek($start, 'Begin')
      $buf = New-Object byte[] ($len - $start)
      $null = $fs.Read($buf, 0, $buf.Length)
      return [System.Text.Encoding]::UTF8.GetString($buf)
    } finally { $fs.Close() }
  } catch { return "" }
}

function Fmt-Time([datetime]$t) {
  $s = $t.ToString("HH\:mm")
  if ($t.Date -ne (Get-Date).Date) { $s += " (" + $t.ToString("ddd") + ")" }
  return $s
}

function Bar([double]$f, [int]$w = 20) {
  $f = [Math]::Max([double]0, [Math]::Min([double]1, [double]$f)); $n = [int][Math]::Floor($f * $w + 0.5)
  return "[" + ("#" * $n) + ("." * ($w - $n)) + "]"
}

# State of one queue from its log: total, finished, failed, measured durations, current run id.
function Get-QueueState([string]$q) {
  $log = Join-Path $logDir ("queue_{0}.log" -f $q)
  $st = @{ total = 0; already = 0; done = 0; failed = 0; durations = @(); current = $null; started = $false; finished = $false }
  if (-not (Test-Path $log)) { return $st }
  $lines = @(Get-Content $log)
  $hdr = -1
  for ($i = $lines.Count - 1; $i -ge 0; $i--) {
    if ($lines[$i] -match "=== queue $([regex]::Escape($q)): (\d+) runs, (\d+) already complete ===") { $hdr = $i; break }
  }
  if ($hdr -lt 0) { return $st }
  $st.total = [int]$Matches[1]; $st.already = [int]$Matches[2]
  $open = $null
  for ($i = $hdr + 1; $i -lt $lines.Count; $i++) {
    $l = $lines[$i]
    if ($l -match "--- (\S+)\s+\(headroom") { $open = $Matches[1]; $st.started = $true }
    elseif ($l -match "(\S+) DONE in ([\d.]+) min") { $st.done++; $st.durations += [double]$Matches[2]; $open = $null }
    elseif ($l -match "(\S+) FAILED again") { $st.failed++; $open = $null }
    elseif ($l -match "=== queue .* finished") { $st.finished = $true; $open = $null }
  }
  $st.current = $open
  return $st
}

# Live position of one run from the newest .err.log tqdm fragment.
function Get-RunState([string]$runId) {
  $err = Get-ChildItem $logDir -Filter ("{0}.*.err.log" -f $runId) -ErrorAction SilentlyContinue |
         Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if (-not $err) { return $null }
  $start = $err.CreationTime
  if ($err.Name -match "\.(\d{8}-\d{6})\.err\.log$") {
    $start = [datetime]::ParseExact($Matches[1], "yyyyMMdd-HHmmss", $null)
  }
  $frags = (Read-Tail $err.FullName) -split "[`r`n]"
  $hit = $null
  for ($i = $frags.Count - 1; $i -ge 0; $i--) {
    if ($frags[$i] -match "epoch\s+(\d+)/(\d+)\s+\[(\w+)\s*\]:\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)") { $hit = $Matches; break }
  }
  $r = @{ start = $start; epoch = 0; epochs = 30; stage = "setup"; frac = 0.0 }
  if ($hit) {
    $r.epoch = [int]$hit[1]; $r.epochs = [int]$hit[2]; $r.stage = $hit[3]
    $r.frac = (($r.epoch - 1) + [double]$hit[4] / [double]$hit[5]) / $r.epochs
  }
  return $r
}

function Show-Progress {
  $now = Get-Date
  $out = New-Object System.Collections.Generic.List[string]
  $cursor = $now          # when the next not-yet-running step can start
  $totalMin = 0.0; $doneMin = 0.0

  # Step 0: control scoring (infer_last), measured 22 s per checkpoint on 30 Sep.
  if (-not $NoScoring) {
    $have = @($scoringFiles | Where-Object { Test-Path $_ }).Count
    $busy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
              Where-Object { $_.CommandLine -match 'research\.v5\.infer_last' }).Count -gt 0
    $totalMin += 2; $doneMin += 2 * $have / 4
    if ($have -eq 4) { $out.Add("  [done] control scoring      4/4 files") }
    elseif ($busy)   { $out.Add("  [run ] control scoring      $have/4 files (~22 s each, measured)"); $cursor = $now.AddMinutes(2 * (4 - $have) / 4) }
    else             { $out.Add("  [wait] control scoring      $have/4 files - NOT STARTED"); $cursor = $now.AddMinutes(2) }
  }

  for ($qi = 0; $qi -lt $QueueList.Count; $qi++) {
    $q = $QueueList[$qi]
    $est = $MinuteList[[Math]::Min($qi, $MinuteList.Count - 1)]
    $st = Get-QueueState $q
    $n = if ($st.total -gt 0) { $st.total } else {
      @(Get-Content (Join-Path $repo ("results\v5\queue_{0}.txt" -f $q)) -ErrorAction SilentlyContinue |
        Where-Object { $_.Trim() -and -not $_.Trim().StartsWith("#") }).Count }
    $per = $est; $tag = "est"
    if ($st.durations.Count -gt 0) { $per = ($st.durations | Measure-Object -Average).Average; $tag = "measured" }
    $finished = $st.already + $st.done + $st.failed
    $totalMin += $n * $per; $doneMin += $finished * $per

    if (-not $st.started) {
      $eta = $cursor.AddMinutes($n * $per)
      $out.Add(("  [wait] {0,-5} 0/{1} runs  {2}  starts ~{3}, ends ~{4}  ({5:N1} min/run {6})" -f $q, $n, (Bar 0), (Fmt-Time $cursor), (Fmt-Time $eta), $per, $tag))
      $cursor = $eta; continue
    }
    if ($st.finished -or -not $st.current) {
      $msg = if ($st.failed) { "  FAILED: $($st.failed)" } else { "" }
      $out.Add(("  [done] {0,-5} {1}/{2} runs  {3}  {4:N1} min/run measured{5}" -f $q, $finished, $n, (Bar 1), $per, $msg))
      continue
    }
    $run = Get-RunState $st.current
    $frac = 0.0; $runEta = $now.AddMinutes($per)
    if ($run) {
      $frac = $run.frac
      $elapsed = ($now - $run.start).TotalMinutes
      if ($frac -ge 0.05) { $runEta = $run.start.AddMinutes($elapsed / $frac) } else { $runEta = $run.start.AddMinutes($per) }
    }
    $doneMin += $frac * $per
    $left = [Math]::Max(0, $n - $finished - 1)
    $qEta = $runEta.AddMinutes($left * $per)
    $out.Add(("  [run ] {0,-5} {1}/{2} runs  {3}  queue ends ~{4}  ({5:N1} min/run {6})" -f $q, $finished, $n, (Bar (($finished + $frac) / [Math]::Max(1, $n))), (Fmt-Time $qEta), $per, $tag))
    if ($run) {
      $out.Add(("         now: {0}  epoch {1}/{2} [{3}]  {4} {5:P0}  run ends ~{6}" -f $st.current, $run.epoch, $run.epochs, $run.stage, (Bar $frac 12), $frac, (Fmt-Time $runEta)))
    }
    $cursor = $qEta
  }

  $chain = if ($totalMin -gt 0) { $doneMin / $totalMin } else { 0 }
  Write-Host ("{0}  chain {1}  {2:P0}  ->  all done ~{3}" -f $now.ToString("HH\:mm\:ss"), (Bar $chain 30), $chain, (Fmt-Time $cursor))
  $out | ForEach-Object { Write-Host $_ }
}

if ($Watch -gt 0) {
  while ($true) { Clear-Host; Show-Progress; Start-Sleep -Seconds $Watch }
} else { Show-Progress }
