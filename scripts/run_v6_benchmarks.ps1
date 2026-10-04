# V6 candidate benchmarks (docs/V6_RUNSHEET.md section A11.4 step 1): one full-fine-tune step-time and
# peak-VRAM measurement per Tier 1-2 candidate, so no V6 cost is ever quoted unmeasured.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_v6_benchmarks.ps1
#
# * Refuses to start while a train_v4 / train_v5 process holds the GPU (same rule as run_v5_queue C1),
#   or while the V5 queue mutex is held.
# * Skips a configuration whose JSON already exists, so an interrupted run resumes.
# * A failure (e.g. CUDA out of memory) is logged and the loop moves on; the large ViTs are then
#   retried once with --grad-checkpointing.
# * Synthetic batches, no weights downloaded, nothing trained of record.

$ErrorActionPreference = "Continue"
$repo   = Split-Path -Parent $PSScriptRoot
$py     = "C:\Users\RAJ\Downloads\Capstone\.venv\Scripts\python.exe"
$outDir = Join-Path $repo "results\v6\benchmarks"
$log    = Join-Path $outDir "run_v6_benchmarks.log"
New-Item -ItemType Directory -Force $outDir | Out-Null
function Log($m) { $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH.mm.ss"), $m; $line; Add-Content $log $line }

$busy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
          Where-Object { $_.CommandLine -match 'research\.v[45]\.train_v[45]|research\.v5\.infer_last' })
if ($busy.Count -gt 0) { Log "a training / inference process holds the GPU; refusing to start."; exit 3 }
$queue = New-Object System.Threading.Mutex($false, "Global\capstone_v5_queue")
try { $free = $queue.WaitOne(0) } catch [System.Threading.AbandonedMutexException] { $free = $true }
if (-not $free) { Log "the V5 queue is running; refusing to start."; exit 3 }

# tag, size. Tier 1 then Tier 2 (section A11.2), then the ADAE-recipe members (section A12.3). SwinV2 window8 needs 256 px.
$configs = @(
  @("convnext_tiny.fb_in22k_ft_in1k", 224),
  @("eva02_small_patch14_224.mim_in22k", 224),
  @("tf_efficientnetv2_s.in21k_ft_in1k", 224),
  @("caformer_s18.sail_in22k_ft_in1k", 224),
  @("vit_small_patch14_reg4_dinov2.lvd142m", 224),
  @("seresnext50_32x4d.racm_in1k", 224),
  @("edgenext_small.usi_in1k", 224),
  @("maxvit_tiny_tf_224.in1k", 224),
  @("convnextv2_tiny.fcmae_ft_in22k_in1k", 224),
  @("swinv2_tiny_window8_256.ms_in1k", 256),
  @("hiera_small_224.mae_in1k_ft_in1k", 224),
  @("convnext_small.dinov3_lvd1689m", 224),
  @("eva02_base_patch14_224.mim_in22k", 224),
  @("vit_base_patch16_dinov3.lvd1689m", 224),
  @("vit_base_patch16_siglip_224.v2_webli", 224),
  # ADAE-recipe members (V6_RUNSHEET section A12.3), at the higher resolutions the recipe relies on
  @("convnext_tiny.fb_in22k_ft_in1k", 384),
  @("tf_efficientnet_b3.ns_jft_in1k", 384),
  @("tf_efficientnet_b4.ns_jft_in1k", 384),
  @("tf_efficientnet_b5.ns_jft_in1k", 384),
  @("tf_efficientnet_b5.ns_jft_in1k", 456),
  @("seresnext101_32x4d.gluon_in1k", 384),
  @("resnest101e.in1k", 384)
)
$started = Get-Date
Log "=== V6 benchmarks: $($configs.Count) configurations ==="
$k = 0; $ran = 0
foreach ($c in $configs) {
  $k++
  $tag = $c[0]; $size = $c[1]
  $json = Join-Path $outDir ("{0}_{1}.json" -f $tag, $size)
  if (Test-Path $json) { Log ("[{0}/{1}] skip {2} {3} (exists)" -f $k, $configs.Count, $tag, $size); continue }
  # Progress counter: ETA from the measured mean time of the configurations run so far in this session.
  $el = ((Get-Date) - $started).TotalMinutes
  $eta = if ($ran -gt 0) { " | ETA " + (Get-Date).AddMinutes($el / $ran * ($configs.Count - $k + 1)).ToString("HH\:mm") + " (measured mean {0:N1} min/config)" -f ($el / $ran) } else { " | ETA after the first configuration" }
  Log ("[{0}/{1}] --- {2} @ {3} | elapsed {4:N1} min{5}" -f $k, $configs.Count, $tag, $size, $el, $eta)
  $ran++
  & $py (Join-Path $repo "scripts\gpu_benchmark.py") --timm --arch $tag --sizes $size --batch 32 `
      --images 15294 --epochs 30 --workers 2 --json-out $json
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path $json)) {
    $ck = Join-Path $outDir ("{0}_{1}_gradckpt.json" -f $tag, $size)
    Log "FAILED $tag $size (exit $LASTEXITCODE); retrying with --grad-checkpointing"
    & $py (Join-Path $repo "scripts\gpu_benchmark.py") --timm --grad-checkpointing --arch $tag --sizes $size `
        --batch 32 --images 15294 --epochs 30 --workers 2 --json-out $ck
    if ($LASTEXITCODE -ne 0) { Log "FAILED again $tag $size" }
  }
}
Log ("=== done in {0:N1} min ===" -f ((Get-Date) - $started).TotalMinutes)
try { $queue.ReleaseMutex() } catch { }
