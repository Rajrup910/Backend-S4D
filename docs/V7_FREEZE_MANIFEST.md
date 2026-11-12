# V7 Freeze Manifest (candidate, 12 Nov 2026; gate rebuilt 13 Nov 02:00)

**Status: CANDIDATE.** It becomes the freeze record only when (1) the working tree below is committed and (2) the owner
signs Gate 0. Until then the hashes describe the files on disk, not a commit. Verdict and reasoning: `docs/V7_AUDIT_REPORT.md`.

## 1. Code state

| Item | Value |
|---|---|
| Branch | `v5-confirmation-review2` |
| HEAD at audit | `5f17939` (12 Nov audit commit); the 13 Nov fixes go in the freeze commit, still to be made |
| Working tree | the 13 Nov fixes (gate builder, promotion P2, tests, docs, rebuilt receipts), uncommitted until the owner says yes |
| Frozen / protected file lists | stored in the Gate 0 record at signing (`research/v7/plan_freeze.py`); not yet written |
| Gate 0 | unsigned |
| Owner decisions | plan §13.5, 24 of 24 written on 12 Nov 2026 |

## 2. Environment

| Item | Value |
|---|---|
| Python | 3.12.10 (`C:\Users\RAJ\Downloads\Capstone\.venv`, junction to D:) |
| OS | Windows 11 Home, build 26200 |
| torch / torchvision / timm | 2.11.0+cu128 / 0.26.0+cu128 / 1.0.28 |
| CUDA / cuDNN | 12.8 / 9.19.0 (91900) |
| numpy / pandas / scikit-learn / scipy / Pillow / OpenCV | 2.4.4 / 3.0.5 / 1.9.0 / 1.18.0 / 12.2.0 / 5.0.0 |
| Full package list | `results/v7/audit_2026-11-12/raw/pip_freeze.txt` (87 lines; no lock file exists) |
| GPU | NVIDIA RTX 5050 Laptop, 8.55 GB, sm_120, driver 610.62 |
| RAM / commit | 15.2 GB / about 7.5 GB commit headroom (measured 16 Sep 2026); `--num-workers 2` |
| Disk at audit | C: 27 GB free; D: 64 GB free |

## 3. Data (SHA-256; full list `results/v7/audit_2026-11-12/raw/sha256_freeze_candidate.txt`)

| File | SHA-256 | Note |
|---|---|---|
| `ml/data/v7/master_v7.csv` | `65f6ea39…efd1d7` | 57,488 rows; matches receipt |
| `ml/data/v7/links_v7.csv` | `9b4aa6d3…72bab6` | 27,162 confirmed, 137,170 at-risk |
| `results/v7/data_gate_receipt.json` | `71e956a1…404a1e` | written 13 Nov 02:00 (patient ids loaded, audit A11), passed |
| `ml/data/manifest_v4.csv` | `43be6392…f8c2f9` | labels, ages, S71 folds |
| `ml/configs/class_mapping.json` | `acd01d00…7d05` | label map |
| `results/v7/added_pool_v7/added_train_v7.csv` | `cfe48f19…a4b2` | V7 pool, 35,183 rows |
| `results/v7/hiba_train/hiba_train.csv` | `3154c997…620b39` | HIBA, training only |
| `results/v5/young_data/extra_train.csv` | `e6e0a84a…1ce` | young extras, 2,078 |
| `results/v7/v6_reference_manifest.json` | `cb965179…7f2811` | banked V6 comparator (checkpoints + fold 1–4 TTA) |
| MIDAS labels | `6bdb6f40…aa` (`data_roles_v7.json`) | sealed; never read |

**Roles:**
- Development: 15,294 images in S71 folds 0–4 (3,059 / 3,059 / 3,059 / 3,059 / 3,058).
- Fold 0 is used to select.
- Folds 1–4 (12,235 images / 6,989 groups) are read once against the banked V6 vote.
- Training-only: added pool 25,471; extras 2,078; HIBA 1,199; `reserved` 4,733; `val` 2,270; `ham_val` 1,532.
- Never trained on: `ham_test` (1,502).
- Sealed external: MIDAS, 765 primary images.

**Gate removals:** 361 rows from every run, plus 476/413/437/375/457 for folds 0–4, and 9,616/2,097/1,559 for the
BCN/HAM/MSKCC holdouts.

## 4. Model and recipe (first night: fold-0 screens)

**Common recipe** (V6, through `research/v6/train_v6.py` and the V4 constants):
- **Two stages:** a 3-epoch frozen-backbone head at LR 1e-3, then full fine-tune at base LR 1e-4.
- **Optimiser and loss:** AdamW with weight decay 1e-4 (anchor default); cosine schedule in ticks of one V5-pool epoch,
  30 epochs nominal, cut at the step budget.
- **Regularisation:** label smoothing 0.05; class weights by effective number (β 0.999); gradient clip 5.0; AMP; SWA
  (`_swad`) and `_last` checkpoints.
- **Step budget:** held at V6's (decision 7: 13,410 optimiser steps on fold 0).
- **Train transform:** RandomResizedCrop (0.8–1.0) + horizontal and vertical flips + ±20° rotation + ColorJitter
  (0.15/0.15/0.10/0.02).
- **Eval transform:** Resize + CenterCrop.
- **Normalisation:** per member (`research/v6/members.py`).
- **Seed and workers:** seed 42; `--num-workers 2`.

| Build | Member | Size | Change vs V6 | Time (basis) |
|---|---|---|---|---|
| R1 | ConvNeXt-T IN-22k anchor | 224 | AdamW weight decay 0.05 | 55.6 min (MEASURED, V6 member) |
| B3 | anchor | 384, batch 16 × accum 2 | CE + one-way pAUC surrogate at FPR ≤ 0.2 | 88.8 min (MEASURED) |
| B5 | anchor | 224 | lesion-only melanin scaling k ∈ [1/2, 2], log-uniform, p 0.5 | about 66 min (EXTRAPOLATED) |
| R2 | DINOv2-B/14 | 224 | layer decay 0.75, top LR 4e-5; needs an R4 receipt | 78.4 min (MEASURED) |
| R3 | PanDerm-B | 224 | layer decay 0.80; needs an R4 receipt | 64.5 min (MEASURED) |
| R3b | PanDerm-B | 224 | unscaled LR 5e-4 (source `UNVERIFIED`), decay 0.65; needs an R4 receipt | 64.5 min (MEASURED) |

**Selection:** fold 0 only, by each build's drop rule (`research/v7/builds.py`). Folds 1–4 are read only after a
candidate is frozen (P1–P3, plan §14.3).

## 5. Outputs and naming

- **Run id:** `{build}_{member_label}_f0_s42_v70` (`--run-tag v70`). Summaries go to `results/v7/runs/`, predictions to
  `results/v7/preds/`, checkpoints to `ml/checkpoints/v7/`.
- **Overwrites:** a finished summary refuses a second write, and the chain skips any finished run.
- **Logs:** `results/v7/logs/first_night/` (`chain.log` and one log per step).
- **Pause:** create `results/v7/PAUSE` to stop the chain before its next step.

## 6. Exact commands, in run order (owner runs them; PowerShell, repo root)

The "Ends" column assumes a 20:00 start; the durations are what carry over to any start time. **Measured** means from file timestamps or V6 run logs.

| # | Step | Command | Time | Ends (from 20:00) | Basis | Must finish before next |
|:-:|---|---|---|---|---|---|
| 1 | Commit the tree | owner's yes, then `git add` / `git commit` (no AI attribution) | 3 min | 20:03 | — | yes |
| 2 | Sign Gate 0 | `& $py -m research.v7.plan_freeze --sign --owner-approved --owner "NAME"`, then `--status` (must print "signed and unchanged") | 2 min | 20:05 | — | yes |
| 3 | Rehearsal of the chain | `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_v7_first_night.ps1 -DryRun` | 1 min | 20:06 | MEASURED (12 Nov 19:33) | yes |
| 4 | The night | `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_v7_first_night.ps1` (launch via WMI outside the app's container, per the overnight routine) | about 7.3 h: gate + verifier 2 min, preflight 2 min, six smokes about 4 min, three R4 checks about 12 min, six screens 418 min | about 03:25 | smokes MEASURED (12 Nov 17:47–17:50); R4 MEASURED as rehearsals (11 Nov, about 4 min each); screens MEASURED per V6 member, B5 EXTRAPOLATED | one GPU job at a time; nothing else on the GPU |
| 5 | Morning read | read fold 0; apply each drop rule | 30 min | — | — | before any folds 1–4 run |

`$py = "C:\Users\RAJ\Downloads\Capstone\.venv\Scripts\python.exe"`. **The step-by-step commands the chain runs** are in
`results/v7/freeze_audit/AUDIT_REPORT.md` §J and `results/v7/audit_2026-11-12/raw/first_night_dryrun.txt`.

**What stays the owner's:**
- Windows Update deferred;
- the lid open and mains connected;
- C: at 5 GB free or more (27 GB now).

## 7. Known limitations carried into the run

- Re-photograph guard about 0.67 of same-lesion pairs where IDs are hidden; no patient ids for HAM or BCN.
- M1 over-merges at least about 7% of cross-lesion merges. This is conservative for leakage.
- Folds 1–4 were read at V6-5 and mined for V7's hypotheses (audit A2). A subgroup gain there is exploratory.
- MIDAS is a prospectively recruited, two-site (Stanford, Cleveland Clinic) general-dermatology cohort. All ages are
  included, and there are 69 under-40 images. It cannot confirm an under-40 result. MIDAS also holds non-biopsied control
  lesions. Our 765-image read keeps biopsied lesions only, by the code's rule (`iscontrol == no`,
  `research/v7/midas_prepare.py`), so its negatives are biopsied benign lesions: a harder and less typical set than a
  screening population.
- One seed per member. Seed noise was measured for the anchor only (pair SD 0.00824 at V6-5). P1's bar sits at the
  one-sided 80th percentile of that noise, so pure seed noise passes it 20% of the time by construction.
- PanDerm and DINOv2 pretraining corpora are unauditable. PanDerm weights are CC-BY-NC-ND. PanDerm's public repository lists
  HIBA among its benchmark datasets, so HIBA was in PanDerm's development loop. HIBA is training-only in V7, but V6's HIBA
  read (0.7348) is not fully external for the PanDerm member.
- Patient IDs reach the gate only where an archive publishes them: all of ISIC 2020, DERM12345 and HIBA, 1,039 of 2,078
  young extras and 417 development rows. HAM, BCN and MILK10k publish none.
- No dependency lock file.
