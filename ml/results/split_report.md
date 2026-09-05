# Train / Validation / Test Split

**Generated:** 2026-08-12  
**Seed:** 42  
**Split file:** `ml/configs/splits/split_combined.csv`  
**Requested lesion fractions:** train 70% / val 15% / test 15%

## Method

Images are **not** split directly. HAM10000 contains multiple photographs of the same
physical lesion, identified by `lesion_id`. The split is performed over lesions,
stratified by diagnosis, and every image inherits its lesion's split. This guarantees
that no two images of the same lesion can appear on both sides of the train/test
boundary, so the test score measures generalisation to unseen lesions.

This is verified programmatically after assignment (`assert_no_leakage`), not assumed.

## Sizes

| Split | Lesions | Images | Image share |
|---|---:|---:|---:|
| train | 6,451 | 8,454 | 69.7% |
| val | 1,382 | 1,864 | 15.4% |
| test | 1,383 | 1,803 | 14.9% |

> Image shares deviate slightly from the requested lesion fractions because lesions
> contribute different numbers of images. This is expected and correct -- the alternative
> (forcing exact image proportions) would require breaking lesion groups.

## Class distribution per split

| Class | train | val | test | total |
|---|---:|---:|---:|---:|
| `akiec` | 731 | 161 | 165 | 1,057 |
| `bcc` | 945 | 217 | 197 | 1,359 |
| `bkl` | 923 | 205 | 206 | 1,334 |
| `df` | 84 | 15 | 16 | 115 |
| `mel` | 818 | 180 | 167 | 1,165 |
| `nv` | 4,852 | 1,068 | 1,029 | 6,949 |
| `vasc` | 101 | 18 | 23 | 142 |
| **total** | 8,454 | 1,864 | 1,803 | 12,121 |

## Reproducibility

The split is committed to version control as a CSV of `image_id, lesion_id, class_code, split`. Training reads that file rather than
re-deriving the split, so a future scikit-learn release cannot silently change which
images are in the test set.
