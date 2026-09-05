# Machine Learning

Dataset preparation, training, evaluation, model comparison and Grad-CAM.

Nothing here is imported by the API server at runtime — the backend loads a checkpoint and
`configs/class_mapping.json`, and nothing else.

## Layout

```
configs/
├── class_mapping.json      7 classes, translations, malignancy tiers — SINGLE SOURCE OF TRUTH
├── training_config.yaml    every hyperparameter
└── splits/split_v1.csv     the committed, reproducible split
preprocessing/              prepare · validate · split · transforms · dataset
training/                   common.py (factory, weights, checkpoints) · train.py
evaluation/                 metrics · plots · evaluate · compare_models
explainability/             gradcam · generate_gradcam
checkpoints/                local weights (git-ignored)
results/                    metrics, figures, reports (git-ignored)
```

## Order of operations

```bash
.venv/Scripts/python.exe -m ml.preprocessing.prepare_dataset
```

```bash
.venv/Scripts/python.exe -m ml.preprocessing.validate_dataset
```

```bash
.venv/Scripts/python.exe -m ml.preprocessing.split_dataset
```

```bash
.venv/Scripts/python.exe -m ml.training.train --arch resnet50
```

```bash
.venv/Scripts/python.exe -m ml.training.train --arch efficientnet_b0
```

```bash
.venv/Scripts/python.exe -m ml.evaluation.evaluate --checkpoint ml/checkpoints/resnet50_best.pt --calibrate
```

```bash
.venv/Scripts/python.exe -m ml.evaluation.compare_models
```

```bash
.venv/Scripts/python.exe -m ml.explainability.generate_gradcam --checkpoint ml/checkpoints/efficientnet_b0_best.pt
```

Every script takes `--help` and every default comes from `configs/training_config.yaml`.

## Four rules this code enforces

**1. The split is grouped by `lesion_id`.** HAM10000 has multiple photographs of the same
lesion. Splitting at random puts near-identical images in both train and test and inflates
the result. `assert_no_leakage()` raises if any lesion spans a boundary — and a test asserts
that the guard itself fails when it should. See [`docs/dataset.md`](../docs/dataset.md).

**2. The test set is touched once.** Checkpoints are selected on **validation macro F1**,
chosen before any test evaluation. Repeatedly checking the test set turns it into a second
validation set and the final number stops meaning anything.

**3. Accuracy is never the headline.** `nv` is two thirds of the data; answering "mole"
every time scores ~67% and misses every melanoma. Macro F1, balanced accuracy, per-class
recall and sensitivity to escalating classes are what get reported.

**4. Nothing is fabricated.** Every number in the report comes from `results/`. If the model
achieves 82%, report 82%.

## Training is resumable

```bash
.venv/Scripts/python.exe -m ml.training.train --arch resnet50 --resume
```

`<arch>_last.pt` is written every epoch, `<arch>_best.pt` only on improvement. A Windows
update at 3am costs one epoch, not the run.

## Class weighting

Default is effective-number weighting (Cui et al., CVPR 2019), not inverse frequency. On
HAM10000-like counts inverse frequency spans ~58× and destabilises training; effective
number spans ~9×. Switch with `class_weighting: none | inverse | effective_number` in the
config — the ablation is worth reporting.

## Checkpoints carry their provenance

Each records the architecture, class codes, class-mapping version, image size, epoch,
selection metric and full config. Loading refuses if the class codes disagree with the
current mapping, because serving predictions where index 4 means `mel` to the model and
something else to the API is exactly the kind of silent error that matters here.

## Experiment log

Every run appends to `results/experiments.csv` (plus a JSONL sidecar): seed, hyperparameters,
augmentation, metrics, timings, device, checkpoint path.

## Tests

```bash
.venv/Scripts/python.exe -m pytest tests/ -v
```

Runs without the dataset. Covers split integrity and leakage detection, class weighting,
model construction and freezing, transforms, metrics, and `ml`↔`backend` parity.
