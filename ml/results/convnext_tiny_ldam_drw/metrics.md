# Evaluation — convnext_tiny

**Checkpoint:** `ml/checkpoints/convnext_tiny-ldam_drw_best.pt`  
**Split:** test (1,502 images)  
**Selected on:** validation macro_f1 = 0.7385 (epoch 27)  
**Class mapping:** v1.0.0  
**Temperature:** 1.000 (uncalibrated)

## Headline metrics

| Metric | Value |
|---|---:|
| Macro F1 | **0.7256** |
| Balanced accuracy | **0.7063** |
| Weighted F1 | 0.8367 |
| Accuracy | 0.8402 |
| Cohen's kappa | 0.6877 |
| Macro ROC-AUC (OvR) | 0.9625 |
| Expected calibration error | 0.2102 |

> Accuracy is listed but is not the headline. `nv` is roughly two thirds of this
> dataset, so a model that answers "mole" every time scores around 0.67 accuracy
> while missing every melanoma. Macro F1 and per-class recall are the numbers that
> describe whether this model is useful.

## Per-class performance

| Class | Malignancy | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| `akiec` Actinic keratosis | premalignant | 0.5970 | **0.7692** | 0.6723 | 52 |
| `bcc` Basal cell carcinoma | malignant | 0.7284 | **0.8310** | 0.7763 | 71 |
| `bkl` Benign keratosis | benign | 0.7559 | **0.5749** | 0.6531 | 167 |
| `df` Dermatofibroma | benign | 0.7857 | **0.5500** | 0.6471 | 20 |
| `mel` Melanoma | malignant | 0.6519 | **0.6168** | 0.6338 | 167 |
| `nv` Mole | benign | 0.9029 | **0.9353** | 0.9188 | 1,004 |
| `vasc` Vascular lesion | benign | 0.9333 | **0.6667** | 0.7778 | 21 |

## Screening view

Collapsing the seven classes into "needs escalation" (`akiec`, `bcc`, `mel`) versus not:

| Metric | Value |
|---|---:|
| Sensitivity | **0.7310** |
| Specificity | 0.9224 |
| Missed serious cases (false negatives) | **78** |
| False alarms | 94 |

> In a screening tool the false negatives in that table are the number that matters.
> A missed melanoma is a materially worse outcome than an unnecessary referral, and
> the report should discuss this asymmetry rather than averaging it away.

## Inference latency

| Metric | Value |
|---|---:|
| Mean | 4.8 ms |
| Median | 4.8 ms |
| p95 | 4.9 ms |
| Device | cuda |
| Parameters | 27,825,511 |
| Model size | 106.1 MB |

## Classification report

```
              precision    recall  f1-score   support

       akiec     0.5970    0.7692    0.6723        52
         bcc     0.7284    0.8310    0.7763        71
         bkl     0.7559    0.5749    0.6531       167
          df     0.7857    0.5500    0.6471        20
         mel     0.6519    0.6168    0.6338       167
          nv     0.9029    0.9353    0.9188      1004
        vasc     0.9333    0.6667    0.7778        21

    accuracy                         0.8402      1502
   macro avg     0.7650    0.7063    0.7256      1502
weighted avg     0.8387    0.8402    0.8367      1502
```

## Figures

- `confusion_matrix.png`
- `reliability_diagram.png`
- `training_curves.png`
