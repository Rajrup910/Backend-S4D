# Evaluation — convnext_small

**Checkpoint:** `ml/checkpoints/convnext_small_best.pt`  
**Split:** test (1,502 images)  
**Selected on:** validation macro_f1 = 0.7435 (epoch 9)  
**Class mapping:** v1.0.0  
**Temperature:** 0.853

## Headline metrics

| Metric | Value |
|---|---:|
| Macro F1 | **0.7250** |
| Balanced accuracy | **0.7281** |
| Weighted F1 | 0.8448 |
| Accuracy | 0.8469 |
| Cohen's kappa | 0.7023 |
| Macro ROC-AUC (OvR) | 0.9447 |
| Expected calibration error | 0.0613 |

> Accuracy is listed but is not the headline. `nv` is roughly two thirds of this
> dataset, so a model that answers "mole" every time scores around 0.67 accuracy
> while missing every melanoma. Macro F1 and per-class recall are the numbers that
> describe whether this model is useful.

## Per-class performance

| Class | Malignancy | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| `akiec` Actinic keratosis | premalignant | 0.7273 | **0.7692** | 0.7477 | 52 |
| `bcc` Basal cell carcinoma | malignant | 0.7353 | **0.7042** | 0.7194 | 71 |
| `bkl` Benign keratosis | benign | 0.7115 | **0.6647** | 0.6873 | 167 |
| `df` Dermatofibroma | benign | 0.4839 | **0.7500** | 0.5882 | 20 |
| `mel` Melanoma | malignant | 0.7063 | **0.6048** | 0.6516 | 167 |
| `nv` Mole | benign | 0.9109 | **0.9373** | 0.9239 | 1,004 |
| `vasc` Vascular lesion | benign | 0.8750 | **0.6667** | 0.7568 | 21 |

## Screening view

Collapsing the seven classes into "needs escalation" (`akiec`, `bcc`, `mel`) versus not:

| Metric | Value |
|---|---:|
| Sensitivity | **0.7138** |
| Specificity | 0.9513 |
| Missed serious cases (false negatives) | **83** |
| False alarms | 59 |

> In a screening tool the false negatives in that table are the number that matters.
> A missed melanoma is a materially worse outcome than an unnecessary referral, and
> the report should discuss this asymmetry rather than averaging it away.

## Inference latency

| Metric | Value |
|---|---:|
| Mean | 12.0 ms |
| Median | 11.5 ms |
| p95 | 18.3 ms |
| Device | cuda |
| Parameters | 49,460,071 |
| Model size | 188.7 MB |

## Classification report

```
              precision    recall  f1-score   support

       akiec     0.7273    0.7692    0.7477        52
         bcc     0.7353    0.7042    0.7194        71
         bkl     0.7115    0.6647    0.6873       167
          df     0.4839    0.7500    0.5882        20
         mel     0.7063    0.6048    0.6516       167
          nv     0.9109    0.9373    0.9239      1004
        vasc     0.8750    0.6667    0.7568        21

    accuracy                         0.8469      1502
   macro avg     0.7357    0.7281    0.7250      1502
weighted avg     0.8452    0.8469    0.8448      1502
```

## Figures

- `confusion_matrix.png`
- `reliability_diagram.png`
- `training_curves.png`
