# Evaluation — efficientnet_b0

**Checkpoint:** `ml/checkpoints/efficientnet_b0_best.PAD-only.pt`  
**Split:** test (314 images)  
**Selected on:** validation macro_f1 = 0.5125 (epoch 20)  
**Class mapping:** v1.0.0  
**Temperature:** 0.954

## Headline metrics

| Metric | Value |
|---|---:|
| Macro F1 | **0.4660** |
| Balanced accuracy | **0.6671** |
| Weighted F1 | 0.7719 |
| Accuracy | 0.7707 |
| Cohen's kappa | 0.6724 |
| Macro ROC-AUC (OvR) | nan |
| Expected calibration error | 0.0655 |

> Accuracy is listed but is not the headline. `nv` is roughly two thirds of this
> dataset, so a model that answers "mole" every time scores around 0.67 accuracy
> while missing every melanoma. Macro F1 and per-class recall are the numbers that
> describe whether this model is useful.

## Per-class performance

| Class | Malignancy | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| `akiec` Actinic keratosis | premalignant | 0.8478 | **0.7290** | 0.7839 | 107 |
| `bcc` Basal cell carcinoma | malignant | 0.8450 | **0.8450** | 0.8450 | 129 |
| `bkl` Benign keratosis | benign | 0.5854 | **0.7059** | 0.6400 | 34 |
| `df` Dermatofibroma | benign | 0.0000 | **0.0000** | 0.0000 | 0 |
| `mel` Melanoma | malignant | 0.3333 | **0.2500** | 0.2857 | 8 |
| `nv` Mole | benign | 0.6304 | **0.8056** | 0.7073 | 36 |
| `vasc` Vascular lesion | benign | 0.0000 | **0.0000** | 0.0000 | 0 |

## Screening view

Collapsing the seven classes into "needs escalation" (`akiec`, `bcc`, `mel`) versus not:

| Metric | Value |
|---|---:|
| Sensitivity | **0.9180** |
| Specificity | 0.9571 |
| Missed serious cases (false negatives) | **20** |
| False alarms | 3 |

> In a screening tool the false negatives in that table are the number that matters.
> A missed melanoma is a materially worse outcome than an unnecessary referral, and
> the report should discuss this asymmetry rather than averaging it away.

## Inference latency

| Metric | Value |
|---|---:|
| Mean | 8.4 ms |
| Median | 6.3 ms |
| p95 | 16.4 ms |
| Device | cuda |
| Parameters | 4,016,515 |
| Model size | 15.3 MB |

## Classification report

```
              precision    recall  f1-score   support

       akiec     0.8478    0.7290    0.7839       107
         bcc     0.8450    0.8450    0.8450       129
         bkl     0.5854    0.7059    0.6400        34
          df     0.0000    0.0000    0.0000         0
         mel     0.3333    0.2500    0.2857         8
          nv     0.6304    0.8056    0.7073        36
        vasc     0.0000    0.0000    0.0000         0

    accuracy                         0.7707       314
   macro avg     0.4631    0.4765    0.4660       314
weighted avg     0.7802    0.7707    0.7719       314
```

## Figures

- `confusion_matrix.png`
- `reliability_diagram.png`
- `training_curves.png`
