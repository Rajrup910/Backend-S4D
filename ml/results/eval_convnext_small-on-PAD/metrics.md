# Evaluation — convnext_small

**Checkpoint:** `ml/checkpoints/convnext_small_best.PAD-only.pt`  
**Split:** test (314 images)  
**Selected on:** validation macro_f1 = 0.6087 (epoch 18)  
**Class mapping:** v1.0.0  
**Temperature:** 1.221

## Headline metrics

| Metric | Value |
|---|---:|
| Macro F1 | **0.4757** |
| Balanced accuracy | **0.6677** |
| Weighted F1 | 0.7794 |
| Accuracy | 0.7866 |
| Cohen's kappa | 0.6900 |
| Macro ROC-AUC (OvR) | nan |
| Expected calibration error | 0.1079 |

> Accuracy is listed but is not the headline. `nv` is roughly two thirds of this
> dataset, so a model that answers "mole" every time scores around 0.67 accuracy
> while missing every melanoma. Macro F1 and per-class recall are the numbers that
> describe whether this model is useful.

## Per-class performance

| Class | Malignancy | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| `akiec` Actinic keratosis | premalignant | 0.8523 | **0.7009** | 0.7692 | 107 |
| `bcc` Basal cell carcinoma | malignant | 0.7972 | **0.8837** | 0.8382 | 129 |
| `bkl` Benign keratosis | benign | 0.6829 | **0.8235** | 0.7467 | 34 |
| `df` Dermatofibroma | benign | 0.0000 | **0.0000** | 0.0000 | 0 |
| `mel` Melanoma | malignant | 1.0000 | **0.1250** | 0.2222 | 8 |
| `nv` Mole | benign | 0.7073 | **0.8056** | 0.7532 | 36 |
| `vasc` Vascular lesion | benign | 0.0000 | **0.0000** | 0.0000 | 0 |

## Screening view

Collapsing the seven classes into "needs escalation" (`akiec`, `bcc`, `mel`) versus not:

| Metric | Value |
|---|---:|
| Sensitivity | **0.9385** |
| Specificity | 0.9571 |
| Missed serious cases (false negatives) | **15** |
| False alarms | 3 |

> In a screening tool the false negatives in that table are the number that matters.
> A missed melanoma is a materially worse outcome than an unnecessary referral, and
> the report should discuss this asymmetry rather than averaging it away.

## Inference latency

| Metric | Value |
|---|---:|
| Mean | 11.8 ms |
| Median | 10.0 ms |
| p95 | 18.6 ms |
| Device | cuda |
| Parameters | 49,460,071 |
| Model size | 188.7 MB |

## Classification report

```
              precision    recall  f1-score   support

       akiec     0.8523    0.7009    0.7692       107
         bcc     0.7972    0.8837    0.8382       129
         bkl     0.6829    0.8235    0.7467        34
          df     0.0000    0.0000    0.0000         0
         mel     1.0000    0.1250    0.2222         8
          nv     0.7073    0.8056    0.7532        36
        vasc     0.0000    0.0000    0.0000         0

    accuracy                         0.7866       314
   macro avg     0.5771    0.4770    0.4757       314
weighted avg     0.7985    0.7866    0.7794       314
```

## Figures

- `confusion_matrix.png`
- `reliability_diagram.png`
- `training_curves.png`
