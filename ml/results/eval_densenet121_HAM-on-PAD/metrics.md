# Evaluation — densenet121

**Checkpoint:** `ml/checkpoints/densenet121_best.HAM-only.pt`  
**Split:** test (314 images)  
**Selected on:** validation macro_f1 = 0.7207 (epoch 13)  
**Class mapping:** v1.0.0  
**Temperature:** 5.517

## Headline metrics

| Metric | Value |
|---|---:|
| Macro F1 | **0.1271** |
| Balanced accuracy | **0.2449** |
| Weighted F1 | 0.2445 |
| Accuracy | 0.2389 |
| Cohen's kappa | 0.0781 |
| Macro ROC-AUC (OvR) | nan |
| Expected calibration error | 0.0525 |

> Accuracy is listed but is not the headline. `nv` is roughly two thirds of this
> dataset, so a model that answers "mole" every time scores around 0.67 accuracy
> while missing every melanoma. Macro F1 and per-class recall are the numbers that
> describe whether this model is useful.

## Per-class performance

| Class | Malignancy | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| `akiec` Actinic keratosis | premalignant | 0.3429 | **0.1121** | 0.1690 | 107 |
| `bcc` Basal cell carcinoma | malignant | 0.6154 | **0.2481** | 0.3536 | 129 |
| `bkl` Benign keratosis | benign | 0.0645 | **0.0588** | 0.0615 | 34 |
| `df` Dermatofibroma | benign | 0.0000 | **0.0000** | 0.0000 | 0 |
| `mel` Melanoma | malignant | 0.0000 | **0.0000** | 0.0000 | 8 |
| `nv` Mole | benign | 0.1883 | **0.8056** | 0.3053 | 36 |
| `vasc` Vascular lesion | benign | 0.0000 | **0.0000** | 0.0000 | 0 |

## Screening view

Collapsing the seven classes into "needs escalation" (`akiec`, `bcc`, `mel`) versus not:

| Metric | Value |
|---|---:|
| Sensitivity | **0.4221** |
| Specificity | 0.9714 |
| Missed serious cases (false negatives) | **141** |
| False alarms | 2 |

> In a screening tool the false negatives in that table are the number that matters.
> A missed melanoma is a materially worse outcome than an unnecessary referral, and
> the report should discuss this asymmetry rather than averaging it away.

## Inference latency

| Metric | Value |
|---|---:|
| Mean | 15.2 ms |
| Median | 11.9 ms |
| p95 | 22.3 ms |
| Device | cuda |
| Parameters | 6,961,031 |
| Model size | 26.6 MB |

## Classification report

```
              precision    recall  f1-score   support

       akiec     0.3429    0.1121    0.1690       107
         bcc     0.6154    0.2481    0.3536       129
         bkl     0.0645    0.0588    0.0615        34
          df     0.0000    0.0000    0.0000         0
         mel     0.0000    0.0000    0.0000         8
          nv     0.1883    0.8056    0.3053        36
        vasc     0.0000    0.0000    0.0000         0

    accuracy                         0.2389       314
   macro avg     0.1730    0.1749    0.1271       314
weighted avg     0.3982    0.2389    0.2445       314
```

## Figures

- `confusion_matrix.png`
- `reliability_diagram.png`
- `training_curves.png`
