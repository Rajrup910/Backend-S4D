# Evaluation — convnext_small

**Checkpoint:** `ml/checkpoints/convnext_small_best.HAM-only.pt`  
**Split:** test (314 images)  
**Selected on:** validation macro_f1 = 0.7435 (epoch 9)  
**Class mapping:** v1.0.0  
**Temperature:** 4.848

## Headline metrics

| Metric | Value |
|---|---:|
| Macro F1 | **0.1716** |
| Balanced accuracy | **0.2816** |
| Weighted F1 | 0.2835 |
| Accuracy | 0.2548 |
| Cohen's kappa | 0.1274 |
| Macro ROC-AUC (OvR) | nan |
| Expected calibration error | 0.0588 |

> Accuracy is listed but is not the headline. `nv` is roughly two thirds of this
> dataset, so a model that answers "mole" every time scores around 0.67 accuracy
> while missing every melanoma. Macro F1 and per-class recall are the numbers that
> describe whether this model is useful.

## Per-class performance

| Class | Malignancy | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| `akiec` Actinic keratosis | premalignant | 0.4103 | **0.1495** | 0.2192 | 107 |
| `bcc` Basal cell carcinoma | malignant | 0.7714 | **0.2093** | 0.3293 | 129 |
| `bkl` Benign keratosis | benign | 0.1512 | **0.3824** | 0.2167 | 34 |
| `df` Dermatofibroma | benign | 0.0000 | **0.0000** | 0.0000 | 0 |
| `mel` Melanoma | malignant | 0.0000 | **0.0000** | 0.0000 | 8 |
| `nv` Mole | benign | 0.3243 | **0.6667** | 0.4364 | 36 |
| `vasc` Vascular lesion | benign | 0.0000 | **0.0000** | 0.0000 | 0 |

## Screening view

Collapsing the seven classes into "needs escalation" (`akiec`, `bcc`, `mel`) versus not:

| Metric | Value |
|---|---:|
| Sensitivity | **0.3238** |
| Specificity | 0.8714 |
| Missed serious cases (false negatives) | **165** |
| False alarms | 9 |

> In a screening tool the false negatives in that table are the number that matters.
> A missed melanoma is a materially worse outcome than an unnecessary referral, and
> the report should discuss this asymmetry rather than averaging it away.

## Inference latency

| Metric | Value |
|---|---:|
| Mean | 11.9 ms |
| Median | 12.3 ms |
| p95 | 16.8 ms |
| Device | cuda |
| Parameters | 49,460,071 |
| Model size | 188.7 MB |

## Classification report

```
              precision    recall  f1-score   support

       akiec     0.4103    0.1495    0.2192       107
         bcc     0.7714    0.2093    0.3293       129
         bkl     0.1512    0.3824    0.2167        34
          df     0.0000    0.0000    0.0000         0
         mel     0.0000    0.0000    0.0000         8
          nv     0.3243    0.6667    0.4364        36
        vasc     0.0000    0.0000    0.0000         0

    accuracy                         0.2548       314
   macro avg     0.2367    0.2011    0.1716       314
weighted avg     0.5103    0.2548    0.2835       314
```

## Figures

- `confusion_matrix.png`
- `reliability_diagram.png`
- `training_curves.png`
