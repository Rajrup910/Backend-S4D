# Evaluation — densenet121

**Checkpoint:** `ml/checkpoints/densenet121_best.combined.pt`  
**Split:** test (314 images)  
**Selected on:** validation macro_f1 = 0.7430 (epoch 18)  
**Class mapping:** v1.0.0  
**Temperature:** 1.000 (uncalibrated)

## Headline metrics

| Metric | Value |
|---|---:|
| Macro F1 | **0.4678** |
| Balanced accuracy | **0.6404** |
| Weighted F1 | 0.7476 |
| Accuracy | 0.7484 |
| Cohen's kappa | 0.6259 |
| Macro ROC-AUC (OvR) | nan |
| Expected calibration error | 0.0534 |

> Accuracy is listed but is not the headline. `nv` is roughly two thirds of this
> dataset, so a model that answers "mole" every time scores around 0.67 accuracy
> while missing every melanoma. Macro F1 and per-class recall are the numbers that
> describe whether this model is useful.

## Per-class performance

| Class | Malignancy | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| `akiec` Actinic keratosis | premalignant | 0.8100 | **0.7043** | 0.7535 | 115 |
| `bcc` Basal cell carcinoma | malignant | 0.7586 | **0.8527** | 0.8029 | 129 |
| `bkl` Benign keratosis | benign | 0.5946 | **0.6667** | 0.6286 | 33 |
| `df` Dermatofibroma | benign | 0.0000 | **0.0000** | 0.0000 | 0 |
| `mel` Melanoma | malignant | 0.5000 | **0.3333** | 0.4000 | 6 |
| `nv` Mole | benign | 0.7407 | **0.6452** | 0.6897 | 31 |
| `vasc` Vascular lesion | benign | 0.0000 | **0.0000** | 0.0000 | 0 |

## Screening view

Collapsing the seven classes into "needs escalation" (`akiec`, `bcc`, `mel`) versus not:

| Metric | Value |
|---|---:|
| Sensitivity | **0.9320** |
| Specificity | 0.7500 |
| Missed serious cases (false negatives) | **17** |
| False alarms | 16 |

> In a screening tool the false negatives in that table are the number that matters.
> A missed melanoma is a materially worse outcome than an unnecessary referral, and
> the report should discuss this asymmetry rather than averaging it away.

## Inference latency

| Metric | Value |
|---|---:|
| Mean | 18.6 ms |
| Median | 13.9 ms |
| p95 | 26.4 ms |
| Device | cuda |
| Parameters | 6,961,031 |
| Model size | 26.6 MB |

## Classification report

```
              precision    recall  f1-score   support

       akiec     0.8100    0.7043    0.7535       115
         bcc     0.7586    0.8527    0.8029       129
         bkl     0.5946    0.6667    0.6286        33
          df     0.0000    0.0000    0.0000         0
         mel     0.5000    0.3333    0.4000         6
          nv     0.7407    0.6452    0.6897        31
        vasc     0.0000    0.0000    0.0000         0

    accuracy                         0.7484       314
   macro avg     0.4863    0.4575    0.4678       314
weighted avg     0.7535    0.7484    0.7476       314
```

## Figures

- `confusion_matrix.png`
- `reliability_diagram.png`
- `training_curves.png`
