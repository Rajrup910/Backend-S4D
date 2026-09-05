# Evaluation — efficientnet_b3

**Checkpoint:** `ml/checkpoints/efficientnet_b3_best.combined.pt`  
**Split:** test (314 images)  
**Selected on:** validation macro_f1 = 0.7500 (epoch 15)  
**Class mapping:** v1.0.0  
**Temperature:** 1.000 (uncalibrated)

## Headline metrics

| Metric | Value |
|---|---:|
| Macro F1 | **0.4797** |
| Balanced accuracy | **0.6640** |
| Weighted F1 | 0.7821 |
| Accuracy | 0.7866 |
| Cohen's kappa | 0.6830 |
| Macro ROC-AUC (OvR) | nan |
| Expected calibration error | 0.0562 |

> Accuracy is listed but is not the headline. `nv` is roughly two thirds of this
> dataset, so a model that answers "mole" every time scores around 0.67 accuracy
> while missing every melanoma. Macro F1 and per-class recall are the numbers that
> describe whether this model is useful.

## Per-class performance

| Class | Malignancy | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| `akiec` Actinic keratosis | premalignant | 0.8000 | **0.8000** | 0.8000 | 115 |
| `bcc` Basal cell carcinoma | malignant | 0.8015 | **0.8140** | 0.8077 | 129 |
| `bkl` Benign keratosis | benign | 0.7500 | **0.6364** | 0.6885 | 33 |
| `df` Dermatofibroma | benign | 0.0000 | **0.0000** | 0.0000 | 0 |
| `mel` Melanoma | malignant | 0.5000 | **0.1667** | 0.2500 | 6 |
| `nv` Mole | benign | 0.7368 | **0.9032** | 0.8116 | 31 |
| `vasc` Vascular lesion | benign | 0.0000 | **0.0000** | 0.0000 | 0 |

## Screening view

Collapsing the seven classes into "needs escalation" (`akiec`, `bcc`, `mel`) versus not:

| Metric | Value |
|---|---:|
| Sensitivity | **0.9480** |
| Specificity | 0.8281 |
| Missed serious cases (false negatives) | **13** |
| False alarms | 11 |

> In a screening tool the false negatives in that table are the number that matters.
> A missed melanoma is a materially worse outcome than an unnecessary referral, and
> the report should discuss this asymmetry rather than averaging it away.

## Inference latency

| Metric | Value |
|---|---:|
| Mean | 11.0 ms |
| Median | 10.3 ms |
| p95 | 13.7 ms |
| Device | cuda |
| Parameters | 10,706,991 |
| Model size | 40.8 MB |

## Classification report

```
              precision    recall  f1-score   support

       akiec     0.8000    0.8000    0.8000       115
         bcc     0.8015    0.8140    0.8077       129
         bkl     0.7500    0.6364    0.6885        33
          df     0.0000    0.0000    0.0000         0
         mel     0.5000    0.1667    0.2500         6
          nv     0.7368    0.9032    0.8116        31
        vasc     0.0000    0.0000    0.0000         0

    accuracy                         0.7866       314
   macro avg     0.5126    0.4743    0.4797       314
weighted avg     0.7834    0.7866    0.7821       314
```

## Figures

- `confusion_matrix.png`
- `reliability_diagram.png`
- `training_curves.png`
