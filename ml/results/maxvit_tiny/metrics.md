# Evaluation — maxvit_tiny

**Checkpoint:** `ml/checkpoints/maxvit_tiny_best.pt`  
**Split:** test (1,502 images)  
**Selected on:** validation macro_f1 = 0.7176 (epoch 18)  
**Class mapping:** v1.0.0  
**Temperature:** 1.000 (uncalibrated)

## Headline metrics

| Metric | Value |
|---|---:|
| Macro F1 | **0.7525** |
| Balanced accuracy | **0.7504** |
| Weighted F1 | 0.8592 |
| Accuracy | 0.8602 |
| Cohen's kappa | 0.7326 |
| Macro ROC-AUC (OvR) | 0.9306 |
| Expected calibration error | 0.1146 |

> Accuracy is listed but is not the headline. `nv` is roughly two thirds of this
> dataset, so a model that answers "mole" every time scores around 0.67 accuracy
> while missing every melanoma. Macro F1 and per-class recall are the numbers that
> describe whether this model is useful.

## Per-class performance

| Class | Malignancy | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| `akiec` Actinic keratosis | premalignant | 0.6056 | **0.8269** | 0.6992 | 52 |
| `bcc` Basal cell carcinoma | malignant | 0.7901 | **0.9014** | 0.8421 | 71 |
| `bkl` Benign keratosis | benign | 0.7817 | **0.6647** | 0.7184 | 167 |
| `df` Dermatofibroma | benign | 0.7143 | **0.5000** | 0.5882 | 20 |
| `mel` Melanoma | malignant | 0.6727 | **0.6647** | 0.6687 | 167 |
| `nv` Mole | benign | 0.9268 | **0.9333** | 0.9300 | 1,004 |
| `vasc` Vascular lesion | benign | 0.8889 | **0.7619** | 0.8205 | 21 |

## Screening view

Collapsing the seven classes into "needs escalation" (`akiec`, `bcc`, `mel`) versus not:

| Metric | Value |
|---|---:|
| Sensitivity | **0.8000** |
| Specificity | 0.9299 |
| Missed serious cases (false negatives) | **58** |
| False alarms | 85 |

> In a screening tool the false negatives in that table are the number that matters.
> A missed melanoma is a materially worse outcome than an unnecessary referral, and
> the report should discuss this asymmetry rather than averaging it away.

## Inference latency

| Metric | Value |
|---|---:|
| Mean | 24.1 ms |
| Median | 22.3 ms |
| p95 | 38.3 ms |
| Device | cuda |
| Parameters | 30,407,119 |
| Model size | 116.0 MB |

## Classification report

```
              precision    recall  f1-score   support

       akiec     0.6056    0.8269    0.6992        52
         bcc     0.7901    0.9014    0.8421        71
         bkl     0.7817    0.6647    0.7184       167
          df     0.7143    0.5000    0.5882        20
         mel     0.6727    0.6647    0.6687       167
          nv     0.9268    0.9333    0.9300      1004
        vasc     0.8889    0.7619    0.8205        21

    accuracy                         0.8602      1502
   macro avg     0.7686    0.7504    0.7525      1502
weighted avg     0.8615    0.8602    0.8592      1502
```

## Figures

- `confusion_matrix.png`
- `reliability_diagram.png`
- `training_curves.png`
