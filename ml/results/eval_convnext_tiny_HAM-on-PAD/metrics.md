# Evaluation — convnext_tiny

**Checkpoint:** `ml/checkpoints/convnext_tiny_best.HAM-only.pt`  
**Split:** test (314 images)  
**Selected on:** validation macro_f1 = 0.7482 (epoch 15)  
**Class mapping:** v1.0.0  
**Temperature:** 3.987

## Headline metrics

| Metric | Value |
|---|---:|
| Macro F1 | **0.1639** |
| Balanced accuracy | **0.2992** |
| Weighted F1 | 0.2743 |
| Accuracy | 0.2643 |
| Cohen's kappa | 0.1226 |
| Macro ROC-AUC (OvR) | nan |
| Expected calibration error | 0.0290 |

> Accuracy is listed but is not the headline. `nv` is roughly two thirds of this
> dataset, so a model that answers "mole" every time scores around 0.67 accuracy
> while missing every melanoma. Macro F1 and per-class recall are the numbers that
> describe whether this model is useful.

## Per-class performance

| Class | Malignancy | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| `akiec` Actinic keratosis | premalignant | 0.3519 | **0.1776** | 0.2360 | 107 |
| `bcc` Basal cell carcinoma | malignant | 0.8889 | **0.1860** | 0.3077 | 129 |
| `bkl` Benign keratosis | benign | 0.2063 | **0.3824** | 0.2680 | 34 |
| `df` Dermatofibroma | benign | 0.0000 | **0.0000** | 0.0000 | 0 |
| `mel` Melanoma | malignant | 0.0000 | **0.0000** | 0.0000 | 8 |
| `nv` Mole | benign | 0.2160 | **0.7500** | 0.3354 | 36 |
| `vasc` Vascular lesion | benign | 0.0000 | **0.0000** | 0.0000 | 0 |

## Screening view

Collapsing the seven classes into "needs escalation" (`akiec`, `bcc`, `mel`) versus not:

| Metric | Value |
|---|---:|
| Sensitivity | **0.3361** |
| Specificity | 0.9714 |
| Missed serious cases (false negatives) | **162** |
| False alarms | 2 |

> In a screening tool the false negatives in that table are the number that matters.
> A missed melanoma is a materially worse outcome than an unnecessary referral, and
> the report should discuss this asymmetry rather than averaging it away.

## Inference latency

| Metric | Value |
|---|---:|
| Mean | 6.6 ms |
| Median | 4.8 ms |
| p95 | 10.7 ms |
| Device | cuda |
| Parameters | 27,825,511 |
| Model size | 106.1 MB |

## Classification report

```
              precision    recall  f1-score   support

       akiec     0.3519    0.1776    0.2360       107
         bcc     0.8889    0.1860    0.3077       129
         bkl     0.2063    0.3824    0.2680        34
          df     0.0000    0.0000    0.0000         0
         mel     0.0000    0.0000    0.0000         8
          nv     0.2160    0.7500    0.3354        36
        vasc     0.0000    0.0000    0.0000         0

    accuracy                         0.2643       314
   macro avg     0.2376    0.2137    0.1639       314
weighted avg     0.5322    0.2643    0.2743       314
```

## Figures

- `confusion_matrix.png`
- `reliability_diagram.png`
- `training_curves.png`
