# Evaluation — efficientnet_b3

**Checkpoint:** `ml/checkpoints/efficientnet_b3_best.HAM-only.pt`  
**Split:** test (314 images)  
**Selected on:** validation macro_f1 = 0.7118 (epoch 15)  
**Class mapping:** v1.0.0  
**Temperature:** 5.165

## Headline metrics

| Metric | Value |
|---|---:|
| Macro F1 | **0.1130** |
| Balanced accuracy | **0.2504** |
| Weighted F1 | 0.2218 |
| Accuracy | 0.2293 |
| Cohen's kappa | 0.0808 |
| Macro ROC-AUC (OvR) | nan |
| Expected calibration error | 0.0259 |

> Accuracy is listed but is not the headline. `nv` is roughly two thirds of this
> dataset, so a model that answers "mole" every time scores around 0.67 accuracy
> while missing every melanoma. Macro F1 and per-class recall are the numbers that
> describe whether this model is useful.

## Per-class performance

| Class | Malignancy | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| `akiec` Actinic keratosis | premalignant | 0.2800 | **0.0654** | 0.1061 | 107 |
| `bcc` Basal cell carcinoma | malignant | 0.7381 | **0.2403** | 0.3626 | 129 |
| `bkl` Benign keratosis | benign | 0.0435 | **0.0294** | 0.0351 | 34 |
| `df` Dermatofibroma | benign | 0.0000 | **0.0000** | 0.0000 | 0 |
| `mel` Melanoma | malignant | 0.0000 | **0.0000** | 0.0000 | 8 |
| `nv` Mole | benign | 0.1701 | **0.9167** | 0.2870 | 36 |
| `vasc` Vascular lesion | benign | 0.0000 | **0.0000** | 0.0000 | 0 |

## Screening view

Collapsing the seven classes into "needs escalation" (`akiec`, `bcc`, `mel`) versus not:

| Metric | Value |
|---|---:|
| Sensitivity | **0.2992** |
| Specificity | 0.9571 |
| Missed serious cases (false negatives) | **171** |
| False alarms | 3 |

> In a screening tool the false negatives in that table are the number that matters.
> A missed melanoma is a materially worse outcome than an unnecessary referral, and
> the report should discuss this asymmetry rather than averaging it away.

## Inference latency

| Metric | Value |
|---|---:|
| Mean | 12.3 ms |
| Median | 9.7 ms |
| p95 | 19.2 ms |
| Device | cuda |
| Parameters | 10,706,991 |
| Model size | 40.8 MB |

## Classification report

```
              precision    recall  f1-score   support

       akiec     0.2800    0.0654    0.1061       107
         bcc     0.7381    0.2403    0.3626       129
         bkl     0.0435    0.0294    0.0351        34
          df     0.0000    0.0000    0.0000         0
         mel     0.0000    0.0000    0.0000         8
          nv     0.1701    0.9167    0.2870        36
        vasc     0.0000    0.0000    0.0000         0

    accuracy                         0.2293       314
   macro avg     0.1760    0.1788    0.1130       314
weighted avg     0.4229    0.2293    0.2218       314
```

## Figures

- `confusion_matrix.png`
- `reliability_diagram.png`
- `training_curves.png`
