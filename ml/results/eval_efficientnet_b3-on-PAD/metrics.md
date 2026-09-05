# Evaluation — efficientnet_b3

**Checkpoint:** `ml/checkpoints/efficientnet_b3_best.PAD-only.pt`  
**Split:** test (314 images)  
**Selected on:** validation macro_f1 = 0.5251 (epoch 12)  
**Class mapping:** v1.0.0  
**Temperature:** 0.850

## Headline metrics

| Metric | Value |
|---|---:|
| Macro F1 | **0.4634** |
| Balanced accuracy | **0.6500** |
| Weighted F1 | 0.7413 |
| Accuracy | 0.7420 |
| Cohen's kappa | 0.6276 |
| Macro ROC-AUC (OvR) | nan |
| Expected calibration error | 0.0937 |

> Accuracy is listed but is not the headline. `nv` is roughly two thirds of this
> dataset, so a model that answers "mole" every time scores around 0.67 accuracy
> while missing every melanoma. Macro F1 and per-class recall are the numbers that
> describe whether this model is useful.

## Per-class performance

| Class | Malignancy | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| `akiec` Actinic keratosis | premalignant | 0.8090 | **0.6729** | 0.7347 | 107 |
| `bcc` Basal cell carcinoma | malignant | 0.7986 | **0.8605** | 0.8284 | 129 |
| `bkl` Benign keratosis | benign | 0.5641 | **0.6471** | 0.6027 | 34 |
| `df` Dermatofibroma | benign | 0.0000 | **0.0000** | 0.0000 | 0 |
| `mel` Melanoma | malignant | 0.5000 | **0.3750** | 0.4286 | 8 |
| `nv` Mole | benign | 0.6098 | **0.6944** | 0.6494 | 36 |
| `vasc` Vascular lesion | benign | 0.0000 | **0.0000** | 0.0000 | 0 |

## Screening view

Collapsing the seven classes into "needs escalation" (`akiec`, `bcc`, `mel`) versus not:

| Metric | Value |
|---|---:|
| Sensitivity | **0.9385** |
| Specificity | 0.9286 |
| Missed serious cases (false negatives) | **15** |
| False alarms | 5 |

> In a screening tool the false negatives in that table are the number that matters.
> A missed melanoma is a materially worse outcome than an unnecessary referral, and
> the report should discuss this asymmetry rather than averaging it away.

## Inference latency

| Metric | Value |
|---|---:|
| Mean | 13.1 ms |
| Median | 10.5 ms |
| p95 | 20.3 ms |
| Device | cuda |
| Parameters | 10,706,991 |
| Model size | 40.8 MB |

## Classification report

```
              precision    recall  f1-score   support

       akiec     0.8090    0.6729    0.7347       107
         bcc     0.7986    0.8605    0.8284       129
         bkl     0.5641    0.6471    0.6027        34
          df     0.0000    0.0000    0.0000         0
         mel     0.5000    0.3750    0.4286         8
          nv     0.6098    0.6944    0.6494        36
        vasc     0.0000    0.0000    0.0000         0

    accuracy                         0.7420       314
   macro avg     0.4688    0.4643    0.4634       314
weighted avg     0.7475    0.7420    0.7413       314
```

## Figures

- `confusion_matrix.png`
- `reliability_diagram.png`
- `training_curves.png`
