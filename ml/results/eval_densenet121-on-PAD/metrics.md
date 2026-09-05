# Evaluation — densenet121

**Checkpoint:** `ml/checkpoints/densenet121_best.PAD-only.pt`  
**Split:** test (314 images)  
**Selected on:** validation macro_f1 = 0.5553 (epoch 12)  
**Class mapping:** v1.0.0  
**Temperature:** 0.907

## Headline metrics

| Metric | Value |
|---|---:|
| Macro F1 | **0.4403** |
| Balanced accuracy | **0.6257** |
| Weighted F1 | 0.7492 |
| Accuracy | 0.7516 |
| Cohen's kappa | 0.6424 |
| Macro ROC-AUC (OvR) | nan |
| Expected calibration error | 0.0665 |

> Accuracy is listed but is not the headline. `nv` is roughly two thirds of this
> dataset, so a model that answers "mole" every time scores around 0.67 accuracy
> while missing every melanoma. Macro F1 and per-class recall are the numbers that
> describe whether this model is useful.

## Per-class performance

| Class | Malignancy | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| `akiec` Actinic keratosis | premalignant | 0.8280 | **0.7196** | 0.7700 | 107 |
| `bcc` Basal cell carcinoma | malignant | 0.8045 | **0.8295** | 0.8168 | 129 |
| `bkl` Benign keratosis | benign | 0.5610 | **0.6765** | 0.6133 | 34 |
| `df` Dermatofibroma | benign | 0.0000 | **0.0000** | 0.0000 | 0 |
| `mel` Melanoma | malignant | 0.3333 | **0.1250** | 0.1818 | 8 |
| `nv` Mole | benign | 0.6364 | **0.7778** | 0.7000 | 36 |
| `vasc` Vascular lesion | benign | 0.0000 | **0.0000** | 0.0000 | 0 |

## Screening view

Collapsing the seven classes into "needs escalation" (`akiec`, `bcc`, `mel`) versus not:

| Metric | Value |
|---|---:|
| Sensitivity | **0.9098** |
| Specificity | 0.9000 |
| Missed serious cases (false negatives) | **22** |
| False alarms | 7 |

> In a screening tool the false negatives in that table are the number that matters.
> A missed melanoma is a materially worse outcome than an unnecessary referral, and
> the report should discuss this asymmetry rather than averaging it away.

## Inference latency

| Metric | Value |
|---|---:|
| Mean | 16.3 ms |
| Median | 12.4 ms |
| p95 | 23.1 ms |
| Device | cuda |
| Parameters | 6,961,031 |
| Model size | 26.6 MB |

## Classification report

```
              precision    recall  f1-score   support

       akiec     0.8280    0.7196    0.7700       107
         bcc     0.8045    0.8295    0.8168       129
         bkl     0.5610    0.6765    0.6133        34
          df     0.0000    0.0000    0.0000         0
         mel     0.3333    0.1250    0.1818         8
          nv     0.6364    0.7778    0.7000        36
        vasc     0.0000    0.0000    0.0000         0

    accuracy                         0.7516       314
   macro avg     0.4519    0.4469    0.4403       314
weighted avg     0.7548    0.7516    0.7492       314
```

## Figures

- `confusion_matrix.png`
- `reliability_diagram.png`
- `training_curves.png`
