# Evaluation — densenet121

**Checkpoint:** `ml/checkpoints/densenet121_best.PADwarm.pt`  
**Split:** test (314 images)  
**Selected on:** validation macro_f1 = 0.5473 (epoch 12)  
**Class mapping:** v1.0.0  
**Temperature:** 0.969

## Headline metrics

| Metric | Value |
|---|---:|
| Macro F1 | **0.4945** |
| Balanced accuracy | **0.6865** |
| Weighted F1 | 0.7603 |
| Accuracy | 0.7611 |
| Cohen's kappa | 0.6565 |
| Macro ROC-AUC (OvR) | nan |
| Expected calibration error | 0.0622 |

> Accuracy is listed but is not the headline. `nv` is roughly two thirds of this
> dataset, so a model that answers "mole" every time scores around 0.67 accuracy
> while missing every melanoma. Macro F1 and per-class recall are the numbers that
> describe whether this model is useful.

## Per-class performance

| Class | Malignancy | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| `akiec` Actinic keratosis | premalignant | 0.7885 | **0.7664** | 0.7773 | 107 |
| `bcc` Basal cell carcinoma | malignant | 0.8145 | **0.7829** | 0.7984 | 129 |
| `bkl` Benign keratosis | benign | 0.6667 | **0.6471** | 0.6567 | 34 |
| `df` Dermatofibroma | benign | 0.0000 | **0.0000** | 0.0000 | 0 |
| `mel` Melanoma | malignant | 0.7500 | **0.3750** | 0.5000 | 8 |
| `nv` Mole | benign | 0.6327 | **0.8611** | 0.7294 | 36 |
| `vasc` Vascular lesion | benign | 0.0000 | **0.0000** | 0.0000 | 0 |

## Screening view

Collapsing the seven classes into "needs escalation" (`akiec`, `bcc`, `mel`) versus not:

| Metric | Value |
|---|---:|
| Sensitivity | **0.9303** |
| Specificity | 0.9286 |
| Missed serious cases (false negatives) | **17** |
| False alarms | 5 |

> In a screening tool the false negatives in that table are the number that matters.
> A missed melanoma is a materially worse outcome than an unnecessary referral, and
> the report should discuss this asymmetry rather than averaging it away.

## Inference latency

| Metric | Value |
|---|---:|
| Mean | 11.5 ms |
| Median | 11.5 ms |
| p95 | 11.7 ms |
| Device | cuda |
| Parameters | 6,961,031 |
| Model size | 26.6 MB |

## Classification report

```
              precision    recall  f1-score   support

       akiec     0.7885    0.7664    0.7773       107
         bcc     0.8145    0.7829    0.7984       129
         bkl     0.6667    0.6471    0.6567        34
          df     0.0000    0.0000    0.0000         0
         mel     0.7500    0.3750    0.5000         8
          nv     0.6327    0.8611    0.7294        36
        vasc     0.0000    0.0000    0.0000         0

    accuracy                         0.7611       314
   macro avg     0.5218    0.4904    0.4945       314
weighted avg     0.7671    0.7611    0.7603       314
```

## Figures

- `confusion_matrix.png`
- `reliability_diagram.png`
- `training_curves.png`
