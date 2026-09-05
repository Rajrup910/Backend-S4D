# Evaluation — efficientnet_b3

**Checkpoint:** `ml/checkpoints/efficientnet_b3_best.PADwarm.pt`  
**Split:** test (314 images)  
**Selected on:** validation macro_f1 = 0.5354 (epoch 14)  
**Class mapping:** v1.0.0  
**Temperature:** 0.919

## Headline metrics

| Metric | Value |
|---|---:|
| Macro F1 | **0.4773** |
| Balanced accuracy | **0.6765** |
| Weighted F1 | 0.7581 |
| Accuracy | 0.7611 |
| Cohen's kappa | 0.6580 |
| Macro ROC-AUC (OvR) | nan |
| Expected calibration error | 0.0809 |

> Accuracy is listed but is not the headline. `nv` is roughly two thirds of this
> dataset, so a model that answers "mole" every time scores around 0.67 accuracy
> while missing every melanoma. Macro F1 and per-class recall are the numbers that
> describe whether this model is useful.

## Per-class performance

| Class | Malignancy | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| `akiec` Actinic keratosis | premalignant | 0.8333 | **0.6542** | 0.7330 | 107 |
| `bcc` Basal cell carcinoma | malignant | 0.8088 | **0.8527** | 0.8302 | 129 |
| `bkl` Benign keratosis | benign | 0.5778 | **0.7647** | 0.6582 | 34 |
| `df` Dermatofibroma | benign | 0.0000 | **0.0000** | 0.0000 | 0 |
| `mel` Melanoma | malignant | 0.6667 | **0.2500** | 0.3636 | 8 |
| `nv` Mole | benign | 0.6739 | **0.8611** | 0.7561 | 36 |
| `vasc` Vascular lesion | benign | 0.0000 | **0.0000** | 0.0000 | 0 |

## Screening view

Collapsing the seven classes into "needs escalation" (`akiec`, `bcc`, `mel`) versus not:

| Metric | Value |
|---|---:|
| Sensitivity | **0.9098** |
| Specificity | 0.9857 |
| Missed serious cases (false negatives) | **22** |
| False alarms | 1 |

> In a screening tool the false negatives in that table are the number that matters.
> A missed melanoma is a materially worse outcome than an unnecessary referral, and
> the report should discuss this asymmetry rather than averaging it away.

## Inference latency

| Metric | Value |
|---|---:|
| Mean | 9.8 ms |
| Median | 9.8 ms |
| p95 | 10.3 ms |
| Device | cuda |
| Parameters | 10,706,991 |
| Model size | 40.8 MB |

## Classification report

```
              precision    recall  f1-score   support

       akiec     0.8333    0.6542    0.7330       107
         bcc     0.8088    0.8527    0.8302       129
         bkl     0.5778    0.7647    0.6582        34
          df     0.0000    0.0000    0.0000         0
         mel     0.6667    0.2500    0.3636         8
          nv     0.6739    0.8611    0.7561        36
        vasc     0.0000    0.0000    0.0000         0

    accuracy                         0.7611       314
   macro avg     0.5086    0.4832    0.4773       314
weighted avg     0.7731    0.7611    0.7581       314
```

## Figures

- `confusion_matrix.png`
- `reliability_diagram.png`
- `training_curves.png`
