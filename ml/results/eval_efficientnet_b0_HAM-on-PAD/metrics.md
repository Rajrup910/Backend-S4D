# Evaluation — efficientnet_b0

**Checkpoint:** `ml/checkpoints/efficientnet_b0_best.HAM-only.pt`  
**Split:** test (314 images)  
**Selected on:** validation macro_f1 = 0.7249 (epoch 23)  
**Class mapping:** v1.0.0  
**Temperature:** 4.712

## Headline metrics

| Metric | Value |
|---|---:|
| Macro F1 | **0.1371** |
| Balanced accuracy | **0.2582** |
| Weighted F1 | 0.2710 |
| Accuracy | 0.2611 |
| Cohen's kappa | 0.1187 |
| Macro ROC-AUC (OvR) | nan |
| Expected calibration error | 0.0399 |

> Accuracy is listed but is not the headline. `nv` is roughly two thirds of this
> dataset, so a model that answers "mole" every time scores around 0.67 accuracy
> while missing every melanoma. Macro F1 and per-class recall are the numbers that
> describe whether this model is useful.

## Per-class performance

| Class | Malignancy | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| `akiec` Actinic keratosis | premalignant | 0.3684 | **0.0654** | 0.1111 | 107 |
| `bcc` Basal cell carcinoma | malignant | 0.7414 | **0.3333** | 0.4599 | 129 |
| `bkl` Benign keratosis | benign | 0.0714 | **0.0588** | 0.0645 | 34 |
| `df` Dermatofibroma | benign | 0.0000 | **0.0000** | 0.0000 | 0 |
| `mel` Melanoma | malignant | 0.0000 | **0.0000** | 0.0000 | 8 |
| `nv` Mole | benign | 0.2013 | **0.8333** | 0.3243 | 36 |
| `vasc` Vascular lesion | benign | 0.0000 | **0.0000** | 0.0000 | 0 |

## Screening view

Collapsing the seven classes into "needs escalation" (`akiec`, `bcc`, `mel`) versus not:

| Metric | Value |
|---|---:|
| Sensitivity | **0.3525** |
| Specificity | 0.9429 |
| Missed serious cases (false negatives) | **158** |
| False alarms | 4 |

> In a screening tool the false negatives in that table are the number that matters.
> A missed melanoma is a materially worse outcome than an unnecessary referral, and
> the report should discuss this asymmetry rather than averaging it away.

## Inference latency

| Metric | Value |
|---|---:|
| Mean | 7.9 ms |
| Median | 6.0 ms |
| p95 | 16.3 ms |
| Device | cuda |
| Parameters | 4,016,515 |
| Model size | 15.3 MB |

## Classification report

```
              precision    recall  f1-score   support

       akiec     0.3684    0.0654    0.1111       107
         bcc     0.7414    0.3333    0.4599       129
         bkl     0.0714    0.0588    0.0645        34
          df     0.0000    0.0000    0.0000         0
         mel     0.0000    0.0000    0.0000         8
          nv     0.2013    0.8333    0.3243        36
        vasc     0.0000    0.0000    0.0000         0

    accuracy                         0.2611       314
   macro avg     0.1975    0.1844    0.1371       314
weighted avg     0.4609    0.2611    0.2710       314
```

## Figures

- `confusion_matrix.png`
- `reliability_diagram.png`
- `training_curves.png`
