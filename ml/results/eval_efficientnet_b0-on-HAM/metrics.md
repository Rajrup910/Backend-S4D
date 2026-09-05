# Evaluation — efficientnet_b0

**Checkpoint:** `ml/checkpoints/efficientnet_b0_best.HAM-only.pt`  
**Split:** test (1,502 images)  
**Selected on:** validation macro_f1 = 0.7249 (epoch 23)  
**Class mapping:** v1.0.0  
**Temperature:** 0.781

## Headline metrics

| Metric | Value |
|---|---:|
| Macro F1 | **0.7264** |
| Balanced accuracy | **0.7390** |
| Weighted F1 | 0.8327 |
| Accuracy | 0.8342 |
| Cohen's kappa | 0.6802 |
| Macro ROC-AUC (OvR) | 0.9539 |
| Expected calibration error | 0.0319 |

> Accuracy is listed but is not the headline. `nv` is roughly two thirds of this
> dataset, so a model that answers "mole" every time scores around 0.67 accuracy
> while missing every melanoma. Macro F1 and per-class recall are the numbers that
> describe whether this model is useful.

## Per-class performance

| Class | Malignancy | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| `akiec` Actinic keratosis | premalignant | 0.5600 | **0.8077** | 0.6614 | 52 |
| `bcc` Basal cell carcinoma | malignant | 0.7826 | **0.7606** | 0.7714 | 71 |
| `bkl` Benign keratosis | benign | 0.8529 | **0.5210** | 0.6468 | 167 |
| `df` Dermatofibroma | benign | 0.6087 | **0.7000** | 0.6512 | 20 |
| `mel` Melanoma | malignant | 0.5714 | **0.6467** | 0.6067 | 167 |
| `nv` Mole | benign | 0.9092 | **0.9273** | 0.9181 | 1,004 |
| `vasc` Vascular lesion | benign | 0.8500 | **0.8095** | 0.8293 | 21 |

## Screening view

Collapsing the seven classes into "needs escalation" (`akiec`, `bcc`, `mel`) versus not:

| Metric | Value |
|---|---:|
| Sensitivity | **0.7862** |
| Specificity | 0.9134 |
| Missed serious cases (false negatives) | **62** |
| False alarms | 105 |

> In a screening tool the false negatives in that table are the number that matters.
> A missed melanoma is a materially worse outcome than an unnecessary referral, and
> the report should discuss this asymmetry rather than averaging it away.

## Inference latency

| Metric | Value |
|---|---:|
| Mean | 8.5 ms |
| Median | 6.0 ms |
| p95 | 16.2 ms |
| Device | cuda |
| Parameters | 4,016,515 |
| Model size | 15.3 MB |

## Classification report

```
              precision    recall  f1-score   support

       akiec     0.5600    0.8077    0.6614        52
         bcc     0.7826    0.7606    0.7714        71
         bkl     0.8529    0.5210    0.6468       167
          df     0.6087    0.7000    0.6512        20
         mel     0.5714    0.6467    0.6067       167
          nv     0.9092    0.9273    0.9181      1004
        vasc     0.8500    0.8095    0.8293        21

    accuracy                         0.8342      1502
   macro avg     0.7336    0.7390    0.7264      1502
weighted avg     0.8425    0.8342    0.8327      1502
```

## Figures

- `confusion_matrix.png`
- `reliability_diagram.png`
- `training_curves.png`
