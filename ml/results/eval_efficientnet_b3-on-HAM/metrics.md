# Evaluation — efficientnet_b3

**Checkpoint:** `ml/checkpoints/efficientnet_b3_best.pt`  
**Split:** test (1,502 images)  
**Selected on:** validation macro_f1 = 0.7118 (epoch 15)  
**Class mapping:** v1.0.0  
**Temperature:** 0.711

## Headline metrics

| Metric | Value |
|---|---:|
| Macro F1 | **0.6894** |
| Balanced accuracy | **0.7126** |
| Weighted F1 | 0.8240 |
| Accuracy | 0.8276 |
| Cohen's kappa | 0.6654 |
| Macro ROC-AUC (OvR) | 0.9434 |
| Expected calibration error | 0.0481 |

> Accuracy is listed but is not the headline. `nv` is roughly two thirds of this
> dataset, so a model that answers "mole" every time scores around 0.67 accuracy
> while missing every melanoma. Macro F1 and per-class recall are the numbers that
> describe whether this model is useful.

## Per-class performance

| Class | Malignancy | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| `akiec` Actinic keratosis | premalignant | 0.4667 | **0.8077** | 0.5915 | 52 |
| `bcc` Basal cell carcinoma | malignant | 0.7179 | **0.7887** | 0.7517 | 71 |
| `bkl` Benign keratosis | benign | 0.6885 | **0.5030** | 0.5813 | 167 |
| `df` Dermatofibroma | benign | 0.4815 | **0.6500** | 0.5532 | 20 |
| `mel` Melanoma | malignant | 0.6870 | **0.5389** | 0.6040 | 167 |
| `nv` Mole | benign | 0.9093 | **0.9382** | 0.9235 | 1,004 |
| `vasc` Vascular lesion | benign | 0.8889 | **0.7619** | 0.8205 | 21 |

## Screening view

Collapsing the seven classes into "needs escalation" (`akiec`, `bcc`, `mel`) versus not:

| Metric | Value |
|---|---:|
| Sensitivity | **0.7276** |
| Specificity | 0.9274 |
| Missed serious cases (false negatives) | **79** |
| False alarms | 88 |

> In a screening tool the false negatives in that table are the number that matters.
> A missed melanoma is a materially worse outcome than an unnecessary referral, and
> the report should discuss this asymmetry rather than averaging it away.

## Inference latency

| Metric | Value |
|---|---:|
| Mean | 9.4 ms |
| Median | 9.4 ms |
| p95 | 9.7 ms |
| Device | cuda |
| Parameters | 10,706,991 |
| Model size | 40.8 MB |

## Classification report

```
              precision    recall  f1-score   support

       akiec     0.4667    0.8077    0.5915        52
         bcc     0.7179    0.7887    0.7517        71
         bkl     0.6885    0.5030    0.5813       167
          df     0.4815    0.6500    0.5532        20
         mel     0.6870    0.5389    0.6040       167
          nv     0.9093    0.9382    0.9235      1004
        vasc     0.8889    0.7619    0.8205        21

    accuracy                         0.8276      1502
   macro avg     0.6914    0.7126    0.6894      1502
weighted avg     0.8297    0.8276    0.8240      1502
```

## Figures

- `confusion_matrix.png`
- `reliability_diagram.png`
- `training_curves.png`
