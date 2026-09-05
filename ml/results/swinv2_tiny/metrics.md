# Evaluation — swinv2_tiny

**Checkpoint:** `ml/checkpoints/swinv2_tiny_best.pt`  
**Split:** test (1,502 images)  
**Selected on:** validation macro_f1 = 0.7412 (epoch 14)  
**Class mapping:** v1.0.0  
**Temperature:** 1.000 (uncalibrated)

## Headline metrics

| Metric | Value |
|---|---:|
| Macro F1 | **0.7273** |
| Balanced accuracy | **0.7116** |
| Weighted F1 | 0.8347 |
| Accuracy | 0.8382 |
| Cohen's kappa | 0.6849 |
| Macro ROC-AUC (OvR) | 0.9038 |
| Expected calibration error | 0.1077 |

> Accuracy is listed but is not the headline. `nv` is roughly two thirds of this
> dataset, so a model that answers "mole" every time scores around 0.67 accuracy
> while missing every melanoma. Macro F1 and per-class recall are the numbers that
> describe whether this model is useful.

## Per-class performance

| Class | Malignancy | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| `akiec` Actinic keratosis | premalignant | 0.6129 | **0.7308** | 0.6667 | 52 |
| `bcc` Basal cell carcinoma | malignant | 0.7826 | **0.7606** | 0.7714 | 71 |
| `bkl` Benign keratosis | benign | 0.6588 | **0.6707** | 0.6647 | 167 |
| `df` Dermatofibroma | benign | 0.8333 | **0.5000** | 0.6250 | 20 |
| `mel` Melanoma | malignant | 0.6567 | **0.5269** | 0.5847 | 167 |
| `nv` Mole | benign | 0.9081 | **0.9353** | 0.9215 | 1,004 |
| `vasc` Vascular lesion | benign | 0.8571 | **0.8571** | 0.8571 | 21 |

## Screening view

Collapsing the seven classes into "needs escalation" (`akiec`, `bcc`, `mel`) versus not:

| Metric | Value |
|---|---:|
| Sensitivity | **0.6621** |
| Specificity | 0.9398 |
| Missed serious cases (false negatives) | **98** |
| False alarms | 73 |

> In a screening tool the false negatives in that table are the number that matters.
> A missed melanoma is a materially worse outcome than an unnecessary referral, and
> the report should discuss this asymmetry rather than averaging it away.

## Inference latency

| Metric | Value |
|---|---:|
| Mean | 12.1 ms |
| Median | 10.4 ms |
| p95 | 21.1 ms |
| Device | cuda |
| Parameters | 27,569,851 |
| Model size | 105.2 MB |

## Classification report

```
              precision    recall  f1-score   support

       akiec     0.6129    0.7308    0.6667        52
         bcc     0.7826    0.7606    0.7714        71
         bkl     0.6588    0.6707    0.6647       167
          df     0.8333    0.5000    0.6250        20
         mel     0.6567    0.5269    0.5847       167
          nv     0.9081    0.9353    0.9215      1004
        vasc     0.8571    0.8571    0.8571        21

    accuracy                         0.8382      1502
   macro avg     0.7585    0.7116    0.7273      1502
weighted avg     0.8346    0.8382    0.8347      1502
```

## Figures

- `confusion_matrix.png`
- `reliability_diagram.png`
- `training_curves.png`
