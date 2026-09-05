# Evaluation — efficientnet_b3

**Checkpoint:** `ml/checkpoints/efficientnet_b3_best.combined.pt`  
**Split:** test (1,489 images)  
**Selected on:** validation macro_f1 = 0.7500 (epoch 15)  
**Class mapping:** v1.0.0  
**Temperature:** 1.000 (uncalibrated)

## Headline metrics

| Metric | Value |
|---|---:|
| Macro F1 | **0.7019** |
| Balanced accuracy | **0.7158** |
| Weighted F1 | 0.8294 |
| Accuracy | 0.8281 |
| Cohen's kappa | 0.6714 |
| Macro ROC-AUC (OvR) | 0.9500 |
| Expected calibration error | 0.0986 |

> Accuracy is listed but is not the headline. `nv` is roughly two thirds of this
> dataset, so a model that answers "mole" every time scores around 0.67 accuracy
> while missing every melanoma. Macro F1 and per-class recall are the numbers that
> describe whether this model is useful.

## Per-class performance

| Class | Malignancy | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| `akiec` Actinic keratosis | premalignant | 0.5139 | **0.7400** | 0.6066 | 50 |
| `bcc` Basal cell carcinoma | malignant | 0.7797 | **0.6765** | 0.7244 | 68 |
| `bkl` Benign keratosis | benign | 0.7984 | **0.5723** | 0.6667 | 173 |
| `df` Dermatofibroma | benign | 0.6667 | **0.6250** | 0.6452 | 16 |
| `mel` Melanoma | malignant | 0.5412 | **0.6522** | 0.5915 | 161 |
| `nv` Mole | benign | 0.9188 | **0.9188** | 0.9188 | 998 |
| `vasc` Vascular lesion | benign | 0.7037 | **0.8261** | 0.7600 | 23 |

## Screening view

Collapsing the seven classes into "needs escalation" (`akiec`, `bcc`, `mel`) versus not:

| Metric | Value |
|---|---:|
| Sensitivity | **0.7276** |
| Specificity | 0.8992 |
| Missed serious cases (false negatives) | **76** |
| False alarms | 122 |

> In a screening tool the false negatives in that table are the number that matters.
> A missed melanoma is a materially worse outcome than an unnecessary referral, and
> the report should discuss this asymmetry rather than averaging it away.

## Inference latency

| Metric | Value |
|---|---:|
| Mean | 10.9 ms |
| Median | 10.6 ms |
| p95 | 12.6 ms |
| Device | cuda |
| Parameters | 10,706,991 |
| Model size | 40.8 MB |

## Classification report

```
              precision    recall  f1-score   support

       akiec     0.5139    0.7400    0.6066        50
         bcc     0.7797    0.6765    0.7244        68
         bkl     0.7984    0.5723    0.6667       173
          df     0.6667    0.6250    0.6452        16
         mel     0.5412    0.6522    0.5915       161
          nv     0.9188    0.9188    0.9188       998
        vasc     0.7037    0.8261    0.7600        23

    accuracy                         0.8281      1489
   macro avg     0.7032    0.7158    0.7019      1489
weighted avg     0.8380    0.8281    0.8294      1489
```

## Figures

- `confusion_matrix.png`
- `reliability_diagram.png`
- `training_curves.png`
