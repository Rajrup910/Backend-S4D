# Evaluation — resnet50

**Checkpoint:** `ml/checkpoints/resnet50_best.PADwarm.pt`  
**Split:** test (314 images)  
**Selected on:** validation macro_f1 = 0.5373 (epoch 11)  
**Class mapping:** v1.0.0  
**Temperature:** 0.993

## Headline metrics

| Metric | Value |
|---|---:|
| Macro F1 | **0.4668** |
| Balanced accuracy | **0.6511** |
| Weighted F1 | 0.7659 |
| Accuracy | 0.7675 |
| Cohen's kappa | 0.6637 |
| Macro ROC-AUC (OvR) | nan |
| Expected calibration error | 0.0839 |

> Accuracy is listed but is not the headline. `nv` is roughly two thirds of this
> dataset, so a model that answers "mole" every time scores around 0.67 accuracy
> while missing every melanoma. Macro F1 and per-class recall are the numbers that
> describe whether this model is useful.

## Per-class performance

| Class | Malignancy | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| `akiec` Actinic keratosis | premalignant | 0.7818 | **0.8037** | 0.7926 | 107 |
| `bcc` Basal cell carcinoma | malignant | 0.8387 | **0.8062** | 0.8221 | 129 |
| `bkl` Benign keratosis | benign | 0.6364 | **0.6176** | 0.6269 | 34 |
| `df` Dermatofibroma | benign | 0.0000 | **0.0000** | 0.0000 | 0 |
| `mel` Melanoma | malignant | 0.4000 | **0.2500** | 0.3077 | 8 |
| `nv` Mole | benign | 0.6667 | **0.7778** | 0.7179 | 36 |
| `vasc` Vascular lesion | benign | 0.0000 | **0.0000** | 0.0000 | 0 |

## Screening view

Collapsing the seven classes into "needs escalation" (`akiec`, `bcc`, `mel`) versus not:

| Metric | Value |
|---|---:|
| Sensitivity | **0.9508** |
| Specificity | 0.9000 |
| Missed serious cases (false negatives) | **12** |
| False alarms | 7 |

> In a screening tool the false negatives in that table are the number that matters.
> A missed melanoma is a materially worse outcome than an unnecessary referral, and
> the report should discuss this asymmetry rather than averaging it away.

## Inference latency

| Metric | Value |
|---|---:|
| Mean | 6.1 ms |
| Median | 4.7 ms |
| p95 | 14.2 ms |
| Device | cuda |
| Parameters | 23,522,375 |
| Model size | 89.7 MB |

## Classification report

```
              precision    recall  f1-score   support

       akiec     0.7818    0.8037    0.7926       107
         bcc     0.8387    0.8062    0.8221       129
         bkl     0.6364    0.6176    0.6269        34
          df     0.0000    0.0000    0.0000         0
         mel     0.4000    0.2500    0.3077         8
          nv     0.6667    0.7778    0.7179        36
        vasc     0.0000    0.0000    0.0000         0

    accuracy                         0.7675       314
   macro avg     0.4748    0.4651    0.4668       314
weighted avg     0.7665    0.7675    0.7659       314
```

## Figures

- `confusion_matrix.png`
- `reliability_diagram.png`
- `training_curves.png`
