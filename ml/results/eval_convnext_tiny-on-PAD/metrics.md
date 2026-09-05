# Evaluation — convnext_tiny

**Checkpoint:** `ml/checkpoints/convnext_tiny_best.PAD-only.pt`  
**Split:** test (314 images)  
**Selected on:** validation macro_f1 = 0.5644 (epoch 14)  
**Class mapping:** v1.0.0  
**Temperature:** 1.172

## Headline metrics

| Metric | Value |
|---|---:|
| Macro F1 | **0.4690** |
| Balanced accuracy | **0.6544** |
| Weighted F1 | 0.7610 |
| Accuracy | 0.7675 |
| Cohen's kappa | 0.6618 |
| Macro ROC-AUC (OvR) | nan |
| Expected calibration error | 0.0518 |

> Accuracy is listed but is not the headline. `nv` is roughly two thirds of this
> dataset, so a model that answers "mole" every time scores around 0.67 accuracy
> while missing every melanoma. Macro F1 and per-class recall are the numbers that
> describe whether this model is useful.

## Per-class performance

| Class | Malignancy | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| `akiec` Actinic keratosis | premalignant | 0.7938 | **0.7196** | 0.7549 | 107 |
| `bcc` Basal cell carcinoma | malignant | 0.7810 | **0.8295** | 0.8045 | 129 |
| `bkl` Benign keratosis | benign | 0.6842 | **0.7647** | 0.7222 | 34 |
| `df` Dermatofibroma | benign | 0.0000 | **0.0000** | 0.0000 | 0 |
| `mel` Melanoma | malignant | 1.0000 | **0.1250** | 0.2222 | 8 |
| `nv` Mole | benign | 0.7317 | **0.8333** | 0.7792 | 36 |
| `vasc` Vascular lesion | benign | 0.0000 | **0.0000** | 0.0000 | 0 |

## Screening view

Collapsing the seven classes into "needs escalation" (`akiec`, `bcc`, `mel`) versus not:

| Metric | Value |
|---|---:|
| Sensitivity | **0.9426** |
| Specificity | 0.9286 |
| Missed serious cases (false negatives) | **14** |
| False alarms | 5 |

> In a screening tool the false negatives in that table are the number that matters.
> A missed melanoma is a materially worse outcome than an unnecessary referral, and
> the report should discuss this asymmetry rather than averaging it away.

## Inference latency

| Metric | Value |
|---|---:|
| Mean | 6.7 ms |
| Median | 4.9 ms |
| p95 | 10.7 ms |
| Device | cuda |
| Parameters | 27,825,511 |
| Model size | 106.1 MB |

## Classification report

```
              precision    recall  f1-score   support

       akiec     0.7938    0.7196    0.7549       107
         bcc     0.7810    0.8295    0.8045       129
         bkl     0.6842    0.7647    0.7222        34
          df     0.0000    0.0000    0.0000         0
         mel     1.0000    0.1250    0.2222         8
          nv     0.7317    0.8333    0.7792        36
        vasc     0.0000    0.0000    0.0000         0

    accuracy                         0.7675       314
   macro avg     0.5701    0.4674    0.4690       314
weighted avg     0.7748    0.7675    0.7610       314
```

## Figures

- `confusion_matrix.png`
- `reliability_diagram.png`
- `training_curves.png`
