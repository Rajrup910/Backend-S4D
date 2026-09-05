# Evaluation — convnext_tiny

**Checkpoint:** `ml/checkpoints/convnext_tiny_best.PADwarm.pt`  
**Split:** test (314 images)  
**Selected on:** validation macro_f1 = 0.5702 (epoch 15)  
**Class mapping:** v1.0.0  
**Temperature:** 1.169

## Headline metrics

| Metric | Value |
|---|---:|
| Macro F1 | **0.5429** |
| Balanced accuracy | **0.7571** |
| Weighted F1 | 0.8117 |
| Accuracy | 0.8121 |
| Cohen's kappa | 0.7266 |
| Macro ROC-AUC (OvR) | nan |
| Expected calibration error | 0.0790 |

> Accuracy is listed but is not the headline. `nv` is roughly two thirds of this
> dataset, so a model that answers "mole" every time scores around 0.67 accuracy
> while missing every melanoma. Macro F1 and per-class recall are the numbers that
> describe whether this model is useful.

## Per-class performance

| Class | Malignancy | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| `akiec` Actinic keratosis | premalignant | 0.8286 | **0.8131** | 0.8208 | 107 |
| `bcc` Basal cell carcinoma | malignant | 0.8346 | **0.8605** | 0.8473 | 129 |
| `bkl` Benign keratosis | benign | 0.7429 | **0.7647** | 0.7536 | 34 |
| `df` Dermatofibroma | benign | 0.0000 | **0.0000** | 0.0000 | 0 |
| `mel` Melanoma | malignant | 0.6250 | **0.6250** | 0.6250 | 8 |
| `nv` Mole | benign | 0.7879 | **0.7222** | 0.7536 | 36 |
| `vasc` Vascular lesion | benign | 0.0000 | **0.0000** | 0.0000 | 0 |

## Screening view

Collapsing the seven classes into "needs escalation" (`akiec`, `bcc`, `mel`) versus not:

| Metric | Value |
|---|---:|
| Sensitivity | **0.9754** |
| Specificity | 0.8857 |
| Missed serious cases (false negatives) | **6** |
| False alarms | 8 |

> In a screening tool the false negatives in that table are the number that matters.
> A missed melanoma is a materially worse outcome than an unnecessary referral, and
> the report should discuss this asymmetry rather than averaging it away.

## Inference latency

| Metric | Value |
|---|---:|
| Mean | 4.7 ms |
| Median | 4.4 ms |
| p95 | 5.8 ms |
| Device | cuda |
| Parameters | 27,825,511 |
| Model size | 106.1 MB |

## Classification report

```
              precision    recall  f1-score   support

       akiec     0.8286    0.8131    0.8208       107
         bcc     0.8346    0.8605    0.8473       129
         bkl     0.7429    0.7647    0.7536        34
          df     0.0000    0.0000    0.0000         0
         mel     0.6250    0.6250    0.6250         8
          nv     0.7879    0.7222    0.7536        36
        vasc     0.0000    0.0000    0.0000         0

    accuracy                         0.8121       314
   macro avg     0.5456    0.5408    0.5429       314
weighted avg     0.8119    0.8121    0.8117       314
```

## Figures

- `confusion_matrix.png`
- `reliability_diagram.png`
- `training_curves.png`
