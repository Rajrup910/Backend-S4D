# Evaluation — resnet50

**Checkpoint:** `ml/checkpoints/resnet50_best.combined.pt`  
**Split:** test (314 images)  
**Selected on:** validation macro_f1 = 0.7408 (epoch 8)  
**Class mapping:** v1.0.0  
**Temperature:** 1.000 (uncalibrated)

## Headline metrics

| Metric | Value |
|---|---:|
| Macro F1 | **0.4478** |
| Balanced accuracy | **0.6101** |
| Weighted F1 | 0.7136 |
| Accuracy | 0.7166 |
| Cohen's kappa | 0.5799 |
| Macro ROC-AUC (OvR) | nan |
| Expected calibration error | 0.0604 |

> Accuracy is listed but is not the headline. `nv` is roughly two thirds of this
> dataset, so a model that answers "mole" every time scores around 0.67 accuracy
> while missing every melanoma. Macro F1 and per-class recall are the numbers that
> describe whether this model is useful.

## Per-class performance

| Class | Malignancy | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| `akiec` Actinic keratosis | premalignant | 0.6800 | **0.8870** | 0.7698 | 115 |
| `bcc` Basal cell carcinoma | malignant | 0.8300 | **0.6434** | 0.7249 | 129 |
| `bkl` Benign keratosis | benign | 0.5882 | **0.6061** | 0.5970 | 33 |
| `df` Dermatofibroma | benign | 0.0000 | **0.0000** | 0.0000 | 0 |
| `mel` Melanoma | malignant | 0.5000 | **0.3333** | 0.4000 | 6 |
| `nv` Mole | benign | 0.7200 | **0.5806** | 0.6429 | 31 |
| `vasc` Vascular lesion | benign | 0.0000 | **0.0000** | 0.0000 | 0 |

## Screening view

Collapsing the seven classes into "needs escalation" (`akiec`, `bcc`, `mel`) versus not:

| Metric | Value |
|---|---:|
| Sensitivity | **0.9440** |
| Specificity | 0.7188 |
| Missed serious cases (false negatives) | **14** |
| False alarms | 18 |

> In a screening tool the false negatives in that table are the number that matters.
> A missed melanoma is a materially worse outcome than an unnecessary referral, and
> the report should discuss this asymmetry rather than averaging it away.

## Inference latency

| Metric | Value |
|---|---:|
| Mean | 4.6 ms |
| Median | 4.6 ms |
| p95 | 4.7 ms |
| Device | cuda |
| Parameters | 23,522,375 |
| Model size | 89.7 MB |

## Classification report

```
              precision    recall  f1-score   support

       akiec     0.6800    0.8870    0.7698       115
         bcc     0.8300    0.6434    0.7249       129
         bkl     0.5882    0.6061    0.5970        33
          df     0.0000    0.0000    0.0000         0
         mel     0.5000    0.3333    0.4000         6
          nv     0.7200    0.5806    0.6429        31
        vasc     0.0000    0.0000    0.0000         0

    accuracy                         0.7166       314
   macro avg     0.4740    0.4358    0.4478       314
weighted avg     0.7325    0.7166    0.7136       314
```

## Figures

- `confusion_matrix.png`
- `reliability_diagram.png`
- `training_curves.png`
