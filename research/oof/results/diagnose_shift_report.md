# A.4 -- OOF Fold-Model vs. Frozen Full-Train Model: Validation Shift Diagnostic

Fold models trained on ~80% of train, never saw val during training or checkpoint selection -- same as the frozen full-train checkpoints. If the 5-fold-bagged prediction is systematically weaker or less confident than the frozen model on the identical val split, an OOF-fitted calibrator/threshold handed to the frozen model risks over-sharpening.

| Arch | Macro-F1 (fold-bagged) | Macro-F1 (frozen) | ΔF1 | Mean max-prob (fold) | Mean max-prob (frozen) | Δconf | ECE (fold) | ECE (frozen) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `convnext_tiny` | 0.7847 | 0.7482 | +0.0365 | 0.6963 | 0.7711 | -0.0748 | 0.1699 | 0.1146 |
| `convnext_small` | 0.7735 | 0.7435 | +0.0300 | 0.7316 | 0.7851 | -0.0534 | 0.1346 | 0.0965 |
| `densenet121` | 0.7584 | 0.7210 | +0.0374 | 0.7073 | 0.7434 | -0.0361 | 0.1517 | 0.1008 |
| `efficientnet_b0` | 0.7052 | 0.7249 | -0.0197 | 0.6720 | 0.7353 | -0.0633 | 0.1724 | 0.1060 |
| `efficientnet_b3` | 0.7384 | 0.7068 | +0.0316 | 0.6760 | 0.7009 | -0.0248 | 0.1712 | 0.1471 |
| `resnet50` | 0.7596 | 0.7401 | +0.0195 | 0.6850 | 0.7385 | -0.0536 | 0.1773 | 0.1212 |

Negative ΔF1 / Δconf means the fold-bagged model is weaker/less confident than the frozen model -- the direction the plan predicts (fold models see less data). A gap here is not itself disqualifying; it quantifies the stacking mismatch A.4 asks OOF-fitted results to state honestly, not a reason to discard OOF fitting.
