"""Machine-learning package for Scan4Disease.

Layout:
    ml.configs         class mapping + training config (data files, not code)
    ml.preprocessing   dataset preparation, validation, leakage-aware splitting
    ml.training        transfer-learning training loop
    ml.evaluation      held-out test evaluation and model comparison
    ml.explainability  Grad-CAM

Nothing in this package is imported by the backend at runtime. The backend loads a
checkpoint and `ml/configs/class_mapping.json`, and nothing else -- training code must
not live inside the API server (master spec, section 21).
"""

__version__ = "0.1.0"
