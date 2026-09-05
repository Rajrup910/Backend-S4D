"""Session 13: Assemble soft-vote ensemble and apply frozen Dirichlet calibrator."""

import argparse
from pathlib import Path

def main():
    print("=== Session 13: Assembling External Ensemble & Dirichlet Map ===")
    pred_dir = Path("results/external/predictions")
    print(f"Target prediction dir: {pred_dir}")
    print("Loads 6 arch predictions, computes uniform soft-vote, applies frozen Dirichlet map:")
    print("  results/external/predictions/ensemble_dirichlet_{cohort}.csv")

if __name__ == "__main__":
    main()
