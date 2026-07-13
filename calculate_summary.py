import pandas as pd
from pathlib import Path


def calculate_and_display_summary():
    results_dir = Path("results")
    root_dir = Path(".")

    # Look for summary CSV files either in root or inside results/
    synthetic_csv = (
        results_dir / "baseline_summary_synthetic.csv"
        if (results_dir / "baseline_summary_synthetic.csv").exists()
        else root_dir / "baseline_summary_synthetic.csv"
    )
    real_csv = (
        results_dir / "baseline_summary_real.csv"
        if (results_dir / "baseline_summary_real.csv").exists()
        else root_dir / "baseline_summary_real.csv"
    )

    # 1. Synthetic Dataset Summary
    if synthetic_csv.exists():
        print("=" * 80)
        print(" SYNTHETIC DATASET BASELINE EVALUATION SUMMARY")
        print("=" * 80)
        df_synth = pd.read_csv(synthetic_csv)
        summary_synth = (
            df_synth.groupby("baseline")
            .agg(
                {
                    "psnr": "mean",
                    "ssim": "mean",
                    "cer": "mean",
                    "wer": "mean",
                    "field_accuracy": "mean",
                    "trace_f1": "mean",
                    "faithfulness_auc": "mean",
                    "ece": "mean",
                    "abstention_rate": "mean",
                }
            )
            .reset_index()
        )
        print(summary_synth.to_string(index=False))
        print("\n")
    else:
        print("Synthetic evaluation results not found.")

    # 2. Real Dataset Summary
    if real_csv.exists():
        print("=" * 80)
        print(" REAL-WORLD DATASET BASELINE EVALUATION SUMMARY")
        print("=" * 80)
        df_real = pd.read_csv(real_csv)
        summary_real = (
            df_real.groupby("baseline")
            .agg(
                {
                    "cer": "mean",
                    "wer": "mean",
                    "field_accuracy": "mean",
                    "high_risk_accuracy": "mean",
                    "abstention_rate": "mean",
                    "accepted_field_accuracy": "mean",
                }
            )
            .reset_index()
        )
        print(summary_real.to_string(index=False))
        print("=" * 80)
    else:
        print("Real evaluation results not found yet. Make sure run_evaluation.py finishes!")


if __name__ == "__main__":
    calculate_and_display_summary()