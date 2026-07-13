import csv
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

from baselines import BaselineName, run_baseline
from dataset_utils import build_synthetic_dataset
from metrics import (
    compute_abstention_metrics,
    compute_ece,
    compute_field_accuracy,
    compute_faithfulness_auc,
    compute_image_metrics,
    compute_trace_f1,
)


def evaluate_synthetic_dataset(synthetic_dir: Path, results_dir: Path):
    metadata_path = synthetic_dir / "metadata.csv"
    if not metadata_path.exists():
        print(f"Synthetic metadata.csv not found at {metadata_path}. Skipping synthetic evaluation.")
        return

    with metadata_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"\n--- Running Synthetic Dataset Evaluation ({len(rows)} samples) ---", flush=True)
    all_rows = []
    baselines_to_run = list(BaselineName)

    for row in tqdm(rows, desc="Evaluating Synthetic Data", unit="img"):
        clean_path = Path(row["clean_path"])
        degraded_path = Path(row["degraded_path"])
        gt_fields = {"date": "", "receipt_no": "", "tax_id": "", "tax_amount": "", "total_amount": ""}

        try:
            clean_image = np.array(Image.open(clean_path).convert("RGB"))
        except Exception as e:
            print(f"\nFailed to load clean image {clean_path}: {e}", flush=True)
            continue

        for baseline in baselines_to_run:
            try:
                result = run_baseline(str(degraded_path), baseline)
                image_metrics = compute_image_metrics(clean_image, result.restored_image)
                field_metrics = compute_field_accuracy(gt_fields, result.fields)

                mask_path = Path(row["mask_path"])
                mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
                if mask_image is None:
                    mask_image = np.zeros(clean_image.shape[:2], dtype=np.uint8)

                pred_map = result.authenticity_map if result.authenticity_map is not None else np.zeros(mask_image.shape, dtype=np.float32)

                trace_f1 = compute_trace_f1(mask_image, pred_map)
                faith_auc = compute_faithfulness_auc(mask_image, pred_map)
                ece_score = compute_ece(mask_image, pred_map)
                abst_metrics = compute_abstention_metrics(gt_fields, result.fields, result.abstained_fields or [])

                all_rows.append(
                    {
                        "image_id": row["image_id"],
                        "baseline": baseline.value,
                        "severity": row.get("severity", "unknown"),
                        "psnr": image_metrics["psnr"],
                        "ssim": image_metrics["ssim"],
                        "cer": field_metrics.cer,
                        "wer": field_metrics.wer,
                        "field_accuracy": field_metrics.overall_accuracy,
                        "high_risk_accuracy": field_metrics.high_risk_accuracy,
                        "trace_f1": trace_f1,
                        "faithfulness_auc": faith_auc,
                        "ece": ece_score,
                        "abstention_rate": abst_metrics.abstention_rate,
                        "accepted_field_accuracy": abst_metrics.accepted_field_accuracy,
                        "unsupported_acceptance_rate": abst_metrics.unsupported_acceptance_rate,
                        "precision_on_accepted_high_risk": abst_metrics.precision_on_accepted_high_risk,
                    }
                )
            except Exception as e:
                print(f"\nError running baseline {baseline.value} on synthetic {row['image_id']}: {e}", flush=True)

    if all_rows:
        csv_path = results_dir / "baseline_summary_synthetic.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)

        md_path = results_dir / "baseline_summary_synthetic.md"
        with md_path.open("w", encoding="utf-8") as f:
            f.write("# Synthetic Baseline Evaluation Summary\n\n")
            f.write("| image_id | baseline | severity | psnr | ssim | cer | wer | field_accuracy | high_risk_accuracy | trace_f1 | faithfulness_auc | ece | abstention_rate | accepted_field_accuracy | unsupported_acceptance_rate | precision_on_accepted_high_risk |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
            for r in all_rows:
                f.write(
                    f"| {r['image_id']} | {r['baseline']} | {r['severity']} | {r['psnr']:.2f} | {r['ssim']:.3f} | {r['cer']:.3f} | {r['wer']:.3f} | {r['field_accuracy']:.3f} | {r['high_risk_accuracy']:.3f} | {r['trace_f1']:.3f} | {r['faithfulness_auc']:.3f} | {r['ece']:.3f} | {r['abstention_rate']:.3f} | {r['accepted_field_accuracy']:.3f} | {r['unsupported_acceptance_rate']:.3f} | {r['precision_on_accepted_high_risk']:.3f} |\n"
                )
        print(f"Synthetic evaluation saved to {csv_path} and {md_path}")


def evaluate_real_dataset(real_dir: Path, results_dir: Path):
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    
    # Scan Datasets/Real life/ directly for image files, filtering out files starting with 'X5'
    real_images = [
        f for f in sorted(real_dir.iterdir())
        if f.is_file() 
        and f.suffix.lower() in valid_exts 
        #and not f.name.startswith("X5")
    ]

    if not real_images:
        print(f"\nNo real receipt images found in {real_dir} (excluding 'X5...' files). Skipping real evaluation.")
        return

    print(f"\n--- Running Real-World Dataset Evaluation ({len(real_images)} samples) ---", flush=True)
    all_rows = []
    baselines_to_run = list(BaselineName)

    for img_path in tqdm(real_images, desc="Evaluating Real Receipts", unit="img"):
        image_id = img_path.stem
        # Default placeholder fields for real receipts without manual transcription
        gt_fields = {"date": "", "receipt_no": "", "tax_id": "", "tax_amount": "", "total_amount": ""}

        for baseline in baselines_to_run:
            try:
                result = run_baseline(str(img_path), baseline)
                field_metrics = compute_field_accuracy(gt_fields, result.fields)
                abst_metrics = compute_abstention_metrics(gt_fields, result.fields, result.abstained_fields or [])

                all_rows.append(
                    {
                        "image_id": image_id,
                        "baseline": baseline.value,
                        "cer": field_metrics.cer,
                        "wer": field_metrics.wer,
                        "field_accuracy": field_metrics.overall_accuracy,
                        "high_risk_accuracy": field_metrics.high_risk_accuracy,
                        "abstention_rate": abst_metrics.abstention_rate,
                        "accepted_field_accuracy": abst_metrics.accepted_field_accuracy,
                        "unsupported_acceptance_rate": abst_metrics.unsupported_acceptance_rate,
                        "precision_on_accepted_high_risk": abst_metrics.precision_on_accepted_high_risk,
                        "notes": result.notes or "",
                    }
                )
            except Exception as e:
                print(f"\nError running baseline {baseline.value} on real receipt {image_id}: {e}", flush=True)

    if all_rows:
        csv_path = results_dir / "baseline_summary_real.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)

        md_path = results_dir / "baseline_summary_real.md"
        with md_path.open("w", encoding="utf-8") as f:
            f.write("# Real-World Baseline Evaluation Summary\n\n")
            f.write("| image_id | baseline | cer | wer | field_accuracy | high_risk_accuracy | abstention_rate | accepted_field_accuracy | unsupported_acceptance_rate | precision_on_accepted_high_risk |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|\n")
            for r in all_rows:
                f.write(
                    f"| {r['image_id']} | {r['baseline']} | {r['cer']:.3f} | {r['wer']:.3f} | {r['field_accuracy']:.3f} | {r['high_risk_accuracy']:.3f} | {r['abstention_rate']:.3f} | {r['accepted_field_accuracy']:.3f} | {r['unsupported_acceptance_rate']:.3f} | {r['precision_on_accepted_high_risk']:.3f} |\n"
                )
        print(f"Real-world evaluation saved to {csv_path} and {md_path}")


def main():
    workspace = Path(__file__).parent
    dataset_dir = workspace / "Datasets" / "Real life"
    clean_dir = dataset_dir / "clean"
    synthetic_dir = dataset_dir / "synthetic"
    results_dir = workspace / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # COMMENT OUT IF ONLY NEED ON REAL DATASWT NOT SYNTHETIC
    # if not synthetic_dir.exists():
    #     print("Synthetic dataset directory not found. Generating synthetic dataset...", flush=True)
    #     build_synthetic_dataset(str(clean_dir), str(synthetic_dir), target_count=100)

    # 1. Run evaluation on synthetic dataset
    # evaluate_synthetic_dataset(synthetic_dir, results_dir)

    # 2. Run evaluation on real dataset directly in 'Datasets/Real life/'
    evaluate_real_dataset(dataset_dir, results_dir)


if __name__ == "__main__":
    main()