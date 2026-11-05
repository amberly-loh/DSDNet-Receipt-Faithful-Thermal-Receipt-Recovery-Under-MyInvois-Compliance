from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn.functional as F
from scipy.stats import friedmanchisquare, wilcoxon

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dsdnet_receipt.config import load_config  # noqa: E402
from dsdnet_receipt.dsdnet_adapter import DSDNetReceiptAdapter, gaussian_kernel  # noqa: E402
from dsdnet_receipt.images import read_gray  # noqa: E402
from dsdnet_receipt.metrics import psnr, ssim  # noqa: E402
from dsdnet_receipt.validation import stroke_fidelity_metrics  # noqa: E402


STAGES = (1, 2, 4, 6, 8)
KERNEL_FACTORS = (0.50, 0.75, 1.00, 1.25, 1.50)
METRICS = ("ssim", "l1", "objective", "stroke_recall", "deleted_stroke_rate", "false_stroke_rate")
HIGHER_IS_BETTER = {"ssim", "stroke_recall"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def centre_crop_pair(clean: np.ndarray, degraded: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = clean.shape
    if height < size or width < size:
        import cv2

        scale = max(size / height, size / width)
        clean = cv2.resize(clean, (int(np.ceil(width * scale)), int(np.ceil(height * scale))), interpolation=cv2.INTER_CUBIC)
        degraded = cv2.resize(degraded, (clean.shape[1], clean.shape[0]), interpolation=cv2.INTER_CUBIC)
        height, width = clean.shape
    top = (height - size) // 2
    left = (width - size) // 2
    return clean[top : top + size, left : left + size], degraded[top : top + size, left : left + size]


def metric_row(clean: np.ndarray, restored_tensor: torch.Tensor, foreground: np.ndarray) -> dict[str, float | int]:
    restored = (restored_tensor.squeeze().detach().cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    fidelity = stroke_fidelity_metrics(clean, restored, foreground)
    clean_tensor = torch.from_numpy(clean).float().div_(255).unsqueeze(0).unsqueeze(0).to(restored_tensor.device)
    l1_value = float(F.l1_loss(restored_tensor, clean_tensor).detach().cpu())
    ssim_value = float(ssim(clean, restored))
    return {
        "psnr": float(psnr(clean, restored)),
        "ssim": ssim_value,
        "l1": l1_value,
        "objective": l1_value + 0.2 * (1.0 - ssim_value),
        **fidelity,
    }


def source_means(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    return frame.groupby(["source_id", group], as_index=False)[list(METRICS)].mean()


def summary_by(frame: pd.DataFrame, group: str) -> list[dict[str, float | int]]:
    source = source_means(frame, group)
    rows = []
    for value, part in source.groupby(group):
        row: dict[str, float | int] = {group: float(value) if isinstance(value, (float, np.floating)) else int(value)}
        row["sources"] = int(part["source_id"].nunique())
        for metric in METRICS:
            row[f"mean_{metric}"] = float(part[metric].mean())
            row[f"sd_{metric}"] = float(part[metric].std(ddof=1))
        rows.append(row)
    return rows


def holm_adjust(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.ones(len(pvalues), dtype=float)
    running = 0.0
    count = len(pvalues)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(pvalues[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def paired_bootstrap(delta: np.ndarray, draws: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = delta.size
    samples = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        samples[index] = float(np.mean(delta[rng.integers(0, n, n)]))
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def paired_tests(frame: pd.DataFrame, group: str, reference: float | int, comparisons: list[float | int], family: str) -> list[dict[str, object]]:
    source = source_means(frame, group)
    outputs: list[dict[str, object]] = []
    raw_p: list[float] = []
    for comparison in comparisons:
        left = source[source[group] == reference].set_index("source_id")
        right = source[source[group] == comparison].set_index("source_id")
        common = sorted(set(left.index) & set(right.index))
        for metric in METRICS:
            reference_values = left.loc[common, metric].to_numpy(float)
            comparison_values = right.loc[common, metric].to_numpy(float)
            if metric in HIGHER_IS_BETTER:
                alternative = "greater"
                benefit = reference_values - comparison_values
            else:
                alternative = "less"
                benefit = comparison_values - reference_values
            try:
                statistic, pvalue = wilcoxon(reference_values, comparison_values, alternative=alternative, zero_method="wilcox")
            except ValueError:
                statistic, pvalue = 0.0, 1.0
            ci_low, ci_high = paired_bootstrap(benefit, 10000, 5600 + len(outputs))
            outputs.append({
                "family": family,
                "metric": metric,
                "reference": reference,
                "comparison": comparison,
                "n_sources": len(common),
                "alternative": alternative,
                "reference_mean": float(np.mean(reference_values)),
                "comparison_mean": float(np.mean(comparison_values)),
                "mean_benefit": float(np.mean(benefit)),
                "median_benefit": float(np.median(benefit)),
                "benefit_ci95_low": ci_low,
                "benefit_ci95_high": ci_high,
                "wilcoxon_statistic": float(statistic),
                "p_raw": float(pvalue),
            })
            raw_p.append(float(pvalue))
    adjusted = holm_adjust(raw_p)
    for row, pvalue in zip(outputs, adjusted):
        row["p_holm"] = pvalue
        row["significant_0_05"] = bool(pvalue < 0.05)
    return outputs


def severity_analysis(validation_records: Path) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    records = pd.read_csv(validation_records)
    records["objective"] = records["l1"] + 0.2 * (1.0 - records["ssim"])
    severity_order = {"light": 0, "moderate": 1, "severe": 2}
    records["severity_order"] = records["severity"].map(severity_order)
    source = records.groupby(["source_id", "method", "severity", "severity_order"], as_index=False)[list(METRICS)].mean()
    summary = []
    for (method, severity, order), part in source.groupby(["method", "severity", "severity_order"]):
        row: dict[str, object] = {"method": method, "severity": severity, "severity_order": int(order), "sources": int(part.source_id.nunique())}
        for metric in METRICS:
            row[f"mean_{metric}"] = float(part[metric].mean())
            row[f"sd_{metric}"] = float(part[metric].std(ddof=1))
        summary.append(row)

    proposed = source[source["method"] == "Proposed_DSDNet_Receipt"]
    wide = {metric: proposed.pivot(index="source_id", columns="severity", values=metric).dropna() for metric in METRICS}
    tests = []
    for metric, table in wide.items():
        statistic, pvalue = friedmanchisquare(table["light"], table["moderate"], table["severe"])
        tests.append({"metric": metric, "test": "Friedman", "n_sources": int(len(table)), "statistic": float(statistic), "p_raw": float(pvalue)})
        pairs = [("light", "moderate"), ("moderate", "severe"), ("light", "severe")]
        pair_rows = []
        for first, second in pairs:
            first_values = table[first].to_numpy(float)
            second_values = table[second].to_numpy(float)
            if metric in HIGHER_IS_BETTER:
                benefit = first_values - second_values
                alternative = "greater"
            else:
                benefit = second_values - first_values
                alternative = "less"
            try:
                stat, p = wilcoxon(first_values, second_values, alternative=alternative, zero_method="wilcox")
            except ValueError:
                stat, p = 0.0, 1.0
            pair_rows.append({
                "metric": metric,
                "test": "paired Wilcoxon",
                "first": first,
                "second": second,
                "n_sources": int(len(table)),
                "alternative": alternative,
                "statistic": float(stat),
                "p_raw": float(p),
                "mean_first_minus_second": float(np.mean(first_values - second_values)),
            })
        adjusted = holm_adjust([float(row["p_raw"]) for row in pair_rows])
        for row, p_adj in zip(pair_rows, adjusted):
            row["p_holm_within_metric"] = p_adj
            row["significant_0_05"] = bool(p_adj < 0.05)
        tests.extend(pair_rows)
    return pd.DataFrame(summary).sort_values(["method", "severity_order"]), tests, summary


def error_analysis(validation_records: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    records = pd.read_csv(validation_records)
    records["objective"] = records["l1"] + 0.2 * (1.0 - records["ssim"])
    source = records.groupby(["source_id", "method"], as_index=False)[list(METRICS)].mean()
    proposed = source[source.method == "Proposed_DSDNet_Receipt"].set_index("source_id")
    direct = source[source.method == "A_direct"].set_index("source_id")
    common = sorted(set(proposed.index) & set(direct.index))
    comparison = pd.DataFrame(index=common)
    for metric in METRICS:
        comparison[f"proposed_{metric}"] = proposed.loc[common, metric]
        comparison[f"direct_{metric}"] = direct.loc[common, metric]
    comparison["objective_improvement"] = comparison["direct_objective"] - comparison["proposed_objective"]
    comparison["recall_improvement"] = comparison["proposed_stroke_recall"] - comparison["direct_stroke_recall"]
    comparison["false_stroke_change"] = comparison["proposed_false_stroke_rate"] - comparison["direct_false_stroke_rate"]
    comparison["mixed_recall_gain_false_stroke_cost"] = (comparison.recall_improvement > 0) & (comparison.false_stroke_change > 0)
    comparison = comparison.reset_index(names="source_id")
    report = {
        "sources": int(len(comparison)),
        "objective_improved_count": int((comparison.objective_improvement > 0).sum()),
        "objective_worsened_count": int((comparison.objective_improvement < 0).sum()),
        "recall_improved_count": int((comparison.recall_improvement > 0).sum()),
        "false_stroke_improved_count": int((comparison.false_stroke_change < 0).sum()),
        "mixed_recall_gain_false_stroke_cost_count": int(comparison.mixed_recall_gain_false_stroke_cost.sum()),
        "worst_proposed_objective_sources": comparison.nlargest(10, "proposed_objective")[["source_id", "proposed_objective", "proposed_ssim", "proposed_stroke_recall", "proposed_false_stroke_rate"]].to_dict("records"),
        "largest_false_stroke_cost_sources": comparison.nlargest(10, "false_stroke_change")[["source_id", "false_stroke_change", "recall_improvement", "objective_improvement"]].to_dict("records"),
        "negative_objective_sources": comparison[comparison.objective_improvement < 0][["source_id", "objective_improvement", "recall_improvement", "false_stroke_change"]].sort_values("objective_improvement").to_dict("records"),
    }
    return comparison, report


def save_plots(output: Path, stage_summary: pd.DataFrame, kernel_summary: pd.DataFrame, severity_summary: pd.DataFrame, error_frame: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "figure.dpi": 160})
    colors = {"Proposed_DSDNet_Receipt": "#0072B2", "A_direct": "#777777", "B_otsu": "#D55E00", "B_sauvola": "#009E73", "C_perceptual": "#CC79A7"}

    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)
    for axis, metric, label in zip(axes.flat, ("objective", "ssim", "stroke_recall", "false_stroke_rate"), ("Validation objective (lower)", "SSIM (higher)", "Stroke recall (higher)", "False-stroke rate (lower)")):
        axis.plot(stage_summary.stage, stage_summary[f"mean_{metric}"], marker="o", color="#0072B2")
        axis.set_xticks(list(STAGES))
        axis.set_xlabel("Retained DSDNet stages")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    figure.savefig(output / "figure_stage_depth_ablation.png", bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    axes[0].plot(kernel_summary.kernel_factor, kernel_summary.mean_objective, marker="o", color="#D55E00")
    axes[0].axvline(1.0, color="#333333", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Applied blur-kernel factor", ylabel="Validation objective (lower)")
    axes[1].plot(kernel_summary.kernel_factor, kernel_summary.mean_ssim, marker="o", color="#0072B2")
    axes[1].axvline(1.0, color="#333333", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Applied blur-kernel factor", ylabel="SSIM (higher)")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.savefig(output / "figure_kernel_sensitivity.png", bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    for method, part in severity_summary.groupby("method"):
        part = part.sort_values("severity_order")
        axes[0].plot(part.severity_order, part.mean_ssim, marker="o", label=method, color=colors.get(method))
        axes[1].plot(part.severity_order, part.mean_stroke_recall, marker="o", label=method, color=colors.get(method))
    for axis, ylabel in zip(axes, ("SSIM", "Stroke recall")):
        axis.set_xticks([0, 1, 2], ["Light", "Moderate", "Severe"])
        axis.set_xlabel("Synthetic degradation severity")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=6.5, loc="best")
    figure.savefig(output / "figure_severity_robustness.png", bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    mixed = error_frame.mixed_recall_gain_false_stroke_cost
    axis.scatter(error_frame.loc[~mixed, "recall_improvement"], error_frame.loc[~mixed, "false_stroke_change"], s=22, alpha=0.7, color="#0072B2", label="No recall/false-stroke trade-off")
    axis.scatter(error_frame.loc[mixed, "recall_improvement"], error_frame.loc[mixed, "false_stroke_change"], s=28, alpha=0.8, color="#D55E00", label="Recall gain with false-stroke cost")
    axis.axhline(0, color="#333333", linewidth=0.8)
    axis.axvline(0, color="#333333", linewidth=0.8)
    axis.set(xlabel="Proposed minus direct stroke recall", ylabel="Proposed minus direct false-stroke rate")
    axis.legend(fontsize=7)
    axis.grid(alpha=0.2)
    figure.savefig(output / "figure_error_tradeoff.png", bbox_inches="tight")
    plt.close(figure)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Claim-dependent ablation, sensitivity, robustness and failure analysis")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=REPO / "artifacts" / "protocol_lock" / "generation_manifest_611_sources.jsonl")
    parser.add_argument("--validation-records", type=Path, default=REPO / "artifacts" / "validation_restoration" / "checkpoint_43000_locked_validation_20261102" / "records.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve()
    manifest = args.manifest.resolve()
    validation_records = args.validation_records.resolve()
    training_path = REPO / "configs" / "dsdnet_receipt_train.json"
    default_path = REPO / "configs" / "default.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    protocol, _ = load_config(default_path)
    variants = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    variants = [row for row in variants if row["project_partition"] == "validation"]
    source_ids = sorted({row["source_id"] for row in variants})

    network = training["network"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DSDNetReceiptAdapter(
        protocol["dsdnet"]["source_path"],
        stages=int(network["stages"]),
        branches=int(network["maxout_branches"]),
        prior_channels=int(network["prior_channels"]),
        data_channels=int(network["data_channels"]),
    ).to(device)
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()
    crop_size = int(training["training"]["crop_height"])

    stage_rows: list[dict[str, object]] = []
    kernel_rows: list[dict[str, object]] = []
    started_all = time.perf_counter()
    for index, variant in enumerate(variants, start=1):
        clean, degraded = centre_crop_pair(read_gray(variant["clean_reference"]), read_gray(variant["degraded_path"]), crop_size)
        foreground_path = variant.get("clean_foreground") or str(Path(variant["degraded_path"]).with_name("clean_foreground.png"))
        foreground, _ = centre_crop_pair(read_gray(foreground_path), read_gray(foreground_path), crop_size)
        foreground = foreground > 127
        observed = torch.from_numpy(degraded).float().div_(255).unsqueeze(0).unsqueeze(0).to(device)
        sigma = float(variant["parameters"]["blur_sigma"])

        with torch.inference_mode():
            nominal_output = model(observed, gaussian_kernel(sigma).to(device), retain_diagnostics=True)
        for stage in STAGES:
            values = metric_row(clean, nominal_output["stage_states"][stage - 1].clamp(0, 1), foreground)
            stage_rows.append({
                "source_id": variant["source_id"], "variant_id": variant["variant_id"], "severity": variant["severity"],
                "stage": stage, "nominal_blur_sigma": sigma, **values,
            })
        kernel_rows.append({
            "source_id": variant["source_id"], "variant_id": variant["variant_id"], "severity": variant["severity"],
            "kernel_factor": 1.0, "nominal_blur_sigma": sigma, "applied_blur_sigma": sigma,
            **metric_row(clean, nominal_output["restored"].clamp(0, 1), foreground),
        })
        del nominal_output
        for factor in KERNEL_FACTORS:
            if factor == 1.0:
                continue
            applied_sigma = sigma * factor
            with torch.inference_mode():
                restored = model(observed, gaussian_kernel(applied_sigma).to(device), retain_diagnostics=False)["restored"].clamp(0, 1)
            kernel_rows.append({
                "source_id": variant["source_id"], "variant_id": variant["variant_id"], "severity": variant["severity"],
                "kernel_factor": factor, "nominal_blur_sigma": sigma, "applied_blur_sigma": applied_sigma,
                **metric_row(clean, restored, foreground),
            })
        print(f"Claim-dependent evaluation {index}/{len(variants)}", flush=True)

    stage_frame = pd.DataFrame(stage_rows)
    kernel_frame = pd.DataFrame(kernel_rows)
    stage_frame.to_csv(output / "stage_ablation_records.csv", index=False)
    kernel_frame.to_csv(output / "kernel_sensitivity_records.csv", index=False)
    stage_summary_rows = summary_by(stage_frame, "stage")
    kernel_summary_rows = summary_by(kernel_frame, "kernel_factor")
    write_csv(output / "table_stage_ablation.csv", stage_summary_rows)
    write_csv(output / "table_kernel_sensitivity.csv", kernel_summary_rows)

    stage_tests = paired_tests(stage_frame, "stage", 8, [1, 2, 4, 6], "stage_depth")
    kernel_tests = paired_tests(kernel_frame, "kernel_factor", 1.0, [0.5, 0.75, 1.25, 1.5], "kernel_factor")
    write_csv(output / "stage_ablation_paired_tests.csv", stage_tests)
    write_csv(output / "kernel_sensitivity_paired_tests.csv", kernel_tests)

    severity_summary, severity_tests, severity_summary_rows = severity_analysis(validation_records)
    severity_summary.to_csv(output / "table_severity_robustness.csv", index=False)
    write_csv(output / "severity_statistical_tests.csv", severity_tests)
    error_frame, error_report = error_analysis(validation_records)
    error_frame.to_csv(output / "source_error_analysis.csv", index=False)
    (output / "error_failure_report.json").write_text(json.dumps(error_report, indent=2), encoding="utf-8")

    stage_summary = pd.DataFrame(stage_summary_rows).sort_values("stage")
    kernel_summary = pd.DataFrame(kernel_summary_rows).sort_values("kernel_factor")
    save_plots(output, stage_summary, kernel_summary, severity_summary, error_frame)

    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        git_head = "unavailable"
    summary = {
        "analysis_scope": "claim-dependent development validation; no fresh independent test",
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": int(payload.get("iteration", -1)),
        "validation_sources": len(source_ids),
        "validation_variants": len(variants),
        "crop_size": crop_size,
        "stage_depths": list(STAGES),
        "kernel_factors": list(KERNEL_FACTORS),
        "stage_ablation": stage_summary_rows,
        "kernel_sensitivity": kernel_summary_rows,
        "stage_tests": stage_tests,
        "kernel_tests": kernel_tests,
        "severity_robustness": severity_summary_rows,
        "severity_tests": severity_tests,
        "error_failure_analysis": error_report,
        "interpretation_boundaries": [
            "Intermediate-stage evaluation isolates inference depth but does not retrain a smaller architecture.",
            "Kernel sensitivity perturbs the supplied blur estimate and does not model every natural degradation.",
            "Synthetic severity results are development validation, not an independent natural-receipt test.",
            "Negative and mixed findings are retained, including false-stroke trade-offs and source-level failures.",
        ],
        "runtime_seconds": time.perf_counter() - started_all,
        "provenance": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "git_head": git_head,
            "checkpoint_sha256": sha256(checkpoint),
            "manifest_sha256": sha256(manifest),
            "validation_records_sha256": sha256(validation_records),
            "training_config_sha256": sha256(training_path),
            "runtime_config_sha256": sha256(default_path),
            "script_sha256": sha256(Path(__file__).resolve()),
        },
    }
    (output / "claim_dependent_analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "runtime_seconds": summary["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
