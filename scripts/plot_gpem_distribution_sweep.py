#!/usr/bin/env python3
"""
Plot GPEM Distribution Sweep Results

Visualizes AMOTA comparison and improvement (static vs GPEM linear vs GPEM quadratic) 
across the 8-step detector+localizer sweep. Two panels: AMOTA comparison and AMOTA improvement %.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import json
import re
from pathlib import Path

STEP_LABELS = [
    "100% DETR3D\n100% ORB",
    "95% DETR3D\n90% ORB",
    "90% DETR3D\n80% ORB",
    "80% DETR3D\n70% ORB",
    "70% DETR3D\n60% ORB",
    "60% DETR3D\n50/50 loc",
    "50% DETR3D\n40% ORB",
    "33/33/33 det\n50/50 loc",
]


def plot_distribution_sweep_results(results_dir):
    results_path = Path(results_dir)
    summary_file = results_path / "summary.json"

    if not summary_file.exists():
        print(f"Error: summary.json not found in {results_dir}")
        print("Pass the path to a GPEM_Distribution_Sweep results folder (e.g. results/GPEM_Distribution_Sweep_<timestamp>).")
        return

    with open(summary_file) as f:
        summary = json.load(f)

    # Parse step_N_<variant> where variant is static, gpem, gpem_linear, or gpem_quadratic
    by_step = {}
    for config_name, data in summary.get("results", {}).items():
        # Try new 3-variant format first: static, gpem_linear, gpem_quadratic
        m = re.match(r"gpem_dist_sweep_step_(\d+)_(static|gpem_linear|gpem_quadratic|gpem)", config_name)
        if m:
            step_idx = int(m.group(1))
            variant = m.group(2)
            if step_idx not in by_step:
                by_step[step_idx] = {}
            by_step[step_idx][variant] = data

    steps = sorted(by_step.keys())
    if not steps:
        print("Error: no gpem_dist_sweep_step_* configs found in summary.json")
        return

    # Determine which variants are present
    first_step_data = by_step[steps[0]]
    has_linear = "gpem_linear" in first_step_data
    has_quadratic = "gpem_quadratic" in first_step_data
    has_legacy_gpem = "gpem" in first_step_data and not has_linear
    
    # Build model list based on what's available
    # Format: (variant_key, label, color, marker)
    models = [("static", "Static Average", "#FF6B6B", "o")]
    
    if has_linear:
        models.append(("gpem_linear", "GPEM Linear", "#4ECDC4", "s"))
    if has_quadratic:
        models.append(("gpem_quadratic", "GPEM Quadratic", "#9B59B6", "^"))
    if has_legacy_gpem:
        models.append(("gpem", "GPEM Model", "#4ECDC4", "s"))
    
    print(f"Found {len(models)} model variants: {[m[1] for m in models]}")

    # Extract data for all models
    model_data = {}
    for variant_key, label, color, marker in models:
        try:
            amota_means = np.array([by_step[s][variant_key]["avg_global_amota_mean"] for s in steps])
            amota_stds = np.array([by_step[s][variant_key]["avg_global_amota_std"] for s in steps])
            model_data[variant_key] = {
                "label": label,
                "color": color,
                "marker": marker,
                "amota_means": amota_means,
                "amota_stds": amota_stds,
            }
        except KeyError as e:
            print(f"Warning: Missing data for {variant_key} at some steps: {e}")

    x = np.arange(len(steps))
    labels = [STEP_LABELS[i] if i < len(STEP_LABELS) else f"Step {i}" for i in steps]

    title_suffix = "A/B/C" if len(model_data) == 3 else "A/B"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"GPEM Distribution Sweep: {title_suffix} Comparison (AMOTA)",
        fontsize=14,
        fontweight="bold",
    )

    # Plot 1: AMOTA comparison
    ax1 = axes[0]
    for variant_key, data in model_data.items():
        ax1.plot(x, data["amota_means"], f'{data["marker"]}-', label=data["label"], 
                linewidth=2.5, markersize=8, color=data["color"])
        ax1.fill_between(x, 
                         data["amota_means"] - data["amota_stds"], 
                         data["amota_means"] + data["amota_stds"], 
                         alpha=0.2, color=data["color"])
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    ax1.set_ylabel("Global AMOTA", fontweight="bold", fontsize=11)
    ax1.set_title("AMOTA Comparison with Standard Deviation", fontweight="bold", fontsize=12)
    ax1.legend(fontsize=10, loc="best")
    ax1.grid(True, alpha=0.3)

    # Plot 2: AMOTA improvement % (relative to static)
    ax2 = axes[1]
    static_data = model_data.get("static")
    if static_data is None:
        print("Error: Static model data not found")
        return
    
    improvement_colors = ['#27AE60', '#3498DB', '#E74C3C']
    color_idx = 0
    
    for variant_key, data in model_data.items():
        if variant_key == "static":
            continue
        
        improvements = (data["amota_means"] - static_data["amota_means"]) / np.maximum(static_data["amota_means"], 0.001) * 100
        improvement_stds = np.sqrt(
            (data["amota_stds"] / np.maximum(static_data["amota_means"], 0.001)) ** 2
            + (data["amota_means"] * static_data["amota_stds"] / np.maximum(static_data["amota_means"] ** 2, 0.001)) ** 2
        ) * 100
        
        ax2.plot(x, improvements, f'{data["marker"]}-', linewidth=2.5, markersize=8, 
                color=improvement_colors[color_idx], label=f'{data["label"]} vs Static')
        ax2.fill_between(x, improvements - improvement_stds, improvements + improvement_stds, 
                        alpha=0.2, color=improvement_colors[color_idx])
        color_idx += 1
    
    ax2.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    ax2.set_ylabel("Improvement (%)", fontweight="bold", fontsize=11)
    ax2.set_title("GPEM AMOTA Improvement over Static", fontweight="bold", fontsize=12)
    ax2.legend(fontsize=10, loc="best")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = results_path / "gpem_distribution_sweep_plot.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✓ Plot saved to: {output_file}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("GPEM DISTRIBUTION SWEEP SUMMARY STATISTICS")
    print("=" * 70)
    
    for variant_key, data in model_data.items():
        print(f"\n{data['label']}:")
        print(f"  Mean AMOTA: {np.mean(data['amota_means']):.4f} ± {np.mean(data['amota_stds']):.4f}")
        
        if variant_key != "static":
            improvements = (data["amota_means"] - static_data["amota_means"]) / np.maximum(static_data["amota_means"], 0.001) * 100
            print(f"  Average improvement vs Static: {np.mean(improvements):.2f}%")
            print(f"  Min improvement: {np.min(improvements):.2f}% (Step {steps[np.argmin(improvements)]})")
            print(f"  Max improvement: {np.max(improvements):.2f}% (Step {steps[np.argmax(improvements)]})")
            print(f"  Wins over Static: {sum(1 for imp in improvements if imp > 0)}/{len(steps)} steps")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        print("Usage: python plot_gpem_distribution_sweep.py <results_directory>")
        print("\nPass the path to an existing GPEM_Distribution_Sweep results folder (containing summary.json).")
        print("This script only plots; it does not run experiments (no --warmup, --runs, etc.).")
        print("\nExample:")
        print("  python plot_gpem_distribution_sweep.py results/GPEM_Distribution_Sweep_2026-02-27_151504")
        sys.exit(1)
    plot_distribution_sweep_results(sys.argv[1])
