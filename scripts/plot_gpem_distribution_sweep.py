#!/usr/bin/env python3
"""
Plot GPEM Distribution Sweep Results

Generates per-filter, per-metric plots (4 filters x 5 metrics = 20 plots)
matching the penetration sweep output style. Each plot has two panels:
left = absolute metric comparison, right = improvement vs filter-matched baseline.
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

# Colors by covariance mode (same as penetration sweep)
COV_COLORS = {
    "baseline":  "#95A5A6",
    "static":    "#E74C3C",
    "linear":    "#2ECC71",
    "quadratic": "#9B59B6",
}

# Variant key -> (covariance_type, filter_group, short_label)
VARIANT_INFO = {
    "baseline":         ("baseline",  "Kalman", "Baseline"),
    "static":           ("static",    "Kalman", "Static"),
    "gpem_linear":      ("linear",    "Kalman", "GPEM Linear"),
    "gpem_quadratic":   ("quadratic", "Kalman", "GPEM Quadratic"),
    "ci_baseline":      ("baseline",  "CI",     "CI + Baseline"),
    "ci_static":        ("static",    "CI",     "CI + Static"),
    "ci_gpem_linear":   ("linear",    "CI",     "CI + GPEM Linear"),
    "ci_gpem_quadratic":("quadratic", "CI",     "CI + GPEM Quadratic"),
    "akf_baseline":     ("baseline",  "AKF",    "AKF + Baseline"),
    "akf_static":       ("static",    "AKF",    "AKF + Static"),
    "akf_gpem_linear":  ("linear",    "AKF",    "AKF + GPEM Linear"),
    "akf_gpem_quadratic":("quadratic","AKF",    "AKF + GPEM Quadratic"),
    "pf_baseline":      ("baseline",  "PF",     "PF + Baseline"),
    "pf_static":        ("static",    "PF",     "PF + Static"),
    "pf_gpem_linear":   ("linear",    "PF",     "PF + GPEM Linear"),
    "pf_gpem_quadratic":("quadratic", "PF",     "PF + GPEM Quadratic"),
}

# Baseline key for each filter group
FILTER_BASELINES = {
    "Kalman": "baseline",
    "CI":     "ci_baseline",
    "AKF":    "akf_baseline",
    "PF":     "pf_baseline",
}

FILTER_ORDER = ["Kalman", "CI", "AKF", "PF"]

METRICS = [
    ("amota", "AMOTA", True),
    ("amotp", "AMOTP (m)", False),
    ("hota", "HOTA", True),
    ("deta", "DetA", True),
    ("assa", "AssA", True),
]

METRIC_KEYS_IN_SUMMARY = {
    "amota": ("avg_global_amota_mean", "avg_global_amota_std"),
    "amotp": ("avg_global_amotp_mean", "avg_global_amotp_std"),
    "hota":  ("avg_hota_mean", "avg_hota_std"),
    "deta":  ("avg_deta_mean", "avg_deta_std"),
    "assa":  ("avg_assa_mean", "avg_assa_std"),
}


def plot_distribution_sweep_results(results_dir):
    results_path = Path(results_dir)
    summary_file = results_path / "summary.json"

    if not summary_file.exists():
        print(f"Error: summary.json not found in {results_dir}")
        return

    with open(summary_file) as f:
        summary = json.load(f)

    # Parse gpem_dist_sweep_step_N_<variant>
    variant_pattern = "|".join(re.escape(k) for k in VARIANT_INFO)
    regex = re.compile(rf"gpem_dist_sweep_step_(\d+)_({variant_pattern})")

    by_step = {}
    for config_name, data in summary.get("results", {}).items():
        m = regex.match(config_name)
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
    all_variants = set()
    for step_data in by_step.values():
        all_variants.update(step_data.keys())

    # Group available variants by filter
    filter_groups = {}
    for variant_key in sorted(all_variants, key=lambda k: list(VARIANT_INFO.keys()).index(k) if k in VARIANT_INFO else 999):
        if variant_key not in VARIANT_INFO:
            continue
        cov_type, filter_name, label = VARIANT_INFO[variant_key]
        if filter_name not in filter_groups:
            filter_groups[filter_name] = []
        filter_groups[filter_name].append(variant_key)

    present_filters = [f for f in FILTER_ORDER if f in filter_groups]
    print(f"Found {len(all_variants)} variants across {len(present_filters)} filters: {present_filters}")

    # Extract metric arrays for each variant
    # variant_data[variant_key][metric_key] = {"means": array, "stds": array}
    variant_data = {}
    for variant_key in all_variants:
        if variant_key not in VARIANT_INFO:
            continue
        variant_data[variant_key] = {}
        for metric_key, _, _ in METRICS:
            mean_key, std_key = METRIC_KEYS_IN_SUMMARY[metric_key]
            means = []
            stds = []
            for s in steps:
                d = by_step[s].get(variant_key)
                if d:
                    means.append(d.get(mean_key, 0.0))
                    stds.append(d.get(std_key, 0.0))
                else:
                    means.append(0.0)
                    stds.append(0.0)
            variant_data[variant_key][metric_key] = {
                "means": np.array(means),
                "stds": np.array(stds),
            }

    x = np.arange(len(steps))
    tick_labels = [STEP_LABELS[i] if i < len(STEP_LABELS) else f"Step {i}" for i in steps]

    # Check if HOTA data exists
    has_hota = False
    for vk in variant_data:
        if np.any(variant_data[vk]["hota"]["means"] > 0):
            has_hota = True
            break

    active_metrics = METRICS[:2]  # Always AMOTA, AMOTP
    if has_hota:
        active_metrics = METRICS  # All 5

    # Generate one figure per (filter, metric)
    for filter_name in present_filters:
        group_variants = filter_groups[filter_name]
        baseline_key = FILTER_BASELINES[filter_name]

        for metric_key, ylabel, higher_is_better in active_metrics:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f'{filter_name} Filter: {ylabel} vs Detector/Localizer Distribution',
                         fontsize=14, fontweight='bold')

            # Left: absolute comparison
            ax_left = axes[0]
            for vk in group_variants:
                cov_type, _, label = VARIANT_INFO[vk]
                color = COV_COLORS[cov_type]
                means = variant_data[vk][metric_key]["means"]
                stds = variant_data[vk][metric_key]["stds"]
                # Short label (strip filter prefix)
                short = label.replace("CI + ", "").replace("AKF + ", "").replace("PF + ", "")
                ax_left.plot(x, means, 'o-', label=short,
                             linewidth=2.5, markersize=8, color=color)
                ax_left.fill_between(x, means - stds, means + stds,
                                     alpha=0.15, color=color)

            ax_left.set_xticks(x)
            ax_left.set_xticklabels(tick_labels, fontsize=7, rotation=15, ha="right")
            ax_left.set_ylabel(ylabel, fontweight='bold', fontsize=11)
            ax_left.set_title(f'{ylabel} Comparison', fontweight='bold', fontsize=12)
            ax_left.legend(fontsize=10, loc='best')
            ax_left.grid(True, alpha=0.3)
            if metric_key in ("hota", "deta", "assa"):
                ax_left.set_ylim(0, 1.05)

            # Right: improvement vs filter-matched baseline
            ax_right = axes[1]
            if baseline_key in variant_data:
                ref_means = variant_data[baseline_key][metric_key]["means"]
                ref_stds = variant_data[baseline_key][metric_key]["stds"]

                for vk in group_variants:
                    if vk == baseline_key:
                        continue
                    cov_type, _, label = VARIANT_INFO[vk]
                    color = COV_COLORS[cov_type]
                    means = variant_data[vk][metric_key]["means"]
                    stds = variant_data[vk][metric_key]["stds"]

                    if higher_is_better:
                        improvements = (means - ref_means) / np.maximum(np.abs(ref_means), 1e-9) * 100
                    else:
                        improvements = (ref_means - means) / np.maximum(np.abs(ref_means), 1e-9) * 100
                    imp_stds = np.sqrt(
                        (stds / np.maximum(np.abs(ref_means), 1e-9))**2 +
                        (means * ref_stds / np.maximum(ref_means**2, 1e-9))**2
                    ) * 100

                    short = label.replace("CI + ", "").replace("AKF + ", "").replace("PF + ", "")
                    ax_right.plot(x, improvements, 'o-', linewidth=2.5, markersize=8,
                                  color=color, label=short)
                    ax_right.fill_between(x, improvements - imp_stds, improvements + imp_stds,
                                          alpha=0.15, color=color)

            ax_right.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax_right.set_xticks(x)
            ax_right.set_xticklabels(tick_labels, fontsize=7, rotation=15, ha="right")
            ax_right.set_ylabel('Improvement (%)', fontweight='bold', fontsize=11)
            imp_note = "(positive = better)" if higher_is_better else "(positive = closer to GT)"
            ax_right.set_title(f'{ylabel} Improvement vs Baseline {imp_note}',
                               fontweight='bold', fontsize=12)
            ax_right.legend(fontsize=10, loc='best')
            ax_right.grid(True, alpha=0.3)

            plt.tight_layout()
            fname = f"gpem_dist_{filter_name.lower()}_{metric_key}_plot.png"
            output_file = results_path / fname
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"  {filter_name} {ylabel} -> {fname}")
            plt.close(fig)

    # Combined cross-filter plots: one line per filter, fixed covariance mode
    FILTER_COLORS = {
        "Kalman": "#3498DB",
        "CI":     "#E74C3C",
        "AKF":    "#2ECC71",
        "PF":     "#F39C12",
    }
    FILTER_MARKERS = {
        "Kalman": "o",
        "CI":     "s",
        "AKF":    "^",
        "PF":     "D",
    }
    # Which covariance variant key to use per filter for "linear"
    COMBINED_VARIANTS = {
        "gpem_linear": {
            "Kalman": "gpem_linear",
            "CI":     "ci_gpem_linear",
            "AKF":    "akf_gpem_linear",
            "PF":     "pf_gpem_linear",
        },
    }
    for cov_label, filter_map in COMBINED_VARIANTS.items():
        for metric_key, ylabel, higher_is_better in active_metrics:
            available = [(f, vk) for f, vk in filter_map.items()
                         if vk in variant_data and f in present_filters]
            if len(available) < 2:
                continue

            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            fig.suptitle(f'{ylabel} Across Filters ({cov_label.replace("_", " ").title()})',
                         fontsize=14, fontweight='bold')

            for filter_name, vk in available:
                means = variant_data[vk][metric_key]["means"]
                stds = variant_data[vk][metric_key]["stds"]
                color = FILTER_COLORS[filter_name]
                marker = FILTER_MARKERS[filter_name]
                ax.plot(x, means, marker=marker, linestyle='-', label=filter_name,
                        linewidth=2.5, markersize=8, color=color)
                ax.fill_between(x, means - stds, means + stds, alpha=0.15, color=color)

            ax.set_xticks(x)
            ax.set_xticklabels(tick_labels, fontsize=7, rotation=15, ha="right")
            ax.set_ylabel(ylabel, fontweight='bold', fontsize=11)
            ax.set_xlabel('Detector / Localizer Distribution', fontweight='bold', fontsize=11)
            ax.legend(fontsize=10, loc='best')
            ax.grid(True, alpha=0.3)
            if metric_key in ("hota", "deta", "assa"):
                ax.set_ylim(0, 1.05)

            plt.tight_layout()
            fname = f"gpem_dist_combined_{cov_label}_{metric_key}_plot.png"
            output_file = results_path / fname
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"  Combined {cov_label} {ylabel} -> {fname}")
            plt.close(fig)

    # Print summary statistics per filter group (matching penetration sweep style)
    print("\n" + "=" * 70)
    print("GPEM DISTRIBUTION SWEEP SUMMARY STATISTICS")
    print("=" * 70)

    for filter_name in present_filters:
        group_variants = filter_groups[filter_name]
        baseline_key = FILTER_BASELINES[filter_name]

        for vk in group_variants:
            cov_type, _, label = VARIANT_INFO[vk]
            d = variant_data[vk]

            print(f"\n{label}:")
            print(f"  Mean AMOTA: {np.mean(d['amota']['means']):.4f} +/- {np.mean(d['amota']['stds']):.4f}")
            print(f"  Mean AMOTP: {np.mean(d['amotp']['means']):.4f}m +/- {np.mean(d['amotp']['stds']):.4f}")
            if has_hota:
                print(f"  Mean HOTA:  {np.mean(d['hota']['means']):.4f} +/- {np.mean(d['hota']['stds']):.4f}")
                print(f"  Mean DetA:  {np.mean(d['deta']['means']):.4f} | Mean AssA: {np.mean(d['assa']['means']):.4f}")

            if vk == baseline_key or baseline_key not in variant_data:
                continue

            ref_amota = variant_data[baseline_key]["amota"]["means"]
            ref_amotp = variant_data[baseline_key]["amotp"]["means"]
            amota_means = d["amota"]["means"]
            amotp_means = d["amotp"]["means"]

            amota_imp = (amota_means - ref_amota) / np.maximum(np.abs(ref_amota), 1e-9) * 100
            ref_label = VARIANT_INFO[baseline_key][2]
            print(f"  AMOTA vs {ref_label}: {np.mean(amota_imp):+.2f}%")
            print(f"    Min: {np.min(amota_imp):+.2f}% (Step {steps[np.argmin(amota_imp)]})")
            print(f"    Max: {np.max(amota_imp):+.2f}% (Step {steps[np.argmax(amota_imp)]})")
            print(f"    Wins: {sum(1 for imp in amota_imp if imp > 0)}/{len(steps)} steps")

            if np.any(ref_amotp > 0):
                amotp_imp = (ref_amotp - amotp_means) / np.maximum(np.abs(ref_amotp), 1e-9) * 100
                print(f"  AMOTP vs {ref_label}: {np.mean(amotp_imp):+.2f}% (positive = closer to GT)")
                print(f"    Min: {np.min(amotp_imp):+.2f}% (Step {steps[np.argmin(amotp_imp)]})")
                print(f"    Max: {np.max(amotp_imp):+.2f}% (Step {steps[np.argmax(amotp_imp)]})")
                print(f"    Wins: {sum(1 for imp in amotp_imp if imp > 0)}/{len(steps)} steps")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        print("Usage: python plot_gpem_distribution_sweep.py <results_directory>")
        print("\nPass the path to an existing GPEM_Distribution_Sweep results folder (containing summary.json).")
        print("This script only plots; it does not run experiments.")
        print("\nExample:")
        print("  python plot_gpem_distribution_sweep.py results/GPEM_Distribution_Sweep_2026-02-27_151504")
        sys.exit(1)
    plot_distribution_sweep_results(sys.argv[1])
