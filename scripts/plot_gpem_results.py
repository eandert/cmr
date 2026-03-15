#!/usr/bin/env python3
"""
Plot GPEM Test Results - Supports Static, Linear, and Quadratic GPEM

Visualizes AMOTA comparison and improvement with confidence bands for all three models.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
import sys
import json
from pathlib import Path

def plot_gpem_results_simple(results_dir):
    """Plot GPEM test results - supports 3 model variants."""
    
    results_path = Path(results_dir)
    summary_file = results_path / "summary.json"
    
    if not summary_file.exists():
        print(f"Error: summary.json not found in {results_dir}")
        return
    
    with open(summary_file) as f:
        summary = json.load(f)
    
    # Extract data for all three model types
    potential_av_rates = np.array([1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
    
    # Color by covariance mode, linestyle/marker by filter type
    # Colors: baseline=grey, static=red, linear=teal, quadratic=purple
    cov_colors = {
        "baseline": "#95A5A6",
        "static":   "#E74C3C",
        "linear":   "#2ECC71",
        "quadratic":"#9B59B6",
    }
    # Linestyles and markers by filter type
    filter_styles = {
        "kalman": ("-",  "o"),   # solid
        "ci":     ("--", "s"),   # dashed
        "akf":    ("-.", "^"),   # dash-dot
        "pf":     (":",  "D"),   # dotted, diamond
    }

    # Model configurations: (key_pattern, label, color, marker, linestyle)
    # Order: baseline, static, linear, quadratic for each filter
    models = [
        # Kalman filter group
        ("baseline_av_{rate:.1f}pct",         "Baseline",            cov_colors["baseline"], filter_styles["kalman"][1], filter_styles["kalman"][0]),
        ("static_cov_av_{rate:.1f}pct",       "Static",              cov_colors["static"],   filter_styles["kalman"][1], filter_styles["kalman"][0]),
        ("gpem_linear_av_{rate:.1f}pct",      "GPEM Linear",         cov_colors["linear"],   filter_styles["kalman"][1], filter_styles["kalman"][0]),
        ("gpem_quadratic_av_{rate:.1f}pct",   "GPEM Quadratic",      cov_colors["quadratic"],filter_styles["kalman"][1], filter_styles["kalman"][0]),
        # CI filter group
        ("ci_baseline_av_{rate:.1f}pct",      "CI + Baseline",       cov_colors["baseline"], filter_styles["ci"][1],     filter_styles["ci"][0]),
        ("ci_static_av_{rate:.1f}pct",        "CI + Static",         cov_colors["static"],   filter_styles["ci"][1],     filter_styles["ci"][0]),
        ("ci_gpem_linear_av_{rate:.1f}pct",   "CI + GPEM Linear",    cov_colors["linear"],   filter_styles["ci"][1],     filter_styles["ci"][0]),
        ("ci_gpem_quadratic_av_{rate:.1f}pct","CI + GPEM Quadratic", cov_colors["quadratic"],filter_styles["ci"][1],     filter_styles["ci"][0]),
        # AKF filter group
        ("akf_baseline_av_{rate:.1f}pct",     "AKF + Baseline",      cov_colors["baseline"], filter_styles["akf"][1],    filter_styles["akf"][0]),
        ("akf_static_av_{rate:.1f}pct",       "AKF + Static",        cov_colors["static"],   filter_styles["akf"][1],    filter_styles["akf"][0]),
        ("akf_gpem_linear_av_{rate:.1f}pct",  "AKF + GPEM Linear",   cov_colors["linear"],   filter_styles["akf"][1],    filter_styles["akf"][0]),
        ("akf_gpem_quadratic_av_{rate:.1f}pct","AKF + GPEM Quadratic",cov_colors["quadratic"],filter_styles["akf"][1],   filter_styles["akf"][0]),
        # PF filter group
        ("pf_baseline_av_{rate:.1f}pct",      "PF + Baseline",       cov_colors["baseline"], filter_styles["pf"][1],     filter_styles["pf"][0]),
        ("pf_static_av_{rate:.1f}pct",        "PF + Static",         cov_colors["static"],   filter_styles["pf"][1],     filter_styles["pf"][0]),
        ("pf_gpem_linear_av_{rate:.1f}pct",   "PF + GPEM Linear",    cov_colors["linear"],   filter_styles["pf"][1],     filter_styles["pf"][0]),
        ("pf_gpem_quadratic_av_{rate:.1f}pct","PF + GPEM Quadratic", cov_colors["quadratic"],filter_styles["pf"][1],     filter_styles["pf"][0]),
    ]

    # Fallback for old naming convention (gpem_model_av_X instead of gpem_linear_av_X)
    legacy_models = [
        ("static_cov_av_{rate:.1f}pct", "Static", "#FF6B6B", "o", "-"),
        ("gpem_model_av_{rate:.1f}pct", "GPEM Model", "#4ECDC4", "s", "-"),
    ]

    # Check which model set we have
    test_rate = potential_av_rates[0]
    has_new_format = f"gpem_linear_av_{test_rate:.1f}pct" in summary['results']
    has_quadratic = f"gpem_quadratic_av_{test_rate:.1f}pct" in summary['results']
    has_baseline = f"baseline_av_{test_rate:.1f}pct" in summary['results']
    has_ci = f"ci_gpem_linear_av_{test_rate:.1f}pct" in summary['results']
    has_akf = f"akf_gpem_linear_av_{test_rate:.1f}pct" in summary['results']
    has_pf = f"pf_gpem_linear_av_{test_rate:.1f}pct" in summary['results']

    if not has_new_format:
        models = legacy_models
        print("Note: Using legacy 2-model format (static vs gpem)")
    else:
        active_models = []
        # Kalman: baseline, static, linear, quadratic
        if has_baseline:
            active_models.append(models[0])
        active_models.append(models[1])  # Static
        active_models.append(models[2])  # Linear
        if has_quadratic:
            active_models.append(models[3])
        # CI: baseline, static, linear, quadratic
        if has_ci:
            active_models.append(models[4])   # CI + Baseline
            active_models.append(models[5])   # CI + Static
            active_models.append(models[6])   # CI + GPEM Linear
            active_models.append(models[7])   # CI + GPEM Quadratic
        # AKF: baseline, static, linear, quadratic
        if has_akf:
            active_models.append(models[8])   # AKF + Baseline
            active_models.append(models[9])   # AKF + Static
            active_models.append(models[10])  # AKF + GPEM Linear
            active_models.append(models[11])  # AKF + GPEM Quadratic
        # PF: baseline, static, linear, quadratic
        if has_pf:
            active_models.append(models[12])  # PF + Baseline
            active_models.append(models[13])  # PF + Static
            active_models.append(models[14])  # PF + GPEM Linear
            active_models.append(models[15])  # PF + GPEM Quadratic
        models = active_models
    
    # Collect data for each model
    model_data = {m[1]: {"rates": [], "amota_means": [], "amota_stds": [], "amotp_means": [], "amotp_stds": [],
                          "hota_means": [], "hota_stds": [], "deta_means": [], "deta_stds": [],
                          "assa_means": [], "assa_stds": []} for m in models}
    
    for rate in potential_av_rates:
        all_found = True
        for key_pattern, label, *_ in models:
            key = key_pattern.format(rate=rate)
            if key not in summary['results']:
                all_found = False
                break
        
        if all_found:
            for key_pattern, label, *_ in models:
                key = key_pattern.format(rate=rate)
                res = summary['results'][key]
                model_data[label]["rates"].append(rate)
                model_data[label]["amota_means"].append(res['avg_global_amota_mean'])
                model_data[label]["amota_stds"].append(res['avg_global_amota_std'])
                model_data[label]["amotp_means"].append(res.get('avg_global_amotp_mean', 0))
                model_data[label]["amotp_stds"].append(res.get('avg_global_amotp_std', 0))
                model_data[label]["hota_means"].append(res.get('avg_hota_mean', 0))
                model_data[label]["hota_stds"].append(res.get('avg_hota_std', 0))
                model_data[label]["deta_means"].append(res.get('avg_deta_mean', 0))
                model_data[label]["deta_stds"].append(res.get('avg_deta_std', 0))
                model_data[label]["assa_means"].append(res.get('avg_assa_mean', 0))
                model_data[label]["assa_stds"].append(res.get('avg_assa_std', 0))
    
    # Convert to numpy arrays
    for label in model_data:
        model_data[label]["rates"] = np.array(model_data[label]["rates"])
        model_data[label]["amota_means"] = np.array(model_data[label]["amota_means"])
        model_data[label]["amota_stds"] = np.array(model_data[label]["amota_stds"])
        model_data[label]["amotp_means"] = np.array(model_data[label]["amotp_means"])
        model_data[label]["amotp_stds"] = np.array(model_data[label]["amotp_stds"])
        model_data[label]["hota_means"] = np.array(model_data[label]["hota_means"])
        model_data[label]["hota_stds"] = np.array(model_data[label]["hota_stds"])
        model_data[label]["deta_means"] = np.array(model_data[label]["deta_means"])
        model_data[label]["deta_stds"] = np.array(model_data[label]["deta_stds"])
        model_data[label]["assa_means"] = np.array(model_data[label]["assa_means"])
        model_data[label]["assa_stds"] = np.array(model_data[label]["assa_stds"])
    
    # Get av_rates from the first model that has data
    av_rates = None
    for m in models:
        if len(model_data[m[1]]["rates"]) > 0:
            av_rates = model_data[m[1]]["rates"]
            break
    if av_rates is None:
        print("Error: No valid data found in summary.json")
        return

    # Map each model label to its filter-matched baseline label
    def _get_baseline_label(label):
        """Return the baseline label for the same filter group."""
        if label.startswith("CI + "):
            return "CI + Baseline"
        elif label.startswith("AKF + "):
            return "AKF + Baseline"
        elif label.startswith("PF + "):
            return "PF + Baseline"
        else:
            return "Baseline"

    # Group models by filter type
    filter_groups = {
        "Kalman": {"prefix": "", "models": []},
        "CI":     {"prefix": "CI + ", "models": []},
        "AKF":    {"prefix": "AKF + ", "models": []},
        "PF":     {"prefix": "PF + ", "models": []},
    }
    for m in models:
        label = m[1]
        if label.startswith("CI + "):
            filter_groups["CI"]["models"].append(m)
        elif label.startswith("AKF + "):
            filter_groups["AKF"]["models"].append(m)
        elif label.startswith("PF + "):
            filter_groups["PF"]["models"].append(m)
        else:
            filter_groups["Kalman"]["models"].append(m)

    # Remove empty filter groups
    filter_groups = {k: v for k, v in filter_groups.items() if len(v["models"]) > 0}

    # Metrics to plot: (metric_key, ylabel, higher_is_better)
    all_metrics = [
        ("amota", "AMOTA", True),
        ("amotp", "AMOTP (m)", False),
    ]
    has_hota = any(np.any(model_data[m[1]]["hota_means"] > 0) for m in models if len(model_data[m[1]]["hota_means"]) > 0)
    if has_hota:
        all_metrics += [
            ("hota", "HOTA", True),
            ("deta", "DetA", True),
            ("assa", "AssA", True),
        ]

    # Generate one figure per metric per filter type
    for filter_name, group in filter_groups.items():
        group_models = group["models"]
        for metric_key, ylabel, higher_is_better in all_metrics:
            fig_h, axes_h = plt.subplots(1, 2, figsize=(14, 5))
            fig_h.suptitle(f'{filter_name} Filter: {ylabel} vs AV Penetration Rate',
                           fontsize=14, fontweight='bold')

            # Left: comparison with confidence bands
            ax_left = axes_h[0]
            for key_pattern, label, color, marker, ls in group_models:
                data = model_data[label]
                means = data[f"{metric_key}_means"]
                stds = data[f"{metric_key}_stds"]
                if len(means) == 0:
                    continue
                # Shorten label for per-filter plot (remove filter prefix)
                short_label = label.replace("CI + ", "").replace("AKF + ", "").replace("PF + ", "")
                ax_left.plot(data["rates"], means, marker=marker, linestyle='-', label=short_label,
                             linewidth=2.5, markersize=8, color=color)
                ax_left.fill_between(data["rates"], means - stds, means + stds,
                                     alpha=0.15, color=color)
            ax_left.set_xlabel('AV Injection Rate (%)', fontweight='bold', fontsize=11)
            ax_left.set_ylabel(ylabel, fontweight='bold', fontsize=11)
            ax_left.set_title(f'{ylabel} Comparison with Standard Deviation', fontweight='bold', fontsize=12)
            ax_left.legend(fontsize=10, loc='best')
            ax_left.grid(True, alpha=0.3)
            ax_left.set_xlim(0, 105)
            if metric_key in ("hota", "deta", "assa"):
                ax_left.set_ylim(0, 1.05)

            # Right: improvement vs baseline
            ax_right = axes_h[1]
            for key_pattern, label, color, marker, ls in group_models:
                ref_lbl = _get_baseline_label(label)
                if label == ref_lbl:
                    continue
                if ref_lbl not in model_data or len(model_data[ref_lbl]["rates"]) == 0:
                    continue
                data = model_data[label]
                means = data[f"{metric_key}_means"]
                stds = data[f"{metric_key}_stds"]
                ref_means_h = model_data[ref_lbl][f"{metric_key}_means"]
                ref_stds_h = model_data[ref_lbl][f"{metric_key}_stds"]
                if len(means) != len(ref_means_h) or len(means) == 0:
                    continue

                if higher_is_better:
                    improvements = (means - ref_means_h) / np.maximum(np.abs(ref_means_h), 1e-9) * 100
                else:
                    improvements = (ref_means_h - means) / np.maximum(np.abs(ref_means_h), 1e-9) * 100
                improvement_stds = np.sqrt(
                    (stds / np.maximum(np.abs(ref_means_h), 1e-9))**2 +
                    (means * ref_stds_h / np.maximum(ref_means_h**2, 1e-9))**2
                ) * 100

                short_label = label.replace("CI + ", "").replace("AKF + ", "").replace("PF + ", "")
                ax_right.plot(data["rates"], improvements, marker=marker, linestyle='-',
                              linewidth=2.5, markersize=8, color=color, label=short_label)
                ax_right.fill_between(data["rates"],
                                      improvements - improvement_stds,
                                      improvements + improvement_stds,
                                      alpha=0.15, color=color)

            ax_right.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax_right.set_xlabel('AV Injection Rate (%)', fontweight='bold', fontsize=11)
            ax_right.set_ylabel('Improvement (%)', fontweight='bold', fontsize=11)
            imp_note = "(positive = better)" if higher_is_better else "(positive = closer to GT)"
            ax_right.set_title(f'{ylabel} Improvement vs Baseline {imp_note}', fontweight='bold', fontsize=12)
            ax_right.legend(fontsize=10, loc='best')
            ax_right.grid(True, alpha=0.3)
            ax_right.set_xlim(0, 105)

            plt.tight_layout()
            fname = f"gpem_{filter_name.lower()}_{metric_key}_plot.png"
            output_file_h = results_path / fname
            plt.savefig(output_file_h, dpi=150, bbox_inches='tight')
            print(f"  {filter_name} {ylabel} -> {fname}")
            plt.close(fig_h)

    # Combined cross-filter plots: one line per filter, fixed covariance mode
    filter_color_map = {
        "Kalman": "#3498DB",
        "CI":     "#E74C3C",
        "AKF":    "#2ECC71",
        "PF":     "#F39C12",
    }
    filter_marker_map = {
        "Kalman": "o",
        "CI":     "s",
        "AKF":    "^",
        "PF":     "D",
    }
    combined_variants = {
        "gpem_linear": {
            "Kalman": "GPEM Linear",
            "CI":     "CI + GPEM Linear",
            "AKF":    "AKF + GPEM Linear",
            "PF":     "PF + GPEM Linear",
        },
    }
    for cov_label, filter_map in combined_variants.items():
        for metric_key, ylabel, higher_is_better in all_metrics:
            available = [(f, lbl) for f, lbl in filter_map.items()
                         if lbl in model_data and len(model_data[lbl]["rates"]) > 0]
            if len(available) < 2:
                continue

            fig_c, ax_c = plt.subplots(1, 1, figsize=(10, 5))
            fig_c.suptitle(f'{ylabel} Across Filters ({cov_label.replace("_", " ").title()})',
                           fontsize=14, fontweight='bold')

            for filter_name, lbl in available:
                data = model_data[lbl]
                means = data[f"{metric_key}_means"]
                stds = data[f"{metric_key}_stds"]
                color = filter_color_map[filter_name]
                marker = filter_marker_map[filter_name]
                ax_c.plot(data["rates"], means, marker=marker, linestyle='-', label=filter_name,
                          linewidth=2.5, markersize=8, color=color)
                ax_c.fill_between(data["rates"], means - stds, means + stds, alpha=0.15, color=color)

            ax_c.set_xlabel('AV Injection Rate (%)', fontweight='bold', fontsize=11)
            ax_c.set_ylabel(ylabel, fontweight='bold', fontsize=11)
            ax_c.legend(fontsize=10, loc='best')
            ax_c.grid(True, alpha=0.3)
            ax_c.set_xlim(0, 105)
            if metric_key in ("hota", "deta", "assa"):
                ax_c.set_ylim(0, 1.05)

            plt.tight_layout()
            fname_c = f"gpem_combined_{cov_label}_{metric_key}_plot.png"
            output_file_c = results_path / fname_c
            plt.savefig(output_file_c, dpi=150, bbox_inches='tight')
            print(f"  Combined {cov_label} {ylabel} -> {fname_c}")
            plt.close(fig_c)

    # Print summary statistics — compare against filter-matched baselines
    print("\n" + "="*70)
    print("GPEM TEST SUMMARY STATISTICS")
    print("="*70)

    for key_pattern, label, color, marker, ls in models:
        data = model_data[label]
        if len(data["rates"]) == 0:
            continue
        print(f"\n{label}:")
        print(f"  Mean AMOTA: {np.mean(data['amota_means']):.4f} +/- {np.mean(data['amota_stds']):.4f}")
        print(f"  Mean AMOTP: {np.mean(data['amotp_means']):.4f}m +/- {np.mean(data['amotp_stds']):.4f}")
        if np.any(data['hota_means'] > 0):
            print(f"  Mean HOTA:  {np.mean(data['hota_means']):.4f} +/- {np.mean(data['hota_stds']):.4f}")
            print(f"  Mean DetA:  {np.mean(data['deta_means']):.4f} | Mean AssA: {np.mean(data['assa_means']):.4f}")

        ref_lbl = _get_baseline_label(label)
        if label != ref_lbl and ref_lbl in model_data and len(model_data[ref_lbl]["rates"]) > 0:
            ref_amota = model_data[ref_lbl]["amota_means"]
            ref_amotp = model_data[ref_lbl]["amotp_means"]
            if len(data["amota_means"]) == len(ref_amota):
                amota_imp = (data["amota_means"] - ref_amota) / np.maximum(np.abs(ref_amota), 1e-9) * 100
                print(f"  AMOTA vs {ref_lbl}: {np.mean(amota_imp):+.2f}%")
                print(f"    Min: {np.min(amota_imp):+.2f}% (@ {av_rates[np.argmin(amota_imp)]:.1f}% AV)")
                print(f"    Max: {np.max(amota_imp):+.2f}% (@ {av_rates[np.argmax(amota_imp)]:.1f}% AV)")
                print(f"    Wins: {sum(1 for imp in amota_imp if imp > 0)}/{len(av_rates)} rates")
                if np.any(ref_amotp > 0):
                    amotp_imp = (ref_amotp - data["amotp_means"]) / np.maximum(np.abs(ref_amotp), 1e-9) * 100
                    print(f"  AMOTP vs {ref_lbl}: {np.mean(amotp_imp):+.2f}% (positive = closer to GT)")
                    print(f"    Min: {np.min(amotp_imp):+.2f}% (@ {av_rates[np.argmin(amotp_imp)]:.1f}% AV)")
                    print(f"    Max: {np.max(amotp_imp):+.2f}% (@ {av_rates[np.argmax(amotp_imp)]:.1f}% AV)")
                    print(f"    Wins: {sum(1 for imp in amotp_imp if imp > 0)}/{len(av_rates)} rates")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_gpem_results.py <results_directory>")
        print("\nExample:")
        print("  python plot_gpem_results.py results/GPEM_Test_Mode_2026-02-12_164112")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    plot_gpem_results_simple(results_dir)
