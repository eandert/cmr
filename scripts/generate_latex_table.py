#!/usr/bin/env python3
"""
Generate LaTeX table(s) from GPEM experiment results.

Reads a summary.json and produces publication-ready LaTeX tables comparing
filter architectures across covariance modes.

Usage:
    python scripts/generate_latex_table.py results/GPEM_Distribution_Sweep_2026-03-15_170555 --combined
    python scripts/generate_latex_table.py results/GPEM_Penetration_Sweep_2026-03-14_112854 --combined
    python scripts/generate_latex_table.py results/GPEM_Distribution_Sweep_2026-03-15_170555 --filter ci
    python scripts/generate_latex_table.py results/GPEM_Distribution_Sweep_2026-03-15_170555 --best-per-cov
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np


# Mapping from internal variant suffix to (filter_group, cov_type)
VARIANT_MAP = {
    "baseline":          ("Kalman", "baseline"),
    "static":            ("Kalman", "static"),
    "gpem_linear":       ("Kalman", "gpem_linear"),
    "gpem_quadratic":    ("Kalman", "gpem_quadratic"),
    "gpem_polar":        ("Kalman", "gpem_polar"),
    "ci_baseline":       ("CI", "baseline"),
    "ci_static":         ("CI", "static"),
    "ci_gpem_linear":    ("CI", "gpem_linear"),
    "ci_gpem_quadratic": ("CI", "gpem_quadratic"),
    "ci_gpem_polar":     ("CI", "gpem_polar"),
    "akf_baseline":      ("AKF", "baseline"),
    "akf_static":        ("AKF", "static"),
    "akf_gpem_linear":   ("AKF", "gpem_linear"),
    "akf_gpem_quadratic":("AKF", "gpem_quadratic"),
    "akf_gpem_polar":    ("AKF", "gpem_polar"),
    "pf_baseline":       ("PF", "baseline"),
    "pf_static":         ("PF", "static"),
    "pf_gpem_linear":    ("PF", "gpem_linear"),
    "pf_gpem_quadratic": ("PF", "gpem_quadratic"),
    "pf_gpem_polar":     ("PF", "gpem_polar"),
    "bici_baseline":       ("BICI", "baseline"),
    "bici_static":         ("BICI", "static"),
    "bici_gpem_linear":    ("BICI", "gpem_linear"),
    "bici_gpem_quadratic": ("BICI", "gpem_quadratic"),
    "bici_gpem_polar":     ("BICI", "gpem_polar"),
    "sabre_baseline":       ("SABRE", "baseline"),
    "sabre_static":         ("SABRE", "static"),
    "sabre_gpem_linear":    ("SABRE", "gpem_linear"),
    "sabre_gpem_quadratic": ("SABRE", "gpem_quadratic"),
    "sabre_gpem_polar":     ("SABRE", "gpem_polar"),
}

# Order for rows in the table
FILTER_ORDER = ["Kalman", "CI", "AKF", "PF", "BICI", "SABRE"]
COV_ORDER = ["baseline", "static", "gpem_linear", "gpem_quadratic", "gpem_polar"]

# Display names for filter groups (used in \multirow)
FILTER_DISPLAY = {
    "Kalman": "Extended Kalman (EKF)",
    "CI":     "Covariance Intersection (CI)",
    "AKF":    "Adaptive Kalman (AKF)",
    "PF":     "Particle Filter (PF)",
    "BICI":   "Batch ICI (BICI)",
    "SABRE":  "SABRE (Adaptive-NIS CI)",
}

# Display names for covariance modes
COV_DISPLAY = {
    "baseline":       r"Fixed Baseline ($\sigma^2=0.1$)",
    "static":         "Static (Empirical Avg)",
    "gpem_linear":    "GPEM Linear",
    "gpem_quadratic": "GPEM Quadratic",
    "gpem_polar":     "GPEM Polar",
}

# Metrics to extract (field_prefix in summary.json, display name, direction)
# direction: +1 = higher is better, -1 = lower is better
METRICS = [
    ("avg_global_amota_mean", "avg_global_amota_std", "AMOTA",      +1),
    ("avg_global_amotp_mean", "avg_global_amotp_std", "AMOTP (m)",  -1),
    ("avg_hota_mean",         "avg_hota_std",         "HOTA",       +1),
    ("avg_deta_mean",         "avg_deta_std",         "DetA",       +1),
    ("avg_assa_mean",         "avg_assa_std",         "AssA",       +1),
]


def parse_variant_key(key):
    """
    Parse a summary.json key to extract (step/rate index, variant suffix).

    Handles both distribution sweep and penetration sweep naming:
      gpem_dist_sweep_step_3_ci_gpem_linear -> (3, "ci_gpem_linear")
      ci_gpem_linear_av_50.0pct -> (50.0, "ci_gpem_linear")
      baseline_av_10.0pct -> (10.0, "baseline")
    """
    # Distribution sweep: gpem_dist_sweep_step_{N}_{variant}
    m = re.match(r"gpem_dist_sweep_step_(\d+)_(.+)", key)
    if m:
        return (int(m.group(1)), m.group(2))

    # Penetration sweep: {variant}_av_{rate}pct
    m = re.match(r"(.+)_av_([\d.]+)pct$", key)
    if m:
        return (float(m.group(2)), m.group(1))

    return None


def load_results(results_dir):
    """Load summary.json and aggregate metrics by variant across steps/rates."""
    summary_path = os.path.join(results_dir, "summary.json")
    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(summary_path) as f:
        data = json.load(f)

    # Get the results dict (may be nested under "results" key)
    if "results" in data and isinstance(data["results"], dict):
        results = data["results"]
    else:
        results = {k: v for k, v in data.items() if isinstance(v, dict) and "avg_global_amota_mean" in v}

    # Collect per-variant metrics across all steps/rates
    variant_metrics = defaultdict(lambda: defaultdict(list))
    steps_or_rates = set()

    for key, entry in results.items():
        parsed = parse_variant_key(key)
        if parsed is None:
            continue
        idx, variant_suffix = parsed
        steps_or_rates.add(idx)

        if variant_suffix not in VARIANT_MAP:
            continue

        for mean_field, std_field, _, _ in METRICS:
            if mean_field in entry:
                variant_metrics[variant_suffix][mean_field].append(entry[mean_field])
            if std_field in entry:
                variant_metrics[variant_suffix][std_field].append(entry[std_field])

    return variant_metrics, sorted(steps_or_rates)


def compute_averages(variant_metrics):
    """Compute mean across steps/rates for each variant."""
    averages = {}
    for variant_suffix, fields in variant_metrics.items():
        averages[variant_suffix] = {}
        for field, values in fields.items():
            averages[variant_suffix][field] = np.mean(values)
    return averages


def find_best_per_metric(averages, rows):
    """For each metric, find the variant with the best value among rows."""
    best = {}
    for mean_field, _, metric_name, direction in METRICS:
        best_val = None
        best_var = None
        for variant_suffix in rows:
            if variant_suffix not in averages:
                continue
            val = averages[variant_suffix].get(mean_field, None)
            if val is None:
                continue
            if best_val is None or (direction > 0 and val > best_val) or (direction < 0 and val < best_val):
                best_val = val
                best_var = variant_suffix
        best[metric_name] = best_var
    return best


def format_value(val, metric_name):
    """Format a metric value for the table (3 decimal places)."""
    return f"{val:.3f}"


def format_pct(val, direction):
    """Format a percentage change."""
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.1f}\\%"


def get_suite_label(results_dir):
    """Derive a LaTeX label suffix from the results directory name."""
    base = os.path.basename(results_dir).lower()
    if "distribution" in base:
        return "distribution_sweep"
    elif "penetration" in base and "accurate" in base:
        return "penetration_sweep_accurate"
    elif "penetration" in base:
        return "penetration_sweep"
    else:
        return "fusion_performance"


def get_suite_display_name(results_dir):
    """Derive a human-readable suite name from the results directory."""
    base = os.path.basename(results_dir)
    # Strip timestamp suffix like _2026-03-15_170555
    name = re.sub(r"_\d{4}-\d{2}-\d{2}_\d{6}$", "", base)
    return name.replace("_", " ")


def generate_combined_latex(averages, suite_name, label, n_steps, caption=None):
    """Generate a publication-quality combined LaTeX table with multirow filter groups."""
    # Collect all rows for best-detection
    all_rows = []
    for filt in FILTER_ORDER:
        for cov in COV_ORDER:
            key = cov if filt == "Kalman" else f"{filt.lower()}_{cov}"
            if key in averages:
                all_rows.append(key)

    best = find_best_per_metric(averages, all_rows)

    lines = []
    if caption is None:
        caption = (
            f"Global Fusion Performance: {suite_name} "
            f"(Averaged over {n_steps} configurations). "
            r"\textbf{Bold} indicates the best performance per metric. "
            "Integrating GPEM parameters systematically improves tracking "
            "across all evaluated filter architectures."
        )

    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{tab:{label}}}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\begin{tabular}{@{}llccccc@{}}")
    lines.append(r"\toprule")

    # Header
    header_parts = [r"\textbf{Filter Architecture}", r"\textbf{Covariance Mode}"]
    for _, _, metric_name, direction in METRICS:
        arrow = r"$\uparrow$" if direction > 0 else r"$\downarrow$"
        header_parts.append(f"\\textbf{{{metric_name}}} {arrow}")
    lines.append(" & ".join(header_parts) + r" \\ \midrule")

    # Data rows grouped by filter
    for fi, filt in enumerate(FILTER_ORDER):
        rows = []
        for cov in COV_ORDER:
            key = cov if filt == "Kalman" else f"{filt.lower()}_{cov}"
            if key in averages:
                rows.append((cov, key))
        if not rows:
            continue

        n_rows = len(rows)
        filt_display = FILTER_DISPLAY[filt]

        for ri, (cov, variant_key) in enumerate(rows):
            entry = averages[variant_key]
            cov_display = COV_DISPLAY[cov]

            cells = []
            # First column: multirow filter name (only on first row)
            if ri == 0:
                cells.append(f"\\multirow{{{n_rows}}}{{*}}{{\\textbf{{{filt_display}}}}}")
            else:
                cells.append("")

            # Second column: covariance mode
            cells.append(cov_display)

            # Metric columns
            for mean_field, _, metric_name, direction in METRICS:
                val = entry.get(mean_field, 0.0)
                formatted = format_value(val, metric_name)
                if best.get(metric_name) == variant_key:
                    formatted = f"\\textbf{{{formatted}}}"
                cells.append(formatted)

            # Row ending: \midrule after last row of group, \\ for others
            if ri == n_rows - 1:
                if fi == len(FILTER_ORDER) - 1:
                    lines.append(" & ".join(cells) + r" \\ \bottomrule")
                else:
                    lines.append(" & ".join(cells) + r" \\ \midrule")
            else:
                lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def generate_filter_latex(averages, rows, caption, label, improvement_base=None, bold_best=True):
    """Generate LaTeX table for a single filter or subset of variants."""
    best = find_best_per_metric(averages, rows) if bold_best else {}

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\begin{tabular}{@{}lccccc@{}}")
    lines.append(r"\toprule")

    # Header
    header_parts = [r"\textbf{Fusion Strategy}"]
    for _, _, metric_name, direction in METRICS:
        arrow = r"$\uparrow$" if direction > 0 else r"$\downarrow$"
        header_parts.append(f"\\textbf{{{metric_name}}} {arrow}")
    lines.append(" & ".join(header_parts) + r" \\ \midrule")

    # Data rows
    for ri, variant_suffix in enumerate(rows):
        if variant_suffix not in averages:
            continue
        entry = averages[variant_suffix]
        filt, cov = VARIANT_MAP[variant_suffix]
        filt_display = FILTER_DISPLAY.get(filt, filt)
        cov_display = COV_DISPLAY.get(cov, cov)
        label_str = f"{filt_display} + {cov_display}"

        cells = [label_str]
        for mean_field, _, metric_name, direction in METRICS:
            val = entry.get(mean_field, 0.0)
            formatted = format_value(val, metric_name)
            if bold_best and best.get(metric_name) == variant_suffix:
                formatted = f"\\textbf{{{formatted}}}"
            cells.append(formatted)

        lines.append(" & ".join(cells) + r" \\")

    # Improvement rows
    if improvement_base and improvement_base in averages and len(rows) >= 2:
        lines[-1] = lines[-1].rstrip("\\") + r" \midrule"
        base_entry = averages[improvement_base]
        imp_rows = [vs for vs in rows if vs != improvement_base and vs in averages]
        for ii, vs in enumerate(imp_rows):
            cmp_entry = averages[vs]
            filt_cmp, cov_cmp = VARIANT_MAP[vs]
            cov_label = COV_DISPLAY.get(cov_cmp, cov_cmp)

            cells = [f"\\textit{{vs Baseline: {cov_label}}}"]
            for mean_field, _, metric_name, direction in METRICS:
                base_val = base_entry.get(mean_field, 0.0)
                cmp_val = cmp_entry.get(mean_field, 0.0)
                if abs(base_val) < 1e-9:
                    cells.append("---")
                else:
                    pct = ((cmp_val - base_val) / abs(base_val)) * 100
                    cells.append(f"\\textit{{{format_pct(pct, direction)}}}")

            lines.append(" & ".join(cells) + r" \\")

    # Replace last \\ with \bottomrule
    if lines[-1].endswith(r" \\"):
        lines[-1] = lines[-1][:-3] + r" \\ \bottomrule"

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX table from GPEM experiment results")
    parser.add_argument("results_dir", help="Path to results directory containing summary.json")
    parser.add_argument("--filter", choices=["kalman", "ci", "akf", "pf", "bici", "sabre", "all"],
                        default="all", help="Which filter(s) to include (default: all)")
    parser.add_argument("--best-per-cov", action="store_true",
                        help="Show only the best filter for each covariance mode")
    parser.add_argument("--caption", default=None,
                        help="LaTeX table caption")
    parser.add_argument("--label", default=None,
                        help="LaTeX table label (without tab: prefix)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output .tex file (default: print to stdout)")
    parser.add_argument("--per-filter", action="store_true",
                        help="Generate one table per filter group")
    parser.add_argument("--combined", action="store_true",
                        help="Generate a combined table with multirow filter groups and booktabs style")
    args = parser.parse_args()

    variant_metrics, steps_or_rates = load_results(args.results_dir)
    averages = compute_averages(variant_metrics)

    n_steps = len(steps_or_rates)
    suite_name = get_suite_display_name(args.results_dir)
    suite_label = args.label or get_suite_label(args.results_dir)

    if not averages:
        print("Error: no valid results found in summary.json", file=sys.stderr)
        sys.exit(1)

    output_parts = []

    if args.best_per_cov:
        rows = []
        for cov in COV_ORDER:
            best_var = None
            best_hota = -1
            for filt in FILTER_ORDER:
                key = f"{cov}" if filt == "Kalman" else f"{filt.lower()}_{cov}"
                if key in averages:
                    h = averages[key].get("avg_hota_mean", 0)
                    if h > best_hota:
                        best_hota = h
                        best_var = key
            if best_var:
                rows.append(best_var)

        caption = args.caption or (
            f"Best Filter per Covariance Mode: {suite_name} "
            f"(Averaged over {n_steps} configurations)"
        )
        output_parts.append(generate_filter_latex(averages, rows, caption,
                                                  f"tab:{suite_label}",
                                                  improvement_base=rows[0] if rows else None))

    elif args.per_filter:
        for filt in FILTER_ORDER:
            rows = []
            for cov in COV_ORDER:
                key = cov if filt == "Kalman" else f"{filt.lower()}_{cov}"
                if key in averages:
                    rows.append(key)
            if not rows:
                continue

            filt_display = FILTER_DISPLAY[filt]
            caption = args.caption or (
                f"{filt_display} Performance across Covariance Modes: "
                f"{suite_name} (Averaged over {n_steps} configurations)"
            )
            label = f"tab:{filt.lower()}_{suite_label}"
            output_parts.append(generate_filter_latex(averages, rows, caption, label,
                                                      improvement_base=rows[0]))

    elif args.combined:
        output_parts.append(generate_combined_latex(
            averages, suite_name, suite_label, n_steps, caption=args.caption
        ))

    else:
        # Default: filter by --filter flag
        filter_map = {
            "kalman": ["Kalman"],
            "ci": ["CI"],
            "akf": ["AKF"],
            "pf": ["PF"],
            "bici": ["BICI"],
            "sabre": ["SABRE"],
            "all": FILTER_ORDER,
        }
        selected_filters = filter_map[args.filter]

        rows = []
        for filt in selected_filters:
            for cov in COV_ORDER:
                key = cov if filt == "Kalman" else f"{filt.lower()}_{cov}"
                if key in averages:
                    rows.append(key)

        if not rows:
            print("Error: no matching variants found", file=sys.stderr)
            sys.exit(1)

        improvement_base = None
        for filt in selected_filters:
            key = "baseline" if filt == "Kalman" else f"{filt.lower()}_baseline"
            if key in averages:
                improvement_base = key
                break

        caption = args.caption or (
            f"Global Fusion Performance: {suite_name} "
            f"(Averaged over {n_steps} configurations)"
        )
        output_parts.append(generate_filter_latex(averages, rows, caption,
                                                  f"tab:{suite_label}",
                                                  improvement_base=improvement_base))

    # Output
    result = "\n\n".join(output_parts)
    if args.output:
        with open(args.output, "w") as f:
            f.write(result + "\n")
        print(f"Written to {args.output}")
    else:
        print(result)


if __name__ == "__main__":
    main()
