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
    
    # Model configurations: (key_pattern, label, color, marker)
    models = [
        ("static_cov_av_{rate:.1f}pct", "Static Average", "#FF6B6B", "o"),
        ("gpem_linear_av_{rate:.1f}pct", "GPEM Linear", "#4ECDC4", "s"),
        ("gpem_quadratic_av_{rate:.1f}pct", "GPEM Quadratic", "#9B59B6", "^"),
    ]
    
    # Fallback for old naming convention (gpem_model_av_X instead of gpem_linear_av_X)
    legacy_models = [
        ("static_cov_av_{rate:.1f}pct", "Static Average", "#FF6B6B", "o"),
        ("gpem_model_av_{rate:.1f}pct", "GPEM Model", "#4ECDC4", "s"),
    ]
    
    # Check which model set we have
    test_rate = potential_av_rates[0]
    has_new_format = f"gpem_linear_av_{test_rate:.1f}pct" in summary['results']
    has_quadratic = f"gpem_quadratic_av_{test_rate:.1f}pct" in summary['results']
    
    if not has_new_format:
        models = legacy_models
        print("Note: Using legacy 2-model format (static vs gpem)")
    elif not has_quadratic:
        models = models[:2]  # Only static and linear
        print("Note: Quadratic results not found, plotting static vs linear only")
    
    # Collect data for each model
    model_data = {m[1]: {"rates": [], "amota_means": [], "amota_stds": []} for m in models}
    
    for rate in potential_av_rates:
        all_found = True
        for key_pattern, label, _, _ in models:
            key = key_pattern.format(rate=rate)
            if key not in summary['results']:
                all_found = False
                break
        
        if all_found:
            for key_pattern, label, _, _ in models:
                key = key_pattern.format(rate=rate)
                res = summary['results'][key]
                model_data[label]["rates"].append(rate)
                model_data[label]["amota_means"].append(res['avg_global_amota_mean'])
                model_data[label]["amota_stds"].append(res['avg_global_amota_std'])
    
    # Convert to numpy arrays
    for label in model_data:
        model_data[label]["rates"] = np.array(model_data[label]["rates"])
        model_data[label]["amota_means"] = np.array(model_data[label]["amota_means"])
        model_data[label]["amota_stds"] = np.array(model_data[label]["amota_stds"])
    
    # Get baseline model (static) for improvement calculations
    baseline_label = models[0][1]
    if len(model_data[baseline_label]["rates"]) == 0:
        print("Error: No valid data found in summary.json")
        return
    
    av_rates = model_data[baseline_label]["rates"]
    baseline_means = model_data[baseline_label]["amota_means"]
    baseline_stds = model_data[baseline_label]["amota_stds"]
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    title_suffix = "A/B/C" if len(models) == 3 else "A/B"
    fig.suptitle(f'GPEM Penetration Sweep: {title_suffix} Comparison Results (AMOTA)', fontsize=14, fontweight='bold')
    
    # Plot 1: AMOTA with confidence bands
    ax1 = axes[0]
    
    for key_pattern, label, color, marker in models:
        data = model_data[label]
        ax1.plot(data["rates"], data["amota_means"], f'{marker}-', label=label, 
                linewidth=2.5, markersize=8, color=color)
        ax1.fill_between(data["rates"], 
                         data["amota_means"] - data["amota_stds"],
                         data["amota_means"] + data["amota_stds"],
                         alpha=0.2, color=color)
    
    ax1.set_xlabel('AV Injection Rate (%)', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Global AMOTA', fontweight='bold', fontsize=11)
    ax1.set_title('AMOTA Comparison with Standard Deviation', fontweight='bold', fontsize=12)
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 105)
    
    # Plot 2: AMOTA Improvement with confidence bands (relative to baseline)
    ax2 = axes[1]
    
    improvement_colors = ['#27AE60', '#3498DB', '#E74C3C']  # Green, Blue, Red
    
    for idx, (key_pattern, label, color, marker) in enumerate(models[1:], start=0):  # Skip baseline
        data = model_data[label]
        
        # Calculate improvement over baseline
        improvements = (data["amota_means"] - baseline_means) / baseline_means * 100
        
        # Calculate std dev of improvement using error propagation
        improvement_stds = np.sqrt((data["amota_stds"]/baseline_means)**2 + 
                                   (data["amota_means"] * baseline_stds / baseline_means**2)**2) * 100
        
        ax2.plot(data["rates"], improvements, f'{marker}-', linewidth=2.5, markersize=8, 
                color=improvement_colors[idx], label=f'{label} vs Static')
        ax2.fill_between(data["rates"],
                         improvements - improvement_stds,
                         improvements + improvement_stds,
                         alpha=0.2, color=improvement_colors[idx])
    
    # Zero line
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    ax2.set_xlabel('AV Injection Rate (%)', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Improvement (%)', fontweight='bold', fontsize=11)
    ax2.set_title('GPEM AMOTA Improvement over Static Model', fontweight='bold', fontsize=12)
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 105)
    
    plt.tight_layout()
    
    # Save figure
    output_file = results_path / "gpem_comparison_plot.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_file}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("GPEM TEST SUMMARY STATISTICS")
    print("="*70)
    
    for key_pattern, label, color, marker in models:
        data = model_data[label]
        print(f"\n{label}:")
        print(f"  Mean AMOTA: {np.mean(data['amota_means']):.4f} ± {np.mean(data['amota_stds']):.4f}")
        
        if label != baseline_label:
            improvements = (data["amota_means"] - baseline_means) / baseline_means * 100
            print(f"  Average improvement vs Static: {np.mean(improvements):.2f}%")
            print(f"  Min improvement: {np.min(improvements):.2f}% (@ {av_rates[np.argmin(improvements)]:.1f}% AV)")
            print(f"  Max improvement: {np.max(improvements):.2f}% (@ {av_rates[np.argmax(improvements)]:.1f}% AV)")
            print(f"  Wins over Static: {sum(1 for imp in improvements if imp > 0)}/{len(av_rates)} rates")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_gpem_results.py <results_directory>")
        print("\nExample:")
        print("  python plot_gpem_results.py results/GPEM_Test_Mode_2026-02-12_164112")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    plot_gpem_results_simple(results_dir)
