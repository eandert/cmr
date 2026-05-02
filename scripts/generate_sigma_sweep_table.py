#!/usr/bin/env python3
"""
Generate publication-quality sigma_a sweep table and heatmap for paper appendix.

Produces:
  - LaTeX table: σ_a × filter avg HOTA (all configs equally weighted)
  - Heatmap PDF: σ_a vs filter, colored by avg GPEM HOTA
  - Line plot PDF: σ_a on x-axis, avg HOTA per filter on y-axis

Usage:
    python3 scripts/generate_sigma_sweep_table.py [sweep_output_file]
"""
import json, glob, re, sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Default to most recent sweep output
SWEEP_FILE = sys.argv[1] if len(sys.argv) > 1 else None

# Require explicit path
if SWEEP_FILE is None:
    print("Usage: python3 scripts/generate_sigma_sweep_table.py <sweep_output_file>")
    sys.exit(1)

if SWEEP_FILE is None:
    print("No sweep output found. Pass path as argument.")
    sys.exit(1)

print(f"Using sweep output: {SWEEP_FILE}")

# Parse step info
output = open(SWEEP_FILE).read()
blocks = re.split(r'(\[\d+/33\][^\n]+)', output)
step_info = []
for i in range(1, len(blocks), 2):
    m = re.search(r'σ_a=([0-9.]+)\s+(fast_\w+)', blocks[i])
    if m: step_info.append((float(m.group(1)), m.group(2)))

# Find result dirs from same date
# Extract date from first dir
date_str = None
for pattern in ['results/GPEM_Distribution_Sweep_2026-*']:
    dirs = sorted(glob.glob(pattern))
    if dirs:
        m = re.search(r'(\d{4}-\d{2}-\d{2})', dirs[-1])
        if m:
            date_str = m.group(1)
            break

if date_str is None:
    date_str = '2026-03-30'

all_dist = sorted(glob.glob(f'results/GPEM_Distribution_Sweep_{date_str}_*'))
all_pen = sorted(glob.glob(f'results/GPEM_Penetration_Sweep_{date_str}_*'))
print(f"Found {len(all_dist)} dist + {len(all_pen)} pen dirs for {date_str}")

# Filter definitions
filter_names = ['EKF', 'CI', 'BICI', 'AKF', 'PF']
filter_gpem_modes = {
    'EKF': ['gpem_linear', 'gpem_quadratic', 'gpem_polar', 'static'],
    'CI':  ['ci_gpem_linear', 'ci_gpem_quadratic', 'ci_gpem_polar', 'ci_static'],
    'BICI':['bici_gpem_linear', 'bici_gpem_quadratic', 'bici_gpem_polar', 'bici_static'],
    'AKF': ['akf_gpem_linear', 'akf_gpem_quadratic', 'akf_gpem_polar', 'akf_static'],
    'PF':  ['pf_gpem_linear', 'pf_gpem_quadratic', 'pf_gpem_polar', 'pf_static'],
}
filter_baselines = {
    'EKF': 'baseline', 'CI': 'ci_baseline', 'BICI': 'bici_baseline',
    'AKF': 'akf_baseline', 'PF': 'pf_baseline',
}

sigmas = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

# Collect: sigma -> filter -> [hota_values]
# Also: sigma -> filter -> [baseline_hota_values]
gpem_hota = {s: {f: [] for f in filter_names} for s in sigmas}
base_hota = {s: {f: [] for f in filter_names} for s in sigmas}

for si, (sigma, map_name) in enumerate(step_info):
    if si >= len(all_dist) or si >= len(all_pen): continue
    for dirpath in [all_dist[si], all_pen[si]]:
        try:
            r = json.load(open(f'{dirpath}/summary.json')).get('results', {})
        except: continue
        for fname in filter_names:
            # Baseline HOTA
            bkey = filter_baselines[fname]
            for ck, cv in r.items():
                if ck.endswith(f'_{bkey}') or ck.startswith(f'{bkey}_av_'):
                    h = cv.get('avg_hota_mean', 0)
                    if h > 0: base_hota[sigma][fname].append(h)
            # GPEM modes HOTA
            for mode in filter_gpem_modes[fname]:
                for ck, cv in r.items():
                    if ck.endswith(f'_{mode}') or ck.startswith(f'{mode}_av_'):
                        h = cv.get('avg_hota_mean', 0)
                        if h > 0: gpem_hota[sigma][fname].append(h)

# ============================================================
# 1. HEATMAP
# ============================================================
valid_sigmas = [s for s in sigmas if any(gpem_hota[s][f] for f in filter_names)]
data_matrix = np.zeros((len(valid_sigmas), len(filter_names)))
for i, s in enumerate(valid_sigmas):
    for j, f in enumerate(filter_names):
        vals = gpem_hota[s][f]
        data_matrix[i, j] = np.mean(vals) if vals else 0

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto', vmin=0.3, vmax=1.0)
ax.set_xticks(range(len(filter_names)))
ax.set_xticklabels(filter_names, fontsize=11)
ax.set_yticks(range(len(valid_sigmas)))
ax.set_yticklabels([f'{s:.0f}' for s in valid_sigmas], fontsize=10)
ax.set_xlabel('Filter', fontsize=12)
ax.set_ylabel('σ_a (m/s²)', fontsize=12)
ax.set_title('Average GPEM HOTA by σ_a and Filter\n(across all maps, penetration levels, and fleet compositions)', fontsize=11)

# Annotate cells
for i in range(len(valid_sigmas)):
    for j in range(len(filter_names)):
        val = data_matrix[i, j]
        color = 'white' if val > 0.75 else 'black'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=8, color=color)

plt.colorbar(im, label='HOTA')
plt.tight_layout()
plt.savefig('results/sigma_sweep_heatmap.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results/sigma_sweep_heatmap.png', dpi=150, bbox_inches='tight')
print("Saved: results/sigma_sweep_heatmap.pdf")

# ============================================================
# 2. LINE PLOT
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

colors = {'EKF': '#1f77b4', 'CI': '#ff7f0e', 'BICI': '#2ca02c', 'AKF': '#d62728', 'PF': '#9467bd'}
markers = {'EKF': 'o', 'CI': 's', 'BICI': 'D', 'AKF': '^', 'PF': 'v'}

# Left: GPEM HOTA
for fname in filter_names:
    xs, ys = [], []
    for s in valid_sigmas:
        vals = gpem_hota[s][fname]
        if vals:
            xs.append(s)
            ys.append(np.mean(vals))
    if xs:
        ax1.plot(xs, ys, f'-{markers[fname]}', color=colors[fname], label=f'{fname} GPEM', linewidth=1.5, markersize=5)

ax1.set_xlabel('σ_a (m/s²)', fontsize=12)
ax1.set_ylabel('Average HOTA', fontsize=12)
ax1.set_title('GPEM HOTA vs σ_a', fontsize=12)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0.2, 1.0)

# Right: Baseline HOTA
for fname in filter_names:
    xs, ys = [], []
    for s in valid_sigmas:
        vals = base_hota[s][fname]
        if vals:
            xs.append(s)
            ys.append(np.mean(vals))
    if xs:
        ax2.plot(xs, ys, f'-{markers[fname]}', color=colors[fname], label=f'{fname} Base', linewidth=1.5, markersize=5, linestyle='--')

ax2.set_xlabel('σ_a (m/s²)', fontsize=12)
ax2.set_ylabel('Average HOTA', fontsize=12)
ax2.set_title('Baseline HOTA vs σ_a', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.2, 1.0)

plt.suptitle('Process Noise Sensitivity: σ_a Sweep\n(equal weight: 3 maps × 8 distribution steps × 13 penetration levels per σ_a)', fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig('results/sigma_sweep_lines.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results/sigma_sweep_lines.png', dpi=150, bbox_inches='tight')
print("Saved: results/sigma_sweep_lines.pdf")

# ============================================================
# 3. LATEX TABLE
# ============================================================
print("\n% LaTeX table for appendix")
print("\\begin{table}[h]")
print("\\centering")
print("\\caption{Average GPEM HOTA by $\\sigma_a$ and filter architecture, equally weighted across all maps, penetration levels, and fleet compositions.}")
print("\\label{tab:sigma_sweep}")
print("\\begin{tabular}{r" + "c" * len(filter_names) + "c}")
print("\\toprule")
print("$\\sigma_a$ & " + " & ".join(filter_names) + " & All \\\\")
print("\\midrule")

best_sigma_all = None
best_val_all = 0
for s in valid_sigmas:
    row_vals = []
    for fname in filter_names:
        vals = gpem_hota[s][fname]
        v = np.mean(vals) if vals else 0
        row_vals.append(v)
    all_vals = []
    for fname in filter_names:
        all_vals.extend(gpem_hota[s][fname])
    avg_all = np.mean(all_vals) if all_vals else 0
    if avg_all > best_val_all:
        best_val_all = avg_all
        best_sigma_all = s

    cells = []
    for v in row_vals:
        cells.append(f'{v:.3f}')
    cells.append(f'{avg_all:.3f}')
    print(f'{s:.0f} & ' + ' & '.join(cells) + ' \\\\')

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

print(f"\nBest σ_a (max all-filter avg GPEM HOTA): {best_sigma_all}")

# CI/BICI specific best
best_ci = max(valid_sigmas, key=lambda s: np.mean(gpem_hota[s]['CI'] + gpem_hota[s]['BICI']) if gpem_hota[s]['CI'] else 0)
print(f"Best σ_a (max CI/BICI avg GPEM HOTA): {best_ci}")
