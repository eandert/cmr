#!/usr/bin/env python3
"""
Generate 3x2 sigma_a sweep heatmap for paper appendix.

Top row: Average across all GPEM modes
Bottom row: Best-of-any-GPEM-mode per config

Usage:
    python3 scripts/generate_sigma_sweep_heatmap.py [sweep_output_file]
"""
import json, glob, re, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# Find sweep output
SWEEP_FILE = sys.argv[1] if len(sys.argv) > 1 else None
if SWEEP_FILE is None:
    print("Usage: python3 scripts/generate_sigma_sweep_heatmap.py <sweep_output_file>")
    sys.exit(1)
print(f"Using: {SWEEP_FILE}")

# Parse step info
output = open(SWEEP_FILE).read()
blocks = re.split(r'(\[\d+/\d+\][^\n]+)', output)
step_info = []
for i in range(1, len(blocks), 2):
    m = re.search(r'σ_a=([0-9.]+)\s+(fast_\w+)', blocks[i])
    if m: step_info.append((float(m.group(1)), m.group(2)))

# Find result dirs matching this sweep — take the last N dirs where N = len(step_info)
n_steps = len(step_info)
all_dist_candidates = sorted(glob.glob('results/GPEM_Distribution_Sweep_*'))
all_pen_candidates = sorted(glob.glob('results/GPEM_Penetration_Sweep_*'))
all_dist = all_dist_candidates[-n_steps:] if len(all_dist_candidates) >= n_steps else all_dist_candidates
all_pen = all_pen_candidates[-n_steps:] if len(all_pen_candidates) >= n_steps else all_pen_candidates
print(f"Found {len(all_dist)} dist + {len(all_pen)} pen dirs (last {n_steps} of each)")

# Config
filter_modes = {
    'EKF': ['gpem_linear', 'gpem_quadratic', 'gpem_polar', 'static'],
    'CI':  ['ci_gpem_linear', 'ci_gpem_quadratic', 'ci_gpem_polar', 'ci_static'],
    'BICI':['bici_gpem_linear', 'bici_gpem_quadratic', 'bici_gpem_polar', 'bici_static'],
    'AKF': ['akf_gpem_linear', 'akf_gpem_quadratic', 'akf_gpem_polar', 'akf_static'],
    'PF':  ['pf_gpem_linear', 'pf_gpem_quadratic', 'pf_gpem_polar', 'pf_static'],
    'SABRE':['sabre_gpem_linear', 'sabre_gpem_quadratic', 'sabre_gpem_polar', 'sabre_static'],
}
filter_baselines = {'EKF': 'baseline', 'AKF': 'akf_baseline', 'PF': 'pf_baseline',
                    'CI': 'ci_baseline', 'BICI': 'bici_baseline', 'SABRE': 'sabre_baseline'}
map_labels = {'fast_city': 'City', 'fast_highway': 'Highway', 'fast_rural': 'Rural'}
sigmas = sorted(set(s for s, _ in step_info))
fnames = ['EKF', 'AKF', 'PF', 'CI', 'BICI', 'SABRE']

# Collect data
gpem_per_cfg = {s: {f: {m: [] for m in map_labels} for f in fnames} for s in sigmas}
base_per_cfg = {s: {f: {m: [] for m in map_labels} for f in fnames} for s in sigmas}

for si, (sigma, map_name) in enumerate(step_info):
    if sigma not in gpem_per_cfg or si >= len(all_dist) or si >= len(all_pen): continue
    for dirpath in [all_dist[si], all_pen[si]]:
        try: r = json.load(open(f'{dirpath}/summary.json')).get('results', {})
        except: continue
        for fname in fnames:
            for mode in filter_modes[fname]:
                for ck, cv in r.items():
                    if ck.endswith(f'_{mode}') or ck.startswith(f'{mode}_av_'):
                        h = cv.get('avg_hota_mean', 0)
                        if h > 0: gpem_per_cfg[sigma][fname][map_name].append((mode, ck, h))
            bkey = filter_baselines[fname]
            for ck, cv in r.items():
                if ck.endswith(f'_{bkey}') or ck.startswith(f'{bkey}_av_'):
                    h = cv.get('avg_hota_mean', 0)
                    if h > 0: base_per_cfg[sigma][fname][map_name].append((bkey, ck, h))

# Build matrices
avg_gpem = np.zeros((len(sigmas), len(fnames)))
avg_pct = np.zeros((len(sigmas), len(fnames)))
best_gpem = np.zeros((len(sigmas), len(fnames)))
best_pct = np.zeros((len(sigmas), len(fnames)))

for i, s in enumerate(sigmas):
    for j, f in enumerate(fnames):
        g_all, b_all, best_pairs = [], [], []
        for mk in map_labels:
            g_vals = [h for _, _, h in gpem_per_cfg[s][f][mk]]
            b_vals = [h for _, _, h in base_per_cfg[s][f][mk]]
            g_all.extend(g_vals); b_all.extend(b_vals)
            b_by_cfg = {bck: bh for _, bck, bh in base_per_cfg[s][f][mk]}
            for cfg_id, bh in b_by_cfg.items():
                best_gh = max((gh for _, _, gh in gpem_per_cfg[s][f][mk]), default=-1)
                if best_gh > 0: best_pairs.append((bh, best_gh))
        avg_g = np.mean(g_all) if g_all else 0
        avg_b = np.mean(b_all) if b_all else 0
        avg_gpem[i, j] = avg_g
        avg_pct[i, j] = ((avg_g - avg_b) / max(abs(avg_b), 0.001)) * 100 if b_all else 0
        if best_pairs:
            best_gpem[i, j] = np.mean([g for _, g in best_pairs])
            best_pct[i, j] = ((np.mean([g for _, g in best_pairs]) - np.mean([b for b, _ in best_pairs])) /
                              max(abs(np.mean([b for b, _ in best_pairs])), 0.001)) * 100

# Colormaps
delta_cmap = LinearSegmentedColormap.from_list('rwg', [
    (0.0, '#c62828'), (0.2, '#ef5350'), (0.35, '#ffee58'),
    (0.5, '#ffffff'), (0.65, '#a5d6a7'), (0.8, '#43a047'), (1.0, '#1b5e20')])

colors = {'EKF': '#1f77b4', 'AKF': '#d62728', 'PF': '#9467bd', 'CI': '#ff7f0e', 'BICI': '#2ca02c', 'SABRE': '#e377c2'}
markers_d = {'EKF': 'o', 'AKF': '^', 'PF': 'v', 'CI': 's', 'BICI': 'D', 'SABRE': 'P'}
sigma_labels = [f'{s:.0f}' if s >= 1 else f'{s}' for s in sigmas]

def draw_hota(ax, matrix, title):
    im = ax.imshow(matrix, cmap='Blues', aspect='auto', vmin=0.3, vmax=0.99)
    ax.set_xticks(range(len(fnames))); ax.set_xticklabels(fnames, fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(sigmas))); ax.set_yticklabels(sigma_labels, fontsize=10)
    ax.set_ylabel('σ_a (m/s²)', fontsize=11); ax.set_title(title, fontsize=11, fontweight='bold')
    for i in range(len(sigmas)):
        for j in range(len(fnames)):
            val = matrix[i, j]; color = 'white' if val > 0.65 else 'black'
            weight = 'bold' if abs(val - matrix[:, j].max()) < 0.001 else 'normal'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=9, color=color, fontweight=weight)
    return im

def draw_pct(ax, matrix, title, vmin=-30, vmax=60):
    im = ax.imshow(matrix, cmap=delta_cmap, aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(fnames))); ax.set_xticklabels(fnames, fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(sigmas))); ax.set_yticklabels(sigma_labels, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    for i in range(len(sigmas)):
        for j in range(len(fnames)):
            val = matrix[i, j]; bg_norm = (val - vmin) / (vmax - vmin)
            color = 'white' if bg_norm > 0.7 else 'black'
            ax.text(j, i, f'{val:+.1f}%', ha='center', va='center', fontsize=9, color=color)
    return im

def draw_lines(ax, matrix, title, ylim_top):
    for fname in fnames:
        ys = [matrix[i, fnames.index(fname)] for i in range(len(sigmas))]
        ax.plot(sigmas, ys, f'-{markers_d[fname]}', color=colors[fname], label=fname,
                linewidth=2, markersize=6, markeredgecolor='white', markeredgewidth=0.5)
        peak_idx = int(np.argmax(ys))
        ax.plot(sigmas[peak_idx], ys[peak_idx], markers_d[fname], color=colors[fname],
                markersize=11, markeredgecolor='black', markeredgewidth=1.5, zorder=5)
    ax.set_xlabel('σ_a (m/s²)', fontsize=11); ax.set_ylabel('GPEM HOTA', fontsize=11)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right'); ax.grid(True, alpha=0.3); ax.set_ylim(0.15, ylim_top)

# Figure
fig, axes = plt.subplots(2, 3, figsize=(18, 10), gridspec_kw={'hspace': 0.45, 'wspace': 0.35})

# Top row: Average
im1 = draw_hota(axes[0, 0], avg_gpem, '(a) Average GPEM HOTA\n(mean of linear, quadratic, polar, static)')
fig.colorbar(im1, ax=axes[0, 0], shrink=0.7, pad=0.02, label='HOTA')

im2 = draw_pct(axes[0, 1], avg_pct, '(b) Average GPEM % Improvement\nover Baseline (all modes averaged)')
fig.colorbar(im2, ax=axes[0, 1], shrink=0.7, pad=0.02, label='% Improvement')

draw_lines(axes[0, 2], avg_gpem, '(c) Optimal σ_a per Filter\n(average mode)', 0.95)

# Bottom row: Best
im4 = draw_hota(axes[1, 0], best_gpem, '(d) Best GPEM Mode HOTA\n(max of linear, quadratic, polar, static)')
fig.colorbar(im4, ax=axes[1, 0], shrink=0.7, pad=0.02, label='HOTA')

im5 = draw_pct(axes[1, 1], best_pct, '(e) Best GPEM Mode % Improvement\nover Baseline (best mode per config)')
fig.colorbar(im5, ax=axes[1, 1], shrink=0.7, pad=0.02, label='% Improvement')

draw_lines(axes[1, 2], best_gpem, '(f) Optimal σ_a per Filter\n(best mode)', 1.02)

fig.suptitle('Process Noise Tuning (σ_a Sweep) — Average vs Best GPEM Mode\n'
             'Equal weight: 3 maps × 8 distribution steps × 13 penetration levels',
             fontsize=13, fontweight='bold', y=1.01)

plt.savefig('results/sigma_sweep_heatmap.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results/sigma_sweep_heatmap.png', dpi=150, bbox_inches='tight')
plt.savefig('docs/sigma_sweep_heatmap.pdf', dpi=300, bbox_inches='tight')
plt.savefig('docs/sigma_sweep_heatmap.png', dpi=150, bbox_inches='tight')
print('Saved to results/ and docs/')
