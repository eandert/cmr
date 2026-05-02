#!/usr/bin/env python3
"""
Generate heatmaps for the SABRE + AKF adaptation parameter sweep.
Sweeps: adapt_rate (α) × window_size (w)
Metric: Average GPEM HOTA across 3 maps × (8 dist + 13 pen) configs
"""
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

TASK_OUTPUT = '/tmp/claude-1000/-home-rave-test-cmr/050b252f-b3e4-47f1-8d41-744ac5c6bcf7/tasks/bao5n0txr.output'

# ── Parse results ───────────────────────────────────────────────
output = open(TASK_OUTPUT).read()
blocks = re.split(r'(\[\d+/144\][^\n]+)', output)

results = {}  # (alpha, window) -> {metric: [values]}

for i in range(1, len(blocks), 2):
    header = blocks[i]
    body = blocks[i+1] if i+1 < len(blocks) else ''

    m = re.search(r'α=([0-9.]+)\s+w=(\d+)\s+(fast_\w+)\s+(\w+)', header)
    if not m:
        continue
    alpha, window = float(m.group(1)), int(m.group(2))
    key = (alpha, window)
    if key not in results:
        results[key] = {
            'akf_baseline': [], 'akf_gpem': [],
            'sabre_baseline': [], 'sabre_gpem': [],
        }

    for line in body.split('\n'):
        hota_m = re.search(r'HOTA = ([0-9.]+)', line)
        if not hota_m:
            continue
        hota = float(hota_m.group(1))

        if 'akf_baseline' in line:
            results[key]['akf_baseline'].append(hota)
        elif 'akf_gpem_linear' in line:
            results[key]['akf_gpem'].append(hota)
        elif 'sabre_baseline' in line:
            results[key]['sabre_baseline'].append(hota)
        elif 'sabre_gpem_linear' in line:
            results[key]['sabre_gpem'].append(hota)

# ── Build grids ─────────────────────────────────────────────────
alphas = sorted(set(a for a, w in results.keys()))
windows = sorted(set(w for a, w in results.keys()))

def make_grid(metric_key):
    grid = np.zeros((len(alphas), len(windows)))
    for i, a in enumerate(alphas):
        for j, w in enumerate(windows):
            vals = results.get((a, w), {}).get(metric_key, [])
            grid[i, j] = np.mean(vals) if vals else np.nan
    return grid

akf_gpem_grid = make_grid('akf_gpem')
akf_base_grid = make_grid('akf_baseline')
akf_delta_grid = akf_gpem_grid - akf_base_grid
sabre_gpem_grid = make_grid('sabre_gpem')
sabre_base_grid = make_grid('sabre_baseline')
sabre_delta_grid = sabre_gpem_grid - sabre_base_grid

# ── Find best configs ───────────────────────────────────────────
akf_best_idx = np.unravel_index(np.nanargmax(akf_gpem_grid), akf_gpem_grid.shape)
sabre_best_idx = np.unravel_index(np.nanargmax(sabre_gpem_grid), sabre_gpem_grid.shape)

akf_best_a, akf_best_w = alphas[akf_best_idx[0]], windows[akf_best_idx[1]]
sabre_best_a, sabre_best_w = alphas[sabre_best_idx[0]], windows[sabre_best_idx[1]]

print(f'AKF best:   α={akf_best_a}, w={akf_best_w}, GPEM HOTA={akf_gpem_grid[akf_best_idx]:.4f}')
print(f'SABRE best: α={sabre_best_a}, w={sabre_best_w}, GPEM HOTA={sabre_gpem_grid[sabre_best_idx]:.4f}')

# ── Plot ────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

alpha_labels = [f'{a:.2f}' for a in alphas]
window_labels = [str(w) for w in windows]

def plot_heatmap(ax, grid, title, best_idx, cmap='Blues', fmt='.3f', vmin=None, vmax=None):
    if vmin is None:
        vmin = np.nanmin(grid)
    if vmax is None:
        vmax = np.nanmax(grid)
    im = ax.imshow(grid, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels(window_labels)
    ax.set_yticks(range(len(alphas)))
    ax.set_yticklabels(alpha_labels)
    ax.set_xlabel('Window Size', fontsize=11)
    ax.set_ylabel('Adapt Rate (α)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')

    # Annotate cells
    for i in range(len(alphas)):
        for j in range(len(windows)):
            val = grid[i, j]
            if np.isnan(val):
                continue
            color = 'white' if (val - vmin) / max(vmax - vmin, 1e-9) > 0.6 else 'black'
            weight = 'bold' if (i, j) == best_idx else 'normal'
            ax.text(j, i, f'{val:{fmt}}', ha='center', va='center',
                    fontsize=9, color=color, fontweight=weight)

    # Mark best cell with a border instead of a star
    if best_idx is not None:
        from matplotlib.patches import Rectangle
        rect = Rectangle((best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1,
                          linewidth=3, edgecolor='red', facecolor='none', zorder=5)
        ax.add_patch(rect)

    plt.colorbar(im, ax=ax, shrink=0.8)

# Row 1: AKF
plot_heatmap(axes[0, 0], akf_base_grid, 'AKF Baseline HOTA', akf_best_idx, cmap='Oranges')
plot_heatmap(axes[0, 1], akf_gpem_grid, 'AKF GPEM HOTA', akf_best_idx, cmap='Blues')
plot_heatmap(axes[0, 2], akf_delta_grid, 'AKF GPEM Improvement (Δ)', akf_best_idx, cmap='RdYlGn',
             vmin=-0.01, vmax=max(0.05, np.nanmax(akf_delta_grid)), fmt='+.3f')

# Row 2: SABRE
plot_heatmap(axes[1, 0], sabre_base_grid, 'SABRE Baseline HOTA', sabre_best_idx, cmap='Oranges')
plot_heatmap(axes[1, 1], sabre_gpem_grid, 'SABRE GPEM HOTA', sabre_best_idx, cmap='Blues')
plot_heatmap(axes[1, 2], sabre_delta_grid, 'SABRE GPEM Improvement (Δ)', sabre_best_idx, cmap='RdYlGn',
             vmin=-0.01, vmax=max(0.05, np.nanmax(sabre_delta_grid)), fmt='+.3f')

fig.suptitle(f'Adaptation Parameter Sweep — σ_a and Baseline R Locked\n'
             f'Best AKF: α={akf_best_a}, w={akf_best_w} | '
             f'Best SABRE: α={sabre_best_a}, w={sabre_best_w}\n'
             f'Equal weight: 3 maps × 8 dist steps × 13 pen levels',
             fontsize=13, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('results/adaptation_sweep_heatmap.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results/adaptation_sweep_heatmap.png', dpi=150, bbox_inches='tight')
plt.savefig('docs/adaptation_sweep_heatmap.pdf', dpi=300, bbox_inches='tight')
plt.savefig('docs/adaptation_sweep_heatmap.png', dpi=150, bbox_inches='tight')

print('\nSaved to results/ and docs/')

# ── Full table ──────────────────────────────────────────────────
print(f'\n{"":>5} {"":>3} | {"AKF base":>9} {"AKF gpem":>9} {"AKF Δ":>9} | {"SABRE base":>11} {"SABRE gpem":>11} {"SABRE Δ":>9}')
print('-' * 80)
for i, a in enumerate(alphas):
    for j, w in enumerate(windows):
        ab = akf_base_grid[i, j]
        ag = akf_gpem_grid[i, j]
        sb = sabre_base_grid[i, j]
        sg = sabre_gpem_grid[i, j]
        marker = ' ***' if (i, j) == akf_best_idx or (i, j) == sabre_best_idx else ''
        print(f'{a:>5.2f} {w:>3} | {ab:>9.4f} {ag:>9.4f} {ag-ab:>+9.4f} | {sb:>11.4f} {sg:>11.4f} {sg-sb:>+9.4f}{marker}')
