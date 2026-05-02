#!/usr/bin/env python3
"""
Generate baseline R sweep plot from permanently saved sweep data.
Reads from results/baseline_sweep_data/ which contains summary.json
files named by R value, map, and suite type.
"""
import json, os, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = 'results/baseline_sweep_data'

# ── Scan saved data ─────────────────────────────────────────────
fnames = ['EKF', 'AKF', 'PF', 'CI', 'BICI', 'SABRE']
filter_baselines_keys = {
    'EKF': 'baseline', 'AKF': 'akf_baseline', 'PF': 'pf_baseline',
    'CI': 'ci_baseline', 'BICI': 'bici_baseline', 'SABRE': 'sabre_baseline',
}
filter_modes = {
    'EKF': ['gpem_linear', 'gpem_quadratic', 'gpem_polar', 'static'],
    'AKF': ['akf_gpem_linear', 'akf_gpem_quadratic', 'akf_gpem_polar', 'akf_static'],
    'PF':  ['pf_gpem_linear', 'pf_gpem_quadratic', 'pf_gpem_polar', 'pf_static'],
    'CI':  ['ci_gpem_linear', 'ci_gpem_quadratic', 'ci_gpem_polar', 'ci_static'],
    'BICI':['bici_gpem_linear', 'bici_gpem_quadratic', 'bici_gpem_polar', 'bici_static'],
    'SABRE':['sabre_gpem_linear', 'sabre_gpem_quadratic', 'sabre_gpem_polar', 'sabre_static'],
}

# Parse directory names: R{value}_{map}_{suite}
baselines_set = set()
base_h = {}   # R -> filter -> [hota]
gpem_h = {}   # R -> filter -> [hota]
gpem_by_mode = {}  # R -> filter -> mode -> [hota]

for entry in sorted(os.listdir(DATA_DIR)):
    summary_path = os.path.join(DATA_DIR, entry, 'summary.json')
    if not os.path.exists(summary_path):
        continue

    m = re.match(r'R([0-9.]+)_(fast_\w+)_(dist|pen)', entry)
    if not m:
        continue

    baseline_r = float(m.group(1))
    baselines_set.add(baseline_r)

    if baseline_r not in base_h:
        base_h[baseline_r] = {f: [] for f in fnames}
        gpem_h[baseline_r] = {f: [] for f in fnames}
        gpem_by_mode[baseline_r] = {f: {m: [] for m in filter_modes[f]} for f in fnames}

    r = json.load(open(summary_path)).get('results', {})

    for fname in fnames:
        # Baseline HOTA
        bkey = filter_baselines_keys[fname]
        for ck, cv in r.items():
            if ck.endswith(f'_{bkey}') or ck.startswith(f'{bkey}_av_'):
                h = cv.get('avg_hota_mean', 0)
                if h > 0:
                    base_h[baseline_r][fname].append(h)

        # GPEM HOTA per mode
        for mode in filter_modes[fname]:
            for ck, cv in r.items():
                if ck.endswith(f'_{mode}') or ck.startswith(f'{mode}_av_'):
                    h = cv.get('avg_hota_mean', 0)
                    if h > 0:
                        gpem_h[baseline_r][fname].append(h)
                        gpem_by_mode[baseline_r][fname][mode].append(h)

baselines_vals = sorted(baselines_set)
print(f'Baseline R values: {baselines_vals}')
for br in baselines_vals:
    n = len(base_h[br]['EKF'])
    print(f'  R={br:.2f}: {n} data points')

# ── Plot ────────────────────────────────────────────────────────
colors = {'EKF': '#1f77b4', 'AKF': '#d62728', 'PF': '#9467bd',
          'CI': '#ff7f0e', 'BICI': '#2ca02c', 'SABRE': '#e377c2'}
markers = {'EKF': 'o', 'AKF': '^', 'PF': 'v', 'CI': 's', 'BICI': 'D', 'SABRE': 'P'}

fig, axes = plt.subplots(1, 4, figsize=(24, 6))

# (a) Baseline HOTA vs R
ax = axes[0]
for fname in fnames:
    xs, ys = [], []
    for br in baselines_vals:
        if base_h[br][fname]:
            xs.append(br)
            ys.append(np.mean(base_h[br][fname]))
    if xs:
        ax.plot(xs, ys, f'-{markers[fname]}', color=colors[fname], label=fname,
                linewidth=2, markersize=7, markeredgecolor='white', markeredgewidth=0.5)
        peak_idx = int(np.argmax(ys))
        ax.plot(xs[peak_idx], ys[peak_idx], markers[fname], color=colors[fname],
                markersize=12, markeredgecolor='black', markeredgewidth=1.5, zorder=5)
ax.set_xlabel('Baseline R', fontsize=12)
ax.set_ylabel('Average HOTA', fontsize=12)
ax.set_title('(a) Baseline HOTA vs Baseline R', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.35, 0.9)

# (b) GPEM HOTA vs R (sanity — should be flat)
ax = axes[1]
for fname in fnames:
    xs, ys = [], []
    for br in baselines_vals:
        if gpem_h[br][fname]:
            xs.append(br)
            ys.append(np.mean(gpem_h[br][fname]))
    if xs:
        ax.plot(xs, ys, f'-{markers[fname]}', color=colors[fname], label=fname,
                linewidth=2, markersize=7, markeredgecolor='white', markeredgewidth=0.5)
ax.set_xlabel('Baseline R', fontsize=12)
ax.set_ylabel('Average HOTA', fontsize=12)
ax.set_title('(b) GPEM HOTA vs Baseline R\n(sanity check — should be flat)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.35, 0.9)

# (c) Average GPEM improvement over baseline (mean across all modes)
ax = axes[2]
for fname in fnames:
    xs, ys = [], []
    for br in baselines_vals:
        if base_h[br][fname] and gpem_h[br][fname]:
            xs.append(br)
            ys.append(np.mean(gpem_h[br][fname]) - np.mean(base_h[br][fname]))
    if xs:
        ax.plot(xs, ys, f'-{markers[fname]}', color=colors[fname], label=fname,
                linewidth=2, markersize=7, markeredgecolor='white', markeredgewidth=0.5)
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
ax.set_xlabel('Baseline R', fontsize=12)
ax.set_ylabel('\u0394HOTA (GPEM \u2212 Baseline)', fontsize=12)
ax.set_title('(c) Avg GPEM Improvement over Baseline\n(mean across all GPEM modes)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (d) Best-mode GPEM improvement over baseline
ax = axes[3]
for fname in fnames:
    xs, ys = [], []
    for br in baselines_vals:
        if not base_h[br][fname]:
            continue
        bv = np.mean(base_h[br][fname])
        best_mode_hota = None
        for mode in filter_modes[fname]:
            vals = gpem_by_mode[br][fname][mode]
            if vals:
                mode_avg = np.mean(vals)
                if best_mode_hota is None or mode_avg > best_mode_hota:
                    best_mode_hota = mode_avg
        if best_mode_hota is not None:
            xs.append(br)
            ys.append(best_mode_hota - bv)
    if xs:
        ax.plot(xs, ys, f'-{markers[fname]}', color=colors[fname], label=fname,
                linewidth=2, markersize=7, markeredgecolor='white', markeredgewidth=0.5)
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
ax.set_xlabel('Baseline R', fontsize=12)
ax.set_ylabel('\u0394HOTA (GPEM \u2212 Baseline)', fontsize=12)
ax.set_title('(d) Best-Mode GPEM Improvement\n(best single GPEM mode per filter)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Add fleet-weighted average and chosen R annotations to panels (a), (c), (d)
R_CHOSEN = 0.2
R_FLEET_AVG = 0.2075  # computed: perc(0.0755) + loc(0.1320) = 0.2075
for ax in [axes[0], axes[2], axes[3]]:
    ylo, yhi = ax.get_ylim()
    ax.axvline(x=R_FLEET_AVG, color='gray', linestyle=':', linewidth=1.5, alpha=0.8, zorder=1)
    ax.axvline(x=R_CHOSEN, color='black', linestyle='-', linewidth=1.5, alpha=0.7, zorder=1)
    ax.annotate(f'chosen R={R_CHOSEN}', xy=(R_CHOSEN, yhi),
                xytext=(R_CHOSEN + 0.04, yhi - (yhi - ylo) * 0.05),
                fontsize=8, fontweight='bold', ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='black'))
    ax.annotate(f'fleet avg\nR={R_FLEET_AVG:.2f}', xy=(R_FLEET_AVG, yhi),
                xytext=(R_FLEET_AVG + 0.04, yhi - (yhi - ylo) * 0.15),
                fontsize=8, ha='left', va='top', color='gray',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray'))

fig.suptitle('Baseline R Sweep (0.01\u20131.0) \u2014 Per-Filter \u03c3_a Locked at Optimal\n'
             'Equal weight: 3 maps \u00d7 8 distribution steps \u00d7 13 penetration levels',
             fontsize=13, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('results/baseline_sweep.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results/baseline_sweep.png', dpi=150, bbox_inches='tight')
plt.savefig('docs/baseline_sweep.pdf', dpi=300, bbox_inches='tight')
plt.savefig('docs/baseline_sweep.png', dpi=150, bbox_inches='tight')

# ── Summary table ───────────────────────────────────────────────
print(f'\n{"R":>5} | {"EKF":>7} | {"AKF":>7} | {"PF":>7} | {"CI":>7} | {"BICI":>7} | {"SABRE":>7}')
print(f'{"":>5} | {"base":>7} | {"base":>7} | {"base":>7} | {"base":>7} | {"base":>7} | {"base":>7}')
print('-' * 62)
for br in baselines_vals:
    row = f'{br:>5.2f} |'
    for fname in fnames:
        bv = np.mean(base_h[br][fname]) if base_h[br][fname] else 0
        row += f' {bv:>.3f} |'
    print(row)

print('\n=== PEAK BASELINE HOTA PER FILTER ===')
for fname in fnames:
    best_r, best_h = 0, 0
    for br in baselines_vals:
        bv = np.mean(base_h[br][fname]) if base_h[br][fname] else 0
        if bv > best_h:
            best_h = bv
            best_r = br
    print(f'  {fname:>5}: R={best_r:.2f}  HOTA={best_h:.4f}')

print('\nSaved to results/ and docs/')
