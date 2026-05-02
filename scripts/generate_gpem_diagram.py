#!/usr/bin/env python3
"""
Generate two publication-quality diagrams comparing covariance modeling:
  Figure 1: Traditional (static/baseline) — identical R, colored by observer
  Figure 2: GPEM (heteroscedastic) — unique R per detection

Both share the same scene, target positions, AND error-injected detection
positions (crosshairs offset from true position by sampled errors).

Output:
    docs/figures/covariance_baseline.pdf / .png
    docs/figures/covariance_gpem.pdf / .png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Sensor MSE data ──────────────────────────────────────────────────

CENTERPOINT_MSE = {
    10: (0.016, 0.009), 30: (0.040, 0.026),
    50: (0.076, 0.052), 70: (0.124, 0.088),
}
DETR3D_MSE = {
    10: (0.132, 0.034), 30: (0.190, 0.054),
    50: (0.258, 0.079), 70: (0.337, 0.110),
}
BEVFUSION_MSE = {
    10: (0.018, 0.008), 30: (0.042, 0.026),
    50: (0.075, 0.053), 70: (0.118, 0.090),
}

KISS_ICP_MSE = (0.006, 0.007)
NOISY_GPS_MSE = (0.095, 0.095)
CIS_LOC_MSE = (0.0, 0.0)

BASELINE_AVG_MSE = (0.065, 0.035)

COL_A = '#2166AC'
COL_B = '#B2182B'
COL_CIS = '#1B7837'

SCALE = 12
RNG = np.random.RandomState(42)  # Fixed seed for reproducible error samples

# ── Scene layout ─────────────────────────────────────────────────────

CAV_A_POS = (-62, 1.5)
CAV_B_POS = (1.5, -62)
CIS_POS = (0, 0)

TARGETS = [
    {'pos': (-30, -2), 'heading': 0, 'label': 'T1'},     # eastbound lane
    {'pos': (2, -30), 'heading': 90, 'label': 'T2'},      # northbound lane
    {'pos': (28, 2), 'heading': 180, 'label': 'T3'},      # westbound lane
    {'pos': (-2, 18), 'heading': 270, 'label': 'T4'},     # southbound lane
]

OBSERVERS = [
    {'name': 'CAV A', 'pos': CAV_A_POS, 'heading': 0,
     'det': CENTERPOINT_MSE, 'loc': KISS_ICP_MSE, 'color': COL_A},
    {'name': 'CAV B', 'pos': CAV_B_POS, 'heading': np.pi / 2,
     'det': DETR3D_MSE, 'loc': NOISY_GPS_MSE, 'color': COL_B},
    {'name': 'CIS', 'pos': CIS_POS, 'heading': 0,
     'det': BEVFUSION_MSE, 'loc': CIS_LOC_MSE, 'color': COL_CIS},
]


# ── Geometry helpers ──────────────────────────────────────────────────

def make_rotated_cov(var_r, var_l, angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return R @ np.diag([var_r, var_l]) @ R.T


def draw_ellipse(ax, center, cov, n_std=2, **kw):
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-8)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * n_std * np.sqrt(vals)
    if w < 0.15 and h < 0.15:
        return
    ax.add_patch(patches.Ellipse(center, w, h, angle=angle, **kw))


def draw_crosshair(ax, center, size=1.2, color='black', lw=0.8, zorder=9):
    """Draw a small + crosshair marking the detected position."""
    x, y = center
    ax.plot([x - size, x + size], [y, y], '-', color=color, lw=lw, zorder=zorder)
    ax.plot([x, x], [y - size, y + size], '-', color=color, lw=lw, zorder=zorder)


def draw_vehicle(ax, center, heading_deg, length=4.5, width=2.0, **kw):
    rect = patches.FancyBboxPatch(
        (-length / 2, -width / 2), length, width,
        boxstyle="round,pad=0.08", **kw)
    t = (plt.matplotlib.transforms.Affine2D()
         .rotate_deg(heading_deg)
         .translate(center[0], center[1]) + ax.transData)
    rect.set_transform(t)
    ax.add_patch(rect)


def draw_road(ax, start, end, width=7.0):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = np.hypot(dx, dy)
    angle = np.degrees(np.arctan2(dy, dx))
    rect = patches.Rectangle(
        (0, -width / 2), length, width,
        facecolor='#ededed', edgecolor='#d5d5d5', linewidth=0.4, zorder=0)
    t = (plt.matplotlib.transforms.Affine2D()
         .rotate_deg(angle).translate(start[0], start[1]) + ax.transData)
    rect.set_transform(t)
    ax.add_patch(rect)
    n = int(length / 8)
    for i in range(n):
        frac = (i + 0.5) / n
        cx = start[0] + frac * dx
        cy = start[1] + frac * dy
        d = patches.Rectangle((-1.8, -0.1), 3.6, 0.2,
                               facecolor='white', edgecolor='none', zorder=1)
        t2 = (plt.matplotlib.transforms.Affine2D()
              .rotate_deg(angle).translate(cx, cy) + ax.transData)
        d.set_transform(t2)
        ax.add_patch(d)


def interp_mse(table, dist):
    dists = sorted(table.keys())
    if dist <= dists[0]:
        return table[dists[0]]
    if dist >= dists[-1]:
        return table[dists[-1]]
    for i in range(len(dists) - 1):
        if dists[i] <= dist <= dists[i + 1]:
            f = (dist - dists[i]) / (dists[i + 1] - dists[i])
            r0, l0 = table[dists[i]]
            r1, l1 = table[dists[i + 1]]
            return (r0 + f * (r1 - r0), l0 + f * (l1 - l0))


def sample_error_offset(obs, target_pos):
    """Sample a realistic error offset in world frame for this observer→target."""
    dx = target_pos[0] - obs['pos'][0]
    dy = target_pos[1] - obs['pos'][1]
    dist = np.hypot(dx, dy)
    angle = np.arctan2(dy, dx)

    # Detection error (radial + lateral in sensor frame)
    r_mse, l_mse = interp_mse(obs['det'], dist)
    err_r = RNG.normal(0, np.sqrt(r_mse))
    err_l = RNG.normal(0, np.sqrt(l_mse))

    # Localization error (longitudinal + lateral in ego frame)
    err_long = RNG.normal(0, np.sqrt(obs['loc'][0])) if obs['loc'][0] > 0 else 0
    err_lat = RNG.normal(0, np.sqrt(obs['loc'][1])) if obs['loc'][1] > 0 else 0

    # Rotate detection error to world frame
    c, s = np.cos(angle), np.sin(angle)
    det_x = c * err_r - s * err_l
    det_y = s * err_r + c * err_l

    # Rotate localization error to world frame (by ego heading)
    c2, s2 = np.cos(obs['heading']), np.sin(obs['heading'])
    loc_x = c2 * err_long - s2 * err_lat
    loc_y = s2 * err_long + c2 * err_lat

    return (det_x + loc_x, det_y + loc_y)


def draw_scene(ax):
    ax.set_aspect('equal')
    ax.set_xlim(-78, 78)
    ax.set_ylim(-72, 38)
    ax.axis('off')

    draw_road(ax, (-80, 0), (80, 0))
    draw_road(ax, (0, -80), (0, 40))

    ax.plot(0, 0, 's', color=COL_CIS, markersize=6, zorder=10,
            markeredgecolor='white', markeredgewidth=0.8)
    ax.text(2.5, 3.5, 'CIS', fontsize=6, fontweight='bold', color=COL_CIS, zorder=10)

    draw_vehicle(ax, CAV_A_POS, 0, length=4.0, width=1.8,
                 facecolor=COL_A, edgecolor='white', linewidth=1.2, zorder=8)
    ax.text(CAV_A_POS[0], CAV_A_POS[1] + 5.5, 'CAV A',
            fontsize=5.5, ha='center', color=COL_A, fontweight='bold', zorder=10)

    draw_vehicle(ax, CAV_B_POS, 90, length=4.0, width=1.8,
                 facecolor=COL_B, edgecolor='white', linewidth=1.2, zorder=8)
    ax.text(CAV_B_POS[0] + 6, CAV_B_POS[1], 'CAV B',
            fontsize=5.5, ha='left', color=COL_B, fontweight='bold', zorder=10)

    for t in TARGETS:
        draw_vehicle(ax, t['pos'], t['heading'], length=3.5, width=1.6,
                     facecolor='#888888', edgecolor='#555555', linewidth=0.6,
                     alpha=0.5, zorder=3)
        ax.text(t['pos'][0] + 2.5, t['pos'][1] + 2.5, t['label'],
                fontsize=5.5, color='#555555', fontweight='bold', zorder=10)

    # Scale bar
    bx, by = -72, -67
    ax.plot([bx, bx + 15], [by, by], 'k-', linewidth=1.5, zorder=10)
    ax.plot([bx, bx], [by - 1, by + 1], 'k-', linewidth=1, zorder=10)
    ax.plot([bx + 15, bx + 15], [by - 1, by + 1], 'k-', linewidth=1, zorder=10)
    ax.text(bx + 7.5, by - 3.5, '15 m', fontsize=5.5, ha='center')


# ── Pre-compute shared error offsets ─────────────────────────────────

DETECTION_OFFSETS = {}
for obs in OBSERVERS:
    for t in TARGETS:
        key = (obs['name'], t['label'])
        offset = sample_error_offset(obs, t['pos'])
        # Scale for visibility (same SCALE as ellipses)
        DETECTION_OFFSETS[key] = (offset[0] * SCALE * 0.7, offset[1] * SCALE * 0.7)


# ── Figure 1: Baseline ───────────────────────────────────────────────

def generate_baseline():
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 3.2))
    fig.patch.set_facecolor('white')
    draw_scene(ax)

    for obs in OBSERVERS:
        for t in TARGETS:
            key = (obs['name'], t['label'])
            off = DETECTION_OFFSETS[key]
            det_pos = (t['pos'][0] + off[0], t['pos'][1] + off[1])

            dx = t['pos'][0] - obs['pos'][0]
            dy = t['pos'][1] - obs['pos'][1]
            angle = np.arctan2(dy, dx)

            # Same fixed R for all, colored by observer
            cov = make_rotated_cov(
                BASELINE_AVG_MSE[0] * SCALE**2,
                BASELINE_AVG_MSE[1] * SCALE**2,
                angle)

            draw_ellipse(ax, det_pos, cov, n_std=2,
                         fill=False, linewidth=0.8, linestyle='-',
                         edgecolor=obs['color'], alpha=0.5, zorder=7)

            draw_crosshair(ax, det_pos, size=1.0, color=obs['color'],
                           lw=0.7, zorder=9)

    ax.text(55, -55,
            r'Traditional: $\mathbf{R} = \sigma^2_{avg} \cdot \mathbf{I}$'
            '\n'
            'Same covariance for every\n'
            'detection regardless of\n'
            'sensor type or distance',
            fontsize=6, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f5f5',
                      edgecolor='#555555', linewidth=0.8),
            zorder=10, linespacing=1.4)

    # Left legend: observer colors
    obs_handles = [
        Line2D([0], [0], color=COL_A, linewidth=1.5, label='CAV A (CenterPoint + ICP-to-HD Map (clean seed))'),
        Line2D([0], [0], color=COL_B, linewidth=1.5, label='CAV B (DETR3D + ICP-to-HD Map (noisy seed))'),
        Line2D([0], [0], color=COL_CIS, linewidth=1.5, label='CIS (BEV Fusion, fixed)'),
    ]
    leg1 = ax.legend(handles=obs_handles, loc='upper left', fontsize=5.5,
              framealpha=0.95, edgecolor='#cccccc', borderpad=0.5,
              handlelength=1.5)
    leg1.set_zorder(20)
    ax.add_artist(leg1)

    # Right legend: covariance + detected position
    style_handles = [
        Line2D([0], [0], color='gray', lw=1.3, ls='-', label=r'Fixed $\mathbf{R}$ covariance'),
        Line2D([0], [0], color='gray', marker='+', markersize=6, lw=0,
               label='Detected position'),
    ]
    leg2 = ax.legend(handles=style_handles, loc='upper right', fontsize=5.5,
              framealpha=0.95, edgecolor='#cccccc', borderpad=0.5,
              handlelength=1.5)
    leg2.set_zorder(20)

    fig.tight_layout(pad=0.3)
    for ext in ['pdf', 'png']:
        p = OUTPUT_DIR / f"covariance_baseline.{ext}"
        fig.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {p}")
    plt.close(fig)


# ── Figure 2: GPEM ───────────────────────────────────────────────────

def generate_gpem():
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 3.2))
    fig.patch.set_facecolor('white')
    draw_scene(ax)

    for obs in OBSERVERS:
        for t in TARGETS:
            key = (obs['name'], t['label'])
            off = DETECTION_OFFSETS[key]
            det_pos = (t['pos'][0] + off[0], t['pos'][1] + off[1])
            col = obs['color']

            dx = t['pos'][0] - obs['pos'][0]
            dy = t['pos'][1] - obs['pos'][1]
            dist = np.hypot(dx, dy)
            angle = np.arctan2(dy, dx)

            # Perception covariance
            r_mse, l_mse = interp_mse(obs['det'], dist)
            perc_cov = make_rotated_cov(r_mse * SCALE**2, l_mse * SCALE**2, angle)

            # Localization covariance
            if obs['loc'][0] > 0 or obs['loc'][1] > 0:
                loc_cov = make_rotated_cov(
                    obs['loc'][0] * SCALE**2,
                    obs['loc'][1] * SCALE**2,
                    obs['heading'])
            else:
                loc_cov = np.zeros((2, 2))

            total_cov = perc_cov + loc_cov

            # Perception: dashed
            draw_ellipse(ax, det_pos, perc_cov, n_std=2,
                         fill=False, linewidth=0.8, linestyle='--',
                         edgecolor=col, alpha=0.5, zorder=6)

            # Localization: dotted
            if obs['loc'][0] > 0.001 or obs['loc'][1] > 0.001:
                draw_ellipse(ax, det_pos, loc_cov, n_std=2,
                             fill=False, linewidth=0.8, linestyle=':',
                             edgecolor=col, alpha=0.35, zorder=5)

            # Combined: solid, matching dashed/dotted line style
            draw_ellipse(ax, det_pos, total_cov, n_std=2,
                         fill=False, linewidth=0.8, linestyle='-',
                         edgecolor=col, alpha=0.5, zorder=7)

            # Crosshair at detected position
            draw_crosshair(ax, det_pos, size=1.0, color=col, lw=0.7, zorder=9)

    # Annotations
    ax.text(-45, 12,
            r'CAV A$\rightarrow$T1: $R$=0.036 m$^2$',
            fontsize=5.5, color=COL_A, ha='center',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor=COL_A, alpha=0.8, linewidth=0.6), zorder=10)

    ax.text(13, -47,
            r'CAV B$\rightarrow$T2: $R$=0.285 m$^2$',
            fontsize=5.5, color=COL_B, ha='center',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor=COL_B, alpha=0.8, linewidth=0.6), zorder=10)

    ax.text(42, 12,
            r'CIS$\rightarrow$T3: $R$=0.042 m$^2$',
            fontsize=5.5, color=COL_CIS, ha='center',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor=COL_CIS, alpha=0.8, linewidth=0.6), zorder=10)

    ax.text(55, -58,
            r'$\mathbf{R}(d,v) = \Sigma_{perc}(d) + \Sigma_{loc}(v)$'
            '\n'
            r'MSE$(d) = \mathrm{bias}(d)^2 + \mathrm{var}(d)$',
            fontsize=6, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f5f5',
                      edgecolor='#333333', linewidth=0.8),
            zorder=10, linespacing=1.5)

    # Left legend: observer colors
    obs_handles = [
        Line2D([0], [0], color=COL_A, linewidth=1.5, label='CAV A (CenterPoint + ICP-to-HD Map (clean seed))'),
        Line2D([0], [0], color=COL_B, linewidth=1.5, label='CAV B (DETR3D + ICP-to-HD Map (noisy seed))'),
        Line2D([0], [0], color=COL_CIS, linewidth=1.5, label='CIS (BEV Fusion, fixed)'),
    ]
    leg1 = ax.legend(handles=obs_handles, loc='upper left', fontsize=5,
              framealpha=0.95, edgecolor='#cccccc', borderpad=0.5,
              handlelength=1.5)
    leg1.set_zorder(20)
    ax.add_artist(leg1)  # keep first legend when adding second

    # Right legend: line styles
    style_handles = [
        Line2D([0], [0], color='gray', lw=0.9, ls='--', label=r'$\Sigma_{perc}(d)$ perception'),
        Line2D([0], [0], color='gray', lw=0.9, ls=':', label=r'$\Sigma_{loc}(v)$ localization'),
        Line2D([0], [0], color='gray', lw=1.8, ls='-', label=r'$\mathbf{R}_{total}$ combined'),
        Line2D([0], [0], color='gray', marker='+', markersize=6, lw=0,
               label='Detected position'),
    ]
    leg2 = ax.legend(handles=style_handles, loc='upper right', fontsize=5,
              framealpha=0.95, edgecolor='#cccccc', borderpad=0.5,
              handlelength=1.5)
    leg2.set_zorder(20)

    fig.tight_layout(pad=0.3)
    for ext in ['pdf', 'png']:
        p = OUTPUT_DIR / f"covariance_gpem.{ext}"
        fig.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {p}")
    plt.close(fig)


if __name__ == "__main__":
    generate_baseline()
    generate_gpem()
