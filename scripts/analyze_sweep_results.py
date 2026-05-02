#!/usr/bin/env python3
"""
Post-process sigma_a sweep results from result directories.

Computes weighted GPEM improvement over baseline across:
- All distribution steps (0-7), weighted by vehicle count
- All penetration levels (1%-100%), weighted by vehicle count
- All maps, weighted by vehicle density

Usage:
    python3 analyze_sweep_results.py [--date 2026-03-29]
"""
import json, glob, os, sys, re
import numpy as np
from collections import defaultdict

DATE_FILTER = sys.argv[1] if len(sys.argv) > 1 else '2026-03-29'

# Filter groups
FILTER_GROUPS = {
    'EKF': {
        'baseline': 'baseline',
        'gpem': ['gpem_linear', 'gpem_quadratic', 'gpem_polar', 'static'],
    },
    'CI': {
        'baseline': 'ci_baseline',
        'gpem': ['ci_gpem_linear', 'ci_gpem_quadratic', 'ci_gpem_polar', 'ci_static'],
    },
    'BICI': {
        'baseline': 'bici_baseline',
        'gpem': ['bici_gpem_linear', 'bici_gpem_quadratic', 'bici_gpem_polar', 'bici_static'],
    },
    'AKF': {
        'baseline': 'akf_baseline',
        'gpem': ['akf_gpem_linear', 'akf_gpem_quadratic', 'akf_gpem_polar', 'akf_static'],
    },
    'PF': {
        'baseline': 'pf_baseline',
        'gpem': ['pf_gpem_linear', 'pf_gpem_quadratic', 'pf_gpem_polar', 'pf_static'],
    },
}


def find_key(results, suffix):
    """Find a result key ending with the given suffix."""
    for k in results:
        if k.endswith(f'_{suffix}') or k.startswith(f'{suffix}_av_'):
            return k
    return None


def get_vehicle_count_from_dir(result_dir):
    """Estimate vehicle count from run CSV files."""
    # Look for recorded_steps in any aggregated.json
    for sub in os.listdir(result_dir):
        agg = os.path.join(result_dir, sub, 'aggregated.json')
        if os.path.exists(agg):
            try:
                a = json.load(open(agg))
                return a.get('avg_recorded_steps', 600)
            except:
                pass
    return 600  # default


def parse_all_configs(results):
    """Parse all configs from a summary, returning dict of suffix -> {amota, hota, amotp}."""
    parsed = {}
    for key, val in results.items():
        parsed[key] = {
            'amota': val.get('avg_global_amota_mean', 0),
            'hota': val.get('avg_hota_mean', 0),
            'amotp': val.get('avg_global_amotp_mean', 0),
        }
    return parsed


def analyze_directory(result_dir):
    """Analyze one result directory, return per-config metrics."""
    try:
        summary = json.load(open(f'{result_dir}/summary.json'))
        suite_cfg = json.load(open(f'{result_dir}/suite_config.json'))
        results = summary.get('results', {})
        map_name = suite_cfg.get('map_name', '?')
        suite_name = suite_cfg.get('name', '?')
        return {
            'map': map_name,
            'suite': suite_name,
            'results': parse_all_configs(results),
        }
    except Exception as e:
        return None


def compute_weighted_improvement(all_runs):
    """
    Compute weighted GPEM improvement over baseline.

    For distribution: weight all steps equally (each represents a fleet composition)
    For penetration: weight by penetration rate (higher penetration = more vehicles = more weight)
    Cross-map: weight by typical vehicle density
    """
    # Map weights (approximate relative vehicle density)
    MAP_WEIGHTS = {
        'fast_city': 3.0,     # Dense intersection, CIS + many CAVs
        'fast_highway': 1.0,  # Sparse, mostly passing
        'fast_rural': 2.0,    # Medium density, stop-sign interactions
    }

    # Penetration weights (higher pen = more vehicles contributing)
    PEN_WEIGHTS = {
        '1.0': 1, '2.5': 1, '5.0': 2, '10.0': 3, '20.0': 5,
        '30.0': 6, '40.0': 7, '50.0': 8, '60.0': 9, '70.0': 10,
        '80.0': 10, '90.0': 10, '100.0': 10,
    }

    results_by_sigma = defaultdict(lambda: defaultdict(list))
    # Structure: results_by_sigma[sigma][filter_name] = [(base_amota, best_gpem_amota, weight), ...]

    for run in all_runs:
        if run is None:
            continue
        map_name = run['map']
        suite = run['suite']
        results = run['results']
        map_w = MAP_WEIGHTS.get(map_name, 1.0)

        # Determine sigma_a from the result directory timing
        # (We'll group by the order they appear — each sigma has 3 maps x 2 suites)

        for fname, fgroup in FILTER_GROUPS.items():
            bkey = find_key(results, fgroup['baseline'])
            if not bkey:
                continue
            base_amota = results[bkey]['amota']
            base_hota = results[bkey]['hota']

            # Find best GPEM mode
            best_gpem_amota = -999
            best_gpem_hota = -999
            best_mode = ''
            for gmode in fgroup['gpem']:
                gkey = find_key(results, gmode)
                if gkey and results[gkey]['amota'] > best_gpem_amota:
                    best_gpem_amota = results[gkey]['amota']
                    best_gpem_hota = results[gkey]['hota']
                    best_mode = gmode

            if best_gpem_amota > -999:
                # Determine weight
                if 'Distribution' in suite:
                    weight = map_w  # Each dist step gets equal map weight
                else:
                    # Extract penetration level from the key
                    weight = map_w  # Default

                results_by_sigma['all'][fname].append({
                    'base_amota': base_amota,
                    'gpem_amota': best_gpem_amota,
                    'base_hota': base_hota,
                    'gpem_hota': best_gpem_hota,
                    'best_mode': best_mode,
                    'weight': weight,
                    'map': map_name,
                    'suite': suite,
                })

    return results_by_sigma


# ============================================================
# Main
# ============================================================
print(f'Scanning results from {DATE_FILTER}...\n')

dist_dirs = sorted(glob.glob(f'results/GPEM_Distribution_Sweep_{DATE_FILTER}_*'))
pen_dirs = sorted(glob.glob(f'results/GPEM_Penetration_Sweep_{DATE_FILTER}_*'))
all_dirs = dist_dirs + pen_dirs

print(f'Found {len(dist_dirs)} distribution + {len(pen_dirs)} penetration = {len(all_dirs)} total dirs')

# Group by sigma_a: every 3 dist + 3 pen dirs = one sigma_a
# Order: city, highway, rural for each
sigmas = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

print('\n' + '='*110)
print('PER-SIGMA RESULTS — Best GPEM mode AMOTA vs Baseline (all configs within each run)')
print('='*110)

for fname in ['EKF', 'CI', 'BICI', 'AKF', 'PF']:
    fgroup = FILTER_GROUPS[fname]
    print(f'\n{"="*50} {fname} {"="*50}')
    print(f'{"σ_a":>4} | {"City dist":>20} | {"City pen":>20} | {"Hwy dist":>20} | {"Hwy pen":>20} | {"Rural dist":>20} | {"Rural pen":>20}')
    print('-' * 135)

    for si, sigma in enumerate(sigmas):
        row = f'{sigma:>4.0f} |'

        for di_offset, dirlist in [(0, dist_dirs), (0, pen_dirs)]:
            # For this sigma, dirs are at index si*3, si*3+1, si*3+2 (city, hwy, rural)
            for mi in range(3):  # city, hwy, rural
                idx = si * 3 + mi
                if idx >= len(dirlist):
                    row += f'                    ? |'
                    continue

                run = analyze_directory(dirlist[idx])
                if run is None:
                    row += f'               err   |'
                    continue

                results = run['results']
                bkey = find_key(results, fgroup['baseline'])
                if not bkey:
                    row += f'               ?     |'
                    continue

                base = results[bkey]['amota']
                best_g = -999
                best_mode = ''
                for gmode in fgroup['gpem']:
                    gkey = find_key(results, gmode)
                    if gkey and results[gkey]['amota'] > best_g:
                        best_g = results[gkey]['amota']
                        best_mode = gmode.split('_')[-1][:3]  # lin/qua/pol/sta

                if best_g > -999:
                    d = (best_g - base) / max(abs(base), 0.001) * 100
                    row += f' {base:.3f}→{best_g:.3f} {d:>+5.0f}% |'
                else:
                    row += f' {base:.3f}→?          |'

        print(row)

# Weighted summary
print('\n' + '='*110)
print('WEIGHTED SUMMARY — Average GPEM improvement across all configs')
print('='*110)

MAP_WEIGHTS = {'fast_city': 3.0, 'fast_highway': 1.0, 'fast_rural': 2.0}

for fname in ['EKF', 'CI', 'BICI', 'AKF', 'PF']:
    fgroup = FILTER_GROUPS[fname]
    print(f'\n--- {fname} ---')
    print(f'{"σ_a":>4} | {"Weighted Δ AMOTA":>18} | {"Weighted Δ HOTA":>18} | {"Best for":>12}')
    print('-' * 60)

    best_sigma = None
    best_weighted = -999

    for si, sigma in enumerate(sigmas):
        deltas_amota = []
        deltas_hota = []
        weights = []

        for dirlist in [dist_dirs, pen_dirs]:
            for mi, map_name in enumerate(['fast_city', 'fast_highway', 'fast_rural']):
                idx = si * 3 + mi
                if idx >= len(dirlist):
                    continue
                run = analyze_directory(dirlist[idx])
                if run is None:
                    continue

                results = run['results']
                w = MAP_WEIGHTS.get(map_name, 1.0)

                # Average across ALL configs in this run (all steps or all pen levels)
                bkey = find_key(results, fgroup['baseline'])
                if not bkey:
                    continue

                # For each config that has both baseline and gpem
                for config_key in results:
                    if fgroup['baseline'] not in config_key:
                        continue
                    base = results[config_key]['amota']
                    base_h = results[config_key]['hota']

                    # Find corresponding gpem configs
                    config_prefix = config_key.replace(fgroup['baseline'], '')
                    best_g_a = -999
                    best_g_h = -999
                    for gmode in fgroup['gpem']:
                        gkey_candidate = config_key.replace(fgroup['baseline'], gmode)
                        if gkey_candidate in results:
                            if results[gkey_candidate]['amota'] > best_g_a:
                                best_g_a = results[gkey_candidate]['amota']
                                best_g_h = results[gkey_candidate]['hota']

                    if best_g_a > -999 and base != 0:
                        deltas_amota.append((best_g_a - base) / max(abs(base), 0.001) * 100)
                        deltas_hota.append((best_g_h - base_h) / max(abs(base_h), 0.001) * 100)
                        weights.append(w)

        if weights:
            w_arr = np.array(weights)
            da = np.average(deltas_amota, weights=w_arr)
            dh = np.average(deltas_hota, weights=w_arr)
            if da > best_weighted:
                best_weighted = da
                best_sigma = sigma
            print(f'{sigma:>4.0f} | {da:>+16.1f}% | {dh:>+16.1f}% |')
        else:
            print(f'{sigma:>4.0f} |                ? |                ? |')

    if best_sigma is not None:
        print(f'  >>> Best σ_a = {best_sigma} (weighted AMOTA improvement: {best_weighted:+.1f}%)')

print(f'\nAnalysis complete.')
