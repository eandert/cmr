#!/usr/bin/env python3
"""
Sweep small baseline R values (0.01–0.04) to complete the left side of the baseline sweep chart.
Uses per-filter sigma_a already locked in filter_config.py.
"""
import re, subprocess, sys, time

baselines = [0.01, 0.02, 0.03, 0.04]
maps = ['fast_city', 'fast_highway', 'fast_rural']
suites = ['gpem_distribution_sweep', 'gpem_penetration_sweep']
COMMON_DIST = '--runs 1 --ff-list 3000 --seed 42 -j 13 --warmup 100 --record 1200'
COMMON_PEN  = '--runs 1 --ff-list 3000 --seed 42 -j 13 --warmup 100 --record 600'
TRACI_PATH = 'src/traci_interface.py'

with open(TRACI_PATH, 'r') as f:
    original_traci = f.read()

total = len(baselines) * len(maps) * 2
done = 0
start = time.time()

try:
    for baseline in baselines:
        # Patch baseline value
        patched = re.sub(
            r'# Override: use fixed [0-9.]+ for baseline testing\n    baseline_var = [0-9.]+',
            f'# Override: use fixed {baseline} for baseline testing\n    baseline_var = {baseline}',
            original_traci
        )
        with open(TRACI_PATH, 'w') as f:
            f.write(patched)

        for map_name in maps:
            for suite in suites:
                done += 1
                elapsed = time.time() - start
                eta = (elapsed / done) * (total - done) / 60 if done > 0 else 0
                common = COMMON_DIST if 'distribution' in suite else COMMON_PEN
                label = 'dist' if 'distribution' in suite else 'pen'
                print(f'\n[{done}/{total}] baseline={baseline} {map_name} {label} (ETA: {eta:.0f}m)')
                sys.stdout.flush()

                cmd = f'python3 src/experiment_runner.py --suite {suite} {common} --map {map_name}'.split()
                try:
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
                except subprocess.TimeoutExpired:
                    print(f'  TIMEOUT — skipping')
                    continue

                # Print key results
                for line in proc.stdout.split('\n'):
                    if '📊' not in line:
                        continue
                    for pattern in ['step_7_baseline:', 'step_7_gpem_linear:',
                                   'step_7_ci_baseline:', 'step_7_ci_gpem_linear:',
                                   'step_7_sabre_baseline:', 'step_7_sabre_gpem_linear:',
                                   'step_7_akf_baseline:', 'step_7_akf_gpem_linear:',
                                   'baseline_av_100.0pct:', 'gpem_linear_av_100.0pct:',
                                   'ci_baseline_av_100.0pct:', 'ci_gpem_linear_av_100.0pct:',
                                   'sabre_baseline_av_100.0pct:', 'sabre_gpem_linear_av_100.0pct:',
                                   'akf_baseline_av_100.0pct:', 'akf_gpem_linear_av_100.0pct:']:
                        if pattern in line:
                            if any(p in line for p in ['pf_', 'bici_']) and not line.strip().startswith('📊 ' + pattern.split(':')[0]):
                                continue
                            print(f'  {line.strip()[2:]}')
                            break

                if proc.returncode != 0:
                    print(f'  FAILED')

finally:
    with open(TRACI_PATH, 'w') as f:
        f.write(original_traci)
    print(f'\nRestored original traci_interface.py. Total time: {(time.time()-start)/60:.1f}m')
