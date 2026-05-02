#!/usr/bin/env python3
"""
Sweep SABRE and AKF adaptation parameters.
σ_a and baseline R are locked from previous sweeps.

Sweeps:
  SABRE: adapt_rate (0.05, 0.1, 0.15, 0.2, 0.3, 0.5) × window_size (10, 15, 20, 30)
  AKF:   ADAPT_ALPHA (0.05, 0.1, 0.15, 0.2, 0.3, 0.5) × WINDOW_SIZE (10, 15, 20, 30)
"""
import re, subprocess, sys, time, json, glob

maps = ['fast_city', 'fast_highway', 'fast_rural']
COMMON_DIST = '--runs 1 --ff-list 3000 --seed 42 -j 13 --warmup 100 --record 1200'
COMMON_PEN  = '--runs 1 --ff-list 3000 --seed 42 -j 13 --warmup 100 --record 600'

LUT_PATH = 'src/filters/source_reliability_lut.py'
AKF_PATH = 'src/filters/adaptive_kalman.py'

adapt_rates = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
window_sizes = [10, 15, 20, 30]

with open(LUT_PATH, 'r') as f:
    original_lut = f.read()
with open(AKF_PATH, 'r') as f:
    original_akf = f.read()

# We sweep adapt_rate and window_size together for both SABRE and AKF
# (same value for both — they share the concept)
configs = []
for ar in adapt_rates:
    for ws in window_sizes:
        configs.append((ar, ws))

total = len(configs) * len(maps) * 2  # dist + pen
done = 0
start = time.time()

try:
    for adapt_rate, window_size in configs:
        # Patch SABRE LUT
        patched_lut = re.sub(r'adapt_rate:\s*float\s*=\s*[0-9.]+', f'adapt_rate: float = {adapt_rate}', original_lut)
        patched_lut = re.sub(r'window_size:\s*int\s*=\s*\d+', f'window_size: int = {window_size}', patched_lut)
        with open(LUT_PATH, 'w') as f:
            f.write(patched_lut)

        # Patch AKF
        patched_akf = re.sub(r'ADAPT_ALPHA\s*=\s*[0-9.]+', f'ADAPT_ALPHA = {adapt_rate}', original_akf)
        patched_akf = re.sub(r'WINDOW_SIZE\s*=\s*\d+', f'WINDOW_SIZE = {window_size}', patched_akf)
        with open(AKF_PATH, 'w') as f:
            f.write(patched_akf)

        for map_name in maps:
            for suite in ['gpem_distribution_sweep', 'gpem_penetration_sweep']:
                done += 1
                elapsed = time.time() - start
                eta = (elapsed / done) * (total - done) / 60 if done > 0 else 0
                label = 'dist' if 'distribution' in suite else 'pen'
                common = COMMON_DIST if 'distribution' in suite else COMMON_PEN
                print(f'\n[{done}/{total}] α={adapt_rate} w={window_size} {map_name} {label} (ETA: {eta:.0f}m)')
                sys.stdout.flush()

                cmd = f'python3 src/experiment_runner.py --suite {suite} {common} --map {map_name}'.split()
                try:
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
                except subprocess.TimeoutExpired:
                    print(f'  TIMEOUT')
                    continue

                if proc.returncode != 0:
                    print(f'  FAILED')
                    continue

                # Print SABRE and AKF results
                for line in proc.stdout.split('\n'):
                    if '📊' not in line:
                        continue
                    for pattern in ['step_7_sabre_baseline:', 'step_7_sabre_gpem_linear:',
                                   'step_7_akf_baseline:', 'step_7_akf_gpem_linear:',
                                   'sabre_baseline_av_100.0pct:', 'sabre_gpem_linear_av_100.0pct:',
                                   'akf_baseline_av_100.0pct:', 'akf_gpem_linear_av_100.0pct:']:
                        if pattern in line:
                            print(f'  {line.strip()[2:]}')
                            break

finally:
    with open(LUT_PATH, 'w') as f:
        f.write(original_lut)
    with open(AKF_PATH, 'w') as f:
        f.write(original_akf)
    print(f'\nRestored originals. Total time: {(time.time()-start)/60:.1f}m')
