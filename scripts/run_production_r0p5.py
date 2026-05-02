#!/usr/bin/env python3
"""
Production runs for paper results on fast maps with baseline R=0.5.
Runs both penetration and distribution sweeps on all 3 fast maps.
Updated baseline R to 0.5 for more defensible GPEM advantage demonstration.
"""
import subprocess, sys, time

maps = ['fast_city', 'fast_highway', 'fast_rural']
suites = ['gpem_penetration_sweep', 'gpem_distribution_sweep']

FF = "3000,5990,8980,11970,14960,17950,20940,23930,26920,29900"
SEED = 42
WORKERS = 8
WARMUP = 100
RECORD = 6000

total = len(maps) * len(suites)
done = 0
start = time.time()

print(f"Starting production runs with baseline R=0.5")
print(f"Total configurations: {total} (3 maps × 2 suites)")
print(f"Each configuration: 10 independent trials\n")

for map_name in maps:
    for suite in suites:
        done += 1
        elapsed = time.time() - start
        eta = (elapsed / done) * (total - done) / 60 if done > 0 else 0
        label = 'pen' if 'penetration' in suite else 'dist'
        print(f'\n[{done}/{total}] {map_name} {label} (ETA: {eta:.0f}m)')
        sys.stdout.flush()

        cmd = [
            'python3', 'src/experiment_runner.py',
            '--suite', suite,
            '--runs', '10',
            '--ff-list', FF,
            '--seed', str(SEED),
            '-j', str(WORKERS),
            '--warmup', str(WARMUP),
            '--record', str(RECORD),
            '--map', map_name,
        ]

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=86400)
        except subprocess.TimeoutExpired:
            print(f'  TIMEOUT')
            continue

        if proc.returncode != 0:
            print(f'  FAILED (rc={proc.returncode})')
            # Print last few lines of stderr
            for line in proc.stderr.strip().split('\n')[-5:]:
                print(f'    {line}')
            continue

        # Print key summary results
        for line in proc.stdout.split('\n'):
            if '📊' not in line:
                continue
            for pattern in ['step_7_baseline:', 'step_7_gpem_linear:',
                           'step_7_ci_baseline:', 'step_7_ci_gpem_linear:',
                           'step_7_bici_baseline:', 'step_7_bici_gpem_linear:',
                           'step_7_sabre_baseline:', 'step_7_sabre_gpem_linear:',
                           'step_7_akf_baseline:', 'step_7_akf_gpem_linear:',
                           'step_7_pf_baseline:', 'step_7_pf_gpem_linear:',
                           'baseline_av_100.0pct:', 'gpem_linear_av_100.0pct:',
                           'ci_baseline_av_100.0pct:', 'ci_gpem_linear_av_100.0pct:',
                           'bici_baseline_av_100.0pct:', 'bici_gpem_linear_av_100.0pct:',
                           'sabre_baseline_av_100.0pct:', 'sabre_gpem_linear_av_100.0pct:',
                           'akf_baseline_av_100.0pct:', 'akf_gpem_linear_av_100.0pct:',
                           'pf_baseline_av_100.0pct:', 'pf_gpem_linear_av_100.0pct:']:
                if pattern in line:
                    print(f'  {line.strip()[2:]}')
                    break

        print(f'  Done in {(time.time() - start) / 60:.0f}m total')

print(f'\nAll production runs complete. Total time: {(time.time()-start)/60:.1f}m')
