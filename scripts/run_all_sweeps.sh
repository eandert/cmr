#!/bin/bash
set -e
COMMON="--runs 1 --ff-list 3000 --seed 42 -j 12 --warmup 100 --record 600"

echo "=== Distribution Sweep: Highway ==="
python3 src/experiment_runner.py --suite gpem_distribution_sweep $COMMON --map highway

echo "=== Distribution Sweep: City ==="
python3 src/experiment_runner.py --suite gpem_distribution_sweep $COMMON --map city

echo "=== Distribution Sweep: Rural ==="
python3 src/experiment_runner.py --suite gpem_distribution_sweep $COMMON --map rural

echo "=== Penetration Sweep: Highway ==="
python3 src/experiment_runner.py --suite gpem_penetration_sweep $COMMON --map highway

echo "=== Penetration Sweep: City ==="
python3 src/experiment_runner.py --suite gpem_penetration_sweep $COMMON --map city

echo "=== Penetration Sweep: Rural ==="
python3 src/experiment_runner.py --suite gpem_penetration_sweep $COMMON --map rural

echo "=== ALL DONE ==="
