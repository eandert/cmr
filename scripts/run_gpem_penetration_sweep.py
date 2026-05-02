#!/usr/bin/env python3
"""
GPEM Penetration Sweep Runner

This script runs the GPEM (Generalized Parameterized Error Modeling) penetration sweep suite.
It performs an A/B comparison test between:
  - A: Static covariance estimates for localization and detection
  - B: GPEM parameterized error model for heterogeneous fleet

The test varies the percentage of AVs across different injection rates:
1%, 2.5%, 5%, 10%, 20%, ..., 100%

Usage:
    python scripts/run_gpem_penetration_sweep.py [options]

Example:
    python scripts/run_gpem_penetration_sweep.py --runs 10 --parallel 4
"""

import sys
import os

# Add parent src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from experiment_runner import ExperimentRunner, create_gpem_penetration_sweep_suite
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Run GPEM Penetration Sweep: A/B test of static covariance vs parameterized error model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "-n", "--runs",
        type=int,
        default=10,
        help="Number of runs per configuration (default: 10)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="results",
        help="Output directory for results (default: results)"
    )
    
    parser.add_argument(
        "--warmup",
        type=int,
        default=600,
        help="Number of warmup steps before recording (default: 600)"
    )
    
    parser.add_argument(
        "--record",
        type=int,
        default=6000,
        help="Number of recording steps (default: 6000)"
    )
    
    parser.add_argument(
        "-j", "--parallel",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1 = sequential)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=True,
        help="Show detailed progress information"
    )
    
    args = parser.parse_args()
    
    # Create the GPEM penetration sweep suite
    print("\n" + "="*70)
    print("  GPEM PENETRATION SWEEP - A/B Comparison: Static vs Parameterized Model")
    print("="*70)
    print()
    print("  Configuration:")
    print(f"    • AV Injection Rates: 1%, 2.5%, 5%, 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100%")
    print(f"    • Runs per rate: {args.runs}")
    print(f"    • Total configurations: 26 (13 rates * 2 models)")
    print(f"    • Total runs: {26 * args.runs}")
    print(f"    • Warmup steps: {args.warmup}")
    print(f"    • Recording steps: {args.record}")
    print(f"    • Parallel workers: {args.parallel}")
    print("="*70)
    print()
    
    # Create the suite
    suite = create_gpem_penetration_sweep_suite(
        runs_per_config=args.runs,
        warmup_steps=args.warmup,
        record_steps=args.record,
    )
    
    # Run the experiments
    runner = ExperimentRunner(output_dir=args.output)
    results = runner.run_suite(suite, verbose=args.verbose, parallel=args.parallel)
    
    # After suite completion, automatically generate plot
    try:
        print("\n📊 Generating GPEM comparison plot...")
        # Since we're in scripts/, we need to go up one level to find where plot_gpem_results might be
        # but it's in the same directory as this script.
        from plot_gpem_results import plot_gpem_results_simple
        plot_gpem_results_simple(runner.current_experiment_dir)
    except Exception as e:
        print(f"  ⚠️  Failed to generate plot: {e}")

    print("\n" + "="*70)
    print(f"Results saved to: {runner.current_experiment_dir}")
    print("="*70)
    print()


if __name__ == "__main__":
    main()
