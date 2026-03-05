#!/usr/bin/env python3
"""
GPEM Penetration Sweep Results Analyzer

This script analyzes the results from a GPEM penetration sweep run and generates:
- Summary statistics
- Comparison plots
- Statistical tests
- Improvement analysis

Usage:
    python scripts/analyze_gpem_penetration_sweep.py <results_directory>
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from statistics import mean, stdev


class GPEMResultsAnalyzer:
    """Analyzer for GPEM penetration sweep results."""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        if not self.results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")
        
        self.summary_file = self.results_dir / "summary.json"
        if not self.summary_file.exists():
            raise FileNotFoundError(f"Summary file not found: {self.summary_file}")
        
        with open(self.summary_file) as f:
            self.summary = json.load(f)
        
        # Determine AV rates from keys in summary if possible
        self.av_rates = []
        for key in self.summary.get('results', {}).keys():
            if key.startswith("static_cov_av_") and key.endswith("pct"):
                try:
                    rate = float(key.replace("static_cov_av_", "").replace("pct", ""))
                    if rate not in self.av_rates:
                        self.av_rates.append(rate)
                except ValueError:
                    continue
        
        if not self.av_rates:
            self.av_rates = [1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        
        self.av_rates.sort()
    
    def extract_model_data(self) -> Dict[float, Dict]:
        """Extract data for both models at each AV rate."""
        data = {}
        
        for rate in self.av_rates:
            # Use fixed precision to match keys
            static_key = f"static_cov_av_{rate:.1f}pct"
            gpem_key = f"gpem_model_av_{rate:.1f}pct"
            
            data[rate] = {
                'static': self.summary['results'].get(static_key, {}),
                'gpem': self.summary['results'].get(gpem_key, {}),
            }
        
        return data
    
    def print_comparison_table(self):
        """Print a formatted comparison table."""
        data = self.extract_model_data()
        
        print("\n" + "="*90)
        print("GPEM PENETRATION SWEEP: MODEL COMPARISON RESULTS")
        print("="*90)
        print()
        print(f"{'AV Rate':<10} | {'Static AMOTA':<20} | {'GPEM AMOTA':<20} | {'Improvement':<15}")
        print("-"*90)
        
        total_improvements = []
        
        for rate in self.av_rates:
            static = data[rate]['static']
            gpem = data[rate]['gpem']
            
            if static and gpem:
                static_mean = static.get('avg_global_amota_mean', 0)
                static_std = static.get('avg_global_amota_std', 0)
                gpem_mean = gpem.get('avg_global_amota_mean', 0)
                gpem_std = gpem.get('avg_global_amota_std', 0)
                
                improvement = gpem_mean - static_mean
                improvement_pct = (improvement / static_mean * 100) if static_mean != 0 else 0
                
                total_improvements.append(improvement)
                
                improvement_str = f"{improvement:+.4f} ({improvement_pct:+.1f}%)"
                if improvement > 0:
                    improvement_str += " ✓"
                elif improvement < 0:
                    improvement_str += " ✗"
                
                print(f"{rate:>7.1f}% | {static_mean:.4f}±{static_std:.4f}  | "
                      f"{gpem_mean:.4f}±{gpem_std:.4f}  | {improvement_str:<15}")
        
        print("-"*90)
        
        # Summary statistics
        if total_improvements:
            avg_improvement = mean(total_improvements)
            print(f"\nAverage improvement: {avg_improvement:+.4f}")
            
            if len(total_improvements) > 1:
                std_improvement = stdev(total_improvements)
                print(f"Std dev of improvements: {std_improvement:.4f}")
            
            # Count wins
            wins = sum(1 for imp in total_improvements if imp > 0)
            losses = sum(1 for imp in total_improvements if imp < 0)
            ties = len(total_improvements) - wins - losses
            
            print(f"\nGPEM Model Performance: {wins} wins, {losses} losses, {ties} ties")
            print(f"Win rate: {wins/len(total_improvements)*100:.1f}%")
    
    def print_generalization_analysis(self):
        """Analyze generalization (consistency across AV rates)."""
        data = self.extract_model_data()
        
        print("\n" + "="*90)
        print("GENERALIZATION ANALYSIS (Lower std = Better generalization)")
        print("="*90)
        print()
        
        static_stds = []
        gpem_stds = []
        
        for rate in self.av_rates:
            static = data[rate]['static']
            gpem = data[rate]['gpem']
            
            if static:
                static_stds.append(static.get('avg_global_amota_std', 0))
            if gpem:
                gpem_stds.append(gpem.get('avg_global_amota_std', 0))
        
        if static_stds:
            static_avg_std = mean(static_stds)
            print(f"Static model average std dev:  {static_avg_std:.6f}")
        
        if gpem_stds:
            gpem_avg_std = mean(gpem_stds)
            print(f"GPEM model average std dev:    {gpem_avg_std:.6f}")
        
        if static_stds and gpem_stds:
            diff = static_avg_std - gpem_avg_std
            pct = (diff / static_avg_std * 100) if static_avg_std != 0 else 0
            
            print(f"\nDifference: {diff:+.6f} ({pct:+.1f}%)")
            if diff > 0:
                print("✓ GPEM is more consistent (better generalization)")
            elif diff < 0:
                print("✗ Static is more consistent")
            else:
                print("~ Equivalent consistency")
    
    def print_av_sensitivity(self):
        """Analyze sensitivity to AV percentage changes."""
        data = self.extract_model_data()
        
        print("\n" + "="*90)
        print("AV PERCENTAGE SENSITIVITY ANALYSIS")
        print("="*90)
        print()
        
        static_means = []
        gpem_means = []
        
        for rate in self.av_rates:
            static = data[rate]['static']
            gpem = data[rate]['gpem']
            
            if static:
                static_means.append(static.get('avg_global_amota_mean', 0))
            if gpem:
                gpem_means.append(gpem.get('avg_global_amota_mean', 0))
        
        if len(static_means) > 1:
            static_range = max(static_means) - min(static_means)
            static_var = stdev(static_means) if len(static_means) > 1 else 0
            print(f"Static model:")
            print(f"  Range: {static_range:.4f}")
            print(f"  Std dev: {static_var:.6f}")
        
        if len(gpem_means) > 1:
            gpem_range = max(gpem_means) - min(gpem_means)
            gpem_var = stdev(gpem_means) if len(gpem_means) > 1 else 0
            print(f"\nGPEM model:")
            print(f"  Range: {gpem_range:.4f}")
            print(f"  Std dev: {gpem_var:.6f}")
        
        if len(static_means) > 1 and len(gpem_means) > 1:
            print(f"\nSensitivity comparison:")
            if static_var > gpem_var:
                print(f"✓ GPEM is more stable ({gpem_var:.6f} vs {static_var:.6f})")
            else:
                print(f"✗ Static is more stable ({static_var:.6f} vs {gpem_var:.6f})")
    
    def generate_report(self):
        """Generate and print full analysis report."""
        print("\n" + "="*90)
        print("GPEM PENETRATION SWEEP: FULL ANALYSIS REPORT")
        print("="*90)
        print(f"Results directory: {self.results_dir}")
        print(f"Suite name: {self.summary.get('suite_name', 'Unknown')}")
        print(f"Total runs: {self.summary.get('total_runs', 0)}")
        
        self.print_comparison_table()
        self.print_generalization_analysis()
        self.print_av_sensitivity()
        
        print("\n" + "="*90)
        print("ANALYSIS COMPLETE")
        print("="*90)
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze GPEM penetration sweep results"
    )
    
    parser.add_argument(
        "results_dir",
        type=str,
        help="Path to GPEM results directory"
    )
    
    args = parser.parse_args()
    
    try:
        analyzer = GPEMResultsAnalyzer(args.results_dir)
        analyzer.generate_report()
    
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
