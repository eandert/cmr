"""
Experiment Runner for CMR Simulation

This module provides a high-level interface for running simulation experiments
with multiple configurations, repeated runs, and organized result output.

Usage:
    from experiment_runner import ExperimentRunner
    
    runner = ExperimentRunner(output_dir="results")
    results = runner.run_suite(experiment_suite)
"""

import os
import sys
import json
import csv
import time
import signal
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from statistics import mean, stdev
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Global flag for graceful shutdown
_shutdown_requested = False

def _signal_handler(signum, frame):
    """Handle Ctrl+C by setting shutdown flag."""
    global _shutdown_requested
    _shutdown_requested = True
    print("\n\n⚠️  Shutdown requested (Ctrl+C). Terminating workers...")
    sys.stdout.flush()

def _kill_orphan_sumo_processes():
    """Kill any orphan SUMO processes that may have been left running."""
    import subprocess
    try:
        # Kill SUMO processes - use SIGKILL (-9) for immediate termination
        result = subprocess.run(
            ["pkill", "-9", "-f", "sumo"],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            print("  ✓ Killed orphan SUMO processes")
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass  # pkill not available or timed out

# Progress bar
from tqdm import tqdm

# Import simulation components
from config_templates import get_perfect_config, get_normal_sensors_config


class LiveDashboard:
    """Display a live updating dashboard in terminal."""
    
    def __init__(self):
        self.num_lines = 12
        self.initialized = False
        self.last_update = 0
        self.update_interval = 0.1  # Update at most every 100ms
    
    def _move_up(self, lines: int):
        """Move cursor up N lines."""
        if lines > 0:
            sys.stdout.write(f"\033[{lines}A")
    
    def _clear_line(self):
        """Clear current line."""
        sys.stdout.write("\033[2K\r")
    
    def _progress_bar(self, current: int, total: int, width: int = 20) -> str:
        """Create a text progress bar."""
        if total <= 0:
            return "░" * width
        pct = min(current / total, 1.0)
        filled = int(width * pct)
        return "█" * filled + "░" * (width - filled)
    
    def update(self, stats: Dict[str, Any], force: bool = False):
        """Update the dashboard display in place."""
        now = time.time()
        if not force and (now - self.last_update) < self.update_interval:
            return
        self.last_update = now
        
        if self.initialized:
            self._move_up(self.num_lines)
        
        # Extract stats with defaults
        test_name = stats.get('test_name', 'Unknown')[:20]
        run_num = stats.get('run_num', 0)
        total_runs = stats.get('total_runs', 0)
        config_num = stats.get('config_num', 0)
        total_configs = stats.get('total_configs', 0)
        step = stats.get('step', 0)
        total_steps = stats.get('total_steps', 0)
        cav_count = stats.get('cav_count', 0)
        cav_amota = stats.get('cav_amota', 0)
        cis_amota = stats.get('cis_amota', 0)
        global_amota = stats.get('global_amota', 0)
        recorded = stats.get('recorded_steps', 0)
        
        # Calculate percentages
        step_pct = (step / total_steps * 100) if total_steps > 0 else 0
        overall_done = stats.get('overall_completed', 0)
        overall_total = stats.get('overall_total', 1)
        overall_pct = (overall_done / overall_total * 100) if overall_total > 0 else 0
        
        # Build progress bars
        step_bar = self._progress_bar(step, total_steps, 25)
        overall_bar = self._progress_bar(overall_done, overall_total, 25)
        
        lines = [
            f"╔{'═' * 62}╗",
            f"║  🧪 Test: {test_name:<20} │ Config {config_num}/{total_configs} │ Run {run_num}/{total_runs}  ║",
            f"╠{'═' * 62}╣",
            f"║  📊 Overall:  {overall_bar} {overall_pct:5.1f}%       ║",
            f"║  📈 Current:  {step_bar} {step_pct:5.1f}%       ║",
            f"╠{'═' * 62}╣",
            f"║  🚗 CAVs: {cav_count:>4}  │  CAV AMOTA:  {cav_amota:>6.4f}                   ║",
            f"║  🚦 CIS AMOTA: {cis_amota:>6.4f}  │  Global AMOTA: {global_amota:>6.4f}          ║",
            f"╠{'═' * 62}╣",
            f"║  Step: {step:>6} / {total_steps:<6}  │  Recorded: {recorded:>6} steps            ║",
            f"╚{'═' * 62}╝",
            "",  # Empty line for spacing
        ]
        
        for line in lines:
            self._clear_line()
            print(line)
        
        sys.stdout.flush()
        self.initialized = True
    
    def finish(self):
        """Mark display as finished."""
        if self.initialized:
            self._move_up(self.num_lines)
            for _ in range(self.num_lines):
                self._clear_line()
                print()
        self.initialized = False


# Global dashboard instance
_dashboard = None

def get_dashboard() -> LiveDashboard:
    """Get or create the global dashboard."""
    global _dashboard
    if _dashboard is None:
        _dashboard = LiveDashboard()
    return _dashboard


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment variant."""
    name: str
    config: Dict[str, Any]
    description: str = ""
    # When set, one simulation run produces three synced outputs (static, gpem_linear, gpem_quadratic)
    triple_fusion_output_names: Optional[List[str]] = None  # e.g. ["static_cov_av_10pct", "gpem_linear_av_10pct", "gpem_quadratic_av_10pct", "baseline_av_10pct"]


@dataclass
class ExperimentSuite:
    """Definition of a complete experiment suite."""
    name: str
    description: str = ""
    runs_per_config: int = 10
    baseline: Optional[ExperimentConfig] = None
    variants: List[ExperimentConfig] = field(default_factory=list)
    
    # Data collection timing
    fast_forward_steps: int = 0      # Steps before any processing starts (quick simulation)
    fast_forward_range: Optional[Tuple[int, int]] = None  # Range for randomized FF steps
    warmup_steps: int = 600          # Steps before recording starts (full systems running)
    record_steps: int = 6000         # Number of steps to record
    max_steps: Optional[int] = None  # Max steps (None = run to completion)
    
    # Error injection timing (relative to simulation start, not warmup)
    error_injection_start: Optional[int] = None  # None = use warmup_steps
    error_injection_end: Optional[int] = None    # None = until end


@dataclass
class RunResult:
    """Results from a single simulation run."""
    config_name: str
    run_id: int
    
    # Timing
    total_steps: int
    recorded_steps: int
    wall_time_seconds: float
    
    # Aggregated metrics
    cav_metrics: Dict[str, Any] = field(default_factory=dict)
    cis_metrics: Dict[str, Any] = field(default_factory=dict)
    global_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Computed averages
    avg_cav_amota: float = 0.0
    avg_cis_amota: float = 0.0
    avg_global_amota: float = 0.0
    avg_cav_amotp: float = 0.0
    avg_cis_amotp: float = 0.0
    avg_global_amotp: float = 0.0
    
    # Performance metrics
    avg_step_time: float = 0.0
    avg_cis_detection_time: float = 0.0
    avg_cav_detection_time: float = 0.0
    avg_global_fusion_time: float = 0.0
    
    # HOTA metrics (per-run, already final — not averaged across frames)
    hota: float = 0.0
    deta: float = 0.0
    assa: float = 0.0

    # Perception scoring (Conclave)
    perception_scores: Dict[str, float] = field(default_factory=dict)
    perception_anomalies: Dict[str, bool] = field(default_factory=dict)
    num_perception_anomalies: int = 0


@dataclass
class AggregatedResults:
    """Aggregated results across multiple runs of the same config."""
    config_name: str
    num_runs: int
    
    # AMOTA statistics
    avg_global_amota_mean: float = 0.0
    avg_global_amota_std: float = 0.0
    avg_cis_amota_mean: float = 0.0
    avg_cis_amota_std: float = 0.0
    avg_cav_amota_mean: float = 0.0
    avg_cav_amota_std: float = 0.0
    
    # AMOTP statistics
    avg_global_amotp_mean: float = 0.0
    avg_global_amotp_std: float = 0.0
    avg_cis_amotp_mean: float = 0.0
    avg_cis_amotp_std: float = 0.0
    avg_cav_amotp_mean: float = 0.0
    avg_cav_amotp_std: float = 0.0
    
    # HOTA statistics
    avg_hota_mean: float = 0.0
    avg_hota_std: float = 0.0
    avg_deta_mean: float = 0.0
    avg_deta_std: float = 0.0
    avg_assa_mean: float = 0.0
    avg_assa_std: float = 0.0

    # Perception scoring statistics
    avg_perception_anomalies_mean: float = 0.0
    avg_perception_anomalies_std: float = 0.0
    total_anomaly_detections: int = 0
    
    # All individual run results
    runs: List[RunResult] = field(default_factory=list)


def _run_simulation_worker(args: Tuple[str, int, Dict, str, Optional[Dict]]) -> Tuple[str, int, Dict, float, Optional[str]]:
    """
    Worker function for running a single simulation in a separate process.
    
    Args:
        args: Tuple of (config_name, run_id, config, config_dir, shared_progress)
        
    Returns:
        Tuple of (config_name, run_id, sim_result, wall_time, error_message)
    """
    import io
    
    # Put worker in new process group so Ctrl+C doesn't reach it or its subprocesses (SUMO)
    try:
        os.setpgrp()
    except OSError:
        pass  # May fail if already a process group leader
    
    # Ignore SIGINT in worker - let the parent process handle it
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    if len(args) >= 5:
        config_name, run_id, config, config_dir, shared_progress = args[0], args[1], args[2], args[3], args[4]
    else:
        config_name, run_id, config, config_dir = args[0], args[1], args[2], args[3]
        shared_progress = None
    
    # Redirect stdout/stderr to suppress worker output
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    
    try:
        # Import here to ensure fresh import in each process
        from traci_interface import run_simulation, shutdown_process_pool
        
        # Prepare config
        config = config.copy()
        
        # Add a simple progress callback that updates the shared dictionary
        if shared_progress is not None:
            job_key = f"{config_name}#{run_id}"
            def simple_callback(step, total, stats):
                shared_progress[job_key] = (step, total)
            config["_progress_callback"] = simple_callback
        
        # Disable internal parallelization in worker processes to avoid nested pools
        config["use_parallel"] = False
        
        start_time = time.time()
        try:
            sim_result = run_simulation(config)
            wall_time = time.time() - start_time
            return (config_name, run_id, sim_result, wall_time, None)
        except Exception as e:
            wall_time = time.time() - start_time
            return (config_name, run_id, None, wall_time, str(e))
        finally:
            # Clean up any process pool created in this worker
            try:
                shutdown_process_pool()
            except:
                pass
    finally:
        # Restore stdout/stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr



class ExperimentRunner:
    """
    High-level runner for simulation experiments.
    
    Handles:
    - Running multiple configurations
    - Repeated runs per configuration
    - Result aggregation and statistics
    - Organized output to files
    """
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        self.current_experiment_dir = None
        
    def run_suite(self, suite: ExperimentSuite, verbose: bool = True, parallel: int = 1) -> Dict[str, AggregatedResults]:
        """
        Run a complete experiment suite.
        
        Args:
            suite: ExperimentSuite defining all configurations to run
            verbose: If True, show detailed progress bars and stats
            parallel: Number of simulations to run concurrently (default: 1 = sequential)
            
        Returns:
            Dictionary mapping config names to aggregated results
        """
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.current_experiment_dir = os.path.join(
            self.output_dir, 
            f"{suite.name.replace(' ', '_')}_{timestamp}"
        )
        os.makedirs(self.current_experiment_dir, exist_ok=True)
        
        # Save suite configuration
        self._save_suite_config(suite)
        
        # Generate randomized fast-forward steps if a range is provided
        # IMPORTANT: The same FF value is used for the same run_id across ALL configs.
        # This ensures fair comparison: run 1 of static, linear, and quadratic all
        # start from the same simulation state (same FF steps).
        import random
        if suite.fast_forward_range:
            ff_min, ff_max = suite.fast_forward_range
            # Fixed seed ensures reproducible FF values across runs
            rng = random.Random(42) 
            self.ff_steps_per_run = [rng.randint(ff_min, ff_max) for _ in range(suite.runs_per_config)]
            print(f"  🎲 Randomized FF range: {ff_min}-{ff_max} steps")
            print(f"  📋 FF steps per run_id (shared across all configs): {self.ff_steps_per_run}")
        else:
            self.ff_steps_per_run = [suite.fast_forward_steps] * suite.runs_per_config
        
        all_results: Dict[str, AggregatedResults] = {}
        
        # Build list of all configs to run
        configs_to_run = []
        if suite.baseline:
            configs_to_run.append(("baseline", suite.baseline))
        for variant in suite.variants:
            configs_to_run.append((variant.name, variant))
        
        total_runs = len(configs_to_run) * suite.runs_per_config
        
        print(f"\n{'═'*60}")
        print(f"  🚀 Experiment Suite: {suite.name}")
        print(f"{'═'*60}")
        print(f"  📊 Configurations: {len(configs_to_run)}")
        print(f"  🔄 Runs per config: {suite.runs_per_config}")
        print(f"  📈 Total runs: {total_runs}")
        print(f"  ⏳ Warmup steps: {suite.warmup_steps}")
        print(f"  📝 Recording steps: {suite.record_steps}")
        print(f"  🔀 Parallel workers: {parallel}")
        print(f"  📁 Output: {self.current_experiment_dir}")
        print(f"{'═'*60}\n")
        
        # Use parallel execution if parallel > 1
        if parallel > 1:
            return self._run_suite_parallel(suite, configs_to_run, total_runs, parallel, verbose)
        
        # Sequential execution (original logic)
        completed_runs = 0
        dashboard = get_dashboard() if verbose else None
        
        # Run each configuration
        for config_idx, (config_name, experiment_config) in enumerate(configs_to_run):
            config_dir = os.path.join(self.current_experiment_dir, config_name.replace(' ', '_'))
            os.makedirs(config_dir, exist_ok=True)
            output_names = getattr(experiment_config, 'triple_fusion_output_names', None)

            if output_names:
                # Triple fusion: one sim per run_id produces 3 synced outputs
                for out_name in output_names:
                    os.makedirs(os.path.join(self.current_experiment_dir, out_name.replace(' ', '_')), exist_ok=True)
                run_results_by_out = {out: [] for out in output_names}
                for run_id in range(1, suite.runs_per_config + 1):
                    run_config = self._prepare_run_config(experiment_config.config, suite, run_id)
                    run_config["_progress_callback"] = self._create_progress_callback(
                        config_name, run_id, suite.runs_per_config,
                        config_idx + 1, len(configs_to_run), completed_runs, total_runs, verbose
                    )
                    from traci_interface import run_simulation
                    start_time = time.time()
                    sim_result = run_simulation(run_config)
                    wall_time = time.time() - start_time
                    triple_results = self._process_triple_sim_result(sim_result, run_id, output_names)
                    for i, out_name in enumerate(output_names):
                        triple_results[i].wall_time_seconds = wall_time
                        run_results_by_out[out_name].append(triple_results[i])
                        self._save_run_result(os.path.join(self.current_experiment_dir, out_name.replace(' ', '_')), triple_results[i])
                    completed_runs += 1
                for out_name in output_names:
                    aggregated = self._aggregate_results(out_name, run_results_by_out[out_name])
                    all_results[out_name] = aggregated
                    self._save_aggregated_results(os.path.join(self.current_experiment_dir, out_name.replace(' ', '_')), aggregated)
                    print(f"  ✅ {out_name}: AMOTA = {aggregated.avg_global_amota_mean:.4f} (±{aggregated.avg_global_amota_std:.4f}) | AMOTP = {aggregated.avg_global_amotp_mean:.4f}m (±{aggregated.avg_global_amotp_std:.4f})")
            else:
                run_results = []
                for run_id in range(1, suite.runs_per_config + 1):
                    run_config = self._prepare_run_config(experiment_config.config, suite, run_id)
                    run_config["_progress_callback"] = self._create_progress_callback(
                        config_name, run_id, suite.runs_per_config,
                        config_idx + 1, len(configs_to_run), completed_runs, total_runs, verbose
                    )
                    start_time = time.time()
                    result = self._run_single(config_name, run_id, run_config)
                    result.wall_time_seconds = time.time() - start_time
                    run_results.append(result)
                    completed_runs += 1
                    self._save_run_result(config_dir, result)
                aggregated = self._aggregate_results(config_name, run_results)
                all_results[config_name] = aggregated
                self._save_aggregated_results(config_dir, aggregated)
                print(f"  ✅ {config_name}: AMOTA = {aggregated.avg_global_amota_mean:.4f} (±{aggregated.avg_global_amota_std:.4f}) | AMOTP = {aggregated.avg_global_amotp_mean:.4f}m (±{aggregated.avg_global_amotp_std:.4f})")

            if dashboard:
                dashboard.finish()
        
        # Save summary
        self._save_summary(suite, all_results)
        
        print(f"\n{'='*60}")
        print("Experiment Suite Complete!")
        print(f"Results saved to: {self.current_experiment_dir}")
        print(f"{'='*60}\n")
        
        return all_results
    
    def _prepare_run_config(self, base_config: Dict, suite: ExperimentSuite, run_id: int) -> Dict:
        """Prepare a config with timing settings from the suite."""
        config = copy.deepcopy(base_config)
        
        # Get fast-forward steps for this specific run ID
        # run_id is 1-indexed
        ff_steps = self.ff_steps_per_run[run_id - 1]
        
        # Add timing settings
        config["fast_forward_steps"] = ff_steps
        config["warmup_steps"] = suite.warmup_steps
        config["record_steps"] = suite.record_steps
        config["max_steps"] = suite.max_steps
        config["error_injection_start"] = suite.error_injection_start or (ff_steps + suite.warmup_steps)
        config["error_injection_end"] = suite.error_injection_end
        
        # Disable verbose output from simulation (we handle progress ourselves)
        config["quiet_mode"] = True
        
        return config
    
    def _create_progress_callback(self, config_name: str, run_id: int, total_runs: int,
                                   config_idx: int, total_configs: int, 
                                   overall_completed: int, overall_total: int, verbose: bool,
                                   shared_progress: Optional[Dict] = None):
        """Create a callback function for simulation progress updates."""
        dashboard = get_dashboard()
        job_key = f"{config_name}#{run_id}"
        
        def callback(step: int, total: int, stats: Dict[str, Any]):
            # Update shared progress if available (multiprocessing safe)
            if shared_progress is not None:
                shared_progress[job_key] = (step, total)
            
            if not verbose:
                return
            
            dashboard.update({
                'test_name': config_name,
                'run_num': run_id,
                'total_runs': total_runs,
                'config_num': config_idx,
                'total_configs': total_configs,
                'step': step,
                'total_steps': total,
                'cav_count': stats.get('cav_count', 0),
                'cav_amota': stats.get('cav_amota', 0),
                'cis_amota': stats.get('cis_amota', 0),
                'global_amota': stats.get('global_amota', 0),
                'recorded_steps': stats.get('recorded_steps', 0),
                'overall_completed': overall_completed,
                'overall_total': overall_total,
            })

        
        def cleanup():
            pass  # Dashboard persists across runs
        
        callback.cleanup = cleanup
        return callback
    
    def _run_single(self, config_name: str, run_id: int, config: Dict) -> RunResult:
        """Run a single simulation and return results."""
        # Import here to avoid circular imports
        from traci_interface import run_simulation
        
        # Run the simulation
        sim_result = run_simulation(config)
        
        # Extract metrics
        cav_metrics = sim_result.get("total_cav_metrics", {})
        cis_metrics = sim_result.get("total_cis_metrics", {})
        global_metrics = sim_result.get("total_global_metrics", {})
        
        # Calculate averages
        cav_amota_frames = sim_result.get("cav_amota_frames_count", 1)
        cis_amota_frames = sim_result.get("cis_amota_frames_count", 1)
        global_amota_frames = sim_result.get("global_amota_frames_count", 1)
        cav_amotp_frames = sim_result.get("cav_amotp_frames_count", 1)
        cis_amotp_frames = sim_result.get("cis_amotp_frames_count", 1)
        global_amotp_frames = sim_result.get("global_amotp_frames_count", 1)
        
        avg_cav_amota = cav_metrics.get("amota", 0) / max(cav_amota_frames, 1)
        avg_cis_amota = cis_metrics.get("amota", 0) / max(cis_amota_frames, 1)
        avg_global_amota = global_metrics.get("amota", 0) / max(global_amota_frames, 1)
        avg_cav_amotp = cav_metrics.get("amotp", 0) / max(cav_amotp_frames, 1)
        avg_cis_amotp = cis_metrics.get("amotp", 0) / max(cis_amotp_frames, 1)
        avg_global_amotp = global_metrics.get("amotp", 0) / max(global_amotp_frames, 1)
        
        return RunResult(
            config_name=config_name,
            run_id=run_id,
            total_steps=sim_result.get("iterations", 0),
            recorded_steps=sim_result.get("recorded_steps", sim_result.get("iterations", 0)),
            wall_time_seconds=0,  # Will be set by caller
            cav_metrics=cav_metrics,
            cis_metrics=cis_metrics,
            global_metrics=global_metrics,
            avg_cav_amota=avg_cav_amota,
            avg_cis_amota=avg_cis_amota,
            avg_global_amota=avg_global_amota,
            avg_cav_amotp=avg_cav_amotp,
            avg_cis_amotp=avg_cis_amotp,
            avg_global_amotp=avg_global_amotp,
            avg_step_time=sim_result.get("avg_step_time", 0),
            avg_cis_detection_time=sim_result.get("avg_cis_detection_time", 0),
            avg_cav_detection_time=sim_result.get("avg_cav_detection_time", 0),
            avg_global_fusion_time=sim_result.get("avg_global_fusion_time", 0),
            hota=sim_result.get("global_hota", {}).get("hota", 0.0),
            deta=sim_result.get("global_hota", {}).get("deta", 0.0),
            assa=sim_result.get("global_hota", {}).get("assa", 0.0),
        )

    def _aggregate_results(self, config_name: str, runs: List[RunResult]) -> AggregatedResults:
        """Aggregate results across multiple runs."""
        if not runs:
            return AggregatedResults(config_name=config_name, num_runs=0)
        
        global_amotas = [r.avg_global_amota for r in runs]
        cis_amotas = [r.avg_cis_amota for r in runs]
        cav_amotas = [r.avg_cav_amota for r in runs]
        global_amotps = [r.avg_global_amotp for r in runs]
        cis_amotps = [r.avg_cis_amotp for r in runs]
        cav_amotps = [r.avg_cav_amotp for r in runs]
        
        # Aggregate perception scoring metrics
        anomaly_counts = [r.num_perception_anomalies for r in runs]
        total_anomalies = sum(anomaly_counts)

        # Aggregate HOTA metrics
        hotas = [r.hota for r in runs]
        detas = [r.deta for r in runs]
        assas = [r.assa for r in runs]

        return AggregatedResults(
            config_name=config_name,
            num_runs=len(runs),
            avg_global_amota_mean=mean(global_amotas),
            avg_global_amota_std=stdev(global_amotas) if len(global_amotas) > 1 else 0,
            avg_cis_amota_mean=mean(cis_amotas),
            avg_cis_amota_std=stdev(cis_amotas) if len(cis_amotas) > 1 else 0,
            avg_cav_amota_mean=mean(cav_amotas),
            avg_cav_amota_std=stdev(cav_amotas) if len(cav_amotas) > 1 else 0,
            avg_global_amotp_mean=mean(global_amotps),
            avg_global_amotp_std=stdev(global_amotps) if len(global_amotps) > 1 else 0,
            avg_cis_amotp_mean=mean(cis_amotps),
            avg_cis_amotp_std=stdev(cis_amotps) if len(cis_amotps) > 1 else 0,
            avg_cav_amotp_mean=mean(cav_amotps),
            avg_cav_amotp_std=stdev(cav_amotps) if len(cav_amotps) > 1 else 0,
            avg_hota_mean=mean(hotas),
            avg_hota_std=stdev(hotas) if len(hotas) > 1 else 0,
            avg_deta_mean=mean(detas),
            avg_deta_std=stdev(detas) if len(detas) > 1 else 0,
            avg_assa_mean=mean(assas),
            avg_assa_std=stdev(assas) if len(assas) > 1 else 0,
            avg_perception_anomalies_mean=mean(anomaly_counts),
            avg_perception_anomalies_std=stdev(anomaly_counts) if len(anomaly_counts) > 1 else 0,
            total_anomaly_detections=total_anomalies,
            runs=runs,
        )
    
    def _run_suite_parallel(self, suite: ExperimentSuite, configs_to_run: List[Tuple[str, Any]], 
                            total_runs: int, parallel: int, verbose: bool) -> Dict[str, AggregatedResults]:
        """
        Run experiment suite with parallel execution.
        
        Args:
            suite: ExperimentSuite defining all configurations
            configs_to_run: List of (config_name, experiment_config) tuples
            total_runs: Total number of runs
            parallel: Number of parallel workers
            verbose: Whether to show progress
            
        Returns:
            Dictionary mapping config names to aggregated results
        """
        all_results: Dict[str, AggregatedResults] = {}
        
        # Use a Manager for shared progress dictionary
        manager = multiprocessing.Manager()
        shared_progress = manager.dict()
        
        # Result keys: expand with triple_fusion_output_names when present (one run -> 3 result buckets)
        result_keys = []
        for name, exp in configs_to_run:
            out_names = getattr(exp, 'triple_fusion_output_names', None)
            result_keys.extend(out_names if out_names else [name])

        # Build list of all jobs to run
        jobs = []
        for config_name, experiment_config in configs_to_run:
            config_dir = os.path.join(self.current_experiment_dir, config_name.replace(' ', '_'))
            os.makedirs(config_dir, exist_ok=True)
            output_names = getattr(experiment_config, 'triple_fusion_output_names', None)
            if output_names:
                for out_name in output_names:
                    os.makedirs(os.path.join(self.current_experiment_dir, out_name.replace(' ', '_')), exist_ok=True)

            for run_id in range(1, suite.runs_per_config + 1):
                run_config = self._prepare_run_config(experiment_config.config, suite, run_id)
                run_config["_simulation_label"] = f"sim_{config_name}_{run_id}_{int(time.time() * 1000) % 100000}"
                jobs.append((config_name, run_id, run_config, config_dir, shared_progress, output_names))
        
        print(f"  Starting {len(jobs)} simulations with {parallel} parallel workers...\n")
        
        # Track results by config (one bucket per output name, so 39 for GPEM penetration triple fusion)
        results_by_config: Dict[str, List[RunResult]] = {k: [] for k in result_keys}
        completed = 0
        failed = 0
        start_time = time.time()
        
        # Track active jobs for display
        active_jobs: Dict[Any, Tuple[str, int, float]] = {}  # future -> (config_name, run_id, start_time)
        last_lines_printed = 0  # Track number of lines printed for clearing
        progress_lock = threading.Lock()
        
        # Progress display helper
        def update_progress_display():
            nonlocal last_lines_printed
            with progress_lock:
                elapsed = time.time() - start_time
                # Calculate overall progress bar
                progress_pct = (completed + failed) / total_runs if total_runs > 0 else 0
                bar_width = 25
                filled = int(bar_width * progress_pct)
                overall_bar = '█' * filled + '░' * (bar_width - filled)
                
                # Estimate time remaining
                if completed + failed > 0:
                    avg_time = elapsed / (completed + failed)
                    remaining = avg_time * (total_runs - completed - failed)
                    eta_str = f"ETA: {remaining/60:.1f}m" if remaining > 60 else f"ETA: {remaining:.0f}s"
                else:
                    eta_str = "ETA: calculating..."
                
                # Build worker status with individual progress bars
                worker_lines = []
                # Only show up to 'parallel' workers from active_jobs
                active_list = list(active_jobs.items())[:parallel]
                
                # Show active workers
                for i, (future, (cfg, rid, job_start)) in enumerate(active_list):
                    job_elapsed = time.time() - job_start
                    short_name = cfg[:12] + '..' if len(cfg) > 14 else cfg
                    
                    # Get actual progress from shared dictionary
                    job_key = f"{cfg}#{rid}"
                    current_step, total_steps = shared_progress.get(job_key, (0, 100))
                    
                    # Calculate worker progress bar
                    bar_width_worker = 10
                    if total_steps > 0:
                        worker_pct = current_step / total_steps
                        filled_worker = int(bar_width_worker * worker_pct)
                        bar_str = '█' * filled_worker + '·' * (bar_width_worker - filled_worker)
                    else:
                        bar_str = '·' * bar_width_worker

                    
                    worker_lines.append(f"    W{i+1:<2} {short_name:14s} #{rid:<2} [{bar_str}] {job_elapsed:5.1f}s")

                
                # Clear previous lines
                if last_lines_printed > 0:
                    # Move cursor to beginning of first line we printed and clear everything below
                    sys.stdout.write(f"\r\x1b[{last_lines_printed-1}A\x1b[J")
                
                # Build final output
                output = []
                output.append(f"  [{overall_bar}] {completed+failed}/{total_runs} | ✓{completed} ✗{failed} | {eta_str}")
                
                if worker_lines:
                    output.extend(worker_lines)
                
                # Print everything
                sys.stdout.write("\n".join(output))
                last_lines_printed = len(output)
                sys.stdout.flush()

        # Background refresh thread to keep animations smooth
        def refresh_loop():
            while not _shutdown_requested and (completed + failed) < total_runs:
                update_progress_display()
                time.sleep(0.1)
        
        # Run jobs in parallel
        global _shutdown_requested
        _shutdown_requested = False
        
        # Register signal handler for clean Ctrl+C
        original_sigint = signal.signal(signal.SIGINT, _signal_handler)
        
        executor = ProcessPoolExecutor(max_workers=parallel)
        try:
            # Submit all jobs and track them
            futures = {}
            for job in jobs:
                if _shutdown_requested:
                    break
                future = executor.submit(_run_simulation_worker, job)
                futures[future] = job
                with progress_lock:
                    active_jobs[future] = (job[0], job[1], time.time())
            
            # Start refresh thread
            refresher = threading.Thread(target=refresh_loop, daemon=True)
            refresher.start()
            
            # Initial display
            update_progress_display()
            
            # Process results as they complete
            for future in as_completed(futures):
                if _shutdown_requested:
                    break
                    
                job = futures[future]
                config_name, run_id = job[0], job[1]
                config_dir = job[3]
                output_names = job[5] if len(job) > 5 else None
                
                # Remove from active jobs
                with progress_lock:
                    if future in active_jobs:
                        del active_jobs[future]
                
                try:
                    result_tuple = future.result(timeout=3600)  # 1 hour timeout per job
                    cfg_name, r_id, sim_result, wall_time, error = result_tuple
                    
                    if error:
                        failed += 1
                        # Clear progress before printing error
                        with progress_lock:
                            if last_lines_printed > 0:
                                sys.stdout.write(f"\r\x1b[{last_lines_printed-1}A\x1b[J")
                                last_lines_printed = 0
                        sys.stdout.write(f"  ❌ {cfg_name} run {r_id}: {error[:50]}...\n" if len(str(error)) > 50 else f"  ❌ {cfg_name} run {r_id}: {error}\n")
                    else:
                        if output_names and sim_result.get("triple_global_metrics"):
                            # One run produced three synced outputs (static, gpem_linear, gpem_quadratic)
                            triple_results = self._process_triple_sim_result(sim_result, r_id, output_names)
                            for i, out_name in enumerate(output_names):
                                result_i = triple_results[i]
                                result_i.wall_time_seconds = wall_time
                                results_by_config[out_name].append(result_i)
                                out_dir = os.path.join(self.current_experiment_dir, out_name.replace(' ', '_'))
                                self._save_run_result(out_dir, result_i)
                        else:
                            result = self._process_sim_result(cfg_name, r_id, sim_result)
                            result.wall_time_seconds = wall_time
                            results_by_config[cfg_name].append(result)
                            self._save_run_result(config_dir, result)
                        completed += 1
                        
                except Exception as e:
                    failed += 1
                    # Clear progress before printing error
                    with progress_lock:
                        if last_lines_printed > 0:
                            sys.stdout.write(f"\r\x1b[{last_lines_printed-1}A\x1b[J")
                            last_lines_printed = 0
                    sys.stdout.write(f"  ❌ {config_name} run {run_id}: {str(e)[:50]}\n")
                
                # Display update after each completion (the thread also updates)
                update_progress_display()
                
        except KeyboardInterrupt:
            _shutdown_requested = True
            print("\n\n⚠️  Interrupted! Cancelling pending jobs...")
            
        finally:
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_sigint)
            
            # Cancel all pending futures
            if _shutdown_requested:
                print("  Cancelling pending futures...")
                for future in futures:
                    future.cancel()
            
            # Shutdown executor - terminate workers immediately if interrupted
            print("  Shutting down worker pool...")
            try:
                if _shutdown_requested:
                    # Force terminate
                    executor.shutdown(wait=False, cancel_futures=True)
                    # Give processes a moment to terminate
                    time.sleep(0.5)
                else:
                    executor.shutdown(wait=True, cancel_futures=False)
            except TypeError:
                # Python < 3.9 doesn't support cancel_futures
                executor.shutdown(wait=False if _shutdown_requested else True)
            
            if _shutdown_requested:
                # Kill any orphan SUMO processes
                _kill_orphan_sumo_processes()
                print("  ✓ Shutdown complete.\n")
                
            # Shut down the manager to clean up shared resources
            try:
                manager.shutdown()
            except:
                pass
                
            if _shutdown_requested:
                return all_results
        
        # Final newline after progress bar
        print("\n")
        
        # Aggregate results for each config (result_keys includes triple-fusion output names)
        print("  Aggregating results...")
        for config_name in result_keys:
            config_dir = os.path.join(self.current_experiment_dir, config_name.replace(' ', '_'))
            run_results = results_by_config[config_name]
            
            # Sort by run_id for consistent ordering
            run_results.sort(key=lambda r: r.run_id)
            
            if run_results:
                aggregated = self._aggregate_results(config_name, run_results)
                all_results[config_name] = aggregated
                self._save_aggregated_results(config_dir, aggregated)
                print(f"  📊 {config_name}: AMOTA = {aggregated.avg_global_amota_mean:.4f} "
                      f"(±{aggregated.avg_global_amota_std:.4f}) | "
                      f"HOTA = {aggregated.avg_hota_mean:.4f} "
                      f"(DetA={aggregated.avg_deta_mean:.4f}, AssA={aggregated.avg_assa_mean:.4f})")
        
        # Save summary
        self._save_summary(suite, all_results)
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Experiment Suite Complete!")
        print(f"  ✓ Succeeded: {completed}")
        print(f"  ✗ Failed: {failed}")
        print(f"  ⏱ Total time: {total_time/60:.1f} minutes")
        print(f"  📁 Results: {self.current_experiment_dir}")
        print(f"{'='*60}\n")
        
        return all_results
    
    def _process_triple_sim_result(self, sim_result: Dict, run_id: int, output_names: List[str]) -> List[RunResult]:
        """Process triple_global_metrics from one run into RunResults (baseline, static, gpem_linear, gpem_quadratic per filter)."""
        cav_metrics = sim_result.get("total_cav_metrics", {})
        cis_metrics = sim_result.get("total_cis_metrics", {})
        triple_metrics = sim_result.get("triple_global_metrics", {})
        triple_amota_counts = sim_result.get("triple_global_amota_frames_count", {})
        triple_amotp_counts = sim_result.get("triple_global_amotp_frames_count", {})
        triple_hota = sim_result.get("triple_global_hota", {})
        method_keys = ["baseline", "static", "gpem_linear", "gpem_quadratic",
                       "ci_baseline", "ci_static", "ci_gpem_linear", "ci_gpem_quadratic",
                       "akf_baseline", "akf_static", "akf_gpem_linear", "akf_gpem_quadratic",
                       "pf_baseline", "pf_static", "pf_gpem_linear", "pf_gpem_quadratic"]
        results = []
        for i, (out_name, key) in enumerate(zip(output_names, method_keys)):
            global_metrics = triple_metrics.get(key, {})
            amota_frames = triple_amota_counts.get(key, 1)
            amotp_frames = triple_amotp_counts.get(key, 1)
            avg_global_amota = global_metrics.get("amota", 0) / max(amota_frames, 1)
            avg_global_amotp = global_metrics.get("amotp", 0) / max(amotp_frames, 1)
            hota_result = triple_hota.get(key, {})
            results.append(RunResult(
                config_name=out_name,
                run_id=run_id,
                total_steps=sim_result.get("iterations", 0),
                recorded_steps=sim_result.get("recorded_steps", sim_result.get("iterations", 0)),
                wall_time_seconds=0,
                cav_metrics=cav_metrics,
                cis_metrics=cis_metrics,
                global_metrics=global_metrics,
                avg_cav_amota=cav_metrics.get("amota", 0) / max(sim_result.get("cav_amota_frames_count", 1), 1),
                avg_cis_amota=cis_metrics.get("amota", 0) / max(sim_result.get("cis_amota_frames_count", 1), 1),
                avg_global_amota=avg_global_amota,
                avg_cav_amotp=cav_metrics.get("amotp", 0) / max(sim_result.get("cav_amotp_frames_count", 1), 1),
                avg_cis_amotp=cis_metrics.get("amotp", 0) / max(sim_result.get("cis_amotp_frames_count", 1), 1),
                avg_global_amotp=avg_global_amotp,
                avg_step_time=sim_result.get("avg_step_time", 0),
                avg_cis_detection_time=sim_result.get("avg_cis_detection_time", 0),
                avg_cav_detection_time=sim_result.get("avg_cav_detection_time", 0),
                avg_global_fusion_time=sim_result.get("avg_global_fusion_time", 0),
                hota=hota_result.get("hota", 0.0),
                deta=hota_result.get("deta", 0.0),
                assa=hota_result.get("assa", 0.0),
            ))
        return results

    def _process_sim_result(self, config_name: str, run_id: int, sim_result: Dict) -> RunResult:
        """Process simulation result dictionary into RunResult."""
        cav_metrics = sim_result.get("total_cav_metrics", {})
        cis_metrics = sim_result.get("total_cis_metrics", {})
        global_metrics = sim_result.get("total_global_metrics", {})
        
        cav_amota_frames = sim_result.get("cav_amota_frames_count", 1)
        cis_amota_frames = sim_result.get("cis_amota_frames_count", 1)
        global_amota_frames = sim_result.get("global_amota_frames_count", 1)
        
        avg_cav_amota = cav_metrics.get("amota", 0) / max(cav_amota_frames, 1)
        avg_cis_amota = cis_metrics.get("amota", 0) / max(cis_amota_frames, 1)
        avg_global_amota = global_metrics.get("amota", 0) / max(global_amota_frames, 1)
        
        # Extract perception scoring results
        perception_scores = sim_result.get("perception_scores", {})
        perception_anomalies = sim_result.get("perception_anomalies", {})
        num_anomalies = sum(1 for a in perception_anomalies.values() if a)
        
        return RunResult(
            config_name=config_name,
            run_id=run_id,
            total_steps=sim_result.get("iterations", 0),
            recorded_steps=sim_result.get("recorded_steps", sim_result.get("iterations", 0)),
            wall_time_seconds=0,
            cav_metrics=cav_metrics,
            cis_metrics=cis_metrics,
            global_metrics=global_metrics,
            avg_cav_amota=avg_cav_amota,
            avg_cis_amota=avg_cis_amota,
            avg_global_amota=avg_global_amota,
            avg_step_time=sim_result.get("avg_step_time", 0),
            avg_cis_detection_time=sim_result.get("avg_cis_detection_time", 0),
            avg_cav_detection_time=sim_result.get("avg_cav_detection_time", 0),
            avg_global_fusion_time=sim_result.get("avg_global_fusion_time", 0),
            hota=sim_result.get("global_hota", {}).get("hota", 0.0),
            deta=sim_result.get("global_hota", {}).get("deta", 0.0),
            assa=sim_result.get("global_hota", {}).get("assa", 0.0),
            perception_scores=perception_scores,
            perception_anomalies=perception_anomalies,
            num_perception_anomalies=num_anomalies,
        )

    def _save_suite_config(self, suite: ExperimentSuite):
        """Save the experiment suite configuration."""
        config_path = os.path.join(self.current_experiment_dir, "suite_config.json")
        
        suite_dict = {
            "name": suite.name,
            "description": suite.description,
            "runs_per_config": suite.runs_per_config,
            "warmup_steps": suite.warmup_steps,
            "record_steps": suite.record_steps,
            "max_steps": suite.max_steps,
            "error_injection_start": suite.error_injection_start,
            "error_injection_end": suite.error_injection_end,
            "baseline": {
                "name": suite.baseline.name,
                "description": suite.baseline.description,
            } if suite.baseline else None,
            "variants": [
                {"name": v.name, "description": v.description}
                for v in suite.variants
            ],
        }
        
        with open(config_path, 'w') as f:
            json.dump(suite_dict, f, indent=2)
    
    def _save_run_result(self, config_dir: str, result: RunResult):
        """Save individual run result to CSV."""
        run_path = os.path.join(config_dir, f"run_{result.run_id:02d}.csv")
        
        with open(run_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            writer.writerow(["config_name", result.config_name])
            writer.writerow(["run_id", result.run_id])
            writer.writerow(["total_steps", result.total_steps])
            writer.writerow(["recorded_steps", result.recorded_steps])
            writer.writerow(["wall_time_seconds", result.wall_time_seconds])
            writer.writerow(["avg_cav_amota", result.avg_cav_amota])
            writer.writerow(["avg_cis_amota", result.avg_cis_amota])
            writer.writerow(["avg_global_amota", result.avg_global_amota])
            
            # Write detailed metrics
            for key, value in result.cav_metrics.items():
                writer.writerow([f"cav_{key}", value])
            for key, value in result.cis_metrics.items():
                writer.writerow([f"cis_{key}", value])
            for key, value in result.global_metrics.items():
                writer.writerow([f"global_{key}", value])
    
    def _save_aggregated_results(self, config_dir: str, aggregated: AggregatedResults):
        """Save aggregated results to JSON."""
        agg_path = os.path.join(config_dir, "aggregated.json")
        
        agg_dict = {
            "config_name": aggregated.config_name,
            "num_runs": aggregated.num_runs,
            "avg_global_amota_mean": aggregated.avg_global_amota_mean,
            "avg_global_amota_std": aggregated.avg_global_amota_std,
            "avg_global_amotp_mean": aggregated.avg_global_amotp_mean,
            "avg_global_amotp_std": aggregated.avg_global_amotp_std,
            "avg_cis_amota_mean": aggregated.avg_cis_amota_mean,
            "avg_cis_amota_std": aggregated.avg_cis_amota_std,
            "avg_cis_amotp_mean": aggregated.avg_cis_amotp_mean,
            "avg_cis_amotp_std": aggregated.avg_cis_amotp_std,
            "avg_cav_amota_mean": aggregated.avg_cav_amota_mean,
            "avg_cav_amota_std": aggregated.avg_cav_amota_std,
            "avg_cav_amotp_mean": aggregated.avg_cav_amotp_mean,
            "avg_cav_amotp_std": aggregated.avg_cav_amotp_std,
            # Perception scoring
            "avg_perception_anomalies_mean": aggregated.avg_perception_anomalies_mean,
            "avg_perception_anomalies_std": aggregated.avg_perception_anomalies_std,
            "total_anomaly_detections": aggregated.total_anomaly_detections,
        }
        
        with open(agg_path, 'w') as f:
            json.dump(agg_dict, f, indent=2)
    
    def _save_summary(self, suite: ExperimentSuite, all_results: Dict[str, AggregatedResults]):
        """Save overall summary of the experiment suite."""
        summary_path = os.path.join(self.current_experiment_dir, "summary.json")
        
        summary = {
            "suite_name": suite.name,
            "completed_at": datetime.now().isoformat(),
            "total_runs": sum(r.num_runs for r in all_results.values()),
            "results": {}
        }
        
        for config_name, agg in all_results.items():
            summary["results"][config_name] = {
                "num_runs": agg.num_runs,
                "avg_global_amota_mean": agg.avg_global_amota_mean,
                "avg_global_amota_std": agg.avg_global_amota_std,
                "avg_global_amotp_mean": agg.avg_global_amotp_mean,
                "avg_global_amotp_std": agg.avg_global_amotp_std,
                "avg_cis_amota_mean": agg.avg_cis_amota_mean,
                "avg_cis_amota_std": agg.avg_cis_amota_std,
                "avg_cis_amotp_mean": agg.avg_cis_amotp_mean,
                "avg_cis_amotp_std": agg.avg_cis_amotp_std,
                "avg_cav_amota_mean": agg.avg_cav_amota_mean,
                "avg_cav_amota_std": agg.avg_cav_amota_std,
                "avg_cav_amotp_mean": agg.avg_cav_amotp_mean,
                "avg_cav_amotp_std": agg.avg_cav_amotp_std,
                # HOTA metrics
                "avg_hota_mean": agg.avg_hota_mean,
                "avg_hota_std": agg.avg_hota_std,
                "avg_deta_mean": agg.avg_deta_mean,
                "avg_deta_std": agg.avg_deta_std,
                "avg_assa_mean": agg.avg_assa_mean,
                "avg_assa_std": agg.avg_assa_std,
                # Perception scoring
                "avg_perception_anomalies_mean": agg.avg_perception_anomalies_mean,
                "avg_perception_anomalies_std": agg.avg_perception_anomalies_std,
                "total_anomaly_detections": agg.total_anomaly_detections,
            }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Build baseline lookup: for each config, find its matching baseline
        # e.g. "gpem_linear_av_10.0pct" -> "baseline_av_10.0pct"
        # e.g. "ci_static_av_10.0pct" -> "ci_baseline_av_10.0pct"
        # e.g. "akf_gpem_quadratic_av_10.0pct" -> "akf_baseline_av_10.0pct"
        def _find_baseline_name(config_name: str) -> str:
            """Find the matching baseline config name for a given config."""
            if config_name.startswith("pf_"):
                suffix = config_name.split("pf_", 1)[1]
                for prefix in ("static_av_", "static_cov_av_", "gpem_linear_av_", "gpem_quadratic_av_", "baseline_av_"):
                    if suffix.startswith(prefix):
                        return "pf_baseline_av_" + suffix.split("av_", 1)[1]
                for prefix in ("static_", "gpem_linear_", "gpem_quadratic_", "baseline_"):
                    if f"_{prefix}" in config_name:
                        return config_name.replace(f"_{prefix}", "_baseline_").replace("_baseline_baseline_", "_baseline_")
            elif config_name.startswith("akf_"):
                suffix = config_name.split("akf_", 1)[1]
                # Replace the method part with "baseline"
                for prefix in ("static_av_", "static_cov_av_", "gpem_linear_av_", "gpem_quadratic_av_", "baseline_av_"):
                    if suffix.startswith(prefix):
                        return "akf_baseline_av_" + suffix.split("av_", 1)[1]
                # dist sweep format
                for prefix in ("static_", "gpem_linear_", "gpem_quadratic_", "baseline_"):
                    if f"_{prefix}" in config_name:
                        return config_name.replace(f"_{prefix}", "_baseline_").replace("_baseline_baseline_", "_baseline_")
            elif config_name.startswith("ci_"):
                suffix = config_name.split("ci_", 1)[1]
                for prefix in ("static_av_", "static_cov_av_", "gpem_linear_av_", "gpem_quadratic_av_", "baseline_av_"):
                    if suffix.startswith(prefix):
                        return "ci_baseline_av_" + suffix.split("av_", 1)[1]
                for prefix in ("static_", "gpem_linear_", "gpem_quadratic_", "baseline_"):
                    if f"_{prefix}" in config_name:
                        return config_name.replace(f"_{prefix}", "_baseline_").replace("_baseline_baseline_", "_baseline_")
            else:
                # Kalman filter group (no prefix)
                for prefix in ("static_cov_av_", "gpem_linear_av_", "gpem_quadratic_av_", "baseline_av_"):
                    if config_name.startswith(prefix):
                        return "baseline_av_" + config_name.split("av_", 1)[1]
                # dist sweep format: gpem_dist_sweep_step_0_static -> gpem_dist_sweep_step_0_baseline
                if "_static" in config_name or "_gpem_linear" in config_name or "_gpem_quadratic" in config_name:
                    for method in ("_static", "_gpem_linear", "_gpem_quadratic"):
                        if config_name.endswith(method):
                            return config_name[:-len(method)] + "_baseline"
            return ""

        def _pct_improvement(value: float, baseline_value: float) -> str:
            """Calculate percentage improvement (higher AMOTA = better, lower AMOTP = better)."""
            if baseline_value == 0:
                return "N/A"
            pct = (value - baseline_value) / abs(baseline_value) * 100
            return f"{pct:+.2f}%"

        # Also save a CSV for easy viewing
        csv_path = os.path.join(self.current_experiment_dir, "summary.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Config", "Runs",
                "Global AMOTA (mean)", "Global AMOTA (std)", "AMOTA vs baseline",
                "Global AMOTP (mean)", "Global AMOTP (std)", "AMOTP vs baseline",
                "HOTA (mean)", "HOTA (std)", "HOTA vs baseline",
                "DetA (mean)", "DetA (std)",
                "AssA (mean)", "AssA (std)",
                "CIS AMOTA (mean)", "CIS AMOTA (std)",
                "CAV AMOTA (mean)", "CAV AMOTA (std)",
                "Anomalies (mean)", "Anomalies (std)", "Total Anomalies"
            ])
            for config_name, agg in all_results.items():
                baseline_name = _find_baseline_name(config_name)
                baseline_agg = all_results.get(baseline_name)
                amota_pct = _pct_improvement(agg.avg_global_amota_mean, baseline_agg.avg_global_amota_mean) if baseline_agg else ""
                amotp_pct = _pct_improvement(agg.avg_global_amotp_mean, baseline_agg.avg_global_amotp_mean) if baseline_agg else ""
                hota_pct = _pct_improvement(agg.avg_hota_mean, baseline_agg.avg_hota_mean) if baseline_agg else ""
                writer.writerow([
                    config_name, agg.num_runs,
                    f"{agg.avg_global_amota_mean:.4f}", f"{agg.avg_global_amota_std:.4f}", amota_pct,
                    f"{agg.avg_global_amotp_mean:.4f}", f"{agg.avg_global_amotp_std:.4f}", amotp_pct,
                    f"{agg.avg_hota_mean:.4f}", f"{agg.avg_hota_std:.4f}", hota_pct,
                    f"{agg.avg_deta_mean:.4f}", f"{agg.avg_deta_std:.4f}",
                    f"{agg.avg_assa_mean:.4f}", f"{agg.avg_assa_std:.4f}",
                    f"{agg.avg_cis_amota_mean:.4f}", f"{agg.avg_cis_amota_std:.4f}",
                    f"{agg.avg_cav_amota_mean:.4f}", f"{agg.avg_cav_amota_std:.4f}",
                    f"{agg.avg_perception_anomalies_mean:.2f}", f"{agg.avg_perception_anomalies_std:.2f}",
                    agg.total_anomaly_detections,
                ])


# Convenience functions for creating experiment suites
def create_sensor_comparison_suite(runs_per_config: int = 10) -> ExperimentSuite:
    """Create a standard sensor comparison experiment suite."""
    from config_templates import (
        get_perfect_config, 
        get_normal_sensors_config,
    )
    
    return ExperimentSuite(
        name="Sensor_Comparison",
        description="Compare different sensor configurations",
        runs_per_config=runs_per_config,
        warmup_steps=600,
        record_steps=6000,
        baseline=ExperimentConfig(
            name="perfect_sensors",
            config=get_perfect_config(),
            description="Perfect sensors with no noise or errors"
        ),
        variants=[
            ExperimentConfig(
                name="normal_sensors",
                config=get_normal_sensors_config(),
                description="Normal sensors with realistic noise"
            ),
        ]
    )


def create_pointpillars_comparison_suite(runs_per_config: int = 10) -> ExperimentSuite:
    """
    Create experiment suite comparing baseline (perfect) vs PointPillars OS1_128.
    
    This is the main comparison suite for evaluating the regression-tested
    PointPillars error model against a perfect baseline.
    
    Args:
        runs_per_config: Number of runs for each configuration (default: 10)
    
    Returns:
        ExperimentSuite configured for baseline vs PointPillars comparison
    """
    from config_templates import (
        get_perfect_config,
        get_pointpillars_os1_128_config,
    )
    
    return ExperimentSuite(
        name="PointPillars_OS1_128_vs_Baseline",
        description="Compare perfect baseline against regression-tested PointPillars OS1_128 LiDAR",
        runs_per_config=runs_per_config,
        warmup_steps=600,
        record_steps=6000,
        baseline=ExperimentConfig(
            name="baseline",
            config=get_perfect_config(),
            description="Perfect sensors with no noise or errors"
        ),
        variants=[
            ExperimentConfig(
                name="pointpillars_os1_128",
                config=get_pointpillars_os1_128_config(),
                description="PointPillars with OS1_128 LiDAR - regression tested error model"
            ),
        ]
    )


def create_error_injection_suite(
    base_config: Dict,
    error_types: List[str],
    error_levels: List[int],
    runs_per_config: int = 10
) -> ExperimentSuite:
    """Create an error injection experiment suite."""
    variants = []
    
    for error_type in error_types:
        for error_level in error_levels:
            config = copy.deepcopy(base_config)
            config["error_type"] = error_type
            config["error_level"] = error_level
            
            variants.append(ExperimentConfig(
                name=f"{error_type}_level_{error_level}",
                config=config,
                description=f"Error type {error_type} at level {error_level}"
            ))
    
    return ExperimentSuite(
        name="Error_Injection",
        description="Test system robustness under various error conditions",
        runs_per_config=runs_per_config,
        warmup_steps=600,
        record_steps=6000,
        error_injection_start=600,  # Start injecting after warmup
        baseline=ExperimentConfig(
            name="no_errors",
            config=base_config,
            description="Baseline with no error injection"
        ),
        variants=variants
    )


def create_comprehensive_error_sweep_suite(
    runs_per_config: int = 10,
    extrinsics_degrees: List[int] = None,
    localization_percentages: List[int] = None,
    malicious_percentages: List[int] = None,
    warmup_steps: int = 600,
    record_steps: int = 6000,
    use_trust_scoring: bool = True,
) -> ExperimentSuite:
    """
    Create a comprehensive error severity sweep experiment suite.
    
    10% of CAVs have errors (fixed), severity varies by error type.
    Uses PointPillars OS1_128 as the base sensor configuration.
    
    Severity levels (error_level parameter):
    - SINGLE_SENSOR_EXTRINSICS: 1-10 degrees of angular error
      (also proportional position error: 1deg = 0.1m, 10deg = 1.0m)
    - LOCALIZATION: 10-100% of max error (max = 1m position, 10deg angle)
    - MALICIOUS_*: 10-100% probability per frame
    
    Note: MULTI_SENSOR_EXTRINSICS is skipped since PointPillars uses a single sensor.
    
    Args:
        runs_per_config: Number of runs per configuration (default: 10)
        extrinsics_degrees: List of degree values for extrinsics (default: 1-10)
        localization_percentages: List of percentages for localization (default: 10-100 by 10)
        malicious_percentages: List of percentages for malicious (default: 10-100 by 10)
        warmup_steps: Steps before recording/error injection starts
        record_steps: Number of steps to record
    
    Returns:
        ExperimentSuite with all error type/severity combinations
    """
    from config_templates import get_pointpillars_os1_128_config
    
    # Default severity levels
    if extrinsics_degrees is None:
        extrinsics_degrees = list(range(1, 11))  # 1-10 degrees
    if localization_percentages is None:
        localization_percentages = list(range(10, 101, 10))  # 10, 20, ..., 100%
    if malicious_percentages is None:
        malicious_percentages = list(range(10, 101, 10))  # 10, 20, ..., 100%
    
    variants = []
    
    # Extrinsics: 1-10 degrees of error
    for deg in extrinsics_degrees:
        config = get_pointpillars_os1_128_config()
        config["error_type"] = "SINGLE_SENSOR_EXTRINSICS"
        config["error_level"] = deg  # Severity: degrees of error
        config["use_trust_scoring"] = use_trust_scoring
        
        variants.append(ExperimentConfig(
            name=f"extrinsics_{deg}deg",
            config=config,
            description=f"Sensor calibration error: {deg} degrees (10% of CAVs)"
        ))
    
    # Localization: 10-100% of max error
    for pct in localization_percentages:
        config = get_pointpillars_os1_128_config()
        config["error_type"] = "LOCALIZATION"
        config["error_level"] = pct  # Severity: percentage of max error
        config["use_trust_scoring"] = use_trust_scoring
        
        variants.append(ExperimentConfig(
            name=f"localization_{pct}pct",
            config=config,
            description=f"Localization error: {pct}% of max (10% of CAVs)"
        ))
    
    # Malicious Removal: 10-100% probability per frame
    for pct in malicious_percentages:
        config = get_pointpillars_os1_128_config()
        config["error_type"] = "MALICIOUS_REMOVAL"
        config["error_level"] = pct  # Severity: probability percentage
        config["use_trust_scoring"] = use_trust_scoring
        
        variants.append(ExperimentConfig(
            name=f"mal_remove_{pct}pct",
            config=config,
            description=f"Malicious removal: {pct}% prob/frame (10% of CAVs)"
        ))
    
    # Malicious Addition: 10-100% probability per frame
    for pct in malicious_percentages:
        config = get_pointpillars_os1_128_config()
        config["error_type"] = "MALICIOUS_ADDITION"
        config["error_level"] = pct  # Severity: probability percentage
        config["use_trust_scoring"] = use_trust_scoring
        
        variants.append(ExperimentConfig(
            name=f"mal_add_{pct}pct",
            config=config,
            description=f"Malicious addition: {pct}% prob/frame (10% of CAVs)"
        ))
    
    # Malicious Convoy: 10-100% probability per frame
    for pct in malicious_percentages:
        config = get_pointpillars_os1_128_config()
        config["error_type"] = "MALICIOUS_CONVOY"
        config["error_level"] = pct  # Severity: probability percentage
        config["use_trust_scoring"] = use_trust_scoring
        
        variants.append(ExperimentConfig(
            name=f"mal_convoy_{pct}pct",
            config=config,
            description=f"Malicious convoy: {pct}% prob/frame (10% of CAVs)"
        ))
    
    # Baseline: PointPillars with no errors
    baseline_config = get_pointpillars_os1_128_config()
    baseline_config["error_type"] = "None"
    baseline_config["error_level"] = 0
    baseline_config["use_trust_scoring"] = use_trust_scoring
    
    return ExperimentSuite(
        name="Error_Severity_Sweep",
        description="Test error severity levels with 10% of CAVs affected, PointPillars OS1_128",
        runs_per_config=runs_per_config,
        warmup_steps=warmup_steps,
        record_steps=record_steps,
        error_injection_start=warmup_steps,  # Errors start after warmup
        baseline=ExperimentConfig(
            name="baseline_no_errors",
            config=baseline_config,
            description="PointPillars OS1_128 baseline with no injected errors"
        ),
        variants=variants
    )


def create_gpem_penetration_sweep_suite(
    runs_per_config: int = 10,
    av_injection_rates: List[float] = None,
    warmup_steps: int = 600,
    record_steps: int = 6000,
) -> ExperimentSuite:
    """
    Create GPEM (Generalized Parameterized Error Modeling) penetration sweep suite.

    One SUMO run per AV rate; three synced fusion outputs (static, linear, quadratic) per run.
    - A: Static covariance estimates (averaged)
    - B: GPEM linear (distance-dependent linear regression)
    - C: GPEM quadratic (distance-dependent quadratic regression)

    All use 33/33/33 detector split (BEV_FUSION, CENTERPOINT, DETR3D) and 50/50 localizers (CT_ICP, ORB_SLAM3).
    """
    from config_templates import get_pointpillars_os1_128_config

    if av_injection_rates is None:
        av_injection_rates = [1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

    variants = []
    det_alloc = [(1.0 / 3, "BEV_FUSION"), (1.0 / 3, "CENTERPOINT"), (1.0 / 3, "DETR3D")]
    loc_alloc = [(0.5, "CT_ICP"), (0.5, "ORB_SLAM3")]

    for rate in av_injection_rates:
        cfg = get_pointpillars_os1_128_config()
        cfg["cav_probability"] = rate / 100.0
        cfg["error_type"] = "None"
        cfg["error_level"] = 0
        cfg["use_trust_scoring"] = False
        cfg["use_gpem_model"] = False  # sensors use static for sampling; triple fusion overwrites cov per stream
        cfg["gpem_triple_fusion"] = True
        cfg["detector_allocation"] = det_alloc
        cfg["localizer_allocation"] = loc_alloc

        output_names = [
            f"baseline_av_{rate:.1f}pct",
            f"static_cov_av_{rate:.1f}pct",
            f"gpem_linear_av_{rate:.1f}pct",
            f"gpem_quadratic_av_{rate:.1f}pct",
            f"ci_baseline_av_{rate:.1f}pct",
            f"ci_static_av_{rate:.1f}pct",
            f"ci_gpem_linear_av_{rate:.1f}pct",
            f"ci_gpem_quadratic_av_{rate:.1f}pct",
            f"akf_baseline_av_{rate:.1f}pct",
            f"akf_static_av_{rate:.1f}pct",
            f"akf_gpem_linear_av_{rate:.1f}pct",
            f"akf_gpem_quadratic_av_{rate:.1f}pct",
            f"pf_baseline_av_{rate:.1f}pct",
            f"pf_static_av_{rate:.1f}pct",
            f"pf_gpem_linear_av_{rate:.1f}pct",
            f"pf_gpem_quadratic_av_{rate:.1f}pct",
        ]
        variants.append(ExperimentConfig(
            name=f"av_{rate:.1f}pct",
            config=cfg,
            description=f"Triple fusion (baseline/static/linear/quadratic) @ {rate:.1f}% AVs",
            triple_fusion_output_names=output_names,
        ))

    return ExperimentSuite(
        name="GPEM_Penetration_Sweep",
        description="One run per AV rate; four synced outputs (baseline, static, GPEM linear, GPEM quadratic) per run",
        runs_per_config=runs_per_config,
        warmup_steps=warmup_steps,
        record_steps=record_steps,
        error_injection_start=None,
        error_injection_end=None,
        baseline=None,
        variants=variants,
    )


def create_gpem_penetration_sweep_accurate_suite(
    runs_per_config: int = 10,
    av_injection_rates: List[float] = None,
    warmup_steps: int = 600,
    record_steps: int = 6000,
) -> ExperimentSuite:
    """
    Same as gpem_penetration_sweep but with accurate localizers (1/4 error).

    Tests whether reducing localization error lets GPEM's advantage emerge.
    Uses kiss_icp_accurate and orb_slam3_accurate instead of CT_ICP and ORB_SLAM3.
    """
    from config_templates import get_pointpillars_os1_128_config

    if av_injection_rates is None:
        av_injection_rates = [1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

    variants = []
    det_alloc = [(1.0 / 3, "BEV_FUSION"), (1.0 / 3, "CENTERPOINT"), (1.0 / 3, "DETR3D")]
    loc_alloc = [(0.5, "kiss_icp_accurate"), (0.5, "orb_slam3_accurate")]

    for rate in av_injection_rates:
        cfg = get_pointpillars_os1_128_config()
        cfg["cav_probability"] = rate / 100.0
        cfg["error_type"] = "None"
        cfg["error_level"] = 0
        cfg["use_trust_scoring"] = False
        cfg["use_gpem_model"] = False
        cfg["gpem_triple_fusion"] = True
        cfg["detector_allocation"] = det_alloc
        cfg["localizer_allocation"] = loc_alloc

        output_names = [
            f"baseline_av_{rate:.1f}pct",
            f"static_cov_av_{rate:.1f}pct",
            f"gpem_linear_av_{rate:.1f}pct",
            f"gpem_quadratic_av_{rate:.1f}pct",
            f"ci_baseline_av_{rate:.1f}pct",
            f"ci_static_av_{rate:.1f}pct",
            f"ci_gpem_linear_av_{rate:.1f}pct",
            f"ci_gpem_quadratic_av_{rate:.1f}pct",
            f"akf_baseline_av_{rate:.1f}pct",
            f"akf_static_av_{rate:.1f}pct",
            f"akf_gpem_linear_av_{rate:.1f}pct",
            f"akf_gpem_quadratic_av_{rate:.1f}pct",
            f"pf_baseline_av_{rate:.1f}pct",
            f"pf_static_av_{rate:.1f}pct",
            f"pf_gpem_linear_av_{rate:.1f}pct",
            f"pf_gpem_quadratic_av_{rate:.1f}pct",
        ]
        variants.append(ExperimentConfig(
            name=f"av_{rate:.1f}pct",
            config=cfg,
            description=f"Triple fusion (baseline/static/linear/quadratic) @ {rate:.1f}% AVs — accurate localizers (÷4)",
            triple_fusion_output_names=output_names,
        ))

    return ExperimentSuite(
        name="GPEM_Penetration_Sweep_Accurate",
        description="Same as penetration sweep but with 1/4 localization error (kiss_icp_accurate, orb_slam3_accurate)",
        runs_per_config=runs_per_config,
        warmup_steps=warmup_steps,
        record_steps=record_steps,
        error_injection_start=None,
        error_injection_end=None,
        baseline=None,
        variants=variants,
    )


def create_gpem_penetration_sweep_static_matching_suite(
    runs_per_config: int = 10,
    av_injection_rates: List[float] = None,
    warmup_steps: int = 600,
    record_steps: int = 6000,
) -> ExperimentSuite:
    """
    Same as gpem_penetration_sweep_accurate but with static matching covariance.

    All three streams use the same static covariance for Mahalanobis gating (matching),
    but their own GPEM/static covariance for Kalman fusion. This isolates whether
    matching or fusion is the bottleneck for GPEM.
    """
    from config_templates import get_pointpillars_os1_128_config

    if av_injection_rates is None:
        av_injection_rates = [1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

    variants = []
    det_alloc = [(1.0 / 3, "BEV_FUSION"), (1.0 / 3, "CENTERPOINT"), (1.0 / 3, "DETR3D")]
    loc_alloc = [(0.5, "kiss_icp_accurate"), (0.5, "orb_slam3_accurate")]

    for rate in av_injection_rates:
        cfg = get_pointpillars_os1_128_config()
        cfg["cav_probability"] = rate / 100.0
        cfg["error_type"] = "None"
        cfg["error_level"] = 0
        cfg["use_trust_scoring"] = False
        cfg["use_gpem_model"] = False
        cfg["gpem_triple_fusion"] = True
        cfg["use_static_matching"] = True
        cfg["detector_allocation"] = det_alloc
        cfg["localizer_allocation"] = loc_alloc

        output_names = [
            f"baseline_av_{rate:.1f}pct",
            f"static_cov_av_{rate:.1f}pct",
            f"gpem_linear_av_{rate:.1f}pct",
            f"gpem_quadratic_av_{rate:.1f}pct",
            f"ci_baseline_av_{rate:.1f}pct",
            f"ci_static_av_{rate:.1f}pct",
            f"ci_gpem_linear_av_{rate:.1f}pct",
            f"ci_gpem_quadratic_av_{rate:.1f}pct",
            f"akf_baseline_av_{rate:.1f}pct",
            f"akf_static_av_{rate:.1f}pct",
            f"akf_gpem_linear_av_{rate:.1f}pct",
            f"akf_gpem_quadratic_av_{rate:.1f}pct",
            f"pf_baseline_av_{rate:.1f}pct",
            f"pf_static_av_{rate:.1f}pct",
            f"pf_gpem_linear_av_{rate:.1f}pct",
            f"pf_gpem_quadratic_av_{rate:.1f}pct",
        ]
        variants.append(ExperimentConfig(
            name=f"av_{rate:.1f}pct",
            config=cfg,
            description=f"Triple fusion w/ static matching @ {rate:.1f}% AVs — accurate localizers (÷4)",
            triple_fusion_output_names=output_names,
        ))

    return ExperimentSuite(
        name="GPEM_Penetration_Sweep_Static_Matching",
        description="Static matching cov for all streams, GPEM cov for fusion only. Accurate localizers (÷4).",
        runs_per_config=runs_per_config,
        warmup_steps=warmup_steps,
        record_steps=record_steps,
        error_injection_start=None,
        error_injection_end=None,
        baseline=None,
        variants=variants,
    )


def create_gpem_distribution_sweep_suite(
    runs_per_config: int = 10,
    av_injection_rate: float = 20.0,
    warmup_steps: int = 600,
    record_steps: int = 6000,
) -> ExperimentSuite:
    """
    GPEM test: combined detector + localizer sweep, comparing static, linear, and quadratic GPEM.

    Single sweep of 8 steps (both dimensions together):
    - Detector: DETR3D 100% -> 95%, 90%, 80%, 70%, 60%, 50%, 33.33% (rest split BEV_FUSION/CENTERPOINT).
    - Localizer: ORB_SLAM3 100% -> 90%, 80%, 70%, 60%, 50%, 40%, 30%, 50%.
    CIS uses the same detector distribution (CIS has no localizer; packages still carry one for interface).
    Each step run three times: static, GPEM linear, GPEM quadratic.
    """
    from config_templates import get_pointpillars_os1_128_config

    # 8 combined (detector, localizer) steps for a smooth curve
    # DETR3D share (rest split equally BEV_FUSION / CENTERPOINT)
    detr3d_pcts = [1.0, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 1.0 / 3]
    # ORB_SLAM3 share
    orb_pcts = [1.0, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.50]  # last point 50% for even split

    sweep_steps = []
    for i in range(8):
        d = detr3d_pcts[i]
        if d >= 0.999:
            det_alloc = [(1.0, "DETR3D"), (0.0, "BEV_FUSION"), (0.0, "CENTERPOINT")]
            det_desc = "100% DETR3D"
        elif d <= 0.34:  # 1/3 even split
            det_alloc = [(1.0 / 3, "DETR3D"), (1.0 / 3, "BEV_FUSION"), (1.0 / 3, "CENTERPOINT")]
            det_desc = "33/33/33 detectors"
        else:
            other = (1.0 - d) / 2.0
            det_alloc = [(d, "DETR3D"), (other, "BEV_FUSION"), (other, "CENTERPOINT")]
            det_desc = f"{100 * d:.0f}% DETR3D, {100 * other:.0f}% BEV, {100 * other:.0f}% CP"
        o = orb_pcts[i]
        loc_alloc = [(1.0 - o, "CT_ICP"), (o, "ORB_SLAM3")]
        if o == 1.0:
            loc_desc = "100% ORB_SLAM3"
        elif o == 0.5:
            loc_desc = "50/50 localizers"
        else:
            loc_desc = f"{100 * (1 - o):.0f}% CT_ICP, {100 * o:.0f}% ORB"
        sweep_steps.append((det_alloc, loc_alloc, f"{det_desc}, {loc_desc}"))

    variants = []
    for step_idx, (det_alloc, loc_alloc, step_desc) in enumerate(sweep_steps):
        config = get_pointpillars_os1_128_config()
        config["cav_probability"] = av_injection_rate / 100.0
        config["error_type"] = "None"
        config["error_level"] = 0
        config["use_trust_scoring"] = False
        config["use_gpem_model"] = False
        config["gpem_triple_fusion"] = True
        config["detector_allocation"] = det_alloc
        config["localizer_allocation"] = loc_alloc
        output_names = [
            f"gpem_dist_sweep_step_{step_idx}_baseline",
            f"gpem_dist_sweep_step_{step_idx}_static",
            f"gpem_dist_sweep_step_{step_idx}_gpem_linear",
            f"gpem_dist_sweep_step_{step_idx}_gpem_quadratic",
            f"gpem_dist_sweep_step_{step_idx}_ci_baseline",
            f"gpem_dist_sweep_step_{step_idx}_ci_static",
            f"gpem_dist_sweep_step_{step_idx}_ci_gpem_linear",
            f"gpem_dist_sweep_step_{step_idx}_ci_gpem_quadratic",
            f"gpem_dist_sweep_step_{step_idx}_akf_baseline",
            f"gpem_dist_sweep_step_{step_idx}_akf_static",
            f"gpem_dist_sweep_step_{step_idx}_akf_gpem_linear",
            f"gpem_dist_sweep_step_{step_idx}_akf_gpem_quadratic",
            f"gpem_dist_sweep_step_{step_idx}_pf_baseline",
            f"gpem_dist_sweep_step_{step_idx}_pf_static",
            f"gpem_dist_sweep_step_{step_idx}_pf_gpem_linear",
            f"gpem_dist_sweep_step_{step_idx}_pf_gpem_quadratic",
        ]
        variants.append(ExperimentConfig(
            name=f"gpem_dist_sweep_step_{step_idx}",
            config=config,
            description=f"Step {step_idx} (triple fusion): {step_desc}",
            triple_fusion_output_names=output_names,
        ))

    return ExperimentSuite(
        name="GPEM_Distribution_Sweep",
        description="One run per step; four synced outputs (baseline, static, GPEM linear, GPEM quadratic) per run",
        runs_per_config=runs_per_config,
        warmup_steps=warmup_steps,
        record_steps=record_steps,
        error_injection_start=None,
        error_injection_end=None,
        baseline=None,
        variants=variants,
    )


def create_trust_comparison_suite(
    runs_per_config: int = 10,
    extrinsics_degrees: List[int] = None,
    localization_percentages: List[int] = None,
    malicious_percentages: List[int] = None,
    high_miss_percentages: List[int] = None,
    lag_steps: List[int] = None,
    warmup_steps: int = 600,
    record_steps: int = 6000,
) -> ExperimentSuite:
    """
    Create a suite that compares trust scoring ON vs OFF for each error type.
    
    Each error configuration is run twice:
    - Once with trust scoring enabled (suffix: _trust)
    - Once with trust scoring disabled (suffix: _notrust)
    
    This allows direct comparison of how trust scoring affects error detection.
    """
    from config_templates import get_pointpillars_os1_128_config
    
    # Default severity levels (use single values for faster comparison)
    if extrinsics_degrees is None:
        extrinsics_degrees = [5]  # 5 degrees
    if localization_percentages is None:
        localization_percentages = [50]  # 50%
    if malicious_percentages is None:
        malicious_percentages = [50]  # 50%
    if high_miss_percentages is None:
        high_miss_percentages = [50]  # 50% additional miss rate
    if lag_steps is None:
        lag_steps = [5]  # 5 steps = 0.5 seconds delay
    
    variants = []
    
    # Helper to create paired configs (trust ON and OFF)
    def add_paired_configs(name: str, error_type: str, error_level: int, description: str):
        for use_trust in [True, False]:
            config = get_pointpillars_os1_128_config()
            config["error_type"] = error_type
            config["error_level"] = error_level
            config["use_trust_scoring"] = use_trust
            
            suffix = "trust" if use_trust else "notrust"
            variants.append(ExperimentConfig(
                name=f"{name}_{suffix}",
                config=config,
                description=f"{description} [trust={'ON' if use_trust else 'OFF'}]"
            ))
    
    # Extrinsics errors
    for deg in extrinsics_degrees:
        add_paired_configs(
            name=f"extrinsics_{deg}deg",
            error_type="SINGLE_SENSOR_EXTRINSICS",
            error_level=deg,
            description=f"Extrinsics {deg}° (10% CAVs)"
        )
    
    # Localization errors
    for pct in localization_percentages:
        add_paired_configs(
            name=f"localization_{pct}pct",
            error_type="LOCALIZATION",
            error_level=pct,
            description=f"Localization {pct}% (10% CAVs)"
        )
    
    # Malicious Removal
    for pct in malicious_percentages:
        add_paired_configs(
            name=f"mal_remove_{pct}pct",
            error_type="MALICIOUS_REMOVAL",
            error_level=pct,
            description=f"Mal. removal {pct}% (10% CAVs)"
        )
    
    # Malicious Addition
    for pct in malicious_percentages:
        add_paired_configs(
            name=f"mal_add_{pct}pct",
            error_type="MALICIOUS_ADDITION",
            error_level=pct,
            description=f"Mal. addition {pct}% (10% CAVs)"
        )
    
    # Malicious Convoy
    for pct in malicious_percentages:
        add_paired_configs(
            name=f"mal_convoy_{pct}pct",
            error_type="MALICIOUS_CONVOY",
            error_level=pct,
            description=f"Mal. convoy {pct}% (10% CAVs)"
        )
    
    # High Miss Rate - sensor misses more than it reports
    for pct in high_miss_percentages:
        add_paired_configs(
            name=f"high_miss_{pct}pct",
            error_type="HIGH_MISS_RATE",
            error_level=pct,
            description=f"High miss rate +{pct}% (10% CAVs)"
        )
    
    # Detection Lag - delayed detections
    for steps in lag_steps:
        add_paired_configs(
            name=f"lag_{steps}steps",
            error_type="DETECTION_LAG",
            error_level=steps,
            description=f"Detection lag {steps} steps (10% CAVs)"
        )
    
    # Baseline with trust ON (primary baseline)
    baseline_config = get_pointpillars_os1_128_config()
    baseline_config["error_type"] = "None"
    baseline_config["error_level"] = 0
    baseline_config["use_trust_scoring"] = True
    
    # Also add baseline with trust OFF as a variant
    baseline_notrust = get_pointpillars_os1_128_config()
    baseline_notrust["error_type"] = "None"
    baseline_notrust["error_level"] = 0
    baseline_notrust["use_trust_scoring"] = False
    variants.insert(0, ExperimentConfig(
        name="baseline_notrust",
        config=baseline_notrust,
        description="Baseline no errors [trust=OFF]"
    ))
    
    return ExperimentSuite(
        name="Trust_Comparison_Sweep",
        description="Compare trust scoring ON vs OFF for each error type",
        runs_per_config=runs_per_config,
        warmup_steps=warmup_steps,
        record_steps=record_steps,
        error_injection_start=warmup_steps,
        baseline=ExperimentConfig(
            name="baseline_trust",
            config=baseline_config,
            description="Baseline no errors [trust=ON]"
        ),
        variants=variants
    )


# Main entry point for running experiments
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run CMR simulation experiments")
    parser.add_argument(
        "--suite", 
        type=str, 
        default="pointpillars",
        choices=["sensor_comparison", "pointpillars", "error_injection", "error_sweep", "trust_comparison", "gpem_penetration_sweep", "gpem_penetration_sweep_accurate", "gpem_penetration_sweep_static_matching", "gpem_distribution_sweep"],
        help="Type of experiment suite to run"
    )
    parser.add_argument(
        "--runs", 
        type=int, 
        default=10,
        help="Number of runs per configuration"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=600,
        help="Number of warmup steps before recording/error injection"
    )
    parser.add_argument(
        "--record",
        type=int,
        default=6000,
        help="Number of steps to record"
    )
    parser.add_argument(
        "--fast-forward",
        type=int,
        default=0,
        help="Number of fast-forward steps (no sensor/fusion processing)"
    )
    parser.add_argument(
        "--ff-min",
        type=int,
        default=None,
        help="Minimum fast-forward steps for randomized range"
    )
    parser.add_argument(
        "--ff-max",
        type=int,
        default=None,
        help="Maximum fast-forward steps for randomized range"
    )
    parser.add_argument(
        "--extrinsics-min",
        type=int,
        default=5,
        help="Min extrinsics severity in degrees for error_sweep (default: 5)"
    )
    parser.add_argument(
        "--extrinsics-max",
        type=int,
        default=5,
        help="Max extrinsics severity in degrees for error_sweep (default: 5)"
    )
    parser.add_argument(
        "--localization-min",
        type=int,
        default=50,
        help="Min localization severity as %% of max for error_sweep (default: 50)"
    )
    parser.add_argument(
        "--localization-max",
        type=int,
        default=50,
        help="Max localization severity as %% of max for error_sweep (default: 50)"
    )
    parser.add_argument(
        "--malicious-min",
        type=int,
        default=50,
        help="Min malicious probability %% for error_sweep (default: 50)"
    )
    parser.add_argument(
        "--malicious-max",
        type=int,
        default=50,
        help="Max malicious probability %% for error_sweep (default: 50)"
    )
    parser.add_argument(
        "--high-miss-min",
        type=int,
        default=50,
        help="Min high miss rate %% for error_sweep (default: 50)"
    )
    parser.add_argument(
        "--high-miss-max",
        type=int,
        default=50,
        help="Max high miss rate %% for error_sweep (default: 50)"
    )
    parser.add_argument(
        "--lag-min",
        type=int,
        default=5,
        help="Min detection lag in steps for error_sweep (default: 5)"
    )
    parser.add_argument(
        "--lag-max",
        type=int,
        default=5,
        help="Max detection lag in steps for error_sweep (default: 5)"
    )
    parser.add_argument(
        "-j", "--parallel", "--j",
        type=int,
        default=1,
        dest="parallel",
        help="Number of simulations to run in parallel (default: 1 = sequential)"
    )
    parser.add_argument(
        "--no-trust-scoring",
        action="store_true",
        help="Disable trust-based filtering in sensor fusion"
    )
    parser.add_argument(
        "--map",
        type=str,
        default=None,
        choices=["single", "tempe_2x3"],
        metavar="MAP",
        help="SUMO map to use (default: single). Use tempe_2x3 for Tempe 2x3 grid map."
    )
    
    parser.add_argument(
        "--variance-floor",
        type=float,
        default=None,
        help="Minimum diagonal variance for GPEM R matrix (e.g., 0.05). Prevents over-confidence at close range."
    )

    args = parser.parse_args()

    # Create experiment suite
    if args.suite == "sensor_comparison":
        suite = create_sensor_comparison_suite(runs_per_config=args.runs)
    elif args.suite == "pointpillars":
        suite = create_pointpillars_comparison_suite(runs_per_config=args.runs)
    elif args.suite == "error_injection":
        from config_templates import get_perfect_config
        suite = create_error_injection_suite(
            base_config=get_perfect_config(),
            error_types=["SINGLE_SENSOR_EXTRINSICS"],
            error_levels=[1, 2, 3, 4, 5],
            runs_per_config=args.runs
        )
    elif args.suite == "error_sweep":
        # Severity-based error sweep: 10% of CAVs, varying severity
        extrinsics_degrees = list(range(args.extrinsics_min, args.extrinsics_max + 1))
        localization_pcts = list(range(args.localization_min, args.localization_max + 1, 10))
        malicious_pcts = list(range(args.malicious_min, args.malicious_max + 1, 10))
        suite = create_comprehensive_error_sweep_suite(
            runs_per_config=args.runs,
            extrinsics_degrees=extrinsics_degrees,
            localization_percentages=localization_pcts,
            malicious_percentages=malicious_pcts,
            warmup_steps=args.warmup,
            record_steps=args.record,
            use_trust_scoring=not args.no_trust_scoring,
        )
    elif args.suite == "trust_comparison":
        # Compare trust scoring ON vs OFF for each error type
        extrinsics_degrees = list(range(args.extrinsics_min, args.extrinsics_max + 1))
        localization_pcts = list(range(args.localization_min, args.localization_max + 1, 10))
        malicious_pcts = list(range(args.malicious_min, args.malicious_max + 1, 10))
        high_miss_pcts = list(range(args.high_miss_min, args.high_miss_max + 1, 10))
        lag_steps_list = list(range(args.lag_min, args.lag_max + 1))
        suite = create_trust_comparison_suite(
            runs_per_config=args.runs,
            extrinsics_degrees=extrinsics_degrees,
            localization_percentages=localization_pcts,
            malicious_percentages=malicious_pcts,
            high_miss_percentages=high_miss_pcts,
            lag_steps=lag_steps_list,
            warmup_steps=args.warmup,
            record_steps=args.record,
        )
    elif args.suite == "gpem_penetration_sweep":
        # GPEM A/B test: Static covariance vs parameterized model
        # Default rates: 1.0%, 2.5%, 5.0%, 10%, 20%, ..., 100%
        suite = create_gpem_penetration_sweep_suite(
            runs_per_config=args.runs,
            av_injection_rates=[1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            warmup_steps=args.warmup,
            record_steps=args.record,
        )
    elif args.suite == "gpem_penetration_sweep_accurate":
        suite = create_gpem_penetration_sweep_accurate_suite(
            runs_per_config=args.runs,
            av_injection_rates=[1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            warmup_steps=args.warmup,
            record_steps=args.record,
        )
    elif args.suite == "gpem_penetration_sweep_static_matching":
        suite = create_gpem_penetration_sweep_static_matching_suite(
            runs_per_config=args.runs,
            av_injection_rates=[1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            warmup_steps=args.warmup,
            record_steps=args.record,
        )
    elif args.suite == "gpem_distribution_sweep":
        # GPEM: sweep detector/localizer distribution from 100% same to even split
        suite = create_gpem_distribution_sweep_suite(
            runs_per_config=args.runs,
            av_injection_rate=10.0,
            warmup_steps=args.warmup,
            record_steps=args.record,
        )
    
    # Override SUMO map if specified
    if args.map is not None:
        from config_templates import with_sumo_map
        if suite.baseline:
            suite.baseline = ExperimentConfig(
                name=suite.baseline.name,
                config=with_sumo_map(suite.baseline.config, args.map),
                description=suite.baseline.description,
            )
        for i, v in enumerate(suite.variants):
            suite.variants[i] = ExperimentConfig(
                name=v.name,
                config=with_sumo_map(v.config, args.map),
                description=v.description,
            )
        print(f"  🗺  Using map: {args.map} (maps/{args.map}/osm.sumocfg)\n")
    
    # Override timing if specified (except for error_sweep/trust_comparison/gpem_penetration_sweep/gpem_distribution_sweep which handle it internally)
    if args.suite not in ["error_sweep", "trust_comparison", "gpem_penetration_sweep", "gpem_penetration_sweep_accurate", "gpem_penetration_sweep_static_matching", "gpem_distribution_sweep"]:
        suite.fast_forward_steps = args.fast_forward
        suite.warmup_steps = args.warmup
        suite.record_steps = args.record
    else:
        # For those that handle it internally, still allow overriding fast_forward
        suite.fast_forward_steps = args.fast_forward
    
    # Apply randomized FF range if provided
    if args.ff_min is not None and args.ff_max is not None:
        suite.fast_forward_range = (args.ff_min, args.ff_max)

    # Inject variance floor into all configs if specified
    if args.variance_floor is not None:
        for v in suite.variants:
            v.config["variance_floor"] = args.variance_floor
        if suite.baseline is not None:
            suite.baseline.config["variance_floor"] = args.variance_floor

    # Print suite info
    total_configs = 1 + len(suite.variants)  # baseline + variants
    total_runs = total_configs * suite.runs_per_config
    ff_timing = f"{suite.fast_forward_range[0]}-{suite.fast_forward_range[1]} (randomized)" if suite.fast_forward_range else f"{suite.fast_forward_steps}"
    
    if args.suite == "trust_comparison":
        trust_status = "COMPARING (ON vs OFF)"
    elif args.suite in ("gpem_penetration_sweep", "gpem_penetration_sweep_accurate", "gpem_penetration_sweep_static_matching", "gpem_distribution_sweep"):
        trust_status = "DISABLED (Model comparison only)"
    else:
        trust_status = "DISABLED" if args.no_trust_scoring else "ENABLED"
    print(f"\n{'='*60}")
    print(f"Experiment Suite: {suite.name}")
    print(f"Description: {suite.description}")
    print(f"Configurations: {total_configs} ({suite.runs_per_config} runs each)")
    print(f"Total runs: {total_runs}")
    print(f"Timing: FF: {ff_timing}, Warmup: {suite.warmup_steps}, Record: {suite.record_steps} steps")
    print(f"Trust scoring: {trust_status}")
    if suite.error_injection_start is not None:
        print(f"Error injection starts at step: {suite.error_injection_start}")
    print(f"{'='*60}\n")
    
    # Run experiments
    runner = ExperimentRunner(output_dir=args.output)
    results = runner.run_suite(suite, parallel=args.parallel)
    
    # Auto-generate plots for GPEM suites
    if args.suite in ("gpem_penetration_sweep", "gpem_penetration_sweep_accurate", "gpem_penetration_sweep_static_matching"):
        try:
            print("\n📊 Generating GPEM comparison plot...")
            sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
            from plot_gpem_results import plot_gpem_results_simple
            plot_gpem_results_simple(runner.current_experiment_dir)
        except Exception as e:
            print(f"  ⚠️  Failed to generate plot: {e}")
    elif args.suite == "gpem_distribution_sweep":
        try:
            print("\n📊 Generating GPEM distribution sweep plot...")
            sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
            from plot_gpem_distribution_sweep import plot_distribution_sweep_results
            plot_distribution_sweep_results(runner.current_experiment_dir)
        except Exception as e:
            print(f"  ⚠️  Failed to generate plot: {e}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if args.suite == "gpem_distribution_sweep":
        # Print all 16 variants per step (same format as penetration sweep)
        import re
        for config_name, agg in results.items():
            print(f"{config_name}: Global AMOTA = {agg.avg_global_amota_mean:.4f} \u00b1 {agg.avg_global_amota_std:.4f} | AMOTP = {agg.avg_global_amotp_mean:.4f}m \u00b1 {agg.avg_global_amotp_std:.4f} | HOTA = {agg.avg_hota_mean:.4f} (DetA={agg.avg_deta_mean:.4f}, AssA={agg.avg_assa_mean:.4f})")
        print("="*60)
    else:
        for config_name, agg in results.items():
            print(f"{config_name}: Global AMOTA = {agg.avg_global_amota_mean:.4f} ± {agg.avg_global_amota_std:.4f} | AMOTP = {agg.avg_global_amotp_mean:.4f}m ± {agg.avg_global_amotp_std:.4f} | HOTA = {agg.avg_hota_mean:.4f} (DetA={agg.avg_deta_mean:.4f}, AssA={agg.avg_assa_mean:.4f})")
    
    # Clean exit
    sys.exit(0)