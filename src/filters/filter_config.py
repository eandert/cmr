"""
Centralized filter tuning parameters.

All filter classes read their parameters from here. This makes it easy to
sweep parameters across all filters simultaneously or tune them independently.

sigma_a controls the process noise matrix Q:
  - Higher sigma_a → P grows faster during prediction → filter trusts measurements more
  - Lower sigma_a → P stays tight → filter trusts its own prediction more
  - For GPEM to help, sigma_a should be high enough that R calibration matters
"""

import os

# Process noise (acceleration std dev in m/s²)
SIGMA_A = {
    "ekf": 10.0,        # Optimal from sweep: broad plateau 8-10
    "ci": 6.0,          # Optimal from sweep: sharp peak at 6
    "bici": 6.0,        # Optimal from sweep: identical to CI
    "akf": 6.0,         # Optimal from sweep: peak at 6, graceful decline
    "pf": 14.0,         # Optimal from sweep: late peak, crashes after 16
    "sabre": 2.0,       # Optimal from sweep: flat 0.75-0.77 across all σ_a, peak at 2
}

# AKF adaptation parameters (optimal from adaptation sweep: α=0.3, w=10)
AKF_PARAMS = {
    "window_size": 10,      # Innovation history window — optimal from sweep
    "adapt_alpha": 0.3,     # Exponential smoothing rate — optimal from sweep
    "sigma_a_min": 0.2,     # Lower bound on adapted sigma_a
    "sigma_a_max": 15.0,    # Upper bound on adapted sigma_a
    "warmup_steps": 5,      # Frames before adaptation starts
}

# SABRE adaptation parameters (optimal from adaptation sweep: α=0.1, w=10)
SABRE_PARAMS = {
    "window_size": 10,      # NIS history window — optimal from sweep
    "adapt_rate": 0.1,      # Reliability exponential smoothing — optimal from sweep
    "stale_timeout": 50,    # Frames before evicting unseen sources
}

# Particle Filter parameters
PF_PARAMS = {
    "n_particles": 500,     # Number of particles
    "resample_threshold": 0.5,  # N_eff / N ratio to trigger resampling
}

# CI/BICI optimization
CI_PARAMS = {
    "minimize": "trace",    # "trace" or "det" for omega optimization
    "ici_fallback": True,   # Fall back to standard CI if ICI produces non-PSD
}

# Baseline R value (flat covariance for non-GPEM comparison)
BASELINE_R = 0.5  # Will be used if override is active in traci_interface

# --- Auto-Q (process noise from the calibrated measurement covariance) ---------
# The qr-ratio autotune lead: optimal position-Q tracks the MAGNITUDE of R, so derive
# sigma_a closed-form from the GPEM mean R instead of sweeping it. Calibrated so
# sigma_a = 2.0 at the DAIR PointPillars detector R ~= 0.025 (per-axis position variance);
# k = 2.0 / sqrt(0.025) ~= 12.6. The optimum is a broad plateau in the small-R (detector)
# regime, so this only bites for large R (e.g. an inflated localizer cov, or a high-variance
# camera detector). See docs/INSTANT_TUNE_SPEC.md and the qr_ratio autotune memory.
SIGMA_A_K = 12.6

# Process-local sigma_a override (filter_type -> value). The filters call get_sigma_a()
# WITHOUT a config (they are constructed deep inside the fusion stack), so the run config
# cannot reach them directly. The run entry point (e.g. run_dair_evaluation.run_one, which
# executes inside each worker process) calls set_sigma_a_overrides() once from the run
# config, populating this so get_sigma_a() picks it up. Empty by default => production
# unaffected. Set per worker; env-inherited and config-pickled paths both reach here.
_SIGMA_A_OVERRIDES: dict = {}


def set_sigma_a_overrides(mapping: dict) -> None:
    """Install a process-local {filter_type: sigma_a} override (idempotent)."""
    global _SIGMA_A_OVERRIDES
    _SIGMA_A_OVERRIDES = {str(k): float(v) for k, v in (mapping or {}).items()}


def clear_sigma_a_overrides() -> None:
    """Remove any process-local override (used by tests / single-process reruns)."""
    global _SIGMA_A_OVERRIDES
    _SIGMA_A_OVERRIDES = {}


def sigma_a_from_R(mean_R: float, k: float = SIGMA_A_K) -> float:
    """Auto-Q law: process-noise sigma_a from the mean per-axis measurement variance R."""
    return k * (max(float(mean_R), 0.0) ** 0.5)


def get_sigma_a(filter_type: str, config: dict = None) -> float:
    """
    Get sigma_a for a filter type, in precedence order:
      1. config["sigma_a_override"][filter_type] — explicit per-call (most specific).
      2. CMR_SIGMA_A_<FILTER> env var — manual sweep / debug override.
      3. process-local override (set_sigma_a_overrides) — the auto-Q value for this run.
      4. SIGMA_A[filter_type] — the hand-tuned default (falls back to 2.0).

    Args:
        filter_type: One of "ekf", "ci", "bici", "akf", "pf", "sabre".
        config: Optional config dict carrying a "sigma_a_override" map.
    """
    if config and "sigma_a_override" in config:
        overrides = config["sigma_a_override"]
        if filter_type in overrides:
            return float(overrides[filter_type])
    # Sweep affordance (default-off): CMR_SIGMA_A_<FILTER> env var overrides for one run.
    env = os.environ.get(f"CMR_SIGMA_A_{filter_type.upper()}")
    if env is not None:
        return float(env)
    if filter_type in _SIGMA_A_OVERRIDES:
        return _SIGMA_A_OVERRIDES[filter_type]
    return SIGMA_A.get(filter_type, 2.0)
