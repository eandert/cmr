"""
Centralized filter tuning parameters.

All filter classes read their parameters from here. This makes it easy to
sweep parameters across all filters simultaneously or tune them independently.

sigma_a controls the process noise matrix Q:
  - Higher sigma_a → P grows faster during prediction → filter trusts measurements more
  - Lower sigma_a → P stays tight → filter trusts its own prediction more
  - For GPEM to help, sigma_a should be high enough that R calibration matters
"""

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


def get_sigma_a(filter_type: str, config: dict = None) -> float:
    """
    Get sigma_a for a filter type, with optional runtime override.

    Args:
        filter_type: One of "ekf", "ci", "bici", "akf", "pf"
        config: Optional config dict. If it contains "sigma_a_override",
                that dict maps filter_type -> value.

    Returns:
        sigma_a value (float)
    """
    if config and "sigma_a_override" in config:
        overrides = config["sigma_a_override"]
        if filter_type in overrides:
            return float(overrides[filter_type])
    return SIGMA_A.get(filter_type, 2.0)
