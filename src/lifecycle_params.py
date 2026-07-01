"""Tier 1 data-driven log_odds lifecycle params.

Loaded from `results/birth_gate_calibration/recommended_lifecycle.csv`
(produced by `scripts/solve_amota_bayes_gate.py`). Per detector, the solver
emits:

    log_lr_miss      — per-miss log-odds decay (negative). Derived to
                       preserve TPs of natural lifetime L_TP under their
                       natural miss rate p_miss_TP:
                           |log_lr_miss| = (logit(p_birth) - kill) / (L_TP·p_miss_TP)
                       Aggressive for short-lifetime detectors, gentle for
                       long-lifetime ones.

    kill_log_odds    — fixed at -3.0 (canonical, log(1/20) ≈ P_real < 0.048).

    p_birth          — pulled from the AMOTA-Bayes gate scalar when non-zero;
                       0.7 fallback. Used as the initial log-odds at birth.

Usage:
    from lifecycle_params import load_lifecycle_params
    lc = load_lifecycle_params("u_pp_cp_lf_tesla")
    if lc is not None:
        # Override SensorFusion defaults
        kw = dict(kill_log_odds=lc.kill_log_odds,
                  log_lr_miss=lc.log_lr_miss,
                  p_tp_birth_gate=lc.p_birth)

Mirrors the lookup discipline of `birth_gate_curves.py`: detector_name keys
include the model's own name (e.g. 'centerpoint_54m_v2v4real_finetune_tesla')
plus all alias names emitted by the solver (e.g. 'u_cpft_cpft_tesla',
'mix_pp_cpzs_tesla'). One row per (detector, alias).
"""
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

_REPO = Path(__file__).resolve().parent.parent
_DEFAULT_LIFECYCLE_CSV = _REPO / "results" / "birth_gate_calibration" / "recommended_lifecycle.csv"


@dataclass
class LifecycleParams:
    """Per-detector log_odds lifecycle parameters.

    Mirrors the kwargs of `SensorFusion.__init__` for the lifecycle-relevant
    knobs. p_birth_logit is not stored separately — the SensorFusion code
    converts p_tp_birth_gate to logit internally; we pass p_birth as the
    probability and let it convert.

    Tier 1.5 additions: confirm_log_odds + tape_score_miss_decay derived
    per detector from p_match_TP. Defaults preserve the canonical values
    for legacy CSVs that don't carry these columns.
    """
    detector: str
    p_birth: float
    log_lr_miss: float
    kill_log_odds: float
    confirm_log_odds: float = 2.996       # log(20) canonical
    tape_score_miss_decay: float = 0.5    # canonical default
    p_miss_tp_observed: float = float("nan")
    l_tp_observed: float = float("nan")
    n_tracks_observed: int = 0


_cache: Dict[str, LifecycleParams] = {}
_loaded_path: Optional[Path] = None


def _load_csv(path: Path) -> Dict[str, LifecycleParams]:
    out: Dict[str, LifecycleParams] = {}
    if not path.exists():
        return out
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                lc = LifecycleParams(
                    detector=row["detector"],
                    p_birth=float(row.get("p_birth", 0.7) or 0.7),
                    log_lr_miss=float(row.get("log_lr_miss", -0.847) or -0.847),
                    kill_log_odds=float(row.get("kill_log_odds", -3.0) or -3.0),
                    confirm_log_odds=float(row.get("confirm_log_odds", 2.996) or 2.996),
                    tape_score_miss_decay=float(row.get("tape_score_miss_decay", 0.5) or 0.5),
                    p_miss_tp_observed=float(row.get("p_miss_tp_observed", 0.0) or 0.0),
                    l_tp_observed=float(row.get("l_tp_observed", 0.0) or 0.0),
                    n_tracks_observed=int(float(row.get("n_tracks_observed", 0) or 0)),
                )
            except (ValueError, KeyError):
                continue
            out[lc.detector] = lc
    return out


def load_lifecycle_params(detector_name: str,
                           lifecycle_csv: Optional[Path] = None
                           ) -> Optional[LifecycleParams]:
    """Look up lifecycle params by detector_name (or alias).

    Returns None if the CSV is absent or detector isn't in it. Callers
    should fall back to runtime defaults in that case.
    """
    global _cache, _loaded_path
    csv_path = Path(lifecycle_csv) if lifecycle_csv else _DEFAULT_LIFECYCLE_CSV
    if _loaded_path != csv_path or not _cache:
        _cache = _load_csv(csv_path)
        _loaded_path = csv_path
    return _cache.get(detector_name)
