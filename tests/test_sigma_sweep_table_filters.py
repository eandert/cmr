"""Regression tests for ``scripts/generate_sigma_sweep_table.py``.

Two guarantees this file pins:

1. **Filter coverage** — the sigma_a sweep appendix artifacts cover all six
   fusion streams (EKF, CI, BICI, AKF, PF, SABRE), matching the LaTeX table and
   the other SUMO plots. SABRE was previously missing.

2. **Exact variant matching** — the summary-key match must be exact. The old
   ``ck.endswith('_' + variant)`` test let the bare EKF variants (``baseline``,
   ``gpem_linear``, …) swallow every prefixed stream (``ci_baseline``,
   ``sabre_gpem_linear``, …), silently turning the EKF column into a six-filter
   pooled average. This pins the precise-match behaviour so it can't regress.

The script runs top-to-bottom at import (it is a CLI, not import-safe), so we
extract the pure helper and the literal config via ``ast`` rather than importing
the module.

Run: python -m pytest tests/test_sigma_sweep_table_filters.py -v
"""
import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_SRC = (REPO / "scripts" / "generate_sigma_sweep_table.py").read_text()


def _load_func(name):
    """Compile and return a single top-level function from the script source."""
    for node in ast.parse(_SRC).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            ns = {"re": re}
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<x>", "exec"), ns)
            return ns[name]
    raise AssertionError(f"{name} not found in script")


def _load_literal(name):
    """Return a top-level literal assignment (list/dict) from the script source."""
    for node in ast.parse(_SRC).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in script")


match = _load_func("_key_matches_variant")
FILTER_NAMES = _load_literal("filter_names")
FILTER_GPEM_MODES = _load_literal("filter_gpem_modes")
FILTER_BASELINES = _load_literal("filter_baselines")

EXPECTED_FILTERS = ["EKF", "CI", "BICI", "AKF", "PF", "SABRE"]


def test_all_six_filters_covered():
    assert set(FILTER_NAMES) == set(EXPECTED_FILTERS)
    assert set(FILTER_GPEM_MODES) == set(EXPECTED_FILTERS)
    assert set(FILTER_BASELINES) == set(EXPECTED_FILTERS)


@pytest.mark.parametrize("key,variant,expected", [
    # Distribution-sweep keys: exact suffix after _step_{N}_
    ("gpem_dist_sweep_step_0_baseline", "baseline", True),
    ("gpem_dist_sweep_step_3_gpem_linear", "gpem_linear", True),
    ("gpem_dist_sweep_step_0_sabre_gpem_linear", "sabre_gpem_linear", True),
    # The bug: bare EKF variants must NOT swallow prefixed streams
    ("gpem_dist_sweep_step_0_ci_baseline", "baseline", False),
    ("gpem_dist_sweep_step_0_sabre_baseline", "baseline", False),
    ("gpem_dist_sweep_step_0_bici_gpem_linear", "gpem_linear", False),
    ("gpem_dist_sweep_step_0_sabre_gpem_linear", "gpem_linear", False),
    # Penetration-sweep keys: exact prefix before _av_{rate}pct
    ("baseline_av_100.0pct", "baseline", True),
    ("sabre_gpem_polar_av_50.0pct", "sabre_gpem_polar", True),
    ("ci_baseline_av_100.0pct", "baseline", False),
])
def test_exact_variant_match(key, variant, expected):
    assert bool(match(key, variant)) is expected


def test_bare_ekf_variant_does_not_pool_other_filters():
    """The concrete failure: 'baseline' matching the whole fleet's baselines."""
    fleet = [
        "gpem_dist_sweep_step_0_baseline",       # EKF (should match)
        "gpem_dist_sweep_step_0_ci_baseline",    # others (must NOT match)
        "gpem_dist_sweep_step_0_bici_baseline",
        "gpem_dist_sweep_step_0_akf_baseline",
        "gpem_dist_sweep_step_0_pf_baseline",
        "gpem_dist_sweep_step_0_sabre_baseline",
    ]
    matched = [k for k in fleet if match(k, "baseline")]
    assert matched == ["gpem_dist_sweep_step_0_baseline"]
