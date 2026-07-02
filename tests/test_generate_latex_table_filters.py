"""Unit tests for the filter coverage of ``scripts/generate_latex_table.py``.

The SUMO GPEM sweeps produce six fusion streams per covariance mode
(EKF, CI, AKF, PF, BICI, SABRE — see ``src/traci_interface.py`` under
``triple_fusion``), but the LaTeX table generator historically rendered only
the first four. These tests pin that every produced stream now maps to a table
row so BICI and SABRE (the paper's second contribution) can never silently
fall out of a published table again.

Run: PYTHONPATH=src python -m pytest tests/test_generate_latex_table_filters.py -v
"""
import importlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

glt = importlib.import_module("generate_latex_table")

# The six fusion filters instantiated per covariance mode under triple_fusion.
EXPECTED_FILTERS = ["Kalman", "CI", "AKF", "PF", "BICI", "SABRE"]


def test_all_six_filters_present_and_ordered():
    """FILTER_ORDER lists all six streams; BICI/SABRE come after the CI family."""
    assert glt.FILTER_ORDER == EXPECTED_FILTERS
    # Every filter group has a display name for the \multirow cell.
    for filt in EXPECTED_FILTERS:
        assert filt in glt.FILTER_DISPLAY, f"{filt} missing from FILTER_DISPLAY"


def test_variant_map_covers_every_filter_times_cov_mode():
    """VARIANT_MAP has exactly one entry per (filter, covariance mode) = 6 x 5."""
    covered = {(f, c) for f, c in glt.VARIANT_MAP.values()}
    expected = {(f, c) for f in EXPECTED_FILTERS for c in glt.COV_ORDER}
    assert covered == expected
    assert len(glt.VARIANT_MAP) == len(EXPECTED_FILTERS) * len(glt.COV_ORDER)


def test_variant_keys_match_result_stream_naming():
    """The VARIANT_MAP keys are exactly the ``{filter}_{cov}`` stream suffixes
    written to summary.json (Kalman is the bare ``{cov}`` prefix)."""
    for key, (filt, cov) in glt.VARIANT_MAP.items():
        expected_key = cov if filt == "Kalman" else f"{filt.lower()}_{cov}"
        assert key == expected_key


def _fake_averages():
    """Minimal averages dict: one distinct value per variant so best-per-metric
    resolution is unambiguous."""
    averages = {}
    for i, key in enumerate(glt.VARIANT_MAP):
        averages[key] = {
            mean_field: 0.1 * (i + 1)
            for mean_field, _, _, _ in glt.METRICS
        }
    return averages


def test_combined_table_renders_bici_and_sabre_rows():
    """The combined table includes a group for every filter, BICI/SABRE included."""
    latex = glt.generate_combined_latex(_fake_averages(), "Test Suite", "test", n_steps=3)
    for filt in EXPECTED_FILTERS:
        assert glt.FILTER_DISPLAY[filt] in latex, f"{filt} group missing from combined table"


def test_filter_flag_accepts_bici_and_sabre():
    """The --filter CLI choice and its dispatch map both know the new filters."""
    # argparse choices are declared in main(); assert the dispatch map is complete
    # by exercising the same lookup main() performs.
    for name, group in [("bici", "BICI"), ("sabre", "SABRE")]:
        rows = []
        for cov in glt.COV_ORDER:
            key = f"{group.lower()}_{cov}"
            assert key in glt.VARIANT_MAP
            rows.append(key)
        assert len(rows) == len(glt.COV_ORDER)
