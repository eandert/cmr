"""Unit tests for v2v4real_replay._active_cov_modes.

Pins the fix where run_scenario eagerly built ALL covariance modes (including
gpem_polar) for every detection even when only one stream was kept — which
crashed on a detector whose GPEM CSV lacks polar columns (e.g. the older
centerpoint_100m CSV). It must now build only the cov modes the active streams
actually consume.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import v2v4real_replay as R  # noqa: E402


def test_single_stream_keeps_only_its_cov():
    assert R._active_cov_modes({"ci_gpem_quadratic"}) == {"gpem_quadratic"}


def test_polar_excluded_when_not_kept():
    # The crash case: keeping only ci_gpem_quadratic must NOT pull in gpem_polar
    # (whose covariance load fails for a CSV without polar columns).
    assert "gpem_polar" not in R._active_cov_modes({"ci_gpem_quadratic"})


def test_all_streams_keep_all_cov_modes():
    assert R._active_cov_modes(set(R.STREAM_KEYS)) == set(R.COV_MODES)


def test_baseline_stream_keeps_baseline():
    assert R._active_cov_modes({"ci_baseline"}) == {"baseline"}


def test_empty_active_set_keeps_nothing():
    assert R._active_cov_modes(set()) == set()


def test_kalman_special_key_resolves():
    # _stream_key('kalman', 'gpem_quadratic') == 'gpem_quadratic' (no prefix);
    # the helper must still resolve it to the gpem_quadratic cov mode.
    assert R._active_cov_modes({"gpem_quadratic"}) == {"gpem_quadratic"}
