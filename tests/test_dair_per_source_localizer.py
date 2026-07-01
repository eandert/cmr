"""Regression tests for the per-source LOCALIZER (loc_cov) path.

`v2v4real_replay.run_scenario` used to build a single GLOBAL localizer and reuse
it for every source, so the localization covariance (loc_cov, added to each
detection's perception covariance per Eq. 18) was identical for a well-localized
RTK vehicle and a poorly-localized roadside infrastructure unit. The localizer
now supports the same "{vehicle}" placeholder the detector models use: a
templated `localizer_name` (DAIR's `dair_loc_{vehicle}`) binds one localizer per
source, so DAIR's infrastructure source can carry the large OU-characterized
localization error while the vehicle stays at the RTK floor. These tests pin:

  1. DATA: the two DAIR localizer CSVs load and produce the intended loc_cov —
     inf >> veh (the whole point of per-source localization).

  2. CONSTRUCTION + SELECTION (the regression gate): a templated localizer_name
     constructs one localizer per meta vehicle AND each source's detections are
     scored with ITS OWN localizer at fusion time; a non-templated name builds a
     single shared localizer reused for every source (byte-identical to the old
     behaviour — V2V4Real is unaffected).

Run: PYTHONPATH=src python -m pytest tests/test_dair_per_source_localizer.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import localizer as cmr_localizer
import v2v4real_replay
from error import ErrorPackage, ErrorType


# --- 1. DATA: the DAIR localizer CSVs produce the intended loc_cov -----------

def test_dair_localizer_csvs_loc_cov():
    """dair_loc_inf carries the OU error; dair_loc_veh stays at the RTK floor."""
    err = ErrorPackage(ErrorType.LOCALIZATION, 0)

    def diag(name):
        loc = cmr_localizer.get_localizer(name, err, use_gpem_model=False,
                                          use_quadratic=False)
        # yaw=0 -> covariance axes align with (longitudinal, lateral).
        return np.diag(loc.get_localization_covariance(15.0, 0.0))

    veh = diag("dair_loc_veh")
    inf = diag("dair_loc_inf")

    # veh = RTK floor (sigma 0.03 m -> var 9e-4) on both axes.
    np.testing.assert_allclose(veh, [9e-4, 9e-4], atol=1e-6)
    # inf = OU train fit: long std 0.482 (var 0.23232), lat std 0.424 (var 0.17978).
    np.testing.assert_allclose(inf, [0.482**2, 0.424**2], rtol=1e-4)
    # The point of per-source localization: inf is far larger than veh.
    assert inf[0] > 100 * veh[0] and inf[1] > 100 * veh[1]


# --- 2. CONSTRUCTION + SELECTION via run_scenario ---------------------------

def _write_min_scenario(tmp_path: Path, vehicles):
    """One-frame scenario (meta + ego/det/gt csvs), one detection per vehicle."""
    d = tmp_path / "val__0001"
    d.mkdir()
    (d / "meta.yaml").write_text(
        "split: val\nscenario: '0001'\nvehicles: [%s]\n" % ", ".join(vehicles)
    )
    ego_hdr = "timestamp,frame_id,vehicle_id,x,y,z,yaw,vx,vy,length,width,height\n"
    ego_rows = "".join(
        f"0.0,0,{v},{i*100.0},0.0,0.0,0.0,0.0,0.0,4.0,2.0,1.5\n"
        for i, v in enumerate(vehicles)
    )
    (d / "ego_state.csv").write_text(ego_hdr + ego_rows)
    det_hdr = ("timestamp,vehicle_id,frame_id,det_idx,x,y,z,yaw,vx,vy,"
               "length,width,height,score,label\n")
    det_rows = "".join(
        f"0.0,{v},0,0,{i*100.0+10.0},0.0,0.0,0.0,0.0,0.0,4.0,2.0,1.5,0.9,car\n"
        for i, v in enumerate(vehicles)
    )
    (d / "detections.csv").write_text(det_hdr + det_rows)
    (d / "gt_objects.csv").write_text(
        "ass_id,obj_id,vehicle_id,frame_id,x,y,z,yaw,length,width,height,label\n"
    )
    return d


class _RecordingLocalizer:
    """Stand-in localizer that records every get_localization_covariance call."""
    def __init__(self, name):
        self.name = name
        self.cov_calls = 0

    def get_localization_covariance(self, velocity, yaw):
        self.cov_calls += 1
        return np.eye(2) * 0.01


def _run_capturing(scenario_dir, localizer_name, detector_name, monkeypatch):
    """Run a 1-stream scenario, capturing constructed localizers by name."""
    built = {}

    def fake_get_localizer(name, error_package, use_gpem_model=False,
                           use_quadratic=False):
        loc = _RecordingLocalizer(name)
        built.setdefault(name, []).append(loc)
        return loc

    monkeypatch.setattr(cmr_localizer, "get_localizer", fake_get_localizer)

    cfg = {
        "detector_name": detector_name,
        "localizer_name": localizer_name,
        "self_report_egos": False,
        "lidar_range": [-1e4, -1e4, -1e4, 1e4, 1e4, 1e4],
        "gt_range":    [-1e4, -1e4, -1e4, 1e4, 1e4, 1e4],
        "detector_max_range": 200.0,
        "score_threshold": 0.0,
        "lifecycle_mode": "ab3dmot",
        "streams_keep": ["sabre_static"],
        "record_tape_for": ["sabre_static"],
    }
    v2v4real_replay.run_scenario(str(scenario_dir), cfg)
    return built


def test_per_source_localizer_resolves_and_is_used(tmp_path, monkeypatch):
    """Templated name -> one localizer per source, each used for its own dets."""
    sc = _write_min_scenario(tmp_path, ["veh", "inf"])
    built = _run_capturing(sc, "dair_loc_{vehicle}", "dair_pp_{vehicle}", monkeypatch)

    # The template resolved to exactly one localizer per meta vehicle (one
    # instance each — no redundant rebuild for the shared fallback).
    assert {"dair_loc_veh", "dair_loc_inf"} <= set(built)
    assert len(built["dair_loc_veh"]) == 1 and len(built["dair_loc_inf"]) == 1
    # SELECTION: each source's single detection was scored with ITS localizer.
    assert built["dair_loc_veh"][0].cov_calls == 1
    assert built["dair_loc_inf"][0].cov_calls == 1


def test_global_localizer_unchanged(tmp_path, monkeypatch):
    """Non-templated name -> a single shared localizer for every source.

    The regression gate: V2V4Real (rtk_v2v4real, no placeholder) builds exactly
    one localizer and reuses it for both sources' detections — identical to the
    pre-per-source behaviour.
    """
    sc = _write_min_scenario(tmp_path, ["astuff", "tesla"])
    built = _run_capturing(sc, "rtk_v2v4real", "pointpillar_v2v4real_{vehicle}",
                           monkeypatch)

    # A non-templated name builds exactly ONE localizer for the configured name
    # (no per-source split) and reuses it for BOTH sources' detections — the
    # pre-per-source behaviour. (Other subsystems may build their own
    # localizers; we assert only the configured-name contract.)
    assert "rtk_v2v4real" in built
    assert len(built["rtk_v2v4real"]) == 1
    assert built["rtk_v2v4real"][0].cov_calls == 2
    # The "{vehicle}" template was NOT engaged for a plain name.
    assert not any(n.startswith("dair_loc") for n in built)
