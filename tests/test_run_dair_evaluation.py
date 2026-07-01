"""Unit tests for the DAIR Phase-B driver helpers (scripts/run_dair_evaluation.py).

These cover the pure logic — scenario discovery, the match-tape → JSON
conversion, and the DAIR config assembly (which bakes in the correctness-critical
defaults: per-source detector models, self-report off, and the WIDE-Z spatial
gate that keeps DAIR's z~=20 m world-frame detections from being silently
dropped by `_in_ego_range`'s absolute-z check). The heavy end-to-end run is
exercised separately; here we guard the contract the off-repo B3 step relies on.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# Load the script as a module (it guards main() under __main__).
_spec = importlib.util.spec_from_file_location(
    "run_dair_evaluation", REPO / "scripts" / "run_dair_evaluation.py"
)
rde = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rde)


def test_tape_to_track_frames_casts_and_sorts():
    # Out-of-order frames with numpy floats (as the tracker emits them). The 3D
    # tape carries (id, x, y, w, l, yaw, score, height, z) — height at index 7,
    # z at index 8 (the match_frame_3d_iou convention).
    tape = [
        {"frame_idx": 5, "tracks": [(np.int64(7), np.float64(1.0), np.float64(2.0),
                                     np.float64(2.0), np.float64(4.5),
                                     np.float64(-1.0), np.float64(0.9),
                                     np.float64(1.6), np.float64(20.1))], "gts": []},
        {"frame_idx": 3, "tracks": [], "gts": []},
    ]
    frames = rde.tape_to_track_frames(tape)
    assert [f["frame_id"] for f in frames] == [3, 5]  # sorted by frame
    t = frames[1]["tracks"][0]
    assert len(t) == 9
    # Every field is a native Python type so json.dump won't choke.
    assert isinstance(t[0], int)
    assert all(isinstance(v, float) for v in t[1:])
    assert t == [7, 1.0, 2.0, 2.0, 4.5, -1.0, 0.9, 1.6, 20.1]


def test_tape_to_track_frames_height_z_none_when_untracked():
    # A track that never saw a 3D detection carries None height/z (B3 falls back
    # to nearest-detection re-attach). Legacy 7-tuples must also degrade to None.
    tape = [
        {"frame_idx": 1, "tracks": [
            (1, 0.0, 0.0, 2.0, 4.0, 0.0, 0.5, None, None),   # untracked vertical
            (2, 1.0, 1.0, 2.0, 4.0, 0.0, 0.5),               # legacy 7-tuple
        ], "gts": []},
    ]
    a, b = rde.tape_to_track_frames(tape)[0]["tracks"]
    assert a == [1, 0.0, 0.0, 2.0, 4.0, 0.0, 0.5, None, None]
    assert b == [2, 1.0, 1.0, 2.0, 4.0, 0.0, 0.5, None, None]


def test_discover_scenarios_filters_splits(tmp_path):
    for name in ("val__0003", "val__0007", "test__0001", "not_a_scenario"):
        d = tmp_path / name
        d.mkdir()
        if "__" in name:
            (d / "meta.yaml").write_text("vehicles: [veh, inf]\n")
    # A dir without meta.yaml must be skipped even though it has "__".
    (tmp_path / "val__9999").mkdir()

    all_found = rde.discover_scenarios(tmp_path)
    assert [d.name for d in all_found] == ["test__0001", "val__0003", "val__0007"]

    val_only = rde.discover_scenarios(tmp_path, splits=["val"])
    assert [d.name for d in val_only] == ["val__0003", "val__0007"]


def test_build_config_bakes_dair_defaults():
    streams = ["sabre_gpem_quadratic"]
    cfg = rde.build_config(streams, {})
    assert cfg["detector_name"] == "dair_pp_{vehicle}"
    assert cfg["self_report_egos"] is False          # veh/inf are not GT objects
    assert cfg["lidar_range"] == rde.DAIR_LIDAR_RANGE  # WIDE-Z gate
    assert cfg["gt_range"] == rde.DAIR_LIDAR_RANGE
    # z gate must be effectively unbounded (DAIR world z ~= 20 m).
    assert cfg["lidar_range"][2] <= -1e3 and cfg["lidar_range"][5] >= 1e3
    assert cfg["streams_keep"] == streams
    assert cfg["record_tape_for"] == streams


def test_build_config_overrides_but_pins_streams():
    streams = ["ab3dmot_baseline", "sabre_gpem_linear"]
    cfg = rde.build_config(streams, {"score_threshold": 0.3, "lifecycle_mode": "log_odds",
                                     # a stray override must NOT desync the tapes
                                     "record_tape_for": ["something_else"]})
    assert cfg["score_threshold"] == 0.3
    assert cfg["lifecycle_mode"] == "log_odds"
    # streams_keep/record_tape_for are always forced to the requested streams.
    assert cfg["streams_keep"] == streams
    assert cfg["record_tape_for"] == streams
