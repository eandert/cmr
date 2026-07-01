"""Unit tests for export_v2v4real_to_cmr.build_pred_by_key (in the mmdet3d clone).

Pins the identity-based preds<->infos pairing that replaced the silent
positional pairing — the bug that placed PointPillar detections in the wrong
frames (~1% det->GT recall). Skipped if the mmdet3d clone isn't on disk.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _export_module():
    """Import the mmdet3d export module via mmdet3d_root from paths.local.yaml."""
    import yaml  # noqa
    pl = REPO / "paths.local.yaml"
    if not pl.is_file():
        pytest.skip("paths.local.yaml not present")
    mm = (yaml.safe_load(pl.read_text()) or {}).get("mmdet3d_root")
    if not mm:
        pytest.skip("mmdet3d_root not set in paths.local.yaml")
    tools = Path(mm) / "tools"
    if not (tools / "export_v2v4real_to_cmr.py").is_file():
        pytest.skip(f"export script not found under {tools}")
    sys.path.insert(0, str(tools))
    try:
        return importlib.import_module("export_v2v4real_to_cmr")
    except Exception as e:  # heavy/missing deps on a fresh clone
        pytest.skip(f"export module import failed: {e}")


def test_maps_scenario_frame_by_identity():
    mod = _export_module()
    meta = [{"scenario": "Day19/sceneA", "timestamp": "000000"},
            {"scenario": "Day19/sceneA", "timestamp": "000001"}]
    out = mod.build_pred_by_key(meta, ["P0", "P1"])
    assert out == {("sceneA", 0): "P0", ("sceneA", 1): "P1"}


def test_length_mismatch_raises():
    mod = _export_module()
    with pytest.raises(ValueError):
        mod.build_pred_by_key([{"scenario": "s", "timestamp": "0"}], ["P0", "P1"])


def test_duplicate_scenario_frame_key_raises():
    mod = _export_module()
    # Both rows reduce to key ("a", 0) → ambiguous (e.g. multi-vehicle preds).
    meta = [{"scenario": "s/a", "timestamp": "0"}, {"scenario": "x/a", "timestamp": "0"}]
    with pytest.raises(ValueError):
        mod.build_pred_by_key(meta, ["P0", "P1"])
