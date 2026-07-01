"""Regression tests for the de-hardcoded per-source lifecycle / birth-gate path.

`v2v4real_replay.run_scenario` used to iterate the literal `("astuff", "tesla")`
pair when building the per-source `data_driven_lifecycle` maps and the
`data_driven_gate` curves, so those two mechanisms silently did nothing for any
dataset whose source vehicles aren't astuff/tesla (e.g. DAIR-V2X-Seq's
veh / inf). The loops now iterate `scenario.meta["vehicles"]`, mirroring the GPEM
error-model path. These tests pin both halves of the contract:

  1. STRICT GENERALIZATION (the regression gate): a V2V4Real-shaped scenario
     (`vehicles: [astuff, tesla]`) requests EXACTLY the same per-source lookups
     and builds EXACTLY the same participant-id-keyed maps as the old hardcoded
     loop. Nothing about V2V4Real changes.

  2. DAIR now resolves: a `vehicles: [veh, inf]` scenario substitutes veh/inf
     into the `{vehicle}` detector name and keys its per-source maps on the
     DAIR participant ids (veh=3, inf=4), which match the runtime
     `_source_id_of` encoding.

The lifecycle / gate loaders are stubbed so the tests are hermetic (no CSV on
disk needed) and so we can capture exactly which lookups each fleet drives.

Run: PYTHONPATH=src python -m pytest tests/test_dair_per_source_generalization.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import lifecycle_params
import birth_gate_curves
import v2v4real_replay


def _write_min_scenario(tmp_path: Path, vehicles, dets_per_vehicle):
    """Write a 1-frame scenario dir (meta + ego/det/gt csvs) for `vehicles`.

    Each vehicle gets one self-detection-free detection so the per-source path
    is exercised. Positions are spread out so nothing matches across sources.
    """
    d = tmp_path / "val__0001"
    d.mkdir()
    (d / "meta.yaml").write_text(
        "split: val\nscenario: '0001'\nvehicles: [%s]\n" % ", ".join(vehicles)
    )
    # ego_state.csv — one ego row per vehicle.
    ego_hdr = ("timestamp,frame_id,vehicle_id,x,y,z,yaw,vx,vy,length,width,height\n")
    ego_rows = "".join(
        f"0.0,0,{v},{i*100.0},0.0,0.0,0.0,0.0,0.0,4.0,2.0,1.5\n"
        for i, v in enumerate(vehicles)
    )
    (d / "ego_state.csv").write_text(ego_hdr + ego_rows)
    # detections.csv — one detection per vehicle.
    det_hdr = ("timestamp,vehicle_id,frame_id,det_idx,x,y,z,yaw,vx,vy,"
               "length,width,height,score,label\n")
    det_rows = "".join(
        f"0.0,{v},0,0,{i*100.0+10.0},0.0,0.0,0.0,0.0,0.0,4.0,2.0,1.5,0.9,car\n"
        for i, v in enumerate(vehicles)
    )
    (d / "detections.csv").write_text(det_hdr + det_rows)
    # gt_objects.csv — empty (header only) is fine.
    (d / "gt_objects.csv").write_text(
        "ass_id,obj_id,vehicle_id,frame_id,x,y,z,yaw,length,width,height,label\n"
    )
    return d


def _run_capturing(scenario_dir, detector_name, monkeypatch):
    """Run a 1-stream scenario, capturing every lifecycle/gate lookup name.

    Returns (lifecycle_lookups, gate_lookups) — the exact `detector_name`
    strings passed to the two loaders, in call order.
    """
    life_calls = []
    gate_calls = []

    def fake_load_lifecycle(name, lifecycle_csv=None):
        life_calls.append(name)
        # Distinct values per source so the per-source BLEND activates and we
        # can assert the participant-id keys downstream.
        per = {
            "pointpillar_v2v4real_astuff": -0.40, "pointpillar_v2v4real_tesla": -0.55,
            "dair_pp_veh": -0.2323, "dair_pp_inf": -0.1261,
        }
        miss = per.get(name, -0.5)
        return lifecycle_params.LifecycleParams(
            detector=name, p_birth=0.0, log_lr_miss=miss,
            kill_log_odds=-3.0, confirm_log_odds=0.8473,
            tape_score_miss_decay=0.5,
        )

    def fake_load_gate(name, alpha, curve_csv=None, polar_dir=None):
        gate_calls.append(name)
        return birth_gate_curves.BirthGateCurve(
            detector=name, alpha=float(alpha), fit_type="quadratic",
            a_quad=0.0, b_quad=0.0, c_quad=0.1, a_lin=0.0, b_lin=0.1,
            n_range_bins=20,
        )

    monkeypatch.setattr(lifecycle_params, "load_lifecycle_params", fake_load_lifecycle)
    monkeypatch.setattr(birth_gate_curves, "load_birth_gate_curve", fake_load_gate)

    cfg = {
        "detector_name": detector_name,
        "localizer_name": "rtk_v2v4real",
        "self_report_egos": False,
        "lidar_range": [-1e4, -1e4, -1e4, 1e4, 1e4, 1e4],
        "gt_range":    [-1e4, -1e4, -1e4, 1e4, 1e4, 1e4],
        "detector_max_range": 200.0,
        "score_threshold": 0.0,
        "lifecycle_mode": "log_odds",
        "data_driven_lifecycle": True,
        "data_driven_gate_alpha": 99.0,
        "data_driven_gate_fit": "quadratic",
        "streams_keep": ["sabre_static"],
        "record_tape_for": ["sabre_static"],
    }
    v2v4real_replay.run_scenario(str(scenario_dir), cfg)
    return life_calls, gate_calls


def test_v2v4real_unchanged(tmp_path, monkeypatch, capsys):
    """V2V4Real ([astuff, tesla]) drives EXACTLY the legacy lookups + maps.

    This is the regression gate: the generalized loop, fed the V2V4Real meta,
    must reproduce the old hardcoded `("astuff", "tesla")` behaviour exactly.
    """
    sc = _write_min_scenario(tmp_path, ["astuff", "tesla"], 1)
    life, gate = _run_capturing(sc, "pointpillar_v2v4real_{vehicle}", monkeypatch)

    # Both loaders were asked for astuff THEN tesla — same order as the old loop.
    assert life == ["pointpillar_v2v4real_astuff", "pointpillar_v2v4real_tesla"]
    assert gate == ["pointpillar_v2v4real_astuff", "pointpillar_v2v4real_tesla"]

    out = capsys.readouterr().out
    # The per-source BLEND must be keyed on the canonical V2V4Real ids (1, 2).
    assert "per-source BLEND active" in out
    assert "log_lr_miss_by_source={1: -0.4, 2: -0.55}" in out


def test_dair_now_resolves(tmp_path, monkeypatch, capsys):
    """DAIR ([veh, inf]) substitutes veh/inf and keys maps on ids 3/4."""
    sc = _write_min_scenario(tmp_path, ["veh", "inf"], 1)
    life, gate = _run_capturing(sc, "dair_pp_{vehicle}", monkeypatch)

    assert life == ["dair_pp_veh", "dair_pp_inf"]
    assert gate == ["dair_pp_veh", "dair_pp_inf"]

    out = capsys.readouterr().out
    assert "per-source BLEND active" in out
    # veh=3, inf=4 in VEHICLE_PARTICIPANT_IDS → the runtime encoding the
    # tracker uses to recover the source. The per-source miss-decay map must
    # therefore key on 3/4 with the veh/inf log_lr_miss the loader returned.
    assert "log_lr_miss_by_source={3: -0.2323, 4: -0.1261}" in out


def test_dair_participant_ids_distinct_and_stable():
    """veh/inf are registered with unique small ids that won't collide with
    astuff/tesla (1/2) under the participant_id * max_id encoding."""
    ids = v2v4real_replay.VEHICLE_PARTICIPANT_IDS
    assert ids["veh"] == 3 and ids["inf"] == 4
    assert len(set(ids.values())) == len(ids)            # no collisions
    import sensor_fusion
    # ids stay well below max_id so encoding is collision-free.
    assert all(v < sensor_fusion.max_id for v in ids.values())
