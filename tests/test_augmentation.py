"""Adversarial tests for the hardened ego-augmentation script.

Target: ``cmr/scripts/augment_v2v4real_with_cav_self_reports.py``.

These tests prove the hardening promises hold (fail-loud on missing dirs,
frame-count mismatch, duplicate-disagreement, missing per-frame poses,
per-vehicle dim drift, per-vehicle mount selection) by monkeypatching the
script's module-level constants to point at synthetic in-tempdir fixtures.

Run from repo root:
    pytest tests/test_augmentation.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# Conventional sys.path setup: src + scripts + configs.
REPO = Path(__file__).resolve().parent.parent
for sub in ("src", "scripts", "configs"):
    sys.path.insert(0, str(REPO / sub))

import augment_v2v4real_with_cav_self_reports as aug  # noqa: E402
from v2v4real_seq_map import (  # noqa: E402
    SCENARIO_TO_SEQ,
    SEQ_TO_SCENARIO,
)
from v2v4real_vehicles import (  # noqa: E402
    DEFAULT_DIMS,
    DIMS_ASSERTION_TOL_M,
    LIDAR_MOUNT_HEIGHT_M,
)


# ── Fixture builders ────────────────────────────────────────────────────────
EGO_CSV_HEADER = (
    "timestamp,vehicle_id,frame_id,ass_id,obj_id,x,y,z,yaw,"
    "length,width,height,label\n"
)


def _ego_row(vehicle: str, fid: int, *, x=0.0, y=0.0, z=0.0, yaw=0.0,
             l=None, w=None, h=None, label="car") -> str:
    """One ego_state.csv row. Defaults use DEFAULT_DIMS for the vehicle."""
    dl, dw, dh = DEFAULT_DIMS[vehicle]
    if l is None: l = dl
    if w is None: w = dw
    if h is None: h = dh
    return (f"0.0,{vehicle},{fid},0,0,{x},{y},{z},{yaw},"
            f"{l},{w},{h},{label}\n")


def _write_ego_csv(scen_dir: Path, *, tesla_frames, astuff_frames,
                   tesla_kwargs=None, astuff_kwargs=None,
                   extra_rows=()):
    """Build a scenario dir with an ego_state.csv that lists tesla/astuff over
    the provided frame iterables (use None to skip that vehicle)."""
    scen_dir.mkdir(parents=True, exist_ok=True)
    body = [EGO_CSV_HEADER.strip()]
    tesla_kwargs = tesla_kwargs or {}
    astuff_kwargs = astuff_kwargs or {}
    if tesla_frames is not None:
        for fid in tesla_frames:
            body.append(_ego_row("tesla", fid, **tesla_kwargs).rstrip())
    if astuff_frames is not None:
        for fid in astuff_frames:
            body.append(_ego_row("astuff", fid, **astuff_kwargs).rstrip())
    for row in extra_rows:
        body.append(row.rstrip())
    (scen_dir / "ego_state.csv").write_text("\n".join(body) + "\n")


def _patch_paths(monkeypatch, root: Path):
    """Redirect all augmenter I/O to a tempdir tree."""
    (root / "cmr_export").mkdir()
    (root / "ab3dmot_dets").mkdir()
    (root / "ab3dmot_gt").mkdir()
    (root / "out_gt").mkdir()
    monkeypatch.setattr(aug, "CMR_EXPORT", root / "cmr_export")
    monkeypatch.setattr(aug, "AB3DMOT_DETS", root / "ab3dmot_dets")
    monkeypatch.setattr(aug, "AB3DMOT_GT", root / "ab3dmot_gt")
    monkeypatch.setattr(aug, "OUT_GT", root / "out_gt")
    monkeypatch.setattr(aug, "OUT_DETS", root / "ab3dmot_dets" / "out_dets")
    monkeypatch.setattr(aug, "OUT_LF_DETS", root / "ab3dmot_dets" / "out_lf")


# ── 1. map_scenarios_to_seqs: missing dir ───────────────────────────────────
def test_map_scenarios_to_seqs_raises_on_missing_dir(tmp_path, monkeypatch):
    """Declared scenario without a matching CMR_EXPORT directory must raise."""
    _patch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(aug, "SCENARIO_TO_SEQ", {"ghost_scenario": 0})
    monkeypatch.setattr(aug, "SEQ_TO_NUM_FRAMES", {0: 5})

    with pytest.raises(FileNotFoundError) as ei:
        aug.map_scenarios_to_seqs()
    assert "ghost_scenario" in str(ei.value)


# ── 2. map_scenarios_to_seqs: frame count mismatch ──────────────────────────
def test_map_scenarios_to_seqs_raises_on_frame_count_mismatch(tmp_path, monkeypatch):
    """Scenario exists but ego_state.csv frame count disagrees with seqmap."""
    _patch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(aug, "SCENARIO_TO_SEQ", {"scenA": 0})
    monkeypatch.setattr(aug, "SEQ_TO_NUM_FRAMES", {0: 10})
    _write_ego_csv(tmp_path / "cmr_export" / "scenA",
                   tesla_frames=range(6), astuff_frames=range(6))

    with pytest.raises(ValueError) as ei:
        aug.map_scenarios_to_seqs()
    msg = str(ei.value)
    assert "tesla frames go 0..5" in msg, msg


# ── 3. read_poses: disagreeing duplicate ────────────────────────────────────
def test_read_poses_raises_on_disagreeing_duplicate(tmp_path):
    """Two rows for (tesla, 0) with different x must raise."""
    scen = tmp_path / "scen"
    _write_ego_csv(
        scen,
        tesla_frames=[0],
        astuff_frames=[0],
        extra_rows=[_ego_row("tesla", 0, x=99.0)],
    )

    with pytest.raises(ValueError) as ei:
        aug.read_poses(scen / "ego_state.csv")
    msg = str(ei.value)
    assert "duplicate" in msg and "disagrees" in msg, msg


# ── 4. read_poses: silent when duplicate agrees ─────────────────────────────
def test_read_poses_silent_when_duplicate_agrees(tmp_path):
    """Two byte-identical rows for (tesla, 0) → no raise, one entry returned."""
    scen = tmp_path / "scen"
    _write_ego_csv(
        scen,
        tesla_frames=[0],
        astuff_frames=[0],
        extra_rows=[_ego_row("tesla", 0)],   # identical to the first tesla row
    )
    poses = aug.read_poses(scen / "ego_state.csv")
    assert ("tesla", 0) in poses
    assert ("astuff", 0) in poses
    assert len(poses) == 2


# ── 5. _emit_ego_self: drifted dims must raise ──────────────────────────────
def test_emit_ego_self_raises_on_dim_drift_above_tol():
    """Length drift > DIMS_ASSERTION_TOL_M must raise with 'dims drift' + name."""
    # tesla default length 4.97; pass 4.0 → drift 0.97 m > 0.05 tol.
    pose = (0.0, 0.0, 0.0, 0.0, 4.0, 1.96, 1.44)
    with pytest.raises(ValueError) as ei:
        aug._emit_ego_self("tesla", pose)
    msg = str(ei.value)
    assert "dims drift" in msg and "tesla" in msg, msg


# ── 6. _emit_ego_self: within tol succeeds ──────────────────────────────────
def test_emit_ego_self_succeeds_within_tol():
    """Drift just under tol returns (0, mount + h/2, 0, 0, l, w, h)."""
    l = 4.97 + 0.04   # drift 0.04 < 0.05 tol
    w = 1.96
    h = 1.44
    pose = (1.0, 2.0, 3.0, 0.7, l, w, h)   # x/y/z/yaw should be IGNORED
    out = aug._emit_ego_self("tesla", pose)
    expected_y = LIDAR_MOUNT_HEIGHT_M["tesla"] + h / 2.0
    assert out == (0.0, expected_y, 0.0, 0.0, l, w, h)


# ── 7. world_to_local_kitti: target == ego → at origin ──────────────────────
def test_world_to_local_kitti_self_is_origin():
    """target == ego must yield (0, dz + h/2, 0, 0, l, w, h) with dz = tz - ez.

    world_to_local_kitti uses the canonical V2V4Real Y convention
    ``kitti_Y = dz + th/2`` (dz = target_world_z - ego_world_z), matching the
    paper-GT-validated helpers build_merged_gt.world_to_kitti_full and
    dump_tracks_to_kitti_mot.global_to_kitti (both ``lid[2] + h/2``, no separate
    LIDAR_MOUNT term — the mount offset is already carried by dz in the shared
    rehydrated world frame). When target == ego, dz == 0 so Y_cam == h/2.
    (The old ``mount + h/2`` expectation double-counted the mount offset; see the
    function's docstring, which documents that re-adding LIDAR_MOUNT was verified
    SPURIOUS against DMS/GT.)"""
    pose = (10.0, 5.0, 0.2, 0.3, 4.97, 1.96, 1.44)
    out = aug.world_to_local_kitti(pose, pose, "tesla")
    expected_y = 1.44 / 2.0   # dz = tz - ez = 0 for self, so Y = h/2
    # X_cam and Z_cam should be exact zero (no float accumulation other than
    # cos/sin of identical eyaw applied to (0, 0)).
    assert math.isclose(out[0], 0.0, abs_tol=1e-12)
    assert math.isclose(out[2], 0.0, abs_tol=1e-12)
    assert math.isclose(out[1], expected_y, abs_tol=1e-12)
    # Yaw delta wraps to 0.
    assert math.isclose(out[3], 0.0, abs_tol=1e-12)
    # Dims pass through.
    assert (out[4], out[5], out[6]) == (4.97, 1.96, 1.44)


# ── 8. world_to_local_kitti: round-trip rotation by -eyaw ───────────────────
def test_world_to_local_kitti_round_trip():
    """X_cam / Z_cam must match a hand-computed rotation by -eyaw of (dx, dy)."""
    ego = (3.0, 4.0, 0.0, 0.5, 4.97, 1.96, 1.44)
    tgt = (10.0, -2.0, 0.0, 1.2, 5.18, 2.03, 1.77)
    out = aug.world_to_local_kitti(tgt, ego, "tesla")
    dx = tgt[0] - ego[0]
    dy = tgt[1] - ego[1]
    c = math.cos(-ego[3]); s = math.sin(-ego[3])
    expected_X = c * dx - s * dy
    expected_Z = s * dx + c * dy
    assert math.isclose(out[0], expected_X, abs_tol=1e-9)
    assert math.isclose(out[2], expected_Z, abs_tol=1e-9)
    # Y_cam = dz + h/2 with dz = tz - ez (canonical V2V4Real convention, no
    # separate mount term — it is baked into dz; see world_to_kitti_full /
    # global_to_kitti and the function docstring). Here tz == ez == 0 so dz == 0.
    expected_Y = (tgt[2] - ego[2]) + tgt[6] / 2.0
    assert math.isclose(out[1], expected_Y, abs_tol=1e-12)


# ── 9. per-vehicle mount selection ──────────────────────────────────────────
def test_world_to_local_kitti_uses_per_vehicle_mount():
    """Passing ego_vehicle='tesla' vs 'astuff' must yield the same X_cam/Z_cam
    (rotation only depends on ego yaw) but POTENTIALLY different Y_cam.

    Today both vehicles share LIDAR_MOUNT_HEIGHT_M = -1.87 so Y_cam is also
    equal; this test passes today and would fail loudly the day the config
    decouples per-vehicle mount heights — which is exactly what we want, since
    a silent change there would corrupt every emitted CAV self-report Y.
    """
    ego = (3.0, 4.0, 0.0, 0.5, 4.97, 1.96, 1.44)
    tgt = (10.0, -2.0, 0.0, 1.2, 5.18, 2.03, 1.77)
    out_t = aug.world_to_local_kitti(tgt, ego, "tesla")
    out_a = aug.world_to_local_kitti(tgt, ego, "astuff")
    # X_cam, Z_cam identical (per-vehicle mount only enters Y_cam).
    assert math.isclose(out_t[0], out_a[0], abs_tol=1e-12)
    assert math.isclose(out_t[2], out_a[2], abs_tol=1e-12)
    # Y_cam: equal today because mounts coincide; the assertion below will
    # fail when configs/v2v4real_vehicles.py decouples them — intentional.
    delta_mount = (LIDAR_MOUNT_HEIGHT_M["tesla"]
                   - LIDAR_MOUNT_HEIGHT_M["astuff"])
    assert math.isclose(out_t[1] - out_a[1], delta_mount, abs_tol=1e-12)


# ── 10. main(): frame completeness fail ────────────────────────────────────-
def test_frame_completeness_raises_on_missing_ego_frame(tmp_path, monkeypatch):
    """Tesla has frames 0..5; astuff is missing frame 3 → main() must raise
    naming the missing vehicle."""
    _patch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(aug, "SCENARIO_TO_SEQ", {"scenA": 0})
    monkeypatch.setattr(aug, "SEQ_TO_NUM_FRAMES", {0: 6})

    astuff_frames = [0, 1, 2, 4, 5]   # missing 3
    _write_ego_csv(tmp_path / "cmr_export" / "scenA",
                   tesla_frames=range(6),
                   astuff_frames=astuff_frames)

    with pytest.raises(ValueError) as ei:
        aug.main()
    msg = str(ei.value)
    assert "missing" in msg and "astuff" in msg, msg


# ── 11. SCENARIO_TO_SEQ ↔ SEQ_TO_SCENARIO bijection ─────────────────────────
def test_seq_to_scenario_bijection():
    """The two halves of the explicit map must round-trip."""
    assert len(SCENARIO_TO_SEQ) == len(SEQ_TO_SCENARIO)
    for scen, seq in SCENARIO_TO_SEQ.items():
        assert SEQ_TO_SCENARIO[seq] == scen, (
            f"forward: {scen}→{seq} but reverse: {seq}→{SEQ_TO_SCENARIO[seq]}"
        )
    for seq, scen in SEQ_TO_SCENARIO.items():
        assert SCENARIO_TO_SEQ[scen] == seq, (
            f"reverse: {seq}→{scen} but forward: {scen}→{SCENARIO_TO_SEQ[scen]}"
        )
