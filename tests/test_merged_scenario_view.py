"""Regression tests for input_bucket.merged_scenario_view per-tree rehydration.

Each per-vehicle cmr_export tree (astuff/, tesla/) is stored in its OWN centred
frame — its meta.yaml ``center_world`` has been subtracted from every x/y/z — and
the two CAVs' centroids differ by tens of metres. ``merged_view_from_bucket`` must
add each tree's centroid back (re-centring onto a single shared anchor frame)
BEFORE concatenating, or the two CAVs' detections and GT land ~80 m apart and
cross-CAV association + GT matching collapse.

That exact bug (rehydration missing from the merged-view path while the S1-S3
import path had it) tanked S4 AMOTA from ~40 to single digits. These tests pin
the behavior so it can't silently regress again:

  * the synthetic tests build two fake per-vehicle trees with known centroids and
    a shared physical object — they always run (no bucket on disk required) and
    assert the shared object's two annotations land on top of each other after merge;
  * the integration test, if the real pp_score0 bucket is present, asserts the
    overlapping-FOV GT actually co-locates (a healthy fraction within the 2 m
    clustering gate) — pre-fix this was ~0.2%, post-fix ~50%.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from input_bucket.merged_scenario_view import merged_view_from_bucket  # noqa: E402

EGO_HDR = "timestamp,vehicle_id,frame_id,x,y,z,yaw,vx,vy,length,width,height"
DET_HDR = ("timestamp,vehicle_id,frame_id,det_idx,x,y,z,yaw,vx,vy,"
           "length,width,height,score,label")
GT_HDR = "timestamp,vehicle_id,frame_id,ass_id,obj_id,x,y,z,yaw,length,width,height,label"


def _write_tree(cmr_export: Path, vehicle: str, scen: str,
                center: tuple[float, float, float],
                world_objs: list[tuple[float, float, float]]) -> None:
    """Write one per-vehicle tree whose CSVs store (world - center) coordinates.

    ``world_objs`` are object positions in the SHARED world frame; each is stored
    relative to this tree's ``center`` (exactly as the upstream exporter does).
    The same object list written into two trees with different centres is the
    cross-CAV co-location the merge must recover.
    """
    cx, cy, cz = center
    d = cmr_export / vehicle / scen
    d.mkdir(parents=True)
    (d / "meta.yaml").write_text(
        f"scenario: {scen}\n"
        f"vehicles: [{vehicle}]\n"
        f"center_world: [{cx:.6f}, {cy:.6f}, {cz:.6f}]\n"
        f"n_detections_total: {len(world_objs)}\n"
        f"n_gt_total: {len(world_objs)}\n"
    )
    # ego: one row, the CAV itself at world origin (stored relative to center).
    (d / "ego_state.csv").write_text(
        EGO_HDR + "\n" +
        f"0.0,{vehicle},0,{-cx:.6f},{-cy:.6f},{-cz:.6f},0.0,0.0,0.0,4.5,2.0,1.6\n"
    )
    det_lines, gt_lines = [DET_HDR], [GT_HDR]
    for i, (wx, wy, wz) in enumerate(world_objs):
        sx, sy, sz = wx - cx, wy - cy, wz - cz   # stored = world - center
        det_lines.append(
            f"0.0,{vehicle},0,{i},{sx:.6f},{sy:.6f},{sz:.6f},0.0,0.0,0.0,"
            f"4.5,2.0,1.6,0.9,car")
        gt_lines.append(
            f"0.0,{vehicle},0,-1,{i},{sx:.6f},{sy:.6f},{sz:.6f},0.0,4.5,2.0,1.6,car")
    (d / "detections.csv").write_text("\n".join(det_lines) + "\n")
    (d / "gt_objects.csv").write_text("\n".join(gt_lines) + "\n")


def _rows(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


@pytest.fixture
def synthetic_bucket(tmp_path: Path) -> Path:
    """A bucket with two per-vehicle trees in DIFFERENT centred frames.

    Both trees observe the SAME shared object at world (10, 20, 0); each also has
    one object only it sees. astuff and tesla centroids differ by (50, -60, 2),
    mimicking the real ~81 m planar delta.
    """
    scen = "scenA"
    cmr_export = tmp_path / "cmr_export"
    shared = (10.0, 20.0, 0.0)
    _write_tree(cmr_export, "astuff", scen, center=(100.0, 200.0, 3.0),
                world_objs=[shared, (40.0, -5.0, 0.0)])           # 1 shared + 1 own
    _write_tree(cmr_export, "tesla", scen, center=(150.0, 260.0, 1.0),
                world_objs=[shared, (-30.0, 70.0, 0.0), (80.0, 12.0, 0.0)])  # 1 shared + 2 own
    return tmp_path


def test_rehydration_aligns_shared_object(synthetic_bucket: Path):
    """The object both CAVs saw must land at the SAME merged position.

    Without per-tree rehydration the astuff copy is offset from the tesla copy by
    the centroid delta (~81 m in production) — the regression that broke S4.
    """
    with merged_view_from_bucket(synthetic_bucket) as merged:
        scen_dir = next(merged.glob("test__*"))
        gt = _rows(scen_dir / "gt_objects.csv")

    a = [(float(r["x"]), float(r["y"])) for r in gt if r["vehicle_id"] == "astuff"]
    t = [(float(r["x"]), float(r["y"])) for r in gt if r["vehicle_id"] == "tesla"]
    # The shared world object must have an astuff row within ~1cm of a tesla row.
    gaps = [min(((ax - tx) ** 2 + (ay - ty) ** 2) ** 0.5 for tx, ty in t)
            for ax, ay in a]
    assert min(gaps) < 0.01, (
        f"no astuff GT object co-locates with a tesla GT object after merge "
        f"(min gap {min(gaps):.3f} m). Per-tree center_world rehydration is "
        f"missing/broken — astuff and tesla are in different frames."
    )


def test_both_vehicles_gt_concatenated(synthetic_bucket: Path):
    """Merged GT must contain BOTH vehicles' rows (not take-first).

    run_scenario keys per-ego GT by vehicle_id and clusters per-vehicle
    annotations; dropping one vehicle's GT silently zeroes its eval.
    """
    with merged_view_from_bucket(synthetic_bucket) as merged:
        scen_dir = next(merged.glob("test__*"))
        gt = _rows(scen_dir / "gt_objects.csv")
        dets = _rows(scen_dir / "detections.csv")

    vids_gt = {r["vehicle_id"] for r in gt}
    vids_det = {r["vehicle_id"] for r in dets}
    assert vids_gt == {"astuff", "tesla"}, f"GT missing a vehicle: {vids_gt}"
    assert vids_det == {"astuff", "tesla"}, f"dets missing a vehicle: {vids_det}"
    assert len(gt) == 5, f"expected 2 astuff + 3 tesla GT rows, got {len(gt)}"


def test_meta_center_world_is_anchor(synthetic_bucket: Path):
    """Merged meta.yaml center_world must be the shared anchor (tesla) centroid."""
    with merged_view_from_bucket(synthetic_bucket) as merged:
        scen_dir = next(merged.glob("test__*"))
        meta = (scen_dir / "meta.yaml").read_text()
    cw_line = next(l for l in meta.splitlines() if l.startswith("center_world:"))
    cw = [float(x) for x in cw_line.split("[")[1].split("]")[0].split(",")]
    assert cw == pytest.approx([150.0, 260.0, 1.0]), (
        f"merged center_world {cw} != tesla anchor [150, 260, 1]")


def test_real_bucket_cross_cav_gt_colocates():
    """Integration: on the real pp_score0 bucket, a healthy fleet-wide fraction of
    one CAV's GT must have the other CAV's GT within the 2 m clustering gate.

    Per-scenario overlap varies a lot with FOV geometry (observed ~3% to ~72%),
    so we aggregate across ALL scenarios — fleet-wide ~48% post-fix vs ~0.2%
    pre-fix (centroid artifact). Skipped on a fresh clone without the bucket.
    """
    bucket = REPO / "data" / "v2v4real_inputs" / "ours" / "detectors" / "pp_score0"
    if not (bucket / "cmr_export").is_dir():
        pytest.skip("pp_score0 bucket not on disk")

    total = matched = 0
    with merged_view_from_bucket(bucket) as merged:
        for scen_dir in merged.glob("test__*"):
            a_by_f: dict = defaultdict(list)
            t_by_f: dict = defaultdict(list)
            for r in _rows(scen_dir / "gt_objects.csv"):
                (a_by_f if r["vehicle_id"] == "astuff" else t_by_f)[
                    r["frame_id"]].append((float(r["x"]), float(r["y"])))
            for f, alist in a_by_f.items():
                tlist = t_by_f.get(f, [])
                for ax, ay in alist:
                    total += 1
                    if any((ax - tx) ** 2 + (ay - ty) ** 2 <= 4.0 for tx, ty in tlist):
                        matched += 1

    assert total > 0, "no astuff GT in merged view"
    frac = matched / total
    assert frac > 0.25, (
        f"only {frac:.1%} of astuff GT co-locates with tesla GT within 2 m "
        f"({matched}/{total}) fleet-wide. Expected ~48% — the per-tree "
        f"rehydration likely regressed (pre-fix this was ~0.2%)."
    )
