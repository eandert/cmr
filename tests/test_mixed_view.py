"""Regression tests for input_bucket.merged_scenario_view CROSS-BUCKET rehydration.

The S4 mixed-detector combos take the astuff per-vehicle tree from one detector
bucket and the tesla tree from ANOTHER (e.g. PP on astuff + CPft on tesla).
The two buckets' trees for the SAME scenario carry DIFFERENT meta.yaml
``center_world`` centroids, so ``mixed_view_from_buckets`` must rehydrate each
tree with its OWN centroid and re-centre both onto the tesla tree's centroid —
exactly the per-tree rehydration ``merged_view_from_bucket`` does within one
bucket (see tests/test_merged_scenario_view.py). Using either tree's centroid
for the other re-introduces the ~80 m S4 coordinate-drift bug.

The synthetic fixture builds TWO buckets, each with both vehicle trees and
per-bucket-per-vehicle centroids that all differ; a shared world object must
co-locate after the mixed merge, and the view must draw astuff from bucket A
and tesla from bucket B (each bucket's other tree is a decoy with different
row counts).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from input_bucket.merged_scenario_view import mixed_view_from_buckets  # noqa: E402

EGO_HDR = "timestamp,vehicle_id,frame_id,x,y,z,yaw,vx,vy,length,width,height"
DET_HDR = ("timestamp,vehicle_id,frame_id,det_idx,x,y,z,yaw,vx,vy,"
           "length,width,height,score,label")
GT_HDR = "timestamp,vehicle_id,frame_id,ass_id,obj_id,x,y,z,yaw,length,width,height,label"

SCEN = "scenA"
SHARED = (10.0, 20.0, 0.0)   # world object both CAVs annotate


def _write_tree(cmr_export: Path, vehicle: str, scen: str,
                center: tuple[float, float, float],
                world_objs: list[tuple[float, float, float]]) -> None:
    """Write one per-vehicle tree whose CSVs store (world - center) coordinates
    (same shape as tests/test_merged_scenario_view.py)."""
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


@pytest.fixture
def two_buckets(tmp_path: Path) -> tuple[Path, Path]:
    """Two detector buckets, each with BOTH vehicle trees, all four trees in
    DIFFERENT centred frames.

    The mixed view must take astuff from bucket A (2 GT rows) and tesla from
    bucket B (3 GT rows); the same-vehicle trees in the OTHER bucket are decoys
    with distinct row counts (4 and 5) so take-from-wrong-bucket is detectable.
    Centroid deltas mimic the real tens-of-metres offsets — and crucially the
    astuff centroid DIFFERS between buckets (different exporter runs), so
    rehydrating the bucket-A astuff tree with anything but its own meta.yaml
    centroid misplaces it.
    """
    bucket_a = tmp_path / "bucketA"
    bucket_b = tmp_path / "bucketB"
    # bucket A: the astuff tree we want (+ decoy tesla tree, 4 objs)
    _write_tree(bucket_a / "cmr_export", "astuff", SCEN, center=(100.0, 200.0, 3.0),
                world_objs=[SHARED, (40.0, -5.0, 0.0)])
    _write_tree(bucket_a / "cmr_export", "tesla", SCEN, center=(-7.0, 13.0, 0.5),
                world_objs=[SHARED, (1.0, 1.0, 0.0), (2.0, 2.0, 0.0), (3.0, 3.0, 0.0)])
    # bucket B: the tesla tree we want (+ decoy astuff tree, 5 objs, with a
    # DIFFERENT astuff centroid than bucket A's astuff tree)
    _write_tree(bucket_b / "cmr_export", "tesla", SCEN, center=(150.0, 260.0, 1.0),
                world_objs=[SHARED, (-30.0, 70.0, 0.0), (80.0, 12.0, 0.0)])
    _write_tree(bucket_b / "cmr_export", "astuff", SCEN, center=(-55.0, 42.0, 9.0),
                world_objs=[SHARED, (4.0, 4.0, 0.0), (5.0, 5.0, 0.0),
                            (6.0, 6.0, 0.0), (7.0, 7.0, 0.0)])
    return bucket_a, bucket_b


def _rows(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def test_cross_bucket_rehydration_aligns_shared_object(two_buckets):
    """The object both CAVs saw must land at the SAME mixed-view position even
    though the astuff and tesla trees come from different buckets with
    different per-tree centroids."""
    bucket_a, bucket_b = two_buckets
    with mixed_view_from_buckets(bucket_a, bucket_b) as merged:
        scen_dir = next(merged.glob("test__*"))
        gt = _rows(scen_dir / "gt_objects.csv")

    a = [(float(r["x"]), float(r["y"])) for r in gt if r["vehicle_id"] == "astuff"]
    t = [(float(r["x"]), float(r["y"])) for r in gt if r["vehicle_id"] == "tesla"]
    gaps = [min(((ax - tx) ** 2 + (ay - ty) ** 2) ** 0.5 for tx, ty in t)
            for ax, ay in a]
    assert min(gaps) < 0.01, (
        f"no astuff GT object co-locates with a tesla GT object after the mixed "
        f"merge (min gap {min(gaps):.3f} m). Per-tree center_world rehydration "
        f"is missing/broken across buckets — astuff and tesla are in different "
        f"frames."
    )


def test_mixed_view_takes_each_vehicle_from_its_own_bucket(two_buckets):
    """astuff must come from bucket A and tesla from bucket B — NOT the decoy
    same-scenario trees the other bucket also carries."""
    bucket_a, bucket_b = two_buckets
    with mixed_view_from_buckets(bucket_a, bucket_b) as merged:
        scen_dir = next(merged.glob("test__*"))
        gt = _rows(scen_dir / "gt_objects.csv")
        dets = _rows(scen_dir / "detections.csv")

    n_a = sum(1 for r in gt if r["vehicle_id"] == "astuff")
    n_t = sum(1 for r in gt if r["vehicle_id"] == "tesla")
    assert n_a == 2, f"astuff GT rows {n_a} != 2 (bucket A's astuff tree; decoy has 5)"
    assert n_t == 3, f"tesla GT rows {n_t} != 3 (bucket B's tesla tree; decoy has 4)"
    vids_det = {r["vehicle_id"] for r in dets}
    assert vids_det == {"astuff", "tesla"}, f"dets missing a vehicle: {vids_det}"


def test_mixed_meta_center_world_is_tesla_bucket_anchor(two_buckets):
    """Merged meta.yaml center_world must be the TESLA tree's centroid from the
    tesla bucket (bucket B), not bucket A's tesla decoy or either astuff tree."""
    bucket_a, bucket_b = two_buckets
    with mixed_view_from_buckets(bucket_a, bucket_b) as merged:
        scen_dir = next(merged.glob("test__*"))
        meta = (scen_dir / "meta.yaml").read_text()
    cw_line = next(l for l in meta.splitlines() if l.startswith("center_world:"))
    cw = [float(x) for x in cw_line.split("[")[1].split("]")[0].split(",")]
    assert cw == pytest.approx([150.0, 260.0, 1.0]), (
        f"merged center_world {cw} != bucket-B tesla anchor [150, 260, 1]")


def test_missing_vehicle_tree_raises(two_buckets, tmp_path):
    """A bucket without the needed per-vehicle tree must fail loudly, not merge
    a partial (single-frame-of-reference) view."""
    bucket_a, _ = two_buckets
    empty = tmp_path / "empty_bucket"
    (empty / "cmr_export").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        with mixed_view_from_buckets(bucket_a, empty):
            pass
