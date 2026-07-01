#!/usr/bin/env python3
"""Build the drift-corrected, co-observation-merged V2V4Real GT ("Ours-merged")
DIRECTLY FROM THE RAW YAML, in the official KITTI frame.

Why a new builder (supersedes build_merged_gt.py's cmr_export path): the old
builder re-merged from cmr_export world boxes and transformed astuff by applying
*tesla's* rotation to astuff's world coords. astuff actually needs the FULL
inter-CAV transform (inv(T_tesla) @ T_astuff, using BOTH CAVs' raw 4x4
lidar_poses) to land with the correct vertical; the cmr_export path corrupted
astuff (and hence every co-observed average), so only ~13% of official boxes
were reproduced. This builder reconstructs the GT the way the official paper_gt
was built and reproduces it (no-drift, no-fuse) box-for-box.

Per scenario (cav0 = tesla = raw dir "0", cav1 = astuff = raw dir "1"):
* Read each CAV's per-frame objects from raw yaml: ``location`` (that CAV's LiDAR
  frame), ``angle`` (degrees), ``extent`` (half-dims), ``ass_id`` (astuff ass_id
  -> matching tesla obj_id), ``obj_type``.
* tesla objects: already in tesla LiDAR frame.
* astuff objects: p_tesla = inv(T_tesla) @ T_astuff @ [loc_astuff, 1]. With
  --drift, the astuff world point is first shifted by the OU drift (tx, ty)
  (origin-invariant translation; the two world frames differ only by an origin
  offset, verified) before mapping into tesla.
* Cluster across CAVs via union-find over ass_id (stable per-scenario track id),
  centroid-average co-observed clusters (mean position/dims, circular-mean yaw),
  keep singletons.
* Emit KITTI: (X, Y, Z) = (p_tesla_x, p_tesla_z, p_tesla_y); (h, w, l) =
  2*(extent_z, extent_y, extent_x); ry = radians(angle_yaw) [+ inter-CAV yaw].

--no-drift builds the same with t=0 (the paper_gt-reproduction validation case).
Output: <out>/{0000..0008}.txt  (default data/v2v4real_inputs/ours_merged_gt_raw/kitti_labels).
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "configs"))
sys.path.insert(0, str(REPO / "src"))
from v2v4real_seq_map import SEQ_TO_SCENARIO, SEQ_TO_NUM_FRAMES  # noqa: E402
from augment_v2v4real_with_cav_self_reports import emit_kitti_gt_row  # noqa: E402
from v2v4real_vehicles import LIDAR_MOUNT_HEIGHT_M, DEFAULT_DIMS  # noqa: E402
from eval_config import TESLA_GT_ID, ASTUFF_GT_ID  # noqa: E402

DEFAULT_RAW_BASE = Path("/media/rave/eddie_drive1/v2v4real/test")
DEFAULT_DRIFT = REPO / "results" / "GT_MERGE_RTK" / "drift"
DEFAULT_OUT = REPO / "data" / "v2v4real_inputs" / "ours_merged_gt_raw" / "kitti_labels"
CAV_DIR = {"tesla": "0", "astuff": "1"}


class UnionFind:
    def __init__(self):
        self.parent: dict = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def circular_mean(angles) -> float:
    return math.atan2(sum(math.sin(a) for a in angles) / len(angles),
                      sum(math.cos(a) for a in angles) / len(angles))


def load_cav(scen_raw: str, cav: str, raw_base: Path):
    """frame -> (T 4x4 lidar->world, {obj_id: (loc(3), yaw_rad, dims(l,w,h), ass_id)})."""
    files = sorted(glob.glob(str(raw_base / scen_raw / CAV_DIR[cav] / "*.yaml")))
    if not files:
        raise SystemExit(f"no raw yaml for {scen_raw}/{CAV_DIR[cav]} under {raw_base}")
    out: dict[int, tuple] = {}
    for fr, f in enumerate(files):
        d = yaml.load(open(f), Loader=yaml.Loader)
        T = np.asarray(d["lidar_pose"], dtype=float)
        objs = {}
        for oid, o in (d.get("vehicles") or {}).items():
            # V2V4Real annotates all vehicles (Car/Van/Truck/...) and the
            # official single-class GT scores them all as "Car"; keep them all.
            loc = np.asarray(o["location"], dtype=float)
            ex = o["extent"]                               # half-dims (x,y,z)
            dims = (2 * ex[0], 2 * ex[1], 2 * ex[2])        # l, w, h
            yaw = math.radians(float(o["angle"][1]))        # angle is in degrees
            objs[int(oid)] = (loc, yaw, dims, int(o.get("ass_id", -1)))
        out[fr] = (T, objs)
    return out


def load_drift(scen: str, drift_dir: Path, use_drift: bool) -> dict[int, tuple]:
    out: dict[int, tuple] = defaultdict(lambda: (0.0, 0.0))
    if not use_drift:
        return out
    path = drift_dir / f"{scen}.csv"
    if not path.exists():
        raise SystemExit(f"missing drift series {path}; run estimate_intercav_rtk_drift.py")
    for r in csv.DictReader(open(path)):
        out[int(r["frame"])] = (float(r["tx_ou"]), float(r["ty_ou"]))
    return out


def _to_kitti(p_tesla: np.ndarray, yaw_tesla: float, dims: tuple) -> tuple:
    """tesla-LiDAR point + yaw + (l,w,h) -> (X,Y,Z,ry,l,w,h) KITTI row tuple."""
    l, w, h = dims
    return (p_tesla[0], p_tesla[2], p_tesla[1], yaw_tesla, l, w, h)


def _ego_boxcenter_lidar(vehicle: str) -> tuple:
    """Box-center of a CAV in its OWN LiDAR frame: at the lidar's (x,y)=(0,0),
    z = ground + h/2 = LIDAR_MOUNT_HEIGHT_M (the ground's lidar-z, negative) + h/2.
    Matches the ground-object z (~-1.1 m) and the augmenter's _emit_ego_self."""
    l, w, h = DEFAULT_DIMS[vehicle]
    return (0.0, 0.0, LIDAR_MOUNT_HEIGHT_M[vehicle] + h / 2.0, (l, w, h))


def build_scenario_rows(seq: int, scen: str, raw_base: Path, drift_dir: Path,
                        use_drift: bool, gt_range=None, add_egos: bool = False):
    scen_raw = scen.split("__", 2)[2]
    tesla = load_cav(scen_raw, "tesla", raw_base)
    astuff = load_cav(scen_raw, "astuff", raw_base)
    drift = load_drift(scen, drift_dir, use_drift)

    # Cluster across CAVs via ass_id (astuff obj -> tesla obj_id=ass_id).
    uf = UnionFind()
    for fr, (_T, objs) in astuff.items():
        for oid, (_loc, _yaw, _dims, ass) in objs.items():
            uf.find(("a", oid))
            if ass >= 0:
                uf.union(("a", oid), ("t", ass))
    for fr, (_T, objs) in tesla.items():
        for oid in objs:
            uf.find(("t", oid))
    roots = sorted({uf.find(n) for n in uf.parent}, key=lambda r: (str(r[0]), r[1]))
    node_tid = {n: i + 1 for i, r in enumerate(roots) for n in uf.parent if uf.find(n) == r}

    n_frames = SEQ_TO_NUM_FRAMES[seq]
    rows: list[str] = []
    n_merged = n_t = n_a = n_ego = 0
    for fr in range(n_frames):
        if fr not in tesla or fr not in astuff:
            continue
        T_t, t_objs = tesla[fr]
        T_a, a_objs = astuff[fr]
        Tt_inv = np.linalg.inv(T_t)
        tx, ty = drift[fr]
        yaw_rel = math.atan2((Tt_inv @ T_a)[1, 0], (Tt_inv @ T_a)[0, 0])

        # cluster -> list of (p_tesla(3), yaw_tesla, dims, src)
        clusters: dict[int, list] = defaultdict(list)
        srcs: dict[int, set] = defaultdict(set)
        for oid, (loc, yaw, dims, _ass) in t_objs.items():
            tid = node_tid[("t", oid)]
            clusters[tid].append((loc.astype(float), yaw, dims))
            srcs[tid].add("t")
        for oid, (loc, yaw, dims, _ass) in a_objs.items():
            tid = node_tid[("a", oid)]
            p_world = (T_a @ np.append(loc, 1.0))[:3]
            p_world[0] += tx
            p_world[1] += ty
            p_tesla = (Tt_inv @ np.append(p_world, 1.0))[:3]
            clusters[tid].append((p_tesla, yaw + yaw_rel, dims))
            srcs[tid].add("a")

        for tid in sorted(clusters):
            members = clusters[tid]
            pos = np.mean([m[0] for m in members], axis=0)
            yaw = circular_mean([m[1] for m in members])
            dims = tuple(np.mean([m[2] for m in members], axis=0))
            if gt_range is not None:                     # tesla-LiDAR-frame box
                xn, yn, zn, xx, yx, zx = gt_range
                if not (xn <= pos[0] <= xx and yn <= pos[1] <= yx
                        and zn <= pos[2] <= zx):
                    continue
            rows.append(emit_kitti_gt_row(fr, tid, _to_kitti(pos, yaw, dims)))
            s = srcs[tid]
            n_merged += s == {"t", "a"}
            n_t += s == {"t"}
            n_a += s == {"a"}

        if add_egos:
            # The two CAVs are real cars that the OTHER CAV tracks, but are absent
            # from the released annotations. Add them from their OWN RTK pose
            # (the raw lidar_pose), drift-corrected for astuff (tesla = the
            # reference frame, so its own pose is exact). Same full inter-CAV
            # transform + drift pipeline as third-party objects.
            tx0, ty0, th0, tdims = _ego_boxcenter_lidar("tesla")
            rows.append(emit_kitti_gt_row(
                fr, TESLA_GT_ID, _to_kitti(np.array([tx0, ty0, th0]), 0.0, tdims)))
            ax0, ay0, az0, adims = _ego_boxcenter_lidar("astuff")
            p_world = (T_a @ np.array([ax0, ay0, az0, 1.0]))[:3]
            p_world[0] += tx
            p_world[1] += ty
            p_a = (Tt_inv @ np.append(p_world, 1.0))[:3]
            if gt_range is None or (gt_range[0] <= p_a[0] <= gt_range[3]
                                    and gt_range[1] <= p_a[1] <= gt_range[4]
                                    and gt_range[2] <= p_a[2] <= gt_range[5]):
                rows.append(emit_kitti_gt_row(
                    fr, ASTUFF_GT_ID, _to_kitti(p_a, yaw_rel, adims)))
                n_ego += 1
            n_ego += 1   # tesla ego always emitted
    return rows, dict(merged=n_merged, tesla_only=n_t, astuff_only=n_a, ego=n_ego)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-base", type=Path, default=DEFAULT_RAW_BASE)
    ap.add_argument("--drift-dir", type=Path, default=DEFAULT_DRIFT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-drift", action="store_true",
                    help="t=0 (validation: should reproduce paper_gt closely)")
    ap.add_argument("--gt-range", type=float, nargs=6, default=None,
                    metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
                    help="tesla-LiDAR-frame box filter (the replay GT_RANGE is "
                         "-100 -40 -5 100 40 3); default None = keep whole world.")
    ap.add_argument("--add-egos", action="store_true",
                    help="Add both CAVs as GT objects from their own RTK pose "
                         "(astuff drift-corrected); reserved ids 90001/90002.")
    args = ap.parse_args()
    use_drift = not args.no_drift
    args.out.mkdir(parents=True, exist_ok=True)

    total = 0
    print(f"{'seq':>3s} {'scenario':38s} {'rows':>6s} {'merged':>7s} {'tOnly':>6s} "
          f"{'aOnly':>6s} {'ego':>5s}")
    for seq in sorted(SEQ_TO_SCENARIO):
        scen = SEQ_TO_SCENARIO[seq]
        rows, st = build_scenario_rows(seq, scen, args.raw_base, args.drift_dir,
                                       use_drift, args.gt_range, args.add_egos)
        (args.out / f"{seq:04d}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
        total += len(rows)
        print(f"{seq:>3d} {scen[7:45]:38s} {len(rows):>6d} {st['merged']:>7d} "
              f"{st['tesla_only']:>6d} {st['astuff_only']:>6d} {st['ego']:>5d}")
    print(f"\n[{'drift-corrected' if use_drift else 'NO-DRIFT (validation)'}] "
          f"wrote {total} rows across 9 sequences to {args.out}")


if __name__ == "__main__":
    main()
