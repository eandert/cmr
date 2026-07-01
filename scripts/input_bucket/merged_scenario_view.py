"""Build a temporary merged-scenario view of a bucket's cmr_export.

The bucket layout is per-vehicle::

    <bucket>/cmr_export/
        astuff/<scenario>/{detections.csv, ego_state.csv, gt_objects.csv, meta.yaml}
        tesla/<scenario>/...

The log_odds tracker's ``run_scenario(scen_dir, cfg)`` expects a SINGLE
scenario dir containing one ``detections.csv`` with BOTH vehicles' rows,
one ``ego_state.csv`` with both vehicles' poses, and one ``gt_objects.csv``
with both vehicles' annotations. This module produces that view as a
temporary directory of ``test__<scenario>/`` merged CSVs.

Each per-vehicle tree is stored in its OWN centred frame (its meta.yaml
``center_world`` is subtracted from every x/y/z), and the two CAVs' centroids
differ by tens of metres. Before concatenating, every tree is re-centred onto a
single shared anchor frame (tesla if present) so the merged scenario is in one
coherent frame — otherwise the two CAVs' detections and GT land ~80 m apart and
cross-CAV association + matching collapse.

Usage::

    with merged_view_from_bucket(bucket_root) as merged_dir:
        scenarios = sorted(merged_dir.iterdir())
        for scen_dir in scenarios:
            result = v2v4real_replay.run_scenario(scen_dir, cfg)
            ...

The context manager cleans up the temp dir on exit.
"""
from __future__ import annotations

import contextlib
import csv
import shutil
import tempfile
from pathlib import Path
from typing import Iterator

import yaml


SUPPORTED_VEHICLES = ("astuff", "tesla")

# CSV columns that are absolute POSITIONS: they get the per-tree centroid offset
# added on merge so every vehicle tree lands in one shared frame. Everything else
# (yaw, dims, score, vx/vy, ids, timestamp, label) is frame-invariant.
_OFFSET_COLS = ("x", "y", "z")


def _load_center_world(scen_dir: Path) -> tuple[float, float, float]:
    """Read the scenario centroid the upstream exporter subtracted from every
    coordinate in this per-vehicle tree.

    Each vehicle tree (astuff/, tesla/) carries its OWN centroid in meta.yaml,
    and every x/y/z in that tree's CSVs is relative to it — the two CAVs' centroids
    differ by tens of metres. Callers must add it back before mixing rows across
    trees. Raises on a missing/malformed meta.yaml rather than silently falling
    back: a silent zero offset is exactly the coordinate-drift bug this guards
    against. Mirrors the same helper in import_ab3dmot_format.py (the S1-S3 path).
    """
    meta_path = scen_dir / "meta.yaml"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"missing meta.yaml at {meta_path}\n"
            f"  Fix: re-run mmdetection3d/tools/export_v2v4real_to_cmr.py with the "
            f"v0.4+ exporter (writes center_world to meta.yaml)."
        )
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    cw = meta.get("center_world")
    if cw is None or len(cw) != 3:
        raise ValueError(
            f"{meta_path} missing or invalid 'center_world' (got {cw!r}).\n"
            f"  Fix: regenerate this cmr_export tree with the v0.4+ exporter."
        )
    return float(cw[0]), float(cw[1]), float(cw[2])


def _merge_csv_rehydrated(
    out_path: Path,
    sources: list[tuple[Path, tuple[float, float, float]]],
) -> int:
    """Concatenate header-bearing CSVs into one shared-frame file.

    ``sources`` is ``[(csv_path, (dx, dy, dz)), ...]``; each source's (dx, dy, dz)
    offset is added to its x/y/z columns (looked up BY HEADER NAME, so the
    differing detections / ego_state / gt_objects schemas all work) before the
    rows are written. The header is emitted once, from the first source. Returns
    the total body row count (excluding the header).
    """
    present = [(p, off) for p, off in sources if p.is_file()]
    if not present:
        out_path.write_text("")
        return 0
    rows = 0
    writer = None
    with open(out_path, "w", newline="") as f_out:
        for path, (dx, dy, dz) in present:
            offs = {"x": dx, "y": dy, "z": dz}
            with open(path, newline="") as f_in:
                reader = csv.reader(f_in)
                header = next(reader, None)
                if header is None:
                    continue
                if writer is None:
                    writer = csv.writer(f_out)
                    writer.writerow(header)
                # Index x/y/z from THIS file's own header so a tree with a
                # different column order is still rehydrated correctly.
                col = {c: header.index(c) for c in _OFFSET_COLS if c in header}
                for row in reader:
                    if not row:
                        continue
                    for c, i in col.items():
                        row[i] = f"{float(row[i]) + offs[c]:.6f}"
                    writer.writerow(row)
                    rows += 1
    return rows


def _merge_meta_yaml(out_path: Path, sources: list[Path],
                     anchor_center: tuple[float, float, float]) -> None:
    """Produce a merged meta.yaml: take the first as base, replace ``vehicles``
    with the union, sum ``n_detections_total`` and ``n_gt_total``, and set
    ``center_world`` to ``anchor_center`` (the shared frame the merged CSVs were
    re-centred onto — see merged_view_from_bucket).

    We do a line-edit rather than parsing YAML to avoid the dep — the upstream
    files are simple key:value with one nested ``n_frames_per_vehicle`` dict.
    """
    if not sources:
        return
    base = sources[0]
    text = base.read_text() if base.is_file() else ""
    # Collect per-source totals to sum.
    totals: dict[str, int] = {"n_detections_total": 0, "n_gt_total": 0}
    vehicles: list[str] = []
    n_frames_lines: list[str] = []
    for src in sources:
        if not src.is_file():
            continue
        for raw in src.read_text().splitlines():
            for key in totals:
                if raw.startswith(f"{key}:"):
                    try:
                        totals[key] += int(raw.split(":", 1)[1].strip())
                    except ValueError:
                        pass
            if raw.startswith("vehicles:"):
                v_str = raw.split(":", 1)[1].strip().strip("[]").replace(" ", "")
                vehicles.extend(v.strip() for v in v_str.split(",") if v.strip())
    # Deduplicate vehicles + write the merged file.
    seen: set[str] = set()
    uniq = []
    for v in vehicles:
        if v in seen:
            continue
        seen.add(v); uniq.append(v)
    out_lines: list[str] = []
    for raw in text.splitlines():
        if raw.startswith("vehicles:"):
            out_lines.append(f"vehicles: [{', '.join(uniq)}]")
        elif raw.startswith("n_detections_total:"):
            out_lines.append(f"n_detections_total: {totals['n_detections_total']}")
        elif raw.startswith("n_gt_total:"):
            out_lines.append(f"n_gt_total: {totals['n_gt_total']}")
        elif raw.startswith("center_world:"):
            cx, cy, cz = anchor_center
            out_lines.append(f"center_world: [{cx:.6f}, {cy:.6f}, {cz:.6f}]")
        else:
            out_lines.append(raw)
    out_path.write_text("\n".join(out_lines) + ("\n" if text and not text.endswith("\n") else ""))


def _collect_scenarios(cmr_export_root: Path) -> list[str]:
    """Return scenario names present under ALL existing vehicle subdirs (intersection)."""
    sets: list[set[str]] = []
    for v in SUPPORTED_VEHICLES:
        vdir = cmr_export_root / v
        if not vdir.is_dir():
            continue
        sets.append({p.name for p in vdir.iterdir() if p.is_dir()})
    if not sets:
        return []
    common = set.intersection(*sets)
    return sorted(common)


@contextlib.contextmanager
def merged_view_from_trees(tree_by_vehicle: dict[str, Path]) -> Iterator[Path]:
    """Yield a temp directory of merged ``test__<scenario>/`` dirs, where EACH
    vehicle's per-vehicle scenario tree may live in a DIFFERENT location.

    ``tree_by_vehicle`` maps vehicle id -> directory containing that vehicle's
    ``<scenario>/{detections.csv, ego_state.csv, gt_objects.csv, meta.yaml}``
    subdirs (e.g. ``<bucketA>/cmr_export/astuff``). Only scenarios present under
    ALL given trees are merged (intersection).

    This is the shared engine behind ``merged_view_from_bucket`` (both vehicles
    from one bucket) and ``mixed_view_from_buckets`` (a different detector
    bucket per vehicle, the S4 mixed-detector combos). The rehydration is per
    TREE: each tree's own meta.yaml ``center_world`` is added back, then every
    tree is re-centred onto a single shared anchor frame (tesla if present,
    else the first vehicle in SUPPORTED_VEHICLES order). For mixed buckets the
    two trees' centroids for the SAME scenario genuinely differ — using one
    tree's centroid for the other re-introduces the ~80 m S4 drift bug.
    """
    vehicles = [v for v in SUPPORTED_VEHICLES if v in tree_by_vehicle]
    unknown = set(tree_by_vehicle) - set(SUPPORTED_VEHICLES)
    if unknown:
        raise ValueError(f"unknown vehicle ids {sorted(unknown)}; "
                         f"supported: {SUPPORTED_VEHICLES}")
    sets: list[set[str]] = []
    for v in vehicles:
        vdir = Path(tree_by_vehicle[v])
        if not vdir.is_dir():
            raise FileNotFoundError(f"per-vehicle tree for '{v}' not found: {vdir}")
        sets.append({p.name for p in vdir.iterdir() if p.is_dir()})
    scenarios = sorted(set.intersection(*sets)) if sets else []
    if not scenarios:
        raise FileNotFoundError(
            f"No common scenarios across per-vehicle trees: "
            f"{ {v: str(tree_by_vehicle[v]) for v in vehicles} }"
        )
    tmp = Path(tempfile.mkdtemp(prefix="cmr_merged_view_"))
    try:
        for scen in scenarios:
            out = tmp / f"test__{scen}"
            out.mkdir()

            # Each per-vehicle tree stores its OWN center_world centroid; all
            # x/y/z in that tree are relative to it (astuff and tesla differ by
            # tens of metres). Re-centre every tree onto a single shared anchor
            # frame (tesla if present, else the first available vehicle) BEFORE
            # concatenating, so the merged scenario is in one coherent frame.
            # Without this the two CAVs' detections land ~80 m apart and both
            # cross-CAV association and GT matching collapse (the S4 regression).
            centers: dict[str, tuple[float, float, float]] = {
                v: _load_center_world(Path(tree_by_vehicle[v]) / scen)
                for v in vehicles
            }
            anchor = centers.get("tesla") or next(iter(centers.values()))
            offsets = {
                v: (c[0] - anchor[0], c[1] - anchor[1], c[2] - anchor[2])
                for v, c in centers.items()
            }

            def srcs(fname: str) -> list[tuple[Path, tuple[float, float, float]]]:
                return [(Path(tree_by_vehicle[v]) / scen / fname, offsets[v])  # noqa: B023
                        for v in vehicles]

            _merge_csv_rehydrated(out / "detections.csv", srcs("detections.csv"))
            _merge_csv_rehydrated(out / "ego_state.csv",  srcs("ego_state.csv"))
            # Concatenate BOTH vehicles' GT (each CAV annotates only the objects
            # it saw — astuff and tesla GT row sets differ). run_scenario clusters
            # the per-vehicle annotations (2 m gate) into physical objects, so it
            # needs both; once both are in the shared anchor frame the two views
            # of one car fall within the clustering gate.
            _merge_csv_rehydrated(out / "gt_objects.csv", srcs("gt_objects.csv"))
            _merge_meta_yaml(
                out / "meta.yaml",
                [Path(tree_by_vehicle[v]) / scen / "meta.yaml" for v in vehicles],
                anchor,
            )
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@contextlib.contextmanager
def merged_view_from_bucket(bucket_root: Path) -> Iterator[Path]:
    """Yield a temp directory of merged ``test__<scenario>/`` dirs.

    The yielded path can be passed directly to scripts that expect the
    legacy "cmr_export with test__-prefixed scenario dirs" layout (e.g.
    dump_tracks_to_kitti_mot.py, v2v4real_replay.run_scenario).
    Auto-cleans up on context exit.
    """
    cmr_export = Path(bucket_root) / "cmr_export"
    trees = {v: cmr_export / v for v in SUPPORTED_VEHICLES
             if (cmr_export / v).is_dir()}
    if not trees or not _collect_scenarios(cmr_export):
        raise FileNotFoundError(
            f"No common scenarios found under {cmr_export} (looking under "
            f"{', '.join(SUPPORTED_VEHICLES)} subdirs). Run --rebuild import first."
        )
    with merged_view_from_trees(trees) as tmp:
        yield tmp


@contextlib.contextmanager
def mixed_view_from_buckets(astuff_bucket: Path, tesla_bucket: Path) -> Iterator[Path]:
    """Yield a merged view where the astuff tree comes from one bucket and the
    tesla tree from ANOTHER (the S4 mixed-detector combos, e.g. PP on astuff +
    CPft on tesla).

    Each bucket must have ``cmr_export/<vehicle>/<scenario>/`` for its vehicle.
    Rehydration is per tree using each tree's own meta.yaml ``center_world``
    (the two buckets' trees for the same scenario have DIFFERENT centroids),
    re-centred onto the tesla tree's centroid — see ``merged_view_from_trees``.
    """
    trees = {
        "astuff": Path(astuff_bucket) / "cmr_export" / "astuff",
        "tesla": Path(tesla_bucket) / "cmr_export" / "tesla",
    }
    for v, d in trees.items():
        if not d.is_dir():
            raise FileNotFoundError(
                f"bucket for '{v}' has no cmr_export/{v} tree: {d}")
    with merged_view_from_trees(trees) as tmp:
        yield tmp
