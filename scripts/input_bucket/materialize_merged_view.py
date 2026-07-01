#!/usr/bin/env python3
"""Materialize a bucket's rehydrated merged-scenario view to a PERSISTENT dir.

``merged_view_from_bucket`` (merged_scenario_view.py) is the verified rehydrator:
it re-centres each per-vehicle cmr_export tree (astuff/, tesla/) — which each
carry their OWN ``center_world`` centroid and otherwise land ~80 m apart — onto a
single shared anchor frame before concatenating. It yields a TEMP dir that is
deleted on context exit.

The benchmark (``scripts/run_v2v4real_benchmark.py`` → ``run_suite``) and
``src/v2v4real_runner.discover_scenarios`` expect a directory of
``test__<scenario>/`` dirs (each with detections.csv / ego_state.csv /
gt_objects.csv / meta.yaml) and do NOT rehydrate themselves. Pointing them at a
raw per-vehicle bucket cmr_export re-introduces the ~80 m coordinate-drift bug.

This tool produces a persistent copy of that rehydrated view so the benchmark's
``export_dir`` can point at it safely. It does NOT re-implement the offset math —
it copies the verified context manager's output verbatim.

Usage::

    python scripts/input_bucket/materialize_merged_view.py \\
        --bucket data/v2v4real_inputs/ours/detectors/pp_score0 \\
        --out    data/v2v4real_merged_views/pp_score0

Idempotent: refuses to overwrite a non-empty --out unless --force.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from input_bucket.merged_scenario_view import merged_view_from_bucket  # noqa: E402


def materialize(bucket_root: Path, out_dir: Path, *, force: bool = False,
                split_prefix: str = "test",
                exclude: list | None = None,
                include_only: list | None = None) -> int:
    """Copy the rehydrated merged view to ``out_dir``.

    ``split_prefix`` renames the rehydrator's ``test__<scen>`` dirs to
    ``<split_prefix>__<scen>`` (the rehydrator hardcodes ``test__``; train-split
    source trees need ``train__`` so v2v4real_runner.discover_scenarios files
    them under the right split). ``exclude`` / ``include_only`` filter on the
    bare scenario name (no split prefix).
    """
    bucket_root = Path(bucket_root)
    out_dir = Path(out_dir)
    exclude_set = set(exclude or [])
    include_set = set(include_only) if include_only else None
    if not (bucket_root / "cmr_export").is_dir():
        raise SystemExit(f"bucket has no cmr_export/: {bucket_root}")

    if out_dir.exists() and any(out_dir.iterdir()):
        if not force:
            raise SystemExit(
                f"{out_dir} is non-empty. Re-run with --force to rebuild it."
            )
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_scen = 0
    with merged_view_from_bucket(bucket_root) as merged_dir:
        for scen in sorted(merged_dir.iterdir()):
            if not scen.is_dir() or not scen.name.startswith("test__"):
                continue
            bare = scen.name[len("test__"):]
            if bare in exclude_set:
                print(f"  [excluded] {bare}")
                continue
            if include_set is not None and bare not in include_set:
                print(f"  [not in include list] {bare}")
                continue
            shutil.copytree(scen, out_dir / f"{split_prefix}__{bare}")
            n_scen += 1
    print(f"[materialized] {n_scen} rehydrated scenarios -> {out_dir}")
    return n_scen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bucket", required=True, type=Path,
                    help="bucket root (the dir containing cmr_export/astuff, cmr_export/tesla)")
    ap.add_argument("--out", required=True, type=Path,
                    help="persistent output dir of test__<scenario>/ merged views")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if --out is non-empty")
    ap.add_argument("--split-prefix", default="test",
                    help="split label for the output dirs "
                         "(<prefix>__<scenario>; default: test)")
    ap.add_argument("--exclude", action="append", default=None,
                    metavar="SCENARIO",
                    help="bare scenario name to skip (repeatable) — e.g. drop "
                         "the train/test-duplicated "
                         "Day19__testoutput_CAV_data_2022-03-15-09-54-40_0 "
                         "from a train view")
    ap.add_argument("--include-file", default=None, type=Path,
                    help="optional file of bare scenario names (one per line); "
                         "only these are materialized")
    args = ap.parse_args()
    include_only = None
    if args.include_file:
        include_only = [ln.strip() for ln in args.include_file.read_text().splitlines()
                        if ln.strip()]
    n = materialize(args.bucket, args.out, force=args.force,
                    split_prefix=args.split_prefix, exclude=args.exclude,
                    include_only=include_only)
    if args.split_prefix == "test" and n != 9:
        print(f"  WARNING: expected 9 test scenarios, got {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
