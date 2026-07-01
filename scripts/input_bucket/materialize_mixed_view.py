#!/usr/bin/env python3
"""Materialize a MIXED-detector merged-scenario view to a PERSISTENT dir.

S4 mixed combos run a DIFFERENT detector per CAV: the astuff per-vehicle tree
comes from one detector bucket and the tesla tree from another (e.g. PP on
astuff + CPft-100m on tesla). ``mixed_view_from_buckets``
(merged_scenario_view.py) is the verified rehydrator: each tree's OWN
meta.yaml ``center_world`` is added back and both trees are re-centred onto
the tesla tree's centroid before concatenation. The two buckets' trees for
the same scenario have DIFFERENT centroids — skipping the per-tree
rehydration re-introduces the ~80 m S4 coordinate-drift bug (cross-CAV GT
co-location collapses from ~48% fleet-wide to ~0.2%).

Like materialize_merged_view.py, this copies the verified context manager's
temp output verbatim to a persistent dir so the benchmark's ``export_dir``
can point at it.

Usage::

    python scripts/input_bucket/materialize_mixed_view.py \\
        --astuff-bucket data/v2v4real_inputs/ours/detectors/pp_score0 \\
        --tesla-bucket  data/v2v4real_inputs/ours/detectors/cp_zeroshot_100m \\
        --out           data/v2v4real_merged_views/mix_pp_cpzs

Idempotent: refuses to overwrite a non-empty --out unless --force.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from input_bucket.merged_scenario_view import mixed_view_from_buckets  # noqa: E402


def materialize_mixed(astuff_bucket: Path, tesla_bucket: Path, out_dir: Path,
                      *, force: bool = False) -> int:
    astuff_bucket, tesla_bucket, out_dir = (
        Path(astuff_bucket), Path(tesla_bucket), Path(out_dir))

    if out_dir.exists() and any(out_dir.iterdir()):
        if not force:
            raise SystemExit(
                f"{out_dir} is non-empty. Re-run with --force to rebuild it."
            )
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_scen = 0
    with mixed_view_from_buckets(astuff_bucket, tesla_bucket) as merged_dir:
        for scen in sorted(merged_dir.iterdir()):
            if not scen.is_dir() or not scen.name.startswith("test__"):
                continue
            shutil.copytree(scen, out_dir / scen.name)
            n_scen += 1
    print(f"[materialized mixed] astuff<-{astuff_bucket.name} "
          f"tesla<-{tesla_bucket.name}: {n_scen} rehydrated scenarios -> {out_dir}")
    return n_scen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--astuff-bucket", required=True, type=Path,
                    help="bucket whose cmr_export/astuff tree supplies the astuff CAV")
    ap.add_argument("--tesla-bucket", required=True, type=Path,
                    help="bucket whose cmr_export/tesla tree supplies the tesla CAV")
    ap.add_argument("--out", required=True, type=Path,
                    help="persistent output dir of test__<scenario>/ merged views")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if --out is non-empty")
    args = ap.parse_args()
    n = materialize_mixed(args.astuff_bucket, args.tesla_bucket, args.out,
                          force=args.force)
    if n != 9:
        print(f"  WARNING: expected 9 test scenarios, got {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
