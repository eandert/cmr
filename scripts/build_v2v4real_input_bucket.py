#!/usr/bin/env python3
"""Build (or rebuild) one canonical input bucket under ``data/v2v4real_inputs/``.

A "bucket" is a folder under ``data/v2v4real_inputs/`` that holds everything
needed to evaluate a single detector lineage or GT variant — detection bytes,
GPEM calibration, manifest, and human-readable README. The full set of buckets
is registered in ``scripts/input_bucket/registry.py``.

This script is the *only* user-facing tool that mutates a bucket. It is
idempotent (re-running with no changes is a no-op + logs SKIP), resumable
(crashes mid-step are picked up by re-running), and dry-run by default (you
must pass ``--apply`` to mutate the filesystem).

Currently-supported steps
-------------------------
* ``manifest`` — walk the bucket's data subdirs, sha256 every file, write
  ``MANIFEST.json``. Always safe to re-run.
* ``readme`` — render ``README.md`` from the registry entry + manifest. Always
  safe to re-run.
* ``import`` — convert ``cmr_export/`` CSVs into ``ab3dmot_detections/``
  KITTI-MOT .txt files.
* ``gpem`` — refit linear/quadratic/polar regression CSVs from upstream preds
  (wraps mmdet3d's ``evaluate_errors.py`` + ``regression_to_cmr_csv.py``).
* ``gate`` — finish the GPEM pipeline: synthesize the count-weighted "overall
  bin" rows the static loader requires, then solve the AMOTA-Bayes birth gate
  + log-odds lifecycle (``solve_amota_bayes_gate.py`` — the authoritative
  gate/lifecycle source the production log_odds config reads). Always run this
  after ``gpem`` (``--rebuild all`` does so automatically).

Examples
--------
Generate the manifest + README for the dmstrack_pp baseline::

    python scripts/build_v2v4real_input_bucket.py --bucket baselines/dmstrack_pp \\
        --rebuild manifest readme --apply

Rebuild every bucket's manifest + readme (sequential, safe)::

    python scripts/build_v2v4real_input_bucket.py --bucket all --rebuild manifest readme --apply

Check what would be done without writing::

    python scripts/build_v2v4real_input_bucket.py --bucket ours/pp_score0 --rebuild manifest
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from input_bucket.manifest import (  # noqa: E402
    BucketUpstream, Manifest, compute_manifest, read_manifest, verify_against_disk,
    write_manifest,
)
from input_bucket.readme import render_readme  # noqa: E402
from input_bucket.refit_detector_gpem import (  # noqa: E402
    GpemRefitError, refit_bucket_gpem,
)
from input_bucket.import_ab3dmot_format import (  # noqa: E402
    ImportError_, import_bucket_to_ab3dmot,
)
from input_bucket.registry import BUCKETS, BucketSpec  # noqa: E402


V2V4REAL_INPUTS_ROOT = REPO / "data" / "v2v4real_inputs"

# Repo venv python — used to run the cmr-side post-GPEM steps (synthesize
# overall bins + derive birth gate) as subprocesses. These are pure-cmr tools
# (no mmdet3d deps), so they run in the repo venv, NOT the mmdet3d interpreter.
VENV_PYTHON = REPO / "venv" / "bin" / "python"
SYNTHESIZE_SCRIPT = REPO / "scripts" / "synthesize_overall_bins.py"
# Authoritative gate + lifecycle source the production log_odds config uses:
# solve_amota_bayes_gate.py emits the AMOTA-Bayes gate curves (α=96 polar,
# 97 static, 98 linear, 99 quadratic) AND recommended_lifecycle.csv. The
# config reads data_driven_gate_alpha=99.0 (quad) + data_driven_lifecycle=True.
# derive_optimal_birth_gate.py remains a standalone tp-preserving (α=1/2) tool
# the production config does NOT read — see step_gate.
SOLVE_GATE_SCRIPT = REPO / "scripts" / "solve_amota_bayes_gate.py"

ALL_STEPS = ("manifest", "readme", "import", "gpem", "gate")
DEFAULT_STEPS = ("manifest", "readme")

log = logging.getLogger("build_bucket")


def _bucket_root(label: str) -> Path:
    return V2V4REAL_INPUTS_ROOT / label


def _resolve_buckets(arg: str) -> List[str]:
    if arg == "all":
        return list(BUCKETS.keys())
    if arg not in BUCKETS:
        raise SystemExit(
            f"unknown bucket: {arg!r}\n"
            f"  known: {', '.join(BUCKETS.keys())}\n"
            f"  or use --bucket all to rebuild every bucket"
        )
    return [arg]


def step_manifest(spec: BucketSpec, apply: bool) -> None:
    root = _bucket_root(spec.label)
    if not root.exists():
        raise SystemExit(
            f"bucket directory {root} does not exist. "
            f"Create the canonical skeleton first (Phase A)."
        )
    upstream = BucketUpstream(
        source_kind=spec.source_kind,
        source_repo_url=spec.source_repo_url,
        source_commit_sha_or_tag=spec.source_commit_sha_or_tag,
        source_artifact_description=spec.source_artifact_description,
    )
    m = compute_manifest(root, spec.label, upstream)
    log.info("  %-32s  %d file(s) hashed", spec.label, len(m.files))
    if apply:
        path = write_manifest(root, m)
        log.info("  wrote %s", path)
    else:
        log.info("  (dry-run; pass --apply to write MANIFEST.json)")


def step_readme(spec: BucketSpec, apply: bool) -> None:
    root = _bucket_root(spec.label)
    if not root.exists():
        raise SystemExit(f"bucket directory {root} does not exist.")
    try:
        m: Manifest | None = read_manifest(root)
    except FileNotFoundError:
        log.warning("  no MANIFEST.json yet for %s; README will skip file summary", spec.label)
        m = None
    text = render_readme(root, spec, m)
    target = root / "README.md"
    if apply:
        target.write_text(text)
        log.info("  wrote %s", target)
    else:
        log.info("  (dry-run; would write %s, %d bytes)", target, len(text))


def step_import(spec: BucketSpec, apply: bool) -> None:
    """Convert ``<bucket>/cmr_export/`` → ``<bucket>/ab3dmot_detections/`` KITTI .txt.

    Detector buckets only. Detections are written in each producing CAV's OWN
    local frame (NOT a shared common frame) — see
    ``input_bucket/import_ab3dmot_format.py`` for the GPEM rationale.
    """
    if spec.kind != "detector":
        raise SystemExit(
            f"--rebuild import is for detector buckets only; {spec.label!r} is "
            f"kind={spec.kind!r}."
        )
    try:
        counts = import_bucket_to_ab3dmot(_bucket_root(spec.label), dry_run=not apply)
    except ImportError_ as e:
        raise SystemExit(f"\nImport failed for {spec.label!r}:\n\n{e}")
    log.info("  files=%d  dets=%d  self_reports=%d  pose_rows=%d  dropped_by_range=%d",
             counts["files_written"], counts["dets"], counts["self_reports"],
             counts["pose_rows"], counts["dets_dropped_by_range_cap"])
    if not apply:
        log.info("  (dry-run; pass --apply to actually write files)")


def step_gpem(spec: BucketSpec, apply: bool) -> None:
    """Refit GPEM regression CSVs from upstream preds.pkl.

    Detector buckets: wraps mmdet3d's ``evaluate_errors.py`` +
    ``regression_to_cmr_csv.py``. Requires ``paths.local.yaml`` with
    ``mmdet3d_root`` set.

    Localizer buckets: not yet supported here. The current localizer CSVs
    in ``ours/localizers/*`` are symlinks-in from ``src/data/sensor_models/``;
    refit them via the legacy ``scripts/import_localizer_binned.py``.
    """
    if spec.kind == "localizer":
        raise SystemExit(
            f"--rebuild gpem on localizer bucket {spec.label!r} is not yet wired up. "
            f"For now, run scripts/import_localizer_binned.py manually and re-link "
            f"the CSVs into the bucket; a future refit_localizer_gpem step will "
            f"automate it."
        )
    if spec.kind != "detector":
        raise SystemExit(
            f"--rebuild gpem is for detector buckets only; {spec.label!r} is "
            f"kind={spec.kind!r}."
        )
    if not apply:
        # Dry-run: show the plan without invoking mmdet3d tools.
        log.info("  (dry-run; pass --apply to invoke mmdet3d tools)")
        for src in spec.gpem_sources:
            log.info("  would refit  %s/%s  ←  %s",
                     src.cmr_basename, src.vehicle or "single", src.preds_relpath)
        if not spec.gpem_sources:
            log.warning("  bucket has no gpem_sources in registry.py")
        return
    try:
        written = refit_bucket_gpem(_bucket_root(spec.label), spec, dry_run=False)
    except GpemRefitError as e:
        raise SystemExit(f"\nGPEM refit failed for {spec.label!r}:\n\n{e}")
    log.info("  wrote %d GPEM CSV file(s) into %s/gpem_calibration/",
             len(written), spec.label)


def _run_cmr_subprocess(cmd: List[str], label: str, apply: bool) -> None:
    """Run a cmr-side helper script (synthesize / derive_gate) as a subprocess.

    Honors the dry-run convention: without --apply we only print the command.
    Streams stdout/stderr live so the user sees the helper's own output.
    """
    pretty = " ".join(str(c) for c in cmd)
    log.info("  %s", pretty)
    if not apply:
        log.info("  (dry-run; pass --apply to execute)")
        return
    proc = subprocess.run(cmd, cwd=REPO)
    if proc.returncode != 0:
        raise SystemExit(
            f"\n{label} step failed (exit {proc.returncode}). Command was:\n"
            f"    {pretty}\n"
            f"  Fix: inspect the output above and re-run."
        )


def step_gate(spec: BucketSpec, apply: bool) -> None:
    """Finish the GPEM pipeline: synthesize overall bins + solve the AMOTA-Bayes gate.

    Two cmr-side steps that MUST follow the GPEM refit (and used to be done by
    hand and get skipped):

      (a) ``synthesize_overall_bins.py`` per cmr_basename — appends the
          count-weighted "overall bin" rows the STATIC-mode error-model loader
          REQUIRES (regression_to_cmr_csv does not emit them).
      (b) ``solve_amota_bayes_gate.py`` for this bucket's cmr_basenames — the
          AUTHORITATIVE gate + lifecycle source the production log_odds config
          uses. It writes the AMOTA-Bayes birth_gate_curves.csv (α=96 polar,
          97 static, 98 linear, 99 quadratic) and recommended_lifecycle.csv
          into ``results/birth_gate_calibration/`` (merge-safe by detector:
          other detectors' rows are preserved). The config reads
          ``data_driven_gate_alpha=99.0`` (quad) + ``data_driven_lifecycle=True``.
          We point the solver at the persistent
          ``<bucket>/gpem_calibration/_error_analysis/<basename>`` dirs that the
          gpem step now preserves (it reads polar_binned_errors.csv per
          detector).

          NOTE: ``derive_optimal_birth_gate.py`` is a SEPARATE standalone
          tp-preserving (α=1/2) tool the production config does NOT read; it is
          intentionally NOT run here. Both tools write birth_gate_curves.csv
          merge-safe-by-detector, so running both would clobber each other per
          detector — the pipeline uses exactly ONE gate tool.

    Detector buckets only.
    """
    if spec.kind != "detector":
        raise SystemExit(
            f"--rebuild gate is for detector buckets only; {spec.label!r} is "
            f"kind={spec.kind!r}."
        )
    basenames = [src.cmr_basename for src in spec.gpem_sources]
    if not basenames:
        log.warning("  bucket has no gpem_sources in registry.py — nothing to gate")
        return

    # (a) synthesize overall bins — one invocation per basename so a failure
    #     names the offending detector. The script no-ops idempotently.
    for basename in basenames:
        _run_cmr_subprocess(
            [str(VENV_PYTHON), str(SYNTHESIZE_SCRIPT), basename],
            label=f"synthesize_overall_bins:{basename}", apply=apply,
        )

    # (b) solve the AMOTA-Bayes gate + lifecycle for this bucket's detectors
    #     (merge-safe write). One invocation covering all basenames, each
    #     pointed at its persistent error_analysis dir from the gpem step.
    err_root = _bucket_root(spec.label) / "gpem_calibration" / "_error_analysis"
    solve_cmd = [str(VENV_PYTHON), str(SOLVE_GATE_SCRIPT), "--detectors", *basenames]
    for basename in basenames:
        solve_cmd += ["--error-analysis-dir", f"{basename}={err_root / basename}"]
    _run_cmr_subprocess(
        solve_cmd,
        label="solve_amota_bayes_gate", apply=apply,
    )


STEP_FUNCS = {
    "manifest": step_manifest,
    "readme":   step_readme,
    "import":   step_import,
    "gpem":     step_gpem,
    "gate":     step_gate,
}

# Execution order when multiple steps are requested (e.g. --rebuild all):
# import → gpem → gate → manifest → readme. The gate step depends on gpem's
# output, and manifest/readme summarize whatever the earlier steps produced.
_STEP_ORDER = ("import", "gpem", "gate", "manifest", "readme")


def run_bucket(label: str, steps: Iterable[str], apply: bool) -> None:
    spec = BUCKETS[label]
    log.info("─── bucket: %s ───", label)
    requested = set(steps)
    ordered = [s for s in _STEP_ORDER if s in requested]
    for step in ordered:
        log.info("step: %s", step)
        STEP_FUNCS[step](spec, apply)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples", 1)[1] if "Examples" in __doc__ else "",
    )
    ap.add_argument(
        "--bucket", required=True,
        help=f"bucket label (one of: {', '.join(BUCKETS)}) or 'all'",
    )
    ap.add_argument(
        "--rebuild", nargs="+", choices=list(ALL_STEPS) + ["all"],
        default=list(DEFAULT_STEPS),
        help=f"steps to run (default: {' '.join(DEFAULT_STEPS)})",
    )
    ap.add_argument(
        "--apply", action="store_true",
        help="execute (default is dry-run)",
    )
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--check-integrity", action="store_true",
                    help="after the run, verify on-disk bytes match the manifest")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    steps = list(ALL_STEPS) if "all" in args.rebuild else args.rebuild
    labels = _resolve_buckets(args.bucket)

    if not args.apply:
        log.info("(dry-run mode — no files will be written; pass --apply to mutate)")

    for label in labels:
        run_bucket(label, steps, apply=args.apply)

    if args.check_integrity:
        log.info("─── integrity check ───")
        any_problem = False
        for label in labels:
            problems = verify_against_disk(_bucket_root(label))
            if problems:
                any_problem = True
                log.error("  %s — %d problem(s):", label, len(problems))
                for p in problems:
                    log.error("    - %s", p)
            else:
                log.info("  %s — OK", label)
        if any_problem:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
