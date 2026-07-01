#!/usr/bin/env python3
"""Run V2V4Real tracker experiments by registered name (or list of names).

Every named experiment in ``configs/v2v4real_experiments.py`` maps deterministically
to one AB3DMOT result directory. This runner is the SOLE entry point — no more
hand-edited bash scripts with hidden flag combinations.

Usage:
  # Single experiment:
  v2v4real_runner.py --experiment LF_V2V4Real_anchor

  # Multiple in sequence (or parallel via --parallel N):
  v2v4real_runner.py --experiments LF_V2V4Real_anchor CoBEVT_V2V4Real_anchor \\
                                   DMSTrack_noinj_anchor --parallel 3

  # Dry run — show the resolved argv but don't execute:
  v2v4real_runner.py --experiment DMS_S3NoNorm_sabre_quadratic_headline --dry-run

  # Idempotent: SKIPs any experiment whose result_dir/data_0 is already populated.
  v2v4real_runner.py --experiment LF_V2V4Real_anchor --force   # re-run anyway
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "configs"))
sys.path.insert(0, str(REPO / "src"))

from v2v4real_experiments import (   # noqa: E402
    EXPERIMENTS, ExperimentConfig, get, list_all,
)
from eval_config import (  # noqa: E402
    AB3DMOT_ROOT, V2V4REAL_RESULTS_ROOT, V2V4REAL_INPUTS_ROOT,
    DMSTRACK_PACKAGE_DIR, DMSTRACK_PACKAGE_PARENT,
)

AB3DMOT_DETECTION_ROOT = AB3DMOT_ROOT / "data" / "v2v4real" / "detection"


# PYTHONPATH must match the bash convention exactly (run_matrix_sweep.sh:
#   PYTHONPATH=.:Xinshuo_PyToolbox:../../cmr/src:../DMSTrack:..
# resolved from cwd=AB3DMOT/). The DMSTRACK_PACKAGE_* entries resolve through
# the cmr/third_party/AB3DMOT symlink to the DMSTrack monorepo that ships the
# DMSTrack Python package; they let `from DMSTrack.model import DMSTrack`
# resolve. Including the cmr-grandparent dir would create a namespace-package
# shadow that breaks the import.
AB3DMOT_PYTHONPATH = ":".join([
    str(AB3DMOT_ROOT),                          # AB3DMOT/
    str(AB3DMOT_ROOT / "Xinshuo_PyToolbox"),    # AB3DMOT/Xinshuo_PyToolbox
    str(REPO / "src"),                          # cmr/src
    str(REPO / "configs"),                      # cmr/configs
    str(DMSTRACK_PACKAGE_DIR),                  # <DMSTrack monorepo>/DMSTrack (the package)
    str(DMSTRACK_PACKAGE_PARENT),               # <DMSTrack monorepo> (parent of package)
])

# Pinned to the cmr venv's python (has traci + scipy + the rest).
PYTHON = REPO / "venv" / "bin" / "python"


def _result_dir(cfg: ExperimentConfig) -> Path:
    return V2V4REAL_RESULTS_ROOT / cfg.result_dir_name() / "data_0"


def ensure_ab3dmot_dets_symlink(cfg: ExperimentConfig) -> None:
    """If ``cfg`` is bucket-backed, set up the forward symlink AB3DMOT reads.

    AB3DMOT resolves detections from ``data/v2v4real/detection/<det_name>_Car_val/``.
    For bucket-backed configs the actual bytes live in the canonical input
    tree under ``cmr/data/v2v4real_inputs/<bucket>/ab3dmot_detections/``.
    This helper places a forward symlink at AB3DMOT's expected path that
    points into the bucket — never the reverse, never out of cmr.

    Idempotent. Fails loudly (with a one-line fix hint) if the bucket or its
    ab3dmot_detections subdir is missing.
    """
    if not cfg.bucket_label:
        return  # legacy config; bytes are at the AB3DMOT path directly
    bucket_dets = V2V4REAL_INPUTS_ROOT / cfg.bucket_label / "ab3dmot_detections"
    if not bucket_dets.is_dir() or not any(bucket_dets.iterdir()):
        raise SystemExit(
            f"\nbucket {cfg.bucket_label!r} has no detections at {bucket_dets}.\n"
            f"  Fix: python scripts/build_v2v4real_input_bucket.py "
            f"--bucket {cfg.bucket_label} --rebuild import --apply"
        )
    link = AB3DMOT_DETECTION_ROOT / f"{cfg.detector}_Car_val"
    target = bucket_dets.resolve()
    if link.is_symlink():
        if link.resolve() == target:
            return  # already pointing at the right place
        link.unlink()
    elif link.exists():
        raise SystemExit(
            f"\n{link} exists as a real directory, not a symlink. Refusing to "
            f"clobber. Fix: rm -rf {link} and re-run (it will be regenerated "
            f"as a symlink into the canonical input tree)."
        )
    AB3DMOT_DETECTION_ROOT.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)
    print(f"[link] {link} -> {target}")


def _is_complete(cfg: ExperimentConfig, min_seqs: int = 9) -> bool:
    """Return True if the result dir has all expected sequence files."""
    d = _result_dir(cfg)
    if not d.is_dir():
        return False
    return len(list(d.glob("*.txt"))) >= min_seqs


def run_one(cfg: ExperimentConfig, *, force: bool = False, dry_run: bool = False) -> int:
    """Run a single experiment. Dispatches on ``cfg.pipeline``.

    Returns the subprocess exit code (0 on success, non-zero or 99 on skip).
    """
    if cfg.pipeline == "log_odds":
        return _run_log_odds(cfg, force=force, dry_run=dry_run)
    return _run_ab3dmot(cfg, force=force, dry_run=dry_run)


def _log_odds_dump_dir(cfg: ExperimentConfig) -> Path:
    """Where the log_odds run writes its KITTI MOT dump for this experiment."""
    return REPO / "results" / f"V2V4Real_KITTI_DUMP_{cfg.name}" / "data_0"


def _log_odds_is_complete(cfg: ExperimentConfig, min_seqs: int = 9) -> bool:
    d = _log_odds_dump_dir(cfg)
    return d.is_dir() and len(list(d.glob("*.txt"))) >= min_seqs


def _log_odds_worker(args):
    """ProcessPoolExecutor-friendly worker: run one scenario, dump KITTI MOT.

    Module-level (picklable). Late-imports heavy modules so each child loads
    its own state — mirrors the pattern in ``src/v2v4real_runner._replay_one``.

    Returns ``(seq_idx, scen_name, n_tracks)``. Raises on any per-scenario
    failure; caller decides whether to surface or continue.
    """
    seq_idx, scen_dir_str, params, dump_stream, out_dir_str = args
    import sys as _sys
    from pathlib import Path as _Path
    repo = _Path(__file__).resolve().parent.parent
    _sys.path.insert(0, str(repo / "src"))
    _sys.path.insert(0, str(repo / "scripts"))
    _sys.path.insert(0, str(repo / "configs"))
    import v2v4real_replay  # noqa: E402
    from dump_tracks_to_kitti_mot import load_tesla_poses, write_kitti_mot  # noqa: E402

    scen_dir = _Path(scen_dir_str)
    out_dir = _Path(out_dir_str)
    result = v2v4real_replay.run_scenario(scen_dir, params)
    poses = load_tesla_poses(scen_dir)
    tape = (result.get("match_tape_per_stream", {}) or {}).get(dump_stream, [])
    out_path = out_dir / f"{seq_idx:04d}.txt"
    n = write_kitti_mot(out_path, tape, poses)
    return seq_idx, scen_dir.name, n


def _run_log_odds(cfg: ExperimentConfig, *, force: bool, dry_run: bool) -> int:
    """End-to-end log_odds run: build merged view → run_scenario per seq →
    write KITTI MOT → invoke evaluate_precomputed_tracks.py.

    Reads ``cfg.log_odds_params`` for the tracker kwargs; ``cfg.bucket_label``
    for the source bucket. Writes to
    ``results/V2V4Real_KITTI_DUMP_<cfg.name>/data_0/0000.txt`` … ``0008.txt``.
    Idempotent (skip-if-complete unless ``--force``).
    """
    if cfg.log_odds_params is None:
        raise SystemExit(
            f"cfg.pipeline=={cfg.pipeline!r} but log_odds_params is None for "
            f"{cfg.name!r}. Either set log_odds_params or change pipeline."
        )
    if not cfg.bucket_label:
        raise SystemExit(
            f"log_odds pipeline requires bucket_label; {cfg.name!r} has none."
        )
    if _log_odds_is_complete(cfg) and not force:
        print(f"[SKIP] {cfg.name}: dump dir already populated ({_log_odds_dump_dir(cfg)})")
        return 99

    bucket_root = REPO / "data" / "v2v4real_inputs" / cfg.bucket_label
    if not bucket_root.is_dir():
        raise SystemExit(f"bucket dir missing: {bucket_root}")

    # Build the run_scenario cfg dict from LogOddsParams + standard defaults.
    params = cfg.log_odds_params.to_params_dict()
    dump_stream = params.pop("dump_stream", "sabre_static")
    # run_scenario expects record_tape_for to be set so it records tracks.
    params["record_tape_for"] = [dump_stream]
    # Stream-subset (fast iteration): if the cfg picked a subset, make sure
    # dump_stream is in it so the KITTI dump always has data.
    if "streams_keep" in params and params["streams_keep"]:
        keep = list(params["streams_keep"])
        if dump_stream not in keep:
            keep.append(dump_stream)
        params["streams_keep"] = keep
    # Apply the same baseline defaults dump_tracks_to_kitti_mot.py uses.
    params.setdefault("localizer_name", "rtk_v2v4real")
    params.setdefault("vehicle_classes", ["car", "truck", "bus", "construction_vehicle"])
    params.setdefault("use_static_matching", True)
    params.setdefault("self_report_egos", False)  # bucket's .txt already has self-reports

    # Worker count: default to min(num_scenarios, cpu_count - 1). Override via
    # CMR_LOG_ODDS_WORKERS env var. Set to 1 to force serial for debugging.
    n_workers_env = os.environ.get("CMR_LOG_ODDS_WORKERS")
    n_workers = int(n_workers_env) if n_workers_env else max(1, (os.cpu_count() or 2) - 1)

    print(f"[LOG_ODDS] {cfg.name}")
    print(f"  bucket:      {cfg.bucket_label}")
    print(f"  dump_stream: {dump_stream}")
    print(f"  streams_keep:{params.get('streams_keep') or 'ALL_30'}")
    print(f"  detector:    {params.get('detector_name')}")
    print(f"  lifecycle:   {params.get('lifecycle_mode')}  ddl={params.get('data_driven_lifecycle')}")
    print(f"  birth_gate:  alpha={params.get('data_driven_gate_alpha')}  fit={params.get('data_driven_gate_fit')}")
    print(f"  workers:     {n_workers}")
    if dry_run:
        print(f"  (dry-run; would build merged view, run {dump_stream!r}, dump to "
              f"{_log_odds_dump_dir(cfg)})")
        return 0

    # Import the heavy deps lazily (only the log_odds path needs them).
    import sys as _sys
    _sys.path.insert(0, str(REPO / "src"))
    _sys.path.insert(0, str(REPO / "scripts"))
    from input_bucket.merged_scenario_view import merged_view_from_bucket
    from concurrent.futures import ProcessPoolExecutor, as_completed

    out_dir = _log_odds_dump_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with merged_view_from_bucket(bucket_root) as merged_dir:
        scenarios = sorted(d for d in merged_dir.iterdir()
                           if d.is_dir() and d.name.startswith("test__"))
        if len(scenarios) != 9:
            raise SystemExit(
                f"Expected 9 test scenarios in merged view, got {len(scenarios)}. "
                f"Check the bucket's cmr_export coverage."
            )
        effective_workers = min(n_workers, len(scenarios))
        print(f"  built merged view with {len(scenarios)} scenarios; "
              f"running tracker on {effective_workers} workers...")
        work_items = [
            (seq_idx, str(scen_dir), params, dump_stream, str(out_dir))
            for seq_idx, scen_dir in enumerate(scenarios)
        ]
        n_written = 0
        n_failed = 0
        if effective_workers == 1:
            # Serial path — for debugging and for the single-scenario perf test.
            for item in work_items:
                seq_idx, scen_name, n = _log_odds_worker(item)
                n_written += n
                print(f"  [{seq_idx + 1}/{len(scenarios)}] {scen_name} → {n} tracks")
        else:
            with ProcessPoolExecutor(max_workers=effective_workers) as ex:
                futures = {ex.submit(_log_odds_worker, item): item[0]
                           for item in work_items}
                done_count = 0
                for fut in as_completed(futures):
                    seq_idx = futures[fut]
                    done_count += 1
                    try:
                        _, scen_name, n = fut.result()
                    except Exception as e:
                        n_failed += 1
                        print(f"  [{done_count}/{len(scenarios)}] seq {seq_idx} FAILED: {e}")
                        continue
                    n_written += n
                    print(f"  [{done_count}/{len(scenarios)}] {scen_name} → {n} tracks")
        if n_failed:
            print(f"  WARNING: {n_failed}/{len(scenarios)} scenarios failed")
    dt = time.time() - t0
    print(f"  dump complete: {n_written} total track rows in {dt:.1f}s → {out_dir}")

    # Score the dump.
    eval_script = REPO / "scripts" / "evaluate_precomputed_tracks.py"
    gt_dir = REPO / "data" / "v2v4real_inputs" / "baselines" / "paper_gt" / "kitti_labels"
    print(f"\n  scoring: {eval_script.name} on {out_dir}")
    proc = subprocess.run(
        [str(PYTHON), str(eval_script),
         "--tracking-dir", str(out_dir),
         "--gt-dir", str(gt_dir),
         "--global-eval"],
        cwd=str(REPO),
    )
    return proc.returncode


def _run_ab3dmot(cfg: ExperimentConfig, *, force: bool, dry_run: bool) -> int:
    """Original AB3DMOT dispatch — the legacy path."""
    if _is_complete(cfg) and not force:
        print(f"[SKIP] {cfg.name}: result dir already populated "
              f"({_result_dir(cfg).parent})")
        return 99  # sentinel: skipped

    # If this experiment reads from a canonical input bucket, place the
    # forward symlink AB3DMOT expects (idempotent; no-op for legacy configs).
    ensure_ab3dmot_dets_symlink(cfg)

    base = [
        str(PYTHON), "main.py",
        "--dataset", "v2v4real",
        "--split", "val",
    ]
    cmd = base + cfg.to_argv()

    print(f"[RUN ] {cfg.name}")
    print(f"       det={cfg.detector}  result_dir={cfg.result_dir_name()}")
    if dry_run:
        print(f"       (dry-run) cwd={AB3DMOT_ROOT}")
        print(f"       (dry-run) PYTHONPATH={AB3DMOT_PYTHONPATH}")
        print(f"       (dry-run) cmd: {' '.join(cmd)}")
        return 0

    env = os.environ.copy()
    env["PYTHONPATH"] = AB3DMOT_PYTHONPATH
    # AB3DMOT writes to ./results/v2v4real/... relatively → must chdir.
    log_path = Path("/tmp") / f"v2v4real_runner_{cfg.name}.log"
    t0 = time.time()
    with open(log_path, "w") as lf:
        proc = subprocess.run(
            cmd, cwd=str(AB3DMOT_ROOT), env=env,
            stdout=lf, stderr=subprocess.STDOUT, check=False,
        )
    rc = proc.returncode
    dt = time.time() - t0
    # AB3DMOT's combine_trk_cat post-process throws a benign AssertionError on
    # some result_suffix patterns AFTER the tracker output is already written.
    # We tolerate non-zero rc and verify by checking the result dir.
    ok = _is_complete(cfg)
    status = "DONE" if ok else f"FAIL (rc={rc})"
    print(f"[{status}] {cfg.name} in {dt:.1f}s (log: {log_path})")
    return 0 if ok else rc


def run_many(cfgs: List[ExperimentConfig], *, force: bool, dry_run: bool,
             parallel: int) -> int:
    if parallel <= 1:
        bad = 0
        for cfg in cfgs:
            rc = run_one(cfg, force=force, dry_run=dry_run)
            if rc not in (0, 99):
                bad += 1
        return bad

    # Parallel: bash-style bounded job control via concurrent.futures.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    bad = 0
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(run_one, cfg, force=force, dry_run=dry_run): cfg
                   for cfg in cfgs}
        for f in as_completed(futures):
            cfg = futures[f]
            try:
                rc = f.result()
                if rc not in (0, 99):
                    bad += 1
            except Exception as e:
                print(f"[FAIL] {cfg.name}: {e}")
                bad += 1
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run V2V4Real tracker experiments by registered name.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--experiment", metavar="NAME",
                   help="single registered experiment to run")
    g.add_argument("--experiments", metavar="NAME", nargs="+",
                   help="multiple registered experiments to run (in series unless --parallel)")
    g.add_argument("--role", metavar="ROLE",
                   choices=["v2v4real_paper_anchor", "dmstrack_paper_anchor",
                            "ablation", "headline", "side_quest"],
                   help="run all experiments with this paper_role")
    ap.add_argument("--parallel", type=int, default=8,
                    help="run up to N experiments concurrently (default 8; tuned for the 32-core dev box). "
                         "Each AB3DMOT subprocess is single-threaded so this is just concurrent subprocesses, "
                         "not multi-threading inside one job.")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve and print argv but don't actually run AB3DMOT")
    ap.add_argument("--force", action="store_true",
                    help="re-run even if result_dir is already populated")
    args = ap.parse_args()

    if args.experiment:
        cfgs = [get(args.experiment)]
    elif args.experiments:
        cfgs = [get(n) for n in args.experiments]
    elif args.role:
        cfgs = list_all(role=args.role)
        if not cfgs:
            print(f"no experiments registered with paper_role={args.role!r}")
            return 1
    else:
        ap.error("must pick exactly one of --experiment / --experiments / --role")

    print(f"running {len(cfgs)} experiment(s), parallel={args.parallel}, dry_run={args.dry_run}")
    bad = run_many(cfgs, force=args.force, dry_run=args.dry_run,
                   parallel=args.parallel)
    if bad:
        print(f"\n{bad}/{len(cfgs)} runs failed")
        return 1
    print(f"\nall {len(cfgs)} run(s) succeeded (or skipped as already-complete)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
