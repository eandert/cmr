"""Refit a detector bucket's GPEM regression CSVs by wrapping mmdet3d tools.

The GPEM regression is the "detector classification" — for each detector we
fit linear / quadratic / polar models of measurement noise vs (range, angle).
The math lives in mmdetection3d (heavy mmcv/mmdet3d deps), so cmr wraps the
existing tools rather than reimplementing them.

Wrapped tools:
  - ``<mmdet3d>/tools/evaluate_errors.py``     (preds.pkl + config + GT → error CSVs)
  - ``<mmdet3d>/tools/regression_to_cmr_csv.py`` (error CSVs → GPEM CSVs in cmr format)

Inputs to the refit (per bucket, from ``BucketSpec.gpem_sources``):
  - ``mmdet3d_config_relpath``: a Python config file under mmdet3d_root
  - ``preds_relpath``: a preds_trainval.pkl under mmdet3d_root
  - ``cmr_basename``: the GPEM CSV stem (e.g. ``pointpillar_v2v4real_astuff``)

Outputs land in ``<bucket>/gpem_calibration/``:
  - ``<cmr_basename>.csv``                     (regression coefficients)
  - ``<cmr_basename>_distributions.csv``       (binned distributions)
  - ``<cmr_basename>_polar_distributions.csv`` (polar-binned distributions)

Setup: requires ``paths.local.yaml`` with ``mmdet3d_root`` pointing at the
developer's mmdetection3d clone. See ``src/local_paths.py``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))
import local_paths  # noqa: E402

from .registry import BucketSpec, GpemSource  # noqa: E402

# Output filenames produced by regression_to_cmr_csv.py for a given basename.
# Plus _polar_calibration.csv which evaluate_errors.py writes via the
# --polar-calibration-out flag — we copy that one from temp ourselves.
GPEM_OUTPUT_SUFFIXES = (
    ".csv",
    "_distributions.csv",
    "_polar_distributions.csv",
    "_polar_calibration.csv",
)


class GpemRefitError(RuntimeError):
    """Raised when the refit pipeline fails — message includes a one-line fix."""


def _run(cmd: list[str], cwd: Path, log_prefix: str, *, dry_run: bool) -> None:
    """Run a subprocess; raise GpemRefitError on failure with a clean message.

    Stdout / stderr stream live to the calling terminal so the user sees
    progress (the upstream tools take 5-10 min per detector). This costs
    visibility-of-each-subprocess's-output for the simpler aesthetic of
    "I see what's happening".
    """
    pretty = " ".join(str(c) for c in cmd)
    print(f"  [{log_prefix}] {pretty}")
    if dry_run:
        print(f"  [{log_prefix}] (dry-run; not executed)")
        return
    proc = subprocess.run(cmd, cwd=cwd)  # stream stdout/stderr live
    if proc.returncode != 0:
        raise GpemRefitError(
            f"{log_prefix} failed (exit {proc.returncode}). "
            f"Command was:\n    {pretty}\n"
            f"Fix: inspect the upstream tool's logs above and re-run."
        )


def _assert_not_blind(path: Path, *, role: str, cmr_basename: str) -> None:
    """Refuse to read from any directory ancestor with a BLIND_DO_NOT_FIT marker.

    Val/test data is blind by protocol — fitting GPEM on it would leak the
    evaluation split into the noise model. We walk up from the path looking
    for a sentinel; if we find one, we raise loudly with a fix hint.
    """
    resolved = path.resolve()
    for ancestor in (resolved, *resolved.parents):
        marker = ancestor / "BLIND_DO_NOT_FIT"
        if marker.is_file():
            raise GpemRefitError(
                f"GPEM refit blocked by BLIND_DO_NOT_FIT sentinel at {marker}\n"
                f"  Tried to read {role} for {cmr_basename!r} from a directory "
                f"that is marked blind (val/test split).\n"
                f"  Fix: point this gpem_source at a TRAIN-split preds.pkl in "
                f"registry.py — never the val/test export."
            )


def _validate_source(mmdet3d_root: Path, src: GpemSource) -> tuple[Path, Path, Path | None]:
    """Resolve config + preds (+ optional aligned ann_file) paths.

    Raises if any required file is missing or blind. Returns
    ``(config, preds, ann_file_or_None)`` — ``ann_file`` is None when the
    GpemSource has no ``ann_file_relpath`` (detector aligns natively).
    """
    cfg = mmdet3d_root / src.mmdet3d_config_relpath
    preds = mmdet3d_root / src.preds_relpath
    if not cfg.is_file():
        raise GpemRefitError(
            f"mmdet3d config not found for {src.cmr_basename}:\n"
            f"  expected: {cfg}\n"
            f"  Fix: re-run the inference / config-generation step in the "
            f"upstream mmdet3d repo, then point this script at the produced .py."
        )
    if not preds.is_file():
        raise GpemRefitError(
            f"preds_trainval.pkl not found for {src.cmr_basename}:\n"
            f"  expected: {preds}\n"
            f"  Fix: run `python tools/dump_test_predictions.py CONFIG CHECKPOINT "
            f"--out {preds.name}` in your mmdet3d clone first."
        )
    ann_file: Path | None = None
    if src.ann_file_relpath:
        ann_file = mmdet3d_root / src.ann_file_relpath
        if not ann_file.is_file():
            raise GpemRefitError(
                f"aligned ann_file not found for {src.cmr_basename}:\n"
                f"  expected: {ann_file}\n"
                f"  This GpemSource declares ann_file_relpath because its preds are\n"
                f"  NOT in the config's default ann_file order (evaluate_errors.py\n"
                f"  pairs preds[i]<->ann_info[i] positionally).\n"
                f"  Fix: produce the 1:1-aligned infos pkl alongside the preds (the\n"
                f"  exporter that wrote {preds.name} should emit it), or clear\n"
                f"  ann_file_relpath in registry.py if this detector aligns natively."
            )
        _assert_not_blind(ann_file, role="ann_file", cmr_basename=src.cmr_basename)
    # Belt-and-suspenders: blind sentinels live alongside val/test exports.
    _assert_not_blind(preds, role="preds.pkl", cmr_basename=src.cmr_basename)
    _assert_not_blind(cfg,   role="config",   cmr_basename=src.cmr_basename)
    return cfg, preds, ann_file


def _python_for_mmdet3d() -> str:
    """Pick the Python interpreter that can import mmdet3d.

    Heuristic: the developer keeps their mmdet3d conda env activated, OR an
    explicit override is set via the ``CMR_MMDET3D_PYTHON`` env var.
    Otherwise we fall back to ``sys.executable`` and hope mmdet3d is installed
    in the active venv (it almost certainly isn't, hence the env-var escape).
    """
    return os.environ.get("CMR_MMDET3D_PYTHON", sys.executable)


def refit_bucket_gpem(
    bucket_root: Path,
    spec: BucketSpec,
    *,
    dry_run: bool,
) -> list[Path]:
    """Refit every GPEM source in ``spec.gpem_sources`` into ``bucket_root/gpem_calibration/``.

    Returns the list of CSV files written. Raises ``GpemRefitError`` on failure
    (caller can decide whether to leave partial output in place).
    """
    if spec.kind != "detector":
        raise GpemRefitError(
            f"--rebuild gpem on bucket {spec.label!r} is for detector buckets; "
            f"this bucket is kind={spec.kind!r}. Use refit_localizer_gpem for "
            f"localizer buckets."
        )
    if not spec.gpem_sources:
        raise GpemRefitError(
            f"bucket {spec.label!r} has no gpem_sources in registry.py — "
            f"add at least one GpemSource(...) entry first."
        )

    mmdet3d_root = local_paths.get("mmdet3d_root")  # raises with friendly msg if unset
    mmdet3d_py = _python_for_mmdet3d()
    eval_tool = mmdet3d_root / "tools" / "evaluate_errors.py"
    regress_tool = mmdet3d_root / "tools" / "regression_to_cmr_csv.py"
    for t in (eval_tool, regress_tool):
        if not t.is_file():
            raise GpemRefitError(
                f"missing upstream tool: {t}\n"
                f"  Fix: your mmdet3d clone at {mmdet3d_root} is older than the "
                f"version that ships these tools. Update or point paths.local.yaml "
                f"at a clone that has them."
            )

    gpem_dir = bucket_root / "gpem_calibration"
    if not dry_run:
        # Replace existing symlinks (or files) before writing fresh ones.
        # Keeps the dir; only removes its children to preserve any side-cars.
        for p in list(gpem_dir.glob("*")):
            if p.is_symlink() or p.is_file():
                p.unlink()
        gpem_dir.mkdir(parents=True, exist_ok=True)

    # Persistent error_analysis root — we keep evaluate_errors' output (with
    # polar_binned_errors.csv) on disk so the downstream AMOTA-Bayes gate solver
    # (solve_amota_bayes_gate.py, which reads polar_binned_errors.csv per
    # detector) has an input after a bucket rebuild. Previously this lived in a
    # tempfile.TemporaryDirectory and got deleted on exit.
    err_analysis_root = gpem_dir / "_error_analysis"

    written: list[Path] = []
    for src in spec.gpem_sources:
        cfg_abs, preds_abs, ann_file_abs = _validate_source(mmdet3d_root, src)
        print(f"\n  ── refitting {src.cmr_basename} ({src.vehicle or 'single'}) ──")

        # Stable per-detector error_analysis dir. Clear/overwrite it at the
        # start of each detector's refit so reruns are clean (no stale rows).
        err_dir = err_analysis_root / src.cmr_basename
        if not dry_run:
            if err_dir.exists():
                shutil.rmtree(err_dir)
            err_dir.mkdir(parents=True, exist_ok=True)
        polar_calib_path = gpem_dir / f"{src.cmr_basename}_polar_calibration.csv"
        fp_records_path = err_dir / f"{src.cmr_basename}_fp_records.csv"

        # Stage 1: evaluate_errors.py — preds + config → error_analysis/
        eval_cmd = [
            mmdet3d_py, str(eval_tool),
            str(cfg_abs),
            "--preds", str(preds_abs),
            "--out-dir", str(err_dir) + "/",
            "--polar-calibration-out", str(polar_calib_path),
            "--fp-records-out", str(fp_records_path),
            # --plot triggers polar_binned_errors.csv generation. Without it,
            # the polar analysis is skipped and regression_to_cmr_csv has no
            # input from which to produce <name>_polar_distributions.csv.
            # Mirrors the working invocation in
            # mmdetection3d/tools/regen_all_polar_calibrations.sh.
            "--plot",
        ]
        if ann_file_abs is not None:
            # Override the config's default ann_file with one aligned 1:1 to
            # the preds. evaluate_errors pairs preds[i]<->ann_info[i]
            # positionally; without this, an OpenCOOD-produced detector's
            # preds would align to the WRONG GT (and the length guard in
            # evaluate_errors.py would fail-hard if the counts differ).
            eval_cmd += [
                "--cfg-options",
                f"test_dataloader.dataset.ann_file={ann_file_abs}",
            ]
        _run(
            eval_cmd,
            cwd=mmdet3d_root,
            log_prefix=f"evaluate_errors:{src.cmr_basename}",
            dry_run=dry_run,
        )

        # Stage 2: regression_to_cmr_csv.py — error_analysis → bucket gpem_calibration
        _run(
            [
                mmdet3d_py, str(regress_tool),
                "--error-analysis", str(err_dir),
                "--out-dir", str(gpem_dir),
                "--name", src.cmr_basename,
            ],
            cwd=mmdet3d_root,
            log_prefix=f"regression_to_cmr_csv:{src.cmr_basename}",
            dry_run=dry_run,
        )

        if not dry_run:
            # evaluate_errors wrote polar_calibration.csv straight into the
            # bucket (polar_calib_path above). regression_to_cmr_csv doesn't
            # touch this file but downstream tools (derive_optimal_birth_gate,
            # the polar-cov sensor loader) expect it next to the others.
            if polar_calib_path.is_file():
                print(f"  [ok] wrote {polar_calib_path}")
            else:
                print(f"  [warn] no polar_calibration.csv produced by evaluate_errors "
                      f"(expected at {polar_calib_path}) — polar-cov streams may fail later")

            # Record what we wrote.
            for suffix in GPEM_OUTPUT_SUFFIXES:
                p = gpem_dir / f"{src.cmr_basename}{suffix}"
                if p.is_file():
                    written.append(p)

    return written
