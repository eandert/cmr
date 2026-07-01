#!/usr/bin/env python3
"""Wire per-vehicle calibration + gate/lifecycle models for the S4 MIXED combos.

The tracker resolves per-vehicle models by substituting the CAV id into the
config's ``detector_name`` (e.g. ``mix_pp_cpzs_{vehicle}``). That mechanism
assumes both vehicles share a name stem, which mixed pairs (different detector
per CAV) break. The cleanest existing-mechanism fix is ALIASES: for each mix
``mix_<a>_<t>`` (a = astuff-side detector, t = tesla-side detector, codes
pp/cpzs/cpft) this script

1. symlinks ``src/data/sensor_models/mix_<a>_<t>_<vehicle><suffix>.csv`` to
   the real detector's calibration files (all 4 suffixes: '', _distributions,
   _polar_calibration, _polar_distributions);
2. appends alias rows (copies of the source detector's rows under the alias
   name) to ``results/birth_gate_calibration/birth_gate_curves.csv`` and
   ``results/birth_gate_calibration/recommended_lifecycle.csv``, which are
   keyed by detector name.

Real stems (current, post NMS-fix / 100m-finetune):

    pp   -> pointpillar_v2v4real_{vehicle}            (per-vehicle fits)
    cpzs -> centerpoint_100m_on_v2v4real              (fleet-wide: SAME files
                                                       for both CAVs)
    cpft -> centerpoint_100m_v2v4real_finetune_{vehicle}  (per-vehicle fits)

Idempotent and self-repairing: re-running never duplicates alias rows — any
existing rows under an alias name are REPLACED with fresh copies of the
current source rows (so stale aliases, e.g. the 2026-05-02 ones that pointed
at pointpillar_v2v4real_score02_* / centerpoint_54m_*, are repaired), and
symlinks already pointing at the right target are left untouched. Files are
only rewritten when something actually changed.

Usage::

    python scripts/input_bucket/alias_mix_models.py            # apply
    python scripts/input_bucket/alias_mix_models.py --dry-run  # show actions
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
SENSOR_MODELS = REPO / "src" / "data" / "sensor_models"
GATE_DIR = REPO / "results" / "birth_gate_calibration"
CURVES_CSV = GATE_DIR / "birth_gate_curves.csv"
LIFECYCLE_CSV = GATE_DIR / "recommended_lifecycle.csv"

VEHICLES = ("astuff", "tesla")
SUFFIXES = ("", "_distributions", "_polar_calibration", "_polar_distributions")

# Detector code -> real calibration stem ('{vehicle}' substituted per CAV;
# cpzs is a single fleet-wide model, so both CAVs alias the same files).
STEM_BY_CODE = {
    "pp": "pointpillar_v2v4real_{vehicle}",
    "cpzs": "centerpoint_100m_on_v2v4real",
    "cpft": "centerpoint_100m_v2v4real_finetune_{vehicle}",
}

# The six mixed pairs: (astuff-side detector, tesla-side detector).
MIXES = [
    ("pp", "cpzs"), ("cpzs", "pp"),
    ("pp", "cpft"), ("cpft", "pp"),
    ("cpzs", "cpft"), ("cpft", "cpzs"),
]


def alias_map() -> dict[str, str]:
    """Return {alias detector name -> real source detector name} for all
    6 mixes x 2 vehicles."""
    out: dict[str, str] = {}
    for a_code, t_code in MIXES:
        for vehicle, code in (("astuff", a_code), ("tesla", t_code)):
            alias = f"mix_{a_code}_{t_code}_{vehicle}"
            out[alias] = STEM_BY_CODE[code].format(vehicle=vehicle)
    return out


def sync_symlinks(aliases: dict[str, str], *, dry_run: bool) -> list[str]:
    """Ensure all sensor-model alias symlinks exist and point at the current
    real files. Returns a human-readable action log."""
    log: list[str] = []
    for alias, source in sorted(aliases.items()):
        for suf in SUFFIXES:
            target = f"{source}{suf}.csv"          # relative link, same dir
            if not (SENSOR_MODELS / target).is_file():
                raise SystemExit(f"missing real calibration file: "
                                 f"{SENSOR_MODELS / target}")
            link = SENSOR_MODELS / f"{alias}{suf}.csv"
            if link.is_symlink():
                if link.readlink() == Path(target):
                    continue                        # already correct
                log.append(f"repoint {link.name}: {link.readlink()} -> {target}")
            elif link.exists():
                log.append(f"replace regular file {link.name} -> {target}")
            else:
                log.append(f"create {link.name} -> {target}")
            if not dry_run:
                link.unlink(missing_ok=True)
                link.symlink_to(target)
    return log


def sync_alias_rows(csv_path: Path, aliases: dict[str, str],
                    *, dry_run: bool) -> list[str]:
    """Make ``csv_path`` (detector-keyed, detector in column 0) carry, for each
    alias, exact copies of the source detector's CURRENT rows. Existing alias
    rows are replaced (never duplicated); the file is rewritten only on change.
    Returns a human-readable action log."""
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]

    log: list[str] = []
    kept = [r for r in body if r and r[0] not in aliases]
    new_body = list(kept)
    for alias, source in sorted(aliases.items()):
        src_rows = [r for r in kept if r and r[0] == source]
        if not src_rows:
            raise SystemExit(
                f"{csv_path.name}: no rows for source detector '{source}' "
                f"(needed by alias '{alias}'). Calibrate the source first.")
        new_body.extend([alias] + r[1:] for r in src_rows)

        old_rows = [r for r in body if r and r[0] == alias]
        want = [[alias] + r[1:] for r in src_rows]
        if old_rows == want:
            continue
        verb = "update" if old_rows else "add"
        log.append(f"{csv_path.name}: {verb} {len(want)} row(s) "
                   f"{alias} <- {source}")

    if log and not dry_run:
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(new_body)
    return log


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="print actions without touching anything")
    args = ap.parse_args()

    aliases = alias_map()
    log = sync_symlinks(aliases, dry_run=args.dry_run)
    log += sync_alias_rows(CURVES_CSV, aliases, dry_run=args.dry_run)
    log += sync_alias_rows(LIFECYCLE_CSV, aliases, dry_run=args.dry_run)

    prefix = "[dry-run] " if args.dry_run else ""
    if not log:
        print(f"{prefix}all {len(aliases)} mix aliases already up to date "
              f"(symlinks + gate curves + lifecycle rows)")
    for line in log:
        print(prefix + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
