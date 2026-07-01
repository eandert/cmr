#!/usr/bin/env python3
"""Create per-mix sensor model symlinks so the existing {vehicle} template
in BENCHMARK_CONFIGS just works for cross-detector mixes.

For each mix combo (e.g., mix_pp_cpft), creates 8 symlinks:
  src/data/sensor_models/mix_pp_cpft_astuff.csv → pointpillar_v2v4real_score02_astuff.csv
  src/data/sensor_models/mix_pp_cpft_astuff_distributions.csv → pointpillar_..._distributions.csv
  src/data/sensor_models/mix_pp_cpft_astuff_polar_calibration.csv → pointpillar_..._polar_calibration.csv
  src/data/sensor_models/mix_pp_cpft_astuff_polar_distributions.csv → pointpillar_..._polar_distributions.csv
  ... + 4 for tesla side

This way `detector_name="mix_pp_cpft_{vehicle}"` resolves to the right per-vehicle model
without any runner code changes.

Usage:
    python scripts/setup_mix_sensor_symlinks.py [--clean] [--dry-run]
"""
import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src/data/sensor_models"

# (mix_name, astuff_target_base, tesla_target_base)
# target_base is the stem of the real sensor model files (without _distributions etc.)
MIXES = [
    ("mix_pp_cpft",     "pointpillar_v2v4real_score02_astuff",       "centerpoint_54m_v2v4real_finetune_tesla"),
    ("mix_cpft_pp",     "centerpoint_54m_v2v4real_finetune_astuff",  "pointpillar_v2v4real_score02_tesla"),
    ("mix_pp_cpzs",     "pointpillar_v2v4real_score02_astuff",       "centerpoint_100m_on_v2v4real"),
    ("mix_cpzs_pp",     "centerpoint_100m_on_v2v4real",              "pointpillar_v2v4real_score02_tesla"),
    ("mix_cpft_cpzs",   "centerpoint_54m_v2v4real_finetune_astuff",  "centerpoint_100m_on_v2v4real"),
    ("mix_cpzs_cpft",   "centerpoint_100m_on_v2v4real",              "centerpoint_54m_v2v4real_finetune_tesla"),
]

# Per-base file suffixes (the loader looks for any of these per stem)
SUFFIXES = [".csv", "_distributions.csv", "_polar_calibration.csv", "_polar_distributions.csv"]


def setup_symlinks(dry_run: bool, clean: bool) -> None:
    if not SRC.exists():
        raise SystemExit(f"sensor_models dir not found: {SRC}")

    n_created = 0
    n_skipped = 0
    n_cleaned = 0

    for mix_name, astuff_base, tesla_base in MIXES:
        for vehicle, target_base in (("astuff", astuff_base), ("tesla", tesla_base)):
            for suffix in SUFFIXES:
                link_name = f"{mix_name}_{vehicle}{suffix}"
                target_name = f"{target_base}{suffix}"
                link_path = SRC / link_name
                target_path = SRC / target_name

                if clean and link_path.is_symlink():
                    if dry_run:
                        print(f"  [dry-run] rm symlink {link_name}")
                    else:
                        link_path.unlink()
                    n_cleaned += 1
                    continue

                if not target_path.exists():
                    print(f"  [skip] target missing: {target_name}")
                    n_skipped += 1
                    continue

                if link_path.exists() or link_path.is_symlink():
                    if link_path.is_symlink() and link_path.resolve() == target_path.resolve():
                        n_skipped += 1
                        continue
                    if dry_run:
                        print(f"  [dry-run] would replace {link_name}")
                    else:
                        link_path.unlink()

                if dry_run:
                    print(f"  [dry-run] {link_name} → {target_name}")
                else:
                    link_path.symlink_to(target_name)  # relative symlink within same dir
                n_created += 1

    if clean:
        print(f"\nCleaned {n_cleaned} symlinks.")
    else:
        print(f"\nCreated/verified {n_created} symlinks; skipped {n_skipped}.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", action="store_true",
                    help="Remove all mix_* symlinks instead of creating them")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print actions without making changes")
    args = ap.parse_args()
    setup_symlinks(args.dry_run, args.clean)


if __name__ == "__main__":
    main()
