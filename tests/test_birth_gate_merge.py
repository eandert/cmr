"""Unit tests for the merge-safe aggregate-CSV writes in derive_optimal_birth_gate.

``derive_optimal_birth_gate.py`` is run per-bucket (one or two detectors at a
time) by ``build_v2v4real_input_bucket.py``'s ``gate`` step. Its two aggregate
outputs — ``birth_gate_curves.csv`` and ``recommended_gates.csv`` — hold one
group of rows PER detector across all buckets, so a per-bucket run must NOT
clobber other detectors' rows. These tests pin that merge contract:

  - writing detector B into a file that already has detector A keeps BOTH;
  - re-writing detector A replaces A's rows and PRESERVES B's.

Run: PYTHONPATH=src python -m pytest tests/test_birth_gate_merge.py -v
"""
import csv
import importlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

gate = importlib.import_module("derive_optimal_birth_gate")


_CURVES_HEADER = ["detector", "alpha", "fit_type",
                  "a_quad", "b_quad", "c_quad",
                  "a_lin", "b_lin",
                  "n_range_bins"]


def _detectors_in(path: Path) -> list[str]:
    """Return the ordered list of detector names (column 0) in a curves CSV."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    assert rows[0] == _CURVES_HEADER, f"unexpected header: {rows[0]}"
    return [r[0] for r in rows[1:] if r]


def _curve_rows_for(path: Path, detector: str) -> list[list[str]]:
    with open(path, newline="") as f:
        return [r for r in csv.reader(f) if r and r[0] == detector]


def test_merge_keeps_existing_detector(tmp_path):
    """Writing detector B into a file holding detector A keeps both A and B."""
    curves = tmp_path / "birth_gate_curves.csv"

    # Seed a fake curves CSV with detector A (3 alpha rows, like the real schema).
    a_rows = [
        ["detector_A", 1.0, "tp_preserve_a1", 0.1, 0.2, 0.3, 0.4, 0.5, 7],
        ["detector_A", 2.0, "tp_preserve_a2", 0.1, 0.2, 0.3, 0.4, 0.5, 7],
        ["detector_A", "", "bayes", 0.1, 0.2, 0.3, 0.4, 0.5, 7],
    ]
    gate._merge_write_csv(curves, _CURVES_HEADER, a_rows, {"detector_A"})
    assert set(_detectors_in(curves)) == {"detector_A"}

    # Now write detector B — A must be preserved.
    b_rows = [
        ["detector_B", 1.0, "tp_preserve_a1", 1.1, 1.2, 1.3, 1.4, 1.5, 5],
        ["detector_B", 2.0, "tp_preserve_a2", 1.1, 1.2, 1.3, 1.4, 1.5, 5],
        ["detector_B", "", "bayes", 1.1, 1.2, 1.3, 1.4, 1.5, 5],
    ]
    gate._merge_write_csv(curves, _CURVES_HEADER, b_rows, {"detector_B"})

    present = _detectors_in(curves)
    assert set(present) == {"detector_A", "detector_B"}
    # A's rows are byte-for-byte what we seeded (round-tripped through csv).
    assert len(_curve_rows_for(curves, "detector_A")) == 3
    assert len(_curve_rows_for(curves, "detector_B")) == 3


def test_merge_replaces_rerun_detector_preserves_others(tmp_path):
    """Re-running detector A replaces A's rows and preserves detector B."""
    curves = tmp_path / "birth_gate_curves.csv"

    a_rows_old = [["detector_A", 1.0, "tp_preserve_a1", 0.1, 0.2, 0.3, 0.4, 0.5, 7]]
    b_rows = [["detector_B", 1.0, "tp_preserve_a1", 1.1, 1.2, 1.3, 1.4, 1.5, 5]]
    gate._merge_write_csv(curves, _CURVES_HEADER, a_rows_old, {"detector_A"})
    gate._merge_write_csv(curves, _CURVES_HEADER, b_rows, {"detector_B"})
    assert set(_detectors_in(curves)) == {"detector_A", "detector_B"}

    # Re-run A with a NEW value — A replaced, B untouched.
    a_rows_new = [["detector_A", 1.0, "tp_preserve_a1", 9.9, 9.9, 9.9, 9.9, 9.9, 7]]
    gate._merge_write_csv(curves, _CURVES_HEADER, a_rows_new, {"detector_A"})

    assert set(_detectors_in(curves)) == {"detector_A", "detector_B"}
    a_after = _curve_rows_for(curves, "detector_A")
    assert len(a_after) == 1
    assert a_after[0][3] == "9.9"  # the replaced value, not the old 0.1
    # B's original row survived intact.
    b_after = _curve_rows_for(curves, "detector_B")
    assert b_after == [["detector_B", "1.0", "tp_preserve_a1",
                        "1.1", "1.2", "1.3", "1.4", "1.5", "5"]]


def test_merge_stale_header_is_replaced(tmp_path):
    """A pre-existing file with a mismatched header is fully replaced, not merged."""
    curves = tmp_path / "birth_gate_curves.csv"
    with open(curves, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["detector", "old_schema_col"])
        w.writerow(["detector_OLD", "stale"])

    rows = [["detector_A", 1.0, "tp_preserve_a1", 0.1, 0.2, 0.3, 0.4, 0.5, 7]]
    gate._merge_write_csv(curves, _CURVES_HEADER, rows, {"detector_A"})

    # The stale-schema detector is dropped; only the new run remains.
    assert _detectors_in(curves) == ["detector_A"]


def test_write_outputs_merges_into_existing(tmp_path, monkeypatch):
    """End-to-end: write_outputs through two runs keeps both detectors' rows."""
    monkeypatch.setattr(gate, "OUT_DIR", tmp_path)

    def _fake_result(name: str, scalar_val: float) -> dict:
        fit = {"quad_a": 0.0, "quad_b": 0.0, "quad_c": scalar_val,
               "lin_a": 0.0, "lin_b": scalar_val, "n_range_bins": 3}
        return {
            "detector": name,
            "n_bins": 4,
            "scalar": {"bayes": scalar_val,
                       "tp_preserve_a1": scalar_val,
                       "tp_preserve_a2": scalar_val},
            "per_range": {},
            "fits": {"bayes": fit, "tp_preserve_a1": fit, "tp_preserve_a2": fit},
            "per_bin": [{"range_lo": 0.0, "range_hi": 10.0, "weight": 1.0}],
        }

    gate.write_outputs([_fake_result("detector_A", 0.30)])
    gate.write_outputs([_fake_result("detector_B", 0.55)])

    curves = tmp_path / "birth_gate_curves.csv"
    rec = tmp_path / "recommended_gates.csv"
    assert set(_detectors_in(curves)) == {"detector_A", "detector_B"}
    with open(rec, newline="") as f:
        rec_detectors = {r[0] for r in csv.reader(f) if r and r[0] != "detector"}
    assert rec_detectors == {"detector_A", "detector_B"}
