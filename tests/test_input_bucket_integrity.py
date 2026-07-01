"""Integrity guard for ``data/v2v4real_inputs/`` buckets.

For each registered bucket, this test:
  1. Verifies a ``MANIFEST.json`` is present and parseable at the current
     schema version.
  2. Re-computes the sha256 of every file the manifest lists and asserts the
     bytes on disk match.
  3. Asserts there are no "extra" files in the bucket's data subdirs that the
     manifest does not know about.
  4. Asserts the bucket's ``README.md`` exists and references at least one
     valid experiment from the registry (or is explicitly marked as having
     no consumers yet).

Run by default (fast — a few seconds per bucket). If it fails after an
intentional change to a bucket, regenerate the affected manifest with::

    python scripts/build_v2v4real_input_bucket.py --bucket <name> \\
        --rebuild manifest readme --apply

then commit the updated MANIFEST.json + README.md alongside the data change.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from input_bucket.manifest import (  # noqa: E402
    SCHEMA_VERSION, read_manifest, verify_against_disk,
)
from input_bucket.registry import BUCKETS  # noqa: E402


V2V4REAL_INPUTS_ROOT = REPO / "data" / "v2v4real_inputs"


def _bucket_ids():
    return list(BUCKETS.keys())


@pytest.fixture(scope="session")
def inputs_root() -> Path:
    if not V2V4REAL_INPUTS_ROOT.exists():
        pytest.skip(
            f"{V2V4REAL_INPUTS_ROOT} does not exist. Run Phase A skeleton + "
            f"`scripts/build_v2v4real_input_bucket.py --bucket all --rebuild manifest readme --apply`."
        )
    return V2V4REAL_INPUTS_ROOT


@pytest.mark.parametrize("bucket_label", _bucket_ids(), ids=lambda s: s.replace("/", "_"))
def test_bucket_manifest_present(inputs_root: Path, bucket_label: str) -> None:
    """Every bucket has a current-schema MANIFEST.json."""
    bucket_root = inputs_root / bucket_label
    manifest_path = bucket_root / "MANIFEST.json"
    assert bucket_root.is_dir(), f"bucket directory missing: {bucket_root}"
    assert manifest_path.is_file(), (
        f"{manifest_path} not found. Generate with:\n"
        f"    python scripts/build_v2v4real_input_bucket.py --bucket {bucket_label} "
        f"--rebuild manifest --apply"
    )
    m = read_manifest(bucket_root)
    assert m.schema_version == SCHEMA_VERSION, (
        f"{manifest_path}: schema_version {m.schema_version} != current {SCHEMA_VERSION}"
    )
    assert m.bucket == bucket_label, (
        f"{manifest_path}: bucket label {m.bucket!r} != expected {bucket_label!r}"
    )


@pytest.mark.parametrize("bucket_label", _bucket_ids(), ids=lambda s: s.replace("/", "_"))
def test_bucket_bytes_match_manifest(inputs_root: Path, bucket_label: str) -> None:
    """Every file in the bucket sha256-matches what the manifest says."""
    bucket_root = inputs_root / bucket_label
    problems = verify_against_disk(bucket_root)
    if problems:
        msg = (
            f"\n{bucket_label}: {len(problems)} integrity problem(s):\n  - "
            + "\n  - ".join(problems)
            + f"\n\nIf this drift is intentional, regenerate with:\n"
            f"    python scripts/build_v2v4real_input_bucket.py --bucket {bucket_label} "
            f"--rebuild manifest --apply"
        )
        pytest.fail(msg)


@pytest.mark.parametrize("bucket_label", _bucket_ids(), ids=lambda s: s.replace("/", "_"))
def test_bucket_readme_present(inputs_root: Path, bucket_label: str) -> None:
    """Every bucket has a README.md."""
    readme = inputs_root / bucket_label / "README.md"
    assert readme.is_file(), (
        f"{readme} not found. Generate with:\n"
        f"    python scripts/build_v2v4real_input_bucket.py --bucket {bucket_label} "
        f"--rebuild readme --apply"
    )
    text = readme.read_text()
    # README is templated; if it doesn't start with the bucket header, the
    # template drifted (e.g. someone hand-edited it). Fail loudly so the
    # template stays canonical.
    expected_header = f"# `{bucket_label}/`"
    assert text.startswith(expected_header), (
        f"{readme} no longer starts with {expected_header!r}. "
        f"It looks hand-edited; the README is generated. "
        f"Encode any custom prose in scripts/input_bucket/registry.py instead."
    )


@pytest.mark.parametrize("bucket_label", _bucket_ids(), ids=lambda s: s.replace("/", "_"))
def test_registry_entry_consistent(inputs_root: Path, bucket_label: str) -> None:
    """The registry entry's ``consumed_by_configs`` references real configs."""
    spec = BUCKETS[bucket_label]
    if not spec.consumed_by_configs:
        pytest.skip(f"{bucket_label}: no configs registered yet (expected pre-Phase F)")
    try:
        from v2v4real_experiments import get as get_experiment  # type: ignore
    except ImportError:
        # configs/ not on sys.path when running outside the cmr environment.
        # Add it and retry once; if that still fails, accept it as a skip
        # rather than masking unrelated import errors.
        sys.path.insert(0, str(REPO / "configs"))
        try:
            from v2v4real_experiments import get as get_experiment  # type: ignore
        except ImportError as e:
            pytest.skip(f"cannot import v2v4real_experiments to validate configs: {e}")
    missing: list[str] = []
    for name in spec.consumed_by_configs:
        try:
            get_experiment(name)
        except KeyError:
            missing.append(name)
    assert not missing, (
        f"{bucket_label}: registry lists configs that don't exist in v2v4real_experiments.py: "
        f"{missing}. Either add them to the registry, remove them from registry.py, "
        f"or fix the misspelling."
    )
