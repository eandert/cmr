"""Guard: cmr must be runnable without any external workstation paths.

This test walks every source/config/test file under cmr and fails if any of
them reference absolute paths to peer repos on a specific workstation. Those
references would break a fresh clone of the open-sourced cmr.

Allowed exceptions (greenlit by ``_ALLOWLIST``):

* Documentation under ``docs/`` and ``results/`` (notes, audits, plans)
* Per-bucket ``README.md`` provenance notes
* The historical ``cmr/results/`` audit reports
* ``src/eval_config.py``: ONE fallback to a workstation path is allowed,
  guarded by ``Path.exists()``, for the AB3DMOT-not-yet-submoduled transition

If you NEED to reference an external path in code, encode it as a registry
entry in ``scripts/input_bucket/registry.py`` (declarative, easy to audit) or
gate it behind a check on ``AB3DMOT_ROOT`` from ``eval_config.py``.

Run by default; should pass in <2 seconds.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# Workstation-specific path prefixes that MUST NOT appear in cmr code.
# Placeholders like ``<your-mmdet3d>`` and ``<dmstrack>`` ARE allowed — they
# document where a developer should fill in their own local clone path
# (typically in docstring usage examples). The forbidden set is concrete
# workstation paths only.
FORBIDDEN_PATTERNS = [
    re.compile(r"/home/rave/test/DMSTrack"),
    re.compile(r"/home/rave/test/mmdetection3d"),
    re.compile(r"/home/rave/test/opencood"),
    re.compile(r"/home/rave/may-old"),
    # /home/rave/test/cmr/ itself is fine when used as a relative anchor in
    # docs; we only catch peers.
]

# Directories scanned for forbidden paths.
SCAN_DIRS = ("src", "scripts", "configs", "tests")

# Files allowed to mention forbidden paths. Each entry pairs a path-suffix
# match with the justification — short list, audited by reviewers.
_ALLOWLIST: list[tuple[str, str]] = [
    # The build CLI's docstring + registry mention paths under
    # mmdetection3d/work_dirs as the upstream artifact source. That's
    # provenance, not a hard runtime dependency.
    ("scripts/input_bucket/registry.py", "upstream provenance metadata only"),
    ("scripts/build_v2v4real_input_bucket.py", "docstring example references"),
    # This test file itself names the forbidden paths.
    ("tests/test_open_source_independence.py", "this test names the forbidden paths"),
    # src/local_paths.py defines the developer-set external paths mechanism;
    # the module's docstring references upstream repo URLs.
    ("src/local_paths.py", "developer local-paths config helper"),
]


def _is_allowlisted(relpath: str) -> bool:
    return any(relpath.endswith(allowed) for allowed, _ in _ALLOWLIST)


def _scan_files() -> list[Path]:
    out: list[Path] = []
    for d in SCAN_DIRS:
        root = REPO / d
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix not in (".py", ".cfg", ".toml", ".yaml", ".yml", ".json", ".sh"):
                continue
            if "__pycache__" in p.parts:
                continue
            out.append(p)
    return sorted(out)


def test_no_workstation_paths_in_code():
    """No cmr code file may hardcode a peer workstation path."""
    violations: list[tuple[str, str, int, str]] = []  # (relpath, pattern, lineno, line)
    for path in _scan_files():
        relpath = str(path.relative_to(REPO))
        if _is_allowlisted(relpath):
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for pat in FORBIDDEN_PATTERNS:
                if pat.search(line):
                    violations.append((relpath, pat.pattern, lineno, line.strip()))
    if violations:
        msg = "\nForbidden workstation paths found in cmr code:\n"
        for relpath, pat, lineno, line in violations:
            msg += f"  {relpath}:{lineno}  matches {pat!r}\n    {line[:120]}\n"
        msg += (
            "\nFix: route external paths through src/eval_config.py constants, "
            "or encode them as registry metadata in scripts/input_bucket/registry.py. "
            "If this is intentional (and existence-gated), add a precise allowlist "
            "entry to _ALLOWLIST in tests/test_open_source_independence.py."
        )
        pytest.fail(msg)


def test_eval_config_has_canonical_constants():
    """eval_config exposes the canonical v2v4real_inputs constants."""
    import sys
    sys.path.insert(0, str(REPO / "src"))
    import eval_config
    expected = [
        "V2V4REAL_INPUTS_ROOT",
        "DMSTRACK_PP_BUCKET", "COBEVT_BUCKET", "PAPER_GT_BUCKET",
        "PP_SCORE0_BUCKET", "CP_ZEROSHOT_100M_BUCKET", "CP_FINETUNE_54M_BUCKET",
        "AUGMENTED_GT_BUCKET",
        "DMSTRACK_PP_DETS", "COBEVT_DETS", "PP_SCORE0_DETS",
        "CP_ZEROSHOT_100M_DETS", "CP_FINETUNE_54M_DETS",
        "PAPER_GT_LABELS", "AUGMENTED_GT_WITH_CAV", "AUGMENTED_GT_WITH_CAV_MERGED",
    ]
    missing = [name for name in expected if not hasattr(eval_config, name)]
    assert not missing, (
        f"eval_config.py is missing canonical constants: {missing}. "
        f"Phase F partial expects all bucket roots + per-bucket detection / GT "
        f"path constants to be exported."
    )
