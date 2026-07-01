"""Developer-set local paths to peer repos (mmdetection3d, DMSTrack, etc.).

THIS MODULE IS NOT NEEDED to run cmr's headline numbers from a fresh clone.
Our pre-computed detection inputs, GPEM calibrations, and augmented GT all
live inside ``cmr/data/v2v4real_inputs/`` (Tier 1) or are fetched by
``scripts/fetch_baselines.py`` (Tier 2). The main eval sweep
(``scripts/v2v4real_runner.py`` + ``scripts/paper_metrics.py``) does NOT
import this module.

This module IS required by developer ingest tooling::

    scripts/build_v2v4real_input_bucket.py --rebuild import   # cmr_export → AB3DMOT format
    scripts/build_v2v4real_input_bucket.py --rebuild gpem     # preds.pkl → GPEM CSVs
    scripts/fetch_baselines.py --use-local-clones             # offline / dev mode

To use these tools, copy the template::

    cp paths.local.yaml.example paths.local.yaml

and edit ``paths.local.yaml`` (gitignored) to point at YOUR local clones of
the upstream repos. The example file lists the keys we recognize.

If a path you need isn't set, you get a friendly error explaining which key
to add to ``paths.local.yaml`` and where the upstream repo lives.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional


_CMR_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _CMR_ROOT / "paths.local.yaml"
_EXAMPLE_CONFIG = _CMR_ROOT / "paths.local.yaml.example"
_ENV_OVERRIDE = "CMR_PATHS_CONFIG"


# Documented keys → (description, example value, upstream repo URL).
KNOWN_KEYS: dict[str, tuple[str, str, str]] = {
    "mmdet3d_root": (
        "Local clone of mmdetection3d (used to refit GPEM regressions and "
        "re-export detector preds → cmr_export format)",
        "/path/to/mmdetection3d",
        "https://github.com/eandert/mmdetection3d (pending publication)",
    ),
    "dmstrack_root": (
        "Local clone of DMSTrack (developer fallback for fetching the DMSTrack "
        "PP baseline detections offline)",
        "/path/to/DMSTrack",
        "https://github.com/eddyhkchiu/DMSTrack",
    ),
    "opencood_root": (
        "Local clone of OpenCOOD (used by cobevt_dets_to_cmr_export.py to "
        "convert KITTI dumps from CoBEVT inference)",
        "/path/to/OpenCOOD",
        "https://github.com/DerrickXuNu/OpenCOOD",
    ),
    "v2v4real_root": (
        "Local clone of V2V4Real (developer fallback for fetching the paper "
        "GT and CoBEVT baseline offline)",
        "/path/to/V2V4Real",
        "https://github.com/ucla-mobility/V2V4Real",
    ),
}


def _config_path() -> Path:
    """The file to load — env override wins over the default location."""
    return Path(os.environ.get(_ENV_OVERRIDE, _DEFAULT_CONFIG))


@lru_cache(maxsize=1)
def _load_config() -> dict[str, str]:
    """Parse ``paths.local.yaml`` once. Tolerates the file being absent.

    We hand-parse a tiny ``key: value`` subset rather than depend on PyYAML so
    the headline eval path stays dependency-light. Comments (``#``) and blank
    lines are ignored. Anything more exotic and we'd suggest the user install
    PyYAML — but for a few path strings, plain text is enough.
    """
    path = _config_path()
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip().strip("'\"")
        if value:
            out[key] = value
    return out


def get(key: str, *, required: bool = True) -> Optional[Path]:
    """Look up a developer-local path by key. Returns ``Path`` or ``None``.

    ``required=True`` (default): raises ``FileNotFoundError`` with a helpful
    message naming the key, the upstream repo, and the line to add to
    ``paths.local.yaml`` if the key is missing or the file doesn't exist.

    ``required=False``: returns ``None`` if the key is unset — useful for
    ingest scripts that have a different fallback (e.g. fetch from URL).
    """
    if key not in KNOWN_KEYS:
        raise KeyError(
            f"unknown local-path key {key!r}. Known keys: "
            f"{sorted(KNOWN_KEYS)}. If you need a new one, add it to "
            f"src/local_paths.py::KNOWN_KEYS first."
        )
    config = _load_config()
    if key in config:
        p = Path(config[key]).expanduser()
        if not p.exists():
            raise FileNotFoundError(
                f"paths.local.yaml has {key!r} = {p!s}, but that directory "
                f"doesn't exist. Edit paths.local.yaml or run the ingest "
                f"script with a different value."
            )
        return p
    if not required:
        return None
    desc, example, upstream = KNOWN_KEYS[key]
    raise FileNotFoundError(
        f"\n\nMissing local path for {key!r}.\n"
        f"  Description: {desc}\n"
        f"  Upstream:    {upstream}\n"
        f"\n  Fix: copy the template, then add the path:\n"
        f"    cp {_EXAMPLE_CONFIG.relative_to(_CMR_ROOT)} {_DEFAULT_CONFIG.relative_to(_CMR_ROOT)}\n"
        f"    # then edit paths.local.yaml:\n"
        f"    {key}: {example}\n"
    )


def keys_set() -> list[str]:
    """Names of keys actually defined in ``paths.local.yaml`` (or env override)."""
    return sorted(_load_config().keys())
