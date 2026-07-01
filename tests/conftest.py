"""Shared pytest config for the cmr test suite.

Adds the ``slow`` marker (so tests marked ``@pytest.mark.slow`` don't trigger
``PytestUnknownMarkWarning``) and a ``--runslow`` opt-in flag. Slow tests are
skipped by default; run them explicitly with::

    pytest tests/ --runslow

or invoke a slow test file directly (its slow-marked tests are auto-included)::

    pytest tests/test_paper_anchor_smoke.py -m slow
"""

import pytest

# Quarantined / superseded tests live under tests/old/ and reference removed
# modules; never collect them in a normal run.
collect_ignore_glob = ["old/*"]


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow-marked tests (the paper-anchor AMOTA smoke suite, etc.)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: tests that take >30s; skipped by default — opt in with --runslow",
    )


def pytest_collection_modifyitems(config, items):
    """Skip slow tests by default unless --runslow or `-m slow` is set."""
    if config.getoption("--runslow") or "slow" in (config.getoption("-m") or ""):
        return
    skip_slow = pytest.mark.skip(reason="slow — opt in with --runslow or -m slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
