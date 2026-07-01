"""Unit tests for the source-local GPEM range mechanism (audit M1).

Covers ``third_party/AB3DMOT/ab3dmot_gpem_injector._range_angle_for`` and its
per-frame offset plumbing:

  - legacy mode (disabled): planar range/bearing from the common-frame origin
  - source-local mode: range measured from the SOURCE CAV's per-frame offset,
    routed on the filter's ``_source_id`` (tesla offset (0,0,0) is
    bit-identical to legacy)
  - heading rotation: bearing expressed in the source CAV's frame; range is
    rotation-invariant
  - fallbacks: unknown source id / missing per-frame offsets → legacy math

The injector imports filterpy + puts cmr/src on sys.path at module import;
no GPEM model needs to be loaded for these tests (_range_angle_for is
model-independent by design).
"""

import math
import sys
from pathlib import Path

import pytest

AB3DMOT_DIR = Path(__file__).resolve().parent.parent / "third_party" / "AB3DMOT"
sys.path.insert(0, str(AB3DMOT_DIR))

import ab3dmot_gpem_injector as gi  # noqa: E402


class FakeFilter:
    def __init__(self, source_id=None):
        if source_id is not None:
            self._source_id = source_id


@pytest.fixture(autouse=True)
def _reset_injector_state():
    """Each test starts from the legacy default (disabled, no offsets)."""
    yield
    gi.enable_gpem_source_range(False)
    gi.set_gpem_source_offsets(None)


def legacy(x_fwd, z_lat):
    return (math.sqrt(x_fwd * x_fwd + z_lat * z_lat),
            math.degrees(math.atan2(x_fwd, z_lat)))


def test_disabled_is_legacy_math():
    gi.enable_gpem_source_range(False)
    gi.set_gpem_source_offsets({"astuff": (10.0, 5.0, 0.0)})
    flt = FakeFilter("astuff")
    assert gi._range_angle_for(flt, 20.0, 5.0) == pytest.approx(legacy(20.0, 5.0))


def test_routes_range_per_source():
    """Same measurement, two source CAVs → two different ranges."""
    gi.enable_gpem_source_range(True)
    # astuff sits 10m ahead / 5m left of tesla in the common frame.
    gi.set_gpem_source_offsets({"astuff": (10.0, 5.0, 0.0),
                                "tesla": (0.0, 0.0, 0.0)})
    det = (20.0, 5.0)  # (fwd, lat) in the common tesla-KITTI frame

    rng_tesla, ang_tesla = gi._range_angle_for(FakeFilter("tesla"), *det)
    rng_astuff, ang_astuff = gi._range_angle_for(FakeFilter("astuff"), *det)

    # tesla-sourced: bit-identical to legacy (offset is the origin).
    assert (rng_tesla, ang_tesla) == pytest.approx(legacy(*det))
    assert rng_tesla == pytest.approx(math.hypot(20.0, 5.0))
    # astuff-sourced: distance from ASTUFF's position, i.e. exactly 10m
    # dead-ahead of astuff.
    assert rng_astuff == pytest.approx(10.0)
    assert ang_astuff == pytest.approx(90.0)  # atan2(fwd=10, lat=0)
    assert rng_astuff != pytest.approx(rng_tesla)


def test_heading_rotation_changes_bearing_not_range():
    gi.enable_gpem_source_range(True)
    det = (20.0, 5.0)
    gi.set_gpem_source_offsets({"astuff": (10.0, 5.0, 0.0)})
    rng0, ang0 = gi._range_angle_for(FakeFilter("astuff"), *det)
    # Same position, source heading rotated +90° vs tesla.
    gi.set_gpem_source_offsets({"astuff": (10.0, 5.0, math.pi / 2.0)})
    rng90, ang90 = gi._range_angle_for(FakeFilter("astuff"), *det)
    assert rng90 == pytest.approx(rng0)          # range is rotation-invariant
    # Bearing convention is atan2(fwd, lat) = 90° − standard angle, so a +90°
    # source heading shifts the reported bearing by +90° (mod 360). Verified
    # against direct source-frame math in the round-trip test below.
    assert (ang90 - ang0) % 360.0 == pytest.approx(90.0)


def test_unknown_source_falls_back_to_legacy():
    gi.enable_gpem_source_range(True)
    gi.set_gpem_source_offsets({"astuff": (10.0, 5.0, 0.0)})
    before = gi._SOURCE_RANGE_STATS["fallback"]
    # source id present but no offset row for it this frame
    out = gi._range_angle_for(FakeFilter("mystery_cav"), 20.0, 5.0)
    assert out == pytest.approx(legacy(20.0, 5.0))
    assert gi._SOURCE_RANGE_STATS["fallback"] == before + 1
    # no source id at all (single-stream mode) → legacy too
    assert gi._range_angle_for(FakeFilter(), 20.0, 5.0) == pytest.approx(legacy(20.0, 5.0))


def test_empty_offsets_frame_falls_back():
    """Frames with no pose rows push {} — every lookup is legacy that frame."""
    gi.enable_gpem_source_range(True)
    gi.set_gpem_source_offsets({})
    assert gi._range_angle_for(FakeFilter("astuff"), 20.0, 5.0) == \
        pytest.approx(legacy(20.0, 5.0))


def test_matches_pose_derived_offset_round_trip():
    """End-to-end synthetic: world poses → main.py-style offsets → range equals
    direct world-frame distance between detection and source CAV."""
    # tesla at world (100, 50), yaw 30°; astuff at world (130, 70), yaw 75°.
    tx, ty, tyaw = 100.0, 50.0, math.radians(30.0)
    sx, sy, syaw = 130.0, 70.0, math.radians(75.0)
    # target at world (150, 90)
    wx, wy = 150.0, 90.0

    def world_to_tesla(px, py):
        dx, dy = px - tx, py - ty
        c, s = math.cos(-tyaw), math.sin(-tyaw)
        return c * dx - s * dy, s * dx + c * dy  # (fwd, lat)

    det_fwd, det_lat = world_to_tesla(wx, wy)
    off_fwd, off_lat = world_to_tesla(sx, sy)
    dyaw = (syaw - tyaw + math.pi) % (2 * math.pi) - math.pi

    gi.enable_gpem_source_range(True)
    gi.set_gpem_source_offsets({"astuff": (off_fwd, off_lat, dyaw)})
    rng, ang = gi._range_angle_for(FakeFilter("astuff"), det_fwd, det_lat)
    assert rng == pytest.approx(math.hypot(wx - sx, wy - sy), abs=1e-9)
    # bearing in the source frame: rotate world delta into astuff's heading
    dx, dy = wx - sx, wy - sy
    c, s = math.cos(-syaw), math.sin(-syaw)
    src_fwd, src_lat = c * dx - s * dy, s * dx + c * dy
    assert ang == pytest.approx(math.degrees(math.atan2(src_fwd, src_lat)), abs=1e-9)
