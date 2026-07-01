"""Unit + acceptance tests for the instant-tune derivations (spec §2.2).

Each §2.2 derivation (F, L, C, K, G, Rng, W) gets a synthetic-stats unit
test with a hand-computable optimum, plus the (L) ACCEPTANCE test from the
spec, evaluated on the REAL cp_finetune_100m calibration:

  - protocol=v2v  → effectively immediate emit (m*=1, confirm_log_odds
                    at-or-below the single-detection/birth log-odds) on all
                    three detectors;
  - protocol=ours → still near-immediate on CPft (manual mh1ma5 evidence:
                    Ours 37.95 → 39.79);
  - protocol=ours → stricter on pp_score0 (soft expectation — encoded as
                    xfail, see test docstring).

Run: pytest tests/test_instant_tune_derivations.py -v
"""
import importlib
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

dic = importlib.import_module("derive_instant_config")


# ---------------------------------------------------------------------------
# Synthetic bin/stats builders
# ---------------------------------------------------------------------------

def make_bin(**kw):
    """One synthetic polar bin row with sane defaults (load_bins schema)."""
    b = {
        "range_lo": 0.0, "range_hi": 5.0, "angle_lo": -180.0, "angle_hi": -170.0,
        "gt_count": 100.0, "missed": 10.0, "n_fp": 50.0,
        "score_mean": 0.7, "score_std": 0.1,
        "fp_score_mean": 0.2, "fp_score_std": 0.05,
        "tp_lifetime_mean": 50.0, "tp_lifetime_std": 10.0,
        "fp_lifetime_mean": 2.0, "fp_lifetime_std": 1.0,
        "n_gt_tracks": 10.0, "n_fp_clusters": 25.0,
        "gt_density_per_scen_mean": 0.3, "fp_density_per_scen_mean": 0.1,
        "fp_per_frame": 0.1,
        "tp_p_match": 0.9, "fp_p_match": 1.0,
        "tp_fragments": 2.0, "fp_fragments": 1.0,
    }
    b.update(kw)
    return b


def make_stats(**kw):
    """Aggregate-stats dict with defaults; overrides via kwargs."""
    s = {
        "n_tp": 10000.0, "n_fp": 1000.0,
        "n_gt_tracks": 100.0, "n_fp_clusters": 500.0,
        "L_tp": 50.0, "L_fp": 2.0,
        "p_match": 0.9, "p_miss": 0.1, "frags": 1.0,
        "mu_tp": 0.7, "sig_tp": 0.1,
    }
    s.update(kw)
    return s


# ---------------------------------------------------------------------------
# (W) protocol weight
# ---------------------------------------------------------------------------

def test_w_ours_coast_amplification():
    """w_ours = (L_fp + L_tp·p_miss)/L_fp with known numbers → 3.5."""
    stats = make_stats(L_fp=2.0, L_tp=50.0, p_miss=0.1)
    out = dic.derive_w_ours(stats)
    # T_coast = min(50·0.1, 10) = 5 → w = (2+5)/2 = 3.5
    assert out["t_coast"] == pytest.approx(5.0)
    assert out["w"] == pytest.approx(3.5)


def test_w_ours_coast_tail_capped():
    """The coast tail is capped at COAST_CAP frames."""
    stats = make_stats(L_fp=2.0, L_tp=80.0, p_miss=0.5)   # raw tail = 40
    out = dic.derive_w_ours(stats)
    assert out["t_coast"] == pytest.approx(dic.COAST_CAP)
    assert out["w"] == pytest.approx((2.0 + dic.COAST_CAP) / 2.0)


def test_w_v2v_pollution_tracks_postgate_fp_mass():
    """w_v2v = GT co-occupancy of the FP mass that SURVIVES the gate.

    Low-score FPs live in empty bins (occ 0.05), high-score FPs in busy
    bins (occ 0.8). With a gate at 0.35 only the busy-bin FPs survive, so
    the pollution fixed point must land at ≈0.8 — not the raw-mass mean.
    """
    rows = [
        make_bin(n_fp=1000.0, fp_score_mean=0.10, fp_score_std=0.01,
                 gt_density_per_scen_mean=0.05),
        make_bin(n_fp=1000.0, fp_score_mean=0.60, fp_score_std=0.01,
                 gt_density_per_scen_mean=0.80),
    ]
    stats = make_stats()
    out = dic.derive_w_v2v(rows, stats, gate_for_w=lambda w: 0.35)
    assert out["w"] == pytest.approx(0.80, abs=1e-3)
    # and with no gate, both populations pollute equally → mass mean 0.425
    out0 = dic.derive_w_v2v(rows, stats, gate_for_w=lambda w: 0.0)
    assert out0["w"] == pytest.approx(0.425, abs=1e-3)


# ---------------------------------------------------------------------------
# (F) score floor
# ---------------------------------------------------------------------------

def test_floor_degenerates_to_zero_for_case_b_bands():
    """A single no-gate (case-B) range band pins the floor at 0."""
    per_range = {"src": {"0-5m": 0.0, "5-10m": 0.30, "10-15m": 0.25}}
    rows = [make_bin()]
    out = dic.derive_floor(per_range, rows)
    assert out["floor"] == 0.0


def test_floor_is_min_gate_over_ranges():
    """All-positive gate curve → floor = min_d gate(d) (below the FP clip)."""
    per_range = {"src": {"0-5m": 0.40, "5-10m": 0.22, "10-15m": 0.30}}
    # FP mixture well above the candidate floor → clip does not bind
    rows = [make_bin(n_fp=1000.0, fp_score_mean=0.5, fp_score_std=0.1)]
    out = dic.derive_floor(per_range, rows)
    assert out["floor"] == pytest.approx(0.22)


def test_floor_clipped_at_fp_mixture_p99():
    """The floor can never exceed (effectively) the whole FP distribution."""
    per_range = {"src": {"0-5m": 0.70}}
    rows = [make_bin(n_fp=1000.0, fp_score_mean=0.10, fp_score_std=0.02)]
    out = dic.derive_floor(per_range, rows)
    # P99 of N(0.10, 0.02) ≈ 0.10 + 2.33·0.02 ≈ 0.147
    assert out["floor"] == pytest.approx(0.1465, abs=5e-3)
    assert out["floor"] < 0.70


def test_floor_fleet_min_across_sources():
    """Per-CAV floors combine with a fleet MIN (protect the weaker CAV)."""
    per_range = {"astuff": {"0-5m": 0.30}, "tesla": {"0-5m": 0.20}}
    rows = [make_bin(n_fp=1000.0, fp_score_mean=0.5, fp_score_std=0.1)]
    out = dic.derive_floor(per_range, rows)
    assert out["floor"] == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# (L) confirm delay
# ---------------------------------------------------------------------------

def test_confirm_immediate_under_fp_ignore():
    """v2v (FP-ignore) prices emitted FPs at zero → J(m) is increasing and
    m* = 1 regardless of how FP-heavy the stream is."""
    stats = make_stats(n_gt_tracks=100.0, frags=2.0, L_fp=3.0)
    out = dic.derive_confirm(stats, "v2v", postgate_clusters=1e6, d_sweep=1.0)
    assert out["m_star"] == 1
    js = list(out["j_table"].values())
    assert js == sorted(js)


def test_confirm_known_optimum_m2_under_ours():
    """Hand-built economics with the optimum at exactly m*=2.

    q = 1−1/L_fp = 0.5; T_coast = min(L_tp·p_miss, 10) = 2;
    fp_price = clusters·d_sweep·(L_fp+T) = 750·0.1·4 = 300;
    step = n_tracks·frags = 100.
    J(1)=300, J(2)=100+150=250, J(3)=200+75=275 → m*=2.
    """
    stats = make_stats(L_fp=2.0, L_tp=20.0, p_miss=0.1,
                       n_gt_tracks=100.0, frags=1.0)
    out = dic.derive_confirm(stats, "ours", postgate_clusters=750.0,
                             d_sweep=0.1)
    assert out["m_star"] == 2
    assert out["j_table"][1] == pytest.approx(300.0)
    assert out["j_table"][2] == pytest.approx(250.0)
    assert out["j_table"][3] == pytest.approx(275.0)


def test_confirm_immediate_for_clean_detector_under_ours():
    """Cheap, sweep-discounted FPs do not justify delaying confirmation."""
    stats = make_stats(L_fp=2.0, L_tp=50.0, p_miss=0.1,
                       n_gt_tracks=1000.0, frags=2.5)
    out = dic.derive_confirm(stats, "ours", postgate_clusters=2000.0,
                             d_sweep=0.01)
    assert out["m_star"] == 1


def test_confirm_log_odds_immediate_equals_birth_logit():
    """m*=1 → confirm_log_odds = logit(p_birth): a single detection (the
    birth itself) confirms the track — the spec's immediate-emit encoding."""
    lo = dic.confirm_log_odds_equivalent(1, p_birth=0.7, p_match=0.9)
    assert lo == pytest.approx(math.log(0.7 / 0.3))


def test_confirm_log_odds_adds_hit_evidence_per_extra_frame():
    lo = dic.confirm_log_odds_equivalent(3, p_birth=0.7, p_match=0.9)
    expect = math.log(0.7 / 0.3) + 2.0 * math.log(0.9 / 0.1)
    assert lo == pytest.approx(expect)


# ---------------------------------------------------------------------------
# (C) coast budget
# ---------------------------------------------------------------------------

def test_coast_v2v_gap_quantile_known_value():
    """ḡ = L_tp·p_miss/(frags−1) = 20·0.2/2 = 2 → geometric keep-rate 0.5 →
    K = ceil(ln 0.05 / ln 0.5) = 5."""
    stats = make_stats(L_tp=20.0, p_miss=0.2, frags=3.0)
    out = dic.derive_coast(stats, "v2v", postgate_clusters=0.0,
                           d_sweep=0.0, m_star=1)
    assert out["gbar"] == pytest.approx(2.0)
    assert out["K"] == 5


def test_coast_v2v_capped():
    """Long-gap detectors hit the COAST_CAP ceiling under v2v."""
    stats = make_stats(L_tp=80.0, p_miss=0.3, frags=2.0)   # ḡ = 24 → clamp 20
    out = dic.derive_coast(stats, "v2v", 0.0, 0.0, 1)
    assert out["K"] == dic.COAST_CAP


def test_coast_ours_marginal_known_value():
    """Marginal bridge-vs-coast balance with hand numbers → K = 3.

    n_gaps = 100·(3−1) = 200, keep = 0.5 (ḡ=2);
    rhs = d_sweep·(n_tracks + clusters·q⁰) = 0.05·(100+900) = 50;
    200·0.5^{K−1} ≥ 50 holds through K=3 (200,100,50) and fails at K=4.
    """
    stats = make_stats(L_tp=20.0, p_miss=0.2, frags=3.0,
                       n_gt_tracks=100.0, L_fp=2.0)
    out = dic.derive_coast(stats, "ours", postgate_clusters=900.0,
                           d_sweep=0.05, m_star=1)
    assert out["K"] == 3


def test_coast_ours_not_longer_than_v2v_for_same_stats():
    """With any positive FP coast cost, ours coasts no longer than v2v."""
    stats = make_stats(L_tp=20.0, p_miss=0.2, frags=3.0,
                       n_gt_tracks=100.0, L_fp=2.0)
    k_v2v = dic.derive_coast(stats, "v2v", 0.0, 0.0, 1)["K"]
    k_ours = dic.derive_coast(stats, "ours", 900.0, 0.05, 1)["K"]
    assert k_ours <= k_v2v


# ---------------------------------------------------------------------------
# (K) sigmoid steepness
# ---------------------------------------------------------------------------

def test_sigmoid_k_recovers_default_at_typical_overlap():
    """σ_TP + σ_FP = 0.2 → k = 2/0.2 = 10 (the hand-set default)."""
    per_bin = [{"case": "A", "n_tp": 100.0, "n_fp": 50.0,
                "tp_sigma": 0.1, "fp_sigma": 0.1}]
    out = dic.derive_sigmoid_k(per_bin)
    assert out["k"] == pytest.approx(10.0)


def test_sigmoid_k_count_weighted_over_case_a():
    """Case-B bins are excluded when case-A bins exist; weighting by counts."""
    per_bin = [
        {"case": "A", "n_tp": 300.0, "n_fp": 100.0, "tp_sigma": 0.05, "fp_sigma": 0.05},
        {"case": "A", "n_tp": 75.0, "n_fp": 25.0, "tp_sigma": 0.20, "fp_sigma": 0.20},
        {"case": "B", "n_tp": 1e6, "n_fp": 1e6, "tp_sigma": 0.50, "fp_sigma": 0.50},
    ]
    out = dic.derive_sigmoid_k(per_bin)
    # weighted σ-sum = (400·0.1 + 100·0.4)/500 = 0.16 → k = 12.5
    assert out["sigma_sum"] == pytest.approx(0.16)
    assert out["k"] == pytest.approx(12.5)


def test_sigmoid_k_falls_back_to_all_bins_without_case_a():
    per_bin = [{"case": "B", "n_tp": 10.0, "n_fp": 10.0,
                "tp_sigma": 0.2, "fp_sigma": 0.2}]
    out = dic.derive_sigmoid_k(per_bin)
    assert out["k"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# (G) association gate
# ---------------------------------------------------------------------------

def test_mahal_gate_is_chi2_quantile_df2():
    """chi2.ppf(0.99, 2) = −2·ln(0.01) ≈ 9.2103."""
    g = dic.derive_mahal_gate()
    assert g == pytest.approx(-2.0 * math.log(0.01))
    assert g == pytest.approx(9.2103403719761818)
    scipy_stats = pytest.importorskip("scipy.stats")
    assert g == pytest.approx(scipy_stats.chi2.ppf(0.99, 2))


# ---------------------------------------------------------------------------
# (Rng) detector max range
# ---------------------------------------------------------------------------

def test_max_range_last_supported_bin():
    rows = [
        {"bin_lo": 0.0, "bin_hi": 10.0, "count": 100.0, "missed": 10.0},
        {"bin_lo": 10.0, "bin_hi": 20.0, "count": 6.0, "missed": 1.0},   # 5 TP ✓
        {"bin_lo": 20.0, "bin_hi": 30.0, "count": 5.0, "missed": 1.0},   # 4 TP ✗
    ]
    assert dic.derive_max_range_from_rows(rows) == pytest.approx(20.0)


def test_max_range_empty():
    assert dic.derive_max_range_from_rows([]) == 0.0


# ---------------------------------------------------------------------------
# (L) ACCEPTANCE — real calibrations (spec §2.2-L / §6 "the crux")
# ---------------------------------------------------------------------------

def _have_sources(detector: str) -> bool:
    return all((src / "polar_binned_errors.csv").is_file()
               for src in dic.DETECTOR_SOURCES[detector].values())


@pytest.fixture(scope="module")
def real_configs():
    """Derive once per detector with available calibration data."""
    out = {}
    for det in dic.DETECTOR_SOURCES:
        if _have_sources(det):
            out[det] = dic.derive_for_detector(det)
    if not out:
        pytest.skip("no real calibration data available")
    return out


@pytest.mark.skipif(not _have_sources("cp_finetune_100m"),
                    reason="cp_finetune_100m calibration not present")
def test_acceptance_cpft_v2v_immediate_emit(real_configs):
    """V2V (FP-ignore) on the REAL CPft calibration must derive immediate
    emit: m*=1 and confirm_log_odds at-or-below the birth (single-detection)
    log-odds increment."""
    cfg = real_configs["cp_finetune_100m"]["v2v"]
    L = cfg["derived"]["L_confirm"]
    assert L["m_star"] == 1
    assert L["confirm_log_odds"] <= L["p_birth_logit"] + 1e-6
    assert cfg["tracker_params"]["ab3dmot_min_hits"] == 1


def test_acceptance_v2v_immediate_emit_all_detectors(real_configs):
    """Spec §2.2-L acceptance: immediate emit under V2V on ALL detectors
    with available calibrations (pp_score0, cp_zeroshot_100m, CPft)."""
    for det, by_proto in real_configs.items():
        L = by_proto["v2v"]["derived"]["L_confirm"]
        assert L["m_star"] == 1, f"{det}: v2v m*={L['m_star']} (want 1)"
        assert L["confirm_log_odds"] <= L["p_birth_logit"] + 1e-6, det


@pytest.mark.skipif(not _have_sources("cp_finetune_100m"),
                    reason="cp_finetune_100m calibration not present")
def test_acceptance_cpft_ours_near_immediate(real_configs):
    """Ours (FP-counted) on the REAL CPft calibration must STILL be
    near-immediate (empirical anchor: mh1ma5 improved CPft Ours
    37.95 → 39.79). Near-immediate = m* ≤ 2 (within one frame of the
    immediate-emit point)."""
    L = real_configs["cp_finetune_100m"]["ours"]["derived"]["L_confirm"]
    assert L["m_star"] <= 2, f"CPft ours m*={L['m_star']} not near-immediate"


@pytest.mark.xfail(reason="soft expectation: post-NMS-fix pp FPs score far "
                          "below the TP confidence band, so the sweep-"
                          "discounted FP price does not justify a confirm "
                          "delay; strictness for pp shows up in the floor/"
                          "gate instead (reported, not fudged)",
                   strict=False)
def test_acceptance_pp_ours_stricter_than_cpft(real_configs):
    """Soft acceptance: pp_score0 (FP-heavy) should derive a STRICTER
    confirm policy than CPft under the FP-counted protocol."""
    if not (_have_sources("pp_score0") and _have_sources("cp_finetune_100m")):
        pytest.skip("need both pp_score0 and cp_finetune_100m calibrations")
    m_pp = real_configs["pp_score0"]["ours"]["derived"]["L_confirm"]["m_star"]
    m_cpft = real_configs["cp_finetune_100m"]["ours"]["derived"]["L_confirm"]["m_star"]
    assert m_pp > m_cpft, f"pp ours m*={m_pp} vs cpft ours m*={m_cpft}"


def test_real_configs_complete_and_sane(real_configs, tmp_path):
    """Every derived config carries the full §2.1+§2.2 bundle with sane
    values, and round-trips through write_config as valid JSON."""
    import copy
    import json
    for det, by_proto in real_configs.items():
        for proto, cfg in by_proto.items():
            tp = cfg["tracker_params"]
            assert 0.0 <= tp["score_threshold"] <= 1.0
            assert 1 <= tp["ab3dmot_min_hits"] <= dic.CONFIRM_MAX
            assert 1 <= tp["ab3dmot_max_age"] <= dic.COAST_CAP
            assert tp["data_driven_gate_k"] > 0
            assert tp["mahal_gate"] == pytest.approx(9.2103, abs=1e-3)
            assert tp["detector_max_range"] >= 50.0
            assert tp["metric_weight_w"] > 0
            assert cfg["derivation_wall_clock_seconds"] < 30.0
            for bn, bg in cfg["birth_gate"].items():
                assert bg["n_bins"] > 0
                assert "quad_a" in bg["fits"]
            # round-trip (deep-copy: write_config pops the polar tables)
            path = dic.write_config(copy.deepcopy(cfg), tmp_path)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["tracker_params"] == json.loads(
                json.dumps(cfg["tracker_params"]))
            for bn in cfg["birth_gate"]:
                assert (path.parent / f"{bn}_polar.csv").is_file()


def test_real_coast_v2v_at_least_ma5_region(real_configs):
    """(C) sanity from the task brief: coasting is cheap under FP-ignore —
    the v2v coast budget must sit at or above the empirically good ma=5."""
    for det, by_proto in real_configs.items():
        assert by_proto["v2v"]["tracker_params"]["ab3dmot_max_age"] >= 5, det
