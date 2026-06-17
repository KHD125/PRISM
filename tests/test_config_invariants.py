"""
test_config_invariants.py
=========================
Pins config.py's STRUCTURAL invariants. config.py is the policy layer that drives every score, yet
it had no dedicated test — so a one-character typo (0.20 -> 0.02 in a weight, a reordered tier, a
band gap) would silently re-balance scoring with a fully green suite. These tests lock STRUCTURE
(sums, completeness, monotonicity, ordering, contiguity, the forensic-flag count) — NOT calibration
VALUES — so they need no market data and cannot be "wrong" by a tuning choice. Sibling of the regime
weight-factory pin (test_regime_detection.py) and the conviction descending-min pin
(test_output_consistency.py); deliberately does NOT re-test get_adaptive_weights sum/non-negativity,
which those already cover.
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

import config as C


# ── sub-score weight blends must sum to exactly 1.0 ──────────────────────────
@pytest.mark.parametrize("name", ["QUALITY_WEIGHTS", "MOMENTUM_WEIGHTS"])
def test_subscore_weight_dicts_sum_to_one(name):
    """quality_score / momentum_score are weighted blends of their components; if the component
    weights don't sum to 1.0 the sub-score silently leaves its 0-100 scale."""
    d = getattr(C, name)
    s = sum(v for v in d.values() if isinstance(v, (int, float)))
    assert abs(s - 1.0) < 1e-9, f"{name} component weights sum to {s}, expected 1.0"


# ── MASTER_PROFILES: complete + QGLP base weights normalized ─────────────────
_PROFILE_KEYS = ["quality_w", "growth_w", "longevity_w", "price_w",
                 "roce_gate", "growth_gate", "peg_gate", "forensic_boost", "priority_cols"]


@pytest.mark.parametrize("profile", sorted(C.MASTER_PROFILES))
def test_master_profile_complete_and_normalized(profile):
    """get_adaptive_weights reads all of _PROFILE_KEYS off every profile (a missing key is a runtime
    KeyError), and the QGLP base weights must sum to 1.0 (the regime cascade re-normalizes, but a
    base that doesn't sum to 1 signals an authoring error)."""
    cfg = C.MASTER_PROFILES[profile]
    missing = [k for k in _PROFILE_KEYS if k not in cfg]
    assert not missing, f"{profile} missing keys {missing} -> get_adaptive_weights would KeyError"
    qglp = [cfg["quality_w"], cfg["growth_w"], cfg["longevity_w"], cfg["price_w"]]
    assert abs(sum(qglp) - 1.0) < 1e-9, f"{profile} QGLP base weights sum to {sum(qglp)}, expected 1.0"
    assert all(0.0 <= w <= 1.0 for w in qglp), f"{profile} has a QGLP weight outside [0,1]: {qglp}"


# ── governance risk shield: monotonic, bounded, full pass-through at 0 ────────
def test_governance_risk_multipliers_monotonic_and_bounded():
    """The asymmetric shield multiplies composite DOWN as risk signals accumulate. It must be a
    contiguous 0..n map, start at 1.0 (no penalty when clean), stay in (0,1], and never increase."""
    m = C.GOVERNANCE_RISK_MULTIPLIERS
    keys = sorted(m)
    assert keys == list(range(len(keys))), f"keys must be contiguous 0..n, got {keys}"
    assert m[0] == 1.0, "0 risk signals must be a full 1.0 multiplier (no penalty)"
    vals = [m[k] for k in keys]
    assert all(0.0 < v <= 1.0 for v in vals), f"multipliers must be in (0,1]: {vals}"
    assert all(a >= b for a, b in zip(vals, vals[1:])), f"multipliers must be non-increasing: {vals}"


# ── MCAP_TIERS: min-only descending first-match needs descending order ───────
def test_mcap_tiers_descending_and_reaches_zero():
    """MCAP_TIERS assigns by highest `min` first-match (no `max`), so it MUST be ordered descending
    by min and reach 0 — else a stock lands in the wrong cap tier (or none)."""
    mins = [t["min"] for t in C.MCAP_TIERS.values()]
    assert mins == sorted(mins, reverse=True), f"MCAP_TIERS must be descending by min; got {mins}"
    assert min(mins) == 0, "MCAP_TIERS must reach 0 so every stock lands in a tier"


# ── scoring band zones: contiguous, non-overlapping, full coverage from 0 ────
@pytest.mark.parametrize("name", ["PEG_ZONES", "PAYBACK_ZONES", "RSI_ZONES", "HIGH_AGE_ZONES"])
def test_score_bands_are_contiguous_and_cover_from_zero(name):
    """Each band maps a metric range to a score; gaps or overlaps mean a value falls into no band
    (or two). Verify: starts at 0, every band min<max, and each band's max == the next band's min."""
    d = getattr(C, name)
    bands = sorted((z["min"], z["max"]) for z in d.values())
    assert bands[0][0] == 0, f"{name} must start at 0; first min={bands[0][0]}"
    for (lo, hi), (nlo, nhi) in zip(bands, bands[1:]):
        assert lo < hi, f"{name} band has min>=max: ({lo},{hi})"
        assert hi == nlo, f"{name} gap/overlap: band ends {hi} but next starts {nlo}"
    assert bands[-1][0] < bands[-1][1], f"{name} last band has min>=max: {bands[-1]}"
    for z in d.values():
        assert "score" in z, f"{name} band missing 'score': {z}"


# ── FORENSIC_MAX_FLAGS must equal the real rf_ column count (CLAUDE.md §5 lock) ─
def test_forensic_max_flags_matches_actual_rf_columns():
    """CLAUDE.md §5 locks FORENSIC_MAX_FLAGS to the count of active rf_ columns and previously only
    a MANUAL command verified it. Automate that lock here."""
    path = os.path.join(os.path.dirname(__file__), "..", "core", "forensic_engine.py")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    rf_cols = set(re.findall(r'df\["(rf_[^"]+)"\]\s*=', src))
    assert C.FORENSIC_MAX_FLAGS == len(rf_cols), (
        f"FORENSIC_MAX_FLAGS={C.FORENSIC_MAX_FLAGS} but found {len(rf_cols)} rf_ columns "
        f"in forensic_engine.py: {sorted(rf_cols)}"
    )
