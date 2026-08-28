"""
test_wealth_tier.py
===================
Contract for the wealth-tier grammar (core/verdict_engine.py — wealth_ep_pct, wealth_vel_pct,
wealth_tier, wealth_warn).

WHAT IT IS. A second, deliberately PRICE-BLIND verdict answering a different question from
verdict_direction: not "is this business good and safe now?" but "is it becoming more valuable?"
Three clocks only:

    A  EP%  = economic_profit / reserves × 100 = ROE − cost of equity    (earning above hurdle)
    B  Vel% = EP velocity / reserves × 100 ≥ 0.5                         (materially improving)
    C  tau  = moat_tau ≥ +0.25                                            (5Y margin spine)
    fading: tau ≤ −0.25 caps everything at WATCH

    BUY★ A·B·C · BUY A·B(flat) · WATCH★ ¬A·B·C (confirmed turnaround) · WATCH one clock /
    momentum broken · AVOID nothing improving · N/A any input unverifiable

PROVENANCE — the grammar was stress-tested BEFORE being built, on ~250 names across three
dry-runs (2026-08-27): the user's watchlists (mostly engine-AVOIDs, where it promoted hidden
improvers like Sarda), the engine's complete 18-stock BUY list (which it split 6/1/0/5/5/1,
demoting decaying legends like Vedant Fashions and Gulf Oil), and the engine's complete 126-stock
WATCH tier (split 36/12/0/43/27/8 — separating Nestle, still compounding, from Colgate at EP%
+70.4 but decaying). The normalization matters: absolute EP is size-biased (it ranked Sarda's
+₹251 Cr above far larger PERCENTAGE improvers); EP/reserves is scale-free and equals ROE − CoE
EXACTLY (one basis — EP is built on reserves. The original build divided by net_worth, a
market_cap ÷ P/B figure on a DIFFERENT equity definition: corr 0.980 but ratio 0.004–14.0 in
the tails, GKW displaying EP% −77.7 against a true −12.1. Fixed 2026-08-28; 8 tiers moved,
none BUY/BUY★).

THE FOUR RULES THESE TESTS DEFEND:
  1. ⚠ NEVER ALTERS THE TIER — the clocks cannot see forensics (16 of 26 BUY★ on one live list
     carried 8+ flags), so risk rides BESIDE the tier, unblended.
  2. STRICT N/A — any missing input yields N/A, never a tier: unverifiable is not passed, and
     equally never condemned. (The Lohia ruling: EP% +24.6 with tau missing stays N/A.)
  3. DISPLAYED-PRECISION CLASSIFICATION — the grammar rounds to what the UI prints (1dp for %,
     2dp for tau) before comparing, so two stocks printing the same number tier the same.
  4. THE FADING CAP BEATS EVERYTHING — improvement without a margin spine stays on probation.

Run with: pytest tests/test_wealth_tier.py -v
"""

import contextlib
import inspect
import io as _io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

import numpy as np
import pandas as pd
import pytest

from core.verdict_engine import (WEALTH_TAU_CONF, WEALTH_TAU_FADE, WEALTH_VEL_MIN,
                                 WEALTH_WARN_FLAGS, compute_verdict)

TIERS = ["BUY★", "BUY", "WATCH★", "WATCH", "AVOID", "N/A"]

_DEFAULTS = {
    "conviction_tier": 1, "composite_score": 90.0,
    "forensic_score": 90.0, "schilit_pass": 1, "red_flag_count": 1,
    "corporate_class": "🏆 GREAT",
    "quality_score": 90.0, "growth_score": 80.0,
    "expected_excess_return": 20.0, "pe": 20.0, "fair_pe_qglp": 30.0,
    "buy_zone_label": "Accumulate", "governance_risk_multiplier": 1.0,
    "data_coverage_pct": 90.0,
    # wealth inputs — defaults make a clean BUY★; each test overrides what it probes.
    # reserves (not net_worth) is the denominator: the SAME equity base EP is built on.
    "reserves": 1000.0, "economic_profit": 100.0,
    "economic_profit_velocity": 50.0, "moat_tau": 1.0,
}


def _tier(**overrides):
    row = {**_DEFAULTS, **overrides}
    out = compute_verdict(pd.DataFrame([row]))
    return out["wealth_tier"].iloc[0]


def _warn(**overrides):
    row = {**_DEFAULTS, **overrides}
    return int(compute_verdict(pd.DataFrame([row]))["wealth_warn"].iloc[0])


# ── 1. The grammar, tier by tier ────────────────────────────────────────────────────────────
def test_all_three_clocks_is_buy_star():
    assert _tier() == "BUY★"                       # EP% +10, Vel% +5, tau +1.00


def test_earning_and_improving_with_flat_tau_is_buy():
    assert _tier(moat_tau=0.0) == "BUY"


def test_confirmed_turnaround_is_watch_star():
    """EP still below the hurdle, but improving with a margin spine — the Entero/Shilpa shelf."""
    assert _tier(economic_profit=-30.0) == "WATCH★"


def test_tau_only_is_watch():
    """The Igarashi case: margin spine, nothing else."""
    assert _tier(economic_profit=-30.0, economic_profit_velocity=-10.0) == "WATCH"


def test_good_business_with_momentum_broken_is_watch():
    """The Jagsonpal case: earning + spine, velocity negative."""
    assert _tier(economic_profit_velocity=-20.0) == "WATCH"


def test_velocity_only_is_watch():
    """Improving, not earning, no spine — one clock."""
    assert _tier(economic_profit=-30.0, moat_tau=0.0) == "WATCH"


def test_level_without_change_is_avoid():
    """The Vedant disease: earning fine, nothing improving, no spine."""
    assert _tier(economic_profit_velocity=0.0, moat_tau=0.0) == "AVOID"


def test_nothing_is_avoid():
    assert _tier(economic_profit=-30.0, economic_profit_velocity=-10.0, moat_tau=0.0) == "AVOID"


# ── 2. The fading cap beats everything ──────────────────────────────────────────────────────
def test_fading_caps_a_full_house_at_watch():
    """Even A·B, margins collapsing → WATCH, never a BUY tier (the Varroc rule)."""
    assert _tier(moat_tau=-0.83) == "WATCH"


def test_fading_without_improvement_is_avoid():
    """The Thermax/Colgate case: earning (even hugely), fading, momentum dead."""
    assert _tier(economic_profit=700.0, economic_profit_velocity=-60.0, moat_tau=-0.83) == "AVOID"


# ── 3. Strict N/A — unverifiable is neither passed nor condemned ────────────────────────────
@pytest.mark.parametrize("hole", [
    {"reserves": np.nan}, {"reserves": 0.0}, {"reserves": -50.0},
    {"economic_profit": np.nan}, {"economic_profit_velocity": np.nan}, {"moat_tau": np.nan},
])
def test_any_missing_input_is_na(hole):
    assert _tier(**hole) == "N/A"


def test_the_lohia_ruling_strong_numbers_with_missing_tau_stay_na():
    """EP% +24.6, Vel% +10.9, tau absent → N/A, deliberately. The row still shows the numbers;
    the TIER refuses to certify on absent evidence."""
    assert _tier(reserves=1000.0, economic_profit=246.0,
                 economic_profit_velocity=109.0, moat_tau=np.nan) == "N/A"


# ── 4. Displayed-precision classification ───────────────────────────────────────────────────
def test_two_stocks_printing_the_same_velocity_tier_the_same():
    """Vel% 0.04 prints +0.0 → below the 0.5 bar; 0.46 prints +0.5 → at it. The comparison runs
    on the printed value, so what you read is what was tiered."""
    below = _tier(economic_profit_velocity=0.4, moat_tau=1.0)     # Vel% = 0.04 → prints 0.0
    at    = _tier(economic_profit_velocity=4.6, moat_tau=1.0)     # Vel% = 0.46 → prints 0.5
    assert below == "WATCH"     # A + C, momentum below the bar
    assert at == "BUY★"


def test_ep_that_prints_zero_is_not_earning():
    """EP% 0.04 prints +0.0 — the reader sees zero, so A must be False."""
    assert _tier(economic_profit=0.4) == "WATCH★"   # ¬A · B · C


# ── 5. ⚠ rides beside the tier, never inside it ────────────────────────────────────────────
def test_warn_fires_on_flags_and_on_schilit():
    assert _warn(red_flag_count=WEALTH_WARN_FLAGS) == 1
    assert _warn(red_flag_count=WEALTH_WARN_FLAGS - 1) == 0
    assert _warn(schilit_pass=0) == 1


def test_warn_never_changes_the_tier():
    """The Krishana rule: 12 flags is a ⚠, not a demotion — blending would hide the tension."""
    assert _tier(red_flag_count=12) == _tier(red_flag_count=0) == "BUY★"
    assert _warn(red_flag_count=12) == 1


# ── 6. Display-only, vectorized, and price-blind ────────────────────────────────────────────
def test_wealth_block_mutates_no_scoring_column():
    frame = pd.DataFrame([_DEFAULTS])
    out = compute_verdict(frame)
    for c in ("composite_score", "conviction_tier", "quality_score", "growth_score"):
        assert float(out[c].iloc[0]) == float(frame[c].iloc[0]), f"{c} was mutated"


def test_wealth_block_is_vectorized():
    """The engine's vectorization mandate — no .apply / iterrows anywhere in compute_verdict."""
    src = inspect.getsource(compute_verdict)
    assert ".apply(" not in src, "compute_verdict grew a row-wise .apply"
    assert "iterrows" not in src, "compute_verdict grew a row loop"


def test_grammar_reads_no_price_column():
    """PRICE-BLIND BY DESIGN: the tier must be computable from the four wealth inputs alone.
    Behavioural proof — removing every price/valuation input must not change the tier."""
    row = {k: v for k, v in _DEFAULTS.items() if k not in ("pe", "fair_pe_qglp", "buy_zone_label")}
    assert compute_verdict(pd.DataFrame([row]))["wealth_tier"].iloc[0] == "BUY★"


# ── 7. Liveness on the real universe ────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def live():
    from data_engine import (coerce_numeric_columns, compute_derived_signals, load_all_csvs,
                             merge_datasets)
    from core import run_scoring_pipeline
    with contextlib.redirect_stdout(_io.StringIO()):
        return run_scoring_pipeline(
            compute_derived_signals(coerce_numeric_columns(merge_datasets(load_all_csvs("local")))))


def test_every_tier_is_alive(live):
    counts = live["wealth_tier"].value_counts()
    for t in TIERS:
        assert counts.get(t, 0) > 0, f"tier {t} fires on zero stocks — dead by the liveness rule"


def test_calibration_bands(live):
    """Shares measured at build time: BUY★ 12% · BUY 5% · WATCH★ 9% · WATCH 32% · AVOID 34% ·
    N/A 8%. Wide bands so a data refresh does not cry wolf; a collapse into one tier fails."""
    share = live["wealth_tier"].value_counts(normalize=True)
    assert 0.04 < share.get("BUY★", 0) < 0.30, f"BUY★ at {share.get('BUY★', 0):.1%}"
    assert share.get("AVOID", 0) < 0.60, "AVOID is swallowing the universe — the engine-verdict disease"
    assert share.get("N/A", 0) < 0.20, "N/A too large — an input's coverage collapsed"


def test_ep_pct_is_exactly_roe_minus_coe(live):
    """THE ONE-BASIS CONTRACT (2026-08-28). EP = reserves × (ROE − CoE) / 100 in data_engine, and
    this engine divides by the SAME reserves base — so EP% ≡ ROE − CoE must hold EXACTLY (fp
    tolerance), for every row where it is defined. This is the drift alarm: if data_engine ever
    changes EP's equity base, or this divisor ever changes, the identity breaks and this fails.
    The original build divided by net_worth (market_cap ÷ P/B — a different equity definition)
    and this identity held only approximately (ratio 0.004–14.0 in the tails)."""
    from config import COST_OF_EQUITY
    known = live["wealth_ep_pct"].notna()
    assert known.sum() > 1500, "EP% coverage collapsed — the identity check has nothing to bite"
    np.testing.assert_allclose(
        live.loc[known, "wealth_ep_pct"].to_numpy(),
        (live.loc[known, "roe"] - COST_OF_EQUITY).to_numpy(),
        rtol=0, atol=1e-6,
        err_msg="wealth_ep_pct no longer equals ROE − CoE — the EP numerator and the wealth "
                "denominator have drifted onto different equity bases",
    )


def test_tiers_mean_what_they_claim_on_live_data(live):
    """Every BUY★ must actually satisfy its own definition — the grammar and the data agree."""
    b = live[live["wealth_tier"] == "BUY★"]
    assert (np.round(b["wealth_ep_pct"], 1) > 0).all()
    assert (np.round(b["wealth_vel_pct"], 1) >= WEALTH_VEL_MIN).all()
    assert (np.round(b["moat_tau"], 2) >= WEALTH_TAU_CONF).all()
    na = live[live["wealth_tier"] == "N/A"]
    assert (na["wealth_ep_pct"].isna() | na["wealth_vel_pct"].isna() | na["moat_tau"].isna()).all()


def test_the_two_verdicts_are_genuinely_different_lenses(live):
    """The point of the whole build: the wealth tier must NOT collapse into verdict_direction.
    Measured at build time, the engine's 18 BUYs split across four wealth tiers."""
    cross = pd.crosstab(live["verdict_direction"], live["wealth_tier"])
    assert (cross.astype(bool).sum(axis=1) >= 2).all(), (
        "every engine verdict maps to a single wealth tier — the lenses have collapsed into one"
    )
