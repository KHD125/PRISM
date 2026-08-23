"""
test_mosl_consistency_substrate.py
==================================
Contract for the MOSL Wealth-Creation SHARED SUBSTRATE — the derived signals that the MOSL
framework gates read. Bugs here propagate to many frameworks at once, which is exactly what
happened: `consistency_champion` feeds `consistent_in_volatile_flag` (+5 composite AND the
🌪️ framework pill), `enduring_vc_flag`, and criterion 7 of `wcs_score`.

── consistency_champion — 27th WCS §2.1, verbatim ──
    "For a company to be deemed a Consistent, it should meet the following 3 criteria –
     1. Over a 15-year period, its annual PAT should not fall by over 10% more than thrice
        (twice if the period is 10 years);
     2. No fall in PAT should be greater than 50%; and
     3. The terminal year PAT should not be lower than the initial year PAT."

The study's Exhibit 2 worked examples pin the reading:
    Company B  4 de-growths, only 3 greater than 10%      -> Consistent
    Company E  only 2 de-growths, but 1 greater than 50%  -> Volatile
    Company G  only 1 de-growth,  but greater than 50%    -> Volatile   <- criterion 2 alone disqualifies

THE BUG THIS FILE PINS: every criterion in the study is expressed as YoY PAT GROWTH % (Exhibit 1),
and the study's universe is the top-500 wealth creators — all profitable. A de-growth % is undefined
on a negative base, so a loss year cannot be scored against criteria 1-2. The engine resolved that by
SKIPPING loss years (`~_both_pos | ...`), which inverted the book: a profit->loss collapse is a >100%
de-growth — Company G's exact disqualifier — yet it became a free pass. On live data 169 of 908
"consistency champions" had a loss year, 94 had two or more; Supreme Infrastructure (four loss years,
-1,426 Cr) and GMR Airports (four) were certified Consistents, and Bombay Burmah ranked 14th.
Criterion 3 had the mirror defect: a missing initial year auto-PASSED (`~_5yb_available | ...`).

Run with: pytest tests/test_mosl_consistency_substrate.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from config import COST_OF_EQUITY
from data_engine import compute_derived_signals
from test_data_quality_fixes import _frame


def _champion(pat_oldest_to_newest, pat_gr_5y: float = 15.0) -> int:
    """Run one 6-point PAT history (oldest -> newest) through the real engine."""
    p5, p4, p3, p2, p1, p0 = pat_oldest_to_newest
    out = compute_derived_signals(_frame(
        n=1,
        pat=[p0], pat_1yb=[p1], pat_2yb=[p2],
        pat_3yb=[p3], pat_4yb=[p4], pat_5yb=[p5],
        pat_gr_5y=[pat_gr_5y], pat_gr_10y=[pat_gr_5y],
    ))
    return int(out["consistency_champion"].iloc[0])


# ══════════════════════════════════════════════════════════════════════
# Criterion 2 — "No fall in PAT should be greater than 50%"
# ══════════════════════════════════════════════════════════════════════

def test_single_fall_greater_than_50pct_disqualifies():
    """Exhibit 2, Company G: only 1 de-growth, but >50% -> Volatile."""
    assert _champion([100, 110, 40, 120, 140, 160]) == 0


def test_profit_to_loss_collapse_disqualifies():
    """A profit -> loss swing is a de-growth of more than 100% — the worst possible
    criterion-2 failure. The engine used to SKIP it because one side was not positive."""
    assert _champion([100, 110, -20, 120, 140, 160]) == 0


def test_sustained_loss_years_disqualify():
    """GMR Airports' real shape: four loss years, then a turnaround. Not a Consistent."""
    assert _champion([-750, -848, -828, -816, 472, 757]) == 0


def test_deepening_loss_disqualifies():
    """Supreme Infrastructure's real shape — losses worsening, then a spike."""
    assert _champion([-919, -1201, -1175, -1426, 5796, 6137]) == 0


def test_current_year_loss_disqualifies():
    assert _champion([100, 110, 120, 130, 140, -10]) == 0


# ══════════════════════════════════════════════════════════════════════
# Criterion 1 — at most one fall >10% across the 5 transitions
# ══════════════════════════════════════════════════════════════════════

def test_one_decline_over_10pct_still_qualifies():
    """110 -> 95 is a 13.6% fall: one de-growth is inside the 3-in-15 / 2-in-10 allowance."""
    assert _champion([100, 110, 95, 120, 140, 160]) == 1


def test_two_declines_over_10pct_disqualify():
    """110 -> 95 and 120 -> 100 are both >10% falls."""
    assert _champion([100, 110, 95, 120, 100, 160]) == 0


def test_shallow_declines_under_10pct_do_not_count():
    """Every step falls, but never by more than 10% — still Consistent per criterion 1."""
    assert _champion([100, 96, 92, 89, 86, 101]) == 1


# ══════════════════════════════════════════════════════════════════════
# Criterion 3 — "terminal year PAT should not be lower than the initial year"
# ══════════════════════════════════════════════════════════════════════

def test_terminal_below_initial_disqualifies():
    assert _champion([200, 190, 185, 180, 175, 170]) == 0


def test_terminal_equal_to_initial_qualifies():
    """The book says 'not be lower' — equality passes."""
    assert _champion([100, 105, 102, 108, 104, 100]) == 1


def test_missing_initial_year_does_not_auto_pass():
    """An unverifiable criterion is not a satisfied one — this used to auto-PASS."""
    assert _champion([np.nan, 110, 120, 130, 140, 160]) == 0


def test_missing_middle_year_does_not_auto_pass():
    assert _champion([100, 110, np.nan, 130, 140, 160]) == 0


# ══════════════════════════════════════════════════════════════════════
# The positive case must survive all of the above
# ══════════════════════════════════════════════════════════════════════

def test_genuine_compounder_still_qualifies():
    assert _champion([100, 110, 125, 140, 160, 180]) == 1


# ══════════════════════════════════════════════════════════════════════
# economic_profit_spread — no fabricated hurdle miss from a data hole
# ══════════════════════════════════════════════════════════════════════

def _spread_frame():
    return compute_derived_signals(_frame(n=2, roce=[np.nan, 25.0]))


def test_missing_roce_leaves_economic_profit_spread_undefined():
    """roce.fillna(0) made this exactly -COST_OF_EQUITY for 84 live rows — a fabricated
    'earns 12pp below its cost of capital' verdict manufactured out of a missing input."""
    assert np.isnan(_spread_frame()["economic_profit_spread"].iloc[0])


def test_reported_roce_still_gives_the_spread():
    assert _spread_frame()["economic_profit_spread"].iloc[1] == 25.0 - COST_OF_EQUITY


# ══════════════════════════════════════════════════════════════════════
# wcs_score (0-10) — previously had ZERO test coverage
# ══════════════════════════════════════════════════════════════════════

def _wcs(**over):
    return compute_derived_signals(_frame(n=1, **over))["wcs_score"].iloc[0]


def test_wcs_score_stays_within_zero_and_ten():
    s = compute_derived_signals(_frame(n=25))["wcs_score"]
    assert s.min() >= 0 and s.max() <= 10


def test_wcs_score_is_zero_when_nothing_is_reported():
    """Every input NaN must score 0 — no criterion may be satisfied by a data hole."""
    assert _wcs(pe=[np.nan], peg=[np.nan], roce_med_5y=[np.nan], roce=[np.nan],
                pat_gr_5y=[np.nan], pb_ratio=[np.nan], price_to_sales=[np.nan],
                market_cap=[np.nan], pat=[np.nan]) == 0


def test_wcs_score_rises_when_a_criterion_is_met():
    """ROCE >= 15 is criterion 5 — turning it on must add exactly one point."""
    base = _wcs(roce_med_5y=[5.0],  roce=[5.0])
    lift = _wcs(roce_med_5y=[25.0], roce=[25.0])
    assert lift == base + 1


# ══════════════════════════════════════════════════════════════════════
# All-Data grid must not render an undefined spread as 0.0%
# ══════════════════════════════════════════════════════════════════════

def test_all_data_shows_undefined_ep_spread_as_na():
    """`g()` defaults NaN to 0, so an unreported spread printed as "0.0%" — a real number."""
    import re
    from streamlit.testing.v1 import AppTest

    def _app():
        import streamlit as st
        from ui.ui_tearsheet import render_raw_signals
        render_raw_signals(st.session_state["stock_row"])

    at = AppTest.from_function(_app)
    at.session_state["stock_row"] = pd.Series({
        "name": "No ROCE Ltd", "economic_profit_spread": np.nan,
    })
    at.run(timeout=60)
    assert not at.exception, at.exception
    html = " ".join(str(md.value) for md in at.markdown)
    mt = re.search(r'EP Spread.*?ts-raw-val">([^<]*)<', html, re.S)
    assert mt and mt.group(1).strip() == "N/A", mt and mt.group(1)


# ══════════════════════════════════════════════════════════════════════
# Moat Endurance Factor (17th WCS) — no verdicts fabricated from data holes
# ══════════════════════════════════════════════════════════════════════
# MEF = current ROCE ÷ 10-yr median ROCE. The old construction filled the NUMERATOR
# with 0 and the no-denominator case with 0.0 — so a stock with UNREPORTED ROCE got
# MEF = 0.0 exactly, landing in the "🔴 Degrading" label band AND the −8-point
# quality band ("ROCE degraded below 80% of its median") — a fabricated penalty for
# ~84 live rows, the mirror image of the EP −12-spread fabrication. Contract: an
# unknown ratio is NaN; NaN earns the NEUTRAL adjustment and a blank label.

def _mef_frame(roce, roce_med_10y):
    return compute_derived_signals(_frame(
        n=len(roce), roce=roce, roce_med_10y=roce_med_10y,
    ))


def test_missing_current_roce_gives_unknown_mef_not_zero():
    df = _mef_frame([np.nan, 20.0], [15.0, 15.0])
    assert np.isnan(df["moat_endurance_factor"].iloc[0])
    assert df["moat_endurance_factor"].iloc[1] == pytest_approx(20.0 / 15.0)


def test_missing_median_gives_unknown_mef():
    df = _mef_frame([20.0], [np.nan])
    assert np.isnan(df["moat_endurance_factor"].iloc[0])


def test_nonpositive_median_gives_unknown_mef():
    df = _mef_frame([20.0], [-5.0])
    assert np.isnan(df["moat_endurance_factor"].iloc[0])


def test_unknown_mef_is_not_labelled_degrading():
    df = _mef_frame([np.nan], [15.0])
    assert df["mef_label"].iloc[0] not in ("🔴 Degrading", "🟡 Eroding", "✅ Intact", "🟢 Expanding")


def test_unknown_mef_earns_the_neutral_quality_adjustment_not_minus_eight():
    """Two stocks identical except MEF: unknown vs the neutral 0.9 band — their moat
    scores must be EQUAL (0.0 adj), while a genuine 0.5 MEF still takes the −8."""
    from core.scoring_engine import _compute_moat_score
    base = pd.DataFrame({
        "moat_endurance_factor": [np.nan, 0.9, 0.5],
        "roce": [20.0] * 3, "roce_med_5y": [20.0] * 3,
    })
    s = _compute_moat_score(base)
    assert s.iloc[0] == pytest_approx(s.iloc[1])
    assert s.iloc[2] == pytest_approx(s.iloc[1] - 8.0)


def pytest_approx(x):
    import pytest as _pt
    return _pt.approx(x)
