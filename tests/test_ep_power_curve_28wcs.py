"""
test_ep_power_curve_28wcs.py
============================
Contract for the Economic Profit family (MOSL 28th Annual Wealth Creation Study, 2018-2023,
"Hockey-Stick Returns: The Power of Economic Profit").

The study's math (its Equations 1 and 3, verbatim):
    Economic Profit = Accounting Profit − Equity Charge
    Equity Charge   = Net Worth × Cost of Equity
    therefore  EP   = Net Worth × (RoE − CoE)          ... Equation 3

Four invariants this file pins, each one a bug that shipped:

1. NEGATIVE EQUITY -> EP IS UNDEFINED.  E × (RoE − CoE) flips sign when E < 0, so the most
   capital-destroyed companies in the universe scored as the study's BEST state (13 of them,
   led by Tata Teleservices: a −215 Cr loss on −29,659 Cr of equity, labelled 🚀 Hockey Stick
   and paid the top-quintile quality bonus). A wiped-out company has no equity base to charge.

2. MISSING RoE -> EP IS UNDEFINED.  `roe.fillna(0)` made EP = −CoE × Equity, manufacturing an
   Economic Loss out of a data hole for 102 stocks (§5 semantic-truth: never inject a sentinel
   into an intermediate ratio).

3. ONE EQUITY BASIS, BOTH YEARS.  net_worth is market_cap ÷ P/B and has NO prior-year
   counterpart (there is no historical price_to_book), so pairing it against reserves_1yb
   subtracted two different definitions of equity. The EP family uses reserves / reserves_1yb —
   the same balance-sheet line in both years — so the velocity is a true delta.

4. MISSING PRIOR-YEAR EQUITY -> VELOCITY IS UNDEFINED, NOT ZERO.  reserves_1yb = 0 made
   EP(t−1) = 0, so velocity collapsed to EP and every profitable such company auto-qualified
   as 🚀 Hockey Stick (19 of them, including Tata Motors and Timken India).

Plus the study's P: `fw_ep_hockey_stick` is the TEM**P** framework, not TEM. §4 of the study
("Engendering Hockey-Stick valuations") gives the number — median HSR entry P/E was 12x and
70% were ≤ 20x — and states the gate completes the framework.

Run with: pytest tests/test_ep_power_curve_28wcs.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import pytest

from config import COST_OF_EQUITY
from data_engine import compute_derived_signals
from scoring_engine import compute_qglp_score
from test_data_quality_fixes import _frame


# ══════════════════════════════════════════════════════════════════════
# 1. Negative equity — EP is undefined, never a profit
# ══════════════════════════════════════════════════════════════════════

def _negative_equity_frame():
    """Equity wiped out. RoE below CoE, so E × (RoE − CoE) is POSITIVE — the sign flip.
    Six rows so the ep_quintile block (which needs >= 5 valid rows) actually runs."""
    return compute_derived_signals(_frame(
        n=6,
        reserves=[-500.0, -500.0, 500.0, 600.0, 700.0, 800.0],
        reserves_1yb=[-430.0, -430.0, 430.0, 500.0, 600.0, 700.0],
        roe=[5.0, -20.0, 20.0, 22.0, 24.0, 26.0],
        roe_1yb=[5.0, -20.0, 20.0, 20.0, 20.0, 20.0],
    ))


def test_negative_equity_gives_undefined_economic_profit():
    assert np.isnan(_negative_equity_frame()["economic_profit"].iloc[0])


def test_negative_equity_never_reports_positive_economic_profit():
    df = _negative_equity_frame()
    assert (df["economic_profit"].iloc[:2] > 0).sum() == 0


def test_negative_equity_is_not_flagged_economic_profit_positive():
    assert int(_negative_equity_frame()["economic_profit_positive"].iloc[0]) == 0


def test_negative_equity_is_not_labelled_a_value_creating_state():
    label = _negative_equity_frame()["ep_power_curve"].iloc[0]
    assert label not in ("🚀 Hockey Stick", "➖ EP Positive, Not Rising")


def test_negative_equity_is_excluded_from_the_ep_quintile_ranking():
    assert np.isnan(_negative_equity_frame()["ep_quintile"].iloc[0])


# ══════════════════════════════════════════════════════════════════════
# 2. Missing RoE — EP is undefined, never a fabricated Economic Loss
# ══════════════════════════════════════════════════════════════════════

def _missing_roe_frame():
    """Six rows so the ep_quintile block (which needs >= 5 valid rows) actually runs."""
    return compute_derived_signals(_frame(
        n=6,
        reserves=[500.0] * 6,
        reserves_1yb=[430.0] * 6,
        roe=[np.nan, 20.0, 22.0, 24.0, 26.0, 28.0],
        roe_1yb=[np.nan, 20.0, 20.0, 20.0, 20.0, 20.0],
    ))


def test_missing_roe_gives_undefined_economic_profit_not_a_loss():
    assert np.isnan(_missing_roe_frame()["economic_profit"].iloc[0])


def test_missing_roe_is_not_labelled_a_value_trap():
    assert _missing_roe_frame()["ep_power_curve"].iloc[0] not in (
        "📉 Value Trap", "🚀 Hockey Stick", "➖ EP Positive, Not Rising", "📈 Improving"
    )


def test_missing_roe_does_not_earn_the_top_quintile_quality_bonus():
    assert int(_missing_roe_frame()["ep_top_quintile_flag"].iloc[0]) == 0


# ══════════════════════════════════════════════════════════════════════
# 3. One equity basis, both years (reserves / reserves_1yb)
# ══════════════════════════════════════════════════════════════════════

def _two_basis_frame():
    """market_cap ÷ P/B = 1000 but reserves = 500. If EP is built on the balance-sheet line,
    EP = 500 × (22 − 12)/100 = 50 — not the 100 that the market_cap ÷ P/B basis would give."""
    return compute_derived_signals(_frame(
        n=1,
        market_cap=[1000.0],
        price_to_book=[1.0],
        reserves=[500.0],
        reserves_1yb=[430.0],
        roe=[22.0],
        roe_1yb=[22.0],
    ))


def test_economic_profit_is_built_on_the_balance_sheet_equity_line():
    expected = 500.0 * (22.0 - COST_OF_EQUITY) / 100.0
    assert _two_basis_frame()["economic_profit"].iloc[0] == pytest.approx(expected)


def test_prior_year_economic_profit_uses_the_same_line():
    expected = 430.0 * (22.0 - COST_OF_EQUITY) / 100.0
    assert _two_basis_frame()["economic_profit_1yb"].iloc[0] == pytest.approx(expected)


def test_velocity_is_a_single_basis_delta():
    """Both terms on reserves: (500 − 430) × 10/100 = 7. The mixed-basis version gave 57."""
    expected = (500.0 - 430.0) * (22.0 - COST_OF_EQUITY) / 100.0
    assert _two_basis_frame()["economic_profit_velocity"].iloc[0] == pytest.approx(expected)


def test_velocity_is_exactly_current_minus_prior():
    df = _two_basis_frame()
    assert df["economic_profit_velocity"].iloc[0] == pytest.approx(
        df["economic_profit"].iloc[0] - df["economic_profit_1yb"].iloc[0]
    )


# ══════════════════════════════════════════════════════════════════════
# 4. Missing prior-year equity — velocity undefined, not zero
# ══════════════════════════════════════════════════════════════════════

def _no_prior_equity_frame():
    """Profitable today, prior-year balance sheet absent (the 56 reserves_1yb ≤ 0 rows)."""
    return compute_derived_signals(_frame(
        n=2,
        reserves=[500.0, 500.0],
        reserves_1yb=[0.0, np.nan],
        roe=[20.0, 20.0],
        roe_1yb=[np.nan, np.nan],
    ))


def test_missing_prior_equity_gives_undefined_velocity():
    assert _no_prior_equity_frame()["economic_profit_velocity"].isna().all()


def test_missing_prior_equity_does_not_manufacture_a_hockey_stick():
    assert int(_no_prior_equity_frame()["ep_hockey_stick"].sum()) == 0


def test_profitable_with_unknown_trend_is_ep_positive_not_a_value_trap():
    assert _no_prior_equity_frame()["ep_power_curve"].iloc[0] == "➖ EP Positive, Not Rising"


# ══════════════════════════════════════════════════════════════════════
# 5. Sign invariance — EP > 0 must mean RoE > CoE (Equation 3 identity)
# ══════════════════════════════════════════════════════════════════════

def test_ep_sign_tracks_the_roe_versus_coe_spread():
    df = compute_derived_signals(_frame(
        n=4,
        reserves=[500.0] * 4,
        reserves_1yb=[430.0] * 4,
        roe=[COST_OF_EQUITY + 5, COST_OF_EQUITY - 5, COST_OF_EQUITY, 40.0],
        roe_1yb=[10.0] * 4,
    ))
    assert list(df["economic_profit"] > 0) == [True, False, False, True]


# ══════════════════════════════════════════════════════════════════════
# 6. eco_profit_improving — no sentinels, hurdle is the config constant
# ══════════════════════════════════════════════════════════════════════

def _improving_frame():
    return compute_derived_signals(_frame(
        n=3,
        reserves=[500.0] * 3,
        reserves_1yb=[430.0] * 3,
        roe=[11.0, 20.0, 20.0],          # 11 clears a hardcoded 10 but not CoE=12
        roe_1yb=[5.0, np.nan, 15.0],     # row 2's prior year is a data hole
    ))


def test_roe_below_cost_of_equity_is_not_improving():
    assert int(_improving_frame()["eco_profit_improving"].iloc[0]) == 0


def test_missing_prior_roe_is_not_treated_as_zero_improvement():
    assert int(_improving_frame()["eco_profit_improving"].iloc[1]) == 0


def test_genuine_roe_improvement_above_the_hurdle_still_fires():
    assert int(_improving_frame()["eco_profit_improving"].iloc[2]) == 1


# ══════════════════════════════════════════════════════════════════════
# 7. The study's P — fw_ep_hockey_stick is TEMP, not TEM
# ══════════════════════════════════════════════════════════════════════

def _framework_frame():
    """Two identical Hockey-Stick fundamentals; only the entry P/E differs."""
    df = pd.DataFrame({
        "company_id":      ["NSE:A", "NSE:B"],
        "name":            ["Cheap Co", "Expensive Co"],
        "ep_hockey_stick": [1, 1],
        "pe":              [12.0, 45.0],
    })
    return compute_qglp_score(df)["frameworks_passed"]


def test_hockey_stick_at_the_study_entry_pe_passes():
    assert "EP Hockey Stick" in _framework_frame().iloc[0]


def test_hockey_stick_above_the_study_pe_ceiling_is_rejected():
    assert "EP Hockey Stick" not in _framework_frame().iloc[1]


# ══════════════════════════════════════════════════════════════════════
# 8. The card must not render a data hole as a verdict
# ══════════════════════════════════════════════════════════════════════

def _render_ep_card(stock: pd.Series) -> str:
    """Render the real EP Power Curve module in-process and return its markdown."""
    from streamlit.testing.v1 import AppTest

    def _app():
        import streamlit as st
        from ui.ui_tearsheet import render_ep_power_curve_module
        render_ep_power_curve_module(st.session_state["stock_row"])

    at = AppTest.from_function(_app)
    at.session_state["stock_row"] = stock
    at.run(timeout=30)
    assert not at.exception, at.exception
    return " ".join(str(md.value) for md in at.markdown)


def _undefined_ep_stock() -> pd.Series:
    """Equity/RoE unreported — economic_profit is genuinely NaN (133 such rows live)."""
    return pd.Series({
        "name": "No Balance Sheet Ltd",
        "economic_profit": np.nan,
        "economic_profit_velocity": np.nan,
        "economic_profit_positive": 0,
        "ep_power_curve": "",
        "ep_quintile": np.nan,
        "ep_hockey_stick_breakout": 0,
    })


def test_undefined_ep_is_not_rendered_as_zero_rupees():
    assert "₹0 Cr" not in _render_ep_card(_undefined_ep_stock())


def test_undefined_ep_is_not_rendered_as_a_value_trap():
    """Checks the VALUE slots, not the whole HTML blob.

    A plain substring scan used to be enough, but the EP Trajectory card now carries a "?"
    tooltip that legitimately names all four curve states ("...📉 Value Trap = negative and not
    improving") to explain the 2x2. That text is documentation, not a verdict about THIS stock.
    What must never regress is the claim the card actually makes — so assert on the rendered
    value elements (font-weight:900 divs), where an undefined EP must read "Not reported".
    """
    import re
    html = _render_ep_card(_undefined_ep_stock())
    values = re.findall(r'font-weight:900;[^>]*>([^<]*)<', html)
    assert values, "no value elements found — the card structure changed"
    assert not any("Value Trap" in v for v in values), (
        f"an undefined economic profit was rendered as a verdict: {values}"
    )
    assert any("Not reported" in v for v in values), f"expected an explicit unknown: {values}"


def test_undefined_ep_says_why_it_is_blank():
    """Assert the specific unknown-state copy — a bare "—" also comes from the velocity tile."""
    assert "Equity or RoE not reported" in _render_ep_card(_undefined_ep_stock())


def _render_all_data(stock: pd.Series) -> str:
    from streamlit.testing.v1 import AppTest

    def _app():
        import streamlit as st
        from ui.ui_tearsheet import render_raw_signals
        render_raw_signals(st.session_state["stock_row"])

    at = AppTest.from_function(_app)
    at.session_state["stock_row"] = stock
    at.run(timeout=60)
    assert not at.exception, at.exception
    return " ".join(str(md.value) for md in at.markdown)


def _all_data_cell(html: str, label: str) -> str:
    """Pull one All-Data cell's rendered value by its label."""
    import re
    mt = re.search(re.escape(label) + r'.*?ts-raw-val">([^<]*)<', html, re.S)
    assert mt, f"cell {label!r} not rendered"
    return mt.group(1).strip()


def test_all_data_shows_undefined_ep_as_na_not_zero_rupees():
    """The engine now leaves 133 rows' EP genuinely NaN — the grid must not print ₹0 Cr."""
    html = _render_all_data(_undefined_ep_stock())
    assert _all_data_cell(html, "Econ Profit") == "N/A"


def test_all_data_shows_undefined_ep_quintile_as_na():
    html = _render_all_data(_undefined_ep_stock())
    assert _all_data_cell(html, "EP Quintile") == "N/A"


def test_quintile_caption_does_not_reuse_the_hockey_stick_name():
    """One card, one meaning per term: the quintile bar's Q2/Q3 caption used to say
    "Hockey-Stick Zone" while a Q1 stock right below it displayed the 🚀 Hockey Stick
    EP-state label — two different concepts (the breakout LAUNCH ZONE vs the EP state)
    wearing the same name read as a contradiction (seen live on Sarda Energy: Q1 +
    🚀 Hockey Stick under a "Q2/Q3 = Hockey-Stick Zone" caption)."""
    html = _render_ep_card(pd.Series({
        "name": "Q1 Compounder Ltd", "economic_profit": 178.0,
        "economic_profit_velocity": 250.0, "economic_profit_positive": 1,
        "ep_power_curve": "🚀 Hockey Stick", "ep_quintile": 1.0,
        "ep_hockey_stick_breakout": 0,
    }))
    assert "Hockey-Stick Zone" not in html
    assert "Launch Zone" in html


# ══════════════════════════════════════════════════════════════════════
# 9. The stage ladder — 📈 EP Improver repurposed as the Q4/Q5 turnaround
# ══════════════════════════════════════════════════════════════════════
# 28th WCS, Exhibit 10 commentary (verbatim): "Upmoves from Quintile 4 and 5 also
# generate handsome returns, albeit they tend to be speculative in nature as they
# involve mainly turnarounds." Exhibit 25: 14 of the 54 Hockey-Stick-Return
# companies (26%) STARTED in Q4/Q5; the completed turnaround earns the matrix's
# best returns (Q4→Q1 = 34%, Q5→Q1 = 29% CAGR, six-period average), with an HSR
# probability of ~7% (vs 18-19% from Q2/Q3). Before this repurpose the EP Improver
# pill was 99% "EP Hockey Stick but too expensive" (181 of its 183 non-HS passers
# differed ONLY on the P/E gate) while the genuine approaching cohort — EP < 0 and
# climbing, 398 stocks — earned no pill at all. The ladder:
#   📈 EP Improver      = APPROACHING (EP < 0, climbing, full internal confirmation)
#   🏒 EP Hockey Stick  = ARRIVED and buyable (EP > 0, climbing, P/E ≤ 20)
# No price gate on the approaching stage — the study's own note: "P/E is not
# meaningful due to accounting loss" (that cohort still returned 27%).

def _approaching_frame(**over):
    """One stock in the canonical approaching state: EP < 0, EP climbing, and all
    three internal confirmations (RoE up, ROCE trend up, margins up) firing."""
    base = dict(
        n=1,
        reserves=[500.0], reserves_1yb=[430.0],
        roe=[10.0], roe_1yb=[6.0],                 # EP −10 vs −25.8 → velocity +15.8
        roce=[12.0], roce_1yb=[9.0], roce_2yb=[8.0],
        opm_med_5y=[8.0], opm_1yb=[9.0], opm=[10.0], opm_latest_q=[11.0],
    )
    base.update(over)
    return compute_derived_signals(_frame(**base))


def test_canonical_turnaround_is_flagged_approaching():
    assert int(_approaching_frame()["ep_approaching_flag"].iloc[0]) == 1


def test_arrived_stock_is_not_approaching():
    """EP already positive belongs to the Hockey-Stick stage, never this one."""
    df = _approaching_frame(roe=[20.0], roe_1yb=[15.0])
    assert df["economic_profit"].iloc[0] > 0
    assert int(df["ep_approaching_flag"].iloc[0]) == 0


def test_deepening_loss_is_not_approaching():
    """EP falling further below zero is a Value Trap, not a turnaround."""
    df = _approaching_frame(roe=[4.0], roe_1yb=[8.0], roce=[6.0])
    assert int(df["ep_approaching_flag"].iloc[0]) == 0


def test_falling_roe_fails_the_returns_confirmation():
    """EP velocity can turn positive from a shrinking equity base alone — RoE
    direction must confirm the RETURNS are actually improving."""
    df = _approaching_frame(reserves=[300.0], roe=[4.8], roe_1yb=[5.0])
    assert df["economic_profit_velocity"].iloc[0] > 0     # velocity alone would pass
    assert int(df["ep_approaching_flag"].iloc[0]) == 0


def test_flat_roce_fails_the_capital_efficiency_confirmation():
    df = _approaching_frame(roce=[8.0], roce_1yb=[9.0], roce_2yb=[9.0])
    assert int(df["ep_approaching_flag"].iloc[0]) == 0


def test_fading_margins_fail_the_margin_confirmation():
    df = _approaching_frame(opm_med_5y=[12.0], opm_1yb=[11.0], opm=[10.0], opm_latest_q=[9.0])
    assert int(df["ep_approaching_flag"].iloc[0]) == 0


def test_unknown_velocity_is_not_approaching():
    """Unverifiable is not passed: no prior-year equity → no velocity → no flag."""
    df = _approaching_frame(reserves_1yb=[np.nan], roe_1yb=[np.nan])
    assert int(df["ep_approaching_flag"].iloc[0]) == 0


def test_the_two_stages_are_mutually_exclusive():
    """EP < 0 here, EP > 0 there — one stock can never hold both stages."""
    df = compute_derived_signals(_frame(
        n=6,
        reserves=[500.0] * 6, reserves_1yb=[430.0] * 6,
        roe=[10.0, 20.0, 8.0, 25.0, 11.0, 30.0],
        roe_1yb=[6.0, 15.0, 10.0, 20.0, 7.0, 22.0],
        roce=[12.0] * 6, roce_1yb=[9.0] * 6, roce_2yb=[8.0] * 6,
        opm_med_5y=[8.0] * 6, opm_1yb=[9.0] * 6, opm=[10.0] * 6, opm_latest_q=[11.0] * 6,
    ))
    assert int(((df["ep_approaching_flag"] == 1) & (df["ep_hockey_stick"] == 1)).sum()) == 0


def test_ep_improver_pill_now_reads_the_approaching_stage():
    """The framework gate reads ep_approaching_flag — and the OLD gate (RoE improving +
    EP positive + ROCE trend) no longer awards the pill: that recipe was 99% 'Hockey
    Stick but too expensive', an end-run around the study's P."""
    df = pd.DataFrame({
        "company_id": ["NSE:A", "NSE:B"],
        "name": ["Turnaround Co", "Expensive Arrived Co"],
        "ep_approaching_flag":      [1, 0],
        "eco_profit_improving":     [0, 1],
        "economic_profit_positive": [0, 1],
        "d35_roce_trend":           [1.0, 5.0],
    })
    pills = compute_qglp_score(df)["frameworks_passed"]
    assert "EP Improver" in pills.iloc[0]
    assert "EP Improver" not in pills.iloc[1]
