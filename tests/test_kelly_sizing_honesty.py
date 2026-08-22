"""
test_kelly_sizing_honesty.py
============================
Contract for the honesty + reconciliation pass over MOD 5 (Kelly-Minervini sizing) and
the two tearsheet panels that display position sizes. Companion to test_sizing_cockpit.py
(which owns the per-stock edge cases); this file owns the PORTFOLIO-LEVEL truths and the
on-screen reconciliation between the system's two sizing engines.

Audit findings (2026-08-22, live universe = 2,117 stocks) pinned here:

1. TWO SIZING ENGINES, ONE TEARSHEET, NO RECONCILIATION. The Mauboussin EV block emits a
   THESIS size ("High Conviction · 8–12% position"); the Kelly/Minervini block emits the
   EXECUTABLE size today. They answer different questions and legitimately diverge — of
   426 "High Conviction" stocks only 14% had a Kelly weight inside 8–12%, and 190 showed
   0.00% (price at/below its volatility stop — a correct "not now"). Rendered side by side
   with no explanation, that read as a contradiction (same failure class as the Fisher
   module-vs-engine split fixed in commit 7fff308). Contract: the cockpit must surface the
   thesis band NEXT TO the executable weight with the REASON for any divergence.

2. WEIGHTS ARE PER-STOCK, NEVER PORTFOLIO-NORMALIZED. `optimal_portfolio_weight_pct` is
   an independent per-row quantity: top-25-by-rank weights summed to 112.7% of capital.
   Contract: the engine performs NO normalization (growing the universe must not change
   any stock's weight), and the UI says so.

3. THE LABEL OVERCLAIMED. "Quarter-Kelly Risk Managed" implied estimated odds; in truth
   p = 0.35 + 0.30 × normalized trajectory (an uncalibrated momentum rescale) and b is
   near-constant (85.5% of the universe sits on the two constants 1.5 and 1.0). Contract:
   the UI copy says "proxy", and the payoff-ratio's continuous branch stays ALIVE so the
   two-constants degeneracy cannot silently become total.

The b-redefinition itself (upside÷downside) is deliberately NOT made here — it raises
every weight ~55% on an uncalibrated p. It is queued, evidence-gated, in
docs/known-issues.md.

Run with: pytest tests/test_kelly_sizing_honesty.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

import numpy as np
import pandas as pd

from scoring_engine import compute_composite_score


# ══════════════════════════════════════════════════════════════════════
# Engine harness — cross-sectional variance for the OLS, MOD 5 inputs set
# ══════════════════════════════════════════════════════════════════════

def _frame(n: int = 12, **overrides) -> pd.DataFrame:
    """Composite-ready frame of n healthy uptrend stocks with cross-sectional variance."""
    base = {
        "quality_score":    [80.0] * n,
        "momentum_score":   [70.0] * n,
        "governance_bonus": [30.0] * n,
        # OLS residual inputs (need cross-sectional variance)
        "pb_ratio":     np.linspace(1.0, 8.0, n),
        "roe":          np.linspace(5.0, 30.0, n),
        "rev_gr_5y":    np.linspace(5.0, 25.0, n),
        "roce_med_5y":  np.linspace(8.0, 28.0, n),
        "market_cap":   np.linspace(500.0, 50000.0, n),
        # MOD 5 inputs
        "pe":               [25.0] * n,
        "fair_pe_qglp":     [35.0] * n,
        "g_star":           [12.0] * n,
        "fcf_yield":        [4.0] * n,
        "sigma_g":          [20.0] * n,
        "trajectory_score": [0.5] * n,
        "close_price":      [500.0] * n,
        "vstop_value":      [450.0] * n,   # healthy: 10% below price
        "peg":              [1.2] * n,
    }
    for k, v in overrides.items():
        base[k] = [v] * n if not isinstance(v, (list, np.ndarray)) else list(v)
    return pd.DataFrame(base)


# ══════════════════════════════════════════════════════════════════════
# 1. Portfolio-sum property — per-stock, no normalization
# ══════════════════════════════════════════════════════════════════════

def test_doubling_the_universe_leaves_each_weight_unchanged():
    """The definitive no-normalization contract: a weight is a per-row quantity, so an
    exact duplication of the universe (which leaves the OLS fit identical) must leave
    every stock's weight identical. Portfolio normalization would halve them."""
    A = _frame(12)
    B = pd.concat([A, A], ignore_index=True)
    wA = compute_composite_score(A)["optimal_portfolio_weight_pct"]
    wB = compute_composite_score(B)["optimal_portfolio_weight_pct"]
    assert np.allclose(wB.iloc[:12].to_numpy(), wA.to_numpy(), atol=1e-9)
    assert np.allclose(wB.iloc[12:].to_numpy(), wA.to_numpy(), atol=1e-9)


def test_weights_of_a_healthy_universe_sum_far_beyond_100pct():
    """24 healthy stocks each get an independent ~5-10% weight — the sum exceeds one
    portfolio. This is the disclosed behaviour; silent normalization would break it."""
    B = pd.concat([_frame(12), _frame(12)], ignore_index=True)
    total = compute_composite_score(B)["optimal_portfolio_weight_pct"].sum()
    assert total > 100.0, f"expected per-stock weights to sum past 100%, got {total:.1f}%"


# ══════════════════════════════════════════════════════════════════════
# 2. Payoff-ratio distribution — bounds, never NaN, continuous branch alive
# ══════════════════════════════════════════════════════════════════════

def test_payoff_ratio_always_bounded_and_never_nan():
    df = compute_composite_score(_frame(
        12,
        pe=[np.nan] * 4 + [25.0] * 8,
        fair_pe_qglp=[np.nan] * 2 + [35.0] * 10,
    ))
    b = df["payoff_ratio_proxy"]
    assert not b.isna().any(), "payoff_ratio_proxy must never be NaN"
    assert (b >= 1.0).all() and (b <= 4.0).all(), "payoff_ratio_proxy must stay in [1.0, 4.0]"


def test_fairly_valued_branch_is_exactly_the_neutral_constant():
    """residual >= 0 -> the documented neutral 1.5 (the branch 53.3% of the live
    universe sits on)."""
    df = compute_composite_score(_frame(12))
    b = df.loc[df["valuation_residual"] >= 0, "payoff_ratio_proxy"]
    assert len(b) >= 3, "harness precondition: need overvalued rows (OLS residuals span 0)"
    assert (b == 1.5).all()


def test_undervalued_branch_stays_continuous():
    """The degeneracy guard: 85.5% of the live universe sits on two constants (1.5, 1.0);
    the ~11% continuous tail is all the discrimination Kelly's b has left. Undervalued
    stocks with different fair-PE gaps must produce DISTINCT payoff ratios — if this
    collapses to constants too, the Kelly weight becomes a pure momentum rescale."""
    df = compute_composite_score(_frame(
        12,
        pe=np.linspace(12.0, 30.0, 12),   # distinct fair/pe gaps, ratios inside (1, 4)
        fair_pe_qglp=[33.0] * 12,
    ))
    under = df[df["valuation_residual"] < 0]
    assert len(under) >= 3, "harness precondition: need undervalued rows (OLS residuals span 0)"
    ratios = under["payoff_ratio_proxy"]
    assert ratios.nunique() >= 3, (
        f"undervalued payoff ratios collapsed to {ratios.nunique()} value(s) — "
        "the continuous branch died and Kelly lost its last real input"
    )
    expected = (under["fair_pe_qglp"] / under["pe"].clip(lower=1.0)).clip(1.0, 4.0)
    assert np.allclose(ratios.to_numpy(), expected.to_numpy(), atol=1e-12)


# ══════════════════════════════════════════════════════════════════════
# 3. UI reconciliation — the cockpit must explain, not contradict
# ══════════════════════════════════════════════════════════════════════

def _render_cockpit(stock: pd.Series) -> str:
    from streamlit.testing.v1 import AppTest

    def _app():
        import streamlit as st
        from ui.ui_tearsheet import render_valuation_inversion_and_sizing_cockpit
        render_valuation_inversion_and_sizing_cockpit(st.session_state["stock_row"])

    at = AppTest.from_function(_app)
    at.session_state["stock_row"] = stock
    at.run(timeout=60)
    assert not at.exception, at.exception
    return " ".join(str(md.value) for md in at.markdown)


def _stock(**over) -> pd.Series:
    base = {
        "name": "Test Co", "expected_cagr_engine": 18.0, "moat_tau": 0.4,
        "valuation_residual": -0.2, "sepa_score": 5, "sepa_pass": 0,
        "optimal_portfolio_weight_pct": 6.0, "rupee_capital_allocation": 60_000.0,
        "vstop_value": 450.0, "close_price": 500.0,
        "mauboussin_ev_verdict": "High Conviction · 8–12% position",
        "value_creation_velocity": 2.0, "expectations_gap": -1.0, "sepa_vcp_dryup": 0,
    }
    base.update(over)
    return pd.Series(base)


def test_cockpit_shows_thesis_band_beside_executable_weight():
    """The Mauboussin band must appear IN the sizing cockpit, subordinated to the
    executable weight — not only in a separate panel where it reads as a contradiction."""
    html = _render_cockpit(_stock())
    assert "8–12%" in html, "EV thesis band missing from the sizing cockpit"


def test_stopped_out_high_conviction_reads_as_not_now_not_contradiction():
    """The 190-of-426 case: thesis says 8-12%, executable weight is 0 because price is
    at/below the volatility stop. The pair must read as 'target — but no entry now'."""
    html = _render_cockpit(_stock(
        optimal_portfolio_weight_pct=0.0, rupee_capital_allocation=0.0,
        close_price=440.0, vstop_value=450.0,
    ))
    assert "8–12%" in html
    assert "no entry now" in html and "volatility stop" in html


def test_zero_weight_without_stop_breach_names_the_kelly_floor():
    """Weight 0 while price is safely above its stop can only mean raw Kelly <= 0."""
    html = _render_cockpit(_stock(
        optimal_portfolio_weight_pct=0.0, rupee_capital_allocation=0.0,
        close_price=500.0, vstop_value=450.0,
    ))
    assert "edge" in html.lower() and "0.00%" in html


def test_insufficient_edge_with_positive_weight_says_thesis_no_position():
    """The mirror case (313 live rows): EV below the 5% book minimum but a positive
    technical weight — the strip must say the thesis is NO position."""
    html = _render_cockpit(_stock(
        mauboussin_ev_verdict="Insufficient Edge · No position (< 5% min)",
        optimal_portfolio_weight_pct=6.0,
    ))
    assert "no position" in html.lower() and "reference" in html.lower()


def test_cockpit_discloses_per_stock_not_normalized():
    html = _render_cockpit(_stock())
    assert "not portfolio-normalized" in html


def test_cockpit_copy_does_not_overclaim_kelly():
    """'Quarter-Kelly Risk Managed' implied estimated odds; the copy must say proxy."""
    html = _render_cockpit(_stock())
    assert "Quarter-Kelly Risk Managed" not in html
    assert "proxy" in html.lower()


def test_mauboussin_tile_subordinates_band_to_executable_weight():
    """At its source too: the EV verdict tile must carry the executable weight so the
    band never stands alone as an instruction."""
    from streamlit.testing.v1 import AppTest

    def _app():
        import streamlit as st
        from ui.ui_tearsheet import render_mauboussin_radar
        render_mauboussin_radar(st.session_state["stock_row"])

    at = AppTest.from_function(_app)
    at.session_state["stock_row"] = _stock(optimal_portfolio_weight_pct=0.0,
                                           close_price=440.0, vstop_value=450.0)
    at.run(timeout=60)
    assert not at.exception, at.exception
    html = " ".join(str(md.value) for md in at.markdown)
    assert "executable" in html.lower() and "0.00%" in html
