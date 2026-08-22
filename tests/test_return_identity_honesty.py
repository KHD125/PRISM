"""
test_return_identity_honesty.py
===============================
Contract for the honesty pass over the Identity Cockpit's ROW 1 (the three tiles of
`render_valuation_inversion_and_sizing_cockpit`) and the display-only engine column
behind its first tile. Companion to test_sizing_cockpit.py (rows 2+ / Kelly-Minervini)
and test_kelly_sizing_honesty.py (thesis-vs-executable reconciliation).

Audit findings (2026-08-22, live universe = 2,117):

1. `expected_cagr_engine` summed UNCLIPPED fat-tailed proxies and applied a s^2/2
   "variance drag" to `sigma_g` — REVENUE-GROWTH volatility, not return volatility.
   Result: median stock "expected" -13.2%/yr (drag at sigma_g p95=113% is -64%/yr),
   tails from -845% to +300%/yr (Kiri Industries +299.8%, Supreme Infrastructure
   +152%). Contract: the identity is the clean 3-term Bogle/Grinold decomposition —
   growth + cash yield + re-rating — each term clipped to an economically meaningful
   band, no volatility drag until a real return-sigma exists in the data. The column
   is DISPLAY-ONLY (no scoring consumer), so this changes zero scoring numbers.

2. The "OLS Valuation Residual" tile printed a raw ln(P/B) residual ("+0.0132") and
   badged the negative half "Market Underpriced (Alpha)" — OLS residuals are mean-zero
   BY CONSTRUCTION, so the green Alpha label fired for 51.6% of the universe: a coin
   flip presented as insight. Contract: the tile shows the engine's existing bounded
   percentile (`valuation_residual_rank`, MOD 2) as "cheaper than X% of peers", and
   the Alpha wording is gone.

3. The "Decade Moat Trajectory (Tau)" tile overclaimed: `moat_tau` is a 4-point
   OPERATING-MARGIN ladder spanning ~5 years, not returns-on-capital over a decade
   (that description belongs to `roce_tau`). Contract: the tile and its glossary
   entry say what the column actually measures. The tau column itself is untouched —
   it feeds trajectory_score -> Kelly's p.

4. Glossary drift: the cockpit glossary still carried "Recommended Capital Weight"
   with the quarter-Kelly overclaim after the Kelly pass renamed the card to
   "Executable Capital Weight" — the exact module-vs-reference drift class fixed for
   Fisher in commit 7fff308. Contract: glossary keys track the rendered labels.

Run with: pytest tests/test_return_identity_honesty.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

import numpy as np
import pandas as pd

from scoring_engine import compute_composite_score


def _frame(n: int = 12, **overrides) -> pd.DataFrame:
    """Composite-ready frame (mirrors test_sizing_cockpit's harness)."""
    base = {
        "quality_score":    [80.0] * n,
        "momentum_score":   [70.0] * n,
        "governance_bonus": [30.0] * n,
        "pb_ratio":     np.linspace(1.0, 8.0, n),
        "roe":          np.linspace(5.0, 30.0, n),
        "rev_gr_5y":    np.linspace(5.0, 25.0, n),
        "roce_med_5y":  np.linspace(8.0, 28.0, n),
        "market_cap":   np.linspace(500.0, 50000.0, n),
        "pe":               [25.0] * n,
        "fair_pe_qglp":     [35.0] * n,
        "g_star":           [12.0] * n,
        "fcf_yield":        [4.0] * n,
        "sigma_g":          [20.0] * n,
        "trajectory_score": [0.5] * n,
        "close_price":      [500.0] * n,
        "vstop_value":      [450.0] * n,
        "peg":              [1.2] * n,
    }
    for k, v in overrides.items():
        base[k] = [v] * n if not isinstance(v, (list, np.ndarray)) else list(v)
    return pd.DataFrame(base)


# ══════════════════════════════════════════════════════════════════════
# 1. The identity — bounded terms, no growth-volatility drag
# ══════════════════════════════════════════════════════════════════════

def test_distressed_inputs_cannot_fabricate_a_triple_digit_loss_forecast():
    """The live p5 was -116%/yr and the minimum -845%/yr (fcf_yield reaches -902%).
    With each term clipped, the identity floors at -25 -15 -15 = -55%/yr."""
    out = compute_composite_score(_frame(g_star=-330.0, fcf_yield=-900.0,
                                         pe=0.4, fair_pe_qglp=18.0))
    assert (out["expected_cagr_engine"] >= -55.0 - 1e-9).all()


def test_degenerate_inputs_cannot_fabricate_a_triple_digit_gain_forecast():
    """Kiri Industries rendered '+299.8%/yr'. Ceiling: +40 +25 +15 = +80%/yr."""
    out = compute_composite_score(_frame(g_star=100.0, fcf_yield=160.0,
                                         pe=0.4, fair_pe_qglp=18.0))
    assert (out["expected_cagr_engine"] <= 80.0 + 1e-9).all()


def test_growth_volatility_no_longer_drags_the_estimate():
    """sigma_g is REVENUE-GROWTH sigma, not return sigma — a s^2/2 drag on it fabricated
    -64%/yr at the live p95. Two stocks identical except sigma_g must now read the same."""
    calm   = compute_composite_score(_frame(sigma_g=5.0))["expected_cagr_engine"]
    jumpy  = compute_composite_score(_frame(sigma_g=200.0))["expected_cagr_engine"]
    assert np.allclose(calm.to_numpy(), jumpy.to_numpy(), atol=1e-9)


def test_decomposition_terms_sum_exactly_to_the_headline():
    """The tile shows WHY — the three materialized terms must be the whole story."""
    out = compute_composite_score(_frame())
    total = (out["expected_cagr_growth_term"] + out["expected_cagr_yield_term"]
             + out["expected_cagr_rerate_term"])
    assert np.allclose(total.to_numpy(), out["expected_cagr_engine"].to_numpy(), atol=1e-9)


def test_loss_maker_identity_survives_missing_pe():
    """PE NaN (loss-maker): re-rating term drops to 0; identity = g* + FCF yield.
    (Supersedes the old exact-value pin that included the removed drag.)"""
    out = compute_composite_score(_frame(pe=np.nan, fair_pe_qglp=np.nan))
    e = out["expected_cagr_engine"]
    assert not e.isna().any()
    assert abs(e.iloc[0] - 16.0) < 1e-9          # 12 (growth) + 4 (yield) + 0 (re-rating)


# ══════════════════════════════════════════════════════════════════════
# 2-3. The tiles — percentile not raw residual, honest tau window
# ══════════════════════════════════════════════════════════════════════

def _render_row1(stock: pd.Series) -> str:
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
        "name": "Test Co",
        "expected_cagr_engine": 18.0,
        "expected_cagr_growth_term": 12.0,
        "expected_cagr_yield_term": 4.0,
        "expected_cagr_rerate_term": 2.0,
        "moat_tau": 0.4, "valuation_residual": -0.2, "valuation_residual_rank": 18.0,
        "sepa_score": 5, "sepa_pass": 0,
        "optimal_portfolio_weight_pct": 6.0, "rupee_capital_allocation": 60_000.0,
        "vstop_value": 450.0, "close_price": 500.0,
        "mauboussin_ev_verdict": "High Conviction · 8–12% position",
        "value_creation_velocity": 2.0, "expectations_gap": -1.0, "sepa_vcp_dryup": 0,
    }
    base.update(over)
    return pd.Series(base)


def test_valuation_tile_shows_the_percentile_not_the_raw_residual():
    html = _render_row1(_stock())
    assert "Cheaper than 82%" in html
    assert "-0.2000" not in html and "OLS Valuation Residual" not in html


def test_valuation_tile_never_says_alpha_for_a_mean_zero_statistic():
    assert "(Alpha)" not in _render_row1(_stock())


def test_missing_rank_renders_a_blank_not_a_verdict():
    html = _render_row1(_stock(valuation_residual_rank=np.nan))
    assert "Cheaper than" not in html and "No cross-sectional rank" in html


def test_tau_tile_does_not_claim_a_decade_of_moat():
    html = _render_row1(_stock())
    assert "Decade" not in html
    assert "Margin Trend" in html


def test_missing_tau_renders_a_blank_not_a_flat_zero():
    html = _render_row1(_stock(moat_tau=np.nan))
    assert "margin history" in html.lower()


def test_expected_return_tile_shows_its_decomposition():
    html = _render_row1(_stock())
    assert "Re-rating" in html and "Growth" in html


# ══════════════════════════════════════════════════════════════════════
# 4. Glossary tracks the rendered labels (the Fisher-drift class)
# ══════════════════════════════════════════════════════════════════════

def _glossary_source() -> str:
    import io
    path = os.path.join(os.path.dirname(__file__), "..", "ui", "ui_reference_data.py")
    return io.open(path, encoding="utf-8").read()


def test_glossary_has_no_stale_cockpit_keys():
    src = _glossary_source()
    for stale in ("Recommended Capital Weight", "Decade Moat Trajectory",
                  "OLS Valuation Residual", "Expected CAGR Identity"):
        assert stale not in src, f"stale glossary key survived the rename: {stale!r}"


def test_glossary_covers_the_renamed_cockpit_cards():
    src = _glossary_source()
    for key in ("Executable Capital Weight", "Margin Trend", "Price vs Fundamentals",
                "Expected Return Estimate"):
        assert key in src, f"renamed cockpit card missing from glossary: {key!r}"
