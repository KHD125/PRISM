"""
test_sizing_cockpit.py
======================
Contract for MOD 5: Expected Returns & Kelly-Minervini Sizing (the engine columns
behind the 'Value Creation & Expected Return Identity Cockpit').

Bugs fixed 2026-06-12:
  1. Price <= VSTOP (trend broken / stopped out) previously clipped per-share risk to
     Rs 1, which made the Minervini 1%-risk cap astronomically large and let breakdown
     stocks receive full Kelly weight. Contract: price at/below stop => weight 0.
  2. NaN VSTOP (no stop computable — includes implausible VSTOPs nullified by the
     data_engine scale guard) previously propagated NaN through np.minimum into the
     weight and rupee allocation. Contract: missing stop => Kelly-only weight (no
     technical cap), never NaN.
  3. NaN PE (loss-makers) previously propagated NaN through the re-rating drift into
     expected_cagr_engine. Contract: no PE => zero re-rating term, identity still
     defined as g* + FCF yield - variance drag.

Run with: pytest tests/test_sizing_cockpit.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

import pandas as pd
import numpy as np

from scoring_engine import compute_composite_score


def _frame(**overrides) -> pd.DataFrame:
    """Minimal composite-ready frame: healthy uptrend stock, 12 rows for OLS stability."""
    n = 12
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
    base.update({k: [v] * n if not isinstance(v, (list, np.ndarray)) else v
                 for k, v in overrides.items()})
    return pd.DataFrame(base)


def test_healthy_uptrend_gets_positive_bounded_weight():
    out = compute_composite_score(_frame())
    w = out["optimal_portfolio_weight_pct"]
    assert (w > 0).all(), "Healthy stock above its stop must receive a positive weight"
    assert (w <= 20.0).all(), "Weight must never exceed the 20% concentration cap"
    assert not w.isna().any()


def test_price_below_vstop_means_zero_weight():
    """Trend broken: price below the volatility stop = you would be stopped out at entry.
    Minervini rule: no position. The old clip(lower=1) bug gave these stocks FULL Kelly."""
    out = compute_composite_score(_frame(close_price=400.0, vstop_value=450.0))
    assert (out["optimal_portfolio_weight_pct"] == 0.0).all(), (
        "Price <= VSTOP must produce weight 0 (stopped out), not a Kelly allocation"
    )
    assert (out["rupee_capital_allocation"] == 0.0).all()


def test_price_exactly_at_vstop_means_zero_weight():
    out = compute_composite_score(_frame(close_price=450.0, vstop_value=450.0))
    assert (out["optimal_portfolio_weight_pct"] == 0.0).all()


def test_nan_vstop_falls_back_to_kelly_only():
    """No stop computable (missing or nullified by the data_engine scale guard):
    the technical cap cannot be computed — weight = quarter-Kelly, never NaN."""
    out_nan  = compute_composite_score(_frame(vstop_value=np.nan))
    w = out_nan["optimal_portfolio_weight_pct"]
    assert not w.isna().any(), "NaN VSTOP must not propagate NaN into the weight"
    assert (w > 0).all(), "Kelly-only weight must remain positive for a healthy stock"
    assert not out_nan["rupee_capital_allocation"].isna().any()


def test_tight_stop_caps_below_kelly():
    """A very tight stop (risk ~Rs 2/share on Rs 500 price) limits position FAR below
    Kelly: 1% equity risk / Rs 2 = 5,000 shares = Rs 25L > 10L equity... so instead use
    a wide-risk case: stop far below price => Minervini cap small => binds below Kelly."""
    # Risk = 250/share on a 500 stock: max shares = 10,000/250 = 40 => 40*500/1M = 2%
    out = compute_composite_score(_frame(close_price=500.0, vstop_value=250.0))
    assert (out["optimal_portfolio_weight_pct"] <= 2.0 + 1e-9).all(), (
        "Wide per-share risk must cap the weight at the Minervini 1%-risk limit (2%)"
    )


def test_loss_maker_expected_cagr_not_nan():
    """PE NaN (loss-maker): re-rating term must drop to 0, not poison the identity."""
    out = compute_composite_score(_frame(pe=np.nan, fair_pe_qglp=np.nan))
    e = out["expected_cagr_engine"]
    assert not e.isna().any(), "Expected CAGR identity must survive missing PE"
    # g* 12 + fcf_yield 4 - drag (0.2^2/2*100 = 2) = 14
    assert abs(e.iloc[0] - 14.0) < 1e-9


def test_loss_maker_weight_not_nan():
    """PE NaN + negative valuation residual: the payoff ratio must fall back to the
    neutral 1.5, never NaN — otherwise Kelly and the final weight go NaN.
    pb_ratio spread guarantees some rows have negative OLS residuals."""
    out = compute_composite_score(_frame(pe=np.nan, fair_pe_qglp=np.nan))
    assert not out["payoff_ratio_proxy"].isna().any(), (
        "payoff_ratio_proxy must never be NaN (loss-maker + cheap residual leak)"
    )
    assert not out["optimal_portfolio_weight_pct"].isna().any()
    assert not out["rupee_capital_allocation"].isna().any()
