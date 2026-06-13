"""
test_oshaughnessy_value_composite.py
====================================
Contract for the O'Shaughnessy Value Composite (What Works on Wall Street, 4th ed.):
percentile-rank 5 value ratios (P/E, P/B, P/S, EV/EBITDA, P/CF) and average — cheap
(low ratio) = high score (0-100). "Outperforms all single value factors."

CRITICAL: negative ratios (loss-maker, negative book/EBITDA) = distress, must score
WORST (the value-trap guard), not be excluded.

Run with: pytest tests/test_oshaughnessy_value_composite.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from data_engine import compute_derived_signals
from test_data_quality_fixes import _frame


def _vc_frame():
    """3 distinct stocks: cheap / expensive / distressed (negative book & EBITDA)."""
    n = 3
    return compute_derived_signals(_frame(
        n=n,
        pe=[5.0, 50.0, 2.0],                 # cheap / expensive / low-but-distressed
        price_to_book=[1.0, 10.0, -0.5],     # stock 2 negative book
        ps_ratio=[0.5, 5.0, 0.3],
        ev_ebitda=[4.0, 30.0, -20.0],        # stock 2 negative EBITDA
        market_cap=[1000.0, 1000.0, 1000.0],
        operating_cash_flow=[200.0, 50.0, 150.0],
        revenue=[2000.0, 200.0, 3000.0],
    ))


def test_cheap_beats_expensive():
    """The cheap stock (low ratios across the board) scores higher than the expensive one."""
    vc = _vc_frame()["oshaughnessy_value_composite"]
    assert vc.iloc[0] > vc.iloc[1]


def test_distressed_negative_is_penalized_not_rewarded():
    """The distressed stock (negative book + negative EBITDA, despite low PE) must NOT
    outscore the genuinely cheap stock — the value-trap guard."""
    vc = _vc_frame()["oshaughnessy_value_composite"]
    assert vc.iloc[2] < vc.iloc[0], "distressed negative-book stock must not beat the cheap one"


def test_composite_in_range():
    vc = _vc_frame()["oshaughnessy_value_composite"]
    assert vc.dropna().between(0.0, 100.0).all()


def test_requires_min_three_factors():
    """A stock with <3 valid value factors -> NaN composite (not a misleading score)."""
    out = compute_derived_signals(_frame(
        n=3,
        pe=[10.0, np.nan, np.nan], price_to_book=[2.0, np.nan, np.nan],
        ps_ratio=[np.nan, np.nan, np.nan], ev_ebitda=[np.nan, np.nan, np.nan],
        operating_cash_flow=[np.nan, np.nan, np.nan],
    ))
    # rows 1,2 have 0 valid factors -> NaN
    assert out["oshaughnessy_value_composite"].iloc[1] != out["oshaughnessy_value_composite"].iloc[1]  # NaN


# ── Trending Value (O'Shaughnessy flagship: value composite top decile + 6-month momentum) ──

def _trending_frame(crs_cheap):
    """20 stocks; stock 0 is clearly cheapest (top value composite). crs_cheap = its crs_26w."""
    n = 20
    pe = [3.0] + [40.0] * (n - 1)              # stock 0 far cheaper than the rest
    pb = [0.4] + [8.0] * (n - 1)
    ps = [0.2] + [6.0] * (n - 1)
    ev = [3.0] + [25.0] * (n - 1)
    crs = [crs_cheap] + [0.0] * (n - 1)
    return compute_derived_signals(_frame(
        n=n, pe=pe, price_to_book=pb, ps_ratio=ps, ev_ebitda=ev,
        market_cap=[1000.0] * n, operating_cash_flow=[300.0] + [40.0] * (n - 1),
        revenue=[5000.0] + [200.0] * (n - 1), crs_26w=crs,
    ))


def test_trending_value_fires_cheap_and_rising():
    """Cheapest stock (top value composite) WITH positive 6-month momentum -> Trending Value."""
    out = _trending_frame(crs_cheap=15.0)
    assert out["trending_value_flag"].iloc[0] == 1


def test_trending_value_needs_momentum():
    """Same cheap stock but NEGATIVE 6-month momentum -> NOT Trending Value (avoids value trap)."""
    out = _trending_frame(crs_cheap=-15.0)
    assert out["trending_value_flag"].iloc[0] == 0
