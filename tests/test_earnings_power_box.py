"""
test_earnings_power_box.py
==========================
Contract for the Earnings Power Box (Heiserman, "It's Earnings That Count").

Heiserman's trademarked 2×2 crosses two reconstructed income statements:
  • DEFENSIVE  (self-funding)     -> free_cash_flow > 0
  • ENTERPRISING (value creation) -> economic_profit > 0  (Net Income - cost-of-equity charge;
                                     the simplified freefincal Indian-market form)

The distinctive read is the CROSS — value-creation WITH vs WITHOUT self-funding:
  📦 Earnings Power      both positive  (self-funds AND creates value — the safe compounder)
  💰 Cash Cow            FCF>0, EP<=0   (self-funds but earns below cost of equity)
  🚀 Cash-Hungry Grower  FCF<=0, EP>0   (creates value but needs dilutive financing)
  ⚠️ Weakest             both <=0

It is DISPLAY-ONLY: it must never gate, score, or penalize. When either input is missing
the label is blank (semantic truth — no false classification).

net_worth is derived (market_cap / price_to_book, fallback reserves), so economic_profit's
sign is steered here via reserves (net_worth>0) and roe (>COST_OF_EQUITY=12 => EP>0).

Run with: pytest tests/test_earnings_power_box.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from data_engine import compute_derived_signals
from test_data_quality_fixes import _frame


def _epb_frame():
    """5 stocks spanning the four quadrants + one missing-input row.
    reserves=500 (price_to_book NaN) -> net_worth=500>0; roe>12 -> economic_profit>0."""
    return compute_derived_signals(_frame(
        n=5,
        reserves=[500.0] * 5,
        roe=[20.0, 5.0, 20.0, 5.0, 20.0],                 # >12 creates value, <12 destroys
        free_cash_flow=[100.0, 100.0, -50.0, -50.0, np.nan],  # self-funds / burns / unknown
        operating_cash_flow=[np.nan] * 5,                  # leave row 5's FCF genuinely NaN
    ))["earnings_power_box"]


def test_both_positive_is_earnings_power():
    assert _epb_frame().iloc[0] == "📦 Earnings Power"


def test_selffunds_no_value_is_cash_cow():
    assert _epb_frame().iloc[1] == "💰 Cash Cow"


def test_creates_value_burns_cash_is_grower():
    assert _epb_frame().iloc[2] == "🚀 Cash-Hungry Grower"


def test_neither_is_weakest():
    assert _epb_frame().iloc[3] == "⚠️ Weakest"


def test_missing_input_is_blank():
    """No false classification when free_cash_flow is missing."""
    assert _epb_frame().iloc[4] == ""


def test_only_known_labels_emitted():
    allowed = {"📦 Earnings Power", "💰 Cash Cow", "🚀 Cash-Hungry Grower", "⚠️ Weakest", ""}
    assert set(_epb_frame().unique()) <= allowed
