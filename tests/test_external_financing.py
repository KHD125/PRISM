"""
test_external_financing.py
==========================
Contract for the External Financing to Assets signal — Tortoriello's (Quantitative
Strategies) single strongest factor: firms raising external capital underperform
(bottom quintile −15.3%); firms returning capital outperform (+7.1%).

external_financing_to_assets = financing_cash_flow / total_assets * 100.
SIGN: positive = raising capital (dilutive/leveraging); negative = returning
capital (buybacks/debt-cut/dividends). capital_allocation_signal labels it.

Run with: pytest tests/test_external_financing.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from data_engine import compute_derived_signals
from test_data_quality_fixes import _frame


def _ef(fincf, total_assets=1000.0):
    out = compute_derived_signals(_frame(financing_cash_flow=fincf, total_assets=total_assets))
    return out["external_financing_to_assets"].iloc[0], out["capital_allocation_signal"].iloc[0]


def test_returning_capital_negative_financing():
    """financing CF -100 / TA 1000 = -10% -> Returning Capital (disciplined)."""
    val, sig = _ef(-100.0)
    assert np.isclose(val, -10.0)
    assert "Returning Capital" in sig


def test_raising_capital_positive_financing():
    """financing CF +200 / TA 1000 = +20% -> Raising Capital (risk, Tortoriello short side)."""
    val, sig = _ef(200.0)
    assert np.isclose(val, 20.0)
    assert "Raising Capital" in sig


def test_neutral_band():
    """financing CF 0 -> 0% -> Neutral."""
    _, sig = _ef(0.0)
    assert "Neutral" in sig


def test_sign_convention_inflow_is_raising():
    """Positive financing CF (cash inflow) must classify as RAISING, not returning."""
    _, sig_in = _ef(180.0)
    _, sig_out = _ef(-180.0)
    assert "Raising" in sig_in and "Returning" in sig_out
