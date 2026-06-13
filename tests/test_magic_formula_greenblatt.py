"""
test_magic_formula_greenblatt.py
================================
Contract for the Greenblatt Magic Formula earnings-yield fix (2026-06-13).

Book audit (The Little Book That Still Beats the Market, appendix): the Magic
Formula's earnings yield is EBIT / Enterprise Value — NOT net-income/price
(= 100/PE). Greenblatt insists on EBIT/EV to neutralize capital-structure and
tax-rate differences. The framework was using the wrong (net-income) yield;
this locks the book-exact EBIT/EV computation and its use in fw_magic_formula.

Run with: pytest tests/test_magic_formula_greenblatt.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

import numpy as np
import pandas as pd

from data_engine import compute_derived_signals
# Reuse the all-columns-materialized synthetic frame from the data-quality suite.
sys.path.insert(0, os.path.dirname(__file__))
from test_data_quality_fixes import _frame


def test_magic_formula_yield_is_ebit_over_ev():
    """EBIT/EV exact: EBIT=100, EBITDA=140, EV/EBITDA=10 -> EV=1400 -> 100/1400 = 7.14%."""
    out = compute_derived_signals(_frame(ebit=100.0, ebitda=140.0, ev_ebitda=10.0))
    assert np.allclose(out["magic_formula_earnings_yield"], 100.0 / 1400.0 * 100.0)


def test_magic_formula_yield_differs_from_net_income_yield():
    """The book-exact EBIT/EV metric must be a DISTINCT column from net-income
    earnings_yield (=100/PE) — proving the framework no longer uses the wrong one."""
    out = compute_derived_signals(_frame(ebit=100.0, ebitda=140.0, ev_ebitda=10.0, pe=25.0))
    gb = out["magic_formula_earnings_yield"].iloc[0]      # 7.14%
    ni = out["earnings_yield"].iloc[0]                    # 100/25 = 4.0%
    assert abs(gb - ni) > 1.0, "EBIT/EV and net-income yields should genuinely differ"


def test_magic_formula_yield_guards_nonpositive_ev():
    """EV <= 0 (negative ev_ebitda or zero ebitda) -> NaN, never a divide blow-up."""
    out = compute_derived_signals(_frame(ebit=100.0, ebitda=0.0, ev_ebitda=10.0))
    assert out["magic_formula_earnings_yield"].isna().all()


def _magic_df(**overrides):
    """Build a scored frame of qualifying Magic Formula stocks, with optional overrides."""
    from scoring_engine import run_full_scoring
    from forensic_engine import compute_forensic_signals
    base = dict(n=30, ebit=100.0, ebitda=140.0, ev_ebitda=8.0, roce=25.0, pe=22.0,
                market_cap=3000.0)
    base.update(overrides)
    df = _frame(**base)
    df = compute_derived_signals(df)
    df = compute_forensic_signals(df)
    return run_full_scoring(df)


def test_magic_formula_framework_consumes_ebit_ev_yield():
    """A stock cheap on EBIT/EV (>=8%) with ROCE>=20 passes the Magic Formula even
    when its net-income yield would be lower — the whole point of the fix."""
    df = _magic_df()
    assert df["frameworks_passed"].str.contains("Magic Formula", na=False).any()


def test_magic_formula_excludes_financials():
    """Greenblatt Step: eliminate financial stocks — even a qualifying financial must not pass."""
    df = _magic_df(sector="Finance", industry="Finance")
    assert not df["frameworks_passed"].str.contains("Magic Formula", na=False).any()


def test_magic_formula_excludes_utilities():
    """Greenblatt Step: eliminate utilities — a qualifying gas/power distributor must not pass."""
    df = _magic_df(sector="Gas Distribution", industry="Gas Distribution")
    assert not df["frameworks_passed"].str.contains("Magic Formula", na=False).any()


def test_magic_formula_applies_mcap_floor():
    """Greenblatt size floor — a qualifying but sub-₹500 Cr micro-cap must not pass."""
    df = _magic_df(market_cap=200.0)
    assert not df["frameworks_passed"].str.contains("Magic Formula", na=False).any()
