"""
test_quantitative_value_snoa.py
===============================
Contract for the Scaled Net Operating Assets (SNOA) forensic flag added from
Quantitative Value (Gray & Carlisle), Ch.3 — QV's second earnings-manipulation
weapon after STA/accruals. Captures CUMULATIVE balance-sheet accrual build-up.

NOA = Debt + Equity − Cash; SNOA = NOA / lagged Total Assets.
rf_snoa fires when SNOA > 1.0 (NOA exceeds the whole asset base) with an upper
guard < 10 to exclude tiny-denominator data artifacts.

Run with: pytest tests/test_quantitative_value_snoa.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from data_engine import compute_derived_signals
from forensic_engine import compute_forensic_signals
from test_data_quality_fixes import _frame
from config import FORENSIC_MAX_FLAGS


def _snoa_frame(**overrides):
    """Frame with net_worth pinned via reserves (pb path NaN) so SNOA is deterministic."""
    base = dict(reserves=500.0, debt=200.0, cash_equivalents=0.0,
                total_assets=1000.0, total_assets_1yb=900.0, pb_ratio=np.nan)
    base.update(overrides)
    return compute_derived_signals(_frame(**base))


def test_snoa_formula_is_debt_plus_equity_minus_cash_over_lagged_ta():
    """debt 200 + net_worth 500 − cash 0, over lagged TA 900 = 0.778."""
    out = _snoa_frame()
    assert np.allclose(out["scaled_net_operating_assets"], (200.0 + 500.0 - 0.0) / 900.0)


def test_snoa_uses_real_columns_no_proxy():
    """SNOA must move with the real balance-sheet inputs (cash reduces NOA)."""
    low_cash = _snoa_frame(cash_equivalents=0.0)["scaled_net_operating_assets"].iloc[0]
    hi_cash  = _snoa_frame(cash_equivalents=300.0)["scaled_net_operating_assets"].iloc[0]
    assert hi_cash < low_cash, "more cash must lower net operating assets"


def test_rf_snoa_fires_on_bloat():
    """debt 900 + equity 500 − cash 0 over 900 = 1.56 > 1.0 -> flag fires."""
    out = compute_forensic_signals(_snoa_frame(debt=900.0))
    assert (out["rf_snoa"] == 1).all()


def test_rf_snoa_clean_below_threshold():
    """SNOA 0.78 < 1.0 -> flag does not fire."""
    out = compute_forensic_signals(_snoa_frame())
    assert (out["rf_snoa"] == 0).all()


def test_rf_snoa_artifact_guard():
    """Tiny lagged TA -> SNOA >= 10 (data artifact) -> flag suppressed, never fires on noise."""
    out = compute_forensic_signals(_snoa_frame(debt=900.0, total_assets_1yb=50.0))
    # SNOA = 1400/50 = 28 (>=10) -> guarded off
    assert (out["rf_snoa"] == 0).all()


def test_rf_snoa_counted_in_red_flags_and_denominator():
    """rf_snoa is a real rf_ column summed into red_flag_count; FORENSIC_MAX_FLAGS bumped to 28."""
    out = compute_forensic_signals(_snoa_frame(debt=900.0))
    assert "rf_snoa" in out.columns
    assert (out["red_flag_count"] >= 1).all()      # at least the SNOA flag
    assert FORENSIC_MAX_FLAGS == 28
