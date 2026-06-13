"""
test_result_staleness.py
========================
Contract for the result-staleness data-recency guard (compute_derived_signals).

`days_from_result` in the CSV: negative = days SINCE the last reported result (past),
positive = an upcoming scheduled result. We derive:
  • result_age_days  = -days_from_result   (positive = financials this many days stale)
  • result_stale_flag = 1 when result_age_days > 120  (scored on outdated fundamentals;
    often a distress tell — Gensol Engineering sat 477 days stale before its collapse).

Display-only companion to data_coverage_pct: it must NEVER gate or score, and a MISSING
days_from_result must NOT be asserted as stale (semantic truth).

Run with: pytest tests/test_result_staleness.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from data_engine import compute_derived_signals
from test_data_quality_fixes import _frame


def _stale(days_from_result_vals):
    n = len(days_from_result_vals)
    return compute_derived_signals(_frame(n=n, days_from_result=days_from_result_vals))


def test_fresh_result_not_flagged():
    """Reported 13 days ago → age 13 → not stale."""
    out = _stale([-13.0])
    assert out["result_age_days"].iloc[0] == 13.0
    assert out["result_stale_flag"].iloc[0] == 0


def test_stale_result_flagged():
    """Last result 200 days ago → stale (Gensol-class data-recency risk)."""
    out = _stale([-200.0])
    assert out["result_age_days"].iloc[0] == 200.0
    assert out["result_stale_flag"].iloc[0] == 1


def test_threshold_is_strict_120():
    """Exactly 120 days = NOT stale (>120); 121 days = stale."""
    out = _stale([-120.0, -121.0])
    assert out["result_stale_flag"].iloc[0] == 0
    assert out["result_stale_flag"].iloc[1] == 1


def test_upcoming_result_not_stale():
    """A future scheduled result (positive days_from_result) → negative age → not stale."""
    out = _stale([12.0])
    assert out["result_stale_flag"].iloc[0] == 0


def test_missing_not_asserted_stale():
    """Missing days_from_result must NOT be flagged stale (semantic truth)."""
    out = _stale([np.nan])
    assert out["result_stale_flag"].iloc[0] == 0
