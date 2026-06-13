"""
test_smile_kedia.py
===================
Contract for the faithful Vijay Kedia SMILE framework (corrected 2026-06-13).

SMILE = Small in size, Medium experience, Integrity, Large aspiration, Extra-large
potential (Vijay Kedia's multibagger principle; web-sourced — he wrote no book).
Corrections from the prior mislabeled version:
  - attribution Maheshwari -> Kedia
  - "Small" tightened from <15000 Cr (mid-cap) to <=2000 Cr (Kedia small-cap)
  - added the Integrity dimension via management_integrity_score >= 2

All thresholds are engineering proxies — Kedia states no numeric gates.

Run with: pytest tests/test_smile_kedia.py -v
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
from scoring_engine import run_full_scoring
from test_data_quality_fixes import _frame


def _smile_df(**overrides):
    """Kedia-qualifying small-cap: integrity>=2 via low pledge + high promoter."""
    base = dict(n=30, market_cap=1000.0, pat_gr_5y=30.0, roce=25.0,
                pledged_percentage=0.0, promoter_holdings=60.0)
    base.update(overrides)
    df = _frame(**base)
    df = compute_derived_signals(df)
    df = compute_forensic_signals(df)
    return run_full_scoring(df)


def _passes(df):
    return df["frameworks_passed"].str.contains("SMILE", na=False).any()


def test_smile_fires_for_kedia_smallcap():
    """Small-cap (1000 Cr) + growth + ROCE + integrity -> SMILE fires."""
    assert _passes(_smile_df())


def test_smile_rejects_midcap():
    """mcap > 2000 Cr is NOT Kedia's 'small' — must NOT fire (the core correction)."""
    assert not _passes(_smile_df(market_cap=8000.0))


def test_smile_rejects_illiquid_microcap():
    """mcap < 100 Cr (illiquid micro-cap) is excluded by the floor."""
    assert not _passes(_smile_df(market_cap=50.0))


def test_smile_requires_integrity():
    """Low integrity (high pledge + low promoter -> score < 2) must NOT fire."""
    df = _smile_df(pledged_percentage=40.0, promoter_holdings=20.0)
    # integrity = at most 1 of {clean flags, pledge<5, promoter>=50} -> below the >=2 gate
    assert not _passes(df)


def test_smile_requires_growth_and_roce():
    """Aspiration (growth) and efficiency (ROCE) gates still apply."""
    assert not _passes(_smile_df(pat_gr_5y=5.0))
    assert not _passes(_smile_df(roce=8.0))
