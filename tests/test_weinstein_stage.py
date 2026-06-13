"""
test_weinstein_stage.py
=======================
Contract for Weinstein Stage Analysis (Secrets for Profiting in Bull & Bear Markets).
4 stages off the 30-week MA, mapped to a 2×2 of price-vs-30W and 30W-vs-200D:
  Stage 2 Advancing (buy)  : close > 30W AND 30W > 200D
  Stage 1 Basing           : close > 30W AND 30W <= 200D
  Stage 3 Top              : close <= 30W AND 30W > 200D
  Stage 4 Declining (avoid): close <= 30W AND 30W <= 200D

Run with: pytest tests/test_weinstein_stage.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from data_engine import compute_derived_signals
from test_data_quality_fixes import _frame


def _stage(close, sma_30w, sma_200d):
    out = compute_derived_signals(_frame(close_price=close, sma_30w=sma_30w, sma_200d=sma_200d))
    return out["weinstein_stage"].iloc[0]


def test_stage2_advancing():
    """close > 30W > 200D = Stage 2 (Weinstein's buy stage)."""
    assert "Stage 2" in _stage(100.0, 90.0, 80.0)


def test_stage1_basing():
    """close > 30W but 30W <= 200D = Stage 1 (early recovery, crossed up)."""
    assert "Stage 1" in _stage(100.0, 90.0, 95.0)


def test_stage3_top():
    """close <= 30W but 30W > 200D = Stage 3 (early weakness, dropped below MA)."""
    assert "Stage 3" in _stage(80.0, 90.0, 85.0)


def test_stage4_declining():
    """close <= 30W <= 200D = Stage 4 (Weinstein's avoid stage)."""
    assert "Stage 4" in _stage(70.0, 80.0, 90.0)


def test_missing_data_unknown():
    """Missing 30W MA -> Unknown, never a false stage."""
    assert "Unknown" in _stage(100.0, np.nan, 80.0)
