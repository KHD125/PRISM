"""
test_entry_timing_calibration.py
================================
Contract for the 2026-06-12 O'Neil/Minervini entry-timing audit fixes.

FIX 1 — vcp_volume_dryup needs Minervini's "dramatic" contraction:
  Old: vol_sma_10d < vol_sma_50d — true ~half the time for any stock (coin flip,
  fired for 61% of the universe). Minervini (Trade Like a Stock Market Wizard,
  converted p.~9389): "volume dries up DRAMATICALLY, accompanied by tightness in
  price"; p.~8854: "volume dries up considerably". New: 10D average must be below
  70% of the 50D average — a real 30%+ contraction, not noise.

FIX 2 — Tsunami quality bar 70 -> 65:
  The 7-condition technical+governance alignment leaves ~12 candidates on live
  data; quality_score >= 70 then kills ALL of them (post-GRUESOME-penalty quality
  median is ~31, so 70 = top ~6% AND perfect alignment simultaneously -> dead
  signal, 0 of 2107 forever). At 65 the signal fires for ~2 elite stocks — the
  'rarest, highest-conviction' design intent, alive.

Run with: pytest tests/test_entry_timing_calibration.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np

from test_data_quality_fixes import _frame
from data_engine import compute_derived_signals
from scoring_engine import detect_catalysts_and_tsunami


# ═══════════════════════════════════════════════════════════════════════════
# FIX 1 — VCP volume dryup materiality
# ═══════════════════════════════════════════════════════════════════════════

def test_vcp_noise_dip_does_not_fire():
    """10D vol at 96% of 50D vol is random fluctuation — Minervini's VCP needs
    volume drying up 'dramatically', not a coin flip."""
    out = compute_derived_signals(_frame(vol_sma_10d=48.0, vol_sma_50d=50.0))
    assert (out["vcp_volume_dryup"] == 0).all()


def test_vcp_dramatic_contraction_fires():
    """10D vol at 60% of 50D vol = genuine supply exhaustion in the base."""
    out = compute_derived_signals(_frame(vol_sma_10d=30.0, vol_sma_50d=50.0))
    assert (out["vcp_volume_dryup"] == 1).all()


def test_vcp_missing_volume_no_dryup():
    out = compute_derived_signals(_frame(vol_sma_10d=np.nan, vol_sma_50d=50.0))
    assert (out["vcp_volume_dryup"] == 0).all()


# ═══════════════════════════════════════════════════════════════════════════
# FIX 2 — Tsunami quality bar
# ═══════════════════════════════════════════════════════════════════════════

def _tsunami_frame(quality: float) -> pd.DataFrame:
    """All 7 technical/governance conditions aligned; quality under test."""
    n = 10
    return pd.DataFrame({
        "gate_pass":        [1] * n,
        "above_sma200":     [1] * n,
        "vstop_green":      [1] * n,
        "vstop_fresh":      [1] * n,
        "promoter_buying":  [1] * n,
        "change_fii_lq":    [0.5] * n,
        "quality_score":    [quality] * n,
        "crs_aligned":      [1] * n,
        "market_cap":       [3000.0] * n,
    })


def test_tsunami_fires_at_quality_66():
    out = detect_catalysts_and_tsunami(_tsunami_frame(66.0))
    assert (out["tsunami_signal"] == 1).all(), (
        "Full 8-way alignment with quality 66 must fire — the old 70 bar made the "
        "signal DEAD (0 of 2107: post-penalty quality median ~31, 70 unreachable "
        "simultaneously with perfect technical alignment)"
    )


def test_tsunami_blocked_below_65():
    out = detect_catalysts_and_tsunami(_tsunami_frame(60.0))
    assert (out["tsunami_signal"] == 0).all(), (
        "Quality 60 must NOT fire — Tsunami stays the rarest, highest-conviction signal"
    )


def test_tsunami_still_requires_all_technicals():
    frame = _tsunami_frame(80.0)
    frame["vstop_green"] = 0
    out = detect_catalysts_and_tsunami(frame)
    assert (out["tsunami_signal"] == 0).all()
