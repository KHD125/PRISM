"""Contract: the DISPLAY-ONLY per-stock trend modifiers (compute_derived_signals).

Enriches weinstein_stage (direction) + d45_trend_structure (strength) with book-faithful,
confidence-graded path chips: ↩️ Pullback (HIGH edge), 🚀 Breakout (with-trend edge),
⚠️ Bounce / ⚠️ Extended (LOW-confidence cautions). All display-only, NaN-safe, never scored.

Volume: dry-up uses the CLEAN vol_sma ladder (vol_sma_10d<50d); breakout uses raw Volume vs 50D
(book-faithful — under-fires until the source Volume deflation is fixed, by design, no workaround).
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

import numpy as np
import pandas as pd

from data_engine import (compute_derived_signals, COMMON_COLS, RATIO_COLS, INCOME_COLS,
                         BALANCE_COLS, CASHFLOW_COLS, SHAREHOLDING_COLS, TECHNICAL_COLS)

_ALL = set()
for _m in (COMMON_COLS, RATIO_COLS, INCOME_COLS, BALANCE_COLS,
           CASHFLOW_COLS, SHAREHOLDING_COLS, TECHNICAL_COLS):
    _ALL.update(_m.values())


def _frame(**overrides) -> pd.DataFrame:
    """Full-NaN mapped frame (the §6 synthetic pattern) with single-row overrides."""
    base = {"company_id": ["NSE:T"], "name": ["T"]}
    for k, v in overrides.items():
        base[k] = [v]
    df = pd.DataFrame(base)
    missing = sorted(c for c in _ALL if c not in df.columns)
    if missing:
        df = pd.concat([df, pd.DataFrame(np.nan, index=df.index, columns=missing)], axis=1)
    return df


def _run(**ov):
    return compute_derived_signals(_frame(**ov)).iloc[0]


# ── each modifier fires on its book-faithful condition ────────────────────────
def test_pullback_stage2_dip_below_50d_above_30w_on_dryup():
    # Stage 2 (close 95 > 30W 90 > 200D 80); dipped below 50D (100); volume dry-up (10D<50D)
    r = _run(close_price=95.0, sma_30w=90.0, sma_200d=80.0, sma_50d=100.0,
             vol_sma_10d=40.0, vol_sma_50d=50.0)
    assert r["trend_pullback"] == 1
    assert r["trend_modifier"] == "↩️ Pullback"


def test_pullback_does_not_fire_without_dryup():
    r = _run(close_price=95.0, sma_30w=90.0, sma_200d=80.0, sma_50d=100.0,
             vol_sma_10d=60.0, vol_sma_50d=50.0)   # 10D > 50D = NO dry-up
    assert r["trend_pullback"] == 0


def test_breakout_near_52wh_on_volume_expansion_not_extended():
    # Stage 2; within 3% of 52WH; not extended (100 <= 90*1.35); raw vol 100 >= 1.5*50
    r = _run(close_price=100.0, sma_30w=90.0, sma_200d=80.0, dist_52wh=2.0,
             volume=100.0, vol_sma_50d=50.0)
    assert r["trend_breakout"] == 1
    assert r["trend_modifier"] == "🚀 Breakout"


def test_breakout_needs_volume_expansion():
    r = _run(close_price=100.0, sma_30w=90.0, sma_200d=80.0, dist_52wh=2.0,
             volume=40.0, vol_sma_50d=50.0)   # 40 < 1.5*50 = no expansion
    assert r["trend_breakout"] == 0


def test_bounce_stage4_recent_rally_to_declining_30w():
    # Stage 4 (close 88 <= 30W 90, 30W 90 <= 200D 100); within 5% below the MA; recent rally (+3M)
    r = _run(close_price=88.0, sma_30w=90.0, sma_200d=100.0, ret_vs_n500_3m=5.0)
    assert r["trend_bounce"] == 1
    assert r["trend_modifier"] == "⚠️ Bounce"


def test_bounce_does_not_fire_on_a_drifter_without_a_rally():
    # Same Stage-4 position but NO recent rally (3M ≤ 0) → not a bounce, just a drifter
    r = _run(close_price=88.0, sma_30w=90.0, sma_200d=100.0, ret_vs_n500_3m=-4.0)
    assert r["trend_bounce"] == 0


def test_extended_overbought_is_caution_and_outranks_breakout():
    # >1.40x the 30W MA AND RSI>70 — and even if it's also near a high, caution wins
    r = _run(close_price=130.0, sma_30w=90.0, sma_200d=80.0, rsi_14d=75.0, dist_52wh=1.0,
             volume=100.0, vol_sma_50d=50.0)
    assert r["trend_extended"] == 1
    assert r["trend_modifier"] == "⚠️ Extended"   # caution-first priority


def test_plain_stage2_has_no_modifier():
    r = _run(close_price=95.0, sma_30w=90.0, sma_200d=80.0)
    assert r[["trend_pullback", "trend_breakout", "trend_bounce", "trend_extended"]].sum() == 0
    assert r["trend_modifier"] == ""


def test_all_nan_technicals_fire_nothing_and_do_not_crash():
    r = _run()   # everything NaN → weinstein "❔ Unknown" → no flag
    assert r[["trend_pullback", "trend_breakout", "trend_bounce", "trend_extended"]].sum() == 0
    assert r["trend_modifier"] == ""
