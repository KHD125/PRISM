"""
test_momentum.py
================
First contract test for the momentum subsystem (core/scoring_engine momentum components). Pins the
NaN-handling convention — missing inputs score NEUTRAL 50, never a sentinel low — and the ADX/volume
score bands.

Audit finding: the ADX trend sub-signal scored a missing ADX as WEAK (10), not neutral (50), because
np.where collapsed NaN to the final branch BEFORE a now-dead `.fillna(50)`. That broke the
no-sentinel mandate and made ADX the lone NaN-inconsistent momentum input (rs/rsi/volume all
neutral-50 on missing). These tests lock the fix and the convention.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from core.scoring_engine import _compute_trend_score, _compute_rs_score, _compute_volume_score
from config import TREND_SIGNALS, RS_SIGNALS


# ── ADX: missing -> NEUTRAL 50, not weak 10 (the fix) ────────────────────────
def test_adx_nan_scores_neutral_not_weak():
    """Only adx present -> trend_score == adx_score * weight. A missing ADX must score neutral 50
    (RED on the old code, which gives 10)."""
    df = pd.DataFrame({"adx_14w": [np.nan, 30.0, 10.0]})
    t = _compute_trend_score(df)
    w = TREND_SIGNALS["adx_strong"]
    assert t.iloc[0] == pytest.approx(50 * w)    # NaN -> neutral (old: 10*w)
    assert t.iloc[1] == pytest.approx(100 * w)   # 30 -> strong
    assert t.iloc[2] == pytest.approx(10 * w)    # 10 -> weak


def test_adx_band_boundaries():
    """>= boundaries: 25->100, 20->70, 15->40, <15->10."""
    df = pd.DataFrame({"adx_14w": [25.0, 24.9, 20.0, 19.9, 15.0, 14.9]})
    t = _compute_trend_score(df)
    w = TREND_SIGNALS["adx_strong"]
    expected = [100, 70, 70, 40, 40, 10]
    assert [round(v / w) for v in t] == expected


# ── RS: missing -> NEUTRAL 50 (all 3 RS columns present-but-NaN) ──────────────
def test_rs_score_neutral_when_all_crs_nan():
    """_compute_rs_score SKIPS absent columns, so a valid NaN-neutral check must supply ALL THREE
    RS_SIGNALS columns. All-NaN -> each _pct_rank().fillna(50)=50, weights sum to 1 -> rs_score 50."""
    df = pd.DataFrame({c: [np.nan, np.nan, np.nan] for c in RS_SIGNALS})
    rs = _compute_rs_score(df)
    assert np.allclose(rs.values, 50.0)


# ── volume: missing -> NEUTRAL 50; bands ─────────────────────────────────────
def test_volume_score_neutral_on_nan_and_bands():
    assert _compute_volume_score(pd.DataFrame({"vol_ratio": [np.nan]})).iloc[0] == 50.0
    vs = _compute_volume_score(pd.DataFrame({"vol_ratio": [2.0, 1.5, 1.0, 0.7, 0.3]}))
    assert list(vs) == [100.0, 80.0, 60.0, 40.0, 20.0]
