"""
test_regime_detection.py
========================
Contract tests for the market-regime subsystem: core/scoring_engine.detect_market_regime (breadth
consensus) and the regime weight factory config.get_adaptive_weights.

Phase-1 audit found this subsystem entirely UNPINNED and carrying a semantic-truth bug: breadth was
measured with `(s > 0).mean()` over ALL rows, so NaN inputs (~10% of CRS) were silently counted as
"not bull", biasing every vote bearish. These tests pin the 2-of-3 consensus, the NaN-EXCLUDED
breadth (the fix), abstention on missing data, and the weight-cascade invariants (sum==1, no
negative weight) that nothing previously guarded.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from core.scoring_engine import detect_market_regime
from config import MASTER_PROFILES, get_adaptive_weights


def _frac(p, n=100):
    """n-length series: exactly round(p*n) positive (+1.0), the rest negative (-1.0)."""
    k = round(p * n)
    return pd.Series([1.0] * k + [-1.0] * (n - k))


def _px(p, n=100):
    """A (close, sma) pair giving a close>sma fraction of p."""
    k = round(p * n)
    return pd.Series([110.0] * k + [90.0] * (n - k)), pd.Series([100.0] * n)


def _frame(crs50=None, crs26=None, close=None, sma200=None):
    """Build a regime input frame from only the provided signals (omitted = absent column).
    All provided series must share one length (a single market snapshot)."""
    data = {}
    if crs50 is not None:
        data["crs_50d"] = pd.Series(crs50).reset_index(drop=True)
    if crs26 is not None:
        data["crs_26w"] = pd.Series(crs26).reset_index(drop=True)
    if close is not None:
        data["close_price"] = pd.Series(close).reset_index(drop=True)
    if sma200 is not None:
        data["sma_200d"] = pd.Series(sma200).reset_index(drop=True)
    return pd.DataFrame(data)


# ── consensus truth table (full data, no NaN) ────────────────────────────────

def test_all_three_bullish_returns_bull():
    c, m = _px(0.70)
    assert detect_market_regime(_frame(_frac(0.70), _frac(0.70), c, m)) == "BULL"


def test_all_three_bearish_returns_bear():
    c, m = _px(0.30)
    assert detect_market_regime(_frame(_frac(0.30), _frac(0.30), c, m)) == "BEAR"


def test_two_bull_one_bear_returns_bull():
    c, m = _px(0.30)   # 200D bearish, but crs_50d + crs_26w bullish -> 2-of-3 -> BULL
    assert detect_market_regime(_frame(_frac(0.70), _frac(0.70), c, m)) == "BULL"


def test_one_bull_one_bear_one_neutral_returns_sideways():
    c, m = _px(0.30)   # 200D bear ; crs50 bull ; crs26 neutral -> no 2-of-3 -> SIDEWAYS
    assert detect_market_regime(_frame(_frac(0.70), _frac(0.50), c, m)) == "SIDEWAYS"


def test_single_signal_cannot_reach_consensus():
    assert detect_market_regime(_frame(crs50=_frac(0.70))) == "SIDEWAYS"


# ── NaN-bias: breadth must be measured over non-NaN rows ONLY ─────────────────

def test_nan_bias_excluded_from_breadth():
    """RED on the old code: 6/10 and 5/10 INCLUDING NaN read as neutral -> SIDEWAYS. EXCLUDING NaN
    the valid stocks are 100% bullish -> 2 bull votes -> BULL. NaN must never count as 'not bull'."""
    crs50 = [1.0] * 6 + [np.nan] * 4   # excl-NaN: 6/6 = 1.00 bull ; incl-NaN: 6/10 = 0.60 neutral
    crs26 = [1.0] * 5 + [np.nan] * 5   # excl-NaN: 5/5 = 1.00 bull ; incl-NaN: 5/10 = 0.50 neutral
    assert detect_market_regime(_frame(crs50, crs26)) == "BULL"


def test_all_nan_inputs_abstain_to_sideways():
    nan10 = [np.nan] * 10
    assert detect_market_regime(_frame(nan10, nan10, nan10, [100.0] * 10)) == "SIDEWAYS"


# ── 200D vote computed from RAW close/SMA, NaN-excluded (not the biased column) ─

def test_200d_vote_from_raw_close_sma_nan_excluded():
    close = pd.Series([110.0] * 60 + [np.nan] * 40)   # 60 valid, all above SMA -> bull
    sma = pd.Series([100.0] * 60 + [np.nan] * 40)
    # crs_50d bullish too -> 2 bull votes -> BULL
    assert detect_market_regime(_frame(crs50=_frac(0.70), close=close, sma200=sma)) == "BULL"


def test_200d_all_nan_abstains():
    nan100 = [np.nan] * 100
    # only crs_50d votes (bull); 200D abstains, crs_26w absent -> 1 vote -> SIDEWAYS
    assert detect_market_regime(_frame(crs50=_frac(0.70), close=nan100, sma200=nan100)) == "SIDEWAYS"


# ── weight factory invariants (previously unguarded) ─────────────────────────

@pytest.mark.parametrize("profile", sorted(MASTER_PROFILES.keys()))
@pytest.mark.parametrize("regime", ["BULL", "SIDEWAYS", "BEAR"])
def test_adaptive_weights_sum_to_one_and_nonnegative(profile, regime):
    w = get_adaptive_weights(profile, regime)
    four = [w["quality_w"], w["growth_w"], w["longevity_w"], w["price_w"]]
    assert abs(sum(four) - 1.0) < 1e-9, f"{profile}/{regime} weights sum to {sum(four)}"
    assert min(four) >= 0.0, f"{profile}/{regime} has a negative weight: {four}"


def test_unknown_regime_falls_back_to_sideways():
    for profile in sorted(MASTER_PROFILES.keys()):
        w = get_adaptive_weights(profile, "NOT_A_REGIME")
        four = [w["quality_w"], w["growth_w"], w["longevity_w"], w["price_w"]]
        assert abs(sum(four) - 1.0) < 1e-9 and min(four) >= 0.0
        sw = get_adaptive_weights(profile, "SIDEWAYS")
        assert (w["quality_w"], w["growth_w"], w["longevity_w"], w["price_w"]) == \
               (sw["quality_w"], sw["growth_w"], sw["longevity_w"], sw["price_w"])
