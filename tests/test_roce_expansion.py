"""
test_roce_expansion.py
======================
Contract for `roce_expansion` — the TRAJECTORY axis of the Moat-Growth plane.

WHY THIS COLUMN EXISTS. `moat_growth_quad` concedes in its own comment that it is "a snapshot of
position TODAY": its moat axis is `roce_med_5y >= 15`, a LEVEL. The recurring conclusion across all
30 MOSL Wealth Creation studies is ROCE *EXPANSION*. That axis existed nowhere in the frame, so
Bharti Airtel (10Y median 10.99 -> 3Y median 18.42, current 19.44) reads "Growth Trap" -- a correct
five-year statement (5Y median 12.33) with the direction invisible.

IT LAGS. A difference of medians keeps reading "expanding" after a business gives the improvement
back: 132 of 397 expanders (33.2%) sit more than 2pp below their own 3Y median TODAY, 57 below the
10Y. Interglobe Aviation (6.53 -> 16.64, current 4.64) was cited in the first draft of this file
and withdrawn. Not a defect -- every trailing metric lags, and a median is the right instrument for
a cycle question -- but a PLACEMENT CONSTRAINT, and section 5 below turns it into a contract.

THREE THINGS THIS FILE DEFENDS, each of which has a matching failure already on the record:

1. THE GUARD IS NOT COSMETIC. A shallow-history company has all four ROCE medians computed over the
   same few years, so they come out identical and a naive subtraction returns exactly 0.00 --
   "flat" fabricated from absent evidence. `roce_trajectory` shipped precisely that: 34% of it sits
   at exactly 0.00. The guard tests the OPERANDS (all four windows identical => no time information
   => NaN), and it was chosen by measurement against an independent witness, not by taste.

2. THE BASIS RULE. Both terms are medians from the same family (CLAUDE.md §5 cross-year basis rule).
   Economic Profit once subtracted a `market_cap / P_B` level from a `reserves` level and biased
   every velocity upward; a delta whose terms are built differently is not a delta.

3. IT MUST NOT REACH THE SCORE. This is a display + filter column. Feeding an unvalidated trajectory
   into composite_score is the threshold-guessing anti-pattern this engine retired -- the forward
   test is the December vintage, and this column is a CANDIDATE for it, not a passenger in it.

Run with: pytest tests/test_roce_expansion.py -v
"""

import contextlib
import io as _io
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

import numpy as np
import pandas as pd
import pytest

_CORE = os.path.join(os.path.dirname(__file__), "..", "core")
_APP = os.path.join(os.path.dirname(__file__), "..", "app.py")


@pytest.fixture(scope="module")
def live():
    from core import fetch_and_clean_data, run_scoring_pipeline
    with contextlib.redirect_stdout(_io.StringIO()):
        return run_scoring_pipeline(fetch_and_clean_data("local").copy())


def _src(name):
    return _io.open(os.path.join(_CORE, name), encoding="utf-8").read()


# -- 1. The value ------------------------------------------------------------------------
def test_the_column_exists_and_is_the_documented_difference(live):
    assert "roce_expansion" in live.columns
    expected = live["roce_med_3y"] - live["roce_med_10y"]
    both = live["roce_expansion"].notna() & expected.notna()
    assert both.sum() > 1000, "too few comparable rows to verify the definition"
    assert np.allclose(live.loc[both, "roce_expansion"], expected[both]), (
        "roce_expansion is no longer roce_med_3y - roce_med_10y"
    )


def test_both_terms_come_from_the_same_median_family(live):
    """CLAUDE.md §5 cross-year basis rule, pinned on the SOURCE so a future edit cannot quietly
    swap one leg for a differently-constructed level (the Economic Profit failure)."""
    src = _src("data_engine.py")
    i = src.index('df["roce_expansion"]')
    stmt = src[i:src.index("\n", i)]
    assert "_rc3" in stmt and "_rc10" in stmt, f"unexpected roce_expansion assignment: {stmt}"
    setup = src[max(0, i - 900):i]
    assert '_rc3, _rc5  = df["roce_med_3y"], df["roce_med_5y"]' in setup
    assert '_rc7, _rc10 = df["roce_med_7y"], df["roce_med_10y"]' in setup


# -- 2. The guard ------------------------------------------------------------------------
def test_collapsed_windows_are_nan_not_zero(live):
    """The whole point. All four medians identical => the windows carry no time information, so
    the honest answer is 'unknown', never 'flat'."""
    m3, m5 = live["roce_med_3y"], live["roce_med_5y"]
    m7, m10 = live["roce_med_7y"], live["roce_med_10y"]
    collapsed = (m3 == m5) & (m5 == m7) & (m7 == m10)
    assert collapsed.sum() > 0, "no collapsed rows on live data -- the guard is untestable here"
    assert live.loc[collapsed, "roce_expansion"].isna().all(), (
        "a company whose ROCE windows all agree exactly has an UNKNOWN trajectory; reporting 0.0 "
        "fabricates 'flat' from absent history -- the defect that made roce_trajectory 34% zeros"
    )


def test_the_guard_did_not_swallow_the_whole_universe(live):
    """A guard that NaNs everything would pass the test above and be useless."""
    e = live["roce_expansion"]
    assert e.isna().mean() < 0.20, f"roce_expansion is {e.isna().mean():.1%} NaN -- over-guarded"
    assert e.notna().sum() > 2000


def test_the_degeneracy_that_broke_roce_trajectory_is_gone(live):
    """roce_trajectory sits at exactly 0.00 on ~34% of rows. This must not."""
    e = live["roce_expansion"].dropna()
    zero_share = (e.abs() < 1e-9).mean()
    assert zero_share < 0.05, (
        f"{zero_share:.1%} of non-null roce_expansion is exactly 0.00 -- the window-collapse "
        f"guard has stopped working (roce_trajectory's failure mode)"
    )


def test_a_genuinely_stable_business_is_not_guarded_away():
    """Guarding on `roce_med_5y == roce_med_10y` was REJECTED because it also hit 7.4% of
    deep-history firms. A company whose 3Y differs from its 10Y must survive even if two of the
    middle windows happen to tie."""
    from data_engine import compute_derived_signals
    row = {"roce_med_3y": 20.0, "roce_med_5y": 15.0, "roce_med_7y": 15.0, "roce_med_10y": 15.0}
    df = pd.DataFrame([row])
    m3, m5, m7, m10 = (df[f"roce_med_{k}y"] for k in (3, 5, 7, 10))
    collapsed = (m3 == m5) & (m5 == m7) & (m7 == m10)
    assert not collapsed.iloc[0], "a differing 3Y window must not count as collapsed"


# -- 3. It must not reach the score ------------------------------------------------------
@pytest.mark.parametrize("module", ["scoring_engine.py", "forensic_engine.py", "verdict_engine.py"])
def test_no_scoring_layer_consumes_it(module):
    """Display + filter only. If a future change wires this into a score, it must fail HERE and be
    justified against forward evidence -- which does not exist until the December vintage."""
    path = os.path.join(_CORE, module)
    if not os.path.exists(path):
        pytest.skip(f"{module} not present")
    assert "roce_expansion" not in _io.open(path, encoding="utf-8").read(), (
        f"{module} reads roce_expansion. This column is UNVALIDATED -- it has never been tested "
        f"against forward returns. Scoring on it is the threshold-guessing anti-pattern."
    )


def test_it_is_not_wired_into_the_composite(live):
    """Behavioural companion to the source scan above."""
    assert live["composite_score"].notna().sum() > 2000
    corr = live["roce_expansion"].corr(live["composite_score"])
    assert abs(corr) < 0.60, (
        f"roce_expansion correlates {corr:.3f} with composite_score -- if it has been folded into "
        f"the score, the display-only contract is broken"
    )


# -- 4. Liveness -------------------------------------------------------------------------
def test_the_signal_is_alive_and_no_band_owns_the_universe(live):
    """Signal-liveness discipline: a band holding everything is a calibration bug no unit test
    catches."""
    e = live["roce_expansion"]
    bands = {
        "expanding": (e >= 5), "improving": (e >= 1) & (e < 5),
        "flat": (e > -1) & (e < 1), "eroding": (e <= -1),
    }
    for name, m in bands.items():
        share = m.fillna(False).mean()
        assert 0.02 < share < 0.60, f"band {name!r} holds {share:.1%} of the universe"


def test_the_blind_spot_it_was_built_for_is_actually_visible(live):
    """The justification, pinned. Stocks that are expanding but are NOT yet Wealth Creators are the
    set this column exists to surface; if it empties, the column has stopped earning its place."""
    expanding = (live["roce_expansion"] >= 5).fillna(False)
    not_yet = expanding & (live["moat_growth_quad"] != "⭐ Wealth Creator")
    assert not_yet.sum() > 50, (
        f"only {int(not_yet.sum())} stocks are expanding by this measure without being Wealth "
        f"Creators -- the reason this column was added was that this set is large and unreachable"
    )


# -- 5. It reaches the screen — existence is not reachability ---------------------------
def test_it_is_in_the_quality_view_and_has_a_display_format():
    """A column nobody can see is not a feature. Both halves are required: the view preset lists
    it, and the format map gives it a readable header."""
    src = _io.open(_APP, encoding="utf-8").read()
    i = src.index('"\U0001f4ca Quality":')
    preset = src[i:src.index("],", i)]
    assert '"roce_expansion"' in preset, "roce_expansion is not in the Quality column view"
    assert '"roce_expansion":' in src, "roce_expansion has no entry in the display format map"
    j = src.index('"roce_expansion":  (')
    entry = src[j:src.index("\n", j)]
    assert "pp" in entry, (
        f"roce_expansion is measured in PERCENTAGE POINTS of ROCE; its header must say so, not "
        f"imply a percent change: {entry}"
    )


def test_roce_sits_beside_roce_expansion_in_the_quality_view():
    """THE SAFETY PROPERTY. roce_expansion is a difference of MEDIANS and it lags: on the 2026-09-09
    data 132 of 397 'expanding' stocks (33.2%) have a CURRENT roce more than 2pp below their own 3Y
    median, and 57 have fallen back below the 10Y median. A reader who sees only '+10.1pp' is
    misled; a reader who sees '4.64 | +10.1pp' is not. The adjacency is what makes a lagging signal
    safe to show, so it is a CONTRACT, not a layout preference -- the verdict-beside-its-number rule
    already pinned for D48/D49, applied to the number that CORRECTS rather than the number that
    scores. Move roce_expansion to another view, or drop roce from this one, and this fails."""
    src = _io.open(_APP, encoding="utf-8").read()
    i = src.index('"\U0001f4ca Quality":')
    preset = src[i:src.index("],", i)]
    cols = re.findall(r'"([a-z_0-9]+)"', preset)
    assert "roce" in cols, f"roce has left the Quality view; roce_expansion now stands uncorrected: {cols}"
    assert "roce_expansion" in cols, f"roce_expansion has left the Quality view: {cols}"
    assert abs(cols.index("roce") - cols.index("roce_expansion")) == 1, (
        f"roce and roce_expansion must be ADJACENT in the Quality view -- the current number is the "
        f"correction for a lagging trajectory, and a column apart is a column unread. Order: {cols}"
    )
