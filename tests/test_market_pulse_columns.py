"""
test_market_pulse_columns.py
============================
Contract for the Market Pulse tables' column ORDER, and for the one tooltip claim on that tab that
measurement contradicted.

THE DEFECT (2026-08-27, found by reading screenshots of the live tab). Each table defines more
columns than fit its container:

    Tsunami   12 columns, ~8 visible
    QGLP      13 columns, ~8 visible
    Sectors    7 + index, last value clipped mid-digit

`st.dataframe` scrolls, so nothing was unreachable — but `sector` is the widest column in the frame
("Infrastructure Developers & Operators", "Capital Goods - Electrical Equipment") and it sat at
position 5 on the QGLP tab. It pushed `qglp_price` — the "P" in QGLP — off-screen entirely, so a
tab showcasing a four-leg framework displayed one and a half legs. On Sectors, `avg_composite`
rendered as a bar plus a single truncated digit, which reads as broken rather than scrollable.

THE FIX IS ORDER, NOT DELETION. Every column survives and the tables still scroll; the columns
that make each tab's point simply come before the wide context ones. Deleting columns would have
destroyed information to solve a layout problem.

WHY THESE TESTS PIN A PROPERTY, NOT A LIST. Asserting the exact column order would fail on any
future addition, which turns the test into an obstacle. They assert the INVARIANT instead: each
tab's own signal columns precede the wide context columns. A new column can be added anywhere
sensible; a regression that puts `sector` back in front of the QGLP legs fails.

Run with: pytest tests/test_market_pulse_columns.py -v
"""

import contextlib
import io as _io
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

import pandas as pd
import pytest

from data_engine import (coerce_numeric_columns, compute_derived_signals, load_all_csvs,
                         merge_datasets)

_APP = os.path.join(os.path.dirname(__file__), "..", "app.py")

# Columns that are WIDE or merely contextual -- they may appear, but never before the signal.
CONTEXT = {"sector", "market_category", "market_cap"}


@pytest.fixture(scope="module")
def src():
    return _io.open(_APP, encoding="utf-8").read()


@pytest.fixture(scope="module")
def live():
    from core import run_scoring_pipeline
    with contextlib.redirect_stdout(_io.StringIO()):
        return run_scoring_pipeline(
            compute_derived_signals(coerce_numeric_columns(merge_datasets(load_all_csvs("local")))))


def _cols(src, var):
    """The column list literal assigned to `var`."""
    i = src.index(var + " = [c for c in [")
    block = src[i:src.index("]", src.index("[c for c in [", i) + 14) + 1]
    return re.findall(r'"([a-z_]+)"', block)


# -- 1. Nothing was dropped ----------------------------------------------------------------
@pytest.mark.parametrize("var,expected", [
    ("_ts_cols", {"rank", "name", "verdict_direction", "sector", "market_category", "market_cap",
                  "composite_score", "quality_score", "momentum_score", "piotroski_fscore",
                  "smart_money_flow", "buy_zone_label"}),
    ("_q_cols", {"rank", "name", "verdict_direction", "red_flag_count", "sector", "market_cap",
                 "qglp_score", "qglp_quality", "qglp_growth", "qglp_longevity", "qglp_price",
                 "smart_money_flow", "buy_zone_label"}),
])
def test_the_reorder_dropped_no_column(src, var, expected):
    """The fix was ORDER. If a column ever disappears, that is a different change and needs its
    own justification -- a layout problem must not be solved by destroying information."""
    got = set(_cols(src, var))
    assert got == expected, (
        f"{var} changed membership.\n  removed: {sorted(expected - got)}\n  added: {sorted(got - expected)}"
    )


# -- 2. The invariant: signal before context ------------------------------------------------
def test_qglp_legs_come_before_the_wide_context_columns(src):
    """The framework's four legs are the tab's entire subject. `sector` used to precede them and
    pushed qglp_price off-screen."""
    cols = _cols(src, "_q_cols")
    legs = [c for c in cols if c.startswith("qglp_")]
    assert len(legs) == 5, f"expected qglp_score + 4 legs, found {legs}"
    last_leg = max(cols.index(c) for c in legs)
    for ctx in CONTEXT & set(cols):
        assert cols.index(ctx) > last_leg, (
            f"'{ctx}' sits at position {cols.index(ctx)}, before the QGLP legs end at {last_leg}. "
            f"It is one of the widest columns in the frame and will push qglp_price off-screen."
        )


def test_tsunami_evidence_comes_before_the_wide_context_columns(src):
    """Tsunami's claim is that all 7 conviction conditions fire at once, so the evidence for it
    leads."""
    cols = _cols(src, "_ts_cols")
    signal = [c for c in ("composite_score", "quality_score", "momentum_score", "piotroski_fscore")
              if c in cols]
    assert len(signal) >= 3, f"the conviction evidence columns vanished: {cols}"
    last_signal = max(cols.index(c) for c in signal)
    for ctx in CONTEXT & set(cols):
        assert cols.index(ctx) > last_signal, (
            f"'{ctx}' precedes the conviction scores and will crowd them out of view"
        )


def test_sector_score_is_not_the_last_column(src):
    """avg_composite was second-to-last and rendered as a bar plus one truncated digit."""
    cols = _cols(src, "_sec_order")
    assert "avg_composite" in cols
    assert cols.index("avg_composite") <= 2, (
        f"Score sits at position {cols.index('avg_composite')}; it is one of the three figures a "
        f"reader scans first and was being clipped at the right edge"
    )


def test_name_stays_near_the_front(src):
    """Whatever else moves, you must be able to tell WHICH stock a row is."""
    for var in ("_ts_cols", "_q_cols"):
        cols = _cols(src, var)
        assert cols.index("name") <= 1, f"{var}: 'name' moved to position {cols.index('name')}"


# -- 3. The tooltip claim must match the data -----------------------------------------------
def test_pct_qualify_tooltip_no_longer_claims_statistical_robustness(src):
    """It said "Robust to sector size." A percentage is SCALE-FREE -- it stops big sectors
    dominating -- but small ones then reach extremes easily, which is the opposite bias."""
    i = src.index('"pct_qualify":')
    block = src[i:i + 900]
    assert "Robust to sector size" not in block, (
        "the tooltip again claims statistical robustness; measurement contradicts it (see below)"
    )
    assert "SCALE-FREE" in block, "the tooltip no longer says what the metric actually is"
    assert "Count" in block, "the tooltip should send the reader to Count, its missing context"


def test_the_small_sample_claim_in_that_tooltip_is_true(live, src):
    """SELF-VERIFYING: the tooltip states that most of the top sectors are small. That figure is
    checked against live data so the prose cannot go stale -- the same pattern as
    tests/test_reinvestment_rate_data_gap.py."""
    g = (live.groupby("sector")
         .agg(count=("name", "size"), qual=("gate_pass", "mean"))
         .reset_index())
    g = g[g["count"] >= 5].sort_values("qual", ascending=False)
    assert len(g) >= 15, "too few sectors clear the 5-stock floor to judge"
    top10 = g.head(10)
    small = int((top10["count"] < 12).sum())
    assert small >= 6, (
        f"only {small} of the top 10 sectors now hold fewer than 12 stocks. The tooltip claims 8; "
        f"if the distribution has shifted this far, remeasure and rewrite it."
    )
    assert top10["count"].median() < g["count"].median(), (
        "the top-ranked sectors are no longer smaller than the universe median -- the small-sample "
        "bias the tooltip warns about has gone, and the warning should go with it"
    )
