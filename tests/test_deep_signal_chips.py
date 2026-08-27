"""
test_deep_signal_chips.py
=========================
Contract for the two chip strips under the verdict scorecard (ui_tearsheet.render_verdict_scorecard):
🔬 DEEP SIGNALS (WCS · Econ-Profit · VCR · Terms-of-Trade · Cash-Machine) and
⏱️ ENTRY TIMING (RS · Traj · EPS-Accel · Vol).

TWO THINGS THIS FILE DEFENDS.

1. VCR IS SHOWN TO TWO DECIMALS, AND THAT IS DELIBERATE.
   At one decimal, 150 stocks (7.1%) printed "1.0x" while being coloured from values as far apart
   as 1.0433 (green) and 0.9542 (red). The COLOURS were right — 1.0433 passes Buffett's one-dollar
   premise and 0.9542 fails it — but the label was too coarse to show why two chips differed. Two
   decimals cuts the ambiguous set to 16 (0.8%).

   NOTE THE CONTRAST WITH THE TRAJECTORY CARD, because the two look like the same bug and take
   OPPOSITE fixes. There the ±0.5pp cutoff was a NOISE FLOOR, so the right answer was to round
   first and let the boundary read neutral. Here 1.00 is Buffett's premise itself — a real
   economic line — so blurring it would be the error, and the fix is to show more precision.
   Diagnosing "display and threshold disagree" is not enough; you have to ask whether the
   threshold MEANS something before choosing the remedy.

2. NOTHING NEW BELONGS ON THE TIMING STRIP.
   Measured 2026-08-27 against every timing column in the frame: above_sma200 0.80, breakout_score
   0.79, dist_52wh 0.75, crs_52w 0.68, rsi_14d 0.61, dist_52wl 0.45 — all far above the 0.29 the
   four existing chips reach among themselves. The one genuinely orthogonal candidate,
   trend_breakout (0.088), fires on 6 of 2,117 stocks (0.3%) and is ALREADY on screen via
   trend_modifier's "🚀 Breakout". The strip is complete; this file records why so the analysis is
   not redone from scratch, and fails if the premise stops holding.

Run with: pytest tests/test_deep_signal_chips.py -v
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

import ui.ui_tearsheet as T
from data_engine import (coerce_numeric_columns, compute_derived_signals, load_all_csvs,
                         merge_datasets)

_TEARSHEET = os.path.join(os.path.dirname(__file__), "..", "ui", "ui_tearsheet.py")

TIMING_CHIPS = ["rs_score", "trajectory_score", "eps_acceleration", "volume_score"]


@pytest.fixture(scope="module")
def live():
    from core import run_scoring_pipeline
    with contextlib.redirect_stdout(_io.StringIO()):
        return run_scoring_pipeline(
            compute_derived_signals(coerce_numeric_columns(merge_datasets(load_all_csvs("local")))))


@pytest.fixture(scope="module")
def src():
    return _io.open(_TEARSHEET, encoding="utf-8").read()


def _render(row):
    out = []

    class _Rec:
        def markdown(self, *a, **k):
            if a:
                out.append(str(a[0]))

        def __getattr__(self, _n):
            return lambda *a, **k: None

    real = T.st
    try:
        T.st = _Rec()
        T.render_verdict_scorecard(row)
    finally:
        T.st = real
    return " ".join(out)


def _vcr_chip(html):
    """The rendered VCR value, e.g. '1.04x'."""
    m = re.search(r"VCR.*?([\d.]+x)", html, re.S)
    return m.group(1) if m else None


# -- 1. VCR precision ---------------------------------------------------------------------
def test_vcr_renders_two_decimals(live):
    row = live[live["value_creation_ratio"].notna()].iloc[0]
    shown = _vcr_chip(_render(row))
    assert shown is not None, "the VCR chip stopped rendering a value"
    assert re.fullmatch(r"\d+\.\d{2}x", shown), (
        f"VCR rendered {shown!r}; it must show TWO decimals. At one decimal 150 stocks printed "
        f"'1.0x' with opposite colours, because 1.0433 and 0.9542 land on opposite sides of "
        f"Buffett's one-dollar premise and one decimal cannot show it."
    )


def test_two_decimals_actually_shrinks_the_ambiguous_set(live):
    """Measured, not assumed: the fix must still be doing its job on current data."""
    v = live["value_creation_ratio"].dropna()

    def ambiguous(fmt):
        d = pd.DataFrame({"s": v.map(lambda x: fmt.format(x)), "g": v >= 1.0})
        bad = d.groupby("s")["g"].nunique()
        return int(d[d["s"].isin(bad[bad > 1].index)].shape[0])

    one, two = ambiguous("{:.1f}"), ambiguous("{:.2f}")
    assert two < one, f"two decimals no longer reduces ambiguity ({two} vs {one})"
    assert two / len(live) < 0.03, (
        f"{two} stocks ({two/len(live):.1%}) still print a VCR label that appears in both colours; "
        f"the display precision needs another decimal"
    )


def _chip_colour(html, label):
    """The colour of the chip whose content starts with `label`.

    MUST parse the chip's OWN span, not a slice. `_ds()` emits
    `<span style="...color:#xxxxxx;...">LABEL&nbsp;VALUE</span>` -- the colour precedes the label,
    so slicing FORWARD from the label reads past the chip's end and picks up the NEXT chip's
    colour. The first version of this test did exactly that and reported VCR 1.0433 as red.
    """
    m = re.search(r"color:(#[0-9a-fA-F]{6});[^>]*>" + re.escape(label) + r"[<&\s]", html)
    return m.group(1).lower() if m else None


def test_the_vcr_colour_itself_is_correct(live):
    """The colours were never the bug -- pin that, so a future 'fix' does not blur a real line."""
    for val, expect_green in ((1.0433, True), (0.9542, False), (1.0, True), (0.99, False)):
        row = live[live["value_creation_ratio"].notna()].iloc[0].copy()
        row["value_creation_ratio"] = val
        got = _chip_colour(_render(row), "VCR")
        want = (T.COLORS["green"] if expect_green else T.COLORS["red"]).lower()
        assert got == want, (
            f"VCR {val} rendered {got}; expected {want}. >=1.00 must be green (it passes the "
            f"one-dollar premise), below must be red -- the colours were correct before this "
            f"precision change and must stay correct after it."
        )


def test_missing_vcr_renders_a_dash_not_a_number(live):
    row = live.iloc[0].copy()
    row["value_creation_ratio"] = np.nan
    seg = _render(row)
    seg = seg[seg.index("VCR"):][:300]
    assert "—" in seg, "a missing VCR must read as an em-dash, never a fabricated number"


# -- 2. The documented correlation must match reality --------------------------------------
def test_documented_orthogonality_matches_live_data(live, src):
    """Self-checking comment, same pattern as tests/test_reinvestment_rate_data_gap.py. The figure
    is PARSED OUT OF THE COMMENT, so prose and contract cannot drift. The line previously claimed
    0.17 when the true value was 0.29 -- and that is the exact number someone consults to decide
    whether a fifth timing chip clears the bar."""
    m = re.search(r"max pairwise corr ([\d.]+)", src)
    assert m, "the timing strip's orthogonality claim vanished from the comment"
    documented = float(m.group(1))
    c = live[TIMING_CHIPS].corr().abs().copy()
    for k in TIMING_CHIPS:
        c.loc[k, k] = 0.0
    actual = float(c.max().max())
    assert abs(actual - documented) <= 0.08, (
        f"the comment documents a max pairwise correlation of {documented} but live data measures "
        f"{actual:.3f}. Remeasure and rewrite that line -- it is the bar used to judge whether a "
        f"new timing chip adds anything."
    )


# -- 3. No chip may go dead, and none may be added blindly ---------------------------------
@pytest.mark.parametrize("col,green,red", [
    ("wcs_score", lambda s: s >= 5, lambda s: s < 5),
    ("economic_profit", lambda s: s > 0, lambda s: s <= 0),
    ("value_creation_ratio", lambda s: s >= 1.0, lambda s: s < 1.0),
    ("terms_of_trade_spread", lambda s: s > 0, lambda s: s <= 0),
    ("cash_machine_score", lambda s: s >= 50, lambda s: s < 50),
    ("rs_score", lambda s: s >= 70, lambda s: s <= 30),
    ("trajectory_score", lambda s: s >= 0.5, lambda s: s < 0),
    ("eps_acceleration", lambda s: s >= 10, lambda s: s < 0),
    ("volume_score", lambda s: s >= 60, lambda s: s <= 20),
])
def test_no_chip_is_dead_or_unanimous(live, col, green, red):
    """A chip that is the same colour on every stock is decoration. This strip has shipped that
    bug before -- all five Deep Signals chips rendered permanently neutral until 2026-08-24,
    because raw NumPy bools were passed where Python bools were expected."""
    s = live[col]
    ok = s.notna()
    assert ok.mean() > 0.50, f"{col} covers only {ok.mean():.1%} of the universe"
    g, r = (green(s) & ok).mean(), (red(s) & ok).mean()
    assert g > 0.02, f"{col} is green on {g:.1%} of stocks -- effectively dead"
    assert r > 0.02, f"{col} is red on {r:.1%} of stocks -- effectively dead"


def test_no_timing_candidate_has_quietly_become_worth_adding(live, src):
    """The strip is closed on evidence, and this is that evidence. If a candidate ever becomes
    MORE orthogonal than the four already shown are to each other, this fails and asks for a
    fresh judgement rather than letting the 2026-08-27 analysis stand forever unexamined."""
    c = live[TIMING_CHIPS].corr().abs().copy()
    for k in TIMING_CHIPS:
        c.loc[k, k] = 0.0
    bar = float(c.max().max())
    cands = ["above_sma200", "breakout_score", "dist_52wh", "crs_52w", "rsi_14d", "dist_52wl"]
    checked = 0
    for cand in cands:
        if cand not in live.columns or live[cand].dtype.kind not in "if":
            continue
        checked += 1
        cor = max(abs(live[cand].corr(live[k])) for k in TIMING_CHIPS)
        assert cor > bar, (
            f"{cand} now correlates {cor:.3f} with the shown chips, BELOW the {bar:.3f} the four "
            f"reach among themselves. It has become a genuine addition -- re-judge the strip."
        )
    assert checked >= 4, "the candidate list went stale; most columns no longer exist"


def test_the_one_orthogonal_candidate_is_still_too_rare_and_already_shown(live, src):
    """trend_breakout is genuinely orthogonal (0.088) and was still rejected: it fires on 0.3% of
    the universe and already reaches the screen through trend_modifier. Both halves of that
    reasoning are pinned, because either one changing would reopen the decision."""
    assert "trend_breakout" in live.columns
    rate = live["trend_breakout"].fillna(0).mean()
    assert rate < 0.05, (
        f"trend_breakout now fires on {rate:.1%} of stocks (was 0.3%). It is the one timing signal "
        f"orthogonal to the strip; if it is no longer rare, it deserves a chip."
    )
    assert (live["trend_modifier"].astype(str).str.contains("Breakout").sum() > 0), (
        "trend_modifier no longer surfaces Breakout, so trend_breakout is now invisible -- the "
        "reason it was kept off the strip no longer holds"
    )
