"""
test_ep_hockey_stick_tooltip.py
===============================
Contract for the EP Trajectory tooltip on Matrix & WCS (ui_tearsheet.render_ep_power_curve_module).

THE CONFUSION IT ANSWERS (reported 2026-08-27 against Lumax Auto Technologies Ltd). The Matrix &
WCS tab showed "🚀 Hockey Stick" while the Frameworks tab had no matching pill, which reads as the
app contradicting itself. It is not:

    ep_power_curve == "🚀 Hockey Stick"   EP > 0 AND rising              -> 533 stocks
    framework "🏒 EP Hockey Stick"        that PLUS P/E <= 20x           -> 211 stocks

The 28th WCS framework is TEM-P — Trends/Endowment/Moves lift economic profit, but the RETURN
needs a cheap entry (§4). Lumax is EP +₹190 Cr and rising on a P/E of 42.6: TEM, not TEM-P. The
split is deliberate and documented at scoring_engine.py:2880 — gating the FRAMEWORK rather than
the column keeps ep_power_curve a pure fundamentals signal.

So 322 stocks (15.2%) legitimately show the rocket with no pill, and nothing on either page said
why. The fix was the smallest possible one: a sentence appended to the card's existing `tip=`
string. NOT a rename (both labels are the study's own term), and NOT a layout change — an earlier
attempt to add a computed sub-line broke the module, because the cards are built inside one
concatenated expression.

WHAT THESE TESTS DEFEND: that the explanation is present, that its NUMBERS stay true, and that the
change really was text-only.

Run with: pytest tests/test_ep_hockey_stick_tooltip.py -v
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

import ui.ui_tearsheet as T
from data_engine import (coerce_numeric_columns, compute_derived_signals, load_all_csvs,
                         merge_datasets)

_TEARSHEET = os.path.join(os.path.dirname(__file__), "..", "ui", "ui_tearsheet.py")
HOCKEY = "🚀 Hockey Stick"


@pytest.fixture(scope="module")
def live():
    from core import run_scoring_pipeline
    with contextlib.redirect_stdout(_io.StringIO()):
        return run_scoring_pipeline(
            compute_derived_signals(coerce_numeric_columns(merge_datasets(load_all_csvs("local")))))


@pytest.fixture(scope="module")
def tip():
    """The EP Trajectory card's tooltip text, sliced from the source."""
    src = _io.open(_TEARSHEET, encoding="utf-8").read()
    i = src.index('_ep_metric("EP Trajectory"')
    block = src[i:src.index('".)', i) if '".)' in src[i:i + 3000] else i + 3000]
    j = block.index("tip=")
    return block[j:j + 1200]


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
        T.render_ep_power_curve_module(row)
    finally:
        T.st = real
    return " ".join(out)


# -- 1. The explanation is there, and says the right thing ---------------------------------
def test_tooltip_names_the_frameworks_extra_price_gate(tip):
    assert "FRAMEWORK is stricter" in tip, "the tooltip no longer distinguishes the two concepts"
    assert "20x" in tip, "the tooltip does not state the P/E gate that separates them"
    assert "TEM-P" in tip, "the tooltip lost the study's own name for the price-completed framework"


def test_tooltip_reaches_the_rendered_page(live):
    """A tip= argument that never makes it into the HTML would be documentation nobody can read."""
    row = live[live["ep_power_curve"] == HOCKEY].iloc[0]
    html = _render(row)
    assert "FRAMEWORK is stricter" in html, "the tooltip text is not in the rendered output"


# -- 2. Its numbers must stay true -----------------------------------------------------------
def test_the_counts_quoted_in_the_tooltip_still_match_live_data(live, tip):
    """SELF-VERIFYING TEXT, the same pattern as tests/test_reinvestment_rate_data_gap.py: the two
    figures are PARSED OUT OF THE TOOLTIP and checked, so the prose cannot quietly go stale."""
    quoted = [int(n) for n in re.findall(r"\b(\d{3})\b", tip)]
    assert len(quoted) >= 2, f"the tooltip no longer quotes both counts, found {quoted}"
    shown_q, framework_q = quoted[0], quoted[1]

    shown = int((live["ep_power_curve"] == HOCKEY).sum())
    framework = int(live["frameworks_passed"].astype(str)
                    .str.contains("EP Hockey Stick", na=False).sum())
    assert abs(shown - shown_q) <= 60, (
        f"the tooltip says {shown_q} stocks show {HOCKEY}; live data has {shown}. Update the text."
    )
    assert abs(framework - framework_q) <= 60, (
        f"the tooltip says {framework_q} pass the framework; live data has {framework}."
    )
    assert framework < shown, (
        "the framework no longer fires on a SUBSET of the trajectory -- the tooltip's whole "
        "explanation (a stricter price gate) is wrong if this inverts"
    )


def test_the_gap_the_tooltip_explains_actually_exists(live):
    """If every Hockey Stick passed the framework there would be nothing to explain."""
    shown = live["ep_power_curve"] == HOCKEY
    fw = live["frameworks_passed"].astype(str).str.contains("EP Hockey Stick", na=False)
    gap = int((shown & ~fw).sum())
    assert gap > 50, (
        f"only {gap} stocks show the trajectory without the framework pill; the confusion this "
        f"tooltip addresses has gone away and the sentence can be dropped"
    )
    assert int((~shown & fw).sum()) == 0, (
        "a stock passes the EP Hockey Stick FRAMEWORK without showing the trajectory -- the "
        "framework is meant to be a strict subset, so the tooltip's explanation is now wrong"
    )


def test_lumax_is_still_the_worked_case(live):
    """The reported stock, pinned so the docstring's example stays real."""
    m = live[live["name"].str.contains("Lumax Auto", case=False, na=False)]
    if m.empty:
        pytest.skip("Lumax Auto not in this universe")
    r = m.iloc[0]
    assert r["ep_power_curve"] == HOCKEY
    assert float(r["pe"]) > 20.0, "Lumax now passes the price gate; pick a new worked example"


# -- 3. Nothing else about the card moved ----------------------------------------------------
# NOTE: an AST-skeleton diff against the previous commit was written here and REMOVED. It proved
# the edit was string-only, which was worth checking ONCE and was verified before shipping -- but
# as a standing contract it would fail on every legitimate future structural edit to
# ui_tearsheet.py, the most-churned file in the repo. A test that must be deleted to make normal
# work possible is a trap, not an invariant. The behavioural checks below are the durable part.


def test_non_hockey_stick_cards_are_untouched(live):
    """Every other curve state must render exactly as before -- the tooltip is shared, but the
    card's value and colour must not have moved."""
    row = live[live["ep_power_curve"] == "📉 Value Trap"].iloc[0]
    html = _render(row)
    assert "📉 Value Trap" in html
    assert "28th WCS" in html
