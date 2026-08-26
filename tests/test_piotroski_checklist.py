"""
test_piotroski_checklist.py
===========================
Contracts for the Piotroski F-Score checklist on the Forensics tab.

WHY IT EXISTS: all nine F-Score components are computed at 100% coverage, they sum EXACTLY to
`piotroski_fscore`, and until 2026-08-26 not one of them reached the screen — while the score
they add up to was displayed as "Piotroski n/9". The F-Score is a CHECKLIST: which nine pass
matters more than the total, and "4/9" without the failing five is its least useful form.

THE ⚪ STATE IS THE POINT. `f_liquidity_improving` passes for 0 of 2,117 stocks because the
sheet's "Current Ratio 1 Year Back" equals the current ratio on every row, so the engine's
`_cr != _cr_1yb` guard correctly refuses to score it. That caps the ENTIRE universe at 8/9 — the
observed maximum is 8, reached by 93 stocks, and zero stocks score 9. Rendering that as ❌ would
blame every company for a broken source column. It is "unverifiable is not passed" applied in the
other direction: do not CONDEMN on absent evidence either.

Run with: pytest tests/test_piotroski_checklist.py -v
"""

import contextlib
import io as _io
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import pytest

import ui.ui_tearsheet as T
from data_engine import (coerce_numeric_columns, compute_derived_signals, load_all_csvs,
                         merge_datasets)
from forensic_engine import compute_forensic_signals

F9 = [c for c, _l, _c, _t in T._PIOTROSKI_9]


@pytest.fixture(scope="module")
def live():
    with contextlib.redirect_stdout(_io.StringIO()):
        return compute_forensic_signals(compute_derived_signals(
            coerce_numeric_columns(merge_datasets(load_all_csvs("local")))))


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
        T.render_piotroski_checklist(row)
    finally:
        T.st = real
    return " ".join(out)


def _icons(html):
    return re.findall(r">(✅|❌|⚪)</span>", html)


def _row_of(html, label):
    """The window around one row. The icon span comes BEFORE the label span, so slicing forward
    from the label silently excludes it — which made an early version of the ❌ assertion below
    pass vacuously."""
    i = html.index(label)
    return html[max(0, i - 200):i + 400]


# ── 1. The panel decomposes the score it sits under ─────────────────────────────────────
def test_all_nine_components_are_rendered(live):
    assert len(F9) == 9, "the registry is not nine checks"
    for _, row in live.head(25).iterrows():
        assert len(_icons(_render(row))) == 9


def test_pass_count_equals_the_displayed_fscore(live):
    """The panel must never disagree with the number it explains."""
    for _, row in live.head(60).iterrows():
        html = _render(row)
        shown = re.search(r">(\d)</span><span[^>]*>of 9", html)
        assert shown, "the score header did not render"
        assert int(shown.group(1)) == int(row["piotroski_fscore"])
        assert _icons(html).count("✅") == int(row["piotroski_fscore"]), (
            "tick count disagrees with piotroski_fscore — the panel and the score have drifted"
        )


def test_components_sum_to_the_engine_score(live):
    assert (live[F9].sum(axis=1) == live["piotroski_fscore"]).all()


# ── 2. An unevaluable check is ⚪, never ❌ ──────────────────────────────────────────────
def test_liquidity_check_is_unevaluable_not_failed(live):
    """current_ratio_1yb is identical to current_ratio on every live row, so the check cannot
    run. Marking it ❌ would blame the company for a source-data defect."""
    row = live.iloc[0].copy()
    assert float(row["current_ratio"]) == float(row["current_ratio_1yb"]), (
        "fixture assumption broken: the source columns now differ"
    )
    html = _render(row)
    assert "not evaluable" in html
    assert _icons(html).count("⚪") >= 1
    seg = _row_of(html, "Liquidity improving")
    assert "⚪" in seg, "the unevaluable check does not carry the blank marker"
    assert "❌" not in seg, "an unevaluable check was rendered as a failure"


def test_attainable_maximum_is_stated(live):
    """Every stock is capped at 8/9 today and nothing on screen used to say so."""
    html = _render(live.iloc[0])
    m = re.search(r"(\d) check[s]? not evaluable · attainable maximum (\d)", html)
    assert m, "the cap on the attainable score is not disclosed"
    assert int(m.group(1)) == 1 and int(m.group(2)) == 8


def test_it_self_heals_when_the_source_data_is_fixed(live):
    """The ⚪ is DERIVED from the two inputs being equal — not hardcoded to a 'known broken'
    column — so a repaired sheet must restore a real verdict with no code change."""
    row = live.iloc[0].copy()
    row["current_ratio"], row["current_ratio_1yb"] = 2.0, 1.5     # a real improvement
    row["f_liquidity_improving"] = 1
    html = _render(row)
    assert "not evaluable" not in html, "still blind after the inputs were repaired"
    seg = _row_of(html, "Liquidity improving")
    assert "✅" in seg, "a repaired input did not restore a real verdict"


def test_zero_stocks_can_reach_nine_today(live):
    """The measured consequence of the dead check — pinned so the claim above stays true."""
    assert int((live["piotroski_fscore"] == 9).sum()) == 0
    assert int(live["f_liquidity_improving"].sum()) == 0


# ── 3. The row renderer matches the one it deliberately does not share ──────────────────
@pytest.mark.parametrize("value,icon", [
    (True, "✅"), (False, "❌"), (None, "⚪"),
    (np.bool_(True), "✅"), (np.bool_(False), "❌"),
    (np.int64(1), "✅"), (np.int64(0), "❌"), (np.float64(1.0), "✅"),
])
def test_row_icon_for_every_input_shape(value, icon):
    """`_pio_row` is a separate function from `_row` in render_financial_insights ON PURPOSE —
    that one is nested by design and five contracts assert it stays there. They are pinned to
    behave identically here instead. numpy types are the reason: `np.True_ is True` is False, the
    trap that once greyed out 13 of 17 rows on every stock."""
    assert icon in T._pio_row("X", value, "")


def test_missing_component_renders_unknown_not_failure():
    row = pd.Series({"piotroski_fscore": 3, "current_ratio": 1.0, "current_ratio_1yb": 2.0})
    html = _render(row)          # no f_* columns at all
    assert _icons(html).count("⚪") == 9, "absent components must be ⚪, never ❌"


def test_no_panel_without_a_score():
    assert _render(pd.Series({"piotroski_fscore": np.nan})) == ""
