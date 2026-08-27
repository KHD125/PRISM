"""
test_trajectory_card.py
=======================
Contract for the Overview tab's Trajectory card (ui_tearsheet.render_trajectory_card).

WHAT IT IS. The second derivative: not where the business stands, but which way it is moving and
whether the move is speeding up. Nothing else on Overview asks that.

THE TRAP IT WAS BUILT AROUND, and the reason most of this file exists. The engine carries SEVEN
columns whose names all end in `_acceleration`. They are THREE unrelated things:

    pat/rev/ebitda_acceleration  = 3Y CAGR - 5Y CAGR    a true acceleration (a rate of a rate)
    npm/opm_acceleration         = latest QUARTER - the ANNUAL figure 1 year back
    gpm_acceleration             = latest QUARTER - the 5Y MEDIAN          (a third base)
    ebit_acceleration            = EBIT growth - EBITDA growth, SAME window — a D&A-intensity
                                   spread, not a time comparison at all

Rendering all seven under one "Acceleration" heading would have rebuilt the exact defect
tests/test_all_data_yoy_labels.py exists to prevent: figures on different bases stacked in one
column, inviting a comparison that is not valid. So the card splits growth from margin, prints
each margin row's own base, and leaves ebit_acceleration out entirely.

These tests pin that separation, because it is invisible from the rendered page — the numbers look
equally comparable whether or not the bases match. Nothing but a contract keeps them apart.

Run with: pytest tests/test_trajectory_card.py -v
"""

import ast
import contextlib
import inspect
import io as _io
import os
import re
import sys
import textwrap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

import numpy as np
import pandas as pd
import pytest

import ui.ui_tearsheet as T
from data_engine import (coerce_numeric_columns, compute_derived_signals, load_all_csvs,
                         merge_datasets)


@pytest.fixture(scope="module")
def live():
    with contextlib.redirect_stdout(_io.StringIO()):
        return compute_derived_signals(
            coerce_numeric_columns(merge_datasets(load_all_csvs("local"))))


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
        T.render_trajectory_card(row)
    finally:
        T.st = real
    return " ".join(out)


def _text(html):
    return re.sub(r"<[^>]+>", " ", html)


# -- 1. Every basis is stated on the page ------------------------------------------------
def test_growth_section_names_both_windows(live):
    txt = _text(_render(live.iloc[0]))
    assert "3Y CAGR vs 5Y CAGR" in txt, (
        "the growth block no longer says which two windows it subtracts; a bare 'Acceleration' "
        "heading is what made the All Data YoY cells misleading"
    )


def test_every_margin_row_prints_its_own_base(live):
    """NPM/OPM measure against 1Y back, GPM against a 5Y median. Three rows, two bases -- if the
    page does not say so per row, the reader assumes they match."""
    txt = _text(_render(live.iloc[0]))
    assert "1Y back" in txt, "the NPM/OPM base vanished from the page"
    assert "5Y median" in txt, "the GPM base vanished -- it is NOT 1Y back like the rows above it"


def test_margin_block_is_not_called_an_acceleration(live):
    """npm_latest_q - npm_1yb is a first difference of a level. Calling it an acceleration would
    be wrong in the same way the engine's column name is wrong."""
    txt = _text(_render(live.iloc[0]))
    m = re.search(r"Margin Trend[^A-Z]*", txt)
    assert m, "the margin block lost its heading"
    assert "cceleration" not in m.group(0), (
        f"the margin block calls itself an acceleration: {m.group(0)[:90]!r}. It is a CHANGE in a "
        f"margin level, not a rate-of-a-rate."
    )


def test_the_quarterly_margin_rows_are_marked_quarterly(live):
    txt = _text(_render(live.iloc[0]))
    assert re.search(r"\bQ\s+[\d.]+%", txt), (
        "margin rows no longer mark their left-hand figure as a QUARTER; it is being compared "
        "against an annual base and the page must say so"
    )


# -- 2. The concept that must stay out ---------------------------------------------------
def test_ebit_acceleration_is_not_on_this_card():
    """ebit_acceleration = EBIT growth - EBITDA growth over the SAME window. It measures D&A
    intensity, not whether anything is speeding up. It shares a suffix with the real
    accelerations and nothing else, so it is the single most likely thing to be 'helpfully'
    added here later."""
    src = inspect.getsource(T.render_trajectory_card)
    for name in ("_ACCEL_GROWTH", "_ACCEL_MARGIN"):
        src += "\n" + repr(getattr(T, name))
    assert "ebit_acceleration" not in src, (
        "ebit_acceleration was added to the Trajectory card. It is EBIT growth minus EBITDA growth "
        "over one window -- a D&A-intensity spread, not an acceleration. It does not answer this "
        "card's question and cannot be read on the same scale as the rows beside it."
    )


def test_the_row_definitions_pair_each_delta_with_its_true_inputs(live):
    """Guards against a copy-paste that leaves a row showing one metric's delta beside another's
    inputs -- which would look perfectly plausible on screen."""
    for label, key, a, b in T._ACCEL_GROWTH:
        stem = key.replace("_acceleration", "")
        assert a.startswith(stem) and b.startswith(stem), f"{label}: {key} paired with {a}/{b}"
        assert a.endswith("_3y") and b.endswith("_5y"), f"{label}: windows are not 3Y vs 5Y"
    for label, key, a, b, tag in T._ACCEL_MARGIN:
        stem = key.replace("_acceleration", "")
        assert a == f"{stem}_latest_q", f"{label}: left side is not the latest quarter"
        assert b.startswith(stem), f"{label}: {key} paired with base {b}"


# -- 3. Absent evidence must not render as a confident zero ------------------------------
def test_missing_inputs_render_an_em_dash_not_zero(live):
    row = live.iloc[0].copy()
    for _, key, a, b in T._ACCEL_GROWTH:
        row[key] = row[a] = row[b] = np.nan
    for _, key, a, b, _t in T._ACCEL_MARGIN:
        row[key] = row[a] = row[b] = np.nan
    txt = _text(_render(row))
    assert "—" in txt, "a missing delta must read as an em-dash"
    assert "no data" in txt
    assert "+0.0" not in txt and "-0.0" not in txt, (
        "a missing figure rendered as 0.0 -- that reads as 'flat', a confident claim, when the "
        "truth is the number is absent"
    )


def test_headline_says_so_when_nothing_is_measurable(live):
    row = live.iloc[0].copy()
    for _, key, _a, _b in T._ACCEL_GROWTH:
        row[key] = np.nan
    assert "Not enough growth history" in _text(_render(row))


def test_headline_counts_only_what_it_could_measure(live):
    """An absent leg must not be counted as evidence either way."""
    row = live.iloc[0].copy()
    row["pat_acceleration"] = 20.0
    row["rev_acceleration"] = 15.0
    row["ebitda_acceleration"] = np.nan
    txt = _text(_render(row))
    assert "2 up" in txt, f"should report the 2 measurable legs, got: {txt[:220]!r}"
    assert "3 up" not in txt, "the absent leg was counted as if it were measured"
    assert "flat" not in txt.split("Growth Acceleration")[0], (
        "the missing leg was reported as FLAT -- absent evidence rendered as a real answer, which "
        "is the same error as printing 0.0 for a missing number"
    )


def test_headline_accounts_for_every_measured_leg(live):
    """The first cut read "Mixed — 1 up, 1 down of 3" on -1.9/+1.2/-0.1 and left the reader
    subtracting to discover the third leg was flat. A leg inside the dead band is an answer."""
    row = live.iloc[0].copy()
    row["pat_acceleration"], row["rev_acceleration"], row["ebitda_acceleration"] = -1.9, 1.2, -0.1
    head = _text(_render(row)).split("Growth Acceleration")[0]
    for part in ("1 up", "1 down", "1 flat"):
        assert part in head, f"headline {head.strip()!r} does not name the {part!r} leg"


@pytest.mark.parametrize("pat,rev,ebitda,expect", [
    (20.0, 15.0, 10.0, "Accelerating — 3 up"),
    (-20.0, -15.0, -10.0, "Decelerating — 3 down"),
    (20.0, -15.0, 0.0, "Mixed"),
    (0.0, 0.1, -0.2, "Flat — no material move in 3"),
])
def test_headline_matches_the_rows(live, pat, rev, ebitda, expect):
    row = live.iloc[0].copy()
    row["pat_acceleration"], row["rev_acceleration"], row["ebitda_acceleration"] = pat, rev, ebitda
    assert expect in _text(_render(row))


# -- 4. Display-only, and stateless ------------------------------------------------------
def test_card_does_not_mutate_the_row(live):
    row = live.iloc[0].copy()
    before = row.to_dict()
    _render(row)
    after = row.to_dict()
    changed = [k for k in before
               if not (pd.isna(before[k]) and pd.isna(after[k])) and before[k] is not after[k]
               and before[k] != after[k]]
    assert not changed, f"the card mutated the stock row: {changed[:5]}"


def test_delta_is_read_from_the_engine_column_not_recomputed(live):
    """A display that re-derives an engine number drifts from it the moment one side changes --
    the Fisher module/engine-pill lesson. Proven behaviourally: poison ONLY the engine column and
    the page must follow it, which it cannot do if it is subtracting the two inputs itself."""
    row = live[live["pat_acceleration"].notna()].iloc[0].copy()
    row["pat_acceleration"] = 123.4
    assert "+123.4" in _text(_render(row)), (
        "the rendered delta ignored the engine column, so it is being recomputed from the inputs"
    )


def test_renderer_declares_no_widget():
    """ui_tearsheet is bound by the stateless contract; app.py owns session_state. AST, not a
    substring scan -- prose that names a banned call is not a banned call."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(T.render_trajectory_card)))
    used = {f"{n.func.value.id}.{n.func.attr}" for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name)}
    attrs = {f"{n.value.id}.{n.attr}" for n in ast.walk(tree)
             if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
    banned = {"st.button", "st.slider", "st.checkbox", "st.selectbox", "st.text_input",
              "st.number_input", "st.multiselect", "st.session_state", "st.columns", "st.metric"}
    hits = (used | attrs) & banned
    assert not hits, f"stateless/layout contract violated: {sorted(hits)}"


# -- 5. It is wired in, and alive on real data -------------------------------------------
def test_card_is_rendered_on_the_overview_tab():
    src = _io.open(os.path.join(os.path.dirname(__file__), "..", "app.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "render_trajectory_card" in called, "the card exists but nothing calls it"


def test_the_card_is_not_dead_on_live_data(live):
    """Liveness: a card that says the same thing about everyone is decoration."""
    def verdict(row):
        ks = [row.get(k) for _l, k, _a, _b in T._ACCEL_GROWTH]
        ks = [k for k in ks if k is not None and not pd.isna(k)]
        if not ks:
            return "none"
        up, dn = sum(k > 0.5 for k in ks), sum(k < -0.5 for k in ks)
        return "up" if up > dn else "down" if dn > up else "mixed"

    share = live.apply(verdict, axis=1).value_counts(normalize=True)
    assert share.get("up", 0) > 0.05, f"almost nothing reads as accelerating: {share.to_dict()}"
    assert share.get("down", 0) > 0.05, f"almost nothing reads as decelerating: {share.to_dict()}"
    assert share.get("none", 0) < 0.30, (
        f"{share.get('none', 0):.0%} of the universe has no measurable trajectory -- the card is "
        f"mostly empty and the inputs need checking"
    )
