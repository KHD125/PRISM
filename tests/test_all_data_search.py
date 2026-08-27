"""
test_all_data_search.py
=======================
Contract for the All Data tab's signal filter.

WHY IT EXISTS: the tab renders 154 cells across 9 sections — the densest surface in the app —
and had no way to reach one except scrolling, while the Deep Scanner (ds_search) and the
Reference tab (ref_search) both had search.

TWO PROPERTIES THAT MATTER MOST:

1. AN EMPTY QUERY MUST CHANGE NOTHING. A filter that quietly drops a cell when unused would be
   far worse than no filter: every signal on this tab is decision-grade, and the tab's caption
   promises "every final, decision-grade signal the engine computes".

2. IT SEARCHES MEANING, NOT ONLY NAMES. The match runs over each cell's label AND its glossary
   text, so "cost of equity" finds Enduring VC / Dilution Vampire / Growth-Value Trap — none of
   which contain those words in their label. Name-only search on 154 cryptic labels would be
   most of the work for a fraction of the value.

Run with: pytest tests/test_all_data_search.py -v
"""

import contextlib
import io as _io
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import pytest

import ui.ui_tearsheet as T
from data_engine import (coerce_numeric_columns, compute_derived_signals, load_all_csvs,
                         merge_datasets)


@pytest.fixture(scope="module")
def row():
    with contextlib.redirect_stdout(_io.StringIO()):
        df = compute_derived_signals(
            coerce_numeric_columns(merge_datasets(load_all_csvs("local"))))
    return df.iloc[0]


def _render(row, query=""):
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
        T.render_raw_signals(row, query=query)
    finally:
        T.st = real
    return " ".join(out)


def _cells(html):
    return html.count("ts-raw-cell")


def _labels(html):
    return [m.strip() for m in re.findall(r'ts-raw-lbl">([^<]{1,40})', html)]


# ── 1. Unused, it must be invisible ─────────────────────────────────────────────────────
def test_empty_query_renders_every_cell(row):
    assert _cells(_render(row, "")) > 140, "the unfiltered grid lost cells"


def test_empty_query_output_is_byte_identical_to_no_query_at_all(row):
    """The default argument and an explicit blank must produce the same page — and neither may
    emit the search footer."""
    a, b = _render(row), _render(row, "")
    assert a == b
    assert "Showing <b>" not in a and "No signal matches" not in a


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_whitespace_only_query_is_treated_as_empty(row, blank):
    assert _cells(_render(row, blank)) == _cells(_render(row, ""))


# ── 2. It filters, and reports honestly ─────────────────────────────────────────────────
def test_query_narrows_the_grid(row):
    full, hit = _cells(_render(row, "")), _cells(_render(row, "roce"))
    assert 0 < hit < full
    for lbl in _labels(_render(row, "roce")):
        pass  # labels themselves need not contain the token — the tooltip may carry it


def test_footer_counts_match_what_was_actually_rendered(row):
    """The count must be measured, not asserted — this footer was initially placed before the
    sections ran and reported "no match" while cells were on screen."""
    for q in ("roce", "pledge", "growth"):
        html = _render(row, q)
        m = re.search(r"Showing <b>(\d+)</b> of (\d+) signals", html)
        assert m, f"no footer for {q!r}"
        assert int(m.group(1)) == _cells(html), (
            f"footer claims {m.group(1)} but {_cells(html)} cells rendered"
        )
        assert int(m.group(2)) > 140


def test_no_match_says_so_rather_than_rendering_an_empty_page(row):
    html = _render(row, "zzzzqqqq")
    assert _cells(html) == 0
    assert "No signal matches" in html


def test_hidden_sections_are_named(row):
    """Saying WHERE nothing matched is the useful half — otherwise the page just looks short."""
    html = _render(row, "pledge")
    assert "no match in" in html
    assert "Business Quality" in html


# ── 3. Meaning, not just names ──────────────────────────────────────────────────────────
def test_search_matches_the_tooltip_not_only_the_label(row):
    """"cost of equity" appears in no cell LABEL, but several glossary entries."""
    html = _render(row, "cost of equity")
    assert _cells(html) > 0, "meaning-search found nothing — it is matching labels only"
    labels = _labels(html)
    assert not any("cost of equity" in l.lower() for l in labels), (
        "this assertion is meant to prove the match came from the TOOLTIP; a label now "
        "contains the phrase, so pick a different probe"
    )


def test_all_tokens_must_appear(row):
    """Token-AND, like the Reference tab: a two-word query is narrower than either word."""
    both = _cells(_render(row, "roce med"))
    one = _cells(_render(row, "roce"))
    assert 0 < both <= one


def test_search_is_case_insensitive(row):
    assert _cells(_render(row, "ROCE")) == _cells(_render(row, "roce"))


# ── 4. The widget lives in app.py, not here ─────────────────────────────────────────────
def test_renderer_declares_no_widget():
    """ui_tearsheet is bound by the stateless contract — app.py owns session_state.

    AST, not a substring scan: the first version of this test matched "st.session_state" inside
    this very function's DOCSTRING, which explains why the widget lives in app.py. Prose that
    NAMES a banned call is not a banned call — the same trap that made an earlier completeness
    test pass on an import list.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(T.render_raw_signals)))
    used = {
        f"{n.func.value.id}.{n.func.attr}"
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name)
    }
    attrs = {
        f"{n.value.id}.{n.attr}"
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
    }
    banned = {"st.text_input", "st.button", "st.slider", "st.checkbox",
              "st.selectbox", "st.multiselect", "st.session_state"}
    hits = (used | attrs) & banned
    assert not hits, f"stateless renderer performs state-mutating calls: {sorted(hits)}"
