"""
test_ui_smoke.py
================
UI-layer smoke tests via Streamlit's native AppTest harness (streamlit.testing.v1).

WHY AppTest and not Playwright: AppTest executes the REAL render functions in-process
inside pytest — no browser, no server, deterministic, CI-friendly — and catches the
exact bug class unit tests miss: NameErrors in render paths, "+nan%" leaking into HTML,
and every stock showing identical values. (All three happened before these tests existed.)

The fixture runs the full real pipeline on a 400-stock sample of the local CSVs once
per session, then drives the tearsheet render functions through AppTest.

Skips cleanly when the local CSV data folder is absent (public repo has code only).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

import pandas as pd
import numpy as np
import pytest

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Other Resources", "CSV Data")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(_DATA_DIR),
    reason="Local CSV data not present (code-only checkout) — UI smoke tests need real data",
)


@pytest.fixture(scope="module")
def scored_df() -> pd.DataFrame:
    """Full real pipeline on a 400-stock sample (cross-section large enough for ranks)."""
    from core.data_engine import (load_all_csvs, merge_datasets,
                                  coerce_numeric_columns, compute_derived_signals)
    from core.forensic_engine import compute_forensic_signals, apply_forensic_penalty
    from core.scoring_engine import run_full_scoring

    datasets = load_all_csvs("local")
    sample_ids = datasets["ratio"]["company_id"].head(400)
    for name in datasets:
        datasets[name] = datasets[name][datasets[name]["company_id"].isin(sample_ids)]

    df = merge_datasets(datasets)
    df = coerce_numeric_columns(df)
    df = compute_derived_signals(df)
    df = compute_forensic_signals(df)
    df = run_full_scoring(df, "Hybrid", "Balanced")
    df = apply_forensic_penalty(df)
    return df


def _render_tearsheet_app():
    """Mini-app executed by AppTest: renders every major tearsheet module for one stock."""
    import streamlit as st
    from ui.ui_tearsheet import (
        render_stock_hero, render_score_strip, render_sell_alerts_panel,
        render_financial_insights, render_forensic_perimeter, render_guru_frameworks,
        render_mauboussin_radar, render_valuation_inversion_and_sizing_cockpit,
        render_canslim_radar, render_sepa_radar, render_marks_radar,
        render_mosl_wealth_matrix,
    )
    stock = st.session_state["stock_row"]
    render_stock_hero(stock)
    render_score_strip(stock)
    render_sell_alerts_panel(stock)
    render_financial_insights(stock)
    render_forensic_perimeter(stock)
    render_guru_frameworks(stock)
    render_mauboussin_radar(stock)
    render_valuation_inversion_and_sizing_cockpit(stock)
    render_canslim_radar(stock)
    render_sepa_radar(stock)
    render_marks_radar(stock)
    render_mosl_wealth_matrix(stock)


def _run_tearsheet_for(stock: pd.Series):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_render_tearsheet_app)
    at.session_state["stock_row"] = stock
    at.run(timeout=60)
    return at


def _all_rendered_text(at) -> str:
    """Concatenate every markdown body AppTest captured."""
    return " ".join(str(md.value) for md in at.markdown)


def test_tearsheet_renders_top_stock_without_exception(scored_df):
    at = _run_tearsheet_for(scored_df.iloc[0])
    assert not at.exception, f"Tearsheet raised: {[str(e.value) for e in at.exception]}"


def test_tearsheet_renders_worst_stock_without_exception(scored_df):
    """Bottom-ranked stock: maximum missing data / fired flags — the hostile path."""
    at = _run_tearsheet_for(scored_df.iloc[-1])
    assert not at.exception, f"Tearsheet raised: {[str(e.value) for e in at.exception]}"


def test_tearsheet_renders_loss_maker_without_exception(scored_df):
    """NaN-PE stock exercises every loss-maker fallback in the cockpit."""
    loss_makers = scored_df[scored_df["pe"].isna()]
    if loss_makers.empty:
        pytest.skip("No loss-maker in the 400-stock sample")
    at = _run_tearsheet_for(loss_makers.iloc[0])
    assert not at.exception, f"Tearsheet raised: {[str(e.value) for e in at.exception]}"


def test_no_nan_leaks_into_rendered_html(scored_df):
    """The '+nan%' class of display bug: no literal 'nan' may appear in any rendered text."""
    at = _run_tearsheet_for(scored_df.iloc[0])
    text = _all_rendered_text(at).lower()
    # match standalone nan tokens like '+nan%', 'nan%', '₹ nan' — not words containing 'nan'
    import re
    leaks = re.findall(r"[+\-₹ >]nan[%< ]", text)
    assert not leaks, f"NaN leaked into rendered HTML: {leaks[:5]}"


def test_score_strip_differs_between_stocks(scored_df):
    """The 'same values for every stock' regression: two different stocks must render
    different score strips (this failed silently during the Google Sheets bug)."""
    from streamlit.testing.v1 import AppTest

    def _strip_app():
        import streamlit as st
        from ui.ui_tearsheet import render_score_strip
        render_score_strip(st.session_state["stock_row"])

    rendered = []
    for idx in [0, len(scored_df) // 2]:
        at = AppTest.from_function(_strip_app)
        at.session_state["stock_row"] = scored_df.iloc[idx]
        at.run(timeout=30)
        assert not at.exception
        rendered.append(" ".join(str(md.value) for md in at.markdown))
    assert rendered[0] != rendered[1], (
        "Two different stocks rendered IDENTICAL score strips — the flat-scores "
        "regression (wrong data loaded or scores not varying) is back."
    )


def test_payoff_framework_tiles_present_and_per_stock(scored_df):
    """The Mauboussin Ch.13 EV tiles must render, and must differ across stocks
    (the old static matrix showed identical hardcoded numbers for every stock)."""
    a = _all_rendered_text(_run_tearsheet_for(scored_df.iloc[0]))
    b = _all_rendered_text(_run_tearsheet_for(scored_df.iloc[len(scored_df) // 2]))
    assert "Expected Excess Return" in a, "Payoff Framework tiles missing from tearsheet"
    # Isolate the EV tile region loosely: the EV figures must not be identical across stocks
    import re
    ev_a = re.findall(r"Expected Excess Return[^%]*?([+\-]\d+\.\d)%", a)
    ev_b = re.findall(r"Expected Excess Return[^%]*?([+\-]\d+\.\d)%", b)
    assert ev_a and ev_b, "EV value not found in rendered tiles"
    # Not a hard inequality (two stocks CAN coincide) — but identical full tile text
    # for stocks with different inputs indicates the static-matrix regression.
    if ev_a == ev_b:
        upside_a = re.findall(r"Upside Leg[^%]*?\+(\d+\.\d)%", a)
        upside_b = re.findall(r"Upside Leg[^%]*?\+(\d+\.\d)%", b)
        assert upside_a != upside_b, (
            "EV tiles identical for two different stocks — static-matrix regression"
        )
