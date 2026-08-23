"""
test_no_verdict_from_data_holes.py
==================================
One principle, five call sites (data_engine deep audit, 2026-08-23):

    A MISSING INPUT MUST NEVER PRODUCE A VERDICT.

This is the same defect the Moat Endurance Factor carried (unreported ROCE -> MEF 0.0
-> "🔴 Degrading" label + a -8 quality penalty, 177 live rows) and the same principle
behind the "⚠️ Integrity Gates Open" rename. `np.select` defaults and ratio
`.fillna(0)` numerators are where it hides: NaN silently falls past every band into
whatever verdict sits in `default=`, and the user sees an accusation the data never
supported.

Live exposure measured before the fix:
  peg_zone        "🔴 Overpriced"      255 rows had peg = NaN (loss-makers / no growth)
  moat_growth_quad"💀 Wealth Destroyer" 16 rows had NO roce data at all
  cash_machine    "📄 Paper Profits"    24 rows had cfo_to_pat = NaN
  fcf_to_ocf_velocity  0.0 exactly     498 rows (OCF <= 0 / NaN) — latent: its only
                                       consumer reaches the same verdict via default,
                                       but that is exactly how the MEF trap stayed
                                       invisible until a consumer trusted the sentinel
  d28_fcf_to_pat_pct   0.0             24 rows had FCF = NaN (guard is on PAT only)

Contract: unknown -> NaN / "" (blank), never a band, never a penalty. Genuine bad
values still earn their genuine bad verdict.

Run with: pytest tests/test_no_verdict_from_data_holes.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from data_engine import compute_derived_signals
from test_data_quality_fixes import _frame


def _one(**over):
    return compute_derived_signals(_frame(n=1, **over)).iloc[0]


# ── peg_zone ──────────────────────────────────────────────────────────
def test_missing_peg_is_not_branded_overpriced():
    assert _one(peg=[np.nan])["peg_zone"] == ""


def test_genuinely_expensive_peg_still_reads_overpriced():
    assert _one(peg=[4.0])["peg_zone"] == "🔴 Overpriced"


def test_negative_peg_still_reads_declining():
    assert _one(peg=[-1.5])["peg_zone"] == "🔴 Declining"


# ── moat_growth_quad ──────────────────────────────────────────────────
def test_no_roce_or_growth_data_is_not_branded_wealth_destroyer():
    r = _one(roce=[np.nan], roce_med_5y=[np.nan], pat_gr_5y=[np.nan], pat_gr_3y=[np.nan])
    assert r["moat_growth_quad"] == ""


def test_genuine_low_moat_low_growth_still_reads_wealth_destroyer():
    r = _one(roce=[5.0], roce_med_5y=[5.0], pat_gr_5y=[2.0], pat_gr_3y=[2.0])
    assert r["moat_growth_quad"] == "💀 Wealth Destroyer"


def test_strong_moat_and_growth_still_reads_wealth_creator():
    r = _one(roce=[25.0], roce_med_5y=[25.0], pat_gr_5y=[20.0], pat_gr_3y=[20.0])
    assert r["moat_growth_quad"] == "⭐ Wealth Creator"


# ── cash_machine_label ────────────────────────────────────────────────
def test_missing_cash_conversion_is_not_accused_of_paper_profits():
    assert _one(cfo_to_pat=[np.nan])["cash_machine_label"] == ""


def test_genuine_weak_cash_conversion_still_reads_paper_profits():
    assert _one(cfo_to_pat=[40.0])["cash_machine_label"] == "📄 Paper Profits"


# ── the two ratios ────────────────────────────────────────────────────
def test_nonpositive_ocf_leaves_fcf_velocity_undefined_not_zero():
    """OCF <= 0 makes 'FCF as a share of OCF' meaningless — NaN, not a 0.0 that reads
    as 'generates no free cash'."""
    r = _one(operating_cash_flow=[-50.0], free_cash_flow=[-10.0])
    assert np.isnan(r["fcf_to_ocf_velocity"])


def test_healthy_ocf_still_computes_fcf_velocity():
    r = _one(operating_cash_flow=[100.0], free_cash_flow=[60.0])
    assert abs(r["fcf_to_ocf_velocity"] - 0.60) < 1e-9


def test_missing_fcf_leaves_fcf_to_pat_undefined_not_zero_percent():
    """Guard was on PAT only, so a NaN FCF numerator printed '0% of PAT'."""
    r = _one(free_cash_flow=[np.nan], operating_cash_flow=[np.nan],
             capex=[np.nan], pat=[100.0])
    assert np.isnan(r["d28_fcf_to_pat_pct"])


# ── UI: a blank label must render cleanly, never as an empty cell or a dangling separator ──

def _render_all_data(stock):
    from streamlit.testing.v1 import AppTest

    def _app():
        import streamlit as st
        from ui.ui_tearsheet import render_raw_signals
        render_raw_signals(st.session_state["s"])

    at = AppTest.from_function(_app)
    at.session_state["s"] = stock
    at.run(timeout=60)
    assert not at.exception, at.exception
    return " ".join(str(md.value) for md in at.markdown)


def test_blank_peg_zone_renders_as_na_not_an_empty_cell():
    import re
    import pandas as pd
    html = _render_all_data(pd.Series({"name": "No PEG Ltd", "peg_zone": ""}))
    mt = re.search(r'PEG Zone.*?ts-raw-val">([^<]*)<', html, re.S)
    assert mt and mt.group(1).strip() == "N/A", mt and mt.group(1)


def test_blank_peg_zone_leaves_no_dangling_separator():
    """The insights row read '  |  Lynch rule: ...' with an empty zone in front."""
    import io as _io
    import os as _os
    src = _io.open(_os.path.join(_os.path.dirname(__file__), "..", "ui", "ui_tearsheet.py"),
                   encoding="utf-8").read()
    assert 'f"{peg_zone}  |  Lynch rule' not in src, "unconditional separator is back"


# ── Round 2 (cross-check of the sweep itself): the INVERSE class — unearned GOOD ──

def test_missing_debt_data_does_not_earn_full_malik_leverage_points():
    """malik_p5: fillna(0) <= 0 read a MISSING D/E as 'debt-free' and granted full
    points — the inverse fabrication (unearned good). Siblings P4/P6 give 0 for
    missing; genuine zero-debt still earns full credit."""
    genuine = _one(debt_to_equity=[0.0])["malik_checklist_score"]
    missing = _one(debt_to_equity=[np.nan])["malik_checklist_score"]
    levered = _one(debt_to_equity=[2.5])["malik_checklist_score"]
    assert genuine > missing, "missing D/E must not score like debt-free"
    assert missing == levered + 0 or missing <= genuine - 10, "missing must earn the 0 band"


def test_missing_price_distance_is_not_branded_far_from_breakout():
    r = _one(dist_52wh=[np.nan], dist_13wh=[np.nan])
    assert r["d48_breakout_readiness"] == ""


def test_genuinely_distant_stock_still_reads_far():
    r = _one(dist_52wh=[45.0], dist_13wh=[30.0])
    assert r["d48_breakout_readiness"] == "FAR"


def test_missing_rsi_is_not_branded_weak_momentum():
    r = _one(rsi_14d=[np.nan])
    assert r["d49_momentum_quality"] == ""


def test_genuinely_weak_momentum_still_reads_weak():
    r = _one(rsi_14d=[35.0])
    assert r["d49_momentum_quality"] == "WEAK"
