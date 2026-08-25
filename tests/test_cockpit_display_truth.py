"""
test_cockpit_display_truth.py
=============================
Contracts for the Matrix & WCS tab's display truth — three bugs found by reading the rendered
tab for EPack Prefab Technologies on 2026-08-25, all of the same family: a card stating
something the engine never computed.

1. HARDCODED STOP DISTANCE. The stop-loss card's subtitle was the literal string
   "-7-8% Active Perimeter Shield" — a number presented as measurement but never derived. True
   for 95 of 2,117 stocks (4.5%); the real distance spans -20%..+27% (p10-p90). EPack rendered
   "-7-8%" while its stop sat 25.5% BELOW the price, with the card immediately to its right
   correctly reading "+34.2% vs stop" — one row contradicting itself.

2. FABRICATED ZEROS. `value_creation_velocity` (96.0% populated) and `expectations_gap` (68.4%)
   were read with a 0.0 default, so 669 stocks were shown a confident "+0.00%" for a number
   nobody had. A data hole is not a measurement (CLAUDE.md §5).

3. EP LABEL. "✅ EP Positive" carried a green check on 285 of its 299 rows whose economic profit
   was SHRINKING (the other 14 have no prior-year figure; zero are flat or rising) — rendered
   directly beside "EP VELOCITY ₹-12 Cr Descending ↓". Renamed "➖ EP Positive, Not Rising".
   The taxonomy was always sound: "🚀 Hockey Stick" is 0-for-533 on negative velocity.

Run with: pytest tests/test_cockpit_display_truth.py -v
"""

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
from config import COLORS


class _Recorder:
    """Minimal Streamlit stand-in — captures every string the renderer emits."""

    def __init__(self):
        self.out = []

    def _cap(self, *a, **k):
        if a:
            self.out.append(str(a[0]))

    markdown = write = success = info = warning = error = caption = _cap

    def __getattr__(self, _name):
        return lambda *a, **k: None


def _render(**over):
    """Render the sizing cockpit for one synthetic stock and return everything it emitted."""
    base = {
        "close_price": 268.50, "vstop_value": 200.01,
        "optimal_portfolio_weight_pct": 3.92, "rupee_capital_allocation": 39202.80,
        "value_creation_velocity": 6.27, "expectations_gap": -0.94,
        "sepa_vcp_dryup": 0, "sepa_profile_score": 1, "sepa_pass": 0,
    }
    base.update(over)
    rec, real = _Recorder(), T.st
    try:
        T.st = rec
        T.render_valuation_inversion_and_sizing_cockpit(pd.Series(base))
    finally:
        T.st = real
    return " ".join(rec.out)


# ── 1. The stop-loss distance must be COMPUTED ──────────────────────────────────────
def test_stop_distance_is_computed_not_hardcoded():
    html = _render()
    assert "-7-8" not in html, "the hardcoded '-7-8%' stop-loss subtitle is back"
    m = re.search(r"([\d.]+)% below price", html)
    assert m, "the stop card no longer states a computed distance"
    # (268.50 - 200.01) / 268.50 = 25.51%
    assert abs(float(m.group(1)) - 25.5) < 0.1, f"wrong distance: {m.group(1)}"


def test_stop_distance_tracks_the_actual_stock():
    """Two stocks, two distances — a constant string cannot pass this."""
    a = re.search(r"([\d.]+)% below price", _render(close_price=100.0, vstop_value=90.0))
    b = re.search(r"([\d.]+)% below price", _render(close_price=100.0, vstop_value=50.0))
    assert a and b
    assert abs(float(a.group(1)) - 10.0) < 0.1
    assert abs(float(b.group(1)) - 50.0) < 0.1


def test_stop_above_price_is_not_reported_as_a_distance_below():
    """~26% of the universe sits at or below its stop — that must read as trend-broken."""
    html = _render(close_price=100.0, vstop_value=120.0)
    assert "ABOVE price" in html and "trend broken" in html
    assert "below price" not in html


@pytest.mark.parametrize("over", [
    {"close_price": np.nan}, {"vstop_value": np.nan}, {"close_price": 0.0},
])
def test_stop_subtitle_states_nothing_when_inputs_are_missing(over):
    html = _render(**over)
    assert "below price" not in html and "ABOVE price" not in html


# ── 2. Missing inputs must render N/A, never a fabricated zero ──────────────────────
@pytest.mark.parametrize("col", ["value_creation_velocity", "expectations_gap"])
def test_absent_structural_metric_renders_na_not_zero(col):
    html = _render(**{col: np.nan})
    assert "N/A" in html, f"{col} missing → the card must say N/A"
    assert "+0.00%" not in html, f"{col} missing → a fabricated '+0.00%' was rendered"


def test_present_structural_metrics_render_their_values():
    html = _render(value_creation_velocity=6.27, expectations_gap=-0.94)
    assert "+6.27%" in html and "-0.94%" in html


def test_expectations_gap_polarity_matches_the_engine_definition():
    """data_engine ~L3308: positive = priced above what it can deliver; negative = safety."""
    good = _render(expectations_gap=-5.0)
    bad  = _render(expectations_gap=+5.0)
    assert "margin of safety" in good
    assert "priced above what it can deliver" in bad
    assert COLORS["green"] in good


def test_structural_metrics_render_as_cards_not_bare_text():
    """They were three unstyled st.write/st.markdown lines under a card grid."""
    html = _render()
    for label in ("Value Creation Velocity", "Expectations Gap", "Consolidation Base"):
        assert label in html
    # a card carries the cockpit's border/background chrome; bare st.write does not
    assert html.count("border-radius:10px") >= 6


# ── 3. The EP power-curve label must not praise a shrinking company ─────────────────
def _live():
    import contextlib, io as _io
    from core.data_engine import (load_all_csvs, merge_datasets, coerce_numeric_columns,
                                  compute_derived_signals)
    with contextlib.redirect_stdout(_io.StringIO()):
        return compute_derived_signals(coerce_numeric_columns(merge_datasets(load_all_csvs("local"))))


@pytest.fixture(scope="module")
def live():
    return _live()


def test_no_ep_label_reads_positive_while_economic_profit_shrinks(live):
    vel = live["economic_profit_velocity"]
    for label, grp in live.groupby("ep_power_curve"):
        if not label or "✅" not in str(label):
            continue
        shrinking = int((vel[grp.index] < 0).sum())
        assert shrinking == 0, (
            f"{label!r} carries a ✅ but {shrinking} of its {len(grp)} rows have SHRINKING "
            f"economic profit — the label praises what the velocity card contradicts"
        )


def test_discovery_filter_order_matches_the_engine_labels(live):
    """_EPPC_ORDER is a hand-kept list; drift silently drops an option from the filter."""
    import ast
    src = open(os.path.join(os.path.dirname(__file__), "..", "ui", "ui_discovery.py"),
               encoding="utf-8").read()
    order = next(
        (ast.literal_eval(n.value) for n in ast.walk(ast.parse(src))
         if isinstance(n, ast.Assign) and any(
             isinstance(t, ast.Name) and t.id == "_EPPC_ORDER" for t in n.targets)),
        None)
    assert order, "_EPPC_ORDER not found in ui_discovery.py"
    emitted = {str(x) for x in live["ep_power_curve"].unique() if str(x).strip()}
    assert emitted <= set(order), (
        f"engine emits labels the Discovery filter cannot show: {sorted(emitted - set(order))}"
    )
