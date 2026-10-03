"""Contract: the IMPROVEMENT facet of quality_score — profit growth + margin widening.

WHY IT EXISTS (2026-10-03). quality_score was built almost entirely from LEVELS and long histories
(10-year ROCE/ROE medians, 5-year margin medians, 5/10-year growth). A study of the next three years'
winners said the level carried almost nothing and the recent CHANGE carried the signal. Measured on a
one-off StockScans export (Returns 1Y/3Y CAGR, verified against the snapshot and known prices),
traits known at the START of each window:

    trait at the start               3Y Oct23->26   1Y Oct25->26 (falling market)
    profit growth, 2y before            +0.173          +0.186
    margin widening, 2y before          +0.138          +0.190
    margin LEVEL at start               +0.015          +0.129
    improvement (both), top 40% vs rest +11.5 vs +2.9%/yr   -0.6 vs -13.9%

It held in small, mid AND large companies in both windows, was never negative in any cyclicality
tier (deep cyclicals ~0 over 3 years — the Lynch peak-margin trap was looked for and not found), and
on PRISM's own 66d / 28d windows the facet scored +0.069 / +0.110 with near-zero overlap with
quality_score (+0.018). Inside quality at 20%, rebuilt in EXACT engine form, the default mode moved
+0.147/+0.091 -> +0.146/+0.101 and every mode improved in the falling window. Whole-market ranking was
kept on measurement: within-sector/industry ranking lost signal (3Y +0.169 -> +0.148/+0.144) without
fixing the deep-cyclical weakness.

WHAT IS DELIBERATELY NOT IN IT:
  * ROCE change. The vendor half-updated `roce` to the new fiscal year (changed on 61% of rows June ->
    September, rolled into roce_1yb on 1.3%) while roce_1yb / roce_2yb stayed frozen, so any ROCE delta
    today compares mismatched years (CLAUDE.md §5 cross-year basis rule). Add it after the balance
    sheet rolls, with a basis check.
  * A turnaround bonus. Loss two years ago -> profit now has no growth RATE; measured forward edge
    -0.2 / +0.2pp, so the honest treatment is "no profit-growth evidence", not a reward.

20% is a prior, not a fit: 30% scored slightly higher and choosing it would be fitting two windows.
The six level facets are scaled by 0.8 each, so levels stay in — margin LEVEL still helped in the
falling year (+0.129).
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config as C
from core import scoring_engine as SE
from core.data_engine import compute_derived_signals
from tests.test_data_quality_fixes import _frame

_ROOT = Path(__file__).resolve().parent.parent

# the six level facets as they stood BEFORE improvement joined (2026-10-03)
_PRE = {"moat": 0.22, "growth": 0.22, "cash": 0.20, "margin": 0.13, "balance_sheet": 0.13, "valuation": 0.10}


# ── 1. Weights ──────────────────────────────────────────────────────────────────────────────
def test_improvement_is_one_fifth_of_quality_and_levels_are_scaled_not_dropped():
    W = C.QUALITY_WEIGHTS
    assert W["improvement"] == pytest.approx(0.20)
    for k, v in _PRE.items():
        assert W[k] == pytest.approx(v * 0.80), f"{k}: expected {v} x 0.8, got {W[k]}"
    assert set(W) == set(_PRE) | {"improvement"}
    assert sum(W.values()) == pytest.approx(1.0, abs=1e-9)


# ── 2. Inputs: P&L only, like-for-like years ──────────────────────────────────────────────────
def _fn_strings(name):
    src = (_ROOT / "core" / "scoring_engine.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == name)
    return {c.value for c in ast.walk(fn) if isinstance(c, ast.Constant) and isinstance(c.value, str)}


def test_improvement_reads_exactly_the_four_pnl_columns_and_no_roce():
    cols = {s for s in _fn_strings("_compute_improvement_score") if s.isidentifier()}
    assert {"pat", "pat_2yb", "opm", "opm_1yb"} <= cols, cols
    assert not {c for c in cols if "roce" in c or "asset" in c or "debt" in c}, (
        "a balance-sheet input crept in — the vendor's balance sheet lags the P&L by a fiscal year: " + str(cols))


def _imp(**cols):
    return SE._compute_improvement_score(pd.DataFrame(cols)).tolist()


def test_better_profit_growth_and_wider_margins_score_higher():
    s = _imp(pat=[150.0, 100.0, 80.0], pat_2yb=[100.0, 100.0, 100.0],
             opm=[18.0, 15.0, 12.0], opm_1yb=[15.0, 15.0, 15.0])
    assert s[0] > s[1] > s[2]
    assert all(0.0 <= v <= 100.0 for v in s)


def test_a_loss_year_is_not_growth_evidence():
    # loss -> profit, profit -> loss, and a loss in BOTH years: no growth RATE exists; margin
    # evidence alone decides. The both-loss row is the discriminating one: -20 vs -40 is a POSITIVE
    # ratio, so without the profitability guard it would read as a growth rate (sign-mixed rows go
    # NaN through the square root by accident, which is why a test without it missed mutation I3).
    a = _imp(pat=[50.0, -20.0, -20.0, 120.0], pat_2yb=[-10.0, 30.0, -40.0, 100.0],
             opm=[10.0] * 4, opm_1yb=[10.0] * 4)
    b = _imp(pat=[np.nan, np.nan, np.nan, 120.0], pat_2yb=[np.nan, np.nan, np.nan, 100.0],
             opm=[10.0] * 4, opm_1yb=[10.0] * 4)
    assert a == b


def test_no_evidence_is_neutral_not_a_verdict():
    s = _imp(pat=[np.nan, 120.0], pat_2yb=[np.nan, 100.0], opm=[np.nan, 12.0], opm_1yb=[np.nan, 10.0])
    assert s[0] == 50.0


# ── 3. Wiring: the facet really is 20% of quality_score ───────────────────────────────────────
def test_quality_score_carries_exactly_one_fifth_of_improvement(monkeypatch):
    n = 6
    df = compute_derived_signals(_frame(
        n, pat=[90.0, 100.0, 110.0, 120.0, 130.0, 140.0], pat_2yb=[100.0] * n,
        opm=[10.0, 11.0, 12.0, 13.0, 14.0, 15.0], opm_1yb=[12.0] * n))
    on = SE.compute_quality_score(df)
    monkeypatch.setitem(C.QUALITY_WEIGHTS, "improvement", 0.0)
    off = SE.compute_quality_score(df)
    assert on["improvement_score"].nunique() > 1          # the inputs discriminate
    assert np.allclose(on["quality_score"] - off["quality_score"], 0.20 * on["improvement_score"])


# ── 4. Every vintage can compute it (Movers re-scores archived copies with this engine) ───────
def test_every_snapshot_carries_the_inputs():
    snaps = sorted((_ROOT / "Other Resources" / "snapshots").glob("prism_snapshot_*.csv"))
    if not snaps:
        pytest.skip("snapshots are local-only")
    for p in snaps:
        head = set(pd.read_csv(p, nrows=0).columns)
        assert {"pat", "pat_2yb", "opm", "opm_1yb"} <= head, p.name


# ── 5. Surfaces: the facet is visible wherever the facets are shown ───────────────────────────
def test_config_card_counts_its_layers_from_the_weights():
    src = (_ROOT / "app.py").read_text(encoding="utf-8")
    assert "Quality Sub-Weights · 6 Layers" not in src
    assert "len(QUALITY_WEIGHTS)" in src


def test_improvement_appears_in_the_facet_rows_radar_and_quality_view():
    app = (_ROOT / "app.py").read_text(encoding="utf-8")
    comp = (_ROOT / "ui" / "ui_components.py").read_text(encoding="utf-8")
    assert '_qfrow("🚀 Improvement", "improvement_score")' in app
    i = app.index('"📊 Quality":')
    assert "improvement_score" in app[i:app.index("]", i)]
    radar = comp[comp.index("def render_radar_chart"):]
    radar = radar[:radar.index("fig = go.Figure()")]
    assert "improvement_score" in radar
