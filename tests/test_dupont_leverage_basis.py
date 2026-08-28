"""
test_dupont_leverage_basis.py
=============================
Contract for the duPont ROE-attribution leverage ladder (core/data_engine.py — _lev0/_lev1
behind roe_margin_contrib / roe_turnover_contrib / roe_leverage_contrib / roe_leverage_driven).

THE BUG THIS CLOSES (found by an external review 2026-08-28, verified by independent
re-measurement before anything was changed). Both terms of a year-over-year delta must come from
ONE line (the CLAUDE.md §5 cross-year basis rule) — but the leverage ladder mixed two equity
definitions: _lev0 = TA / net_worth (market_cap ÷ P/B) against _lev1 = TA_1yb / net_worth_1yb
(= reserves_1yb). Since net_worth runs above reserves, _lev0 was systematically depressed and
the leverage delta biased NEGATIVE. Measured on the live universe, pre-fix:

    median Δleverage   −0.1026 mixed basis   vs   −0.0086 with reserves both years
    sign flips         409 of 2,015 rows (20.3%)
    roe_leverage_driven ("is this rising ROE just debt?") under-fired at 5.53%

It is the SAME defect the EP family fixed (one basis, both years) — duPont was left behind.
MITIGATING FACT, also measured: the four roe_* duPont columns had ZERO consumers (no scoring, no
UI, no tests) — the bias never reached a screen. This file exists so that if they ARE ever
surfaced, they surface correct, and so the basis cannot silently drift back.

A second latent artifact died with the fix: the old net_worth_1yb (= reserves_1yb.fillna(0))
.clip(lower=1.0) turned a MISSING prior year into ₹1 Cr of equity, making lev1 = TA/1 — garbage
leverage for every stock without prior-year reserves. Missing now propagates NaN.

Run with: pytest tests/test_dupont_leverage_basis.py -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

import numpy as np
import pandas as pd

from data_engine import (BALANCE_COLS, CASHFLOW_COLS, COMMON_COLS, INCOME_COLS, RATIO_COLS,
                         SHAREHOLDING_COLS, TECHNICAL_COLS, compute_derived_signals)

_ALL_MAPPED_COLS = set()
for _m in (COMMON_COLS, RATIO_COLS, INCOME_COLS, BALANCE_COLS,
           CASHFLOW_COLS, SHAREHOLDING_COLS, TECHNICAL_COLS):
    _ALL_MAPPED_COLS.update(_m.values())


def _frame(n: int = 25, **overrides) -> pd.DataFrame:
    """Same fixture discipline as test_data_quality_fixes: every mapped column materialized."""
    base = {
        "company_id":      [f"NSE:T{i}" for i in range(n)],
        "name":            [f"Test Co {i}" for i in range(n)],
        "sector":          ["Chemicals"] * n,
        "industry":        ["Specialty Chemicals"] * n,
        "market_category": ["Mid Cap"] * n,
        "market_cap":      np.linspace(500.0, 50000.0, n),
        "close_price":     [100.0] * n,
        # duPont inputs — a stable base each test overrides
        "npm":               [10.0] * n,
        "npm_1yb":           [10.0] * n,
        "asset_turnover":    [1.0] * n,
        "asset_turnover_1yb": [1.0] * n,
        "total_assets":      [1000.0] * n,
        "total_assets_1yb":  [1000.0] * n,
        "reserves":          [500.0] * n,
        "reserves_1yb":      [500.0] * n,
    }
    for k, v in overrides.items():
        base[k] = [v] * n if not isinstance(v, (list, np.ndarray)) else v
    df = pd.DataFrame(base)
    missing = sorted(c for c in _ALL_MAPPED_COLS if c not in df.columns)
    if missing:
        df = pd.concat([df, pd.DataFrame(np.nan, index=df.index, columns=missing)], axis=1)
    return df


# ── 1. THE ONE-BASIS PROPERTY (the mutation-proof pin) ──────────────────────────────────────
def test_unchanged_equity_and_assets_mean_zero_leverage_delta():
    """Reserves 500 both years, assets 1000 both years → leverage did not move → the leverage
    contribution must be EXACTLY zero. Under the old mixed basis this was nonzero whenever
    net_worth (market_cap ÷ P/B) differed from reserves — i.e. for ~95% of real stocks — which
    is precisely how the bias slipped in. This test fails on the mixed basis and passes on one."""
    out = compute_derived_signals(_frame(price_to_book=3.0))   # P/B present → old basis diverges
    assert np.allclose(out["roe_leverage_contrib"], 0.0), (
        f"leverage 'moved' while equity and assets were flat both years: "
        f"{out['roe_leverage_contrib'].iloc[0]:+.4f} — the two years are on different equity bases"
    )
    assert (out["roe_leverage_driven"] == 0).all()


def test_genuine_deleveraging_reads_negative_and_releveraging_positive():
    """Equity grows, assets flat → leverage fell → negative contribution; and the mirror case."""
    delev = compute_derived_signals(_frame(reserves=600.0, reserves_1yb=400.0))
    relev = compute_derived_signals(_frame(reserves=400.0, reserves_1yb=600.0))
    assert (delev["roe_leverage_contrib"] < 0).all(), "rising equity, flat assets must read as deleveraging"
    assert (relev["roe_leverage_contrib"] > 0).all(), "falling equity, flat assets must read as releveraging"


def test_leverage_driven_fires_only_when_leverage_dominates():
    """The flag's meaning: rising ROE whose dominant positive driver is leverage expansion."""
    out = compute_derived_signals(_frame(reserves=400.0, reserves_1yb=600.0))   # pure releverage
    assert (out["roe_leverage_driven"] == 1).all()
    margin_led = compute_derived_signals(_frame(npm=14.0, npm_1yb=10.0))        # margins, not debt
    assert (margin_led["roe_leverage_driven"] == 0).all()


# ── 2. NaN honesty (the ₹1 Cr phantom equity artifact) ──────────────────────────────────────
def test_missing_prior_year_reserves_propagates_nan_not_phantom_equity():
    """The old ladder turned a missing reserves_1yb into ₹1 Cr of equity (lev1 = TA/1). An
    unverifiable prior year must yield NaN contributions and a 0 flag — never a fabricated
    leverage collapse."""
    out = compute_derived_signals(_frame(reserves_1yb=np.nan))
    assert out["roe_leverage_contrib"].isna().all(), "missing prior equity fabricated a leverage delta"
    assert (out["roe_leverage_driven"] == 0).all(), "a flag fired on absent evidence"


def test_negative_equity_propagates_nan():
    """§5 signed-base guard: leverage on negative equity is meaningless, not a huge number."""
    out = compute_derived_signals(_frame(reserves=-200.0))
    assert out["roe_leverage_contrib"].isna().all()
    assert (out["roe_leverage_driven"] == 0).all()


# ── 3. Structural: both years on the same line ──────────────────────────────────────────────
def test_the_ladder_reads_reserves_both_years():
    """The cross-year basis rule, pinned at the source so a future edit cannot half-revert:
    neither leverage term may read net_worth (the market_cap ÷ P/B figure) again."""
    import io as _io
    import re
    src = _io.open(os.path.join(os.path.dirname(__file__), "..", "core", "data_engine.py"),
                   encoding="utf-8").read()
    i = src.index("_rv0")
    block = src[i:src.index("_lev1_s", i)]
    assert 'df.get("reserves"' in block and 'df.get("reserves_1yb"' in block, (
        "the duPont leverage ladder no longer reads reserves for both years"
    )
    assert 'df.get("net_worth"' not in block, (
        "net_worth is back in the leverage ladder — the mixed-basis bug this file documents"
    )
