"""Contract: the DISPLAY-ONLY cyclicality columns — max_earnings_drawdown_5y + cyclicality_tier.

Two columns shipped into compute_derived_signals (core/data_engine.py), both display-only (never
scored — pinned by the byte-identical composite test below):

  max_earnings_drawdown_5y — deepest TIME-ORDERED peak-to-trough fall in annual PAT over the 6
      available years (current + 5 back). A monotone compounder scores ~0; a commodity that
      collapsed at the trough scores high; >1.0 = the trough went negative. NaN when <4 of 6 present.
  cyclicality_tier(_code) — the a-priori business TYPE from the committed core/cyclicality_map.py
      (industry → sector fallback → "F"). The FINER, industry-level DISPLAY sibling of the
      SECTOR-level cyclical_peak_trap (which DOES feed scoring). Prior(tier) vs realized(drawdown)
      disagreement is itself the signal.

The COMMITTED map (not a runtime CSV read) is what makes this deploy-safe. The coverage guard is
SPLIT (mirroring the project's committed-test / live-ritual pattern): the synthetic miss-detection
below is committed (passes in a code-only CI clone with no data); the LIVE count-of-unmapped==0
tripwire lives in tools/verify.py, where the gitignored real CSV exists.
"""
import sys
import os
import io
import contextlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

import numpy as np
import pandas as pd
import pytest

from data_engine import (compute_derived_signals, COMMON_COLS, RATIO_COLS, INCOME_COLS,
                          BALANCE_COLS, CASHFLOW_COLS, SHAREHOLDING_COLS, TECHNICAL_COLS)
from core.cyclicality_map import INDUSTRY_TIER, SECTOR_TIER_FALLBACK, TIER_LABELS

_ALL_MAPPED_COLS = set()
for _m in (COMMON_COLS, RATIO_COLS, INCOME_COLS, BALANCE_COLS,
           CASHFLOW_COLS, SHAREHOLDING_COLS, TECHNICAL_COLS):
    _ALL_MAPPED_COLS.update(_m.values())


def _frame(n: int = 2, **overrides) -> pd.DataFrame:
    """Full-NaN mapped frame (the §6 synthetic pattern) with per-row overrides. sorted() pins the
    fill order so the fixture's block layout is seed-independent (determinism mandate, §5)."""
    base = {"company_id": [f"NSE:T{i}" for i in range(n)], "name": [f"T{i}" for i in range(n)]}
    for k, v in overrides.items():
        base[k] = list(v) if isinstance(v, (list, np.ndarray)) else [v] * n
    df = pd.DataFrame(base)
    missing = sorted(c for c in _ALL_MAPPED_COLS if c not in df.columns)
    if missing:
        df = pd.concat([df, pd.DataFrame(np.nan, index=df.index, columns=missing)], axis=1)
    return df


# ── map integrity ────────────────────────────────────────────────────────────
def test_map_integrity():
    """Every tier code in either map ∈ {A..F}, has a label, and both dicts are key-sorted (§5)."""
    codes = set(INDUSTRY_TIER.values()) | set(SECTOR_TIER_FALLBACK.values())
    assert codes <= set("ABCDEF"), f"tier code outside A-F: {sorted(codes - set('ABCDEF'))}"
    assert codes <= set(TIER_LABELS), f"tier code without a label: {sorted(codes - set(TIER_LABELS))}"
    assert list(INDUSTRY_TIER) == sorted(INDUSTRY_TIER), "INDUSTRY_TIER must be key-sorted (determinism)"
    assert list(SECTOR_TIER_FALLBACK) == sorted(SECTOR_TIER_FALLBACK), "SECTOR_TIER_FALLBACK must be key-sorted"


# ── synthetic unmapped-industry detection (the committed half of the coverage split) ──
def test_unmapped_industry_is_detectable_and_falls_to_F():
    """An industry+sector absent from both maps must (a) be detectable as a miss and (b) resolve to
    the 'F' catch-all — never crash. (The live count-of-misses==0 tripwire lives in tools/verify.py.)"""
    assert "__UNMAPPED__" not in INDUSTRY_TIER and "__UNMAPPED__" not in SECTOR_TIER_FALLBACK
    out = compute_derived_signals(_frame(n=1, industry="__UNMAPPED__", sector="__UNMAPPED__"))
    assert out["cyclicality_tier_code"].iloc[0] == "F"
    assert out["cyclicality_tier"].iloc[0] == TIER_LABELS["F"]


def test_known_industry_and_sector_fallback_resolve():
    """A mapped industry resolves to its committed tier; a NEW industry with a KNOWN sector uses the
    sector fallback (not 'F') — the chain industry → sector → 'F'."""
    ind = next(iter(INDUSTRY_TIER))               # any mapped industry (sorted-first)
    sec = next(iter(SECTOR_TIER_FALLBACK))        # any mapped sector
    out = compute_derived_signals(_frame(
        n=2,
        industry=[ind, "__NEW_INDUSTRY__"],
        sector=["__UNKNOWN_SECTOR__", sec],
    ))
    assert out["cyclicality_tier_code"].iloc[0] == INDUSTRY_TIER[ind]          # industry wins
    assert out["cyclicality_tier_code"].iloc[1] == SECTOR_TIER_FALLBACK[sec]   # sector fallback


# ── drawdown semantics ───────────────────────────────────────────────────────
def test_drawdown_vshape_high_compounder_zero_shorthistory_nan():
    """V-shape (collapse-then-recover) scores deep; monotone compounder ~0; <4 of 6 years → NaN.
    Columns are newest→oldest (pat, pat_1yb … pat_5yb); the engine reverses to oldest→newest."""
    out = compute_derived_signals(_frame(
        n=3,
        pat    =[100.0, 100.0, 50.0],   # row0 V-shape (recovered) | row1 compounder | row2 short
        pat_1yb=[ 20.0,  80.0, 40.0],
        pat_2yb=[100.0,  60.0, np.nan],
        pat_3yb=[100.0,  40.0, np.nan],
        pat_4yb=[100.0,  20.0, np.nan],
        pat_5yb=[100.0,  10.0, np.nan],
    ))
    dd = out["max_earnings_drawdown_5y"]
    # row0 oldest→newest 100,100,100,100,20,100 → peak 100, trough 20 → dd 0.8
    assert dd.iloc[0] > 0.7, f"V-shape should show deep drawdown, got {dd.iloc[0]}"
    # row1 oldest→newest 10,20,40,60,80,100 monotone up → dd ~0
    assert dd.iloc[1] < 0.05, f"compounder should be ~0, got {dd.iloc[1]}"
    # row2 only 2 of 6 present → NaN
    assert pd.isna(dd.iloc[2]), f"short history (<4 yrs) must be NaN, got {dd.iloc[2]}"


def test_drawdown_negative_trough_exceeds_one():
    """A trough that goes negative yields drawdown > 1.0 — (peak − neg_trough)/peak."""
    out = compute_derived_signals(_frame(
        n=1, pat=[50.0], pat_1yb=[-30.0], pat_2yb=[80.0],
        pat_3yb=[100.0], pat_4yb=[90.0], pat_5yb=[70.0],
    ))
    # oldest→newest 70,90,100,80,-30,50 → peak 100, trough -30 → (100-(-30))/100 = 1.3
    assert out["max_earnings_drawdown_5y"].iloc[0] > 1.0


# ── the KEY invariant: display-only, zero scoring leakage ─────────────────────
@pytest.mark.slow
def test_display_only_composite_byte_identical():
    """Corrupting the two cyclicality columns must NOT change composite_score — proving they feed no
    score (mirrors the verdict-engine display-only contract). compute_derived_signals runs in the
    DATA stage (fetch_and_clean_data), so run_scoring_pipeline never recomputes them: poisoning the
    clean frame's columns persists through scoring, and an identical composite proves no dependency."""
    from core import run_scoring_pipeline
    from core.data_engine import fetch_and_clean_data
    with contextlib.redirect_stdout(io.StringIO()):
        clean = fetch_and_clean_data("local")
        baseline = run_scoring_pipeline(clean.copy(), "Hybrid", "Balanced")
        poisoned_in = clean.copy()
        poisoned_in["max_earnings_drawdown_5y"] = 999.0
        poisoned_in["cyclicality_tier_code"] = "Z"
        poisoned_in["cyclicality_tier"] = "GARBAGE"
        poisoned = run_scoring_pipeline(poisoned_in, "Hybrid", "Balanced")
    pd.testing.assert_series_equal(
        baseline["composite_score"], poisoned["composite_score"], check_names=False)
