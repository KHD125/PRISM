"""
test_score_confidence.py
========================
Contract for the Score Confidence column (data_coverage_pct / data_coverage_label)
behind the tearsheet "🔍 Evidence" badge.

Why this exists (2026-06-13): missing inputs become neutral 50s in percentile
ranking, so a data-starved stock compresses toward "average" while really being
"unknown" — the bug class behind the flat-scores incident, the FCF gap and the
DPR gap. The coverage column makes missing evidence visible per stock.

Contracts:
  1. Every CORE_SCORING_INPUTS name is genuinely present in scoring_engine source
     (the list cannot drift from the ranked signals it claims to describe).
  2. All inputs present -> 100% and "N/N inputs" label.
  3. Coverage arithmetic is exact for partially missing inputs.
  4. Columns absent from the frame entirely count as missing; the metric is always
     defined (no NaN) and bounded [0, 100].

Run with: pytest tests/test_score_confidence.py -v
"""

import sys
import os
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

import pandas as pd
import numpy as np

from scoring_engine import compute_composite_score, CORE_SCORING_INPUTS

_N = len(CORE_SCORING_INPUTS)


def _frame(**overrides) -> pd.DataFrame:
    """Composite-ready frame (mirrors test_sizing_cockpit) with EVERY core scoring
    input populated, so coverage starts at exactly 100%."""
    n = 12
    base = {
        "quality_score":    [80.0] * n,
        "momentum_score":   [70.0] * n,
        "governance_bonus": [30.0] * n,
        # OLS residual inputs (need cross-sectional variance)
        "pb_ratio":     np.linspace(1.0, 8.0, n),
        "roe":          np.linspace(5.0, 30.0, n),
        "roce_med_5y":  np.linspace(8.0, 28.0, n),
        "market_cap":   np.linspace(500.0, 50000.0, n),
        # MOD 5 inputs
        "pe":               [25.0] * n,
        "fair_pe_qglp":     [35.0] * n,
        "g_star":           [12.0] * n,
        "sigma_g":          [20.0] * n,
        "trajectory_score": [0.5] * n,
        "close_price":      [500.0] * n,
        "vstop_value":      [450.0] * n,
    }
    for col in CORE_SCORING_INPUTS:
        base.setdefault(col, [10.0] * n)
    for k, v in overrides.items():
        base[k] = [v] * n if not isinstance(v, (list, np.ndarray)) else v
    return pd.DataFrame(base)


# ─────────────────────────────────────────────────────────────────────────────
# Contract 1 — the list cannot drift from the engine source
# ─────────────────────────────────────────────────────────────────────────────

def test_every_core_input_is_referenced_in_scoring_source():
    src_path = os.path.join(os.path.dirname(__file__), "..", "core", "scoring_engine.py")
    with open(src_path, encoding="utf-8") as fh:
        src = fh.read()
    # Strip the CORE_SCORING_INPUTS definition itself, then require each name to
    # still appear as a quoted column reference somewhere in the scoring logic.
    body = re.sub(r"CORE_SCORING_INPUTS\s*=\s*\[[^\]]*\]", "", src, count=1)
    missing = [c for c in CORE_SCORING_INPUTS if f'"{c}"' not in body]
    assert not missing, (
        f"CORE_SCORING_INPUTS drifted from scoring source — not ranked anywhere: {missing}"
    )


def test_core_inputs_unique_and_nonempty():
    assert _N > 0
    assert len(set(CORE_SCORING_INPUTS)) == _N, "duplicate names inflate coverage weight"


# ─────────────────────────────────────────────────────────────────────────────
# Contract 2 — full evidence -> 100%
# ─────────────────────────────────────────────────────────────────────────────

def test_full_inputs_give_100_pct():
    out = compute_composite_score(_frame())
    assert (out["data_coverage_pct"] == 100.0).all()
    assert (out["data_coverage_label"] == f"{_N}/{_N} inputs").all()


# ─────────────────────────────────────────────────────────────────────────────
# Contract 3 — exact arithmetic on partial evidence
# ─────────────────────────────────────────────────────────────────────────────

def test_missing_inputs_lower_coverage_exactly():
    k = 11  # null out 11 of the core inputs
    nulled = {c: np.nan for c in CORE_SCORING_INPUTS[:k]}
    out = compute_composite_score(_frame(**nulled))
    expected = (_N - k) / _N * 100.0
    assert np.allclose(out["data_coverage_pct"], expected)
    assert (out["data_coverage_label"] == f"{_N - k}/{_N} inputs").all()


# ─────────────────────────────────────────────────────────────────────────────
# Contract 4 — absent columns count as missing; metric always defined & bounded
# ─────────────────────────────────────────────────────────────────────────────

def test_absent_columns_count_as_missing_never_nan():
    df = _frame()
    dropped = [c for c in CORE_SCORING_INPUTS[:7] if c not in
               ("roe", "roce_med_5y", "market_cap")]  # keep residual inputs intact
    out = compute_composite_score(df.drop(columns=dropped))
    expected = (_N - len(dropped)) / _N * 100.0
    assert np.allclose(out["data_coverage_pct"], expected)
    assert not out["data_coverage_pct"].isna().any()
    assert out["data_coverage_pct"].between(0.0, 100.0).all()
