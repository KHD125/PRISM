"""
test_data_quality_fixes.py
==========================
Contract for the 2026-06-12 data-quality fixes in compute_derived_signals:

FIX 1 — True Effective Tax Rate (plan Bug #5, completed):
  effective_tax_rate_pct = (PBT - PAT) / PBT x 100, clipped [0, 60].
  The old proxy (1 - PAT/EBITDA) includes Depreciation + Interest + Tax (live median
  42.1%) and was graded against tax-rate bands 20-35 -> healthy companies failed
  Malik P3. True ETR live median = 25.3% = India's post-2019 corporate rate. PBT
  coverage: 91% of the universe.

FIX 2 — hidden_obligation_growth rebuilt:
  CSV semantics: total_liabilities = WHOLE balance-sheet total (TL/TA = 1.0 for every
  stock), so the old "TL growing faster than debt" condition fired for ANY company
  growing retained earnings — 82% of the universe (noise, polluted Schilit checker 3).
  New: other_liabilities = TL - reserves - debt (payables/provisions/leases), flag
  only when its one-year growth exceeds 5% of total assets (materiality). Live: 38%.

Run with: pytest tests/test_data_quality_fixes.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

import pandas as pd
import numpy as np

from data_engine import (compute_derived_signals, COMMON_COLS, RATIO_COLS, INCOME_COLS,
                         BALANCE_COLS, CASHFLOW_COLS, SHAREHOLDING_COLS, TECHNICAL_COLS)

_ALL_MAPPED_COLS = set()
for _m in (COMMON_COLS, RATIO_COLS, INCOME_COLS, BALANCE_COLS,
           CASHFLOW_COLS, SHAREHOLDING_COLS, TECHNICAL_COLS):
    _ALL_MAPPED_COLS.update(_m.values())


def _frame(n: int = 25, **overrides) -> pd.DataFrame:
    """Minimal frame for compute_derived_signals — mirrors a full merge by materializing
    every mapped column as NaN (the schema guard only covers non-ratio columns)."""
    base = {
        "company_id":      [f"NSE:T{i}" for i in range(n)],
        "name":            [f"Test Co {i}" for i in range(n)],
        "sector":          ["Chemicals"] * n,
        "industry":        ["Specialty Chemicals"] * n,
        "market_category": ["Mid Cap"] * n,
        "market_cap":      np.linspace(500.0, 50000.0, n),
        "close_price":     [100.0] * n,
        # Tax inputs
        "pbt":             [100.0] * n,
        "pat":             [75.0] * n,
        "ebitda":          [140.0] * n,
        # Hidden-obligation inputs (balance-sheet totals INCLUDE equity, like the CSV)
        "total_assets":        [1000.0] * n,
        "total_assets_1yb":    [900.0] * n,
        "total_liabilities":   [1000.0] * n,
        "total_liabilities_1yb": [900.0] * n,
        "reserves":            [500.0] * n,
        "reserves_1yb":        [430.0] * n,
        "debt":                [200.0] * n,
        "debt_1yb":            [200.0] * n,
    }
    for k, v in overrides.items():
        base[k] = [v] * n if not isinstance(v, (list, np.ndarray)) else v
    df = pd.DataFrame(base)
    for col in _ALL_MAPPED_COLS - set(df.columns):
        df[col] = np.nan
    return df


# ═══════════════════════════════════════════════════════════════════════════
# FIX 1 — True Effective Tax Rate
# ═══════════════════════════════════════════════════════════════════════════

def test_effective_tax_rate_basic_identity():
    """PBT 100, PAT 75 -> ETR exactly 25% (India's new-regime corporate rate)."""
    out = compute_derived_signals(_frame())
    assert np.allclose(out["effective_tax_rate_pct"], 25.0)


def test_effective_tax_rate_nan_when_pbt_not_positive():
    out = compute_derived_signals(_frame(pbt=0.0))
    assert out["effective_tax_rate_pct"].isna().all()
    out2 = compute_derived_signals(_frame(pbt=-50.0))
    assert out2["effective_tax_rate_pct"].isna().all()


def test_effective_tax_rate_clipped():
    """Deferred-tax credit (PAT > PBT) floors at 0; one-off charge caps at 60."""
    credit = compute_derived_signals(_frame(pbt=100.0, pat=110.0))
    assert (credit["effective_tax_rate_pct"] == 0.0).all()
    charge = compute_derived_signals(_frame(pbt=100.0, pat=20.0))
    assert (charge["effective_tax_rate_pct"] == 60.0).all()


def test_malik_p3_uses_true_etr_not_ebitda_gap():
    """A healthy company: true ETR 25% (in band) but EBITDA gap 46% (out of old band).
    Old code graded the gap against 20-35 -> zero credit. New code must award FULL P3
    credit, so its malik_checklist_score must beat an identical company whose ETR is
    an abnormal 50%."""
    healthy  = compute_derived_signals(_frame(pbt=100.0, pat=75.0, ebitda=140.0))
    abnormal = compute_derived_signals(_frame(pbt=100.0, pat=50.0, ebitda=140.0))
    assert (healthy["malik_checklist_score"] > abnormal["malik_checklist_score"]).all(), (
        "Malik P3 must grade the TRUE effective tax rate (25% in-band vs 50% abnormal), "
        "not the EBITDA-to-PAT gap"
    )


# ═══════════════════════════════════════════════════════════════════════════
# FIX 2 — hidden_obligation_growth (equity-funded growth must NOT fire)
# ═══════════════════════════════════════════════════════════════════════════

def test_equity_funded_growth_does_not_fire():
    """The old false positive: balance sheet grew 100 purely via retained earnings
    (reserves +70, debt flat, other liabilities +30 = 3% of TA < 5% materiality).
    Old formula: liab_change 100 > debt_change 0 -> FIRED (wrong). New: must not."""
    out = compute_derived_signals(_frame())
    assert (out["hidden_obligation_growth"] == 0).all(), (
        "Retained-earnings growth must not flag hidden obligations — total_liabilities "
        "in this CSV includes equity (TL/TA == 1.0)"
    )


def test_material_other_liability_jump_fires():
    """Other liabilities jump by 80 (TL +100, reserves +20, debt flat) = 8% of TA > 5%."""
    out = compute_derived_signals(_frame(reserves=450.0, reserves_1yb=430.0))
    assert (out["hidden_obligation_growth"] == 1).all()


def test_debt_funded_growth_does_not_fire():
    """TL +100 fully explained by new debt (+100): other liabilities flat -> no flag
    (debt risk is covered by the dedicated debt flags, not this off-BS signal)."""
    out = compute_derived_signals(_frame(
        reserves=430.0, reserves_1yb=430.0,   # equity flat
        debt=300.0, debt_1yb=200.0,           # +100 debt explains the TL growth
    ))
    assert (out["hidden_obligation_growth"] == 0).all()


def test_missing_reserves_conservative_zero():
    out = compute_derived_signals(_frame(reserves=np.nan, reserves_1yb=np.nan))
    assert (out["hidden_obligation_growth"] == 0).all(), (
        "Missing reserves data -> cannot compute other-liabilities -> conservative 0"
    )
