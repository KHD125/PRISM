"""
test_governance_asymmetric.py
=============================
Contract for the Asymmetric Governance Risk Shield (2026-06-12 refactor).

Design principle: negative ownership signals predict DISASTERS far better than
positive signals predict winners. Therefore:
  - Positive signals  → additive governance_bonus (engine), clipped [floor, 100]
  - Negative signals  → governance_risk_multiplier on composite_score (shield)

The four hard risk signals (each counts 1 toward gov_risk_count):
  1. Tier-2 dilution (dilution_flag == 2, 3-10% share dilution)
  2. Promoter 3Y systematic exit  (change_promoter_3y < -5)
  3. Promoter 2Y recent exit      (change_promoter_2y < -3 AND 3Y >= -5)
  4. Low + declining promoter     (promoter_holdings < 40 AND change_promoter_1y < 0)

Multiplier tiers (GOVERNANCE_RISK_MULTIPLIERS in config.py):
  0 signals → x1.00 | 1 → x0.92 | 2 → x0.82 | 3+ → x0.70

Run with: pytest tests/test_governance_asymmetric.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

import pandas as pd
import numpy as np
import pytest

from scoring_engine import compute_governance_bonus, compute_composite_score
from config import GOVERNANCE_RISK_MULTIPLIERS


def _mk(n: int = 4, **kwargs) -> pd.DataFrame:
    return pd.DataFrame({k: [v] * n for k, v in kwargs.items()})


def _clean_base(**overrides) -> dict:
    """A governance-clean stock: no risk signals firing."""
    base = dict(
        promoter_buying=0,
        change_fii_lq=0.0,
        change_dii_lq=0.0,
        inst_convergence=0,
        pledge_falling_1y=0,
        promoter_holdings=55.0,
        change_promoter_1y=0.0,
        change_promoter_2y=0.0,
        change_promoter_3y=0.0,
        fii_holdings=10.0,
        market_cap=10000.0,
        dilution_flag=0,
        insider_trading=np.nan,
    )
    base.update(overrides)
    return base


# ═══════════════════════════════════════════════════════════════════════════
# Risk count + multiplier mapping
# ═══════════════════════════════════════════════════════════════════════════

def test_clean_stock_multiplier_is_one():
    df = compute_governance_bonus(_mk(**_clean_base()))
    assert (df["gov_risk_count"] == 0).all()
    assert (df["governance_risk_multiplier"] == 1.0).all()


def test_tier2_dilution_alone_counts_one():
    df = compute_governance_bonus(_mk(**_clean_base(dilution_flag=2)))
    assert (df["gov_risk_count"] == 1).all()
    assert (df["governance_risk_multiplier"] == GOVERNANCE_RISK_MULTIPLIERS[1]).all()


def test_promoter_3y_exit_counts_one():
    df = compute_governance_bonus(_mk(**_clean_base(change_promoter_3y=-6.0)))
    assert (df["gov_risk_count"] == 1).all()
    assert (df["governance_risk_multiplier"] == GOVERNANCE_RISK_MULTIPLIERS[1]).all()


def test_2y_recent_exit_and_3y_exit_mutually_exclusive():
    """2Y early warning fires ONLY when 3Y has not yet crossed its threshold —
    the same promoter exit must never be double-counted."""
    # 3Y exit fired → 2Y warning suppressed → count stays 1
    df = compute_governance_bonus(
        _mk(**_clean_base(change_promoter_2y=-4.0, change_promoter_3y=-6.0))
    )
    assert (df["gov_risk_count"] == 1).all()
    # 3Y below threshold → 2Y warning fires alone → count 1
    df2 = compute_governance_bonus(
        _mk(**_clean_base(change_promoter_2y=-4.0, change_promoter_3y=-2.0))
    )
    assert (df2["gov_risk_count"] == 1).all()


def test_three_signals_floor_multiplier():
    """Dilution + 3Y exit + low-declining = 3 signals → harshest tier."""
    df = compute_governance_bonus(_mk(**_clean_base(
        dilution_flag=2,
        change_promoter_3y=-6.0,
        promoter_holdings=30.0,
        change_promoter_1y=-2.0,
    )))
    assert (df["gov_risk_count"] == 3).all()
    assert (df["governance_risk_multiplier"] == GOVERNANCE_RISK_MULTIPLIERS[3]).all()


def test_multiplier_tiers_are_monotonic_decreasing():
    tiers = [GOVERNANCE_RISK_MULTIPLIERS[k] for k in sorted(GOVERNANCE_RISK_MULTIPLIERS)]
    assert tiers[0] == 1.0, "Zero risk signals must mean NO penalty (x1.00)"
    assert all(a > b for a, b in zip(tiers, tiers[1:])), (
        "More risk signals must always mean a harsher multiplier"
    )
    assert tiers[-1] >= 0.5, "Governance multiplier must stay milder than forensic x0.50 floor"


# ═══════════════════════════════════════════════════════════════════════════
# Additive bonus is positive-conviction only (negatives moved to multiplier)
# ═══════════════════════════════════════════════════════════════════════════

def test_risk_signals_do_not_drag_additive_bonus():
    """The 4 hard risk signals act through the multiplier, NOT the additive bonus.
    A stock with heavy promoter exit + dilution but zero positives scores bonus ~0,
    while its multiplier carries the penalty."""
    df = compute_governance_bonus(_mk(**_clean_base(
        promoter_holdings=20.0,       # too low for alignment bonus
        change_promoter_1y=-3.0,
        change_promoter_2y=-4.0,
        change_promoter_3y=-6.0,      # systematic exit
        dilution_flag=2,              # tier-2 dilution
    )))
    assert (df["governance_bonus"] >= -5).all(), (
        "Hard risk signals must not stack into the additive bonus anymore — "
        "only the minor tier-1 ESOP deduction (-5) may remain additive."
    )
    assert (df["governance_risk_multiplier"] <= GOVERNANCE_RISK_MULTIPLIERS[2]).all(), (
        "The penalty must have moved to the multiplier (3 signals fired here)."
    )


def test_positive_bonus_unchanged_for_clean_dynasty_stock():
    """Dynasty-mode stock (promoter >= 60%, FII+DII buying): positives still stack."""
    df = compute_governance_bonus(_mk(**_clean_base(
        promoter_holdings=65.0,
        change_fii_lq=0.5,
        change_dii_lq=0.5,
    )))
    # high_alignment 15 + fii 15 + dii 10 = 40
    assert (df["governance_bonus"] == 40.0).all()
    assert (df["governance_risk_multiplier"] == 1.0).all()


# ═══════════════════════════════════════════════════════════════════════════
# Composite integration: multiplier actually bites, proportionally to conviction
# ═══════════════════════════════════════════════════════════════════════════

def _composite_frame(mult: float) -> pd.DataFrame:
    n = 12
    return pd.DataFrame({
        "quality_score":               [80.0] * n,
        "momentum_score":              [70.0] * n,
        "governance_bonus":            [30.0] * n,
        "governance_risk_multiplier":  [mult] * n,
        "pb_ratio":                    np.linspace(1.0, 8.0, n),
        "roe":                         np.linspace(5.0, 30.0, n),
        "rev_gr_5y":                   np.linspace(5.0, 25.0, n),
        "roce_med_5y":                 np.linspace(8.0, 28.0, n),
        "market_cap":                  np.linspace(500.0, 50000.0, n),
        "peg":                         [2.0] * n,
    })


def test_composite_scaled_by_governance_multiplier():
    clean = compute_composite_score(_composite_frame(1.0))
    risky = compute_composite_score(_composite_frame(GOVERNANCE_RISK_MULTIPLIERS[2]))
    ratio = risky["composite_score"].iloc[0] / clean["composite_score"].iloc[0]
    assert abs(ratio - GOVERNANCE_RISK_MULTIPLIERS[2]) < 1e-9, (
        f"Composite must scale by exactly the governance multiplier "
        f"(expected x{GOVERNANCE_RISK_MULTIPLIERS[2]}, got x{ratio:.4f})"
    )


def test_composite_missing_multiplier_column_defaults_to_one():
    """Synthetic frames without governance_risk_multiplier must not crash or penalize."""
    frame = _composite_frame(1.0).drop(columns=["governance_risk_multiplier"])
    out = compute_composite_score(frame)
    base = compute_composite_score(_composite_frame(1.0))
    assert abs(out["composite_score"].iloc[0] - base["composite_score"].iloc[0]) < 1e-9
