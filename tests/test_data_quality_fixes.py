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
import re

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
    # sorted() pins column order: _ALL_MAPPED_COLS is a set, so unsorted iteration varies with
    # PYTHONHASHSEED → column order varies → under Copy-on-Write the block layout (and whether the
    # fragmentation test trips pandas' 100-block threshold) became seed-dependent and flaky.
    missing = sorted(c for c in _ALL_MAPPED_COLS if c not in df.columns)
    if missing:   # one-shot concat (not column-by-column) so the fixture itself doesn't fragment
        df = pd.concat([df, pd.DataFrame(np.nan, index=df.index, columns=missing)], axis=1)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# VSTOP scale-guard diagnostic (Phase-1 audit finding A1)
# ═══════════════════════════════════════════════════════════════════════════

def test_vstop_scale_guard_excludes_missing_from_mismatch_count(capsys):
    """The VSTOP scale guard nullifies implausible (paise-scale) VSTOP values. A stock with
    a GENUINELY MISSING vstop must not be (a) counted or (b) announced as a 'scale mismatch'.
    Three stocks: missing / paise-scale (7684 vs 130 ≈ 59×) / normal (90 vs 100 = 0.9×)."""
    df = _frame(
        n=3,
        vstop_value=[np.nan, 7684.0, 90.0],
        close_price=[100.0, 130.0, 100.0],
    )
    out = compute_derived_signals(df)
    msg = capsys.readouterr().out

    assert pd.isna(out["vstop_value"].iloc[1]), "paise-scale VSTOP should be nullified"
    assert out["vstop_value"].iloc[2] == 90.0, "normal VSTOP should be kept"
    assert pd.isna(out["vstop_value"].iloc[0]), "missing VSTOP stays missing"

    m = re.search(r"nullified for (\d+) stocks", msg)
    assert m is not None, f"expected a VSTOP nullification message, got: {msg!r}"
    assert int(m.group(1)) == 1, (
        f"diagnostic must count ONLY the true scale mismatch (1), not the missing row; got {m.group(1)}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Frame-fragmentation hygiene (Phase-1 audit finding C1)
# ═══════════════════════════════════════════════════════════════════════════

def test_compute_derived_signals_emits_no_fragmentation_warning():
    """compute_derived_signals does 313 single-column inserts; once the block manager passes
    ~100 blocks pandas emits a `DataFrame is highly fragmented` PerformanceWarning on EVERY
    subsequent insert (~33k across the suite, and slow ingest). Periodic df.copy() defrag keeps
    it consolidated. Behavior is unchanged — guarded separately by the byte-identical census —
    so this test pins ONLY the fragmentation regression."""
    frame = _frame()   # build OUTSIDE the capture — fixture construction is not under test
    import warnings as _w
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        compute_derived_signals(frame)
    frag = [x for x in caught if "fragmented" in str(x.message).lower()]
    assert not frag, f"{len(frag)} DataFrame-fragmentation PerformanceWarning(s) still emitted"


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


# ═══════════════════════════════════════════════════════════════════════════
# Ind AS 116 / Debt Restatement Guard (plan item D4 — implemented, was untested)
# ═══════════════════════════════════════════════════════════════════════════

def test_debt_restatement_spike_flagged_and_slope_neutralised():
    """D/E spikes >2.5x after a stable year (lease capitalization pattern):
    flag fires AND de_slope_3y is neutralised to NaN (trend data unreliable)."""
    out = compute_derived_signals(_frame(
        debt_to_equity=1.5, debt_to_equity_1yb=0.5,
        debt_to_equity_2yb=0.5, debt_to_equity_3yb=0.5,
    ))
    assert (out["debt_restatement_suspected"] == 1).all()
    assert out["de_slope_3y"].isna().all(), (
        "de_slope_3y must be NaN when the spike is a suspected restatement"
    )


def test_debt_restatement_genuine_leveraging_not_flagged():
    """Gradual debt build-up (already rising in prior year) is GENUINE leveraging —
    not a restatement; the slope must survive."""
    out = compute_derived_signals(_frame(
        debt_to_equity=1.5, debt_to_equity_1yb=1.0,
        debt_to_equity_2yb=0.5, debt_to_equity_3yb=0.3,
    ))
    assert (out["debt_restatement_suspected"] == 0).all()
    assert not out["de_slope_3y"].isna().any()


def test_debt_restatement_deleveraging_not_flagged():
    """G5 inversion regression: a company AGGRESSIVELY PAYING DOWN debt (D/E dropping)
    must never be flagged — the original bug caught drops instead of spikes."""
    out = compute_derived_signals(_frame(
        debt_to_equity=0.2, debt_to_equity_1yb=0.8,
        debt_to_equity_2yb=0.9, debt_to_equity_3yb=1.0,
    ))
    assert (out["debt_restatement_suspected"] == 0).all()
    assert not out["de_slope_3y"].isna().any()


# ═══════════════════════════════════════════════════════════════════════════
# FIX 12 — Blue Chip Quality (Study 16): equity_shares unit contract
# Census 2026-06-12: the flag fired 0/2107. Root causes: (a) source CSV's
# Dividend Payout Ratio column broken (96% empty, rest negative) — DATA GAP,
# self-heals when the sheet formula is fixed; (b) the Weiss "≥ 5 million shares"
# screen was coded as `>= 5` against an ABSOLUTE share-count column (median
# ~51M) — vacuously true for every stock. These tests lock the corrected
# 5_000_000 absolute threshold and the dpr leg's missing-data behavior.
# ═══════════════════════════════════════════════════════════════════════════

_BLUE_CHIP_LEGS = dict(
    dividend_payout_ratio=30.0,           # Screen 1/2 lit (real data, once sheet fixed)
    roe_med_10y=20.0,                     # Screen 4 lit
    # consistency_champion inputs: smooth rising PAT, positive long CAGR
    pat=75.0, pat_1yb=70.0, pat_2yb=65.0, pat_3yb=60.0, pat_4yb=55.0, pat_5yb=50.0,
    pat_gr_5y=8.0,
)


def test_blue_chip_fires_with_all_legs_and_5m_shares():
    """All four screens lit + 6M absolute shares -> flag must fire."""
    out = compute_derived_signals(_frame(equity_shares=6_000_000.0, **_BLUE_CHIP_LEGS))
    assert (out["blue_chip_quality_flag"] == 1).all()


def test_blue_chip_share_count_is_absolute_not_millions():
    """1M absolute shares is below Weiss's 5M floor -> must NOT fire.
    Regression: the old `>= 5` threshold (written for a millions-unit column)
    passed every stock in the universe, making the screen vacuous."""
    out = compute_derived_signals(_frame(equity_shares=1_000_000.0, **_BLUE_CHIP_LEGS))
    assert (out["blue_chip_quality_flag"] == 0).all()


def test_blue_chip_missing_dpr_never_passes():
    """Missing dividend data -> dpr leg fails closed (fillna(0) < 20).
    Documents the live-data state: with the CSV DPR column broken, the flag
    is correctly 0 rather than firing on garbage."""
    legs = {k: v for k, v in _BLUE_CHIP_LEGS.items() if k != "dividend_payout_ratio"}
    out = compute_derived_signals(_frame(equity_shares=6_000_000.0, **legs))
    assert (out["blue_chip_quality_flag"] == 0).all()
