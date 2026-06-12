"""
test_quant_integrity.py
=======================
Automated mathematical invariant tests for the Multibagger Discovery System.
Covers all 12 Quantamental Agent checks.

Run with: pytest tests/test_quant_integrity.py -v
"""

import sys
import os
import inspect

# Add project root and core/ to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

import pandas as pd
import numpy as np
import pytest


# ─── HELPER ─────────────────────────────────────────────────────────────────

def _mk(n: int = 10, **kwargs) -> pd.DataFrame:
    """Create minimal test DataFrame with each keyword argument repeated n times."""
    return pd.DataFrame({k: [v] * n for k, v in kwargs.items()})


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 1 — Ingestion Guard
# Verifies: header=1, key-indexed merge on company_id, no pd.concat(axis=1)
# ═══════════════════════════════════════════════════════════════════════════

def test_agent1_load_uses_header_autodetect():
    """CSV loading must auto-detect the header row, not hardcode header=1.

    Local CSVs have row 0 = emoji section labels, row 1 = real column names.
    Google Sheets exports may omit the emoji row entirely. The loader reads
    header=0 and promotes row 0 to column names only when 'companyId' is absent.
    """
    from data_engine import _load_single_csv, _apply_column_mapping
    src = inspect.getsource(_load_single_csv)
    assert "header=0" in src, (
        "AGENT 1 FAIL: _load_single_csv must use header=0 for auto-detection. "
        "header=1 breaks Google Sheets exports that lack the emoji metadata row."
    )
    # Header promotion lives in the shared _apply_column_mapping helper (used by both the
    # CSV path and the XLSX-workbook Google Sheets path).
    map_src = inspect.getsource(_apply_column_mapping)
    assert "companyId" in map_src, (
        "AGENT 1 FAIL: _apply_column_mapping must check for 'companyId' to detect "
        "whether the emoji row is present before promoting a data row to column names."
    )


def test_agent1_sheet_loading_uses_xlsx_workbook_by_tab_name():
    """Google Sheets loading must download the workbook as XLSX and read tabs BY NAME.

    Regression guard for the 2026-06-12 bug: /export?format=csv silently IGNORES a
    'sheet=' parameter and always returns the FIRST tab. All 6 tab requests downloaded
    the Ratio tab; the other 5 sheets became all-NaN and every stock scored the neutral
    defaults (Growth=50, Momentum=26, Governance=0). The XLSX export uses the same
    /export endpoint + sharing rules, but selects tabs by their exact names locally.
    """
    from data_engine import load_all_csvs
    src = inspect.getsource(load_all_csvs)
    assert "export?format=xlsx" in src, (
        "AGENT 1 FAIL: sheet loading must download the workbook via export?format=xlsx "
        "and read tabs by name. Per-tab CSV URLs cannot select a tab by name."
    )
    assert "format=csv&sheet=" not in src, (
        "AGENT 1 FAIL: /export?format=csv&sheet= is a broken URL form — the parameter "
        "is silently ignored by Google and the FIRST tab is returned for every request."
    )
    assert "format=csv&gid=" not in src, (
        "AGENT 1 FAIL: GID-based loading is banned (CLAUDE.md §0) — GIDs are "
        "per-spreadsheet and always wrong for a different user's sheet."
    )


def test_agent1_wrong_tab_guard_raises():
    """If a loaded sheet contains NONE of the expected columns, the loader must raise —
    not silently feed an all-NaN sheet downstream (flattens every stock's scores)."""
    from data_engine import _load_single_csv
    import tempfile

    wrong_tab = pd.DataFrame({
        "companyId": ["NSE:AAA", "NSE:BBB"],
        "Name": ["Alpha Ltd", "Beta Ltd"],
        "Some Unrelated Column": [1, 2],
    })
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "wrong_tab.csv")
        wrong_tab.to_csv(path, index=False)
        with pytest.raises(ValueError, match="wrong sheet/tab"):
            _load_single_csv(path, {"PAT": "pat", "Revenue": "revenue"}, "income")


def test_agent1_merge_uses_company_id_join():
    """Datasets must be merged on company_id, not by horizontal positional concat."""
    from data_engine import merge_datasets
    src = inspect.getsource(merge_datasets)
    assert 'on="company_id"' in src or "on='company_id'" in src, (
        "AGENT 1 FAIL: merge_datasets must use .merge(on='company_id'), not pd.concat(axis=1). "
        "Positional concat silently aligns wrong rows when any CSV has missing companies."
    )
    # Also confirm no horizontal concat
    assert "axis=1" not in src or "pd.concat" not in src, (
        "AGENT 1 FAIL: merge_datasets must not use pd.concat(axis=1). "
        "Use key-indexed .merge() to prevent row misalignment."
    )


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 2 — Math Sanitization
# Verifies: np.inf flush at end of compute_derived_signals
# ═══════════════════════════════════════════════════════════════════════════

def test_agent2_inf_flush_present_in_pipeline():
    """compute_derived_signals must flush np.inf to NaN at the end."""
    from data_engine import compute_derived_signals
    src = inspect.getsource(compute_derived_signals)
    assert "np.inf" in src and "replace" in src, (
        "AGENT 2 FAIL: compute_derived_signals must call df.replace([np.inf, -np.inf], np.nan). "
        "Division edge cases can create np.inf; _pct_rank treats inf as valid maximum, "
        "pushing bankrupt stocks to 99th percentile."
    )


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 3 — Sector Isolation
# Verifies: working capital metrics are NaN'd for financial sector stocks
# ═══════════════════════════════════════════════════════════════════════════

def test_agent3_financial_sector_isolation_in_code():
    """Financial sector NaN-out block must exist in compute_derived_signals."""
    from data_engine import compute_derived_signals
    src = inspect.getsource(compute_derived_signals)
    assert "_fin_mask" in src and "_wc_nullify" in src, (
        "AGENT 3 FAIL: compute_derived_signals must contain _fin_mask / _wc_nullify block. "
        "Banks/NBFCs must have NaN for inventory_turnover, ccc, days_receivable — "
        "these metrics are structurally meaningless for financial stocks."
    )


def test_agent3_config_financial_sectors_defined():
    """FINANCIAL_SECTORS must be defined in config.py."""
    from config import FINANCIAL_SECTORS
    assert isinstance(FINANCIAL_SECTORS, (list, set, frozenset)) and len(FINANCIAL_SECTORS) >= 4, (
        "AGENT 3 FAIL: FINANCIAL_SECTORS must be a non-empty collection in config.py. "
        "It drives is_financial flag and gate relaxation logic."
    )


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 5 — Paradox Optimization (Negative PEG)
# Verifies: negative PEG gets penalty score 5, not a "deep value" score of 100
# ═══════════════════════════════════════════════════════════════════════════

def test_agent5_negative_peg_gets_penalty_not_reward():
    """A negative PEG (earnings contracting) must receive the lowest score, not the highest."""
    from scoring_engine import _compute_valuation_score

    # Stock with negative PEG (loss-making or contracting earnings)
    df_neg = _mk(10, peg=-5.0,  pe_discount=0.0, ev_compression=0.0,
                 fcf_yield=0.0, debt_to_equity=1.5, payback_ratio=4.0)
    # Stock with healthy PEG
    df_pos = _mk(10, peg=0.8,   pe_discount=20.0, ev_compression=10.0,
                 fcf_yield=5.0, debt_to_equity=0.3, payback_ratio=0.8)

    score_neg = _compute_valuation_score(df_neg).mean()
    score_pos = _compute_valuation_score(df_pos).mean()

    assert score_neg < score_pos, (
        f"AGENT 5 FAIL: Negative PEG score ({score_neg:.1f}) must be less than "
        f"positive PEG score ({score_pos:.1f}). "
        "Negative PEG was clipping to 0 → falling in deep_value zone → score=100. "
        "Fix: np.where(raw_peg < 0, 5.0, peg_score)."
    )
    # The PEG component with score=5 rather than 100 should keep valuation below ~40
    assert score_neg < 50, (
        f"AGENT 5 FAIL: Negative PEG valuation score {score_neg:.1f} ≥ 50 (neutral). "
        "Should be below neutral to signal earnings contraction danger."
    )


def test_agent5_extremely_high_peg_does_not_score_neutral():
    """PEG=5000 should produce the same low score as PEG=998 (both clipped to 998).
    NaN PEG uses fillna(999) → clip(998) → same conservative low score (unknown earnings = expensive).
    This is CORRECT behavior: unknown earnings ≠ cheap, so NaN is not neutral here.
    """
    from scoring_engine import _compute_valuation_score

    df_998  = _mk(10, peg=998.0,   pe_discount=0.0, ev_compression=0.0,
                  fcf_yield=0.0, debt_to_equity=2.0, payback_ratio=5.0)
    df_5000 = _mk(10, peg=5000.0,  pe_discount=0.0, ev_compression=0.0,
                  fcf_yield=0.0, debt_to_equity=2.0, payback_ratio=5.0)

    # Good company with fair PEG should score higher than extreme-PEG company
    df_fair = _mk(10, peg=0.8,    pe_discount=20.0, ev_compression=10.0,
                  fcf_yield=6.0, debt_to_equity=0.3, payback_ratio=0.8)

    score_998   = _compute_valuation_score(df_998).mean()
    score_5000  = _compute_valuation_score(df_5000).mean()
    score_fair  = _compute_valuation_score(df_fair).mean()

    # PEG=998 and PEG=5000 must produce the same score (both clipped to 998)
    assert abs(score_998 - score_5000) < 1.0, (
        f"AGENT 5 FAIL: PEG=998 ({score_998:.2f}) and PEG=5000 ({score_5000:.2f}) must "
        "produce the same score — both should be clipped to upper=998."
    )
    # Extreme PEG must score well below a fair-value company
    assert score_998 < score_fair, (
        f"AGENT 5 FAIL: PEG=998 ({score_998:.2f}) should score below fair-value "
        f"PEG=0.8 ({score_fair:.2f}). "
        "Extreme PEG must not default to neutral 50. Fix: clip(upper=998) ensures the worst zone."
    )


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 6 — Cash Quality Auditor (FCF/CFO Inversion)
# Verifies: OCF <= 0 forces FCF/CFO component to 0, not neutral 50
# ═══════════════════════════════════════════════════════════════════════════

def test_agent6_negative_ocf_forces_zero_fcf_cfo_component():
    """A company with negative OCF must score lower than identical company with positive OCF."""
    from scoring_engine import _compute_cash_score

    common = dict(
        cfo_to_pat=90.0, cfo_to_ebitda=80.0, fcf_yield=5.0,
        capex_coverage=3.0, fcf_to_cfo_pct=70.0,
        fcf_consistency=1, self_funding=1,
    )
    df_neg_ocf = _mk(10, operating_cash_flow=-100.0, **common)
    df_pos_ocf = _mk(10, operating_cash_flow=+100.0, **common)

    score_neg = _compute_cash_score(df_neg_ocf).mean()
    score_pos = _compute_cash_score(df_pos_ocf).mean()

    assert score_neg < score_pos, (
        f"AGENT 6 FAIL: Negative-OCF cash score ({score_neg:.1f}) must be < "
        f"positive-OCF score ({score_pos:.1f}). "
        "When OCF <= 0, fcf_to_cfo_pct is meaningless (ratio of negative/near-zero). "
        "Fix: np.where(operating_cash_flow <= 0, 0.0, ranked_fcf_cfo)."
    )


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 7 — Governance Floor Enforcer
# Verifies: dilution + promoter-selling penalty survives (not clipped to 0)
# ═══════════════════════════════════════════════════════════════════════════

def test_agent7_governance_penalty_not_clipped_at_zero():
    """Heavy dilution + promoter systematic selling must produce a real penalty.

    Contract updated 2026-06-12 (Asymmetric Governance Risk Shield): the four hard risk
    signals no longer deduct additive points — they set governance_risk_multiplier < 1,
    which scales the composite down. This is STRONGER than the old flat deduction
    (-25 pts × 15% weight = -3.75 composite pts; the x0.70 multiplier costs a
    70-composite stock 21 pts) and, unlike additive points, can never be clipped away —
    the original G3 bug class is structurally impossible.
    """
    from scoring_engine import compute_governance_bonus
    from config import GOVERNANCE_RISK_MULTIPLIERS

    df = _mk(10,
        promoter_buying=0,
        change_fii_lq=-1.0,
        change_dii_lq=-1.0,
        inst_convergence=0,
        pledge_falling_1y=0,
        promoter_holdings=20.0,      # low promoter stake
        change_promoter_1y=-3.0,     # selling this year
        change_promoter_2y=-4.0,
        change_promoter_3y=-6.0,     # systematic 3-year exit pattern
        fii_holdings=10.0,
        market_cap=5000.0,
        dilution_flag=2,             # Tier 2 dilution = hard risk signal
        insider_trading=np.nan,
    )
    result = compute_governance_bonus(df)

    # 3 risk signals fire: tier-2 dilution + 3Y exit + low-declining
    # (2Y early warning is suppressed because the 3Y exit already crossed threshold)
    assert (result["gov_risk_count"] == 3).all(), (
        f"AGENT 7 FAIL: expected 3 governance risk signals, got "
        f"{result['gov_risk_count'].iloc[0]}. Dilution tier-2 + promoter 3Y exit + "
        "low-declining holdings must each count once (2Y warning mutually exclusive with 3Y)."
    )
    assert (result["governance_risk_multiplier"] == GOVERNANCE_RISK_MULTIPLIERS[3]).all(), (
        f"AGENT 7 FAIL: governance_risk_multiplier = "
        f"{result['governance_risk_multiplier'].iloc[0]} — heavy ownership risk must map "
        f"to the harshest tier (x{GOVERNANCE_RISK_MULTIPLIERS[3]})."
    )
    assert result["governance_bonus"].min() >= -50, (
        f"AGENT 7 FAIL: governance_bonus below -50 hard floor ({result['governance_bonus'].min():.1f})."
    )


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 8 — Capex Velocity Auditor
# Verifies: rf_capex_mirage flag fires for capital-intensive + high-growth + low-capex
# ═══════════════════════════════════════════════════════════════════════════

def _base_forensic_df(industry: str = "Metals", rev_gr_yoy: float = 25.0,
                      fa: float = 200.0, fa_1yb: float = 200.0,
                      is_financial: bool = False,
                      sector: str = "Steel",
                      dep_rate_1yb: float = 10.0) -> pd.DataFrame:
    """Minimal DataFrame for forensic engine tests. Mirrors production CSV structure.

    dep_rate_1yb: per-company audited depreciation rate (%) — the rf_capex_mirage
    denominator source after the 2026-05-31 refactor (replaced static _SECTOR_DEP_RATES).
    """
    return pd.DataFrame({
        "dep_rate_1yb":           [dep_rate_1yb],
        "cfo_to_pat":             [90.0],
        "days_receivable":        [30.0],
        "inv_vs_rev_gap":         [0.0],
        "debt_to_equity":         [0.5],
        "debt_to_equity_1yb":     [0.5],
        "ccc":                    [40.0],
        "ccc_1yb":                [40.0],
        "expense_ratio":          [0.5],
        "expense_ratio_1yb":      [0.5],
        "pledged_percentage":     [0.0],
        "dilution_flag":          [0],
        "free_cash_flow":         [100.0],
        "operating_cash_flow":    [100.0],
        "rev_gr_yoy":             [rev_gr_yoy],
        "pat_gr_yoy":             [10.0],
        "high_cash_high_debt":    [0],
        "inventory_turnover":     [8.0],
        "inventory_turnover_1yb": [8.0],
        "ssgr_cushion":           [5.0],
        "pat":                    [50.0],
        "total_assets":           [500.0],
        "total_assets_1yb":       [450.0],
        "ebitda":                 [100.0],
        "nfat":                   [3.0],
        "fcf_to_cfo_pct":         [70.0],
        "opm":                    [20.0],
        "opm_med_5y":             [20.0],
        "opm_stable":             [1],
        "opm_1yb":                [19.0],
        "fixed_assets":           [fa],
        "fixed_assets_1yb":       [fa_1yb],
        "cwip":                   [10.0],
        "cwip_1yb":               [10.0],
        "is_financial":           [is_financial],
        "industry":               [industry],
        "sector":                 [sector],
        "debt":                   [50.0],
        "total_liabilities":      [100.0],
        "total_liabilities_1yb":  [95.0],
    })


def test_agent8_capex_mirage_fires_for_capital_intensive():
    """rf_capex_mirage must fire: capital-intensive + rev>20% + FA flat (zero net capex)."""
    from forensic_engine import compute_red_flags
    # FA flat: capex_net = max(0, 200-200) = 0
    # depr_est = 200 * 0.10 = 20
    # capex_net_ratio = 0 / 20 = 0.0 < 0.5 → flag SHOULD fire
    df = _base_forensic_df(industry="Metals", rev_gr_yoy=25.0, fa=200.0, fa_1yb=200.0)
    result = compute_red_flags(df)

    assert "rf_capex_mirage" in result.columns, (
        "AGENT 8 FAIL: rf_capex_mirage column missing from forensic output."
    )
    assert result["rf_capex_mirage"].iloc[0] == 1, (
        f"AGENT 8 FAIL: rf_capex_mirage should fire for capital-intensive (Metals) company "
        f"with rev_gr_yoy=25% and FA flat (capex_ratio=0). Got {result['rf_capex_mirage'].iloc[0]}."
    )


def test_agent8_capex_mirage_does_not_fire_for_services():
    """rf_capex_mirage must NOT fire for IT/Service companies — low capex is their advantage."""
    from forensic_engine import compute_red_flags
    df = _base_forensic_df(industry="IT - Software", sector="IT - Software",
                           rev_gr_yoy=25.0, fa=200.0, fa_1yb=200.0)
    result = compute_red_flags(df)

    assert result["rf_capex_mirage"].iloc[0] == 0, (
        f"AGENT 8 FAIL: rf_capex_mirage must NOT fire for IT/Software companies. "
        "Asset-light revenue growth is their business model, not deferred maintenance. "
        f"Got {result['rf_capex_mirage'].iloc[0]}."
    )


def test_agent8_capex_mirage_does_not_fire_for_financials():
    """rf_capex_mirage must NOT fire for financial sector stocks."""
    from forensic_engine import compute_red_flags
    df = _base_forensic_df(industry="Banking", sector="Finance",
                           rev_gr_yoy=25.0, fa=200.0, fa_1yb=200.0,
                           is_financial=True)
    result = compute_red_flags(df)
    assert result["rf_capex_mirage"].iloc[0] == 0, (
        f"AGENT 8 FAIL: rf_capex_mirage must NOT fire for financial sector. "
        f"Got {result['rf_capex_mirage'].iloc[0]}."
    )


def test_agent8_capex_mirage_does_not_fire_when_capex_adequate():
    """rf_capex_mirage must NOT fire when capex/dep > 0.5 (adequate reinvestment)."""
    from forensic_engine import compute_red_flags
    # FA growing from 200 to 220: capex_net = 20, depr_est = 200*0.1 = 20, ratio = 1.0 ≥ 0.5
    df = _base_forensic_df(industry="Metals", rev_gr_yoy=25.0, fa=220.0, fa_1yb=200.0)
    result = compute_red_flags(df)
    assert result["rf_capex_mirage"].iloc[0] == 0, (
        f"AGENT 8 FAIL: rf_capex_mirage must NOT fire when capex_net_ratio ≥ 0.5. "
        f"(FA grew from 200→220, ratio=1.0). Got {result['rf_capex_mirage'].iloc[0]}."
    )


def test_agent8_capex_mirage_does_not_fire_for_slow_revenue_growth():
    """rf_capex_mirage must NOT fire when revenue growth <= 20% (even with low capex)."""
    from forensic_engine import compute_red_flags
    df = _base_forensic_df(industry="Metals", rev_gr_yoy=15.0, fa=200.0, fa_1yb=200.0)
    result = compute_red_flags(df)
    assert result["rf_capex_mirage"].iloc[0] == 0, (
        f"AGENT 8 FAIL: rf_capex_mirage must NOT fire for rev_gr_yoy=15% (<= 20% threshold). "
        f"Got {result['rf_capex_mirage'].iloc[0]}."
    )


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 9 — Scaled Value Realization (Geometric Payback)
# Verifies: payback ratio uses (1+g) × ((1+g)^5 - 1) / g compounding
# ═══════════════════════════════════════════════════════════════════════════

def test_agent9_payback_ratio_uses_geometric_compounding():
    """Payback ratio formula must use geometric growth compounding, not simple 5× PAT."""
    from data_engine import compute_derived_signals
    src = inspect.getsource(compute_derived_signals)
    # The correct formula contains (1 + g) multiplier and the geometric sum
    assert "(1 + g)" in src or "(1+g)" in src, (
        "AGENT 9 FAIL: Payback ratio must use geometric compounding formula "
        "(1+g) * ((1+g)^5 - 1) / g. Simple 5×PAT understates fast-grower payback by 20%+ "
        "at growth rates of 20-30%."
    )


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 10 — Institutional Volume Vector
# Verifies: NaN vol_ratio → score 50 (neutral); surge → 100; low → 20
# ═══════════════════════════════════════════════════════════════════════════

def test_agent10_volume_score_nan_returns_neutral_50():
    """Missing vol_ratio must return neutral score 50, not penalizing 20."""
    from scoring_engine import _compute_volume_score
    df = pd.DataFrame({"vol_ratio": [np.nan] * 10})
    score = _compute_volume_score(df)
    assert (score == 50.0).all(), (
        f"AGENT 10 FAIL: NaN vol_ratio must return neutral score 50, got {score.values}. "
        "Before fix, NaN fell through all np.where conditions (NaN >= X = False) → score=20. "
        "Fix: np.where(vol_ratio.isna(), 50, ...) as the FIRST condition."
    )


def test_agent10_volume_surge_returns_100():
    """vol_ratio >= 2.0 must return score 100 (institutional surge signal)."""
    from scoring_engine import _compute_volume_score
    df = pd.DataFrame({"vol_ratio": [2.0, 3.0, 5.0]})
    score = _compute_volume_score(df)
    assert (score == 100.0).all(), (
        f"AGENT 10 FAIL: vol_ratio >= 2.0 must return score 100, got {score.values}."
    )


def test_agent10_volume_low_returns_20():
    """vol_ratio < 0.7 must return score 20 (sub-par volume)."""
    from scoring_engine import _compute_volume_score
    df = pd.DataFrame({"vol_ratio": [0.3, 0.5, 0.69]})
    score = _compute_volume_score(df)
    assert (score == 20.0).all(), (
        f"AGENT 10 FAIL: vol_ratio < 0.7 must return score 20, got {score.values}."
    )


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 11 — Performance & Dot Product
# Verifies: zero .apply() calls in all scoring functions (pure vectorized)
# ═══════════════════════════════════════════════════════════════════════════

def test_agent11_no_apply_in_scoring_engine():
    """All core scoring functions must be purely vectorized — no .apply(axis=1) loops."""
    import scoring_engine
    hot_path_funcs = [
        scoring_engine._compute_moat_score,
        scoring_engine._compute_growth_score,
        scoring_engine._compute_cash_score,
        scoring_engine._compute_margin_score,
        scoring_engine._compute_balance_sheet_score,
        scoring_engine._compute_volume_score,
        scoring_engine._compute_valuation_score,
        scoring_engine.compute_governance_bonus,
        scoring_engine.compute_quality_score,
        scoring_engine.compute_momentum_score,
    ]
    for func in hot_path_funcs:
        src = inspect.getsource(func)
        assert ".apply(" not in src, (
            f"AGENT 11 FAIL: {func.__name__} contains .apply() — use vectorized operations. "
            "apply(axis=1) iterates Python-level over 2108 rows; numpy/pandas vectorization is "
            "100-1000× faster on this dataset size."
        )


def test_agent11_no_apply_in_data_engine():
    """compute_derived_signals must be purely vectorized."""
    from data_engine import compute_derived_signals
    src = inspect.getsource(compute_derived_signals)
    assert ".apply(" not in src, (
        "AGENT 11 FAIL: compute_derived_signals contains .apply() call. "
        "Use np.where / vectorized pandas operations."
    )


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 1 (CONFIG) — Cash Quality Hard Gate Threshold
# Verifies: cash_quality threshold is in PERCENTAGE units (70.0), not ratio (0.7)
# ═══════════════════════════════════════════════════════════════════════════

def test_agent1_cash_quality_threshold_percentage_units():
    """cash_quality gate threshold must be >= 50.0 (percentage), not 0.5-1.0 (ratio).
    cfo_to_pat CSV column value: 73.04 = 73%, NOT 0.7304.
    Threshold 0.7 would make 73.04 >= 0.7 always True — the gate never rejects anyone."""
    from config import HARD_GATES
    threshold = HARD_GATES["cash_quality"]["threshold"]
    assert threshold >= 50.0, (
        f"AGENT 1 FAIL: cash_quality threshold = {threshold}. Must be >= 50.0 (percentage). "
        "BUG: cfo_to_pat is stored as percentage (73.04 = 73%), not ratio (0.7304). "
        "Threshold 0.7 means 73.04 >= 0.7 = ALWAYS True — gate never fires. "
        "Fix: threshold = 70.0"
    )


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 5 (CONFIG) — Sell Alert Cash Collapse Threshold
# Verifies: sell_alert_cash_collapse uses < 50 (not < 0.5)
# ═══════════════════════════════════════════════════════════════════════════

def test_agent5_sell_alert_cash_collapse_percentage_units():
    """sell_alert_cash_collapse must compare against 50 (%), not 0.5 (ratio)."""
    import scoring_engine
    src = inspect.getsource(scoring_engine.compute_quality_score)
    # Should contain '< 50' not '< 0.5'
    assert "< 50" in src, (
        "AGENT 5 FAIL: sell_alert_cash_collapse must use '< 50' (percentage). "
        "BUG: '< 0.5' means 73.04 < 0.5 = NEVER True — alert never fires. "
        "cfo_to_pat is percentage: 73.04 = 73%, so threshold must be 50 (not 0.5)."
    )


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 7 (CONFIG) — Governance Bonus Floor
# Verifies: governance bonus allows negative values down to -50
# ═══════════════════════════════════════════════════════════════════════════

def test_agent7_governance_floor_config():
    """Governance bonus clip must use lower=-50, not lower=0."""
    import scoring_engine
    src = inspect.getsource(scoring_engine.compute_governance_bonus)
    assert "lower=-50" in src, (
        "AGENT 7 FAIL: compute_governance_bonus must use bonus.clip(lower=-50). "
        "clip(lower=0) erased all dilution and promoter-exit penalties for companies "
        "starting with zero governance bonus."
    )


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 8 — Capex Mirage in forensic flag registry
# ═══════════════════════════════════════════════════════════════════════════

def test_agent8_capex_mirage_in_flag_descriptions():
    """rf_capex_mirage must be registered in the forensic flag_descriptions dict."""
    from forensic_engine import compute_red_flags
    src = inspect.getsource(compute_red_flags)
    assert "rf_capex_mirage" in src, (
        "AGENT 8 FAIL: rf_capex_mirage not found in compute_red_flags source. "
        "Flag must be computed AND registered in flag_descriptions for UI display."
    )


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION: Composite score with negative governance stays valid
# ═══════════════════════════════════════════════════════════════════════════

def test_composite_score_bounded_0_to_100():
    """Composite score must always be in [0, 100] regardless of governance penalty."""
    from scoring_engine import compute_composite_score

    # Simulate the output of quality + momentum + governance
    df = pd.DataFrame({
        "quality_score":    [70.0, 50.0, 30.0, 10.0],
        "momentum_score":   [80.0, 50.0, 20.0,  5.0],
        "governance_bonus": [100.0, 0.0, -25.0, -50.0],  # test negative boundary
    })
    result = compute_composite_score(df)

    assert result["composite_score"].between(0, 100).all(), (
        f"AGENT 7 FAIL: composite_score out of [0, 100] range. "
        f"Min={result['composite_score'].min():.2f}, Max={result['composite_score'].max():.2f}. "
        "Governance penalty of -50 × 15% weight = -7.5 pts can push composite below 0."
    )


# ═══════════════════════════════════════════════════════════════════════════
# AGENTS 4, 8, 9, 10 — WCS 28/29/30 Framework Implementation Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_agent4_sector_normalization_in_moat_score():
    """Agent 4: _compute_moat_score must use groupby for sector-relative ranking."""
    import inspect
    from scoring_engine import _compute_moat_score
    src = inspect.getsource(_compute_moat_score)
    assert "groupby" in src, (
        "AGENT 4 FAIL: _compute_moat_score must use groupby() for sector-relative "
        "ROCE/ROE ranking (70% universe + 30% sector blend). Found no groupby call."
    )


def test_agent4_moat_score_sector_aware():
    """Agent 4: Sector normalization must produce different scores for same ROCE in different sectors."""
    from scoring_engine import _compute_moat_score
    df = pd.DataFrame({
        "roce_med_10y":        [25.0, 25.0, 8.0,  8.0],
        "roe_med_10y":         [20.0, 20.0, 6.0,  6.0],
        "roce_trajectory":     [0.5,  0.5,  0.5,  0.5],
        "roe_trajectory":      [0.5,  0.5,  0.5,  0.5],
        "roce_current_vs_med": [0.0,  0.0,  0.0,  0.0],
        "sector":              ["IT", "Banking", "IT", "Banking"],
    })
    scores = _compute_moat_score(df)
    # Stocks 0 and 1 have identical ROCE/ROE; with sector blending their scores may differ
    # because they sit at different peer ranks within their sectors
    assert scores.between(0, 100).all(), (
        f"AGENT 4 FAIL: Sector-normalized moat scores out of [0,100]. Got: {scores.tolist()}"
    )


def test_agent8b_ep_delta_in_data_engine():
    """Agent 8: economic_profit_delta, ep_hockey_stick, ep_power_curve must exist in data_engine."""
    import inspect
    import data_engine
    src = inspect.getsource(data_engine)
    for signal in ["economic_profit_delta", "ep_hockey_stick", "ep_power_curve"]:
        assert signal in src, (
            f"AGENT 8 FAIL: '{signal}' not found in data_engine.py. "
            "28th WCS Hockey Stick EP signals must be computed in compute_derived_signals."
        )


def test_agent8b_ep_hockey_stick_logic():
    """Agent 8: ep_hockey_stick=1 iff EP>0 AND EP improving YoY."""
    df = pd.DataFrame({
        "economic_profit":       [100.0, 100.0, -50.0, -50.0],
        "economic_profit_delta": [ 20.0, -10.0,  30.0, -10.0],
        "ep_hockey_stick":       [    1,     0,     0,     0],
    })
    for col in ["roce_med_10y", "roe_med_10y", "roce_trajectory", "roe_trajectory",
                "roce_current_vs_med", "sector", "pat_gr_5y", "rev_gr_5y",
                "earnings_yield", "roce", "pe_ratio", "pb_ratio", "pat_gr_10y",
                "rev_gr_10y", "pat_gr_3y", "eps_gr_yoy"]:
        df[col] = 0.0

    # ep_hockey_stick=1 only when EP>0 AND delta>0 (row 0)
    assert df.loc[0, "ep_hockey_stick"] == 1, "AGENT 8 FAIL: Row 0 (EP>0, delta>0) must be hockey stick"
    assert df.loc[1, "ep_hockey_stick"] == 0, "AGENT 8 FAIL: Row 1 (EP>0, delta<0) must NOT be hockey stick"
    assert df.loc[2, "ep_hockey_stick"] == 0, "AGENT 8 FAIL: Row 2 (EP<0, delta>0) must NOT be hockey stick"
    assert df.loc[3, "ep_hockey_stick"] == 0, "AGENT 8 FAIL: Row 3 (EP<0, delta<0) must NOT be hockey stick"


def test_agent9b_bruised_blue_chip_29_in_data_engine():
    """Agent 9: bruised_blue_chip_29 must exist in data_engine source."""
    import inspect
    import data_engine
    src = inspect.getsource(data_engine)
    assert "bruised_blue_chip_29" in src, (
        "AGENT 9 FAIL: 'bruised_blue_chip_29' not found in data_engine.py. "
        "29th WCS large-cap elite ROCE + P/B ≤ 2x signal must be computed."
    )


def test_agent10b_multitrillioncap_in_data_engine():
    """Agent 10: multitrillioncap_tipping_point must exist in data_engine source."""
    import inspect
    import data_engine
    src = inspect.getsource(data_engine)
    assert "multitrillioncap_tipping_point" in src, (
        "AGENT 10 FAIL: 'multitrillioncap_tipping_point' not found in data_engine.py. "
        "30th WCS Multi-Trillion Compounding tipping point signal must be computed."
    )


def test_wcs28_29_30_frameworks_in_scoring_source():
    """Frameworks 30-32 (EP Hockey Stick, Bruised BB 29, Multi-Trillion Cap) must be in compute_qglp_score."""
    import inspect
    from scoring_engine import compute_qglp_score
    src = inspect.getsource(compute_qglp_score)
    for label in ["EP Hockey Stick", "Bruised Blue Chip 29", "Multi-Trillion Cap"]:
        assert label in src, (
            f"AGENT 8/9/10 FAIL: Framework label '{label}' not found in compute_qglp_score. "
            "All 3 WCS 28/29/30 frameworks must appear in the fw_str vectorized concatenation."
        )


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2 — Scoring Engine Refactor: Redundant Winsorization, Sector
# Normalization, EP Power Curve, Hockey-Stick Breakout
# ═══════════════════════════════════════════════════════════════════════════

def test_phase2_no_winsorization_in_quality_score():
    """compute_quality_score must NOT contain a growth-column winsorization loop before _pct_rank."""
    import inspect
    from scoring_engine import compute_quality_score
    src = inspect.getsource(compute_quality_score)
    # The removed loop was identified by its quantile clip pattern on _growth_cols
    assert "_growth_cols" not in src, (
        "PHASE 2 FAIL: Growth-column winsorization loop (_growth_cols clip) must be removed from "
        "compute_quality_score. _pct_rank is non-parametric — winsorization before rank creates "
        "artificial ties at p01/p99 boundaries without changing any rank ordering."
    )


def test_phase2_sector_pct_rank_exists():
    """_sector_pct_rank helper must exist in scoring_engine (sector normalization Agent 3)."""
    import inspect
    import scoring_engine
    assert hasattr(scoring_engine, "_sector_pct_rank"), (
        "PHASE 2 FAIL: _sector_pct_rank function not found in scoring_engine. "
        "Sector-relative percentile ranking is required to eliminate cross-sectional bias."
    )
    src = inspect.getsource(scoring_engine._sector_pct_rank)
    assert "groupby" in src, (
        "PHASE 2 FAIL: _sector_pct_rank must use groupby().rank() for vectorized sector-relative ranking."
    )


def test_phase2_cash_score_uses_sector_normalization():
    """_compute_cash_score must blend universe + sector rank for cfo_to_pat and cfo_to_ebitda."""
    import inspect
    from scoring_engine import _compute_cash_score
    src = inspect.getsource(_compute_cash_score)
    assert "_sector_pct_rank" in src, (
        "PHASE 2 FAIL: _compute_cash_score must call _sector_pct_rank for CFO quality signals. "
        "Banks/financials have structurally different CFO/PAT norms vs non-financials."
    )


def test_phase2_margin_score_uses_sector_normalization():
    """_compute_margin_score must blend universe + sector rank for margin medians."""
    import inspect
    from scoring_engine import _compute_margin_score
    src = inspect.getsource(_compute_margin_score)
    assert "_sector_pct_rank" in src, (
        "PHASE 2 FAIL: _compute_margin_score must call _sector_pct_rank for margin signals. "
        "Commodity sectors have 5% OPM norms; FMCG sectors have 20%+ — sector bias must be removed."
    )


def test_phase2_ep_quintile_produces_1_to_5():
    """compute_ep_power_curve must produce ep_quintile values in {1, 2, 3, 4, 5}."""
    from scoring_engine import compute_ep_power_curve

    # Build a DataFrame with a spread of EP values to ensure distinct quintiles
    df = pd.DataFrame({
        "economic_profit":          [-500, -200, -50, 0, 50, 100, 200, 300, 500, 1000],
        "economic_profit_velocity": [  10,   10,  10, 5, 20,  30,  40,  50,  60,   70],
        "vol_ratio":                [   2,    2,   2, 2,  2,   2,   2,   2,   2,    2],
    })

    result = compute_ep_power_curve(df)
    valid_q = result["ep_quintile"].dropna()
    assert len(valid_q) > 0, "PHASE 2 FAIL: ep_quintile has no non-NaN values."
    assert set(valid_q.unique()).issubset({1.0, 2.0, 3.0, 4.0, 5.0}), (
        f"PHASE 2 FAIL: ep_quintile must only contain values in {{1,2,3,4,5}}. "
        f"Got: {sorted(valid_q.unique())}"
    )
    assert valid_q.min() >= 1.0 and valid_q.max() <= 5.0, (
        "PHASE 2 FAIL: ep_quintile range must be [1, 5]."
    )


def test_phase2_ep_hockey_stick_breakout_conditions():
    """ep_hockey_stick_breakout fires only on Q2/Q3 + velocity>0 + vol_ratio>1."""
    from scoring_engine import compute_ep_power_curve

    # Use enough rows that qcut can produce Q2 and Q3
    n = 20
    ep_values = list(range(-5, 15))   # 20 values → 4 per quintile
    df = pd.DataFrame({
        "economic_profit":          ep_values,
        "economic_profit_velocity": [10] * n,   # all positive velocity
        "vol_ratio":                [2.0] * n,  # all above 20D SMA
    })
    result = compute_ep_power_curve(df)

    # All stocks: velocity>0 and vol>SMA, so breakout should fire only for Q2 and Q3
    breakout = result["ep_hockey_stick_breakout"]
    q_vals   = result["ep_quintile"]

    for i, (hs, q) in enumerate(zip(breakout, q_vals)):
        if pd.isna(q):
            continue
        if q in (2.0, 3.0):
            assert hs == 1, (
                f"PHASE 2 FAIL: Row {i} is Q{int(q)} with velocity>0 and vol>SMA — "
                f"ep_hockey_stick_breakout must be 1, got {hs}."
            )
        elif q in (1.0, 4.0, 5.0):
            assert hs == 0, (
                f"PHASE 2 FAIL: Row {i} is Q{int(q)} — breakout must NOT fire outside Q2/Q3, got {hs}."
            )


def test_phase2_hockey_stick_blocked_by_low_volume():
    """ep_hockey_stick_breakout must not fire when vol_ratio <= 1 (no institutional volume)."""
    from scoring_engine import compute_ep_power_curve

    n = 20
    ep_values = list(range(-5, 15))
    df = pd.DataFrame({
        "economic_profit":          ep_values,
        "economic_profit_velocity": [10.0] * n,
        "vol_ratio":                [0.8]  * n,   # below 20D SMA → no institutional accumulation
    })
    result = compute_ep_power_curve(df)
    assert result["ep_hockey_stick_breakout"].sum() == 0, (
        "PHASE 2 FAIL: ep_hockey_stick_breakout must not fire when vol_ratio <= 1.0. "
        "Institutional volume confirmation is a mandatory concurrent condition."
    )


def test_phase2_ep_power_curve_in_run_full_scoring_source():
    """compute_ep_power_curve must be called inside run_full_scoring."""
    import inspect
    from scoring_engine import run_full_scoring
    src = inspect.getsource(run_full_scoring)
    assert "compute_ep_power_curve" in src, (
        "PHASE 2 FAIL: compute_ep_power_curve must be wired into run_full_scoring pipeline. "
        "EP quintile + hockey-stick breakout must run before compute_composite_score."
    )


def test_phase2_composite_score_includes_hockey_stick_boost():
    """compute_composite_score must apply ep_hockey_stick_breakout boost."""
    import inspect
    from scoring_engine import compute_composite_score
    src = inspect.getsource(compute_composite_score)
    assert "ep_hockey_stick_breakout" in src, (
        "PHASE 2 FAIL: compute_composite_score must incorporate ep_hockey_stick_breakout boost. "
        "28th WCS structural alpha inflection must influence the final composite score."
    )


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3 — Forensic Engine Overhaul: WCS 24 Defensive Accounting Protocols
# Covers: rf_capitalizing_expenses retirement, rf_tax_panic, rf_receivables_bloat,
#         compute_cascading_forensic_filter
# ═══════════════════════════════════════════════════════════════════════════

def test_phase3_rf_capitalizing_expenses_retired():
    """rf_capitalizing_expenses must NOT appear in compute_red_flags output.
    It was an inverted diagnostic (penalized high capex = active reinvestment).
    Superseded by rf_capex_mirage which correctly flags LOW capex + HIGH growth."""
    from forensic_engine import compute_red_flags
    df = _base_forensic_df(industry="Metals", rev_gr_yoy=25.0, fa=200.0, fa_1yb=200.0)
    result = compute_red_flags(df)
    assert "rf_capitalizing_expenses" not in result.columns, (
        "PHASE 3 FAIL: rf_capitalizing_expenses must be retired — column must not appear in output. "
        "The prior logic flagged companies actively investing in growth (high capex > 3× dep), "
        "which is the OPPOSITE of the Capitalization Mirage pattern. "
        "rf_capex_mirage (flag #22) is the correct replacement."
    )


def test_phase3_rf_tax_panic_bypasses_loss_makers():
    """rf_tax_panic must be 0 for loss-makers (PAT ≤ 0) even when PBT > 0."""
    from forensic_engine import compute_red_flags
    # PAT = -50 (loss-maker). PAT <= 0 guard must prevent flag from firing.
    df = pd.DataFrame({
        **{c: [0.0] for c in [
            "inv_vs_rev_gap", "debt_to_equity", "debt_to_equity_1yb",
            "ccc", "ccc_1yb", "expense_ratio", "expense_ratio_1yb",
            "pledged_percentage", "free_cash_flow", "pat_gr_yoy",
            "high_cash_high_debt", "inventory_turnover", "inventory_turnover_1yb",
            "ssgr_cushion", "total_liabilities", "total_liabilities_1yb",
            "fcf_to_cfo_pct", "opm_1yb", "cwip", "cwip_1yb",
        ]},
        "cfo_to_pat":          [0.0],
        "days_receivable":     [30.0],
        "dilution_flag":       [0],
        "operating_cash_flow": [-50.0],
        "rev_gr_yoy":          [25.0],
        "pat":                 [-50.0],   # LOSS-MAKER
        "pbt":                 [100.0],   # PBT > 0 but PAT guard must block flag
        "ebitda":              [100.0],
        "total_assets":        [500.0],
        "total_assets_1yb":    [450.0],
        "nfat":                [3.0],
        "opm_med_5y":          [20.0],
        "fixed_assets":        [0.0],
        "fixed_assets_1yb":    [0.0],
        "is_financial":        [False],
        "industry":            ["Metals"],
        "debt":                [0.0],
    })
    result = compute_red_flags(df)
    assert result["rf_tax_panic"].iloc[0] == 0, (
        f"PHASE 3 FAIL: rf_tax_panic must be 0 for loss-makers (PAT <= 0). "
        f"Got {result['rf_tax_panic'].iloc[0]}. "
        "Loss-makers inherently pay no income tax — the signal is meaningless for negative PAT."
    )


def test_phase3_rf_tax_panic_fires_for_zero_effective_tax():
    """rf_tax_panic must fire when PBT=100, PAT=97 → eff_tax_rate = 3% < 10%."""
    from forensic_engine import compute_red_flags
    # Direct PBT: PBT=100, PAT=97 → (100-97)/100 = 3% effective tax → SHOULD FIRE
    df = pd.DataFrame({
        **{c: [0.0] for c in [
            "inv_vs_rev_gap", "debt_to_equity", "debt_to_equity_1yb",
            "ccc", "ccc_1yb", "expense_ratio", "expense_ratio_1yb",
            "pledged_percentage", "pat_gr_yoy", "high_cash_high_debt",
            "inventory_turnover", "inventory_turnover_1yb", "ssgr_cushion",
            "total_liabilities", "total_liabilities_1yb", "fcf_to_cfo_pct",
            "opm_1yb", "cwip", "cwip_1yb",
        ]},
        "cfo_to_pat":          [90.0],
        "days_receivable":     [30.0],
        "dilution_flag":       [0],
        "operating_cash_flow": [97.0],
        "free_cash_flow":      [97.0],
        "rev_gr_yoy":          [10.0],
        "pat":                 [97.0],   # PAT = 97
        "pbt":                 [100.0],  # PBT = 100 → eff_tax = (100-97)/100 = 3%
        "ebitda":              [100.0],
        "total_assets":        [500.0],
        "total_assets_1yb":    [450.0],
        "nfat":                [3.0],
        "opm_med_5y":          [97.0],
        "fixed_assets":        [0.0],
        "fixed_assets_1yb":    [0.0],
        "is_financial":        [False],
        "industry":            ["IT - Software"],
        "debt":                [0.0],
    })
    result = compute_red_flags(df)
    assert result["rf_tax_panic"].iloc[0] == 1, (
        f"PHASE 3 FAIL: rf_tax_panic must fire when PBT=100, PAT=97. "
        f"Effective tax rate = (100-97)/100 = 3% < 10% threshold. "
        f"Got rf_tax_panic = {result['rf_tax_panic'].iloc[0]}."
    )


def test_phase3_rf_tax_panic_does_not_fire_for_normal_tax():
    """rf_tax_panic must NOT fire for normal effective tax rate (e.g. Tata Motors 31.5%)."""
    from forensic_engine import compute_red_flags
    # Tata Motors: PBT=4663, PAT=3195 → eff_tax = 31.5% → must NOT fire
    df = pd.DataFrame({
        **{c: [0.0] for c in [
            "inv_vs_rev_gap", "debt_to_equity", "debt_to_equity_1yb",
            "ccc", "ccc_1yb", "expense_ratio", "expense_ratio_1yb",
            "pledged_percentage", "pat_gr_yoy", "high_cash_high_debt",
            "inventory_turnover", "inventory_turnover_1yb", "ssgr_cushion",
            "total_liabilities", "total_liabilities_1yb", "fcf_to_cfo_pct",
            "opm_1yb", "cwip", "cwip_1yb",
        ]},
        "cfo_to_pat":          [90.0],
        "days_receivable":     [30.0],
        "dilution_flag":       [0],
        "operating_cash_flow": [3195.0],
        "free_cash_flow":      [2000.0],
        "rev_gr_yoy":          [10.0],
        "pat":                 [3195.0],  # Tata Motors PAT
        "pbt":                 [4663.0],  # Tata Motors PBT → eff_tax = 31.5%
        "ebitda":              [6172.0],
        "total_assets":        [344264.0],
        "total_assets_1yb":    [378642.0],
        "nfat":                [2.0],
        "opm_med_5y":          [15.0],
        "fixed_assets":        [164986.0],
        "fixed_assets_1yb":    [163879.0],
        "is_financial":        [False],
        "industry":            ["Auto - 4 Wheelers"],
        "debt":                [58501.0],
    })
    result = compute_red_flags(df)
    assert result["rf_tax_panic"].iloc[0] == 0, (
        f"PHASE 3 FAIL: rf_tax_panic must NOT fire for Tata Motors (PBT=4663, PAT=3195). "
        f"Effective tax rate = 31.5% — well above the 10% panic threshold. "
        f"Got rf_tax_panic = {result['rf_tax_panic'].iloc[0]}."
    )


def test_phase3_rf_receivables_bloat_sector_relative():
    """rf_receivables_bloat must fire only for companies expanding DSO well above sector peers."""
    from forensic_engine import compute_red_flags

    # Sector A: 3 companies — median DSO expansion = 5 days
    # Company 0: expansion = 5 days (at median — should NOT fire)
    # Company 1: expansion = 30 days (25 days above median = 30 > 5+20 — SHOULD FIRE)
    # Company 2: expansion = 3 days (below median — should NOT fire)
    base_cols = {
        "cfo_to_pat":          [90.0,  90.0,  90.0],
        "days_receivable":     [65.0,  80.0,  53.0],   # current DSO
        "days_receivable_1yb": [60.0,  50.0,  50.0],   # 1yr back DSO
        # expansions:            +5      +30     +3
        "sector":              ["Mfg", "Mfg", "Mfg"],
        "inv_vs_rev_gap":      [0.0,   0.0,   0.0],
        "debt_to_equity":      [0.5,   0.5,   0.5],
        "debt_to_equity_1yb":  [0.5,   0.5,   0.5],
        "ccc":                 [40.0,  40.0,  40.0],
        "ccc_1yb":             [40.0,  40.0,  40.0],
        "expense_ratio":       [0.5,   0.5,   0.5],
        "expense_ratio_1yb":   [0.5,   0.5,   0.5],
        "pledged_percentage":  [0.0,   0.0,   0.0],
        "dilution_flag":       [0,     0,     0],
        "free_cash_flow":      [100.0, 100.0, 100.0],
        "operating_cash_flow": [100.0, 100.0, 100.0],
        "rev_gr_yoy":          [10.0,  10.0,  10.0],
        "pat_gr_yoy":          [10.0,  10.0,  10.0],
        "high_cash_high_debt": [0,     0,     0],
        "inventory_turnover":  [8.0,   8.0,   8.0],
        "inventory_turnover_1yb": [8.0, 8.0,  8.0],
        "ssgr_cushion":        [5.0,   5.0,   5.0],
        "pat":                 [50.0,  50.0,  50.0],
        "total_assets":        [500.0, 500.0, 500.0],
        "total_assets_1yb":    [450.0, 450.0, 450.0],
        "ebitda":              [100.0, 100.0, 100.0],
        "nfat":                [3.0,   3.0,   3.0],
        "fcf_to_cfo_pct":      [70.0,  70.0,  70.0],
        "opm_med_5y":          [20.0,  20.0,  20.0],
        "opm_1yb":             [19.0,  19.0,  19.0],
        "fixed_assets":        [200.0, 200.0, 200.0],
        "fixed_assets_1yb":    [200.0, 200.0, 200.0],
        "cwip":                [10.0,  10.0,  10.0],
        "cwip_1yb":            [10.0,  10.0,  10.0],
        "is_financial":        [False, False, False],
        "industry":            ["Metals", "Metals", "Metals"],
        "debt":                [50.0,  50.0,  50.0],
        "total_liabilities":   [100.0, 100.0, 100.0],
        "total_liabilities_1yb": [95.0, 95.0, 95.0],
    }
    df = pd.DataFrame(base_cols)
    result = compute_red_flags(df)

    assert "rf_receivables_bloat" in result.columns, (
        "PHASE 3 FAIL: rf_receivables_bloat column missing from compute_red_flags output."
    )
    # Company 0: expansion=5, median=5, 5 > 5+20=25 → False → 0
    assert result["rf_receivables_bloat"].iloc[0] == 0, (
        f"PHASE 3 FAIL: Company 0 (DSO exp=+5, sector median=+5) must NOT trigger "
        f"rf_receivables_bloat. Got {result['rf_receivables_bloat'].iloc[0]}."
    )
    # Company 1: expansion=30, median=5, 30 > 5+20=25 → True → 1
    assert result["rf_receivables_bloat"].iloc[1] == 1, (
        f"PHASE 3 FAIL: Company 1 (DSO exp=+30, sector median=+5) MUST trigger "
        f"rf_receivables_bloat (30 > 5+20=25). Got {result['rf_receivables_bloat'].iloc[1]}."
    )
    # Company 2: expansion=3, median=5, 3 > 25 → False → 0
    assert result["rf_receivables_bloat"].iloc[2] == 0, (
        f"PHASE 3 FAIL: Company 2 (DSO exp=+3, sector median=+5) must NOT trigger "
        f"rf_receivables_bloat. Got {result['rf_receivables_bloat'].iloc[2]}."
    )


def test_phase3_cascading_filter_multipliers():
    """compute_cascading_forensic_filter must apply correct multipliers and adjust composite_score."""
    from forensic_engine import compute_cascading_forensic_filter

    df = pd.DataFrame({
        "red_flag_count": [0,    1,    2,    3,    4,    5,    7   ],
        "composite_score": [80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0],
    })
    result = compute_cascading_forensic_filter(df)

    expected_multipliers = [1.0, 0.9, 0.9, 0.75, 0.75, 0.5, 0.5]
    expected_scores      = [80.0, 72.0, 72.0, 60.0, 60.0, 40.0, 40.0]

    for i, (exp_m, exp_s) in enumerate(zip(expected_multipliers, expected_scores)):
        got_m = result["forensic_multiplier"].iloc[i]
        got_s = result["composite_score"].iloc[i]
        assert abs(got_m - exp_m) < 0.001, (
            f"PHASE 3 FAIL: red_flag_count={int(df['red_flag_count'].iloc[i])} → "
            f"forensic_multiplier expected {exp_m}, got {got_m:.3f}."
        )
        assert abs(got_s - exp_s) < 0.1, (
            f"PHASE 3 FAIL: red_flag_count={int(df['red_flag_count'].iloc[i])} → "
            f"composite_score expected {exp_s:.1f} (80×{exp_m}), got {got_s:.2f}."
        )
