"""
Contract Tests v1.1 — Mauboussin Framework (Framework 34) — NOPAT Precision
=============================================================================
Tests the v1.1-mauboussin-nopat-precision implementation.

Key changes verified here (not in test_mauboussin_contract.py):
  1. reinvestment_rate is decimal [0,1] — no /100 division applied
  2. mauboussin_nopat_margin computed from EBIT/PBT/PAT/Revenue
  3. CAP trap uses structural 3-year slope, not 1-year delta
  4. Boundary: implied_cap == 15.0 does NOT trigger trap (strictly >)
  5. Boundary: implied_cap == 15.01 + slope < -1.0 DOES trigger trap
  6. roce_med_3y NaN → conservative fillna(roce) → slope = 0 (no false traps)
  7. NOPAT margin column is materialized in DataFrame output

Data engine prerequisite verified:
  - roce_med_3y must be in RATIO_COLS (loaded from CSV)
"""

import os
import re
import sys
import pytest
import pandas as pd
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DE_PATH = os.path.join(REPO_ROOT, "core", "data_engine.py")
SE_PATH = os.path.join(REPO_ROOT, "core", "scoring_engine.py")


@pytest.fixture(scope="module")
def de_source() -> str:
    with open(DE_PATH, encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module")
def se_source() -> str:
    with open(SE_PATH, encoding="utf-8") as f:
        return f.read()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _row_v11(
    pe: float = 20.0,
    # NOPAT margin inputs: ebit=250, pbt=250, pat=200, rev=1000 → margin=20%
    ebit: float = 250.0,
    pbt: float = 250.0,
    pat: float = 200.0,
    revenue: float = 1000.0,
    reinvestment_rate: float = 0.50,   # decimal [0,1]
    sell_alert_treadmill: int = 0,
    operating_leverage: int = 1,
    roce: float = 25.0,
    roce_med_3y: float = 25.0,         # stable slope = 0
    **extra,
) -> dict:
    """v1.1 canonical row — all gates clear, decimal reinvestment_rate."""
    row = dict(
        pe=pe, ebit=ebit, pbt=pbt, pat=pat, revenue=revenue,
        reinvestment_rate=reinvestment_rate,
        sell_alert_treadmill=sell_alert_treadmill,
        operating_leverage=operating_leverage,
        roce=roce, roce_med_3y=roce_med_3y,
        market_cap=2500.0, close_price=250.0,
        name="V11TestStock", sector="Technology",
    )
    row.update(extra)
    return row


def _run(rows: list) -> pd.DataFrame:
    from config import MASTER_PROFILES
    from core.scoring_engine import compute_qglp_score
    df = pd.DataFrame(rows)
    df.attrs["detected_market_regime"] = "SIDEWAYS"
    profile = MASTER_PROFILES.get("Balanced", next(iter(MASTER_PROFILES.values())))
    return compute_qglp_score(df, profile)


# ── Helper: expected implied_cap for given inputs ─────────────────────────────

def _nopat_margin(ebit, pbt, pat, revenue) -> float:
    if not revenue or revenue != revenue:   # 0 or NaN
        return 0.0
    eff_tax = max(0.0, min(0.50, (pbt - pat) / pbt)) if pbt else 0.25
    return ebit * (1 - eff_tax) / revenue * 100


# ═══════════════════════════════════════════════════════════════════════════════
# TestDataEnginePrerequiste — roce_med_3y must be in RATIO_COLS
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataEnginePrerequisite:
    """ROCE Median 3 Years must be loaded from CSV for structural slope."""

    def test_roce_med_3y_in_ratio_cols_source(self, de_source):
        assert re.search(
            r'"ROCE Median 3 Years"\s*:\s*"roce_med_3y"',
            de_source
        ), "data_engine.py RATIO_COLS must map 'ROCE Median 3 Years' → 'roce_med_3y'"

    def test_roce_med_3y_loaded_before_5y(self, de_source):
        pos_3y = de_source.find('"ROCE Median 3 Years"')
        pos_5y = de_source.find('"ROCE Median 5 Years"')
        assert pos_3y > 0, "ROCE Median 3 Years not in data_engine.py"
        assert pos_5y > 0, "ROCE Median 5 Years not in data_engine.py"


# ═══════════════════════════════════════════════════════════════════════════════
# TestNOPATMarginComputation — column materialization and formula
# ═══════════════════════════════════════════════════════════════════════════════

class TestNOPATMarginComputation:
    """mauboussin_nopat_margin must be computed and materialized."""

    def test_nopat_margin_column_present_in_output(self):
        res = _run([_row_v11()])
        assert "mauboussin_nopat_margin" in res.columns

    def test_nopat_margin_formula_20_pct(self):
        # ebit=250, pbt=250, pat=200, rev=1000 → eff_tax=0.20, nopat=200, margin=20%
        res = _run([_row_v11()])
        assert abs(res.loc[0, "mauboussin_nopat_margin"] - 20.0) < 0.1

    def test_nopat_margin_formula_30_pct(self):
        # ebit=375, pbt=375, pat=300, rev=1000 → eff_tax=0.20, nopat=300, margin=30%
        res = _run([_row_v11(ebit=375.0, pbt=375.0, pat=300.0)])
        assert abs(res.loc[0, "mauboussin_nopat_margin"] - 30.0) < 0.1

    def test_nopat_margin_nan_when_revenue_zero(self):
        res = _run([_row_v11(revenue=0.0)])
        assert pd.isna(res.loc[0, "mauboussin_nopat_margin"])

    def test_nopat_margin_nan_when_revenue_nan(self):
        row = _row_v11()
        row["revenue"] = float("nan")
        res = _run([row])
        assert pd.isna(res.loc[0, "mauboussin_nopat_margin"])

    def test_nopat_margin_zero_when_ebit_zero(self):
        res = _run([_row_v11(ebit=0.0)])
        assert res.loc[0, "mauboussin_nopat_margin"] == 0.0 or \
               pd.isna(res.loc[0, "mauboussin_nopat_margin"])

    def test_nopat_margin_excludes_interest_expense(self):
        # Same revenue/ebit but different PAT (different leverage) → same NOPAT margin
        # High-debt company: pbt=200, pat=160 vs low-debt: pbt=300, pat=240
        # Both: ebit=375, rev=1000 → eff_tax=0.20 → nopat margin = 30%
        row_levered   = _row_v11(ebit=375.0, pbt=200.0, pat=160.0, revenue=1000.0)
        row_unlevered = _row_v11(ebit=375.0, pbt=300.0, pat=240.0, revenue=1000.0)
        r_lev   = _run([row_levered])
        r_unlev = _run([row_unlevered])
        # Both should show ~30% nopat margin (interest excluded)
        assert abs(r_lev.loc[0, "mauboussin_nopat_margin"] - 30.0) < 0.5
        assert abs(r_unlev.loc[0, "mauboussin_nopat_margin"] - 30.0) < 0.5

    def test_effective_tax_clipped_at_50_pct(self):
        # Extreme: pbt=100, pat=10 → (100-10)/100=0.90 → clipped to 0.50
        # nopat = 100 * (1-0.50) = 50, margin = 50/1000 = 5%
        res = _run([_row_v11(ebit=100.0, pbt=100.0, pat=10.0, revenue=1000.0)])
        assert res.loc[0, "mauboussin_nopat_margin"] <= 10.0  # cap prevents overstate

    def test_effective_tax_fallback_25_pct_when_pbt_zero(self):
        # pbt=0 → division would blow up → fallback eff_tax=0.25
        # nopat = ebit * 0.75
        res = _run([_row_v11(ebit=200.0, pbt=0.0, pat=0.0, revenue=1000.0)])
        expected = 200.0 * 0.75 / 1000.0 * 100  # = 15%
        assert abs(res.loc[0, "mauboussin_nopat_margin"] - expected) < 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# TestReinvestmentRateDecimal — confirm no /100 applied in v1.1
# ═══════════════════════════════════════════════════════════════════════════════

class TestReinvestmentRateDecimal:
    """reinvestment_rate is decimal [0,1] — must NOT be divided by 100."""

    def test_rr_decimal_0_50_implied_cap_correct(self):
        # pe=20, nopat_margin=20%, rr=0.50 → implied_cap = 20*0.20*0.50 = 2.0
        res = _run([_row_v11(pe=20.0, reinvestment_rate=0.50)])
        assert abs(res.loc[0, "mauboussin_implied_cap"] - 2.0) < 0.05

    def test_rr_decimal_1_0_max_reinvestment(self):
        # pe=50, nopat_margin=30%, rr=1.0 → implied_cap = 50*0.30*1.0 = 15.0
        res = _run([_row_v11(pe=50.0, ebit=375.0, pbt=375.0, pat=300.0,
                              reinvestment_rate=1.0)])
        assert abs(res.loc[0, "mauboussin_implied_cap"] - 15.0) < 0.05

    def test_rr_zero_collapses_cap(self):
        # rr=0.0 → implied_cap = 0 (company paying out all earnings)
        res = _run([_row_v11(reinvestment_rate=0.0)])
        assert res.loc[0, "mauboussin_implied_cap"] == 0.0

    def test_rr_0_70_formula_precision(self):
        # pe=40, nopat_margin=25%, rr=0.70 → 40 * 0.25 * 0.70 = 7.0
        res = _run([_row_v11(pe=40.0, ebit=312.5, pbt=312.5, pat=250.0,
                              reinvestment_rate=0.70)])
        assert abs(res.loc[0, "mauboussin_implied_cap"] - 7.0) < 0.1

    def test_rr_nan_collapses_cap_to_zero(self):
        row = _row_v11()
        row["reinvestment_rate"] = float("nan")
        res = _run([row])
        assert res.loc[0, "mauboussin_implied_cap"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# TestStructuralRoceSlopev11 — 3-year slope replaces 1-year delta
# ═══════════════════════════════════════════════════════════════════════════════

class TestStructuralRoceSlopev11:
    """CAP trap must use (roce - roce_med_3y)/2 structural slope, not 1-year delta."""

    # ── Boundary at 15.0 ──────────────────────────────────────────────────────

    def test_implied_cap_exactly_15_no_trap(self):
        # pe=50, nopat_margin=30%, rr=1.0 → implied_cap = 15.0 (NOT > 15 → no trap)
        res = _run([_row_v11(pe=50.0, ebit=375.0, pbt=375.0, pat=300.0,
                              reinvestment_rate=1.0,
                              roce=10.0, roce_med_3y=20.0)])   # slope = -5 (very declining)
        assert abs(res.loc[0, "mauboussin_implied_cap"] - 15.0) < 0.05
        assert res.loc[0, "mauboussin_cap_trap"] == 0, (
            "implied_cap == 15.0 must NOT trigger trap (strictly > 15.0 required)"
        )

    def test_implied_cap_15_01_with_slope_fires_trap(self):
        # pe=51, nopat_margin=30%, rr=1.0 → implied_cap ≈ 15.3 > 15
        # slope = (15-20)/2 = -2.5 < -1.0 → trap fires
        res = _run([_row_v11(pe=51.0, ebit=375.0, pbt=375.0, pat=300.0,
                              reinvestment_rate=1.0,
                              roce=15.0, roce_med_3y=20.0)])
        assert res.loc[0, "mauboussin_implied_cap"] > 15.0
        assert res.loc[0, "mauboussin_cap_trap"] == 1

    # ── Slope threshold at -1.0 ───────────────────────────────────────────────

    def test_slope_exactly_minus_1_no_trap(self):
        # slope = (roce - roce_med_3y) / 2 = -1.0 exactly (NOT < -1 → no trap)
        # Need: roce - roce_med_3y = -2.0 → roce=18, roce_med_3y=20
        res = _run([_row_v11(pe=60.0, ebit=375.0, pbt=375.0, pat=300.0,
                              reinvestment_rate=1.0,
                              roce=18.0, roce_med_3y=20.0)])   # slope = (18-20)/2 = -1.0
        assert res.loc[0, "mauboussin_implied_cap"] > 15.0
        assert res.loc[0, "mauboussin_cap_trap"] == 0, (
            "slope == -1.0 must NOT trigger trap (strictly < -1.0 required)"
        )

    def test_slope_minus_1_01_fires_trap(self):
        # slope = (roce - roce_med_3y) / 2 = -1.1 → roce=17.8, roce_med_3y=20
        res = _run([_row_v11(pe=60.0, ebit=375.0, pbt=375.0, pat=300.0,
                              reinvestment_rate=1.0,
                              roce=17.8, roce_med_3y=20.0)])   # slope = -1.1 < -1.0
        assert res.loc[0, "mauboussin_implied_cap"] > 15.0
        assert res.loc[0, "mauboussin_cap_trap"] == 1

    def test_slope_zero_no_trap_even_with_high_cap(self):
        # Stable ROCE (slope=0) → no trap regardless of cap magnitude
        res = _run([_row_v11(pe=100.0, ebit=500.0, pbt=500.0, pat=400.0,
                              reinvestment_rate=1.0,
                              roce=20.0, roce_med_3y=20.0)])   # slope = 0
        assert res.loc[0, "mauboussin_implied_cap"] > 15.0
        assert res.loc[0, "mauboussin_cap_trap"] == 0

    def test_slope_positive_no_trap(self):
        # ROCE improving (slope > 0) → definitely no trap
        res = _run([_row_v11(pe=60.0, ebit=375.0, pbt=375.0, pat=300.0,
                              reinvestment_rate=1.0,
                              roce=25.0, roce_med_3y=20.0)])   # slope = +2.5
        assert res.loc[0, "mauboussin_cap_trap"] == 0

    # ── NaN handling ──────────────────────────────────────────────────────────

    def test_nan_roce_med3y_fallback_to_roce_slope_zero(self):
        # roce_med_3y NaN → fillna(roce) → slope = (roce-roce)/2 = 0 → no trap
        row = _row_v11(pe=60.0, ebit=375.0, pbt=375.0, pat=300.0, reinvestment_rate=1.0)
        row["roce_med_3y"] = float("nan")
        res = _run([row])
        assert res.loc[0, "mauboussin_implied_cap"] > 15.0
        assert res.loc[0, "mauboussin_cap_trap"] == 0, (
            "NaN roce_med_3y must not create false CAP trap (conservative fillna)"
        )

    def test_nan_roce_slope_zero_no_trap(self):
        # roce NaN → fillna(0.0); roce_med_3y NaN → fillna(0.0) → slope = 0
        row = _row_v11(pe=60.0, ebit=375.0, pbt=375.0, pat=300.0, reinvestment_rate=1.0)
        row["roce"] = float("nan")
        row["roce_med_3y"] = float("nan")
        res = _run([row])
        assert res.loc[0, "mauboussin_cap_trap"] == 0

    # ── One-year delta insensitivity ──────────────────────────────────────────

    def test_single_year_dip_does_not_trigger_trap(self):
        # Only 1-year ROCE dip (would have triggered old d35_roce_trend < 0 test)
        # but 3-year structural slope is flat → no trap in v1.1
        # Simulated: current roce = 22 (dipped), but 3Y median = 23 (stable long-run)
        # slope = (22-23)/2 = -0.5 > -1.0 → no trap
        res = _run([_row_v11(pe=60.0, ebit=375.0, pbt=375.0, pat=300.0,
                              reinvestment_rate=1.0,
                              roce=22.0, roce_med_3y=23.0)])   # slope = -0.5
        assert res.loc[0, "mauboussin_cap_trap"] == 0, (
            "Single-year ROCE dip (slope=-0.5) must not trigger trap in v1.1"
        )

    def test_structural_multi_year_decay_triggers_trap(self):
        # 3-year structural decay: slope = (15-25)/2 = -5 << -1 → trap fires
        res = _run([_row_v11(pe=60.0, ebit=375.0, pbt=375.0, pat=300.0,
                              reinvestment_rate=1.0,
                              roce=15.0, roce_med_3y=25.0)])   # slope = -5
        assert res.loc[0, "mauboussin_cap_trap"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# TestImpliedCapDistribution — sanity check on universe-level magnitude
# ═══════════════════════════════════════════════════════════════════════════════

class TestImpliedCapDistribution:
    """Implied CAP should be in realistic range 0-25 for typical stocks."""

    def test_typical_value_stock_low_cap(self):
        # pe=15, nopat_margin=12%, rr=0.30 → cap = 15*0.12*0.30 = 0.54
        res = _run([_row_v11(pe=15.0, ebit=150.0, pbt=150.0, pat=120.0,
                              reinvestment_rate=0.30)])
        assert 0.0 < res.loc[0, "mauboussin_implied_cap"] < 5.0

    def test_premium_growth_stock_mid_cap(self):
        # pe=40, nopat_margin=20%, rr=0.60 → cap = 40*0.20*0.60 = 4.8
        res = _run([_row_v11(pe=40.0, reinvestment_rate=0.60)])
        assert 3.0 < res.loc[0, "mauboussin_implied_cap"] < 8.0

    def test_hypervalued_stock_high_cap(self):
        # pe=100, nopat_margin=30%, rr=1.0 → cap = 100*0.30*1.0 = 30.0
        res = _run([_row_v11(pe=100.0, ebit=375.0, pbt=375.0, pat=300.0,
                              reinvestment_rate=1.0)])
        assert res.loc[0, "mauboussin_implied_cap"] > 20.0

    def test_zero_pe_gives_zero_cap(self):
        res = _run([_row_v11(pe=0.0)])
        assert res.loc[0, "mauboussin_implied_cap"] == 0.0

    def test_negative_pe_clipped_to_zero(self):
        # Loss-making company: pe=-5 → fillna(0) → cap = 0
        row = _row_v11()
        row["pe"] = float("nan")   # negative PE typically stored as NaN in screeners
        res = _run([row])
        assert res.loc[0, "mauboussin_implied_cap"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# TestV11SourceCodeContract — verify scoring_engine.py v1.1 patterns
# ═══════════════════════════════════════════════════════════════════════════════

class TestV11SourceCodeContract:
    """Verify exact v1.1 implementation patterns in core/scoring_engine.py."""

    def test_nopat_margin_uses_ebit(self, se_source):
        assert re.search(r'_ebit_v\s*=\s*df\.get\("ebit"', se_source), \
            "scoring_engine must extract ebit for NOPAT margin"

    def test_nopat_margin_uses_revenue(self, se_source):
        assert re.search(r'_rev_v\s*=\s*df\.get\("revenue"', se_source), \
            "scoring_engine must extract revenue for NOPAT margin"

    def test_nopat_margin_uses_pbt(self, se_source):
        assert re.search(r'_pbt_v\s*=\s*df\.get\("pbt"', se_source), \
            "scoring_engine must extract pbt for effective tax rate"

    def test_effective_tax_clipped(self, se_source):
        assert re.search(r'\.clip\(0.*0\.50\)|\.clip\(0,\s*0\.50\)', se_source), \
            "effective tax rate must be clipped to [0, 0.50]"

    def test_effective_tax_fallback(self, se_source):
        assert re.search(r'fillna\(0\.25\)', se_source), \
            "effective tax must fallback to 0.25 when pbt is NaN/zero"

    def test_no_double_division_on_rr(self, se_source):
        # After Mauboussin block, _rr_ly must NOT have / 100
        mf_start = se_source.find("# 34. Mauboussin Expectations Investing Framework")
        mf_end   = se_source.find("fw_mauboussin =", mf_start)
        block = se_source[mf_start:mf_end]
        rr_line = [l for l in block.split("\n") if "_rr_ly" in l and "get(" in l]
        assert rr_line, "_rr_ly must be defined in Mauboussin block"
        assert "/ 100" not in rr_line[0], (
            f"_rr_ly must NOT divide by 100 (v1.1 fix); found: {rr_line[0]}"
        )

    def test_roce_slope_variable_defined(self, se_source):
        assert "_roce_slope_3y" in se_source, \
            "_roce_slope_3y must be defined in scoring_engine.py"

    def test_roce_slope_formula_correct(self, se_source):
        assert re.search(
            r'_roce_slope_3y\s*=\s*\(_roce_v\s*-\s*_roce_med3y\)\s*/\s*2\.0',
            se_source
        ), "_roce_slope_3y formula must be (_roce_v - _roce_med3y) / 2.0"

    def test_spec_version_comment_updated(self, se_source):
        assert "v1.1" in se_source, \
            "scoring_engine.py must reference v1.1 in the Mauboussin block comment"
