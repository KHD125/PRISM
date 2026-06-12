"""
Contract Tests — Forensic Engine Sector-Aware Lookup (BUG #11) + Capex Mirage
=============================================================================
Regression-hardening suite for the sector-specific vectorized DSO mapping in
core/forensic_engine.py, plus the per-company capex-mirage depreciation logic:

  BUG #11 Fix — _SECTOR_DSO_THRESHOLDS:
      rf_high_receivables now uses per-sector DSO alert thresholds instead of
      the old binary is_service → (120 or 75) pattern. This suite verifies that:
        • IT / Construction stocks with DSO=75 do NOT trigger false positives
        • FMCG stocks with DSO=50 correctly trigger (threshold=45)
        • Every sector boundary is tested at threshold-1, threshold, threshold+1
        • NaN DSO always produces 0 regardless of sector

  rf_capex_mirage (Flag 22) — per-company audited depreciation:
      The depreciation denominator now reads each company's audited dep_rate_1yb
      column (data_engine accounting identity), NOT a static per-sector dep dict.
      This suite verifies:
        • A power utility's audited rate (4%) does not trigger on mild FA growth
        • A telecom's audited rate (18%) correctly triggers on revenue surge + low capex
        • Default companies (10%) behave at the documented capex-ratio boundaries

Structure:
    TestSectorDSODictContract     — dict keys match live CSV sector names, value types correct
    TestDSOThresholdBoundaries    — boundary tests at threshold-1, threshold, threshold+1 per sector
    TestDSOFalsePositiveSuppression — IT/Healthcare/Construction at 75d → zero flags
    TestDSOTruePositiveActivation — FMCG/Steel at above-threshold DSO → flag=1
    TestCapexMirageDepRateLogic   — per-company dep_rate_1yb propagates correctly
    TestSectorColumnMissing       — safe fallback when sector column absent entirely

All tests use compute_red_flags() directly — no full pipeline overhead.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# ── Path bootstrap ────────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.forensic_engine import (
    compute_red_flags,
    _SECTOR_DSO_THRESHOLDS,
    _DEFAULT_DSO_THRESHOLD,
)

# ── CSV sector master list (authoritative 84-sector source) ───────────────────
_CSV_SECTOR_NAMES = frozenset({
    "Aerospace & Defence", "Agro Chemicals", "Air Transport Service",
    "Alcoholic Beverages", "Auto Ancillaries", "Automobile",
    "Banks", "Bearings", "Cables",
    "Capital Goods - Electrical Equipment", "Capital Goods-Non Electrical Equipment",
    "Castings, Forgings & Fastners", "Cement", "Cement - Products",
    "Ceramic Products", "Chemicals", "Computer Education",
    "Construction", "Consumer Durables", "Co-Working",
    "Credit Rating Agencies", "Crude Oil & Natural Gas",
    "Diamond, Gems and Jewellery", "Diversified", "Dry cells",
    "E-Commerce/App based Aggregator", "Edible Oil", "Education",
    "Electronics", "Engineering", "Entertainment",
    "Ferro Alloys", "Fertilizers", "Finance",
    "Financial Services", "FMCG", "Gas Distribution",
    "Glass & Glass Products", "Healthcare", "Hotels & Restaurants",
    "Infrastructure Developers & Operators", "Infrastructure Investment Trusts",
    "Insurance", "IT - Hardware", "IT - Software",
    "Leather", "Logistics", "Marine Port & Services",
    "Media - Print/Television/Radio", "Mining & Mineral products", "Miscellaneous",
    "Non Ferrous Metals", "Oil Drill/Allied", "Packaging",
    "Paints/Varnish", "Paper", "Petrochemicals",
    "Pharmaceuticals", "Plantation & Plantation Products", "Plastic products",
    "Plywood Boards/Laminates", "Power Generation & Distribution",
    "Power Infrastructure", "Printing & Stationery", "Quick Service Restaurant",
    "Railways", "Readymade Garments/ Apparells", "Real Estate Investment Trusts",
    "Realty", "Refineries", "Refractories",
    "Retail", "Ship Building", "Shipping",
    "Steel", "Stock/ Commodity Brokers", "Sugar",
    "Telecom Equipment & Infra Services", "Telecom-Handsets/Mobile",
    "Telecom-Service", "Textiles", "Tobacco Products",
    "Trading", "Tyres",
})


# ── Minimal row builder for compute_red_flags ─────────────────────────────────

def _base_row(**overrides) -> dict:
    """Build a minimal row dict that satisfies all column guards in compute_red_flags."""
    base = {
        # Identity
        "sector":                   "Textiles",   # safe default — not in any special map
        # Receivables
        "days_receivable":          50.0,
        "days_receivable_1yb":      48.0,
        # CFO/PAT
        "cfo_to_pat":               85.0,
        # Inventory
        "inv_vs_rev_gap":           0.0,
        # Debt
        "debt_to_equity":           0.3,
        "debt_to_equity_1yb":       0.3,
        # Liquidity / CCC
        "current_ratio":            2.0,
        "current_ratio_1yb":        2.0,
        "ccc":                      30.0,
        "ccc_1yb":                  30.0,
        # Growth
        "rev_gr_yoy":               10.0,
        "pat_gr_yoy":               10.0,
        "ocf_growth":               10.0,
        "pat_gr_5y":                10.0,
        # Governance
        "pledged_percentage":       2.0,
        "equity_shares":            1000.0,
        "equity_shares_1yb":        1000.0,
        "dilution_flag":            0,
        # Cash flow
        "operating_cash_flow":      500.0,
        "free_cash_flow":           400.0,
        "investing_cash_flow":      -100.0,
        "financing_cash_flow":      -50.0,
        # Margins
        "expense_ratio":            0.60,
        "expense_ratio_1yb":        0.60,
        "opm_1yb":                  20.0,
        "opm_med_5y":               20.0,
        "opm":                      20.0,
        "npm":                      18.0,
        # Profitability
        "roa":                      12.0,
        "roa_1yb":                  11.0,
        "roe":                      18.0,
        "roe_1yb":                  17.0,
        "roce":                     22.0,
        "roce_1yb":                 21.0,
        "pat":                      300.0,
        "pbt":                      400.0,
        "ebitda":                   500.0,
        # Balance sheet
        "total_assets":             2000.0,
        "total_assets_1yb":         1900.0,
        "fixed_assets":             600.0,
        "fixed_assets_1yb":         580.0,
        "cwip":                     50.0,
        "cwip_1yb":                 50.0,
        "debt":                     200.0,
        "inventory_turnover":       5.0,
        "inventory_turnover_1yb":   5.0,
        # Working capital
        "asset_turnover":           1.2,
        "asset_turnover_1yb":       1.2,
        "dep_rate":                 10.0,
        "dep_rate_1yb":             10.0,
        # Misc
        "is_financial":             False,
        "ssgr_cushion":             5.0,
        "high_cash_high_debt":      0,
        "hidden_obligation_growth": 0,
        "psu_value_destruction_flag": 0,
        "cwip_conversion":          0.0,
        "nfat":                     3.0,
        "dso_delta_3y":             5.0,
        "inventory_days_change":    5.0,
        "cash_machine_label":       "✅ Cash Machine",
        "cfo_to_ebitda":            95.0,
        "accruals_warning":         0,
        "opm_stability":            0.0,
    }
    base.update(overrides)
    return base


def _run(rows: list) -> pd.DataFrame:
    """Run compute_red_flags on a list of row dicts, return result df."""
    df = pd.DataFrame(rows)
    return compute_red_flags(df)


# ═══════════════════════════════════════════════════════════════════════════════
# TestSectorDSODictContract
# ═══════════════════════════════════════════════════════════════════════════════

class TestSectorDSODictContract:
    """Verify _SECTOR_DSO_THRESHOLDS dict structure and CSV sector name alignment."""

    def test_dict_is_not_empty(self):
        assert len(_SECTOR_DSO_THRESHOLDS) >= 9, "Must have at least 9 sector entries"

    def test_all_values_are_positive_integers(self):
        for sector, threshold in _SECTOR_DSO_THRESHOLDS.items():
            assert isinstance(threshold, int), f"{sector}: threshold must be int, got {type(threshold)}"
            assert threshold > 0, f"{sector}: threshold must be positive, got {threshold}"

    def test_all_keys_exist_in_csv_sector_list(self):
        """Every key in the dict must be a real sector from the production CSV."""
        for sector in _SECTOR_DSO_THRESHOLDS:
            assert sector in _CSV_SECTOR_NAMES, (
                f"Sector '{sector}' in _SECTOR_DSO_THRESHOLDS not found in CSV sector list. "
                f"Possible typo or data drift."
            )

    def test_it_software_threshold_is_120(self):
        assert _SECTOR_DSO_THRESHOLDS["IT - Software"] == 120

    def test_it_hardware_threshold_is_110(self):
        assert _SECTOR_DSO_THRESHOLDS["IT - Hardware"] == 110

    def test_fmcg_threshold_is_45(self):
        assert _SECTOR_DSO_THRESHOLDS["FMCG"] == 45

    def test_pharmaceuticals_threshold_is_80(self):
        assert _SECTOR_DSO_THRESHOLDS["Pharmaceuticals"] == 80

    def test_construction_threshold_is_120(self):
        assert _SECTOR_DSO_THRESHOLDS["Construction"] == 120

    def test_steel_threshold_is_70(self):
        assert _SECTOR_DSO_THRESHOLDS["Steel"] == 70

    def test_telecom_service_threshold_is_90(self):
        assert _SECTOR_DSO_THRESHOLDS["Telecom-Service"] == 90

    def test_capital_goods_electrical_threshold_is_110(self):
        assert _SECTOR_DSO_THRESHOLDS["Capital Goods - Electrical Equipment"] == 110

    def test_capital_goods_non_electrical_threshold_is_110(self):
        assert _SECTOR_DSO_THRESHOLDS["Capital Goods-Non Electrical Equipment"] == 110

    def test_default_dso_threshold_is_75(self):
        assert _DEFAULT_DSO_THRESHOLD == 75

    def test_fmcg_threshold_is_stricter_than_default(self):
        """FMCG (45) is strictly below the default (75) — consumer goods settle faster."""
        assert _SECTOR_DSO_THRESHOLDS["FMCG"] < _DEFAULT_DSO_THRESHOLD

    def test_it_software_threshold_is_above_default(self):
        """IT-Software (120) is above default (75) — long project billing cycles."""
        assert _SECTOR_DSO_THRESHOLDS["IT - Software"] > _DEFAULT_DSO_THRESHOLD


# ═══════════════════════════════════════════════════════════════════════════════
# TestDSOFalsePositiveSuppression
# ═══════════════════════════════════════════════════════════════════════════════

class TestDSOFalsePositiveSuppression:
    """
    Critical regression: high-quality tech/services companies must NOT trigger
    rf_high_receivables for normal contractual DSO levels.

    This is the core false-positive fix — TCS (DSO~65d) and Infosys (DSO~72d)
    were being flagged under the old 75-day universal threshold. With IT-Software
    threshold = 120, both are clean.
    """

    def test_it_software_dso_75_no_flag(self):
        """IT-Software at exactly 75d DSO — well below 120d threshold → no flag."""
        result = _run([_base_row(sector="IT - Software", days_receivable=75.0)])
        assert result["rf_high_receivables"].iloc[0] == 0, (
            "IT-Software DSO=75 must NOT trigger rf_high_receivables (threshold=120). "
            "This would be a false positive for TCS/Infosys-level billing cycles."
        )

    def test_it_software_dso_110_no_flag(self):
        """IT-Software at 110d DSO — still below 120d → no flag."""
        result = _run([_base_row(sector="IT - Software", days_receivable=110.0)])
        assert result["rf_high_receivables"].iloc[0] == 0

    def test_it_hardware_dso_75_no_flag(self):
        """IT-Hardware at 75d DSO — well below 110d threshold → no flag."""
        result = _run([_base_row(sector="IT - Hardware", days_receivable=75.0)])
        assert result["rf_high_receivables"].iloc[0] == 0

    def test_pharmaceuticals_dso_75_no_flag(self):
        """Pharmaceuticals at 75d DSO — below 80d threshold → no flag."""
        result = _run([_base_row(sector="Pharmaceuticals", days_receivable=75.0)])
        assert result["rf_high_receivables"].iloc[0] == 0

    def test_construction_dso_75_no_flag(self):
        """Construction at 75d DSO — well below 120d threshold → no flag."""
        result = _run([_base_row(sector="Construction", days_receivable=75.0)])
        assert result["rf_high_receivables"].iloc[0] == 0

    def test_telecom_service_dso_75_no_flag(self):
        """Telecom-Service at 75d DSO — below 90d threshold → no flag."""
        result = _run([_base_row(sector="Telecom-Service", days_receivable=75.0)])
        assert result["rf_high_receivables"].iloc[0] == 0

    def test_nan_dso_never_flags_any_sector(self):
        """NaN DSO → notna() guard fires → always 0, regardless of sector."""
        for sector in ["IT - Software", "FMCG", "Construction", "Steel", "Textiles"]:
            result = _run([_base_row(sector=sector, days_receivable=float("nan"))])
            assert result["rf_high_receivables"].iloc[0] == 0, (
                f"NaN DSO must never trigger rf_high_receivables for sector='{sector}'"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TestDSOThresholdBoundaries
# ═══════════════════════════════════════════════════════════════════════════════

class TestDSOThresholdBoundaries:
    """
    Boundary tests at (threshold - 1), threshold, (threshold + 1) for each
    sector in _SECTOR_DSO_THRESHOLDS plus the default fallback.

    Invariant: flag fires only when DSO > threshold (strictly greater than).
    """

    @pytest.mark.parametrize("sector,threshold", list(_SECTOR_DSO_THRESHOLDS.items()))
    def test_at_threshold_minus_one_no_flag(self, sector, threshold):
        """DSO = threshold - 1 → below limit → no flag."""
        result = _run([_base_row(sector=sector, days_receivable=float(threshold - 1))])
        assert result["rf_high_receivables"].iloc[0] == 0, (
            f"{sector}: DSO={threshold - 1} must NOT flag (threshold={threshold})"
        )

    @pytest.mark.parametrize("sector,threshold", list(_SECTOR_DSO_THRESHOLDS.items()))
    def test_at_threshold_exactly_no_flag(self, sector, threshold):
        """DSO = threshold exactly → not strictly greater → no flag (> not >=)."""
        result = _run([_base_row(sector=sector, days_receivable=float(threshold))])
        assert result["rf_high_receivables"].iloc[0] == 0, (
            f"{sector}: DSO={threshold} exactly must NOT flag (strict > comparison)"
        )

    @pytest.mark.parametrize("sector,threshold", list(_SECTOR_DSO_THRESHOLDS.items()))
    def test_at_threshold_plus_one_flags(self, sector, threshold):
        """DSO = threshold + 1 → strictly above → flag fires."""
        result = _run([_base_row(sector=sector, days_receivable=float(threshold + 1))])
        assert result["rf_high_receivables"].iloc[0] == 1, (
            f"{sector}: DSO={threshold + 1} must flag (threshold={threshold})"
        )

    def test_default_sector_at_74_no_flag(self):
        """Unknown sector uses default=75. DSO=74 → no flag."""
        result = _run([_base_row(sector="Textiles", days_receivable=74.0)])
        assert result["rf_high_receivables"].iloc[0] == 0

    def test_default_sector_at_75_no_flag(self):
        """Unknown sector, DSO=75 exactly → no flag (strict >)."""
        result = _run([_base_row(sector="Textiles", days_receivable=75.0)])
        assert result["rf_high_receivables"].iloc[0] == 0

    def test_default_sector_at_76_flags(self):
        """Unknown sector, DSO=76 → above 75 default → flag fires."""
        result = _run([_base_row(sector="Textiles", days_receivable=76.0)])
        assert result["rf_high_receivables"].iloc[0] == 1

    def test_unknown_sector_string_uses_default_threshold(self):
        """Unmapped sector string falls back to default=75."""
        result_74 = _run([_base_row(sector="NewSectorAddedIn2027", days_receivable=74.0)])
        result_76 = _run([_base_row(sector="NewSectorAddedIn2027", days_receivable=76.0)])
        assert result_74["rf_high_receivables"].iloc[0] == 0
        assert result_76["rf_high_receivables"].iloc[0] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# TestDSOTruePositiveActivation
# ═══════════════════════════════════════════════════════════════════════════════

class TestDSOTruePositiveActivation:
    """
    Sectors with tight thresholds (FMCG=45, Steel=70) must correctly trigger
    flags for DSO values that are acceptable for IT but problematic for them.
    """

    def test_fmcg_dso_50_flags(self):
        """FMCG at 50d DSO → 50 > 45 → flag fires. Channel-stuffing signal."""
        result = _run([_base_row(sector="FMCG", days_receivable=50.0)])
        assert result["rf_high_receivables"].iloc[0] == 1

    def test_fmcg_dso_44_no_flag(self):
        """FMCG at 44d DSO → 44 < 45 → clean."""
        result = _run([_base_row(sector="FMCG", days_receivable=44.0)])
        assert result["rf_high_receivables"].iloc[0] == 0

    def test_steel_dso_71_flags(self):
        """Steel at 71d DSO → 71 > 70 → flag fires."""
        result = _run([_base_row(sector="Steel", days_receivable=71.0)])
        assert result["rf_high_receivables"].iloc[0] == 1

    def test_pharmaceuticals_dso_81_flags(self):
        """Pharmaceuticals at 81d DSO → 81 > 80 → flag fires."""
        result = _run([_base_row(sector="Pharmaceuticals", days_receivable=81.0)])
        assert result["rf_high_receivables"].iloc[0] == 1

    def test_it_software_same_dso_no_flag(self):
        """Same 81d DSO that flags Pharma does NOT flag IT-Software (threshold=120)."""
        result = _run([_base_row(sector="IT - Software", days_receivable=81.0)])
        assert result["rf_high_receivables"].iloc[0] == 0, (
            "IT-Software at 81d DSO must be clean. Pharma fires; IT does not."
        )

    def test_fmcg_and_it_same_dso_different_results(self):
        """Two rows with identical DSO=75 — FMCG flags, IT-Software does not."""
        rows = [
            _base_row(sector="FMCG",        days_receivable=75.0),
            _base_row(sector="IT - Software", days_receivable=75.0),
        ]
        result = _run(rows)
        assert result["rf_high_receivables"].iloc[0] == 1, "FMCG DSO=75 must flag (threshold=45)"
        assert result["rf_high_receivables"].iloc[1] == 0, "IT-Software DSO=75 must not flag (threshold=120)"

    def test_multi_sector_vector_result_independent(self):
        """3-row batch — each row uses its own sector threshold independently."""
        rows = [
            _base_row(sector="Steel",          days_receivable=71.0),  # 71 > 70 → flag
            _base_row(sector="Construction",   days_receivable=75.0),  # 75 < 120 → clean
            _base_row(sector="Telecom-Service", days_receivable=91.0), # 91 > 90 → flag
        ]
        result = _run(rows)
        assert result["rf_high_receivables"].tolist() == [1, 0, 1]


# ═══════════════════════════════════════════════════════════════════════════════
# TestCapexMirageDepRateLogic
# ═══════════════════════════════════════════════════════════════════════════════

class TestCapexMirageDepRateLogic:
    """
    Verify that rf_capex_mirage uses each company's AUDITED depreciation rate.

    rf_capex_mirage fires when:
        - company is capital-intensive (not IT/Pharma/Healthcare/Financial)
        - rev_gr_yoy > 20%
        - capex_net / depr_est < 0.5
        where depr_est = fixed_assets_1yb * (dep_rate_1yb / 100)

    ARCHITECTURE NOTE (2026-05-31 refactor):
        The denominator is NO LONGER a static _SECTOR_DEP_RATES lookup. It is
        sourced from the per-company `dep_rate_1yb` column, which data_engine
        derives from audited statement line items
        ((EBITDA_1yb - EBIT_1yb) / fixed_assets_1yb * 100). This is strictly
        superior to a sector-wide guess: a telecom with 18% real gear
        obsolescence and a power utility with 4% long-life assets each get
        their OWN rate. The tests below therefore inject `dep_rate_1yb`
        directly to represent each sector's realistic audited rate.
        See test_rf_capex_mirage_refactor.py for the full contract.
    """

    def test_infra_sector_mild_fa_growth_no_capex_mirage(self):
        """
        Power utility audited dep_rate_1yb=4.0%: FA_1YB=1000, FA=1020 (2% FA growth)
        depr_est = 1000 × 0.04 = 40
        capex_net = max(0, 1020-1000) = 20
        capex_ratio = 20/40 = 0.50 → NOT < 0.5 → no flag
        (with flat 0.10: depr_est=100, capex_ratio=20/100=0.20 → would wrongly flag)
        """
        result = _run([_base_row(
            sector="Power Infrastructure",
            dep_rate_1yb=4.0,   # audited long-life power asset rate (per-company)
            fixed_assets=1020.0,
            fixed_assets_1yb=1000.0,
            rev_gr_yoy=25.0,   # revenue growth > 20% — mirage condition active
        )])
        assert result["rf_capex_mirage"].iloc[0] == 0, (
            "Power utility: capex_ratio=0.5 (exact boundary) must NOT trigger "
            "rf_capex_mirage. The company's audited dep_rate_1yb (4.0%) must be used."
        )

    def test_infra_sector_zero_fa_growth_triggers_capex_mirage(self):
        """
        Power utility audited dep_rate_1yb=4.0%: FA flat (no new investment), rev_gr=25%
        depr_est = 1000 × 0.04 = 40
        capex_net = max(0, 1000-1000) = 0
        capex_ratio = 0/40 = 0.0 < 0.5 → flag fires
        """
        result = _run([_base_row(
            sector="Power Infrastructure",
            dep_rate_1yb=4.0,   # audited long-life power asset rate (per-company)
            fixed_assets=1000.0,
            fixed_assets_1yb=1000.0,
            rev_gr_yoy=25.0,
        )])
        assert result["rf_capex_mirage"].iloc[0] == 1, (
            "Power Infrastructure: zero FA growth with 25% rev growth must trigger "
            "rf_capex_mirage (deferred maintenance on long-life power assets)."
        )

    def test_telecom_low_capex_relative_to_high_dep_rate_flags(self):
        """
        Telecom audited dep_rate_1yb=18.0%: FA_1YB=1000, FA=1050 (5% FA growth), rev_gr=25%
        depr_est = 1000 × 0.18 = 180
        capex_net = max(0, 50) = 50
        capex_ratio = 50/180 ≈ 0.278 < 0.5 → flag fires
        (correctly: telecom network equipment needs frequent replacement at 18% dep)
        """
        result = _run([_base_row(
            sector="Telecom-Service",
            dep_rate_1yb=18.0,   # audited high-obsolescence network gear rate (per-company)
            fixed_assets=1050.0,
            fixed_assets_1yb=1000.0,
            rev_gr_yoy=25.0,
        )])
        assert result["rf_capex_mirage"].iloc[0] == 1, (
            "Telecom: capex ratio ≈ 0.278 < 0.5 with 25% rev growth must flag. "
            "Audited dep_rate_1yb (18.0%) correctly identifies under-investment in network gear."
        )

    def test_default_sector_adequate_capex_no_flag(self):
        """
        Default sector (dep=0.10): FA_1YB=1000, FA=1060 (6% FA growth), rev_gr=25%
        depr_est = 1000 × 0.10 = 100
        capex_net = 60
        capex_ratio = 60/100 = 0.60 > 0.5 → no flag
        """
        result = _run([_base_row(
            sector="Textiles",   # default dep=0.10
            fixed_assets=1060.0,
            fixed_assets_1yb=1000.0,
            rev_gr_yoy=25.0,
        )])
        assert result["rf_capex_mirage"].iloc[0] == 0, (
            "Default sector: capex_ratio=0.60 > 0.5 must NOT flag. Adequate reinvestment."
        )

    def test_default_sector_low_capex_flags(self):
        """
        Default sector (dep=0.10): FA_1YB=1000, FA=1010 (1% FA growth), rev_gr=25%
        depr_est = 100, capex_net = 10, capex_ratio = 0.10 < 0.5 → flag
        """
        result = _run([_base_row(
            sector="Textiles",
            fixed_assets=1010.0,
            fixed_assets_1yb=1000.0,
            rev_gr_yoy=25.0,
        )])
        assert result["rf_capex_mirage"].iloc[0] == 1

    def test_it_software_excluded_from_capex_mirage_regardless_of_numbers(self):
        """
        IT-Software is in _HIGH_DSO_SECTORS → _is_capital_intensive = False
        → rf_capex_mirage = 0 regardless of FA growth and rev_gr_yoy.
        IT asset-light model is their ADVANTAGE, not a red flag.
        """
        result = _run([_base_row(
            sector="IT - Software",
            fixed_assets=500.0,
            fixed_assets_1yb=1000.0,   # FA actually declining
            rev_gr_yoy=50.0,            # 50% revenue growth
        )])
        assert result["rf_capex_mirage"].iloc[0] == 0, (
            "IT-Software must NEVER trigger rf_capex_mirage. "
            "High revenue with low capex is their structural moat, not deferred maintenance."
        )

    def test_revenue_growth_below_20_never_flags(self):
        """
        rf_capex_mirage requires rev_gr_yoy > 20%. Below this, never fires.
        """
        result = _run([_base_row(
            sector="Textiles",
            fixed_assets=1000.0,
            fixed_assets_1yb=1000.0,   # zero FA growth
            rev_gr_yoy=19.9,            # just below 20% trigger
        )])
        assert result["rf_capex_mirage"].iloc[0] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# TestSectorColumnMissing
# ═══════════════════════════════════════════════════════════════════════════════

class TestSectorColumnMissing:
    """
    When sector column is absent entirely, both flags must degrade gracefully
    to their default behaviors — no KeyError, no NaN propagation.
    """

    def test_rf_high_receivables_fallback_when_sector_missing(self):
        """No sector column → .get() returns 'Unknown' → fillna(75) → behaves as default."""
        base = _base_row(days_receivable=80.0)
        base.pop("sector")   # remove sector column entirely
        result = _run([base])
        # 80 > 75 (default) → should flag
        assert result["rf_high_receivables"].iloc[0] == 1, (
            "Missing sector column: DSO=80 > default 75 must still flag"
        )

    def test_rf_high_receivables_clean_when_sector_missing_and_dso_below_default(self):
        """No sector column → default=75. DSO=70 → no flag."""
        base = _base_row(days_receivable=70.0)
        base.pop("sector")
        result = _run([base])
        assert result["rf_high_receivables"].iloc[0] == 0

    def test_rf_capex_mirage_fallback_when_sector_missing(self):
        """No sector column → dep_multiplier falls back to 0.10 (default)."""
        base = _base_row(
            fixed_assets=1010.0,
            fixed_assets_1yb=1000.0,
            rev_gr_yoy=25.0,
        )
        base.pop("sector")
        result = _run([base])
        # depr_est = 1000 × 0.10 = 100, capex_net = 10, ratio = 0.10 < 0.5 → flag
        assert result["rf_capex_mirage"].iloc[0] == 1, (
            "Missing sector column: falls back to default dep rate (0.10). "
            "capex_ratio=0.10 < 0.5 must still trigger capex_mirage."
        )

    def test_no_keyerror_when_sector_column_absent(self):
        """compute_red_flags must not raise KeyError when sector column is missing."""
        base = _base_row(days_receivable=50.0)
        base.pop("sector")
        try:
            _run([base])
        except KeyError as e:
            pytest.fail(f"KeyError raised when sector column absent: {e}")
