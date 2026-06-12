"""
Contract Tests — Financial Shenanigans Forensic Engine (Schilit)
================================================================
STAGE 3 of the Schilit integration pipeline.

Tests parse docs/schilit_forensic_specs.json and assert that EVERY threshold,
penalty constant, and checker variable declared in the spec is precisely reflected
in the live core/forensic_engine.py and core/scoring_engine.py code.

Structure:
    TestSchilitSpecLedger           — JSON file structure + required top-level keys
    TestSchilitFunctionExists       — compute_schilit_forensic_score exists in forensic_engine
    TestSchilitScoringConstants     — initial_score=100, per_checker_deduction=15, pass_threshold=70
    TestSchilitCheckerVariables     — 4 canonical checker variable names in function body
    TestSchilitThresholdContract    — JSON thresholds match code literals (regex scan)
    TestSchilitOutputColumns        — all 6 output columns produced on synthetic data
    TestSchilitScoringLogic         — score arithmetic correct on controlled fixture
    TestSchilitPipelineIntegration  — schilit_pass hooked into fw_str + pipeline call chains
"""

import json
import re
import os
import sys
import pytest
import pandas as pd
import numpy as np

# ── Path bootstrap ───────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SPEC_PATH  = os.path.join(REPO_ROOT, "docs", "schilit_forensic_specs.json")
FE_PATH    = os.path.join(REPO_ROOT, "core", "forensic_engine.py")
SE_PATH    = os.path.join(REPO_ROOT, "core", "scoring_engine.py")


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def spec() -> dict:
    """Load the canonical spec ledger once for the entire module."""
    with open(SPEC_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def fe_source() -> str:
    """Full text of forensic_engine.py."""
    with open(FE_PATH, encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module")
def se_source() -> str:
    """Full text of scoring_engine.py."""
    with open(SE_PATH, encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module")
def fe_fn_block(fe_source) -> str:
    """Text of compute_schilit_forensic_score function body only."""
    # Extract from function def to the next top-level function/class
    pattern = r"def compute_schilit_forensic_score\(.*?(?=\ndef |\nclass |\Z)"
    match = re.search(pattern, fe_source, re.DOTALL)
    assert match, "compute_schilit_forensic_score not found in forensic_engine.py"
    return match.group(0)


def _make_minimal_df(n: int = 5) -> pd.DataFrame:
    """Return a minimal DataFrame with all columns that compute_schilit_forensic_score reads.

    v1.1: Added 'days_receivable' for Checker-4 Signal 3 (absolute DSO > 90 days, Schilit Ch.3/4/14).
    v1.2: Added Checker-1 Signal 4 columns (npm, opm), Checker-1 Signal 5 columns
          (asset_turnover, asset_turnover_1yb, opm_1yb), and Checker-2 Signal 3 column
          (cfo_to_ebitda) — all set to clean values that do NOT fire any new signal.
    v1.3: Added Checker-1 Signal 6 columns (fixed_assets, fixed_assets_1yb, dep_rate,
          dep_rate_1yb) — FA flat (no growth), dep_rate flat (no fall) → Signal 6 does NOT fire.
          dep_rate is derived in data_engine.py as (ebitda − ebit) / fixed_assets × 100.
    """
    idx = range(n)
    return pd.DataFrame({
        # ── Checker 1: EMS ─────────────────────────────────────────────────────
        "accruals_warning":        [0] * n,
        "inv_vs_rev_gap":          [0.0] * n,
        "opm_stability":           [0.0] * n,
        # Signal 4 (EMS #3 other-income inflate): npm=5, opm=10 → diff=-5 < 5 → no fire
        "npm":                     [5.0] * n,    # NPM (%) — below OPM
        "opm":                     [10.0] * n,   # OPM (%) — used for S4 and S5 current
        # Signal 5 (EMS #4 expense-cap proxy): AT flat, OPM flat → no fire
        "asset_turnover":          [2.0] * n,    # current asset turnover ratio
        "asset_turnover_1yb":      [2.0] * n,    # 1-year-back AT — same as current, no decline
        "opm_1yb":                 [10.0] * n,   # OPM 1 year back — same as current, no improvement
        # Signal 6 (EMS #4 dep-rate manipulation): FA flat, dep_rate flat → no fire
        "fixed_assets":            [100.0] * n,  # current net fixed assets (₹ Cr)
        "fixed_assets_1yb":        [100.0] * n,  # 1-year-back FA — same → growth=0% < 5% → no fire
        "dep_rate":                [10.0] * n,   # current dep/FA (%) — 10% = 10yr useful life
        "dep_rate_1yb":            [10.0] * n,   # 1YB dep/FA — same → no fall → no fire
        # ── Checker 2: CFS ─────────────────────────────────────────────────────
        "pat_gr_yoy":              [0.0] * n,
        "ocf_growth":              [0.0] * n,
        "cash_machine_label":      ["✅ Solid"] * n,
        # Signal 3 (CFS #3 cfo_ebitda_weak): 80% > 60% threshold → no fire
        "cfo_to_ebitda":           [80.0] * n,   # CFO/EBITDA as percentage (80% = clean)
        # ── Checker 3: KMS Leverage ────────────────────────────────────────────
        "hidden_obligation_growth":[0] * n,
        "high_cash_high_debt":     [0] * n,
        # ── Checker 4: KMS Bloat ───────────────────────────────────────────────
        "sector":                  ["Manufacturing"] * n,
        "is_financial":            [False] * n,
        "dso_delta_3y":            [0.0] * n,
        "inventory_days_change":   [0.0] * n,
        "days_receivable":         [30.0] * n,   # Checker-4 Signal 3: absolute DSO (normal ~30d)
    }, index=idx)


# ═══════════════════════════════════════════════════════════════════════════════
# CLASS 1: SPEC LEDGER STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

class TestSchilitSpecLedger:
    """Verify the JSON spec file exists and has the required structural keys."""

    def test_spec_file_exists(self):
        assert os.path.isfile(SPEC_PATH), f"Missing spec file: {SPEC_PATH}"

    def test_spec_is_valid_json(self, spec):
        # If fixture loaded, JSON is valid. Just assert dict.
        assert isinstance(spec, dict)

    def test_spec_has_meta(self, spec):
        assert "_meta" in spec, "Missing '_meta' top-level key"

    def test_spec_has_scoring_parameters(self, spec):
        assert "scoring_parameters" in spec

    def test_spec_has_all_four_checkers(self, spec):
        for key in ["checker_1_ems", "checker_2_cfs", "checker_3_kms_leverage", "checker_4_kms_bloat"]:
            assert key in spec, f"Missing checker key: {key}"

    def test_spec_checker4_has_signal3_absolute_dso(self, spec):
        """Checker 4 must declare signal_3_absolute_dso (v1.1 addition from full book reading)."""
        assert "signal_3_absolute_dso" in spec["checker_4_kms_bloat"], \
            "signal_3_absolute_dso missing from checker_4_kms_bloat — added after full Schilit book read"

    def test_spec_absolute_dso_threshold_is_90(self, spec):
        """Absolute DSO threshold must be 90.0 days (non-IT/Healthcare sectors)."""
        val = spec["checker_4_kms_bloat"]["signal_3_absolute_dso"]["threshold"]
        assert val == 90.0, f"Expected absolute DSO threshold 90.0, got {val}"

    def test_spec_has_aa_shenanigans_gap_section(self, spec):
        """AA shenanigans gap must be documented (Part 5, Ch.15-17, new in 4th edition)."""
        assert "aa_shenanigans_implementation_gap" in spec, \
            "Missing aa_shenanigans_implementation_gap — new Part 5 of 4th edition must be documented"

    def test_spec_aa_shenanigans_marked_not_implemented(self, spec):
        """AA shenanigans section must be explicitly marked as not implemented."""
        aa = spec["aa_shenanigans_implementation_gap"]
        assert aa.get("implemented") is False, "AA shenanigans must be marked implemented=false"

    # ── v1.2 additions: EMS signals 4+5, CFS signal 3 ──────────────────────────

    def test_spec_checker1_has_signal4_other_income(self, spec):
        """Checker 1 must declare signal_4_other_income_inflate (EMS#3 — NPM−OPM > 5pp, Schilit Ch.5)."""
        assert "signal_4_other_income_inflate" in spec["checker_1_ems"], \
            "signal_4_other_income_inflate missing from checker_1_ems — must be added per Schilit Ch.5 EMS#3"

    def test_spec_checker1_other_income_threshold_is_5(self, spec):
        """Other-income inflate threshold must be 5.0 percentage points (npm − opm > 5.0)."""
        val = spec["checker_1_ems"]["signal_4_other_income_inflate"]["threshold"]
        assert val == 5.0, f"Expected other-income threshold 5.0, got {val}"

    def test_spec_checker1_has_signal5_expense_cap(self, spec):
        """Checker 1 must declare signal_5_expense_cap_proxy (EMS#4 — AOL pattern, Schilit Ch.6)."""
        assert "signal_5_expense_cap_proxy" in spec["checker_1_ems"], \
            "signal_5_expense_cap_proxy missing from checker_1_ems — must be added per Schilit Ch.6 EMS#4"

    def test_spec_checker1_expense_cap_at_decline_threshold_is_0_10(self, spec):
        """Expense-cap proxy: asset_turnover decline threshold must be 0.10."""
        val = spec["checker_1_ems"]["signal_5_expense_cap_proxy"]["threshold_at_decline"]
        assert val == 0.10, f"Expected AT decline threshold 0.10, got {val}"

    def test_spec_checker1_expense_cap_opm_improvement_threshold_is_2(self, spec):
        """Expense-cap proxy: OPM improvement threshold must be 2.0 percentage points."""
        val = spec["checker_1_ems"]["signal_5_expense_cap_proxy"]["threshold_opm_improvement"]
        assert val == 2.0, f"Expected OPM improvement threshold 2.0, got {val}"

    def test_spec_checker2_has_signal3_cfo_ebitda(self, spec):
        """Checker 2 must declare signal_3_cfo_ebitda_weakness (CFS#3 — OCF/EBITDA < 60%, Schilit Ch.12)."""
        assert "signal_3_cfo_ebitda_weakness" in spec["checker_2_cfs"], \
            "signal_3_cfo_ebitda_weakness missing from checker_2_cfs — must be added per Schilit Ch.12 CFS#3"

    def test_spec_checker2_cfo_ebitda_threshold_is_60(self, spec):
        """CFO/EBITDA weakness threshold must be 60.0 (percentage form)."""
        val = spec["checker_2_cfs"]["signal_3_cfo_ebitda_weakness"]["threshold"]
        assert val == 60.0, f"Expected CFO/EBITDA threshold 60.0, got {val}"

    def test_spec_checker2_cfo_ebitda_has_financial_exclusion(self, spec):
        """CFO/EBITDA signal must declare financial_exclusion=true in spec."""
        assert spec["checker_2_cfs"]["signal_3_cfo_ebitda_weakness"].get("financial_exclusion") is True, \
            "signal_3_cfo_ebitda_weakness must have financial_exclusion=true (banks/NBFCs excluded)"

    def test_spec_version_updated_to_1_3(self, spec):
        """Spec version must reflect v1.3 (Signal 6 dep-rate manipulation + EBIT mapping + missing columns gap)."""
        ver = spec["_meta"]["version"]
        assert "1.3" in ver, f"Expected version containing '1.3', got '{ver}'"

    # ── v1.3 additions: EMS Signal 6 (dep-rate manipulation) + missing columns gap ─

    def test_spec_checker1_has_signal6_dep_rate_manip(self, spec):
        """Checker 1 must declare signal_6_depreciation_rate_manipulation (EMS#4 — Schilit Ch.6/9)."""
        assert "signal_6_depreciation_rate_manipulation" in spec["checker_1_ems"], \
            "signal_6_depreciation_rate_manipulation missing from checker_1_ems — " \
            "required for Qwest/WorldCom dep-rate manipulation pattern (Schilit Ch.6 EMS#4)"

    def test_spec_signal6_fa_growth_threshold_is_1_05(self, spec):
        """Signal 6 FA-growth multiplier threshold must be 1.05 (>5% growth required)."""
        val = spec["checker_1_ems"]["signal_6_depreciation_rate_manipulation"]["threshold_fa_growth"]
        assert val == 1.05, f"Expected FA growth threshold 1.05, got {val}"

    def test_spec_signal6_dep_rate_fall_threshold_is_0_80(self, spec):
        """Signal 6 dep-rate fall multiplier must be 0.80 (dep/FA fell >20% YoY)."""
        val = spec["checker_1_ems"]["signal_6_depreciation_rate_manipulation"]["threshold_dep_rate_fall"]
        assert val == 0.8, f"Expected dep-rate fall threshold 0.80, got {val}"

    def test_spec_signal6_has_financial_exclusion(self, spec):
        """Signal 6 must declare financial_exclusion=true (banks have trivially small FA)."""
        assert spec["checker_1_ems"]["signal_6_depreciation_rate_manipulation"].get("financial_exclusion") is True, \
            "signal_6_depreciation_rate_manipulation must have financial_exclusion=true"

    def test_spec_signal6_precomputed_in_data_engine(self, spec):
        """Signal 6 spec must document that dep_rate is precomputed in data_engine.py."""
        precomp = spec["checker_1_ems"]["signal_6_depreciation_rate_manipulation"].get("precomputed_in", "")
        assert "data_engine" in precomp, \
            "signal_6 spec must document dep_rate derivation is in data_engine.py"

    def test_spec_has_missing_csv_columns_gap(self, spec):
        """Spec must document missing_csv_columns_gap (audit of BUGS.TXT gaps vs Screener.in CSV)."""
        assert "missing_csv_columns_gap" in spec, \
            "missing_csv_columns_gap section missing — must document Other Income, " \
            "Intangibles, Contingent Liabilities status after BUGS.TXT analysis"

    def test_spec_missing_columns_gap_has_resolved_ebit(self, spec):
        """missing_csv_columns_gap must document EBIT as RESOLVED (was in CSV but unmapped)."""
        gap = spec["missing_csv_columns_gap"]
        assert "resolved_column" in gap, "missing_csv_columns_gap must document resolved_column (EBIT)"
        assert "ebit" in gap["resolved_column"].get("column", "").lower() or \
               "ebit" in str(gap["resolved_column"]).lower(), \
            "resolved_column must reference EBIT (the column that was in CSV but unmapped)"

    def test_spec_missing_columns_gap_documents_other_income_absent(self, spec):
        """missing_csv_columns_gap must document Other Income as NOT IMPLEMENTED (absent from Screener.in)."""
        gap = spec["missing_csv_columns_gap"]
        assert "column_1_other_income" in gap, \
            "missing_csv_columns_gap must document Other Income status"

    def test_spec_missing_columns_gap_documents_contingent_liabilities(self, spec):
        """missing_csv_columns_gap must document Contingent Liabilities as NOT IMPLEMENTED."""
        gap = spec["missing_csv_columns_gap"]
        assert "column_3_contingent_liabilities" in gap, \
            "missing_csv_columns_gap must document Contingent Liabilities status"

    def test_spec_has_output_column_definitions(self, spec):
        assert "output_column_definitions" in spec

    def test_spec_has_pipeline_integration(self, spec):
        assert "pipeline_integration" in spec

    def test_spec_initial_score_is_100(self, spec):
        val = spec["scoring_parameters"]["initial_score"]["value"]
        assert val == 100.0, f"Expected 100.0, got {val}"

    def test_spec_per_checker_deduction_is_15(self, spec):
        val = spec["scoring_parameters"]["per_checker_deduction"]["value"]
        assert val == 15.0, f"Expected 15.0, got {val}"

    def test_spec_pass_threshold_is_70(self, spec):
        val = spec["scoring_parameters"]["pass_threshold"]["value"]
        assert val == 70.0, f"Expected 70.0, got {val}"

    def test_spec_framework_label(self, spec):
        assert spec["_meta"]["framework_label"] == "Financial Shenanigans"

    def test_spec_all_checker_deductions_match_global(self, spec):
        """Every checker's deduction_points must equal the global per_checker_deduction."""
        global_ded = spec["scoring_parameters"]["per_checker_deduction"]["value"]
        for checker in ["checker_1_ems", "checker_2_cfs", "checker_3_kms_leverage", "checker_4_kms_bloat"]:
            ded = spec[checker]["deduction_points"]
            assert ded == global_ded, f"{checker} deduction {ded} != global {global_ded}"

    def test_spec_all_six_output_columns_defined(self, spec):
        expected = {
            "schilit_forensic_score", "schilit_pass",
            "schilit_ems_flag", "schilit_cfs_flag",
            "schilit_kms_lev_flag", "schilit_kms_bloat_flag"
        }
        defined = set(spec["output_column_definitions"].keys())
        assert expected == defined, f"Output columns mismatch. Expected {expected}, got {defined}"


# ═══════════════════════════════════════════════════════════════════════════════
# CLASS 2: FUNCTION EXISTS IN FORENSIC ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class TestSchilitFunctionExists:
    """Verify compute_schilit_forensic_score is defined in forensic_engine.py."""

    def test_function_defined(self, fe_source):
        assert "def compute_schilit_forensic_score(" in fe_source

    def test_function_takes_df_argument(self, fe_fn_block):
        assert "def compute_schilit_forensic_score(df" in fe_fn_block

    def test_function_returns_df(self, fe_fn_block):
        assert "return df" in fe_fn_block

    def test_function_is_importable(self):
        from core.forensic_engine import compute_schilit_forensic_score
        assert callable(compute_schilit_forensic_score)

    def test_function_produces_dataframe(self):
        from core.forensic_engine import compute_schilit_forensic_score
        df = _make_minimal_df(3)
        result = compute_schilit_forensic_score(df)
        assert isinstance(result, pd.DataFrame)

    def test_function_does_not_mutate_input(self):
        from core.forensic_engine import compute_schilit_forensic_score
        df = _make_minimal_df(3)
        original_cols = set(df.columns)
        compute_schilit_forensic_score(df)
        assert set(df.columns) == original_cols, "Input DataFrame was mutated"


# ═══════════════════════════════════════════════════════════════════════════════
# CLASS 3: SCORING CONSTANTS IN CODE
# ═══════════════════════════════════════════════════════════════════════════════

class TestSchilitScoringConstants:
    """Assert score=100, penalty=15, threshold=70 appear as literals in the function."""

    def test_initial_score_100_in_code(self, fe_fn_block, spec):
        """100.0 must appear as score initializer literal in the function."""
        val = spec["scoring_parameters"]["initial_score"]["value"]
        # Pattern: 100.0 as the start of the scoring expression
        assert re.search(r"100\.0", fe_fn_block), \
            f"Literal {val} not found in compute_schilit_forensic_score"

    def test_per_checker_deduction_15_in_code(self, fe_fn_block, spec):
        """15.0 must appear as the multiplier in the deduction expression."""
        val = spec["scoring_parameters"]["per_checker_deduction"]["value"]
        pattern = rf"\*\s*{re.escape(str(val))}"
        assert re.search(pattern, fe_fn_block), \
            f"Deduction literal '* {val}' not found in compute_schilit_forensic_score"

    def test_pass_threshold_70_in_code(self, fe_fn_block, spec):
        """70.0 must appear as the pass gate threshold in the function."""
        val = spec["scoring_parameters"]["pass_threshold"]["value"]
        pattern = rf">= {re.escape(str(val))}"
        assert re.search(pattern, fe_fn_block), \
            f"Pass threshold literal '>= {val}' not found in compute_schilit_forensic_score"

    def test_score_deduction_uses_multiplication(self, fe_fn_block):
        """Score is computed by subtracting checker * 15.0 — must have astype(float) * 15.0."""
        assert re.search(r"astype\(float\)\s*\*\s*15\.0", fe_fn_block), \
            "Expected 'astype(float) * 15.0' pattern for vectorized deduction"

    def test_score_clipped_to_zero_hundred(self, fe_fn_block):
        """Score must be clipped to [0, 100] range."""
        assert re.search(r"clip\(.*?lower=0", fe_fn_block), \
            "Score must be clipped with lower=0"


# ═══════════════════════════════════════════════════════════════════════════════
# CLASS 4: CHECKER VARIABLE NAMES IN FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

class TestSchilitCheckerVariables:
    """Assert all 4 canonical checker variable names appear in the function body."""

    _CHECKER_VARS = [
        "ems_revenue_expense_manipulation",
        "cfs_cash_flow_distortion",
        "kms_balance_sheet_leverage_trap",
        "kms_inventory_receivables_bloat",
    ]

    @pytest.mark.parametrize("var_name", _CHECKER_VARS)
    def test_checker_variable_in_function(self, fe_fn_block, var_name):
        assert var_name in fe_fn_block, \
            f"Checker variable '{var_name}' not found in compute_schilit_forensic_score"

    def test_all_four_checkers_used_in_score_expression(self, fe_fn_block):
        """Each checker variable must appear in the _score computation block."""
        score_block_match = re.search(
            r"_score\s*=.*?\.clip\(",
            fe_fn_block, re.DOTALL
        )
        assert score_block_match, "Score computation block not found"
        score_block = score_block_match.group(0)
        for var in self._CHECKER_VARS:
            assert var in score_block, \
                f"Checker '{var}' not used in _score computation"

    def test_all_four_flag_columns_assigned(self, fe_fn_block):
        """All 4 schilit_*_flag columns must be assigned in the function."""
        for col in ["schilit_ems_flag", "schilit_cfs_flag",
                    "schilit_kms_lev_flag", "schilit_kms_bloat_flag"]:
            assert f'df["{col}"]' in fe_fn_block or f"df['{col}']" in fe_fn_block, \
                f"Output column '{col}' not assigned in function"


# ═══════════════════════════════════════════════════════════════════════════════
# CLASS 5: THRESHOLD CONTRACT (JSON ↔ CODE)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSchilitThresholdContract:
    """Parse each threshold from the JSON and assert it appears as a literal in the code."""

    def test_checker1_inv_rev_gap_threshold(self, fe_fn_block, spec):
        """inv_vs_rev_gap > 20.0 threshold must appear in code."""
        val = spec["checker_1_ems"]["signal_2_inv_rev_gap"]["threshold"]
        pattern = rf">\s*{re.escape(str(float(val)))}"
        assert re.search(pattern, fe_fn_block), \
            f"inv_vs_rev_gap threshold >{val} not found in function"

    def test_checker1_opm_stability_threshold(self, fe_fn_block, spec):
        """opm_stability > 30.0 threshold must appear in code."""
        val = spec["checker_1_ems"]["signal_3_opm_stability"]["threshold"]
        pattern = rf">\s*{re.escape(str(float(val)))}"
        assert re.search(pattern, fe_fn_block), \
            f"opm_stability threshold >{val} not found in function"

    def test_checker2_pat_gr_yoy_threshold(self, fe_fn_block, spec):
        """pat_gr_yoy > 15.0 threshold must appear in code."""
        val = spec["checker_2_cfs"]["signal_1_earnings_cash_divergence"]["pat_gr_yoy_threshold"]
        pattern = rf">\s*{re.escape(str(float(val)))}"
        assert re.search(pattern, fe_fn_block), \
            f"pat_gr_yoy threshold >{val} not found in function"

    def test_checker2_ocf_growth_threshold(self, fe_fn_block, spec):
        """ocf_growth < -15.0 threshold must appear in code."""
        val = spec["checker_2_cfs"]["signal_1_earnings_cash_divergence"]["ocf_growth_threshold"]
        # val = -15.0; appears as < -15.0
        pattern = rf"<\s*{re.escape(str(float(val)))}"
        assert re.search(pattern, fe_fn_block), \
            f"ocf_growth threshold <{val} not found in function"

    def test_checker2_paper_profits_label(self, fe_fn_block, spec):
        """'📄 Paper Profits' string literal must appear in code."""
        label = spec["checker_2_cfs"]["signal_2_paper_profits"]["label_value"]
        assert label in fe_fn_block, \
            f"cash_machine_label value '{label}' not found in function"

    def test_checker4_dso_delta_3y_threshold(self, fe_fn_block, spec):
        """dso_delta_3y > 15.0 threshold must appear in code."""
        val = spec["checker_4_kms_bloat"]["signal_1_dso_delta_3y"]["threshold"]
        pattern = rf">\s*{re.escape(str(float(val)))}"
        assert re.search(pattern, fe_fn_block), \
            f"dso_delta_3y threshold >{val} not found in function"

    def test_checker4_inventory_days_change_threshold(self, fe_fn_block, spec):
        """inventory_days_change > 20.0 threshold must appear in code."""
        val = spec["checker_4_kms_bloat"]["signal_2_inventory_days_change"]["threshold"]
        pattern = rf">\s*{re.escape(str(float(val)))}"
        assert re.search(pattern, fe_fn_block), \
            f"inventory_days_change threshold >{val} not found in function"

    def test_checker1_accruals_warning_column_used(self, fe_fn_block, spec):
        """accruals_warning column must be accessed in checker 1."""
        col = spec["checker_1_ems"]["signal_1_accruals"]["column"]
        assert col in fe_fn_block, f"Column '{col}' not used in function"

    def test_checker3_hidden_obligation_growth_column_used(self, fe_fn_block, spec):
        """hidden_obligation_growth column must be accessed in checker 3."""
        col = spec["checker_3_kms_leverage"]["signal_1_hidden_obligation"]["column"]
        assert col in fe_fn_block, f"Column '{col}' not used in function"

    def test_checker3_high_cash_high_debt_column_used(self, fe_fn_block, spec):
        """high_cash_high_debt column must be accessed in checker 3."""
        col = spec["checker_3_kms_leverage"]["signal_2_high_cash_high_debt"]["column"]
        assert col in fe_fn_block, f"Column '{col}' not used in function"

    def test_checker4_excludes_high_dso_sectors(self, fe_fn_block):
        """_HIGH_DSO_SECTORS must be referenced in the function for sector exclusion."""
        assert "_HIGH_DSO_SECTORS" in fe_fn_block, \
            "_HIGH_DSO_SECTORS exclusion not applied in checker 4"

    def test_checker4_excludes_financial_stocks(self, fe_fn_block):
        """is_financial must be referenced in the function for financial exclusion."""
        assert "is_financial" in fe_fn_block, \
            "is_financial exclusion not applied in checker 4"

    def test_checker4_absolute_dso_threshold_in_code(self, fe_fn_block, spec):
        """days_receivable > 90.0 absolute DSO threshold must appear in code (Schilit Ch.3/4/14)."""
        val = spec["checker_4_kms_bloat"]["signal_3_absolute_dso"]["threshold"]
        # The code must have: _days_recv > 90.0 (or equivalent literal)
        pattern = rf">\s*{re.escape(str(float(val)))}"
        assert re.search(pattern, fe_fn_block), \
            f"Absolute DSO threshold >{val} not found in compute_schilit_forensic_score"

    def test_checker4_days_receivable_column_used(self, fe_fn_block, spec):
        """days_receivable column must be accessed in checker 4."""
        col = spec["checker_4_kms_bloat"]["signal_3_absolute_dso"]["column"]
        assert col in fe_fn_block, f"Column '{col}' not used in function"

    def test_checker4_logic_has_three_signals(self, fe_fn_block):
        """Checker 4 kms_inventory_receivables_bloat must reference all 3 sub-signals."""
        # All 3 column references must exist in the function body
        for col in ["dso_delta_3y", "inventory_days_change", "days_receivable"]:
            assert col in fe_fn_block, \
                f"Checker 4 signal column '{col}' not referenced in function"

    # ── v1.2: Checker 1 Signal 4 (other-income inflate, EMS #3) ─────────────────

    def test_checker1_signal4_other_income_threshold_in_code(self, fe_fn_block, spec):
        """NPM − OPM > 5.0 threshold literal must appear in compute_schilit_forensic_score (Schilit Ch.5 EMS#3)."""
        val = spec["checker_1_ems"]["signal_4_other_income_inflate"]["threshold"]
        pattern = rf">\s*{re.escape(str(float(val)))}"
        assert re.search(pattern, fe_fn_block), \
            f"Signal 4 other-income threshold >{val} not found in function"

    def test_checker1_npm_column_used(self, fe_fn_block, spec):
        """'npm' column must be accessed in the function for Signal 4."""
        col = spec["checker_1_ems"]["signal_4_other_income_inflate"]["column_a"]
        assert col in fe_fn_block, f"Signal 4 column '{col}' not referenced in function"

    def test_checker1_opm_column_used_for_signal4(self, fe_fn_block, spec):
        """'opm' column must be accessed in the function for Signal 4 (NPM − OPM)."""
        col = spec["checker_1_ems"]["signal_4_other_income_inflate"]["column_b"]
        assert col in fe_fn_block, f"Signal 4 column '{col}' not referenced in function"

    # ── v1.2: Checker 1 Signal 5 (expense-cap proxy, EMS #4) ────────────────────

    def test_checker1_signal5_asset_turnover_column_used(self, fe_fn_block):
        """'asset_turnover' column must be accessed in the function for Signal 5 (Schilit Ch.6 EMS#4)."""
        assert "asset_turnover" in fe_fn_block, \
            "Signal 5 (expense-cap proxy) column 'asset_turnover' not referenced in function"

    def test_checker1_signal5_at_1yb_column_used(self, fe_fn_block):
        """'asset_turnover_1yb' column must be accessed in the function for Signal 5."""
        assert "asset_turnover_1yb" in fe_fn_block, \
            "Signal 5 column 'asset_turnover_1yb' not referenced in function"

    def test_checker1_signal5_at_decline_threshold_in_code(self, fe_fn_block, spec):
        """AT decline threshold 0.10 literal must appear in function (Schilit Ch.6 AOL pattern)."""
        val = spec["checker_1_ems"]["signal_5_expense_cap_proxy"]["threshold_at_decline"]
        pattern = rf">\s*{re.escape(str(float(val)))}"
        assert re.search(pattern, fe_fn_block), \
            f"Signal 5 AT decline threshold >{val} not found in function"

    def test_checker1_signal5_opm_improvement_threshold_in_code(self, fe_fn_block, spec):
        """OPM improvement threshold 2.0 literal must appear in function (Schilit Ch.6 AOL pattern)."""
        val = spec["checker_1_ems"]["signal_5_expense_cap_proxy"]["threshold_opm_improvement"]
        pattern = rf">\s*{re.escape(str(float(val)))}"
        assert re.search(pattern, fe_fn_block), \
            f"Signal 5 OPM improvement threshold >{val} not found in function"

    # ── v1.2: Checker 2 Signal 3 (cfo_to_ebitda < 60%, CFS #3) ─────────────────

    def test_checker2_cfo_ebitda_threshold_in_code(self, fe_fn_block, spec):
        """cfo_to_ebitda < 60.0 literal must appear in compute_schilit_forensic_score (Schilit Ch.12 CFS#3)."""
        val = spec["checker_2_cfs"]["signal_3_cfo_ebitda_weakness"]["threshold"]
        pattern = rf"<\s*{re.escape(str(float(val)))}"
        assert re.search(pattern, fe_fn_block), \
            f"CFS Signal 3 threshold <{val} not found in function"

    def test_checker2_cfo_ebitda_column_used(self, fe_fn_block, spec):
        """'cfo_to_ebitda' column must be accessed in the function for CFS Signal 3."""
        col = spec["checker_2_cfs"]["signal_3_cfo_ebitda_weakness"]["column"]
        assert col in fe_fn_block, f"CFS Signal 3 column '{col}' not referenced in function"

    def test_checker2_cfs_financial_exclusion_present(self, fe_fn_block):
        """CFS Signal 3 must exclude financial stocks — _is_fin_g must appear in checker 2 context."""
        # The function must reference _is_fin_g (shared financial exclusion mask computed at function top)
        assert "_is_fin_g" in fe_fn_block, \
            "Financial exclusion mask '_is_fin_g' not found in function — CFS Signal 3 must exclude financials"

    # ── v1.3: Checker 1 Signal 6 (depreciation-rate manipulation, EMS #4) ───────

    def test_checker1_signal6_fa_growth_threshold_in_code(self, fe_fn_block, spec):
        """Fixed-asset growth threshold 1.05 literal must appear in function (Schilit Ch.6 EMS#4 Qwest/WorldCom)."""
        val = spec["checker_1_ems"]["signal_6_depreciation_rate_manipulation"]["threshold_fa_growth"]
        # Appears as: _fa_scht > _fa_1yb_scht * 1.05
        pattern = rf"\*\s*{re.escape(str(float(val)))}"
        assert re.search(pattern, fe_fn_block), \
            f"Signal 6 FA growth threshold * {val} not found in compute_schilit_forensic_score"

    def test_checker1_signal6_dep_rate_fall_threshold_in_code(self, fe_fn_block, spec):
        """Dep-rate fall threshold 0.80 literal must appear in function (dep/FA fell >20% YoY)."""
        val = spec["checker_1_ems"]["signal_6_depreciation_rate_manipulation"]["threshold_dep_rate_fall"]
        pattern = rf"\*\s*{re.escape(str(float(val)))}"
        assert re.search(pattern, fe_fn_block), \
            f"Signal 6 dep-rate fall threshold * {val} not found in compute_schilit_forensic_score"

    def test_checker1_dep_rate_column_used(self, fe_fn_block, spec):
        """'dep_rate' column must be accessed in the function for Signal 6."""
        col = spec["checker_1_ems"]["signal_6_depreciation_rate_manipulation"]["column_c"]
        assert col in fe_fn_block, f"Signal 6 column '{col}' not referenced in function"

    def test_checker1_dep_rate_1yb_column_used(self, fe_fn_block, spec):
        """'dep_rate_1yb' column must be accessed in the function for Signal 6."""
        col = spec["checker_1_ems"]["signal_6_depreciation_rate_manipulation"]["column_d"]
        assert col in fe_fn_block, f"Signal 6 column '{col}' not referenced in function"

    def test_checker1_fixed_assets_column_used(self, fe_fn_block, spec):
        """'fixed_assets' column must be accessed in the function for Signal 6."""
        col = spec["checker_1_ems"]["signal_6_depreciation_rate_manipulation"]["column_a"]
        assert col in fe_fn_block, f"Signal 6 column '{col}' not referenced in function"

    def test_checker1_fixed_assets_1yb_column_used(self, fe_fn_block, spec):
        """'fixed_assets_1yb' column must be accessed in the function for Signal 6."""
        col = spec["checker_1_ems"]["signal_6_depreciation_rate_manipulation"]["column_b"]
        assert col in fe_fn_block, f"Signal 6 column '{col}' not referenced in function"

    def test_checker1_signal6_dep_rate_manip_var_in_function(self, fe_fn_block):
        """_dep_rate_manip intermediate variable must exist in the function body."""
        assert "_dep_rate_manip" in fe_fn_block, \
            "_dep_rate_manip variable not found in function — Signal 6 mask must be named _dep_rate_manip"

    def test_checker1_ems_mask_has_six_signals(self, fe_fn_block):
        """ems_revenue_expense_manipulation must be built from 6 OR-chained signals."""
        # All 6 signal variable names must appear in the function
        for var in ["_accruals_warn", "_inv_rev_gap", "_opm_stab",
                    "_other_income_inflate", "_expense_cap_proxy", "_dep_rate_manip"]:
            assert var in fe_fn_block, \
                f"EMS mask missing signal variable '{var}' — EMS must have 6 signals total"


# ═══════════════════════════════════════════════════════════════════════════════
# CLASS 6: OUTPUT COLUMNS PRODUCED ON LIVE DATA
# ═══════════════════════════════════════════════════════════════════════════════

class TestSchilitOutputColumns:
    """Run compute_schilit_forensic_score on controlled DataFrames and verify output shape."""

    _REQUIRED_COLS = [
        "schilit_forensic_score", "schilit_pass",
        "schilit_ems_flag", "schilit_cfs_flag",
        "schilit_kms_lev_flag", "schilit_kms_bloat_flag",
    ]

    def test_all_output_columns_present(self):
        from core.forensic_engine import compute_schilit_forensic_score
        result = compute_schilit_forensic_score(_make_minimal_df())
        for col in self._REQUIRED_COLS:
            assert col in result.columns, f"Missing output column: {col}"

    def test_score_column_is_float(self):
        from core.forensic_engine import compute_schilit_forensic_score
        result = compute_schilit_forensic_score(_make_minimal_df())
        assert result["schilit_forensic_score"].dtype == float

    def test_pass_column_is_int(self):
        from core.forensic_engine import compute_schilit_forensic_score
        result = compute_schilit_forensic_score(_make_minimal_df())
        assert result["schilit_pass"].dtype in [int, np.int32, np.int64]

    def test_flag_columns_are_binary(self):
        from core.forensic_engine import compute_schilit_forensic_score
        result = compute_schilit_forensic_score(_make_minimal_df(10))
        for col in ["schilit_ems_flag", "schilit_cfs_flag",
                    "schilit_kms_lev_flag", "schilit_kms_bloat_flag"]:
            unique_vals = set(result[col].unique())
            assert unique_vals.issubset({0, 1}), \
                f"Column {col} has non-binary values: {unique_vals}"

    def test_score_within_valid_range(self):
        from core.forensic_engine import compute_schilit_forensic_score
        result = compute_schilit_forensic_score(_make_minimal_df(10))
        assert result["schilit_forensic_score"].between(0, 100).all(), \
            "schilit_forensic_score out of [0, 100] range"

    def test_no_nan_in_score_column_with_complete_data(self):
        from core.forensic_engine import compute_schilit_forensic_score
        result = compute_schilit_forensic_score(_make_minimal_df(10))
        assert result["schilit_forensic_score"].notna().all(), \
            "NaN found in schilit_forensic_score with complete input data"

    def test_clean_stock_scores_100(self):
        """A stock with no manipulation signals must score exactly 100."""
        from core.forensic_engine import compute_schilit_forensic_score
        df = _make_minimal_df(1)
        result = compute_schilit_forensic_score(df)
        assert result["schilit_forensic_score"].iloc[0] == 100.0

    def test_clean_stock_passes(self):
        from core.forensic_engine import compute_schilit_forensic_score
        result = compute_schilit_forensic_score(_make_minimal_df(1))
        assert result["schilit_pass"].iloc[0] == 1

    def test_output_index_preserved(self):
        from core.forensic_engine import compute_schilit_forensic_score
        df = _make_minimal_df(5)
        result = compute_schilit_forensic_score(df)
        assert list(result.index) == list(df.index)


# ═══════════════════════════════════════════════════════════════════════════════
# CLASS 7: SCORING LOGIC VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestSchilitScoringLogic:
    """Controlled fixture tests verifying exact score arithmetic."""

    def _run(self, overrides: dict) -> pd.DataFrame:
        from core.forensic_engine import compute_schilit_forensic_score
        df = _make_minimal_df(1)
        for col, val in overrides.items():
            df[col] = val
        return compute_schilit_forensic_score(df)

    def test_one_checker_deducts_15(self):
        """Firing only EMS checker → score = 85."""
        result = self._run({"accruals_warning": 1})
        assert result["schilit_forensic_score"].iloc[0] == 85.0

    def test_two_checkers_deduct_30(self):
        """Firing EMS + CFS checkers → score = 70, schilit_pass = 1."""
        result = self._run({
            "accruals_warning": 1,
            "cash_machine_label": "📄 Paper Profits",
        })
        assert result["schilit_forensic_score"].iloc[0] == 70.0
        assert result["schilit_pass"].iloc[0] == 1

    def test_three_checkers_deduct_45(self):
        """Firing EMS + CFS + KMS-lev → score = 55, schilit_pass = 0."""
        result = self._run({
            "accruals_warning": 1,
            "cash_machine_label": "📄 Paper Profits",
            "hidden_obligation_growth": 1,
        })
        assert result["schilit_forensic_score"].iloc[0] == 55.0
        assert result["schilit_pass"].iloc[0] == 0

    def test_all_four_checkers_deduct_60(self):
        """All 4 checkers fire → score = 40, schilit_pass = 0."""
        result = self._run({
            "accruals_warning": 1,
            "cash_machine_label": "📄 Paper Profits",
            "hidden_obligation_growth": 1,
            "dso_delta_3y": 20.0,         # > 15.0 threshold
            "is_financial": False,
            "sector": "Manufacturing",
        })
        assert result["schilit_forensic_score"].iloc[0] == 40.0
        assert result["schilit_pass"].iloc[0] == 0

    def test_pass_boundary_at_70(self):
        """Score exactly 70 must pass (>= threshold, not > threshold)."""
        result = self._run({
            "accruals_warning": 1,
            "cash_machine_label": "📄 Paper Profits",
        })
        assert result["schilit_forensic_score"].iloc[0] == 70.0
        assert result["schilit_pass"].iloc[0] == 1, "Score == 70 must pass (>= not >)"

    def test_inv_vs_rev_gap_fires_at_20_01(self):
        """inv_vs_rev_gap = 20.01 must fire checker 1 (strict >)."""
        result = self._run({"inv_vs_rev_gap": 20.01})
        assert result["schilit_ems_flag"].iloc[0] == 1

    def test_inv_vs_rev_gap_does_not_fire_at_exactly_20(self):
        """inv_vs_rev_gap = 20.0 must NOT fire checker 1 (strict >, not >=)."""
        result = self._run({"inv_vs_rev_gap": 20.0})
        # Only inv_vs_rev_gap = 20.0 — strict > means it does NOT fire
        # (other signals are clean)
        df_check = _make_minimal_df(1)
        df_check["inv_vs_rev_gap"] = 20.0
        from core.forensic_engine import compute_schilit_forensic_score
        res = compute_schilit_forensic_score(df_check)
        assert res["schilit_ems_flag"].iloc[0] == 0, \
            "inv_vs_rev_gap == 20.0 must NOT fire (strict >)"

    def test_opm_stability_fires_at_30_01(self):
        """opm_stability = 30.01 must fire checker 1."""
        result = self._run({"opm_stability": 30.01})
        assert result["schilit_ems_flag"].iloc[0] == 1

    def test_cfs_divergence_requires_both_conditions(self):
        """pat_gr_yoy > 15 alone must NOT fire CFS (requires ALSO ocf_growth < -15)."""
        result = self._run({"pat_gr_yoy": 20.0, "ocf_growth": 5.0})  # OCF growing too
        assert result["schilit_cfs_flag"].iloc[0] == 0

    def test_cfs_divergence_fires_with_both(self):
        """pat_gr_yoy > 15 AND ocf_growth < -15 must fire CFS."""
        result = self._run({"pat_gr_yoy": 20.0, "ocf_growth": -20.0})
        assert result["schilit_cfs_flag"].iloc[0] == 1

    def test_checker4_does_not_fire_for_it_sector(self):
        """IT - Software sector must NOT fire checker 4 regardless of DSO."""
        result = self._run({
            "sector": "IT - Software",
            "dso_delta_3y": 50.0,        # would fire if not excluded
            "inventory_days_change": 40.0,
        })
        assert result["schilit_kms_bloat_flag"].iloc[0] == 0

    def test_checker4_does_not_fire_for_financial_stocks(self):
        """is_financial = True must exclude checker 4."""
        result = self._run({
            "is_financial": True,
            "dso_delta_3y": 50.0,
        })
        assert result["schilit_kms_bloat_flag"].iloc[0] == 0

    def test_checker4_fires_for_manufacturing_over_threshold(self):
        """Manufacturing sector with dso_delta_3y > 15 must fire checker 4."""
        result = self._run({
            "sector": "Manufacturing",
            "is_financial": False,
            "dso_delta_3y": 16.0,
        })
        assert result["schilit_kms_bloat_flag"].iloc[0] == 1

    # ── Signal 3: Absolute DSO > 90 days (added v1.1 after full book reading) ──

    def test_checker4_absolute_dso_fires_at_91(self):
        """days_receivable = 91 for Manufacturing must fire checker 4 (Schilit Ch.3: CA=247d, Ch.4: Hanergy=500d)."""
        result = self._run({
            "sector": "Manufacturing",
            "is_financial": False,
            "days_receivable": 91.0,
            "dso_delta_3y": 0.0,        # ensure only Signal 3 fires
            "inventory_days_change": 0.0,
        })
        assert result["schilit_kms_bloat_flag"].iloc[0] == 1, \
            "Absolute DSO=91 must fire checker 4 for Manufacturing sector"

    def test_checker4_absolute_dso_does_not_fire_at_exactly_90(self):
        """days_receivable = 90.0 must NOT fire (strict >, not >=) — boundary test."""
        result = self._run({
            "sector": "Manufacturing",
            "is_financial": False,
            "days_receivable": 90.0,
            "dso_delta_3y": 0.0,
            "inventory_days_change": 0.0,
        })
        assert result["schilit_kms_bloat_flag"].iloc[0] == 0, \
            "Absolute DSO=90.0 must NOT fire (strict > threshold, boundary case)"

    def test_checker4_absolute_dso_does_not_fire_for_it_software(self):
        """IT - Software sector must NOT fire Signal 3 even with DSO=200d (structural exclusion)."""
        result = self._run({
            "sector": "IT - Software",
            "is_financial": False,
            "days_receivable": 200.0,   # extreme DSO, but excluded for IT sector
            "dso_delta_3y": 0.0,
            "inventory_days_change": 0.0,
        })
        assert result["schilit_kms_bloat_flag"].iloc[0] == 0, \
            "IT - Software must not fire checker 4 even with DSO=200 (60-90d is structural in IT)"

    def test_checker4_absolute_dso_does_not_fire_for_pharmaceuticals(self):
        """Pharmaceuticals sector must NOT fire Signal 3 even with DSO=150d."""
        result = self._run({
            "sector": "Pharmaceuticals",
            "is_financial": False,
            "days_receivable": 150.0,
            "dso_delta_3y": 0.0,
            "inventory_days_change": 0.0,
        })
        assert result["schilit_kms_bloat_flag"].iloc[0] == 0

    def test_checker4_absolute_dso_does_not_fire_for_financial(self):
        """Financial stocks (is_financial=True) must not fire Signal 3 even with DSO=500d."""
        result = self._run({
            "sector": "Banking",
            "is_financial": True,
            "days_receivable": 500.0,
        })
        assert result["schilit_kms_bloat_flag"].iloc[0] == 0

    def test_checker4_any_of_three_signals_fires_checker(self):
        """Each of the 3 signals independently fires checker 4 for Manufacturing sector."""
        from core.forensic_engine import compute_schilit_forensic_score
        base = {
            "sector": "Manufacturing",
            "is_financial": False,
            "dso_delta_3y": 0.0,
            "inventory_days_change": 0.0,
            "days_receivable": 30.0,
        }
        # Signal 1: dso_delta_3y
        r1 = self._run({**base, "dso_delta_3y": 20.0})
        assert r1["schilit_kms_bloat_flag"].iloc[0] == 1, "Signal 1 (dso_delta_3y) should fire"
        # Signal 2: inventory_days_change
        r2 = self._run({**base, "inventory_days_change": 25.0})
        assert r2["schilit_kms_bloat_flag"].iloc[0] == 1, "Signal 2 (inventory_days_change) should fire"
        # Signal 3: days_receivable
        r3 = self._run({**base, "days_receivable": 95.0})
        assert r3["schilit_kms_bloat_flag"].iloc[0] == 1, "Signal 3 (days_receivable) should fire"

    def test_checker4_nan_days_receivable_does_not_fire(self):
        """NaN days_receivable must not fire Signal 3 (fillna(0) → 0 > 90 is False)."""
        from core.forensic_engine import compute_schilit_forensic_score
        df = _make_minimal_df(1)
        df["days_receivable"] = np.nan
        df["dso_delta_3y"] = 0.0
        df["inventory_days_change"] = 0.0
        df["sector"] = "Manufacturing"
        df["is_financial"] = False
        result = compute_schilit_forensic_score(df)
        assert result["schilit_kms_bloat_flag"].iloc[0] == 0, \
            "NaN days_receivable must not fire checker 4 (fillna(0) is safe)"

    # ── v1.2: Checker 1 Signal 4 — Other Income Inflate (EMS #3, Schilit Ch.5) ──

    def test_checker1_signal4_fires_when_npm_exceeds_opm_by_6pp(self):
        """npm=16, opm=10 → npm−opm=6 > 5.0 → EMS checker must fire (Schilit Ch.5 EMS#3)."""
        result = self._run({"npm": 16.0, "opm": 10.0})
        assert result["schilit_ems_flag"].iloc[0] == 1, \
            "NPM−OPM=6pp should fire Checker 1 (non-operating income inflating PAT)"

    def test_checker1_signal4_does_not_fire_at_exactly_5pp(self):
        """npm=15, opm=10 → npm−opm=5.0 must NOT fire (strict >, not >=)."""
        result = self._run({"npm": 15.0, "opm": 10.0,
                            "accruals_warning": 0, "inv_vs_rev_gap": 0.0, "opm_stability": 0.0})
        assert result["schilit_ems_flag"].iloc[0] == 0, \
            "NPM−OPM=5.0 must NOT fire (strict > 5.0, boundary case)"

    def test_checker1_signal4_does_not_fire_when_npm_lt_opm(self):
        """npm=5, opm=10 → npm−opm=−5 < 5.0 → must NOT fire (operations > net income, clean)."""
        result = self._run({"npm": 5.0, "opm": 10.0})
        assert result["schilit_ems_flag"].iloc[0] == 0, \
            "NPM < OPM (operations profitable, low non-op income) must not fire checker 1"

    def test_checker1_signal4_nan_npm_does_not_fire(self):
        """NaN npm → NaN − opm = NaN > 5.0 = False (pandas semantics) — must not fire."""
        from core.forensic_engine import compute_schilit_forensic_score
        df = _make_minimal_df(1)
        df["npm"] = np.nan
        df["accruals_warning"] = 0
        df["inv_vs_rev_gap"] = 0.0
        df["opm_stability"] = 0.0
        result = compute_schilit_forensic_score(df)
        assert result["schilit_ems_flag"].iloc[0] == 0, \
            "NaN npm must not fire Signal 4 (NaN arithmetic → NaN > 5.0 = False)"

    # ── v1.2: Checker 1 Signal 5 — Expense Cap Proxy (EMS #4, Schilit Ch.6) ─────

    def test_checker1_signal5_fires_when_both_conditions_met(self):
        """AT declines 0.15 AND OPM improves 3pp → AOL pattern → EMS checker must fire (Schilit Ch.6)."""
        result = self._run({
            "asset_turnover":     1.80,   # current AT
            "asset_turnover_1yb": 1.95,   # 1YB AT — decline = 0.15 > 0.10
            "opm":                13.0,   # current OPM
            "opm_1yb":            10.0,   # 1YB OPM — improvement = 3.0 > 2.0
            "accruals_warning": 0, "inv_vs_rev_gap": 0.0, "opm_stability": 0.0, "npm": 5.0,
        })
        assert result["schilit_ems_flag"].iloc[0] == 1, \
            "AT decline 0.15 + OPM improvement 3pp should fire Checker 1 (expense capitalisation proxy)"

    def test_checker1_signal5_requires_both_conditions_at_alone_insufficient(self):
        """AT declines 0.15 but OPM FLAT → must NOT fire (requires BOTH conditions, Schilit Ch.6)."""
        result = self._run({
            "asset_turnover":     1.80,
            "asset_turnover_1yb": 1.95,   # AT decline = 0.15
            "opm":                10.0,   # OPM improvement = 0.0 < 2.0 — flat
            "opm_1yb":            10.0,
            "accruals_warning": 0, "inv_vs_rev_gap": 0.0, "opm_stability": 0.0, "npm": 5.0,
        })
        assert result["schilit_ems_flag"].iloc[0] == 0, \
            "AT decline alone (OPM flat) must NOT fire Signal 5 (both conditions required)"

    def test_checker1_signal5_requires_both_conditions_opm_alone_insufficient(self):
        """OPM improves 3pp but AT FLAT → must NOT fire (requires BOTH conditions, Schilit Ch.6)."""
        result = self._run({
            "asset_turnover":     2.0,    # AT flat — no decline
            "asset_turnover_1yb": 2.0,
            "opm":                13.0,   # OPM improvement = 3.0 > 2.0
            "opm_1yb":            10.0,
            "accruals_warning": 0, "inv_vs_rev_gap": 0.0, "opm_stability": 0.0, "npm": 5.0,
        })
        assert result["schilit_ems_flag"].iloc[0] == 0, \
            "OPM improvement alone (AT flat) must NOT fire Signal 5 (both conditions required)"

    def test_checker1_signal5_sub_threshold_at_decline_does_not_fire(self):
        """AT declines 0.08 (clearly below 0.10 threshold) must NOT fire Signal 5.

        Note: Testing exact float boundary (e.g. 2.10 - 2.0) is unreliable because
        binary floating point makes 2.10 - 2.0 = 0.1000000000000000888... > 0.10.
        This test uses 2.20 - 2.12 = 0.08 which is clearly sub-threshold.
        """
        result = self._run({
            "asset_turnover":     2.12,
            "asset_turnover_1yb": 2.20,   # AT decline = 0.08 < 0.10 threshold — must NOT fire
            "opm":                13.0,
            "opm_1yb":            10.0,
            "accruals_warning": 0, "inv_vs_rev_gap": 0.0, "opm_stability": 0.0, "npm": 5.0,
        })
        assert result["schilit_ems_flag"].iloc[0] == 0, \
            "AT decline 0.08 (below 0.10 threshold) must NOT fire Signal 5"

    # ── v1.2: Checker 2 Signal 3 — CFO/EBITDA Weakness (CFS #3, Schilit Ch.12) ──

    def test_checker2_signal3_fires_below_60(self):
        """cfo_to_ebitda=55 for non-financial stock → CFS checker must fire (Schilit Ch.12 CFS#3)."""
        result = self._run({
            "cfo_to_ebitda": 55.0,
            "is_financial": False,
            "pat_gr_yoy": 0.0, "ocf_growth": 0.0, "cash_machine_label": "✅ Solid",
        })
        assert result["schilit_cfs_flag"].iloc[0] == 1, \
            "cfo_to_ebitda=55% must fire Checker 2 (OCF < 60% of EBITDA)"

    def test_checker2_signal3_does_not_fire_at_exactly_60(self):
        """cfo_to_ebitda=60.0 must NOT fire (strict <, not <=) — boundary test."""
        result = self._run({
            "cfo_to_ebitda": 60.0,
            "is_financial": False,
            "pat_gr_yoy": 0.0, "ocf_growth": 0.0, "cash_machine_label": "✅ Solid",
        })
        assert result["schilit_cfs_flag"].iloc[0] == 0, \
            "cfo_to_ebitda=60.0 must NOT fire (strict < threshold, boundary)"

    def test_checker2_signal3_does_not_fire_above_60(self):
        """cfo_to_ebitda=80 must NOT fire (well above 60% threshold)."""
        result = self._run({
            "cfo_to_ebitda": 80.0,
            "is_financial": False,
            "pat_gr_yoy": 0.0, "ocf_growth": 0.0, "cash_machine_label": "✅ Solid",
        })
        assert result["schilit_cfs_flag"].iloc[0] == 0, \
            "cfo_to_ebitda=80% must NOT fire (above 60% threshold)"

    def test_checker2_signal3_does_not_fire_for_financial_stocks(self):
        """is_financial=True must exclude CFO/EBITDA signal even at 30% (banks/NBFCs, Schilit CFS#3)."""
        result = self._run({
            "cfo_to_ebitda": 30.0,    # far below threshold, but financial stock
            "is_financial": True,
            "pat_gr_yoy": 0.0, "ocf_growth": 0.0, "cash_machine_label": "✅ Solid",
        })
        assert result["schilit_cfs_flag"].iloc[0] == 0, \
            "Financial stock must not fire CFS Signal 3 regardless of cfo_to_ebitda value"

    def test_checker2_signal3_nan_does_not_fire(self):
        """NaN cfo_to_ebitda → NaN < 60.0 = False (pandas semantics) — must not fire."""
        from core.forensic_engine import compute_schilit_forensic_score
        df = _make_minimal_df(1)
        df["cfo_to_ebitda"] = np.nan
        df["is_financial"] = False
        df["pat_gr_yoy"] = 0.0
        df["ocf_growth"] = 0.0
        df["cash_machine_label"] = "✅ Solid"
        result = compute_schilit_forensic_score(df)
        assert result["schilit_cfs_flag"].iloc[0] == 0, \
            "NaN cfo_to_ebitda must not fire CFS Signal 3 (NaN < 60.0 = False)"

    def test_checker2_signal3_clean_stock_still_passes_when_only_signal3_absent(self):
        """Base minimal_df (cfo_to_ebitda=80%) must remain clean with all 3 signals checked."""
        from core.forensic_engine import compute_schilit_forensic_score
        result = compute_schilit_forensic_score(_make_minimal_df(1))
        assert result["schilit_cfs_flag"].iloc[0] == 0, \
            "Base minimal_df (cfo_to_ebitda=80%) must not fire CFS checker with new Signal 3"
        assert result["schilit_forensic_score"].iloc[0] == 100.0, \
            "Base minimal_df must still score 100 after adding CFS Signal 3"

    def test_missing_columns_do_not_raise(self):
        """Function must handle empty DataFrame (no columns) without raising."""
        from core.forensic_engine import compute_schilit_forensic_score
        empty = pd.DataFrame(index=range(3))
        result = compute_schilit_forensic_score(empty)
        assert "schilit_forensic_score" in result.columns
        assert "schilit_pass" in result.columns

    def test_nan_inputs_do_not_propagate_to_score(self):
        """NaN in numeric columns must not produce NaN scores."""
        from core.forensic_engine import compute_schilit_forensic_score
        df = _make_minimal_df(3)
        df["inv_vs_rev_gap"] = np.nan
        df["opm_stability"] = np.nan
        df["pat_gr_yoy"] = np.nan
        df["dso_delta_3y"] = np.nan
        result = compute_schilit_forensic_score(df)
        assert result["schilit_forensic_score"].notna().all()

    def test_vectorized_all_rows_processed(self):
        """Result must have same row count as input."""
        from core.forensic_engine import compute_schilit_forensic_score
        df = _make_minimal_df(50)
        result = compute_schilit_forensic_score(df)
        assert len(result) == 50

    # ── v1.3: Checker 1 Signal 6 — Depreciation-Rate Manipulation (EMS #4, Schilit Ch.6/9) ──

    def test_checker1_signal6_fires_when_fa_grows_and_dep_rate_falls(self):
        """FA grows 10% AND dep_rate falls 25% → Qwest/WorldCom pattern → EMS checker must fire.

        Scenario: Company extends asset useful life from 10yr → 14yr.
        dep_rate falls from 10% → 7.1% (−29%, > 20% fall threshold).
        Fixed assets grow from 100 → 110 (+10%, > 5% growth threshold).
        Both conditions met → Signal 6 fires (Schilit Ch.6 EMS#4 + Ch.9 EMS#7).
        """
        result = self._run({
            "fixed_assets":     110.0,   # grew 10% vs 1YB=100 → growth > 5% threshold
            "fixed_assets_1yb": 100.0,
            "dep_rate":          7.0,    # fell from 10% → 7% = -30% fall, > 20% threshold
            "dep_rate_1yb":     10.0,
            "is_financial":     False,
            "accruals_warning": 0, "inv_vs_rev_gap": 0.0, "opm_stability": 0.0,
            "npm": 5.0, "asset_turnover": 2.0, "asset_turnover_1yb": 2.0,
            "opm": 10.0, "opm_1yb": 10.0,
        })
        assert result["schilit_ems_flag"].iloc[0] == 1, \
            "FA grew 10% + dep_rate fell 30% must fire Checker 1 (Qwest/WorldCom dep-rate manipulation)"

    def test_checker1_signal6_does_not_fire_fa_growth_alone(self):
        """FA grows 10% but dep_rate is STABLE → must NOT fire (both conditions required)."""
        result = self._run({
            "fixed_assets":     110.0,
            "fixed_assets_1yb": 100.0,
            "dep_rate":         10.0,   # dep_rate unchanged — no manipulation
            "dep_rate_1yb":     10.0,
            "is_financial": False,
            "accruals_warning": 0, "inv_vs_rev_gap": 0.0, "opm_stability": 0.0,
            "npm": 5.0, "asset_turnover": 2.0, "asset_turnover_1yb": 2.0,
            "opm": 10.0, "opm_1yb": 10.0,
        })
        assert result["schilit_ems_flag"].iloc[0] == 0, \
            "FA growth alone (dep_rate stable) must NOT fire Signal 6 (both conditions required)"

    def test_checker1_signal6_does_not_fire_dep_rate_fall_alone(self):
        """dep_rate falls 30% but FA is FLAT → must NOT fire (flat FA = declining asset base, not manipulation).

        When FA is flat and dep_rate falls, it could mean assets are aging out
        (smaller depreciable base), not life-extension manipulation. Both conditions required.
        """
        result = self._run({
            "fixed_assets":     100.0,   # FA flat — no growth
            "fixed_assets_1yb": 100.0,
            "dep_rate":          7.0,    # dep_rate fell 30% — but without FA growth, not the pattern
            "dep_rate_1yb":     10.0,
            "is_financial": False,
            "accruals_warning": 0, "inv_vs_rev_gap": 0.0, "opm_stability": 0.0,
            "npm": 5.0, "asset_turnover": 2.0, "asset_turnover_1yb": 2.0,
            "opm": 10.0, "opm_1yb": 10.0,
        })
        assert result["schilit_ems_flag"].iloc[0] == 0, \
            "dep_rate fall alone (FA flat, no growth > 5%) must NOT fire Signal 6"

    def test_checker1_signal6_does_not_fire_for_financial_stocks(self):
        """Financial stocks (is_financial=True) must be excluded even with extreme dep manipulation."""
        result = self._run({
            "fixed_assets":     200.0,   # FA doubled — extreme growth
            "fixed_assets_1yb": 100.0,
            "dep_rate":          1.0,    # dep_rate collapsed to 1% from 10% — extreme fall
            "dep_rate_1yb":     10.0,
            "is_financial":     True,    # ← financial exclusion must block the signal
            "accruals_warning": 0, "inv_vs_rev_gap": 0.0, "opm_stability": 0.0,
            "npm": 5.0, "asset_turnover": 2.0, "asset_turnover_1yb": 2.0,
            "opm": 10.0, "opm_1yb": 10.0,
        })
        assert result["schilit_ems_flag"].iloc[0] == 0, \
            "Financial stock must not fire Signal 6 (banks have trivially small FA vs revenue)"

    def test_checker1_signal6_does_not_fire_nan_dep_rate(self):
        """NaN dep_rate → NaN < dep_rate_1yb * 0.80 = False (pandas NaN semantics) — must not fire."""
        from core.forensic_engine import compute_schilit_forensic_score
        df = _make_minimal_df(1)
        df["fixed_assets"]     = 110.0   # FA grew > 5% (condition 1 met)
        df["fixed_assets_1yb"] = 100.0
        df["dep_rate"]         = np.nan  # dep_rate missing → NaN comparison → False → no fire
        df["dep_rate_1yb"]     = 10.0
        df["is_financial"]     = False
        result = compute_schilit_forensic_score(df)
        assert result["schilit_ems_flag"].iloc[0] == 0, \
            "NaN dep_rate must not fire Signal 6 (NaN < threshold = False in pandas)"

    def test_checker1_signal6_sub_threshold_fa_growth_does_not_fire(self):
        """FA grows only 4% (below 5% threshold) → Signal 6 must NOT fire even if dep_rate falls sharply.

        Threshold: fixed_assets > fixed_assets_1yb * 1.05. A 4% FA growth gives:
        fixed_assets (104) vs fixed_assets_1yb * 1.05 (100 * 1.05 = 105) → 104 < 105 → False.
        """
        result = self._run({
            "fixed_assets":     104.0,   # 4% growth — strictly below 5% threshold (104 < 100*1.05=105)
            "fixed_assets_1yb": 100.0,
            "dep_rate":          6.0,    # dep_rate fell 40% (extreme, below threshold) — but FA growth fails
            "dep_rate_1yb":     10.0,
            "is_financial": False,
            "accruals_warning": 0, "inv_vs_rev_gap": 0.0, "opm_stability": 0.0,
            "npm": 5.0, "asset_turnover": 2.0, "asset_turnover_1yb": 2.0,
            "opm": 10.0, "opm_1yb": 10.0,
        })
        assert result["schilit_ems_flag"].iloc[0] == 0, \
            "FA growth of 4% (< 5% threshold) must NOT fire Signal 6"

    def test_checker1_signal6_sub_threshold_dep_rate_fall_does_not_fire(self):
        """dep_rate falls only 15% (below 20% threshold) → Signal 6 must NOT fire.

        Threshold: dep_rate < dep_rate_1yb * 0.80. A 15% fall gives:
        dep_rate (8.5) vs dep_rate_1yb * 0.80 (10 * 0.80 = 8.0) → 8.5 > 8.0 → False.
        """
        result = self._run({
            "fixed_assets":     110.0,   # FA grew 10% — condition 1 met
            "fixed_assets_1yb": 100.0,
            "dep_rate":          8.5,    # fell only 15% from 10%: 8.5 vs threshold 8.0 → above threshold
            "dep_rate_1yb":     10.0,
            "is_financial": False,
            "accruals_warning": 0, "inv_vs_rev_gap": 0.0, "opm_stability": 0.0,
            "npm": 5.0, "asset_turnover": 2.0, "asset_turnover_1yb": 2.0,
            "opm": 10.0, "opm_1yb": 10.0,
        })
        assert result["schilit_ems_flag"].iloc[0] == 0, \
            "dep_rate fall of 15% (< 20% threshold, dep_rate=8.5 > dep_rate_1yb*0.80=8.0) must NOT fire"

    def test_checker1_signal6_clean_base_df_still_scores_100(self):
        """Base minimal_df (FA flat, dep_rate flat) must remain clean after adding Signal 6."""
        from core.forensic_engine import compute_schilit_forensic_score
        result = compute_schilit_forensic_score(_make_minimal_df(1))
        assert result["schilit_ems_flag"].iloc[0] == 0, \
            "Base minimal_df (Signal 6 safe defaults) must not fire EMS checker"
        assert result["schilit_forensic_score"].iloc[0] == 100.0, \
            "Base minimal_df must still score 100 after adding Signal 6 to EMS mask"


# ═══════════════════════════════════════════════════════════════════════════════
# CLASS 8: PIPELINE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestSchilitPipelineIntegration:
    """Verify Schilit is wired into run_forensic_analysis and fw_str."""

    def test_schilit_called_in_run_forensic_analysis(self, fe_source):
        """compute_schilit_forensic_score must be wired into the forensic pipeline.

        Architecture (post-2026-06-11): run_forensic_analysis delegates to
        compute_forensic_signals which contains the actual Schilit call.
        Verify the chain: compute_forensic_signals body contains the call,
        and run_forensic_analysis calls compute_forensic_signals.
        """
        # Verify compute_forensic_signals contains the Schilit call
        sig_match = re.search(
            r"def compute_forensic_signals\(.*?(?=\ndef |\nclass |\Z)",
            fe_source, re.DOTALL
        )
        assert sig_match, "compute_forensic_signals not found in forensic_engine.py"
        assert "compute_schilit_forensic_score" in sig_match.group(0), \
            "compute_schilit_forensic_score not called in compute_forensic_signals"

        # Verify run_forensic_analysis delegates to compute_forensic_signals
        rfa_match = re.search(
            r"def run_forensic_analysis\(.*?(?=\ndef |\nclass |\Z)",
            fe_source, re.DOTALL
        )
        assert rfa_match, "run_forensic_analysis not found in forensic_engine.py"
        assert "compute_forensic_signals" in rfa_match.group(0), \
            "run_forensic_analysis must call compute_forensic_signals"

    def test_schilit_called_after_compute_red_flags(self, fe_source):
        """Schilit must be called after compute_red_flags (ordering dependency)."""
        idx_red_flags = fe_source.find("df = compute_red_flags(df)")
        idx_schilit   = fe_source.find("df = compute_schilit_forensic_score(df)")
        assert idx_red_flags != -1, "compute_red_flags call not found"
        assert idx_schilit   != -1, "compute_schilit_forensic_score call not found"
        assert idx_schilit > idx_red_flags, \
            "compute_schilit_forensic_score must be called AFTER compute_red_flags"

    def test_schilit_not_duplicated_in_run_full_scoring(self, se_source):
        """SINGLE-PASS CONTRACT: run_full_scoring must NOT call Schilit itself.

        Architecture (post-2026-05-31): the forensic suite runs exactly once,
        in forensic_engine.run_forensic_analysis(), which app.get_scored_data()
        invokes AFTER run_full_scoring(). The old duplicate forensic shim inside
        run_full_scoring was surgically removed to stop double-computation on
        every Streamlit state change. This test guards against its reintroduction.
        """
        match = re.search(
            r"def run_full_scoring\(.*?(?=\ndef |\nclass |\Z)",
            se_source, re.DOTALL
        )
        assert match, "run_full_scoring not found in scoring_engine.py"
        fn_body = match.group(0)
        assert "compute_schilit_forensic_score" not in fn_body, (
            "run_full_scoring must NOT call compute_schilit_forensic_score. "
            "Forensic runs once in run_forensic_analysis (single-pass architecture)."
        )

    def test_fw_schilit_variable_defined(self, se_source):
        """fw_schilit variable must be defined in compute_qglp_score."""
        assert "fw_schilit" in se_source, \
            "fw_schilit not defined in scoring_engine.py"

    def test_financial_shenanigans_label_in_fw_str(self, se_source):
        """'Financial Shenanigans' string must appear in the fw_str construction."""
        assert "Financial Shenanigans" in se_source, \
            "'Financial Shenanigans' label not in fw_str in scoring_engine.py"

    def test_fw_schilit_uses_schilit_pass_column(self, se_source):
        """fw_schilit must read from schilit_pass column."""
        assert 'schilit_pass' in se_source, \
            "fw_schilit does not read schilit_pass column"

    def test_fw_str_includes_schilit_in_pipe_format(self, se_source):
        """The fw_str must have the pipe-separated label format."""
        assert '"Financial Shenanigans|"' in se_source or \
               "'Financial Shenanigans|'" in se_source, \
            "fw_str does not include 'Financial Shenanigans|' pipe-format label"

    def test_no_forensic_guard_block_in_run_full_scoring(self, se_source):
        """SINGLE-PASS CONTRACT: the legacy forensic guard must stay removed.

        run_full_scoring must not contain an `if "forensic_score" not in
        df.columns:` shim that re-computes the forensic suite. That early-call
        guard was deleted to avoid running the heavy forensic filters twice;
        run_forensic_analysis is the single source of forensic columns.
        """
        match = re.search(
            r"def run_full_scoring\(.*?(?=\ndef |\nclass |\Z)",
            se_source, re.DOTALL
        )
        assert match, "run_full_scoring not found in scoring_engine.py"
        fn_body = match.group(0)
        assert 'if "forensic_score" not in df.columns:' not in fn_body, (
            "A forensic guard block was re-added to run_full_scoring. Forensic must "
            "run only in run_forensic_analysis (single-pass) to prevent double-computation."
        )
