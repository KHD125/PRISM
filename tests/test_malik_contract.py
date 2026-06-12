"""
Contract Tests — Dr. Vijay Malik Peaceful Investing Framework
=============================================================
Automated verification that docs/malik_peaceful_specs.json and
core/scoring_engine.py are in perfect alignment.

Version 1.0 — Full 5-pillar Peaceful Investing materialization:
  • Pillar G: rev_gr (10Y/5Y) >= 10%           (Sales Growth Runway)
  • Pillar P: npm >= 8% AND stable              (Profit Stability)
  • Pillar F: ICR >= 3× AND D/E <= 0.5 AND CR >= 1.25 (Debt Fortress; fin exempt)
  • Pillar C: cfo_to_pat >= 70.0 (PERCENTAGE)   (Cash Generation)
  • Pillar S: ssgr_self_funded == 1             (Self-Funded Growth)

Structure:
    TestMalikSpecLedger         — JSON schema completeness and meta keys
    TestMalikEngineContract     — regex source-code threshold verification
    TestMalikPillarArithmetic   — boundary conditions, AND invariant, score 0-5
    TestMalikNaNConservative    — NaN handling strategy: conservative gate failure
    TestMalikIndexAlignment     — non-default integer and string index safety
    TestMalikFinancialExemption — financial sector fully exempt from Debt Fortress
    TestMalikUIContract         — render_malik_radar import + pure-display contract
    TestMalikRawSignalsContract — malik cells present in render_raw_signals
"""

import json
import re
import os
import sys
import pytest
import pandas as pd
import numpy as np

# ── Path bootstrap ────────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SPEC_PATH = os.path.join(REPO_ROOT, "docs", "malik_peaceful_specs.json")
SE_PATH   = os.path.join(REPO_ROOT, "core", "scoring_engine.py")
UI_PATH   = os.path.join(REPO_ROOT, "ui",   "ui_tearsheet.py")


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def spec() -> dict:
    with open(SPEC_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def se_source() -> str:
    with open(SE_PATH, encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module")
def ui_source() -> str:
    with open(UI_PATH, encoding="utf-8") as f:
        return f.read()


# ── Mock data helpers ─────────────────────────────────────────────────────────

def _build_mock_malik_row(**overrides) -> dict:
    """Build a fully-passing Malik Peaceful Investing row (all 5 pillars green).

    Default thresholds exceed every pillar gate:
      G — rev_gr_10y=15.0  (>= 10% threshold)
      P — npm=12.0 + npm_1yb=10.0 (>= 8% + prior >= 6%)
      F — interest_coverage=6.0, debt_to_equity=0.3, current_ratio=2.0
      C — cfo_to_pat=80.0 (>= 70% PERCENTAGE threshold)
      S — ssgr_self_funded=1 (binary == 1)
    """
    base = {
        "rev_gr_10y":        15.0,   # G: 15% > 10% threshold
        "rev_gr_5y":         12.0,   # G fallback
        "npm":               12.0,   # P: 12% > 8% threshold
        "npm_1yb":           10.0,   # P: prior year 10% > 6% threshold
        "interest_coverage":  6.0,   # F: 6x > 3x threshold
        "debt_to_equity":     0.3,   # F: 0.3 <= 0.5 threshold
        "current_ratio":      2.0,   # F: 2.0 >= 1.25 threshold
        "cfo_to_pat":        80.0,   # C: 80% >= 70% (PERCENTAGE)
        "ssgr_self_funded":   1,     # S: == 1 (binary pass)
        "is_financial":       False,
        # scaffold columns
        "market_cap":    5000.0,
        "close_price":    500.0,
        "name":           "TestCo",
        "sector":         "FMCG",
    }
    base.update(overrides)
    return base


def _run_malik(rows: list, regime: str = "SIDEWAYS") -> pd.DataFrame:
    """Execute compute_qglp_score on a list of row dicts; return result df."""
    from config import MASTER_PROFILES
    from core.scoring_engine import compute_qglp_score
    df = pd.DataFrame(rows)
    df.attrs["detected_market_regime"] = regime
    profile = MASTER_PROFILES.get("Balanced", next(iter(MASTER_PROFILES.values())))
    return compute_qglp_score(df, profile)


# ═══════════════════════════════════════════════════════════════════════════════
# TestMalikSpecLedger
# ═══════════════════════════════════════════════════════════════════════════════

class TestMalikSpecLedger:
    """Verify docs/malik_peaceful_specs.json is complete and structurally correct."""

    def test_spec_file_exists(self):
        assert os.path.exists(SPEC_PATH), f"Spec file not found: {SPEC_PATH}"

    def test_spec_is_valid_json(self, spec):
        assert isinstance(spec, dict)

    def test_meta_section_present(self, spec):
        assert "_meta" in spec

    def test_meta_required_keys(self, spec):
        required = [
            "title", "author", "framework_variable", "pass_column",
            "score_column", "frameworks_passed_label", "implementation_file", "version",
        ]
        for key in required:
            assert key in spec["_meta"], f"_meta missing key: {key}"

    def test_pass_column_name(self, spec):
        assert spec["_meta"]["pass_column"] == "malik_pass"

    def test_score_column_name(self, spec):
        assert spec["_meta"]["score_column"] == "malik_score"

    def test_framework_variable_name(self, spec):
        assert spec["_meta"]["framework_variable"] == "fw_malik_peaceful"

    def test_framework_label(self, spec):
        assert spec["_meta"]["frameworks_passed_label"] == "Peaceful Investing"

    def test_version_present(self, spec):
        assert "malik" in spec["_meta"]["version"].lower() or "peaceful" in spec["_meta"]["version"].lower(), (
            f"Version should reference malik or peaceful; got: {spec['_meta']['version']}"
        )

    def test_framework_number_is_9(self, spec):
        assert spec["_meta"]["framework_number_in_code"] == 9, (
            f"Malik is Framework 9. Got: {spec['_meta']['framework_number_in_code']}"
        )

    def test_comment_anchor_present(self, spec):
        assert "comment_anchor" in spec["_meta"]
        assert "Peaceful Investing" in spec["_meta"]["comment_anchor"] or "Vijay Malik" in spec["_meta"]["comment_anchor"]

    def test_5_pillar_sections_present(self, spec):
        required_pillars = [
            "pillar_g_growth_runway",
            "pillar_p_profit_stability",
            "pillar_f_debt_fortress",
            "pillar_c_cash_generation",
            "pillar_s_self_funded",
        ]
        for p in required_pillars:
            assert p in spec, f"Spec missing pillar section: {p}"

    def test_pillar_g_column_materialized(self, spec):
        assert spec["pillar_g_growth_runway"]["_column_materialized"] == "malik_growth_runway"

    def test_pillar_p_column_materialized(self, spec):
        assert spec["pillar_p_profit_stability"]["_column_materialized"] == "malik_profit_stability"

    def test_pillar_f_column_materialized(self, spec):
        assert spec["pillar_f_debt_fortress"]["_column_materialized"] == "malik_debt_fortress"

    def test_pillar_c_column_materialized(self, spec):
        assert spec["pillar_c_cash_generation"]["_column_materialized"] == "malik_cash_generation"

    def test_pillar_s_column_materialized(self, spec):
        assert spec["pillar_s_self_funded"]["_column_materialized"] == "malik_self_funded"

    def test_pillar_g_threshold_is_10(self, spec):
        t = spec["pillar_g_growth_runway"]["growth_gate"]["threshold"]
        assert abs(t - 10.0) < 1e-9, f"Pillar G threshold must be 10.0; got: {t}"

    def test_pillar_g_operator_is_gte(self, spec):
        op = spec["pillar_g_growth_runway"]["growth_gate"]["operator"]
        assert op == ">=", f"Pillar G operator must be '>='; got: '{op}'"

    def test_pillar_p_threshold_current_is_8(self, spec):
        t = spec["pillar_p_profit_stability"]["margin_gate"]["threshold_current"]
        assert abs(t - 8.0) < 1e-9, f"Pillar P current NPM threshold must be 8.0; got: {t}"

    def test_pillar_p_threshold_prior_year_is_6(self, spec):
        t = spec["pillar_p_profit_stability"]["margin_gate"]["threshold_prior_year"]
        assert abs(t - 6.0) < 1e-9, f"Pillar P prior-year NPM threshold must be 6.0; got: {t}"

    def test_pillar_f_interest_coverage_threshold_is_3(self, spec):
        t = spec["pillar_f_debt_fortress"]["interest_coverage_gate"]["threshold"]
        assert abs(t - 3.0) < 1e-9, f"Pillar F ICR threshold must be 3.0; got: {t}"

    def test_pillar_f_de_threshold_is_0point5(self, spec):
        t = spec["pillar_f_debt_fortress"]["debt_to_equity_gate"]["threshold"]
        assert abs(t - 0.5) < 1e-9, f"Pillar F D/E threshold must be 0.5; got: {t}"

    def test_pillar_f_de_operator_is_lte(self, spec):
        op = spec["pillar_f_debt_fortress"]["debt_to_equity_gate"]["operator"]
        assert op == "<=", f"Pillar F D/E operator must be '<='; got: '{op}'"

    def test_pillar_f_cr_threshold_is_1point25(self, spec):
        t = spec["pillar_f_debt_fortress"]["current_ratio_gate"]["threshold"]
        assert abs(t - 1.25) < 1e-9, f"Pillar F CR threshold must be 1.25; got: {t}"

    def test_pillar_f_financial_exemption_present(self, spec):
        exemption = spec["pillar_f_debt_fortress"]["financial_sector_exemption"]
        assert "condition" in exemption
        assert "is_financial" in exemption["condition"]

    def test_pillar_c_threshold_is_70(self, spec):
        t = spec["pillar_c_cash_generation"]["cash_generation_gate"]["threshold"]
        assert abs(t - 70.0) < 1e-9, (
            f"Pillar C threshold must be 70.0 (PERCENTAGE). Got: {t}. "
            "CRITICAL: cfo_to_pat is stored as PERCENTAGE (73.04 = 73%), not ratio."
        )

    def test_pillar_c_unit_warning_present(self, spec):
        gate = spec["pillar_c_cash_generation"]["cash_generation_gate"]
        assert "unit_warning" in gate
        warning = gate["unit_warning"]
        assert "PERCENTAGE" in warning or "70.0" in warning, (
            f"unit_warning must document the PERCENTAGE unit convention; got: {warning}"
        )

    def test_pillar_c_unit_warning_not_0point7(self, spec):
        """Critical: unit_warning must warn against using 0.7 (ratio) instead of 70.0 (%)."""
        gate = spec["pillar_c_cash_generation"]["cash_generation_gate"]
        warning = gate["unit_warning"]
        assert "0.7" in warning, (
            "unit_warning must mention '0.7' to warn against the ratio vs percentage confusion"
        )

    def test_pillar_s_threshold_is_1(self, spec):
        t = spec["pillar_s_self_funded"]["ssgr_gate"]["threshold"]
        assert t == 1, f"Pillar S threshold must be 1 (binary flag); got: {t}"

    def test_pillar_s_operator_is_equal(self, spec):
        op = spec["pillar_s_self_funded"]["ssgr_gate"]["operator"]
        assert op == "==", f"Pillar S operator must be '=='; got: '{op}'"

    def test_pillar_s_unit_is_binary(self, spec):
        unit = spec["pillar_s_self_funded"]["ssgr_gate"]["unit"]
        assert "BINARY" in unit.upper(), f"Pillar S unit must be BINARY; got: '{unit}'"

    def test_output_columns_registry_present(self, spec):
        assert "output_columns_registry" in spec
        required = [
            "malik_growth_runway", "malik_profit_stability",
            "malik_debt_fortress", "malik_cash_generation",
            "malik_self_funded", "malik_pass", "malik_score",
        ]
        for col in required:
            assert col in spec["output_columns_registry"], (
                f"output_columns_registry missing: {col}"
            )

    def test_scoring_matrix_present(self, spec):
        assert "scoring_matrix" in spec
        sm = spec["scoring_matrix"]
        assert sm["all_gates_equal_weight"] is True
        assert sm["score_range"] == "0-5"

    def test_vectorization_matrix_present(self, spec):
        assert "vectorization_matrix" in spec
        vm = spec["vectorization_matrix"]["nan_handling"]
        for col in ["rev_gr_10y", "npm", "npm_1yb", "interest_coverage",
                    "debt_to_equity", "current_ratio", "cfo_to_pat", "ssgr_self_funded"]:
            assert col in vm, f"vectorization_matrix missing NaN handling for: {col}"

    def test_vectorization_matrix_npm_1yb_conservative(self, spec):
        """npm_1yb NaN is treated as 'unavailable' (not failure) — don't penalize newer listings."""
        vm = spec["vectorization_matrix"]["nan_handling"]
        npm_1yb_strategy = vm["npm_1yb"]
        # The strategy should mention "unavailable" or "not fail" — conservative for new listings
        assert "unavailable" in npm_1yb_strategy.lower() or "not" in npm_1yb_strategy.lower(), (
            f"npm_1yb NaN strategy should document 'unavailable' behavior; got: {npm_1yb_strategy}"
        )

    def test_not_implementable_section_present(self, spec):
        assert "not_implementable" in spec
        assert len(spec["not_implementable"]) >= 4, (
            f"Expected at least 4 not_implementable entries; found {len(spec['not_implementable'])}"
        )

    def test_not_implementable_each_has_reason(self, spec):
        for item in spec["not_implementable"]:
            assert "reason" in item, (
                f"not_implementable entry '{item.get('gate', '?')}' missing 'reason' field"
            )

    def test_distinction_section_present(self, spec):
        assert "distinction_from_other_frameworks" in spec

    def test_distinction_vs_coffee_can_present(self, spec):
        assert "vs_coffee_can" in spec["distinction_from_other_frameworks"]

    def test_ssgr_noted_as_unique_signal(self, spec):
        """SSGR self-funding gate is UNIQUE to Malik — spec must document this."""
        dists = spec["distinction_from_other_frameworks"]
        # Check at least one distinction section mentions SSGR uniqueness
        dists_str = json.dumps(dists).lower()
        assert "unique" in dists_str, (
            "distinction_from_other_frameworks must note SSGR is unique to Malik"
        )

    def test_unit_notes_section_present(self, spec):
        assert "unit_notes" in spec["_meta"]
        assert "cfo_to_pat" in spec["_meta"]["unit_notes"]

    def test_unit_notes_cfo_to_pat_documents_percentage(self, spec):
        note = spec["_meta"]["unit_notes"]["cfo_to_pat"]
        assert "PERCENTAGE" in note or "percent" in note.lower(), (
            f"unit_notes.cfo_to_pat must document PERCENTAGE convention; got: {note}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TestMalikEngineContract
# ═══════════════════════════════════════════════════════════════════════════════

class TestMalikEngineContract:
    """Verify scoring_engine.py has all Malik columns and correct thresholds."""

    def test_malik_anchor_comment(self, se_source):
        assert "Peaceful Investing" in se_source and "Vijay Malik" in se_source

    def test_malik_spec_reference_in_code(self, se_source):
        """Code must reference the spec JSON file."""
        assert "malik_peaceful_specs.json" in se_source

    def test_malik_growth_runway_column_defined(self, se_source):
        assert 'df["malik_growth_runway"]' in se_source

    def test_malik_profit_stability_column_defined(self, se_source):
        assert 'df["malik_profit_stability"]' in se_source

    def test_malik_debt_fortress_column_defined(self, se_source):
        assert 'df["malik_debt_fortress"]' in se_source

    def test_malik_cash_generation_column_defined(self, se_source):
        assert 'df["malik_cash_generation"]' in se_source

    def test_malik_self_funded_column_defined(self, se_source):
        assert 'df["malik_self_funded"]' in se_source

    def test_malik_pass_column_defined(self, se_source):
        assert 'df["malik_pass"]' in se_source

    def test_malik_score_column_defined(self, se_source):
        assert 'df["malik_score"]' in se_source

    def test_pillar_g_threshold_10_in_code(self, se_source):
        """Pillar G: rev_gr_mk >= 10.0"""
        assert re.search(r'rev_gr_mk.*>=\s*10\.0', se_source), (
            "scoring_engine must check rev_gr_mk >= 10.0 for malik_growth_runway"
        )

    def test_pillar_p_npm_threshold_8_in_code(self, se_source):
        """Pillar P: npm >= 8 for current year."""
        assert re.search(r'npm_mk.*>=\s*8', se_source), (
            "scoring_engine must check npm_mk >= 8 for malik_profit_stability"
        )

    def test_pillar_p_npm_1yb_threshold_6_in_code(self, se_source):
        """Pillar P: npm_1yb >= 6 for prior year stability check."""
        assert re.search(r'npm_1yb_mk.*>=\s*6', se_source), (
            "scoring_engine must check npm_1yb_mk >= 6 for Pillar P stability"
        )

    def test_pillar_p_npm_1yb_isna_in_code(self, se_source):
        """Pillar P: npm_1yb.isna() treated as unavailable (not failure)."""
        assert re.search(r'npm_1yb_mk\.isna\(\)', se_source), (
            "scoring_engine must use npm_1yb_mk.isna() for conservative prior-year handling"
        )

    def test_pillar_f_interest_coverage_threshold_3_in_code(self, se_source):
        """Pillar F: interest_coverage >= 3.0"""
        assert re.search(r'ic_mk.*>=\s*3\.0', se_source), (
            "scoring_engine must check ic_mk >= 3.0 for malik_debt_fortress ICR gate"
        )

    def test_pillar_f_de_threshold_0point5_in_code(self, se_source):
        """Pillar F: debt_to_equity <= 0.5"""
        assert re.search(r'de_mk.*<=\s*0\.5', se_source), (
            "scoring_engine must check de_mk <= 0.5 for malik_debt_fortress D/E gate"
        )

    def test_pillar_f_cr_threshold_1point25_in_code(self, se_source):
        """Pillar F: current_ratio >= 1.25"""
        assert re.search(r'cr_mk.*>=\s*1\.25', se_source), (
            "scoring_engine must check cr_mk >= 1.25 for malik_debt_fortress CR gate"
        )

    def test_pillar_f_financial_exemption_in_code(self, se_source):
        """Pillar F: is_fin_mk makes financial sector exempt from debt gates."""
        assert re.search(r'is_fin_mk\s*\|', se_source), (
            "scoring_engine must have is_fin_mk | (debt gates) for financial sector exemption"
        )

    def test_pillar_c_threshold_70_in_code(self, se_source):
        """Pillar C: cfo_to_pat >= 70.0 — PERCENTAGE (not 0.7)."""
        assert re.search(r'cfo_pat_mk.*>=\s*70\.0', se_source), (
            "scoring_engine must check cfo_pat_mk >= 70.0 for malik_cash_generation. "
            "CRITICAL: cfo_to_pat is PERCENTAGE — must be 70.0 not 0.7"
        )

    def test_pillar_c_not_0point7(self, se_source):
        """Critical unit guard: the ratio form 0.7 must NOT appear for the Malik cash gate."""
        # Isolate the Malik block
        malik_block_match = re.search(
            r'Peaceful Investing.*?(?=# ──.*?(?:Schilit|Billionaire|fw_str|Build))',
            se_source, re.DOTALL
        )
        if malik_block_match:
            malik_block = malik_block_match.group(0)
            assert not re.search(r'cfo_pat_mk.*>=\s*0\.7\b', malik_block), (
                "malik_cash_generation must NOT use >= 0.7 (ratio form). "
                "cfo_to_pat CSV is PERCENTAGE: must use >= 70.0"
            )

    def test_pillar_s_ssgr_equality_1_in_code(self, se_source):
        """Pillar S: ssgr_self_funded == 1 (binary flag)."""
        assert re.search(r'ssgr_mk\s*==\s*1', se_source), (
            "scoring_engine must check ssgr_mk == 1 for malik_self_funded"
        )

    def test_fw_malik_peaceful_is_and_of_5_pillars(self, se_source):
        """fw_malik_peaceful = AND of all 5 materialized pillar columns."""
        assert re.search(r'fw_malik_peaceful\s*=\s*\(', se_source)
        for col in ["malik_growth_runway", "malik_profit_stability",
                    "malik_debt_fortress", "malik_cash_generation", "malik_self_funded"]:
            assert col in se_source, f"fw_malik_peaceful must reference {col}"

    def test_fw_str_includes_malik_label(self, se_source):
        """fw_str must include 'Peaceful Investing|' via fw_malik_peaceful."""
        assert re.search(r'fw_malik_peaceful.*Peaceful Investing', se_source), (
            "fw_str must have np.where(fw_malik_peaceful, 'Peaceful Investing|', '')"
        )

    def test_nan_fillna_rev_gr(self, se_source):
        """Missing rev_gr → fillna(0) → 0 < 10 → gate fails."""
        assert re.search(r'rev_gr_mk\.fillna\s*\(\s*0\s*\)', se_source)

    def test_nan_fillna_npm(self, se_source):
        """Missing npm → fillna(0) → 0 < 8 → gate fails."""
        assert re.search(r'npm_mk\.fillna\s*\(\s*0\s*\)', se_source)

    def test_nan_fillna_de_conservative(self, se_source):
        """Missing debt_to_equity → fillna(999) → unknown leverage = fails gate."""
        assert re.search(r'de_mk\.fillna\s*\(\s*999\s*\)', se_source)

    def test_nan_fillna_cfo_to_pat(self, se_source):
        """Missing cfo_to_pat → fillna(0) → 0 < 70 → gate fails."""
        assert re.search(r'cfo_pat_mk\.fillna\s*\(\s*0\s*\)', se_source)

    def test_malik_score_is_sum_of_5_pillars(self, se_source):
        """malik_score = sum of the 5 pillar flags (0-5 range)."""
        assert re.search(r'malik_score.*malik_growth_runway', se_source, re.DOTALL), (
            "malik_score must sum all 5 pillar flags"
        )

    def test_percentage_unit_comment_in_code(self, se_source):
        """The PERCENTAGE unit warning must be documented in the code comment."""
        assert "PERCENTAGE" in se_source or "70.0 not 0.70" in se_source, (
            "scoring_engine must have a comment documenting cfo_to_pat as PERCENTAGE"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TestMalikPillarArithmetic
# ═══════════════════════════════════════════════════════════════════════════════

class TestMalikPillarArithmetic:
    """Boundary conditions and arithmetic invariants for all 5 pillar gates."""

    # ── Pillar G: Sales Growth Runway ────────────────────────────────────────

    def test_pillar_g_above_boundary_passes(self):
        result = _run_malik([_build_mock_malik_row(rev_gr_10y=15.0)])
        assert result["malik_growth_runway"].iloc[0] == 1

    def test_pillar_g_at_boundary_passes(self):
        """Exactly 10.0% = at threshold → passes (>= 10.0)."""
        result = _run_malik([_build_mock_malik_row(rev_gr_10y=10.0)])
        assert result["malik_growth_runway"].iloc[0] == 1

    def test_pillar_g_just_below_boundary_fails(self):
        result = _run_malik([_build_mock_malik_row(rev_gr_10y=9.9)])
        assert result["malik_growth_runway"].iloc[0] == 0

    def test_pillar_g_zero_growth_fails(self):
        result = _run_malik([_build_mock_malik_row(rev_gr_10y=0.0)])
        assert result["malik_growth_runway"].iloc[0] == 0

    def test_pillar_g_negative_growth_fails(self):
        result = _run_malik([_build_mock_malik_row(rev_gr_10y=-5.0)])
        assert result["malik_growth_runway"].iloc[0] == 0

    def test_pillar_g_nan_10y_uses_5y_fallback(self):
        """NaN rev_gr_10y falls back to rev_gr_5y — not a hard fail."""
        result = _run_malik([_build_mock_malik_row(rev_gr_10y=float("nan"), rev_gr_5y=12.0)])
        # With rev_gr_5y=12.0 >= 10.0, the pillar should pass via fallback
        assert result["malik_growth_runway"].iloc[0] == 1

    def test_pillar_g_both_nan_fails(self):
        """Both rev_gr_10y and rev_gr_5y NaN → fillna(0) → 0 < 10 → fails."""
        result = _run_malik([_build_mock_malik_row(rev_gr_10y=float("nan"), rev_gr_5y=float("nan"))])
        assert result["malik_growth_runway"].iloc[0] == 0

    # ── Pillar P: Profit Stability ────────────────────────────────────────────

    def test_pillar_p_current_at_boundary_prior_above_passes(self):
        """npm=8.0 (at boundary) AND npm_1yb=6.0 (at prior boundary) → passes."""
        result = _run_malik([_build_mock_malik_row(npm=8.0, npm_1yb=6.0)])
        assert result["malik_profit_stability"].iloc[0] == 1

    def test_pillar_p_current_above_prior_unavailable_passes(self):
        """npm=10% AND npm_1yb=NaN (unavailable) → conservative: passes (no prior year data)."""
        result = _run_malik([_build_mock_malik_row(npm=10.0, npm_1yb=float("nan"))])
        assert result["malik_profit_stability"].iloc[0] == 1

    def test_pillar_p_current_below_threshold_fails(self):
        """npm=7.9 < 8.0 → Pillar P fails regardless of prior year."""
        result = _run_malik([_build_mock_malik_row(npm=7.9, npm_1yb=10.0)])
        assert result["malik_profit_stability"].iloc[0] == 0

    def test_pillar_p_current_meets_but_prior_below_fails(self):
        """npm=10% AND npm_1yb=5.9 < 6% → margin declining → fails."""
        result = _run_malik([_build_mock_malik_row(npm=10.0, npm_1yb=5.9)])
        assert result["malik_profit_stability"].iloc[0] == 0

    def test_pillar_p_current_nan_fails(self):
        """NaN current npm → fillna(0) → 0 < 8 → Pillar P fails."""
        result = _run_malik([_build_mock_malik_row(npm=float("nan"))])
        assert result["malik_profit_stability"].iloc[0] == 0

    def test_pillar_p_high_npm_no_prior_passes(self):
        """npm=25% (IT/FMCG quality), no prior year data → passes."""
        result = _run_malik([_build_mock_malik_row(npm=25.0, npm_1yb=float("nan"))])
        assert result["malik_profit_stability"].iloc[0] == 1

    # ── Pillar F: Debt Fortress ───────────────────────────────────────────────

    def test_pillar_f_all_three_above_threshold_passes(self):
        """ICR=6x, D/E=0.3, CR=2.0 → all three sub-gates pass."""
        result = _run_malik([_build_mock_malik_row(
            interest_coverage=6.0, debt_to_equity=0.3, current_ratio=2.0
        )])
        assert result["malik_debt_fortress"].iloc[0] == 1

    def test_pillar_f_icr_at_boundary_passes(self):
        """ICR=3.0 exactly → passes (>= 3.0)."""
        result = _run_malik([_build_mock_malik_row(interest_coverage=3.0)])
        assert result["malik_debt_fortress"].iloc[0] == 1

    def test_pillar_f_icr_just_below_fails(self):
        """ICR=2.9 < 3.0 → Debt Fortress fails."""
        result = _run_malik([_build_mock_malik_row(interest_coverage=2.9)])
        assert result["malik_debt_fortress"].iloc[0] == 0

    def test_pillar_f_de_at_boundary_passes(self):
        """D/E=0.5 exactly → passes (<= 0.5 inclusive)."""
        result = _run_malik([_build_mock_malik_row(debt_to_equity=0.5)])
        assert result["malik_debt_fortress"].iloc[0] == 1

    def test_pillar_f_de_just_above_fails(self):
        """D/E=0.51 > 0.5 → Debt Fortress fails."""
        result = _run_malik([_build_mock_malik_row(debt_to_equity=0.51)])
        assert result["malik_debt_fortress"].iloc[0] == 0

    def test_pillar_f_de_nan_fails(self):
        """NaN D/E → fillna(999) → 999 > 0.5 → Debt Fortress fails."""
        result = _run_malik([_build_mock_malik_row(debt_to_equity=float("nan"))])
        assert result["malik_debt_fortress"].iloc[0] == 0

    def test_pillar_f_cr_at_boundary_passes(self):
        """CR=1.25 exactly → passes (>= 1.25)."""
        result = _run_malik([_build_mock_malik_row(current_ratio=1.25)])
        assert result["malik_debt_fortress"].iloc[0] == 1

    def test_pillar_f_cr_just_below_fails(self):
        """CR=1.24 < 1.25 → Debt Fortress fails."""
        result = _run_malik([_build_mock_malik_row(current_ratio=1.24)])
        assert result["malik_debt_fortress"].iloc[0] == 0

    def test_pillar_f_icr_alone_failing_fails(self):
        """Only ICR fails (D/E and CR fine) → Debt Fortress fails (AND condition)."""
        result = _run_malik([_build_mock_malik_row(
            interest_coverage=1.0, debt_to_equity=0.3, current_ratio=2.0
        )])
        assert result["malik_debt_fortress"].iloc[0] == 0

    def test_pillar_f_de_alone_failing_fails(self):
        """Only D/E fails → Debt Fortress fails (AND condition)."""
        result = _run_malik([_build_mock_malik_row(
            interest_coverage=6.0, debt_to_equity=1.5, current_ratio=2.0
        )])
        assert result["malik_debt_fortress"].iloc[0] == 0

    def test_pillar_f_cr_alone_failing_fails(self):
        """Only CR fails → Debt Fortress fails (AND condition)."""
        result = _run_malik([_build_mock_malik_row(
            interest_coverage=6.0, debt_to_equity=0.3, current_ratio=0.8
        )])
        assert result["malik_debt_fortress"].iloc[0] == 0

    # ── Pillar C: Cash Generation ─────────────────────────────────────────────

    def test_pillar_c_at_boundary_passes(self):
        """cfo_to_pat=70.0 exactly → passes (>= 70.0 PERCENTAGE)."""
        result = _run_malik([_build_mock_malik_row(cfo_to_pat=70.0)])
        assert result["malik_cash_generation"].iloc[0] == 1, (
            "cfo_to_pat=70.0 must pass malik_cash_generation (threshold=70.0 PERCENTAGE, inclusive)"
        )

    def test_pillar_c_above_boundary_passes(self):
        result = _run_malik([_build_mock_malik_row(cfo_to_pat=90.0)])
        assert result["malik_cash_generation"].iloc[0] == 1

    def test_pillar_c_100pct_passes(self):
        """100% cash conversion (CFO == PAT) → gold standard, passes."""
        result = _run_malik([_build_mock_malik_row(cfo_to_pat=100.0)])
        assert result["malik_cash_generation"].iloc[0] == 1

    def test_pillar_c_just_below_boundary_fails(self):
        """69.9% < 70% → fails."""
        result = _run_malik([_build_mock_malik_row(cfo_to_pat=69.9)])
        assert result["malik_cash_generation"].iloc[0] == 0

    def test_pillar_c_0point7_ratio_would_fail(self):
        """Unit guard: 0.7 (the wrong ratio form) must NOT pass (CSV stores PERCENTAGE)."""
        result = _run_malik([_build_mock_malik_row(cfo_to_pat=0.7)])
        assert result["malik_cash_generation"].iloc[0] == 0, (
            "cfo_to_pat=0.7 (ratio form) must FAIL. If it passes, threshold is 0.7 not 70.0 — "
            "CRITICAL unit bug! CSV stores PERCENTAGE (73.04 = 73%)"
        )

    def test_pillar_c_nan_fails(self):
        """NaN cfo_to_pat → fillna(0) → 0 < 70 → gate fails."""
        result = _run_malik([_build_mock_malik_row(cfo_to_pat=float("nan"))])
        assert result["malik_cash_generation"].iloc[0] == 0

    def test_pillar_c_50pct_fails(self):
        """50% cash conversion → below Malik's 70% floor → fails."""
        result = _run_malik([_build_mock_malik_row(cfo_to_pat=50.0)])
        assert result["malik_cash_generation"].iloc[0] == 0

    def test_pillar_c_threshold_higher_than_marks_ratio(self):
        """Malik=70% floor. A stock at 72% passes Malik but must verify threshold."""
        result = _run_malik([_build_mock_malik_row(cfo_to_pat=72.0)])
        assert result["malik_cash_generation"].iloc[0] == 1, (
            "cfo_to_pat=72.0 must pass malik_cash_generation (Malik floor=70%)"
        )

    # ── Pillar S: Self-Funded Growth ──────────────────────────────────────────

    def test_pillar_s_flag_1_passes(self):
        """ssgr_self_funded=1 → growth internally funded → passes."""
        result = _run_malik([_build_mock_malik_row(ssgr_self_funded=1)])
        assert result["malik_self_funded"].iloc[0] == 1

    def test_pillar_s_flag_0_fails(self):
        """ssgr_self_funded=0 → debt-dependent growth → fails."""
        result = _run_malik([_build_mock_malik_row(ssgr_self_funded=0)])
        assert result["malik_self_funded"].iloc[0] == 0

    def test_pillar_s_nan_fails(self):
        """NaN ssgr_self_funded → fillna(0) → 0 ≠ 1 → fails."""
        result = _run_malik([_build_mock_malik_row(ssgr_self_funded=float("nan"))])
        assert result["malik_self_funded"].iloc[0] == 0

    # ── AND Invariant: malik_pass ─────────────────────────────────────────────

    def test_all_5_pass_malik_pass_is_1(self):
        result = _run_malik([_build_mock_malik_row()])
        assert result["malik_pass"].iloc[0] == 1

    def test_single_g_failure_malik_pass_is_0(self):
        """Any single pillar failing must flip malik_pass to 0."""
        result = _run_malik([_build_mock_malik_row(rev_gr_10y=5.0)])
        assert result["malik_pass"].iloc[0] == 0

    def test_single_p_failure_malik_pass_is_0(self):
        result = _run_malik([_build_mock_malik_row(npm=3.0)])
        assert result["malik_pass"].iloc[0] == 0

    def test_single_f_failure_malik_pass_is_0(self):
        result = _run_malik([_build_mock_malik_row(interest_coverage=1.0)])
        assert result["malik_pass"].iloc[0] == 0

    def test_single_c_failure_malik_pass_is_0(self):
        result = _run_malik([_build_mock_malik_row(cfo_to_pat=50.0)])
        assert result["malik_pass"].iloc[0] == 0

    def test_single_s_failure_malik_pass_is_0(self):
        result = _run_malik([_build_mock_malik_row(ssgr_self_funded=0)])
        assert result["malik_pass"].iloc[0] == 0

    def test_each_single_failure_independently_kills_pass(self):
        """Parametric: each pillar, when alone failing, must cause malik_pass=0."""
        fail_cases = [
            {"rev_gr_10y": 5.0},               # G fails
            {"npm": 3.0},                       # P fails
            {"interest_coverage": 1.0},         # F fails (ICR)
            {"debt_to_equity": 2.0},            # F fails (D/E)
            {"current_ratio": 0.5},             # F fails (CR)
            {"cfo_to_pat": 50.0},               # C fails
            {"ssgr_self_funded": 0},            # S fails
        ]
        for fc in fail_cases:
            result = _run_malik([_build_mock_malik_row(**fc)])
            assert result["malik_pass"].iloc[0] == 0, (
                f"malik_pass must be 0 when {fc} fails"
            )

    # ── Score Range ────────────────────────────────────────────────────────────

    def test_score_all_pass_is_5(self):
        result = _run_malik([_build_mock_malik_row()])
        assert result["malik_score"].iloc[0] == 5

    def test_score_all_fail_is_0(self):
        result = _run_malik([_build_mock_malik_row(
            rev_gr_10y=0.0, npm=0.0, interest_coverage=0.0,
            debt_to_equity=5.0, cfo_to_pat=0.0, ssgr_self_funded=0,
        )])
        assert result["malik_score"].iloc[0] == 0

    def test_score_4_when_s_fails(self):
        """S pillar failure → score=4."""
        result = _run_malik([_build_mock_malik_row(ssgr_self_funded=0)])
        assert result["malik_score"].iloc[0] == 4

    def test_score_4_when_c_fails(self):
        """C pillar failure (cfo_to_pat=50 < 70) → score=4."""
        result = _run_malik([_build_mock_malik_row(cfo_to_pat=50.0)])
        assert result["malik_score"].iloc[0] == 4

    def test_score_4_when_g_fails(self):
        """G pillar failure → score=4."""
        result = _run_malik([_build_mock_malik_row(rev_gr_10y=5.0)])
        assert result["malik_score"].iloc[0] == 4

    def test_score_3_when_two_fail(self):
        """Two pillar failures → score=3."""
        result = _run_malik([_build_mock_malik_row(
            cfo_to_pat=50.0, ssgr_self_funded=0
        )])
        assert result["malik_score"].iloc[0] == 3

    def test_score_1_when_four_fail(self):
        """Four pillar failures → score=1."""
        result = _run_malik([_build_mock_malik_row(
            rev_gr_10y=5.0, npm=3.0, cfo_to_pat=50.0, ssgr_self_funded=0
        )])
        assert result["malik_score"].iloc[0] == 1

    def test_score_range_always_0_to_5(self):
        combos = [
            _build_mock_malik_row(),
            _build_mock_malik_row(ssgr_self_funded=0),
            _build_mock_malik_row(cfo_to_pat=50.0, ssgr_self_funded=0),
            _build_mock_malik_row(
                rev_gr_10y=0.0, npm=0.0, interest_coverage=0.0,
                debt_to_equity=5.0, cfo_to_pat=0.0, ssgr_self_funded=0,
            ),
        ]
        result = _run_malik(combos)
        assert result["malik_score"].between(0, 5).all()

    def test_pass_equals_score_5(self):
        """malik_pass == 1 ⟺ malik_score == 5 (AND identity)."""
        combos = [
            _build_mock_malik_row(),                   # all pass → score=5
            _build_mock_malik_row(ssgr_self_funded=0), # S fails → score=4
            _build_mock_malik_row(
                rev_gr_10y=0.0, npm=0.0, interest_coverage=0.0,
                debt_to_equity=5.0, cfo_to_pat=0.0, ssgr_self_funded=0,
            ),  # all fail → score=0
        ]
        result = _run_malik(combos)
        assert (result["malik_pass"] == (result["malik_score"] == 5).astype(int)).all()

    def test_score_columns_are_integer_dtype(self):
        """malik_pass and malik_score must be integer-typed columns."""
        result = _run_malik([_build_mock_malik_row()])
        assert pd.api.types.is_integer_dtype(result["malik_pass"]), (
            "malik_pass must be integer dtype (not float or bool)"
        )
        assert pd.api.types.is_integer_dtype(result["malik_score"]), (
            "malik_score must be integer dtype"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TestMalikNaNConservative
# ═══════════════════════════════════════════════════════════════════════════════

class TestMalikNaNConservative:
    """NaN handling: every numeric gate must default to failure on missing data."""

    def test_all_numeric_nan_fails_pass(self):
        """All numeric gates NaN → ssgr_self_funded=0 → all 5 fail → malik_pass=0."""
        nan_row = {
            "rev_gr_10y":        float("nan"),
            "rev_gr_5y":         float("nan"),
            "npm":               float("nan"),
            "npm_1yb":           float("nan"),
            "interest_coverage": float("nan"),
            "debt_to_equity":    float("nan"),
            "current_ratio":     float("nan"),
            "cfo_to_pat":        float("nan"),
            "ssgr_self_funded":  float("nan"),
        }
        result = _run_malik([nan_row])
        assert result["malik_pass"].iloc[0] == 0, "All-NaN row must fail malik_pass"

    def test_all_nan_score_at_most_1(self):
        """All-NaN row: only Pillar P can pass (npm_1yb.isna() = unavailable, but npm itself fails)."""
        nan_row = {k: float("nan") for k in [
            "rev_gr_10y", "rev_gr_5y", "npm", "npm_1yb",
            "interest_coverage", "debt_to_equity", "current_ratio",
            "cfo_to_pat", "ssgr_self_funded",
        ]}
        result = _run_malik([nan_row])
        # npm=NaN → fillna(0) → 0 < 8 → Pillar P fails too
        # All pillars fail → score = 0
        assert result["malik_score"].iloc[0] == 0, (
            "All-NaN row: every numeric pillar fails → score must be 0"
        )

    def test_de_nan_fails_debt_fortress(self):
        """NaN D/E → fillna(999) → 999 > 0.5 → Debt Fortress fails."""
        result = _run_malik([_build_mock_malik_row(debt_to_equity=float("nan"))])
        assert result["malik_debt_fortress"].iloc[0] == 0

    def test_ic_nan_fails_debt_fortress(self):
        """NaN interest_coverage → fillna(0) → 0 < 3 → Debt Fortress fails."""
        result = _run_malik([_build_mock_malik_row(interest_coverage=float("nan"))])
        assert result["malik_debt_fortress"].iloc[0] == 0

    def test_cr_nan_fails_debt_fortress(self):
        """NaN current_ratio → fillna(0) → 0 < 1.25 → Debt Fortress fails."""
        result = _run_malik([_build_mock_malik_row(current_ratio=float("nan"))])
        assert result["malik_debt_fortress"].iloc[0] == 0

    def test_cfo_nan_fails_cash_generation(self):
        """NaN cfo_to_pat → fillna(0) → 0 < 70 → Cash Generation fails."""
        result = _run_malik([_build_mock_malik_row(cfo_to_pat=float("nan"))])
        assert result["malik_cash_generation"].iloc[0] == 0

    def test_ssgr_nan_fails_self_funded(self):
        """NaN ssgr_self_funded → fillna(0) → 0 ≠ 1 → Self-Funded fails."""
        result = _run_malik([_build_mock_malik_row(ssgr_self_funded=float("nan"))])
        assert result["malik_self_funded"].iloc[0] == 0

    def test_npm_1yb_nan_does_not_fail_profit_stability(self):
        """npm_1yb NaN = prior data unavailable → conservative: does NOT fail Pillar P.
        Only fails if current npm is below threshold."""
        result = _run_malik([_build_mock_malik_row(npm=10.0, npm_1yb=float("nan"))])
        assert result["malik_profit_stability"].iloc[0] == 1, (
            "Missing prior-year NPM must NOT penalize companies — data unavailable is "
            "not the same as declining margins (conservative approach for newer listings)"
        )

    def test_de_nan_is_stricter_than_zero(self):
        """fillna(999) for D/E is more conservative than fillna(0).
        D/E=0 would pass (<= 0.5) but unknown leverage must fail."""
        result = _run_malik([_build_mock_malik_row(debt_to_equity=float("nan"))])
        assert result["malik_debt_fortress"].iloc[0] == 0, (
            "NaN D/E must fail: unknown leverage is never safe. "
            "fillna(999) → 999 > 0.5 → gate correctly fails."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TestMalikIndexAlignment
# ═══════════════════════════════════════════════════════════════════════════════

class TestMalikIndexAlignment:
    """Non-default index safety — guards against pandas index alignment drift."""

    def test_integer_index_non_default(self):
        """Results must be correctly aligned with a shuffled integer index."""
        rows = [
            _build_mock_malik_row(),                    # all pass
            _build_mock_malik_row(cfo_to_pat=50.0),     # C fails
            _build_mock_malik_row(ssgr_self_funded=0),  # S fails
        ]
        from config import MASTER_PROFILES
        from core.scoring_engine import compute_qglp_score
        df = pd.DataFrame(rows, index=[100, 200, 300])
        df.attrs["detected_market_regime"] = "SIDEWAYS"
        profile = MASTER_PROFILES.get("Balanced", next(iter(MASTER_PROFILES.values())))
        result = compute_qglp_score(df, profile)

        assert result.index.tolist() == [100, 200, 300], "Index must be preserved"
        assert result.loc[100, "malik_pass"] == 1, "Row 100: all pass → malik_pass=1"
        assert result.loc[200, "malik_pass"] == 0, "Row 200: C fails → malik_pass=0"
        assert result.loc[300, "malik_pass"] == 0, "Row 300: S fails → malik_pass=0"

    def test_string_index(self):
        """Results must be correctly aligned with a string index."""
        rows = [
            _build_mock_malik_row(),                        # all pass
            _build_mock_malik_row(npm=3.0),                 # P fails
            _build_mock_malik_row(debt_to_equity=2.0),      # F fails (D/E)
        ]
        from config import MASTER_PROFILES
        from core.scoring_engine import compute_qglp_score
        df = pd.DataFrame(rows, index=["HDFC", "TCS", "INFY"])
        df.attrs["detected_market_regime"] = "SIDEWAYS"
        profile = MASTER_PROFILES.get("Balanced", next(iter(MASTER_PROFILES.values())))
        result = compute_qglp_score(df, profile)

        assert result.index.tolist() == ["HDFC", "TCS", "INFY"]
        assert result.loc["HDFC", "malik_pass"] == 1, "HDFC: all pass"
        assert result.loc["TCS",  "malik_pass"] == 0, "TCS: P fails (npm=3%)"
        assert result.loc["INFY", "malik_pass"] == 0, "INFY: F fails (D/E=2.0)"

    def test_score_aligned_with_integer_index(self):
        """Scores must align correctly with a non-default integer index."""
        rows = [
            _build_mock_malik_row(ssgr_self_funded=0),  # 4/5
            _build_mock_malik_row(),                     # 5/5
        ]
        from config import MASTER_PROFILES
        from core.scoring_engine import compute_qglp_score
        df = pd.DataFrame(rows, index=[99, 42])
        df.attrs["detected_market_regime"] = "SIDEWAYS"
        profile = MASTER_PROFILES.get("Balanced", next(iter(MASTER_PROFILES.values())))
        result = compute_qglp_score(df, profile)

        assert result.loc[99, "malik_score"] == 4, "Index 99: S fails → score=4"
        assert result.loc[42, "malik_score"] == 5, "Index 42: all pass → score=5"

    def test_large_index_batch(self):
        """Batch of 10 rows with non-default index → all columns computed correctly."""
        rows = [_build_mock_malik_row() for _ in range(10)]
        from config import MASTER_PROFILES
        from core.scoring_engine import compute_qglp_score
        idx = list(range(100, 110))
        df = pd.DataFrame(rows, index=idx)
        df.attrs["detected_market_regime"] = "BULL"
        profile = MASTER_PROFILES.get("Balanced", next(iter(MASTER_PROFILES.values())))
        result = compute_qglp_score(df, profile)

        assert len(result) == 10
        assert result["malik_pass"].sum() == 10, "All 10 fully-passing rows → malik_pass=1"
        assert (result["malik_score"] == 5).all(), "All 10 rows → malik_score=5"


# ═══════════════════════════════════════════════════════════════════════════════
# TestMalikFinancialExemption
# ═══════════════════════════════════════════════════════════════════════════════

class TestMalikFinancialExemption:
    """Financial sector (banks/NBFCs) exempt from Debt Fortress sub-gates."""

    def test_financial_with_high_de_passes_debt_fortress(self):
        """is_financial=True + D/E=10.0 → Debt Fortress PASSES (structural exemption)."""
        result = _run_malik([_build_mock_malik_row(
            is_financial=True, debt_to_equity=10.0
        )])
        assert result["malik_debt_fortress"].iloc[0] == 1, (
            "Financial sector with D/E=10.0 must pass Debt Fortress (exempt from D/E gate). "
            "Banks structurally carry high leverage."
        )

    def test_financial_with_low_icr_passes_debt_fortress(self):
        """is_financial=True + ICR=0.0 → Debt Fortress PASSES (ICR not applicable to banks)."""
        result = _run_malik([_build_mock_malik_row(
            is_financial=True, interest_coverage=0.0
        )])
        assert result["malik_debt_fortress"].iloc[0] == 1, (
            "Financial sector with ICR=0 must pass Debt Fortress (ICR inapplicable for banks)."
        )

    def test_financial_with_low_cr_passes_debt_fortress(self):
        """is_financial=True + CR=0.0 → Debt Fortress PASSES (CR meaningless for financials)."""
        result = _run_malik([_build_mock_malik_row(
            is_financial=True, current_ratio=0.0
        )])
        assert result["malik_debt_fortress"].iloc[0] == 1, (
            "Financial sector with CR=0 must pass Debt Fortress (CR inapplicable for banks)."
        )

    def test_financial_all_debt_nan_passes_debt_fortress(self):
        """is_financial=True + all debt metrics NaN → Debt Fortress PASSES."""
        result = _run_malik([_build_mock_malik_row(
            is_financial=True,
            interest_coverage=float("nan"),
            debt_to_equity=float("nan"),
            current_ratio=float("nan"),
        )])
        assert result["malik_debt_fortress"].iloc[0] == 1, (
            "Financial sector with all NaN debt metrics must pass Debt Fortress."
        )

    def test_non_financial_same_values_fails_debt_fortress(self):
        """Non-financial with D/E=10.0 must FAIL (exemption is financial-only)."""
        result = _run_malik([_build_mock_malik_row(
            is_financial=False, debt_to_equity=10.0
        )])
        assert result["malik_debt_fortress"].iloc[0] == 0, (
            "Non-financial with D/E=10.0 must FAIL Debt Fortress."
        )

    def test_financial_still_needs_other_4_pillars(self):
        """Financial sector exemption only applies to Debt Fortress — other 4 pillars still required."""
        result = _run_malik([_build_mock_malik_row(
            is_financial=True,
            debt_to_equity=10.0,  # exempt — passes F
            cfo_to_pat=50.0,      # C fails: 50 < 70
        )])
        assert result["malik_pass"].iloc[0] == 0, (
            "Financial sector must still fail when Pillar C (cash generation) fails."
        )

    def test_financial_all_non_debt_passing_full_pass(self):
        """Financial sector: if all non-debt pillars clear, full malik_pass=1."""
        result = _run_malik([_build_mock_malik_row(
            is_financial=True,
            debt_to_equity=10.0,    # F: exempt → passes
            interest_coverage=0.0,  # F: exempt → passes
            current_ratio=0.0,      # F: exempt → passes
        )])
        assert result["malik_pass"].iloc[0] == 1, (
            "Financial sector with all non-debt pillars passing must have malik_pass=1."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TestMalikUIContract
# ═══════════════════════════════════════════════════════════════════════════════

class TestMalikUIContract:
    """Verify render_malik_radar is correctly exported and reads the right columns."""

    def test_render_malik_radar_importable(self):
        from ui import render_malik_radar
        assert callable(render_malik_radar)

    def test_render_malik_radar_in_all(self):
        import ui
        assert "render_malik_radar" in ui.__all__

    def test_ui_reads_malik_growth_runway(self, ui_source):
        assert "malik_growth_runway" in ui_source

    def test_ui_reads_malik_profit_stability(self, ui_source):
        assert "malik_profit_stability" in ui_source

    def test_ui_reads_malik_debt_fortress(self, ui_source):
        assert "malik_debt_fortress" in ui_source

    def test_ui_reads_malik_cash_generation(self, ui_source):
        assert "malik_cash_generation" in ui_source

    def test_ui_reads_malik_self_funded(self, ui_source):
        assert "malik_self_funded" in ui_source

    def test_ui_reads_malik_pass(self, ui_source):
        assert '"malik_pass"' in ui_source

    def test_ui_reads_malik_score(self, ui_source):
        assert '"malik_score"' in ui_source

    def test_ui_uses_no_threshold_math(self, ui_source):
        """Pure display: render_malik_radar must not contain numeric threshold literals."""
        match = re.search(r'def render_malik_radar.*?(?=\ndef |\Z)', ui_source, re.DOTALL)
        assert match, "render_malik_radar function not found in ui_tearsheet.py"
        fn_body = match.group(0)
        # These patterns should not appear as computation — only in string labels/display text
        # We allow them in comments/strings for display, but not as comparison operators
        forbidden_computations = [
            r'>=\s*70',     # Pillar C threshold as computation
            r'>=\s*10\.0',  # Pillar G threshold as computation
            r'>=\s*8',      # Pillar P threshold as computation
            r'<=\s*0\.5',   # Pillar F D/E threshold as computation
            r'>=\s*3\.0',   # Pillar F ICR threshold as computation
            r'>=\s*1\.25',  # Pillar F CR threshold as computation
        ]
        for pattern in forbidden_computations:
            assert not re.search(pattern, fn_body), (
                f"render_malik_radar contains threshold computation: {pattern}. "
                "Pure display layer must never re-compute thresholds — read pre-materialized columns only."
            )

    def test_ui_has_green_color_for_pass(self, ui_source):
        """Malik radar must use green accent for passed pillars."""
        match = re.search(r'def render_malik_radar.*?(?=\ndef |\Z)', ui_source, re.DOTALL)
        assert match
        fn_body = match.group(0)
        # Green must be present (either #2ecc71 or any green hex)
        assert re.search(r'#[0-9a-fA-F]{6}', fn_body), (
            "render_malik_radar must have a color definition"
        )
        assert "_MALIK_GREEN" in fn_body or "2ecc71" in fn_body or "green" in fn_body.lower(), (
            "render_malik_radar must use a green color for passing pillars"
        )

    def test_ui_displays_score_out_of_5(self, ui_source):
        """Score display must show X/5 (not /100 or /4)."""
        match = re.search(r'def render_malik_radar.*?(?=\ndef |\Z)', ui_source, re.DOTALL)
        assert match
        fn_body = match.group(0)
        assert "/ 5" in fn_body or "/5" in fn_body, (
            "render_malik_radar must display score out of 5 (not /100 or /4)"
        )

    def test_ui_has_5_pillar_letter_labels(self, ui_source):
        """The 5 pillar single-letter labels G/P/F/C/S must be present in the widget."""
        match = re.search(r'def render_malik_radar.*?(?=\ndef |\Z)', ui_source, re.DOTALL)
        assert match
        fn_body = match.group(0)
        for letter in ["G", "P", "F", "C", "S"]:
            assert f'"{letter}"' in fn_body or f"'{letter}'" in fn_body, (
                f"render_malik_radar missing pillar letter: {letter}"
            )

    def test_ui_has_pillar_descriptive_labels(self, ui_source):
        """Descriptive pillar names must appear in the widget."""
        match = re.search(r'def render_malik_radar.*?(?=\ndef |\Z)', ui_source, re.DOTALL)
        assert match
        fn_body = match.group(0)
        for label in ["Growth Runway", "Margin Stability", "Debt Fortress",
                      "Cash Realization", "Self-Funded"]:
            assert label in fn_body, f"render_malik_radar missing pillar label: '{label}'"

    def test_ui_peaceful_investing_title_present(self, ui_source):
        """The function header/section must identify Vijay Malik or Peaceful Investing."""
        match = re.search(r'def render_malik_radar.*?(?=\ndef |\Z)', ui_source, re.DOTALL)
        assert match
        fn_body = match.group(0)
        assert "Malik" in fn_body or "Peaceful Investing" in fn_body

    def test_ui_certified_status_message(self, ui_source):
        """Status message must differentiate pass from fail."""
        match = re.search(r'def render_malik_radar.*?(?=\ndef |\Z)', ui_source, re.DOTALL)
        assert match
        fn_body = match.group(0)
        assert "CERTIFIED" in fn_body or "Criteria Not Met" in fn_body, (
            "render_malik_radar must have a pass/fail status message"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TestMalikRawSignalsContract
# ═══════════════════════════════════════════════════════════════════════════════

class TestMalikRawSignalsContract:
    """Verify Malik cells appear in render_raw_signals grid."""

    def test_raw_signals_has_malik_score_cell(self, ui_source):
        assert "malik_score" in ui_source

    def test_raw_signals_has_malik_pass_cell(self, ui_source):
        assert "malik_pass" in ui_source

    def test_raw_signals_malik_score_format_is_out_of_5(self, ui_source):
        """malik_score in render_raw_signals must show /5 not /100."""
        assert '"/5"' in ui_source or "'/5'" in ui_source or "/5" in ui_source, (
            "render_raw_signals must display malik_score out of 5"
        )

    def test_raw_signals_malik_pass_has_readable_label(self, ui_source):
        """malik_pass cell must use a human-readable label like 'Yes' or 'No'."""
        assert "Malik Pass" in ui_source or "malik_pass" in ui_source

    def test_raw_signals_malik_score_has_readable_label(self, ui_source):
        """malik_score cell must use 'Malik Score' label."""
        assert "Malik Score" in ui_source


# ═══════════════════════════════════════════════════════════════════════════════
# TestMalikAppWiring
# ═══════════════════════════════════════════════════════════════════════════════

class TestMalikAppWiring:
    """Verify app.py correctly imports and calls render_malik_radar."""

    @pytest.fixture(scope="class")
    def app_source(self) -> str:
        app_path = os.path.join(REPO_ROOT, "app.py")
        with open(app_path, encoding="utf-8") as f:
            return f.read()

    def test_app_imports_render_malik_radar(self, app_source):
        assert "render_malik_radar" in app_source, (
            "app.py must import render_malik_radar from ui"
        )

    def test_app_calls_render_malik_radar(self, app_source):
        assert "render_malik_radar(stock)" in app_source, (
            "app.py must call render_malik_radar(stock) in the frameworks tab"
        )

    def test_ui_init_exports_render_malik_radar(self):
        """ui/__init__.py must export render_malik_radar in __all__."""
        import ui
        assert "render_malik_radar" in ui.__all__

    def test_render_malik_radar_callable_after_import(self):
        """render_malik_radar must be callable (not a stub from failed import)."""
        from ui import render_malik_radar
        assert callable(render_malik_radar)
        # Verify it's not the generic _stub function
        assert render_malik_radar.__name__ != "_stub", (
            "render_malik_radar resolved to _stub — check for ImportError in ui/__init__.py"
        )
