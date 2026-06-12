"""
test_fisher_contract.py

Contract tests that parse docs/fisher_quality_specs.json and assert that
core/scoring_engine.py honours every threshold in the Philip Fisher
Operating Leverage & Scalability Codex specification.

Four criteria verified:
  Rev Runway   (Points 1 & 2): rev_gr_3y >= 12.0% AND rev_gr_yoy >= 10.0%
  Op Leverage  (Points 5 & 6): d05_rev_minus_exp_gr >= 2.0%
  Pricing Power (Point 4):     opm_acceleration >= 0 OR npm_acceleration >= 0
  Anti-Dilution (Point 13):    dilution_pct <= 1.0%  [fillna(999) = conservative]

Output columns: df["fisher_pass"] (int 0/1) and df["fisher_score"] (int 0-4).
Framework label in frameworks_passed: "Fisher Scalability"
Spec ledger: docs/fisher_quality_specs.json
"""
import functools
import json
import re
import sys
from pathlib import Path

ROOT       = Path(__file__).parent.parent
SPEC_PATH  = ROOT / "docs" / "fisher_quality_specs.json"
SE_PATH    = ROOT / "core" / "scoring_engine.py"


def _load_spec():
    with open(SPEC_PATH, encoding="utf-8") as f:
        return json.load(f)


def _src():
    return SE_PATH.read_text(encoding="utf-8")


def _fisher_block(src: str) -> str:
    """Extract the fw_fisher_scalability block from scoring_engine.py."""
    anchor = "# ── Framework 33: fw_fisher_scalability"
    end_anchor = "# Schilit Financial Shenanigans"
    start = src.find(anchor)
    assert start != -1, (
        "Cannot find fw_fisher_scalability anchor "
        "'# ── Framework 33: fw_fisher_scalability' in scoring_engine.py"
    )
    end = src.find(end_anchor, start)
    assert end != -1, (
        "Cannot find end boundary '# Schilit Financial Shenanigans' of fw_fisher_scalability block"
    )
    return src[start:end]


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — Spec file integrity
# ═══════════════════════════════════════════════════════════════════════════

class TestSpecFileExists:
    def test_spec_file_exists(self):
        """docs/fisher_quality_specs.json must exist."""
        assert SPEC_PATH.exists(), f"Missing spec file: {SPEC_PATH}"

    def test_spec_is_valid_json(self):
        """Spec file must be valid JSON."""
        spec = _load_spec()
        assert isinstance(spec, dict)

    def test_spec_has_meta_block(self):
        spec = _load_spec()
        assert "_meta" in spec, "Spec must contain '_meta' block"

    def test_spec_has_fisher_core_parameters(self):
        spec = _load_spec()
        assert "fisher_core_parameters" in spec, "Spec must contain 'fisher_core_parameters'"

    def test_spec_has_implementation_mapping(self):
        spec = _load_spec()
        assert "implementation_mapping" in spec, "Spec must contain 'implementation_mapping'"

    def test_spec_framework_variable_name(self):
        spec = _load_spec()
        fw_var = spec["implementation_mapping"]["_framework_variable"]
        assert fw_var == "fw_fisher_scalability", (
            f"Expected framework variable 'fw_fisher_scalability', got '{fw_var}'"
        )

    def test_spec_pass_column_name(self):
        spec = _load_spec()
        col = spec["implementation_mapping"]["_pass_column"]
        assert col == "fisher_pass", f"Expected pass column 'fisher_pass', got '{col}'"

    def test_spec_score_column_name(self):
        spec = _load_spec()
        col = spec["implementation_mapping"]["_score_column"]
        assert col == "fisher_score", f"Expected score column 'fisher_score', got '{col}'"

    def test_spec_frameworks_passed_label(self):
        spec = _load_spec()
        label = spec["implementation_mapping"]["_frameworks_passed_label"]
        assert label == "Fisher Scalability", (
            f"Expected label 'Fisher Scalability', got '{label}'"
        )

    def test_spec_score_range_documented(self):
        spec = _load_spec()
        score_range = spec["implementation_mapping"]["_score_range"]
        assert "0-4" in score_range, f"Score range must document '0-4', got: '{score_range}'"


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — Spec threshold values
# ═══════════════════════════════════════════════════════════════════════════

class TestSpecThresholdValues:
    def test_rev_runway_3y_threshold(self):
        spec = _load_spec()
        val = spec["fisher_core_parameters"]["revenue_runway_scalability"]["min_revenue_growth_3years_pct"]
        assert val == 12.0, f"Expected rev_gr_3y threshold 12.0, got {val}"

    def test_rev_runway_yoy_threshold(self):
        spec = _load_spec()
        val = spec["fisher_core_parameters"]["revenue_runway_scalability"]["min_revenue_growth_yoy_pct"]
        assert val == 10.0, f"Expected rev_gr_yoy threshold 10.0, got {val}"

    def test_operating_leverage_threshold(self):
        spec = _load_spec()
        val = spec["fisher_core_parameters"]["operating_leverage_inflection"]["min_rev_minus_exp_growth_delta"]
        assert val == 2.0, f"Expected d05_rev_minus_exp_gr threshold 2.0, got {val}"

    def test_op_lev_code_variable(self):
        spec = _load_spec()
        ref = spec["fisher_core_parameters"]["operating_leverage_inflection"]["code_variable_reference"]
        assert ref == "d05_rev_minus_exp_gr", (
            f"Expected code variable 'd05_rev_minus_exp_gr', got '{ref}'"
        )

    def test_pricing_power_require_positive(self):
        spec = _load_spec()
        val = spec["fisher_core_parameters"]["pricing_power_moat"]["require_positive_margin_acceleration"]
        assert val is True, "Pricing power must require positive margin acceleration"

    def test_pricing_power_or_logic(self):
        spec = _load_spec()
        logic = spec["fisher_core_parameters"]["pricing_power_moat"]["logic"]
        assert "OR" in logic.upper(), f"Pricing power must use OR logic, got: '{logic}'"

    def test_pricing_power_code_variables(self):
        spec = _load_spec()
        refs = spec["fisher_core_parameters"]["pricing_power_moat"]["code_variable_references"]
        assert "opm_acceleration" in refs, "opm_acceleration must be in pricing power refs"
        assert "npm_acceleration" in refs, "npm_acceleration must be in pricing power refs"

    def test_dilution_threshold(self):
        spec = _load_spec()
        val = spec["fisher_core_parameters"]["capital_dilution_shield"]["max_annual_equity_dilution_pct"]
        assert val == 1.0, f"Expected dilution_pct threshold 1.0, got {val}"

    def test_dilution_code_variable(self):
        spec = _load_spec()
        ref = spec["fisher_core_parameters"]["capital_dilution_shield"]["code_variable_reference"]
        assert ref == "dilution_pct", f"Expected code variable 'dilution_pct', got '{ref}'"


# ═══════════════════════════════════════════════════════════════════════════
# PART 3 — Scoring engine source: framework block structure
# ═══════════════════════════════════════════════════════════════════════════

class TestFrameworkBlockExists:
    def test_framework_anchor_exists(self):
        src = _src()
        assert "# ── Framework 33: fw_fisher_scalability" in src, (
            "Missing Framework 33 anchor in scoring_engine.py"
        )

    def test_framework_variable_defined(self):
        block = _fisher_block(_src())
        assert "fw_fisher_scalability" in block, (
            "fw_fisher_scalability variable not found in Framework 33 block"
        )

    def test_fisher_pass_column_assigned(self):
        block = _fisher_block(_src())
        assert 'df["fisher_pass"]' in block, (
            'df["fisher_pass"] assignment not found in Framework 33 block'
        )

    def test_fisher_score_column_assigned(self):
        block = _fisher_block(_src())
        assert 'df["fisher_score"]' in block, (
            'df["fisher_score"] assignment not found in Framework 33 block'
        )

    def test_fisher_pass_is_int_cast(self):
        block = _fisher_block(_src())
        assert "fw_fisher_scalability.astype(int)" in block, (
            "fisher_pass must be fw_fisher_scalability.astype(int)"
        )

    def test_spec_file_reference_in_comment(self):
        block = _fisher_block(_src())
        assert "fisher_quality_specs.json" in block, (
            "Framework 33 block must reference docs/fisher_quality_specs.json"
        )


# ═══════════════════════════════════════════════════════════════════════════
# PART 4 — Sub-gate threshold literals in source code
# ═══════════════════════════════════════════════════════════════════════════

class TestSubGateThresholds:
    """Assert that every threshold number from the spec appears in the code block."""

    def test_rev_3y_column_referenced(self):
        block = _fisher_block(_src())
        assert '"rev_gr_3y"' in block, 'Missing "rev_gr_3y" in Fisher Scalability block'

    def test_rev_yoy_column_referenced(self):
        block = _fisher_block(_src())
        assert '"rev_gr_yoy"' in block, 'Missing "rev_gr_yoy" in Fisher Scalability block'

    def test_rev_3y_threshold_12(self):
        block = _fisher_block(_src())
        assert ">= 12.0" in block, "Revenue runway 3Y threshold 12.0% not found in code block"

    def test_rev_yoy_threshold_10(self):
        block = _fisher_block(_src())
        assert ">= 10.0" in block, "Revenue runway YoY threshold 10.0% not found in code block"

    def test_oplev_column_referenced(self):
        block = _fisher_block(_src())
        assert '"d05_rev_minus_exp_gr"' in block, (
            'Missing "d05_rev_minus_exp_gr" in Fisher Scalability block'
        )

    def test_oplev_threshold_2(self):
        block = _fisher_block(_src())
        assert ">= 2.0" in block, "Operating leverage threshold 2.0% not found in code block"

    def test_opm_acceleration_column_referenced(self):
        block = _fisher_block(_src())
        assert '"opm_acceleration"' in block, (
            'Missing "opm_acceleration" in Fisher Scalability block'
        )

    def test_npm_acceleration_column_referenced(self):
        block = _fisher_block(_src())
        assert '"npm_acceleration"' in block, (
            'Missing "npm_acceleration" in Fisher Scalability block'
        )

    def test_pricing_power_or_logic_in_code(self):
        block = _fisher_block(_src())
        # OR-logic must appear: (opm_acc_fs...) | (npm_acc_fs...)
        assert ") | (" in block, (
            "Pricing power gate must use OR logic '| (' but not found in code block"
        )

    def test_pricing_power_threshold_0_opm(self):
        block = _fisher_block(_src())
        # threshold >= 0 for opm_acceleration
        assert "opm_acc_fs.fillna(0) >= 0" in block, (
            "OPM acceleration threshold '>= 0' not found in code block"
        )

    def test_pricing_power_threshold_0_npm(self):
        block = _fisher_block(_src())
        assert "npm_acc_fs.fillna(0) >= 0" in block, (
            "NPM acceleration threshold '>= 0' not found in code block"
        )

    def test_dilution_column_referenced(self):
        block = _fisher_block(_src())
        assert '"dilution_pct"' in block, 'Missing "dilution_pct" in Fisher Scalability block'

    def test_dilution_threshold_1(self):
        block = _fisher_block(_src())
        assert "<= 1.0" in block, "Anti-dilution threshold <= 1.0% not found in code block"

    def test_dilution_conservative_fillna_999(self):
        block = _fisher_block(_src())
        assert "fillna(999)" in block, (
            "Anti-dilution gate must use fillna(999) — conservative exclusion for missing data. "
            "Not found in code block."
        )


# ═══════════════════════════════════════════════════════════════════════════
# PART 5 — NaN handling strategy
# ═══════════════════════════════════════════════════════════════════════════

class TestNaNHandling:
    def test_rev_3y_fillna_zero(self):
        """Missing rev_gr_3y → fillna(0) → 0 < 12 → gate fails (conservative)."""
        block = _fisher_block(_src())
        assert "rev_3y_fs.fillna(0)" in block, (
            "rev_gr_3y must use fillna(0) to conservatively fail the gate on missing data"
        )

    def test_rev_yoy_fillna_zero(self):
        """Missing rev_gr_yoy → fillna(0) → 0 < 10 → gate fails."""
        block = _fisher_block(_src())
        assert "rev_yoy_fs.fillna(0)" in block, (
            "rev_gr_yoy must use fillna(0) in the revenue runway gate"
        )

    def test_oplev_fillna_zero(self):
        """Missing d05_rev_minus_exp_gr → fillna(0) → 0 < 2 → gate fails."""
        block = _fisher_block(_src())
        assert "oplev_fs.fillna(0)" in block, (
            "d05_rev_minus_exp_gr must use fillna(0) in the operating leverage gate"
        )

    def test_dilution_fillna_999_not_zero(self):
        """dilution_pct must NOT use fillna(0) — zero dilution on missing data is wrong."""
        block = _fisher_block(_src())
        # Should have fillna(999), not fillna(0) for dilution
        assert "dilut_pct_fs.fillna(999)" in block, (
            "dilution_pct must use fillna(999) to conservatively exclude stocks with no share data"
        )
        # Confirm it's NOT fillna(0) — that would incorrectly pass missing stocks
        # (dilut_pct_fs variable must not be used with fillna(0))
        rev_zero_usage = re.findall(r"dilut_pct_fs\.fillna\(0\)", block)
        assert len(rev_zero_usage) == 0, (
            "dilut_pct_fs must NOT use fillna(0) — missing dilution data should exclude (999)"
        )


# ═══════════════════════════════════════════════════════════════════════════
# PART 6 — fw_str wiring
# ═══════════════════════════════════════════════════════════════════════════

class TestFrameworksPassedWiring:
    def test_fisher_scalability_in_fw_str(self):
        """Fisher Scalability must appear in the fw_str builder."""
        src = _src()
        assert '"Fisher Scalability|"' in src, (
            '"Fisher Scalability|" not found in fw_str builder in scoring_engine.py'
        )

    def test_fw_fisher_scalability_used_in_np_where(self):
        """fw_fisher_scalability must be referenced in a np.where() call in fw_str."""
        src = _src()
        assert "np.where(fw_fisher_scalability" in src, (
            "np.where(fw_fisher_scalability ...) not found in fw_str builder"
        )

    def test_fisher_quality_label_still_present(self):
        """fw_fisher (Fisher Quality, Framework 11) must still be wired separately."""
        src = _src()
        assert '"Fisher Quality|"' in src, (
            '"Fisher Quality|" removed from fw_str — must be preserved alongside Fisher Scalability'
        )

    def test_fisher_scalability_after_multitrillioncap(self):
        """Fisher Scalability entry must appear after Multi-Trillion Cap in fw_str."""
        src = _src()
        multi_pos = src.find('"Multi-Trillion Cap|"')
        fisher_pos = src.find('"Fisher Scalability|"')
        assert multi_pos != -1 and fisher_pos != -1, (
            "Both Multi-Trillion Cap and Fisher Scalability must be in fw_str"
        )
        assert fisher_pos > multi_pos, (
            "Fisher Scalability must appear after Multi-Trillion Cap in fw_str builder"
        )

    def test_fisher_scalability_before_schilit(self):
        """Fisher Scalability must appear before Financial Shenanigans in fw_str."""
        src = _src()
        fisher_pos = src.find('"Fisher Scalability|"')
        schilit_pos = src.find('"Financial Shenanigans|"')
        assert fisher_pos != -1, '"Fisher Scalability|" not found in fw_str'
        assert schilit_pos != -1, '"Financial Shenanigans|" not found in fw_str'
        assert fisher_pos < schilit_pos, (
            "Fisher Scalability must appear before Financial Shenanigans in fw_str builder"
        )


# ═══════════════════════════════════════════════════════════════════════════
# PART 7 — fisher_score structure (0-4 composite)
# ═══════════════════════════════════════════════════════════════════════════

class TestFisherScoreStructure:
    def test_fisher_score_sums_four_components(self):
        """fisher_score must sum exactly 4 sub-gate boolean casts."""
        block = _fisher_block(_src())
        # All 4 sub-gate variables must be cast to int in the score summation
        assert "_fs_rev_runway.astype(int)" in block, (
            "_fs_rev_runway.astype(int) not in fisher_score summation"
        )
        assert "_fs_op_lev.astype(int)" in block, (
            "_fs_op_lev.astype(int) not in fisher_score summation"
        )
        assert "_fs_pricing.astype(int)" in block, (
            "_fs_pricing.astype(int) not in fisher_score summation"
        )
        assert "_fs_anti_dilut.astype(int)" in block, (
            "_fs_anti_dilut.astype(int) not in fisher_score summation"
        )

    def test_fw_fisher_scalability_is_and_of_four_sub_gates(self):
        """fw_fisher_scalability must be the AND of all 4 sub-gate masks."""
        block = _fisher_block(_src())
        # All 4 sub-gate variables must appear in the fw definition line
        assert "_fs_rev_runway" in block
        assert "_fs_op_lev" in block
        assert "_fs_pricing" in block
        assert "_fs_anti_dilut" in block

    def test_fw_fisher_scalability_and_not_or(self):
        """The final fw assignment must use & (AND), not | (OR) between sub-gates."""
        block = _fisher_block(_src())
        # Find the fw_fisher_scalability assignment line
        lines = block.split("\n")
        fw_line = next(
            (l for l in lines if "fw_fisher_scalability = _fs_rev" in l), None
        )
        assert fw_line is not None, (
            "Cannot find the fw_fisher_scalability = _fs_rev_runway & ... assignment line"
        )
        assert "&" in fw_line, (
            "fw_fisher_scalability must use & (AND) between sub-gates, not |"
        )


# ═══════════════════════════════════════════════════════════════════════════
# PART 8 — Uniqueness / no duplication vs fw_fisher (Fisher Quality)
# ═══════════════════════════════════════════════════════════════════════════

class TestDistinctFromFisherQuality:
    def test_fw_fisher_quality_still_exists(self):
        """fw_fisher (Fisher Quality, Framework 11) must not be removed or renamed."""
        src = _src()
        assert "fw_fisher = (" in src, (
            "fw_fisher (Fisher Quality, Framework 11) has been removed — it must be preserved"
        )

    def test_two_separate_fisher_variables(self):
        """Both fw_fisher and fw_fisher_scalability must coexist in source."""
        src = _src()
        assert "fw_fisher = (" in src, "fw_fisher (Fisher Quality) missing"
        assert "fw_fisher_scalability = " in src, "fw_fisher_scalability (Fisher Scalability) missing"

    def test_fisher_scalability_uses_different_columns(self):
        """Fisher Scalability block must use d05_rev_minus_exp_gr (not in Fisher Quality block)."""
        block = _fisher_block(_src())
        assert "d05_rev_minus_exp_gr" in block, (
            "Fisher Scalability must reference d05_rev_minus_exp_gr — this is its unique operating leverage signal"
        )

    def test_fisher_quality_does_not_use_d05(self):
        """Fisher Quality (fw_fisher) block must NOT use d05_rev_minus_exp_gr."""
        src = _src()
        # Find the fw_fisher block (Framework 11)
        anchor = "# 11. Fisher Quality"
        end_anchor = "# 12. 100-Bagger Hunter"
        start = src.find(anchor)
        assert start != -1, "Cannot find '# 11. Fisher Quality' anchor"
        end = src.find(end_anchor, start)
        assert end != -1, "Cannot find '# 12. 100-Bagger Hunter' boundary"
        fisher_quality_block = src[start:end]
        assert "d05_rev_minus_exp_gr" not in fisher_quality_block, (
            "d05_rev_minus_exp_gr must not appear in Fisher Quality block — it belongs to Fisher Scalability"
        )


# ═══════════════════════════════════════════════════════════════════════════
# PART 9 — Zero-loop / vectorization compliance
# ═══════════════════════════════════════════════════════════════════════════

class TestVectorizationCompliance:
    def test_no_iterrows_in_fisher_block(self):
        """Fisher Scalability block must not use iterrows()."""
        block = _fisher_block(_src())
        assert "iterrows" not in block, (
            "iterrows() found in Fisher Scalability block — violates zero-loop design principle"
        )

    def test_no_apply_in_fisher_block(self):
        """Fisher Scalability block must not use .apply()."""
        block = _fisher_block(_src())
        # Allow df.get() but not df.apply() or .apply(
        apply_usages = re.findall(r"\.apply\(", block)
        assert len(apply_usages) == 0, (
            f".apply() found {len(apply_usages)} times in Fisher Scalability block — "
            "must use vectorized boolean masks only"
        )

    def test_uses_df_get_for_column_access(self):
        """Fisher Scalability must use df.get() for robust missing-column handling."""
        block = _fisher_block(_src())
        assert "df.get(" in block, (
            "Fisher Scalability block must use df.get() for safe column access"
        )


# ═══════════════════════════════════════════════════════════════════════════
# PART 10 — Column source validation (data_engine.py)
# ═══════════════════════════════════════════════════════════════════════════

class TestColumnSourceValidation:
    """Verify that all columns used by Fisher Scalability are computed in data_engine.py."""

    DE_PATH = ROOT / "core" / "data_engine.py"

    def _de_src(self):
        return self.DE_PATH.read_text(encoding="utf-8")

    def test_rev_gr_3y_mapped_in_data_engine(self):
        de = self._de_src()
        assert '"rev_gr_3y"' in de, 'rev_gr_3y not found as output column in data_engine.py'

    def test_rev_gr_yoy_mapped_in_data_engine(self):
        de = self._de_src()
        assert '"rev_gr_yoy"' in de, 'rev_gr_yoy not found as output column in data_engine.py'

    def test_d05_computed_in_data_engine(self):
        de = self._de_src()
        assert 'd05_rev_minus_exp_gr' in de, (
            'd05_rev_minus_exp_gr not computed in data_engine.py'
        )

    def test_opm_acceleration_computed_in_data_engine(self):
        de = self._de_src()
        assert 'opm_acceleration' in de, (
            'opm_acceleration not computed in data_engine.py'
        )

    def test_npm_acceleration_computed_in_data_engine(self):
        de = self._de_src()
        assert 'npm_acceleration' in de, (
            'npm_acceleration not computed in data_engine.py'
        )

    def test_dilution_pct_computed_in_data_engine(self):
        de = self._de_src()
        assert 'dilution_pct' in de, (
            'dilution_pct not computed in data_engine.py'
        )


# ═══════════════════════════════════════════════════════════════════════════
# PART 11 — fisher_quality_pass output column (v1.1 addition)
# ═══════════════════════════════════════════════════════════════════════════

class TestFisherQualityPassColumn:
    """Contracts for df["fisher_quality_pass"] — the symmetric pass column for fw_fisher."""

    def test_fisher_quality_pass_assigned_in_source(self):
        """fisher_quality_pass must be assigned in scoring_engine.py."""
        src = _src()
        assert 'df["fisher_quality_pass"]' in src, (
            'df["fisher_quality_pass"] assignment not found in scoring_engine.py'
        )

    def test_fisher_quality_pass_uses_fw_fisher(self):
        """fisher_quality_pass must derive from fw_fisher (Framework 11), not fw_fisher_scalability."""
        src = _src()
        assert 'df["fisher_quality_pass"] = fw_fisher.astype(int)' in src, (
            'df["fisher_quality_pass"] must be assigned as fw_fisher.astype(int)'
        )

    def test_fisher_quality_pass_in_fisher_block(self):
        """fisher_quality_pass assignment must reside inside the Framework 33 block."""
        block = _fisher_block(_src())
        assert 'df["fisher_quality_pass"]' in block, (
            'df["fisher_quality_pass"] not found inside Framework 33 block'
        )

    def test_fisher_quality_pass_is_int_cast(self):
        """fisher_quality_pass must be cast to int (binary 0/1), not bool."""
        block = _fisher_block(_src())
        assert "fw_fisher.astype(int)" in block, (
            "fisher_quality_pass must use fw_fisher.astype(int) — not bool"
        )

    def test_spec_documents_fisher_quality_pass(self):
        """docs/fisher_quality_specs.json must document fisher_quality_pass in output_columns."""
        spec = _load_spec()
        assert "output_columns" in spec, "Spec must have 'output_columns' block"
        assert "fisher_quality_pass" in spec["output_columns"], (
            "'fisher_quality_pass' not documented in spec output_columns"
        )

    def test_spec_fisher_quality_pass_dtype(self):
        """Spec must declare fisher_quality_pass dtype as int (0 or 1)."""
        spec = _load_spec()
        dtype_str = spec["output_columns"]["fisher_quality_pass"]["dtype"]
        assert "int" in dtype_str.lower() or "0 or 1" in dtype_str, (
            f"fisher_quality_pass dtype must state 'int (0 or 1)', got: '{dtype_str}'"
        )

    def test_spec_fisher_quality_pass_source_variable(self):
        """Spec must name fw_fisher (Framework 11) as the source."""
        spec = _load_spec()
        source = spec["output_columns"]["fisher_quality_pass"]["source_variable"]
        assert "fw_fisher" in source, (
            f"fisher_quality_pass source_variable must reference fw_fisher, got: '{source}'"
        )


# ═══════════════════════════════════════════════════════════════════════════
# PART 12 — fisher_lifecycle_quadrant output column (v1.1 addition)
# ═══════════════════════════════════════════════════════════════════════════

VALID_QUADRANT_LABELS = frozenset([
    "⚪ Laggard",
    "⚡ Catalyst Play",
    "🐢 Steady Compounder",
    "👑 Apex Winner",
])


class TestFisherLifecycleQuadrant:
    """Contracts for df["fisher_lifecycle_quadrant"] — the 4-quadrant lifecycle matrix."""

    def test_lifecycle_column_assigned_in_source(self):
        """fisher_lifecycle_quadrant must be assigned in scoring_engine.py."""
        src = _src()
        assert 'df["fisher_lifecycle_quadrant"]' in src, (
            'df["fisher_lifecycle_quadrant"] assignment not found in scoring_engine.py'
        )

    def test_lifecycle_uses_np_select(self):
        """fisher_lifecycle_quadrant must use np.select for vectorized computation."""
        block = _fisher_block(_src())
        assert "np.select" in block, (
            "fisher_lifecycle_quadrant must use np.select — not if/else, not apply, not map"
        )

    def test_lifecycle_no_iterrows(self):
        """Lifecycle quadrant block must not use iterrows()."""
        block = _fisher_block(_src())
        assert "iterrows" not in block, (
            "iterrows() found in Fisher Scalability block — violates zero-loop design"
        )

    def test_lifecycle_no_apply(self):
        """Lifecycle quadrant block must not use .apply()."""
        block = _fisher_block(_src())
        apply_usages = re.findall(r"\.apply\(", block)
        assert len(apply_usages) == 0, (
            f".apply() found {len(apply_usages)} times — lifecycle must be fully vectorized"
        )

    def test_all_four_quadrant_labels_in_source(self):
        """All 4 quadrant label strings must appear in scoring_engine.py."""
        src = _src()
        for label in VALID_QUADRANT_LABELS:
            assert label in src, (
                f"Quadrant label '{label}' not found in scoring_engine.py"
            )

    def test_laggard_label_present(self):
        block = _fisher_block(_src())
        assert "⚪ Laggard" in block, '"⚪ Laggard" not in Framework 33 block'

    def test_catalyst_play_label_present(self):
        block = _fisher_block(_src())
        assert "⚡ Catalyst Play" in block, '"⚡ Catalyst Play" not in Framework 33 block'

    def test_steady_compounder_label_present(self):
        block = _fisher_block(_src())
        assert "🐢 Steady Compounder" in block, '"🐢 Steady Compounder" not in Framework 33 block'

    def test_apex_winner_label_present(self):
        block = _fisher_block(_src())
        assert "👑 Apex Winner" in block, '"👑 Apex Winner" not in Framework 33 block'

    def test_default_is_laggard(self):
        """np.select default must be '⚪ Laggard' — the safe fallback for any unclassified row."""
        block = _fisher_block(_src())
        assert 'default="⚪ Laggard"' in block, (
            'np.select default must be "⚪ Laggard" — not empty string or None'
        )

    def test_lifecycle_in_fisher_block(self):
        """fisher_lifecycle_quadrant must be inside the Framework 33 block, not elsewhere."""
        block = _fisher_block(_src())
        assert "fisher_lifecycle_quadrant" in block, (
            "fisher_lifecycle_quadrant not found inside Framework 33 block"
        )

    def test_lifecycle_uses_fisher_quality_pass_not_fw_fisher_directly(self):
        """Lifecycle matrix must reference df['fisher_quality_pass'], not fw_fisher directly."""
        block = _fisher_block(_src())
        assert 'df["fisher_quality_pass"]' in block, (
            'Lifecycle matrix must use df["fisher_quality_pass"] (the materialized column), '
            'not fw_fisher boolean mask directly'
        )

    def test_lifecycle_uses_fisher_pass_not_fw_fisher_scalability_directly(self):
        """Lifecycle matrix must reference df['fisher_pass'], not fw_fisher_scalability directly."""
        block = _fisher_block(_src())
        assert 'df["fisher_pass"]' in block, (
            'Lifecycle matrix must use df["fisher_pass"] (materialized column), '
            'not fw_fisher_scalability boolean mask directly'
        )

    def test_spec_documents_lifecycle_quadrant(self):
        """docs/fisher_quality_specs.json must document fisher_lifecycle_quadrant."""
        spec = _load_spec()
        assert "fisher_lifecycle_quadrant" in spec.get("output_columns", {}), (
            "'fisher_lifecycle_quadrant' not documented in spec output_columns"
        )

    def test_spec_lifecycle_valid_values(self):
        """Spec must list all 4 valid quadrant values."""
        spec = _load_spec()
        valid = spec["output_columns"]["fisher_lifecycle_quadrant"]["valid_values"]
        assert isinstance(valid, list), "valid_values must be a list"
        assert len(valid) == 4, f"Expected 4 quadrant labels, got {len(valid)}"
        for label in VALID_QUADRANT_LABELS:
            assert label in valid, f"'{label}' not in spec valid_values"

    def test_spec_lifecycle_default_value(self):
        """Spec must document default_value as '⚪ Laggard'."""
        spec = _load_spec()
        default = spec["output_columns"]["fisher_lifecycle_quadrant"]["default_value"]
        assert default == "⚪ Laggard", (
            f"Spec lifecycle default must be '⚪ Laggard', got: '{default}'"
        )

    def test_spec_lifecycle_implementation_is_np_select(self):
        """Spec must document np.select as implementation method."""
        spec = _load_spec()
        impl = spec["output_columns"]["fisher_lifecycle_quadrant"]["implementation"]
        assert "np.select" in impl, (
            f"Spec must document 'np.select' as the implementation, got: '{impl}'"
        )


# ═══════════════════════════════════════════════════════════════════════════
# PART 13 — Array fixture: mapping rules mathematical verification
# ═══════════════════════════════════════════════════════════════════════════

class TestLifecycleMappingRulesFixture:
    """Isolated NumPy fixture verifying the 4-quadrant logic without touching scoring_engine.

    Tests the mathematical correctness of the np.select logic independently
    to guard against index alignment drift and boundary condition errors.
    """

    def _compute_quadrant(self, qp_vals, fp_vals):
        """Replicate the np.select logic from scoring_engine.py in isolation."""
        import numpy as np
        import pandas as pd
        n = len(qp_vals)
        idx = range(n)
        fisher_quality_pass = pd.Series(qp_vals, index=idx)
        fisher_pass         = pd.Series(fp_vals,  index=idx)
        _conds = [
            (fisher_quality_pass == 0) & (fisher_pass == 0),
            (fisher_quality_pass == 0) & (fisher_pass == 1),
            (fisher_quality_pass == 1) & (fisher_pass == 0),
            (fisher_quality_pass == 1) & (fisher_pass == 1),
        ]
        _labels = ["⚪ Laggard", "⚡ Catalyst Play", "🐢 Steady Compounder", "👑 Apex Winner"]
        return np.select(_conds, _labels, default="⚪ Laggard")

    def test_both_fail_yields_laggard(self):
        result = self._compute_quadrant([0], [0])
        assert result[0] == "⚪ Laggard", f"(0,0) must be Laggard, got '{result[0]}'"

    def test_quality_fail_scalability_pass_yields_catalyst(self):
        result = self._compute_quadrant([0], [1])
        assert result[0] == "⚡ Catalyst Play", (
            f"(0,1) must be Catalyst Play, got '{result[0]}'"
        )

    def test_quality_pass_scalability_fail_yields_steady(self):
        result = self._compute_quadrant([1], [0])
        assert result[0] == "🐢 Steady Compounder", (
            f"(1,0) must be Steady Compounder, got '{result[0]}'"
        )

    def test_both_pass_yields_apex_winner(self):
        result = self._compute_quadrant([1], [1])
        assert result[0] == "👑 Apex Winner", (
            f"(1,1) must be Apex Winner, got '{result[0]}'"
        )

    def test_all_four_quadrants_in_one_array(self):
        """Verify all 4 states resolve correctly when processed as a single vectorized array."""
        result = self._compute_quadrant([0, 0, 1, 1], [0, 1, 0, 1])
        assert result[0] == "⚪ Laggard",          f"Row 0 (0,0): expected Laggard, got '{result[0]}'"
        assert result[1] == "⚡ Catalyst Play",     f"Row 1 (0,1): expected Catalyst Play, got '{result[1]}'"
        assert result[2] == "🐢 Steady Compounder", f"Row 2 (1,0): expected Steady Compounder, got '{result[2]}'"
        assert result[3] == "👑 Apex Winner",       f"Row 3 (1,1): expected Apex Winner, got '{result[3]}'"

    def test_all_quadrant_outputs_are_in_valid_set(self):
        """No result should fall outside the authorized label set."""
        result = self._compute_quadrant([0, 0, 1, 1], [0, 1, 0, 1])
        for i, val in enumerate(result):
            assert val in VALID_QUADRANT_LABELS, (
                f"Row {i}: result '{val}' is not in the authorized quadrant label set"
            )

    def test_no_index_alignment_drift_with_non_default_index(self):
        """np.select must handle non-default (e.g. shuffled) index without drift."""
        import numpy as np
        import pandas as pd
        idx = [10, 20, 30, 40]
        qp = pd.Series([0, 0, 1, 1], index=idx)
        fp = pd.Series([0, 1, 0, 1], index=idx)
        _conds = [
            (qp == 0) & (fp == 0),
            (qp == 0) & (fp == 1),
            (qp == 1) & (fp == 0),
            (qp == 1) & (fp == 1),
        ]
        _labels = ["⚪ Laggard", "⚡ Catalyst Play", "🐢 Steady Compounder", "👑 Apex Winner"]
        result = np.select(_conds, _labels, default="⚪ Laggard")
        expected = ["⚪ Laggard", "⚡ Catalyst Play", "🐢 Steady Compounder", "👑 Apex Winner"]
        for i, (got, exp) in enumerate(zip(result, expected)):
            assert got == exp, (
                f"Index drift detected at position {i}: expected '{exp}', got '{got}'"
            )


# ═══════════════════════════════════════════════════════════════════════════
# PART 14 — Live data-engine integration tests
# ═══════════════════════════════════════════════════════════════════════════
#
# These three tests exercise the full production pipeline against the actual
# 6-CSV Indian equity dataset (~2107 stocks).  They are end-to-end (not static
# source analysis) so they belong in a separate Part rather than Part 9.
#
# Performance note: _get_production_df() is LRU-cached at module level.
# Only the first test that calls it pays the IO + compute cost (~10-15s on
# an i5 laptop).  All subsequent calls in the same pytest session are free.
# ═══════════════════════════════════════════════════════════════════════════

# ── CSV paths: actual data lives in "Other Resources/CSV Data/" ──────────
# config.py's _get_actual_path resolves to <project_root>/CSV Data/ (a
# non-existent path) when data_source="local".  We bypass this by using
# data_source="upload" and passing the real paths explicitly.
_CSV_DATA_DIR = ROOT / "Other Resources" / "CSV Data"
_UPLOADED_FILES = {
    "ratio":        str(_CSV_DATA_DIR / "Stockscan - Ratio.csv"),
    "income":       str(_CSV_DATA_DIR / "Stockscan - Income Statement.csv"),
    "balance":      str(_CSV_DATA_DIR / "Stockscan - Balance Sheet.csv"),
    "cashflow":     str(_CSV_DATA_DIR / "Stockscan - Cashflow.csv"),
    "shareholding": str(_CSV_DATA_DIR / "Stockscan - Shareholdings.csv"),
    "technical":    str(_CSV_DATA_DIR / "Stockscan - Technicals.csv"),
}


@functools.lru_cache(maxsize=1)
def _get_production_df():
    """
    Full production pipeline: load 6 CSVs → derive 400+ signals → 4-layer scoring.

    LRU-cached so the expensive pipeline (IO + compute) runs ONCE per pytest session.
    Uses 'upload' mode so the explicit paths above are used directly, bypassing
    config.py's local-path resolver which looks in the wrong folder.
    """
    # Ensure core modules are importable when pytest runs from project root
    project_root = str(ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from core.data_engine import fetch_and_clean_data
    from core.scoring_engine import run_full_scoring
    df_raw = fetch_and_clean_data(data_source="upload", uploaded_files=_UPLOADED_FILES)
    return run_full_scoring(df_raw)


class TestFisherLifecycleDataIntegration:
    """
    End-to-end integration tests validating Fisher Lifecycle Quadrant logic
    against the live ~2107-stock Indian equity production dataset.

    Part 14a — Column existence, dtype, and population contracts
    Part 14b — Categorical integrity across the full production universe
    Part 14c — Extreme boundary resilience and fillna strategy verification
    """

    def test_lifecycle_quadrant_production_ingestion(self):
        """
        Invoke fetch_and_clean_data() + run_full_scoring() against the production
        CSVs.  Assert that fisher_quality_pass, fisher_pass, fisher_lifecycle_quadrant,
        and fisher_score are present, fully populated, and correctly typed.
        """
        import pandas as pd
        import numpy as np

        df = _get_production_df()

        # ── Column existence ──
        for col in ("fisher_quality_pass", "fisher_pass",
                    "fisher_score", "fisher_lifecycle_quadrant"):
            assert col in df.columns, (
                f"'{col}' missing from scored production DataFrame"
            )

        # ── Dataset size sanity ──
        assert len(df) >= 2000, (
            f"Expected ≥2000 stocks in production universe, got {len(df)}"
        )

        # ── Binary integer contract: fisher_pass and fisher_quality_pass ──
        for binary_col in ("fisher_pass", "fisher_quality_pass"):
            assert df[binary_col].isna().sum() == 0, (
                f"NaN found in {binary_col} — fillna strategy failed"
            )
            assert df[binary_col].isin([0, 1]).all(), (
                f"{binary_col} contains values other than 0 or 1: "
                f"{df[binary_col].value_counts().to_dict()}"
            )

        # ── fisher_score: clean int, range 0-4 ──
        assert df["fisher_score"].isna().sum() == 0, "NaN in fisher_score"
        assert df["fisher_score"].isin(range(5)).all(), (
            f"fisher_score out of 0-4 range: {df['fisher_score'].value_counts().to_dict()}"
        )

        # ── Lifecycle quadrant: no NaN, string dtype (object or StringDtype) ──
        lc = df["fisher_lifecycle_quadrant"]
        assert lc.isna().sum() == 0, "NaN in fisher_lifecycle_quadrant"
        assert pd.api.types.is_string_dtype(lc), (
            f"fisher_lifecycle_quadrant expected string dtype, got {lc.dtype}"
        )

    def test_lifecycle_quadrant_categorical_integrity(self):
        """
        Vectorially verify that every row in the production universe holds exactly
        one authorized label, and that quadrant-level pass-flag consistency is
        maintained (e.g. every Catalyst Play row must have fisher_pass=1 and
        fisher_quality_pass=0).
        """
        df = _get_production_df()
        col = df["fisher_lifecycle_quadrant"]

        # ── Zero NaN and zero empty-string ──
        assert col.isna().sum() == 0, (
            f"{col.isna().sum()} NaN values in fisher_lifecycle_quadrant"
        )
        assert (col == "").sum() == 0, (
            f"{(col == '').sum()} empty-string values in fisher_lifecycle_quadrant"
        )

        # ── All values within the authorized 4-label set ──
        authorized = frozenset(VALID_QUADRANT_LABELS)  # reuse constant from Part 12
        invalid_mask = ~col.isin(authorized)
        assert invalid_mask.sum() == 0, (
            f"{invalid_mask.sum()} rows have unauthorized quadrant values: "
            f"{col[invalid_mask].unique().tolist()}"
        )

        # ── Default fallback label is present (most stocks are Laggards) ──
        assert "⚪ Laggard" in col.values, (
            "'⚪ Laggard' not found — engine may have failed or used wrong default"
        )

        # ── At least 2 distinct labels (sanity: engine is differentiating) ──
        assert col.nunique() >= 2, (
            f"Only {col.nunique()} distinct quadrant label(s) found — "
            f"engine may not be classifying stocks correctly: "
            f"{col.value_counts().to_dict()}"
        )

        # ── Quadrant ↔ pass-flag consistency ──
        # Apex Winner: quality_pass=1 AND scalability_pass=1
        apex = df[col == "👑 Apex Winner"]
        if len(apex):
            assert (apex["fisher_pass"] == 1).all(), (
                "Apex Winner rows must have fisher_pass=1"
            )
            assert (apex["fisher_quality_pass"] == 1).all(), (
                "Apex Winner rows must have fisher_quality_pass=1"
            )

        # Catalyst Play: quality_pass=0 AND scalability_pass=1
        catalyst = df[col == "⚡ Catalyst Play"]
        if len(catalyst):
            assert (catalyst["fisher_pass"] == 1).all(), (
                "Catalyst Play rows must have fisher_pass=1"
            )
            assert (catalyst["fisher_quality_pass"] == 0).all(), (
                "Catalyst Play rows must have fisher_quality_pass=0"
            )

        # Steady Compounder: quality_pass=1 AND scalability_pass=0
        steady = df[col == "🐢 Steady Compounder"]
        if len(steady):
            assert (steady["fisher_quality_pass"] == 1).all(), (
                "Steady Compounder rows must have fisher_quality_pass=1"
            )
            assert (steady["fisher_pass"] == 0).all(), (
                "Steady Compounder rows must have fisher_pass=0"
            )

        # Laggard: both flags = 0
        laggard = df[col == "⚪ Laggard"]
        if len(laggard):
            assert (laggard["fisher_pass"] == 0).all(), (
                "Laggard rows must have fisher_pass=0"
            )
            assert (laggard["fisher_quality_pass"] == 0).all(), (
                "Laggard rows must have fisher_quality_pass=0"
            )

    def test_lifecycle_extreme_boundary_resilience(self):
        """
        Construct a 3-row mock DataFrame with real-market extreme edge cases:
          Row 0 — Revenue collapse (-95%) with expense blowout and heavy dilution
          Row 1 — All-NaN banking/NBFC row (working capital metrics absent)
          Row 2 — Expense runaway (-350% d05) with NaN dilution data

        Asserts two independent pillars:

        Pillar A — compute_qglp_score() resilience:
            Survives extreme/NaN inputs, produces qglp_score in [0, 100].

        Pillar B — Fisher Scalability fillna strategy correctness:
            * rev/oplev NaN → fillna(0) → fails threshold (conservative gate failure)
            * opm/npm_acceleration NaN → fillna(0) → 0 >= 0 → OR gate passes (neutral)
            * dilution_pct NaN → fillna(999) → 999 > 1 → gate fails (conservative exclusion)
        Exact per-row scores asserted: Row0=0, Row1=1, Row2=2.
        """
        import numpy as np
        import pandas as pd

        project_root = str(ROOT)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from core.scoring_engine import compute_qglp_score

        # ── Extreme mock matrix ──
        mock_df = pd.DataFrame({
            # Fisher Scalability columns
            "rev_gr_3y":            [-95.0,  np.nan,   20.0],
            "rev_gr_yoy":           [-80.0,  np.nan,   15.0],
            "d05_rev_minus_exp_gr": [-400.0, np.nan, -350.0],
            "opm_acceleration":     [-15.0,  np.nan,  np.nan],
            "npm_acceleration":     [-10.0,  np.nan,  np.nan],
            "dilution_pct":         [ 25.0,  np.nan,  np.nan],
            # QGLP scoring columns (all accessed via .get() + .fillna(50))
            "roce":                 [ -5.0,  np.nan,   15.0],
            "pat_gr_5y":            [-30.0,  np.nan,   18.0],
            "eps_gr_5y":            [-25.0,  np.nan,   16.0],
            "roe_med_10y":          [  5.0,  np.nan,   14.0],
            "peg":                  [np.nan, np.nan,    1.5],
        })

        # ─────────────────────────────────────────
        # Pillar A: compute_qglp_score resilience
        # ─────────────────────────────────────────
        qglp_out = compute_qglp_score(mock_df.copy())
        assert "qglp_score" in qglp_out.columns, (
            "qglp_score column absent from compute_qglp_score output"
        )
        assert qglp_out["qglp_score"].isna().sum() == 0, (
            "NaN in qglp_score — .fillna(50) chain failed on extreme input"
        )
        assert (qglp_out["qglp_score"] >= 0).all() and \
               (qglp_out["qglp_score"] <= 100).all(), (
            f"qglp_score out of [0,100] range: {qglp_out['qglp_score'].tolist()}"
        )

        # ─────────────────────────────────────────────────────────────────
        # Pillar B: exact replication of Framework 33 fillna gate logic
        # ─────────────────────────────────────────────────────────────────
        _nan_s = pd.Series(np.nan, index=mock_df.index)
        rev_3y  = mock_df.get("rev_gr_3y",            _nan_s)
        rev_yoy = mock_df.get("rev_gr_yoy",            _nan_s)
        oplev   = mock_df.get("d05_rev_minus_exp_gr",  _nan_s)
        opm_a   = mock_df.get("opm_acceleration",       _nan_s)
        npm_a   = mock_df.get("npm_acceleration",       _nan_s)
        dil     = mock_df.get("dilution_pct",           _nan_s)

        gate_rev   = (rev_3y.fillna(0)  >= 12.0) & (rev_yoy.fillna(0) >= 10.0)
        gate_oplev = oplev.fillna(0)   >= 2.0
        gate_price = (opm_a.fillna(0)  >= 0)    | (npm_a.fillna(0)   >= 0)
        gate_dil   = dil.fillna(999)   <= 1.0

        score = (
            gate_rev.astype(int) + gate_oplev.astype(int) +
            gate_price.astype(int) + gate_dil.astype(int)
        )

        # Row 0 — revenue collapse: all 4 gates fail → score = 0
        #   rev_gr_3y=-95 < 12 → False; d05=-400 < 2 → False;
        #   opm_a=-15 & npm_a=-10 → both <0 → OR=False; dil=25 > 1 → False
        assert int(score.iloc[0]) == 0, (
            f"Row 0 (revenue collapse): expected fisher_score=0, got {score.iloc[0]}"
        )

        # Row 1 — all-NaN banking: only pricing gate passes via fillna(0) → score = 1
        #   rev NaN→0 < 12 → False; d05 NaN→0 < 2 → False;
        #   opm_a NaN→0 >= 0 → True (OR gate passes — neutral, not conservative);
        #   dil NaN→999 > 1 → False (conservative exclusion)
        assert int(score.iloc[1]) == 1, (
            f"Row 1 (all-NaN): expected fisher_score=1 "
            f"(only pricing passes via fillna(0) neutral strategy), got {score.iloc[1]}"
        )

        # Row 2 — expense runaway: rev runway passes + pricing passes (NaN→neutral) → score = 2
        #   rev_gr_3y=20>=12 AND rev_gr_yoy=15>=10 → True;
        #   d05=-350 < 2 → False;
        #   opm_a NaN→0>=0 OR npm_a NaN→0>=0 → True;
        #   dil NaN→999 > 1 → False
        assert int(score.iloc[2]) == 2, (
            f"Row 2 (expense runaway): expected fisher_score=2, got {score.iloc[2]}"
        )

        # All scores are clean ints in 0-4 (no NaN, no float)
        score_ints = score.astype(int)
        assert score_ints.isna().sum() == 0, "NaN in computed fisher_score"
        assert score_ints.isin(range(5)).all(), (
            f"fisher_score out of 0-4: {score_ints.tolist()}"
        )

        # ──────────────────────────────────────────────────────────────────
        # Pillar C: np.select produces valid lifecycle labels from mock data
        # ──────────────────────────────────────────────────────────────────
        fw_scal = gate_rev & gate_oplev & gate_price & gate_dil
        _mock = mock_df.copy()
        _mock["fisher_pass"]         = fw_scal.astype(int)
        _mock["fisher_quality_pass"] = 0   # no quality columns in mock → default 0
        _mock["fisher_score"]        = score_ints

        _conds = [
            (_mock["fisher_quality_pass"] == 0) & (_mock["fisher_pass"] == 0),
            (_mock["fisher_quality_pass"] == 0) & (_mock["fisher_pass"] == 1),
            (_mock["fisher_quality_pass"] == 1) & (_mock["fisher_pass"] == 0),
            (_mock["fisher_quality_pass"] == 1) & (_mock["fisher_pass"] == 1),
        ]
        _lc_labels = ["⚪ Laggard", "⚡ Catalyst Play",
                      "🐢 Steady Compounder", "👑 Apex Winner"]
        _mock["fisher_lifecycle_quadrant"] = np.select(
            _conds, _lc_labels, default="⚪ Laggard"
        )

        valid_set = frozenset(VALID_QUADRANT_LABELS)
        assert all(v in valid_set for v in _mock["fisher_lifecycle_quadrant"]), (
            f"Invalid lifecycle labels in extreme mock: "
            f"{_mock['fisher_lifecycle_quadrant'].tolist()}"
        )
        # All 3 extreme rows have fisher_pass=0 → all Laggard
        assert (_mock["fisher_lifecycle_quadrant"] == "⚪ Laggard").all(), (
            f"Expected all extreme rows to be '⚪ Laggard' (no row passes all 4 gates), "
            f"got: {_mock['fisher_lifecycle_quadrant'].tolist()}"
        )
        assert _mock["fisher_lifecycle_quadrant"].isna().sum() == 0, (
            "NaN in fisher_lifecycle_quadrant output for extreme mock"
        )
