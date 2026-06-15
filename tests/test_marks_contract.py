"""
Contract Tests — Howard Marks Market Cycle & Risk Defensive Shield
==================================================================
Automated verification that docs/marks_cycle_specs.json and
core/scoring_engine.py are in perfect alignment.

Version 1.1 — India-calibrated thresholds (2026-05-25 audit):
  • Pillar L: debt_to_equity < 0.5  (companion Ch.9 "D/E < 0.5")
  • Pillar D: cfo_to_pat >= 80.0    (companion Ch.9 "CFO/PAT > 0.8")

Structure:
    TestMarksSpecLedger          — JSON schema completeness and meta keys
    TestMarksEngineContract      — regex source-code threshold verification
    TestMarksPillarArithmetic    — boundary conditions, AND invariant, score 0-4
    TestMarksIndexAlignment      — non-default integer and string index safety
    TestMarksUIContract          — render_marks_radar import + column reads
    TestMarksRawSignalsContract  — marks cells present in render_raw_signals
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

SPEC_PATH = os.path.join(REPO_ROOT, "docs", "marks_cycle_specs.json")
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

def _build_mock_marks_row(**overrides) -> dict:
    """Build a fully-passing Marks Cycle Shield row (all 4 pillars green).

    Defaults are calibrated for v1.1 India thresholds:
      - debt_to_equity=0.3  (< 0.5 threshold)
      - cfo_to_pat=85.0     (≥ 80.0 threshold)
    """
    base = {
        "mean_reversion_risk": 0,                           # M: no margin spike
        "buy_zone_label":      "🟢 Perfect Entry (Low Risk)",   # P: in buy zone
        "debt_to_equity":      0.3,                         # L: D/E well below 0.5
        "cfo_to_pat":          85.0,                        # D: 85% > 80% threshold
        # scaffold columns
        "market_cap":   5000.0,
        "close_price":  500.0,
        "name":         "TestCo",
        "sector":       "FMCG",
        "is_financial": False,
    }
    base.update(overrides)
    return base


def _run_marks(rows: list, regime: str = "SIDEWAYS") -> pd.DataFrame:
    """Execute compute_qglp_score on a list of row dicts; return result df."""
    from config import MASTER_PROFILES
    from core.scoring_engine import compute_qglp_score
    df = pd.DataFrame(rows)
    df.attrs["detected_market_regime"] = regime
    profile = MASTER_PROFILES.get("Balanced", next(iter(MASTER_PROFILES.values())))
    return compute_qglp_score(df, profile)


# ═══════════════════════════════════════════════════════════════════════════════
# TestMarksSpecLedger
# ═══════════════════════════════════════════════════════════════════════════════

class TestMarksSpecLedger:
    """Verify docs/marks_cycle_specs.json is complete and structurally correct."""

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
        assert spec["_meta"]["pass_column"] == "marks_pass"

    def test_score_column_name(self, spec):
        assert spec["_meta"]["score_column"] == "marks_score"

    def test_framework_variable_name(self, spec):
        assert spec["_meta"]["framework_variable"] == "fw_marks_cycle"

    def test_framework_label(self, spec):
        assert spec["_meta"]["frameworks_passed_label"] == "Marks Cycle Shield"

    def test_version_is_india_calibrated(self, spec):
        """Version must reflect the v1.1 India-calibrated audit (2026-05-25)."""
        assert "1.1" in spec["_meta"]["version"], (
            f"Expected version 1.1 (India-calibrated). Got: {spec['_meta']['version']}"
        )

    def test_4_pillar_sections_present(self, spec):
        required_pillars = [
            "pillar_m_margin_protection",
            "pillar_p_price_value",
            "pillar_l_leverage_discipline",
            "pillar_d_defensive_cash",
        ]
        for p in required_pillars:
            assert p in spec, f"Spec missing pillar section: {p}"

    def test_output_columns_registry_present(self, spec):
        assert "output_columns_registry" in spec
        required = [
            "marks_margin_spike", "marks_price_value",
            "marks_leverage_trap", "marks_defensive_base",
            "marks_pass", "marks_score",
        ]
        for col in required:
            assert col in spec["output_columns_registry"], (
                f"output_columns_registry missing: {col}"
            )

    def test_pillar_m_threshold_is_zero(self, spec):
        assert spec["pillar_m_margin_protection"]["margin_protection_gate"]["threshold"] == 0

    def test_pillar_p_threshold_is_buy_zone_label(self, spec):
        t = spec["pillar_p_price_value"]["price_value_gate"]["threshold"]
        assert "🟢" in t and "Perfect Entry" in t, (
            f"Pillar P threshold must reference the buy zone label; got: {t}"
        )

    def test_pillar_l_threshold_is_0point5(self, spec):
        """India-calibrated: D/E < 0.5 (companion Ch.9 Pillar 1 Quality Floor)."""
        t = spec["pillar_l_leverage_discipline"]["leverage_discipline_gate"]["threshold"]
        assert abs(t - 0.5) < 1e-9, (
            f"Pillar L threshold must be 0.5 (India defensive floor). Got: {t}"
        )

    def test_pillar_l_operator_is_strict_less_than(self, spec):
        """Companion says 'D/E < 0.5' — operator must be strict '<', not '<='."""
        op = spec["pillar_l_leverage_discipline"]["leverage_discipline_gate"]["operator"]
        assert op == "<", (
            f"Pillar L operator must be '<' (strict less than). Got: '{op}'"
        )

    def test_pillar_d_threshold_is_80(self, spec):
        """India-calibrated: CFO/PAT >= 80.0 (companion Ch.9 'CFO/PAT > 0.8')."""
        t = spec["pillar_d_defensive_cash"]["defensive_cash_gate"]["threshold"]
        assert abs(t - 80.0) < 1e-9, (
            f"Pillar D threshold must be 80.0 (India defensive floor). Got: {t}"
        )

    def test_pillar_d_unit_warning_present(self, spec):
        """Critical unit warning must document cfo_to_pat is PERCENTAGE not ratio."""
        gate = spec["pillar_d_defensive_cash"]["defensive_cash_gate"]
        assert "unit_warning" in gate
        warning = gate["unit_warning"]
        assert "80.0" in warning, f"unit_warning must mention '80.0'; got: {warning}"
        assert "0.80" in warning, f"unit_warning must mention '0.80'; got: {warning}"

    def test_pillar_l_audit_provenance_present(self, spec):
        """v1.1 refinement must be traceable to textbook audit."""
        gate = spec["pillar_l_leverage_discipline"]["leverage_discipline_gate"]
        assert "audit_provenance" in gate, (
            "Pillar L must have audit_provenance documenting the 1.0 → 0.5 refinement"
        )

    def test_pillar_d_audit_provenance_present(self, spec):
        """v1.1 refinement must be traceable to textbook audit."""
        gate = spec["pillar_d_defensive_cash"]["defensive_cash_gate"]
        assert "audit_provenance" in gate, (
            "Pillar D must have audit_provenance documenting the 70% → 80% refinement"
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

    def test_vectorization_matrix_present(self, spec):
        assert "vectorization_matrix" in spec
        vm = spec["vectorization_matrix"]["nan_handling"]
        for col in ["mean_reversion_risk", "buy_zone_label", "debt_to_equity", "cfo_to_pat"]:
            assert col in vm, f"vectorization_matrix missing NaN handling for: {col}"

    def test_scoring_matrix_present(self, spec):
        assert "scoring_matrix" in spec
        sm = spec["scoring_matrix"]
        assert sm["all_gates_equal_weight"] is True
        assert sm["score_range"] == "0-4"


# ═══════════════════════════════════════════════════════════════════════════════
# TestMarksEngineContract
# ═══════════════════════════════════════════════════════════════════════════════

class TestMarksEngineContract:
    """Verify scoring_engine.py has all marks columns and correct v1.1 thresholds."""

    def test_marks_anchor_comment(self, se_source):
        assert "Marks Market Cycle & Risk Defensive Shield" in se_source

    def test_marks_margin_spike_column_defined(self, se_source):
        assert 'df["marks_margin_spike"]' in se_source

    def test_marks_price_value_column_defined(self, se_source):
        assert 'df["marks_price_value"]' in se_source

    def test_marks_leverage_trap_column_defined(self, se_source):
        assert 'df["marks_leverage_trap"]' in se_source

    def test_marks_defensive_base_column_defined(self, se_source):
        assert 'df["marks_defensive_base"]' in se_source

    def test_marks_pass_column_defined(self, se_source):
        assert 'df["marks_pass"]' in se_source

    def test_marks_score_column_defined(self, se_source):
        assert 'df["marks_score"]' in se_source

    def test_mean_reversion_risk_equality_zero(self, se_source):
        """Pillar M: mean_reversion_risk == 0 (no margin spike)."""
        assert re.search(r'mean_reversion_risk.*==\s*0', se_source), (
            "scoring_engine must check mean_reversion_risk == 0 for marks_margin_spike"
        )

    def test_buy_zone_label_equality(self, se_source):
        """Pillar P: buy_zone_label == '🟢 Perfect Entry (Low Risk)'."""
        assert "🟢 Perfect Entry (Low Risk)" in se_source, (
            "scoring_engine must check buy_zone_label == '🟢 Perfect Entry (Low Risk)'"
        )

    def test_debt_to_equity_strict_lt_0point5(self, se_source):
        """Pillar L v1.1: debt_to_equity < 0.5 (India companion Ch.9 'D/E < 0.5')."""
        assert re.search(r'debt_to_equity.*<\s*0\.5', se_source), (
            "scoring_engine must check debt_to_equity < 0.5 for marks_leverage_trap "
            "(India-calibrated threshold from companion Ch.9)"
        )

    def test_debt_to_equity_not_lte_1(self, se_source):
        """Pillar L old threshold (<=1.0) must NOT appear in marks block."""
        # Find the marks block specifically
        marks_block_match = re.search(
            r'Marks Market Cycle.*?(?=# ──.*?(?:Schilit|Build|fw_str))',
            se_source, re.DOTALL
        )
        assert marks_block_match, "Could not isolate Marks block in scoring_engine"
        marks_block = marks_block_match.group(0)
        assert not re.search(r'marks_leverage_trap.*<=\s*1\.0', marks_block), (
            "marks_leverage_trap must NOT use old threshold <= 1.0; should be < 0.5"
        )

    def test_cfo_to_pat_gte_80(self, se_source):
        """Pillar D v1.1: cfo_to_pat >= 80.0 (India companion Ch.9 'CFO/PAT > 0.8')."""
        assert re.search(r'cfo_to_pat.*>=\s*80\.0', se_source), (
            "scoring_engine must check cfo_to_pat >= 80.0 for marks_defensive_base "
            "(India-calibrated threshold from companion Ch.9)"
        )

    def test_fw_marks_cycle_is_and_of_4_pillars(self, se_source):
        """fw_marks_cycle = AND of all 4 materialized pillar columns."""
        assert re.search(r'fw_marks_cycle\s*=\s*\(', se_source)
        for col in ["marks_margin_spike", "marks_price_value",
                    "marks_leverage_trap", "marks_defensive_base"]:
            assert col in se_source, f"fw_marks_cycle must reference {col}"

    def test_fw_str_includes_marks_label(self, se_source):
        """fw_str must include 'Marks Cycle Shield|' via fw_marks_cycle."""
        assert re.search(r'fw_marks_cycle.*Marks Cycle Shield', se_source), (
            "fw_str must have np.where(fw_marks_cycle, 'Marks Cycle Shield|', '')"
        )

    def test_nan_fillna_mean_reversion_risk(self, se_source):
        """Missing mean_reversion_risk must default to 0 (no spike = conservative pass)."""
        assert re.search(r'mean_reversion_risk.*fillna\s*\(\s*0\s*\)', se_source)

    def test_nan_fillna_debt_to_equity_conservative(self, se_source):
        """Missing debt_to_equity must default to 999 (leverage unknown = fails: 999 ≥ 0.5)."""
        assert re.search(r'debt_to_equity.*fillna\s*\(\s*999\s*\)', se_source)

    def test_marks_score_is_sum_of_4_pillars(self, se_source):
        """marks_score = sum of the 4 pillar flags (0-4 range)."""
        assert re.search(r'marks_score.*marks_margin_spike', se_source, re.DOTALL), (
            "marks_score must sum all 4 pillar flags"
        )

    def test_companion_calibration_comment_present(self, se_source):
        """v1.1 comments must reference companion chapter for both thresholds."""
        assert "Companion Ch.9" in se_source or "companion" in se_source.lower(), (
            "scoring_engine comment block must reference the companion India calibration"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TestMarksPillarArithmetic
# ═══════════════════════════════════════════════════════════════════════════════

class TestMarksPillarArithmetic:
    """Boundary conditions and arithmetic invariants for all 4 pillar gates."""

    # ── Pillar M: Margin Spike Guard ──────────────────────────────────────────

    def test_pillar_m_no_spike_passes(self):
        result = _run_marks([_build_mock_marks_row(mean_reversion_risk=0)])
        assert result["marks_margin_spike"].iloc[0] == 1

    def test_pillar_m_spike_flag_1_fails(self):
        result = _run_marks([_build_mock_marks_row(mean_reversion_risk=1)])
        assert result["marks_margin_spike"].iloc[0] == 0

    def test_pillar_m_nan_passes_conservatively(self):
        """Missing mean_reversion_risk → fillna(0) → 0 == 0 → gate passes."""
        result = _run_marks([_build_mock_marks_row(mean_reversion_risk=float("nan"))])
        assert result["marks_margin_spike"].iloc[0] == 1

    # ── Pillar P: Price vs Value ───────────────────────────────────────────────

    def test_pillar_p_in_buy_zone_passes(self):
        result = _run_marks([_build_mock_marks_row(buy_zone_label="🟢 Perfect Entry (Low Risk)")])
        assert result["marks_price_value"].iloc[0] == 1

    def test_pillar_p_standard_zone_fails(self):
        result = _run_marks([_build_mock_marks_row(buy_zone_label="🟡 Standard Zone")])
        assert result["marks_price_value"].iloc[0] == 0

    def test_pillar_p_extended_fails(self):
        result = _run_marks([_build_mock_marks_row(buy_zone_label="🔴 Extended (Wait for Pullback)")])
        assert result["marks_price_value"].iloc[0] == 0

    def test_pillar_p_uncharted_fails(self):
        result = _run_marks([_build_mock_marks_row(buy_zone_label="⚪ Uncharted")])
        assert result["marks_price_value"].iloc[0] == 0

    def test_pillar_p_nan_fails_conservatively(self):
        """Missing buy_zone_label → fillna('') → '' != buy zone label → gate fails."""
        result = _run_marks([_build_mock_marks_row(buy_zone_label=float("nan"))])
        assert result["marks_price_value"].iloc[0] == 0

    # ── Pillar L: Leverage Discipline (v1.1 India threshold: D/E < 0.5) ──────

    def test_pillar_l_de_well_below_boundary_passes(self):
        """D/E = 0.3 is well below 0.5 boundary → passes."""
        result = _run_marks([_build_mock_marks_row(debt_to_equity=0.3)])
        assert result["marks_leverage_trap"].iloc[0] == 1

    def test_pillar_l_de_just_below_boundary_passes(self):
        """D/E = 0.49 < 0.5 → passes (just inside strict boundary)."""
        result = _run_marks([_build_mock_marks_row(debt_to_equity=0.49)])
        assert result["marks_leverage_trap"].iloc[0] == 1

    def test_pillar_l_de_at_boundary_fails(self):
        """D/E = 0.5 is NOT < 0.5 (strict boundary) → fails."""
        result = _run_marks([_build_mock_marks_row(debt_to_equity=0.5)])
        assert result["marks_leverage_trap"].iloc[0] == 0, (
            "D/E = 0.5 must fail: threshold is strictly < 0.5 (India companion: 'D/E < 0.5')"
        )

    def test_pillar_l_de_just_above_boundary_fails(self):
        """D/E = 0.51 > 0.5 → fails."""
        result = _run_marks([_build_mock_marks_row(debt_to_equity=0.51)])
        assert result["marks_leverage_trap"].iloc[0] == 0

    def test_pillar_l_de_old_threshold_1_now_fails(self):
        """Regression guard: old threshold 1.0 now FAILS (v1.1 tightening to 0.5)."""
        result = _run_marks([_build_mock_marks_row(debt_to_equity=1.0)])
        assert result["marks_leverage_trap"].iloc[0] == 0, (
            "D/E = 1.0 must now FAIL (v1.1 tightened from <= 1.0 to < 0.5)"
        )

    def test_pillar_l_high_de_fails(self):
        result = _run_marks([_build_mock_marks_row(debt_to_equity=3.5)])
        assert result["marks_leverage_trap"].iloc[0] == 0

    def test_pillar_l_zero_de_passes(self):
        """D/E = 0.0 (debt-free company) must pass."""
        result = _run_marks([_build_mock_marks_row(debt_to_equity=0.0)])
        assert result["marks_leverage_trap"].iloc[0] == 1

    def test_pillar_l_nan_fails_conservatively(self):
        """Missing debt_to_equity → fillna(999) → 999 ≥ 0.5 → gate fails."""
        result = _run_marks([_build_mock_marks_row(debt_to_equity=float("nan"))])
        assert result["marks_leverage_trap"].iloc[0] == 0

    # ── Pillar D: Defensive Cash (v1.1 India threshold: CFO/PAT >= 80.0) ─────

    def test_pillar_d_at_exact_boundary_passes(self):
        """Exactly 80.0% = at threshold → passes (>= 80.0)."""
        result = _run_marks([_build_mock_marks_row(cfo_to_pat=80.0)])
        assert result["marks_defensive_base"].iloc[0] == 1, (
            "cfo_to_pat=80.0 must pass marks_defensive_base (threshold=80.0, inclusive)"
        )

    def test_pillar_d_above_boundary_passes(self):
        result = _run_marks([_build_mock_marks_row(cfo_to_pat=90.0)])
        assert result["marks_defensive_base"].iloc[0] == 1

    def test_pillar_d_100pct_passes(self):
        """100% CFO/PAT (gold standard — CFO >= PAT) must pass."""
        result = _run_marks([_build_mock_marks_row(cfo_to_pat=100.0)])
        assert result["marks_defensive_base"].iloc[0] == 1

    def test_pillar_d_just_below_boundary_fails(self):
        """79.9% < 80% → fails."""
        result = _run_marks([_build_mock_marks_row(cfo_to_pat=79.9)])
        assert result["marks_defensive_base"].iloc[0] == 0

    def test_pillar_d_old_threshold_70_now_fails(self):
        """Regression guard: old threshold 70.0 now FAILS (v1.1 tightening to 80.0)."""
        result = _run_marks([_build_mock_marks_row(cfo_to_pat=70.0)])
        assert result["marks_defensive_base"].iloc[0] == 0, (
            "cfo_to_pat=70.0 must now FAIL (v1.1 tightened from >= 70.0 to >= 80.0)"
        )

    def test_pillar_d_72pct_fails(self):
        """72% < 80% → fails (previously passed at old 70% threshold — regression guard)."""
        result = _run_marks([_build_mock_marks_row(cfo_to_pat=72.0)])
        assert result["marks_defensive_base"].iloc[0] == 0, (
            "cfo_to_pat=72.0 must fail with 80% threshold"
        )

    def test_pillar_d_threshold_less_strict_than_outsider(self):
        """Marks=80% < Outsider CEO=85%. A stock at 82% passes Marks."""
        result = _run_marks([_build_mock_marks_row(cfo_to_pat=82.0)])
        assert result["marks_defensive_base"].iloc[0] == 1, (
            "cfo_to_pat=82.0 must pass marks_defensive_base (Marks=80% < Outsider=85%)"
        )

    def test_pillar_d_nan_fails_conservatively(self):
        result = _run_marks([_build_mock_marks_row(cfo_to_pat=float("nan"))])
        assert result["marks_defensive_base"].iloc[0] == 0

    # ── AND Invariant: marks_pass ─────────────────────────────────────────────

    def test_all_4_pass_marks_pass_is_1(self):
        result = _run_marks([_build_mock_marks_row()])
        assert result["marks_pass"].iloc[0] == 1

    def test_single_pillar_failure_marks_pass_is_0(self):
        """Any single pillar failing must flip marks_pass to 0."""
        for fail_case in [
            {"mean_reversion_risk": 1},
            {"buy_zone_label": "🟡 Standard Zone"},
            {"debt_to_equity": 2.0},
            {"cfo_to_pat": 60.0},
        ]:
            result = _run_marks([_build_mock_marks_row(**fail_case)])
            assert result["marks_pass"].iloc[0] == 0, (
                f"marks_pass must be 0 when {fail_case} fails"
            )

    def test_de_0point5_causes_pass_failure(self):
        """D/E = 0.5 exactly — strict boundary — must cause marks_pass=0."""
        result = _run_marks([_build_mock_marks_row(debt_to_equity=0.5)])
        assert result["marks_pass"].iloc[0] == 0, (
            "D/E = 0.5 must fail marks_pass (strict < 0.5 boundary)"
        )

    def test_cfo_79_causes_pass_failure(self):
        """CFO/PAT = 79.9 — just below boundary — must cause marks_pass=0."""
        result = _run_marks([_build_mock_marks_row(cfo_to_pat=79.9)])
        assert result["marks_pass"].iloc[0] == 0, (
            "cfo_to_pat=79.9 must fail marks_pass (>= 80.0 boundary)"
        )

    # ── Score Range ────────────────────────────────────────────────────────────

    def test_score_all_pass_is_4(self):
        result = _run_marks([_build_mock_marks_row()])
        assert result["marks_score"].iloc[0] == 4

    def test_score_all_fail_is_0(self):
        result = _run_marks([_build_mock_marks_row(
            mean_reversion_risk=1,
            buy_zone_label="🔴 Extended (Wait for Pullback)",
            debt_to_equity=3.0,
            cfo_to_pat=40.0,
        )])
        assert result["marks_score"].iloc[0] == 0

    def test_score_3_when_d_fails(self):
        """D pillar failure (cfo_to_pat=60.0 < 80) → score=3."""
        result = _run_marks([_build_mock_marks_row(cfo_to_pat=60.0)])
        assert result["marks_score"].iloc[0] == 3

    def test_score_3_when_l_fails(self):
        """L pillar failure (D/E=0.5 at strict boundary) → score=3."""
        result = _run_marks([_build_mock_marks_row(debt_to_equity=0.5)])
        assert result["marks_score"].iloc[0] == 3

    def test_score_2_when_two_fail(self):
        result = _run_marks([_build_mock_marks_row(
            cfo_to_pat=60.0, debt_to_equity=2.0
        )])
        assert result["marks_score"].iloc[0] == 2

    def test_score_range_always_0_to_4(self):
        combos = [
            _build_mock_marks_row(),
            _build_mock_marks_row(mean_reversion_risk=1),
            _build_mock_marks_row(debt_to_equity=2.0, cfo_to_pat=50.0),
            _build_mock_marks_row(
                mean_reversion_risk=1,
                buy_zone_label="🔴 Extended (Wait for Pullback)",
                debt_to_equity=3.0, cfo_to_pat=40.0,
            ),
        ]
        result = _run_marks(combos)
        assert result["marks_score"].between(0, 4).all()

    def test_pass_equals_score_4(self):
        """marks_pass == 1 ⟺ marks_score == 4 (AND identity)."""
        combos = [
            _build_mock_marks_row(),                       # all pass → score=4
            _build_mock_marks_row(cfo_to_pat=60.0),        # D fails → score=3
            _build_mock_marks_row(
                mean_reversion_risk=1,
                buy_zone_label="🔴 Extended (Wait for Pullback)",
                debt_to_equity=3.0, cfo_to_pat=40.0,
            ),  # all fail → score=0
        ]
        result = _run_marks(combos)
        assert (result["marks_pass"] == (result["marks_score"] == 4).astype(int)).all()


# ═══════════════════════════════════════════════════════════════════════════════
# TestMarksIndexAlignment
# ═══════════════════════════════════════════════════════════════════════════════

class TestMarksIndexAlignment:
    """Non-default index safety — guards against pandas index alignment drift."""

    def test_integer_index_non_default(self):
        """Results must be correctly aligned with a shuffled integer index."""
        rows = [
            _build_mock_marks_row(),                          # all pass (D/E=0.3, CFO=85)
            _build_mock_marks_row(cfo_to_pat=60.0),           # D fails (60 < 80)
            _build_mock_marks_row(debt_to_equity=2.0),        # L fails (2.0 ≥ 0.5)
        ]
        from config import MASTER_PROFILES
        from core.scoring_engine import compute_qglp_score
        df = pd.DataFrame(rows, index=[100, 200, 300])
        df.attrs["detected_market_regime"] = "SIDEWAYS"
        profile = MASTER_PROFILES.get("Balanced", next(iter(MASTER_PROFILES.values())))
        result = compute_qglp_score(df, profile)

        assert result.index.tolist() == [100, 200, 300], "Index must be preserved"
        assert result.loc[100, "marks_pass"] == 1
        assert result.loc[200, "marks_pass"] == 0   # D pillar failed
        assert result.loc[300, "marks_pass"] == 0   # L pillar failed

    def test_string_index(self):
        """Results must be correctly aligned with a string index."""
        rows = [
            _build_mock_marks_row(cfo_to_pat=85.0),             # all pass
            _build_mock_marks_row(mean_reversion_risk=1),        # M fails
            _build_mock_marks_row(),                             # all pass
        ]
        from config import MASTER_PROFILES
        from core.scoring_engine import compute_qglp_score
        df = pd.DataFrame(rows, index=["HDFC", "TCS", "ITC"])
        df.attrs["detected_market_regime"] = "SIDEWAYS"
        profile = MASTER_PROFILES.get("Balanced", next(iter(MASTER_PROFILES.values())))
        result = compute_qglp_score(df, profile)

        assert result.index.tolist() == ["HDFC", "TCS", "ITC"]
        assert result.loc["HDFC", "marks_pass"] == 1
        assert result.loc["TCS",  "marks_pass"] == 0   # M pillar failed
        assert result.loc["ITC",  "marks_pass"] == 1

    def test_all_nan_row_fails_conservatively(self):
        """A row with all NaN inputs must have marks_pass=0; score<=1."""
        nan_row = {
            "mean_reversion_risk": float("nan"),
            "buy_zone_label":      float("nan"),
            "debt_to_equity":      float("nan"),
            "cfo_to_pat":          float("nan"),
        }
        result = _run_marks([nan_row])
        # mean_reversion_risk NaN → 0 → M passes (fillna 0 → 0 == 0)
        # buy_zone NaN → '' → P fails
        # debt_to_equity NaN → 999 → 999 >= 0.5 → L fails (999 < 0.5 is False)
        # cfo_to_pat NaN → 0 → 0 < 80 → D fails
        assert result["marks_pass"].iloc[0]  == 0, "All-NaN row must fail marks_pass"
        assert result["marks_score"].iloc[0] <= 1, (
            "All-NaN row: only M can pass (fillna(0) == 0); score must be 0 or 1"
        )

    def test_de_nan_fails_leverage_pillar(self):
        """NaN debt_to_equity → fillna(999) → 999 < 0.5 is False → L fails."""
        result = _run_marks([_build_mock_marks_row(debt_to_equity=float("nan"))])
        assert result["marks_leverage_trap"].iloc[0] == 0, (
            "NaN D/E → fillna(999) → 999 >= 0.5 → marks_leverage_trap must be 0"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TestMarksUIContract
# ═══════════════════════════════════════════════════════════════════════════════

class TestMarksUIContract:
    """Verify render_marks_radar is correctly exported and reads the right columns."""

    def test_render_marks_radar_importable(self):
        from ui import render_marks_radar
        assert callable(render_marks_radar)

    def test_render_marks_radar_in_all(self):
        import ui
        assert "render_marks_radar" in ui.__all__

    def test_ui_reads_marks_margin_spike(self, ui_source):
        assert "marks_margin_spike" in ui_source

    def test_ui_reads_marks_price_value(self, ui_source):
        assert "marks_price_value" in ui_source

    def test_ui_reads_marks_leverage_trap(self, ui_source):
        assert "marks_leverage_trap" in ui_source

    def test_ui_reads_marks_defensive_base(self, ui_source):
        assert "marks_defensive_base" in ui_source

    def test_ui_reads_marks_pass(self, ui_source):
        assert '"marks_pass"' in ui_source

    def test_ui_reads_marks_score(self, ui_source):
        assert '"marks_score"' in ui_source

    def test_ui_uses_no_threshold_math(self, ui_source):
        """Pure display: render_marks_radar must not contain numeric threshold literals."""
        match = re.search(r'def render_marks_radar.*?(?=\ndef |\Z)', ui_source, re.DOTALL)
        assert match, "render_marks_radar function not found in ui_tearsheet.py"
        fn_body = match.group(0)
        forbidden = [r'>= 70', r'>= 80', r'<= 1\.0', r'< 0\.5', r'>= 85']
        for pattern in forbidden:
            assert not re.search(pattern, fn_body), (
                f"render_marks_radar contains threshold math: {pattern}. "
                "Pure display layer must never re-compute thresholds."
            )

    def test_ui_has_cyan_teal_color(self, ui_source):
        """Marks radar must use Cyan/Teal (#00CED1) accent as specified."""
        match = re.search(r'def render_marks_radar.*?(?=\ndef |\Z)', ui_source, re.DOTALL)
        assert match
        fn_body = match.group(0)
        assert "00CED1" in fn_body, "render_marks_radar must use Cyan/Teal (#00CED1) accent"

    def test_ui_marks_pillar_labels_present(self, ui_source):
        """The 4 pillar descriptive labels must be present in the widget."""
        match = re.search(r'def render_marks_radar.*?(?=\ndef |\Z)', ui_source, re.DOTALL)
        assert match
        fn_body = match.group(0)
        for label in ["Margin Extreme", "Price vs Value", "Leverage Cushion", "Defensive Cushion"]:
            assert label in fn_body, f"render_marks_radar missing pillar label: {label}"


# ═══════════════════════════════════════════════════════════════════════════════
# TestMarksRawSignalsContract
# ═══════════════════════════════════════════════════════════════════════════════

class TestMarksRawSignalsContract:
    """Verify Marks cells are surfaced in the tearsheet UI (render_marks_radar in the Frameworks
    tab — the All Data pillar grid was removed as a duplicate; whole-source scan stays robust)."""

    def test_raw_signals_has_marks_score_cell(self, ui_source):
        assert "marks_score" in ui_source

    def test_raw_signals_has_marks_pass_cell(self, ui_source):
        assert "marks_pass" in ui_source

    def test_raw_signals_section_header(self, ui_source):
        assert "Marks Cycle Shield" in ui_source or "Marks Cycle" in ui_source
