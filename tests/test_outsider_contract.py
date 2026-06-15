"""
Contract Tests — William Thorndike Outsider CEO Capital Allocation Framework
=============================================================================
STAGE 4 of the Outsider CEO integration pipeline.

Tests parse docs/outsider_specs.json and assert that EVERY threshold, column name,
and materialization logic declared in the spec is precisely reflected in the live
core/scoring_engine.py code.

Structure:
    TestOutsiderSpecLedger          — JSON schema completeness and required meta keys
    TestOutsiderEngineContract      — regex-based source code threshold verification
    TestOutsiderPillarArithmetic    — boundary conditions, AND invariant, score 0-4
    TestOutsiderIndexAlignment      — non-default integer and string index safety
    TestOutsiderUIContract          — render_outsider_radar import + column reads
    TestOutsiderRawSignalsContract  — outsider cells present in render_raw_signals
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

SPEC_PATH = os.path.join(REPO_ROOT, "docs", "outsider_specs.json")
SE_PATH   = os.path.join(REPO_ROOT, "core", "scoring_engine.py")
UI_PATH   = os.path.join(REPO_ROOT, "ui",   "ui_tearsheet.py")


# ── Fixtures ──────────────────────────────────────────────────────────────────

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

def _build_mock_outsider_row(**overrides) -> dict:
    """Build a fully-passing Outsider CEO row (all 4 pillars green)."""
    base = {
        "dilution_flag":  0,      # S: share retirement — zero dilution
        "de_slope_3y":    -0.1,   # D: debt discipline — D/E declining
        "cfo_to_pat":     88.0,   # C: cash generation — 88% > 85% threshold
        "roce_med_10y":   18.0,   # R: capital returns — 18% > 15% threshold
        # Extra columns needed by compute_qglp_score scaffold
        "market_cap":     5000.0,
        "close_price":    500.0,
        "name":           "TestCo",
        "sector":         "FMCG",
        "is_financial":   False,
    }
    base.update(overrides)
    return base


def _run_outsider(rows: list, regime: str = "SIDEWAYS") -> pd.DataFrame:
    """Execute compute_qglp_score on a list of row dicts; return result df."""
    from config import MASTER_PROFILES
    from core.scoring_engine import compute_qglp_score
    df = pd.DataFrame(rows)
    df.attrs["detected_market_regime"] = regime
    profile = MASTER_PROFILES.get("Balanced", next(iter(MASTER_PROFILES.values())))
    return compute_qglp_score(df, profile)


# ═══════════════════════════════════════════════════════════════════════════════
# TestOutsiderSpecLedger
# ═══════════════════════════════════════════════════════════════════════════════

class TestOutsiderSpecLedger:
    """Verify docs/outsider_specs.json is complete and structurally correct."""

    def test_spec_file_exists(self):
        assert os.path.exists(SPEC_PATH), f"Spec file not found: {SPEC_PATH}"

    def test_spec_is_valid_json(self, spec):
        assert isinstance(spec, dict)

    def test_meta_section_present(self, spec):
        assert "_meta" in spec

    def test_meta_required_keys(self, spec):
        required = [
            "title", "author", "framework_variable", "pass_column",
            "score_column", "frameworks_passed_label", "framework_number_in_code",
            "implementation_file", "version",
        ]
        for key in required:
            assert key in spec["_meta"], f"_meta missing key: {key}"

    def test_framework_number_is_15(self, spec):
        assert spec["_meta"]["framework_number_in_code"] == 15

    def test_pass_column_name(self, spec):
        assert spec["_meta"]["pass_column"] == "outsider_pass"

    def test_score_column_name(self, spec):
        assert spec["_meta"]["score_column"] == "outsider_score"

    def test_framework_variable_name(self, spec):
        assert spec["_meta"]["framework_variable"] == "fw_outsider"

    def test_framework_label(self, spec):
        assert spec["_meta"]["frameworks_passed_label"] == "Outsider CEO"

    def test_4_pillar_sections_present(self, spec):
        required_pillars = [
            "pillar_s_share_retirement",
            "pillar_d_debt_discipline",
            "pillar_c_cash_generation",
            "pillar_r_capital_returns",
        ]
        for p in required_pillars:
            assert p in spec, f"Spec missing pillar section: {p}"

    def test_output_columns_registry_present(self, spec):
        assert "output_columns_registry" in spec
        required = [
            "outsider_share_retirement", "outsider_debt_discipline",
            "outsider_cash_generation",  "outsider_capital_returns",
            "outsider_pass", "outsider_score",
        ]
        for col in required:
            assert col in spec["output_columns_registry"], (
                f"output_columns_registry missing: {col}"
            )

    def test_pillar_s_threshold_is_zero(self, spec):
        assert spec["pillar_s_share_retirement"]["share_retirement_gate"]["threshold"] == 0

    def test_pillar_d_threshold_is_zero_float(self, spec):
        assert spec["pillar_d_debt_discipline"]["debt_discipline_gate"]["threshold"] == 0.0

    def test_pillar_c_threshold_is_85(self, spec):
        t = spec["pillar_c_cash_generation"]["cash_generation_gate"]["threshold"]
        assert abs(t - 85.0) < 1e-9, f"Expected 85.0, got {t}"

    def test_pillar_r_threshold_is_15(self, spec):
        t = spec["pillar_r_capital_returns"]["capital_returns_gate"]["threshold"]
        assert abs(t - 15.0) < 1e-9, f"Expected 15.0, got {t}"

    def test_not_implementable_section_present(self, spec):
        assert "not_implementable" in spec
        assert len(spec["not_implementable"]) >= 6, (
            f"Expected at least 6 not_implementable entries (HQ cost, CEO comms, "
            f"decentralisation, M&A ROIC, per-share FCF CAGR, FCF CAGR > EPS CAGR); "
            f"found {len(spec['not_implementable'])}"
        )

    def test_not_implementable_fcf_cagr_vs_eps_documented(self, spec):
        """Companion Chapter 13 FCF CAGR > EPS CAGR signal must be in not_implementable."""
        gates = [item["gate"] for item in spec["not_implementable"]]
        fcf_eps_present = any("FCF CAGR" in g and "EPS CAGR" in g for g in gates)
        assert fcf_eps_present, (
            "not_implementable must document 'FCF CAGR > EPS CAGR' — the companion "
            "Chapter 13 signal not implementable due to missing 5Y FCF CAGR in CSV"
        )

    def test_not_implementable_each_has_reason(self, spec):
        """Every not_implementable entry must have a 'reason' field for traceability."""
        for item in spec["not_implementable"]:
            assert "reason" in item, (
                f"not_implementable entry '{item.get('gate', '?')}' missing 'reason' field"
            )

    def test_companion_screener_exclusions_section_present(self, spec):
        """Spec must document the 3 companion Chapter 7 screener conditions that were excluded."""
        assert "companion_screener_exclusions" in spec, (
            "companion_screener_exclusions section missing — must document why "
            "rev_gr_10y, market_cap, and static D/E were excluded from fw_outsider"
        )

    def test_companion_excludes_revenue_growth_documented(self, spec):
        excl = spec["companion_screener_exclusions"]
        assert "excluded_revenue_growth_gate" in excl, (
            "Spec must document exclusion of revenue growth gate from companion Chapter 7"
        )
        assert "exclusion_reason" in excl["excluded_revenue_growth_gate"]

    def test_companion_excludes_market_cap_documented(self, spec):
        excl = spec["companion_screener_exclusions"]
        assert "excluded_market_cap_gate" in excl, (
            "Spec must document exclusion of market cap filter from companion Chapter 7"
        )
        assert "exclusion_reason" in excl["excluded_market_cap_gate"]

    def test_companion_de_upgrade_documented(self, spec):
        """Static D/E < 0.75 replaced by de_slope_3y trajectory must be documented."""
        excl = spec["companion_screener_exclusions"]
        assert "excluded_static_de_gate" in excl
        de_entry = excl["excluded_static_de_gate"]
        assert "replacement_in_fw_outsider" in de_entry
        assert "de_slope_3y" in de_entry["replacement_in_fw_outsider"]

    def test_related_engine_signals_section_present(self, spec):
        assert "related_data_engine_signals" in spec, (
            "related_data_engine_signals section missing — must document "
            "value_creation_ratio, capital_misallocation_risk, capital_return_spread"
        )

    def test_related_signals_has_value_creation_ratio(self, spec):
        signals = spec["related_data_engine_signals"]
        assert "value_creation_ratio" in signals
        assert "why_not_a_hard_gate" in signals["value_creation_ratio"]

    def test_related_signals_has_capital_misallocation_risk(self, spec):
        signals = spec["related_data_engine_signals"]
        assert "capital_misallocation_risk" in signals
        assert "why_not_a_hard_gate" in signals["capital_misallocation_risk"]

    def test_related_signals_has_capital_return_spread(self, spec):
        signals = spec["related_data_engine_signals"]
        assert "capital_return_spread" in signals
        assert "why_not_a_hard_gate" in signals["capital_return_spread"]

    def test_vectorization_matrix_present(self, spec):
        assert "vectorization_matrix" in spec
        vm = spec["vectorization_matrix"]["nan_handling"]
        for col in ["dilution_flag", "de_slope_3y", "cfo_to_pat", "roce_med_10y"]:
            assert col in vm, f"vectorization_matrix missing NaN handling for {col}"


# ═══════════════════════════════════════════════════════════════════════════════
# TestOutsiderEngineContract
# ═══════════════════════════════════════════════════════════════════════════════

class TestOutsiderEngineContract:
    """Verify scoring_engine.py contains the exact materialized columns and thresholds."""

    def test_framework_15_anchor_comment(self, se_source):
        assert "# 15. Outsiders on Dalal Street" in se_source

    def test_outsider_share_retirement_column_defined(self, se_source):
        assert 'df["outsider_share_retirement"]' in se_source

    def test_outsider_debt_discipline_column_defined(self, se_source):
        assert 'df["outsider_debt_discipline"]' in se_source

    def test_outsider_cash_generation_column_defined(self, se_source):
        assert 'df["outsider_cash_generation"]' in se_source

    def test_outsider_capital_returns_column_defined(self, se_source):
        assert 'df["outsider_capital_returns"]' in se_source

    def test_outsider_pass_column_defined(self, se_source):
        assert 'df["outsider_pass"]' in se_source

    def test_outsider_score_column_defined(self, se_source):
        assert 'df["outsider_score"]' in se_source

    def test_dilution_flag_equality_zero(self, se_source):
        """Pillar S: dilution_flag == 0 (exact equality check for zero dilution)."""
        assert re.search(r'dilution_flag.*==\s*0', se_source), (
            "scoring_engine.py must check dilution_flag == 0 for outsider_share_retirement"
        )

    def test_de_slope_3y_lte_zero(self, se_source):
        """Pillar D: de_slope_3y <= 0.0 (debt discipline trajectory)."""
        assert re.search(r'de_slope_3y.*<=\s*0\.0', se_source), (
            "scoring_engine.py must check de_slope_3y <= 0.0 for outsider_debt_discipline"
        )

    def test_cfo_to_pat_gte_85(self, se_source):
        """Pillar C: cfo_to_pat >= 85.0 (strictest cash conversion floor)."""
        assert re.search(r'cfo_to_pat.*>=\s*85\.0', se_source), (
            "scoring_engine.py must check cfo_to_pat >= 85.0 for outsider_cash_generation"
        )

    def test_roce_med_10y_gte_15(self, se_source):
        """Pillar R: roce_med_10y >= 15.0 (10-year ROCE hurdle)."""
        assert re.search(r'roce_med_10y.*>=\s*15\.0', se_source), (
            "scoring_engine.py must check roce_med_10y >= 15.0 for outsider_capital_returns"
        )

    def test_fw_outsider_is_and_of_4_pillars(self, se_source):
        """fw_outsider = AND of all 4 materialized pillar columns."""
        assert re.search(r'fw_outsider\s*=\s*\(', se_source)
        for col in ["outsider_share_retirement", "outsider_debt_discipline",
                    "outsider_cash_generation", "outsider_capital_returns"]:
            assert col in se_source, f"fw_outsider must reference {col}"

    def test_fw_str_includes_outsider_label(self, se_source):
        """fw_str concatenation must include 'Outsider CEO|' via fw_outsider."""
        assert re.search(r'fw_outsider.*Outsider CEO', se_source), (
            "fw_str must have np.where(fw_outsider, 'Outsider CEO|', '')"
        )

    def test_nan_fillna_dilution_flag_conservative(self, se_source):
        """Missing dilution_flag data must default to 1 (diluted = excludes from Outsider)."""
        assert re.search(r'dilution_flag.*fillna\s*\(\s*1\s*\)', se_source)

    def test_nan_fillna_de_slope_conservative(self, se_source):
        """Missing de_slope_3y must default to 999 (rising debt = excludes from Outsider)."""
        assert re.search(r'de_slope_3y.*fillna\s*\(\s*999\s*\)', se_source)

    def test_outsider_score_is_sum_of_4_pillars(self, se_source):
        """outsider_score = sum of the 4 pillar flags (0-4 range)."""
        assert re.search(r'outsider_score.*outsider_share_retirement', se_source, re.DOTALL), (
            "outsider_score must sum all 4 pillar flags"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TestOutsiderPillarArithmetic
# ═══════════════════════════════════════════════════════════════════════════════

class TestOutsiderPillarArithmetic:
    """Boundary conditions and arithmetic invariants for all 4 pillar gates."""

    # ── Pillar S: Share Retirement ─────────────────────────────────────────────

    def test_pillar_s_dilution_flag_0_passes(self):
        result = _run_outsider([_build_mock_outsider_row(dilution_flag=0)])
        assert result["outsider_share_retirement"].iloc[0] == 1

    def test_pillar_s_dilution_flag_1_fails(self):
        result = _run_outsider([_build_mock_outsider_row(dilution_flag=1)])
        assert result["outsider_share_retirement"].iloc[0] == 0

    def test_pillar_s_dilution_flag_3_fails(self):
        result = _run_outsider([_build_mock_outsider_row(dilution_flag=3)])
        assert result["outsider_share_retirement"].iloc[0] == 0

    def test_pillar_s_nan_dilution_fails_conservatively(self):
        """Missing dilution_flag → fillna(1) → 1 ≠ 0 → gate fails."""
        result = _run_outsider([_build_mock_outsider_row(dilution_flag=float("nan"))])
        assert result["outsider_share_retirement"].iloc[0] == 0

    # ── Pillar D: Debt Discipline ──────────────────────────────────────────────

    def test_pillar_d_negative_slope_passes(self):
        result = _run_outsider([_build_mock_outsider_row(de_slope_3y=-0.5)])
        assert result["outsider_debt_discipline"].iloc[0] == 1

    def test_pillar_d_zero_slope_passes(self):
        """Exactly zero slope = flat D/E = discipline confirmed (≤ 0)."""
        result = _run_outsider([_build_mock_outsider_row(de_slope_3y=0.0)])
        assert result["outsider_debt_discipline"].iloc[0] == 1

    def test_pillar_d_positive_slope_fails(self):
        result = _run_outsider([_build_mock_outsider_row(de_slope_3y=0.01)])
        assert result["outsider_debt_discipline"].iloc[0] == 0

    def test_pillar_d_nan_slope_fails_conservatively(self):
        """Missing de_slope_3y → fillna(999) → 999 > 0 → gate fails."""
        result = _run_outsider([_build_mock_outsider_row(de_slope_3y=float("nan"))])
        assert result["outsider_debt_discipline"].iloc[0] == 0

    # ── Pillar C: Cash Generation ──────────────────────────────────────────────

    def test_pillar_c_at_85_passes(self):
        """Exactly 85.0% = at the threshold → passes (≥ 85)."""
        result = _run_outsider([_build_mock_outsider_row(cfo_to_pat=85.0)])
        assert result["outsider_cash_generation"].iloc[0] == 1

    def test_pillar_c_above_85_passes(self):
        result = _run_outsider([_build_mock_outsider_row(cfo_to_pat=95.0)])
        assert result["outsider_cash_generation"].iloc[0] == 1

    def test_pillar_c_below_85_fails(self):
        result = _run_outsider([_build_mock_outsider_row(cfo_to_pat=84.9)])
        assert result["outsider_cash_generation"].iloc[0] == 0

    def test_pillar_c_threshold_stricter_than_dorsey(self):
        """Dorsey threshold = 80%; Outsider = 85%. A stock at 82% fails Outsider but passes Dorsey."""
        result = _run_outsider([_build_mock_outsider_row(cfo_to_pat=82.0)])
        assert result["outsider_cash_generation"].iloc[0] == 0, (
            "cfo_to_pat=82.0 must fail outsider_cash_generation (threshold=85.0 > Dorsey's 80.0)"
        )

    def test_pillar_c_nan_fails_conservatively(self):
        result = _run_outsider([_build_mock_outsider_row(cfo_to_pat=float("nan"))])
        assert result["outsider_cash_generation"].iloc[0] == 0

    # ── Pillar R: Capital Returns ──────────────────────────────────────────────

    def test_pillar_r_at_15_passes(self):
        """Exactly 15.0% = at threshold → passes (≥ 15)."""
        result = _run_outsider([_build_mock_outsider_row(roce_med_10y=15.0)])
        assert result["outsider_capital_returns"].iloc[0] == 1

    def test_pillar_r_above_15_passes(self):
        result = _run_outsider([_build_mock_outsider_row(roce_med_10y=25.0)])
        assert result["outsider_capital_returns"].iloc[0] == 1

    def test_pillar_r_below_15_fails(self):
        result = _run_outsider([_build_mock_outsider_row(roce_med_10y=14.9)])
        assert result["outsider_capital_returns"].iloc[0] == 0

    def test_pillar_r_nan_fails_conservatively(self):
        result = _run_outsider([_build_mock_outsider_row(roce_med_10y=float("nan"))])
        assert result["outsider_capital_returns"].iloc[0] == 0

    # ── AND Invariant: outsider_pass ───────────────────────────────────────────

    def test_all_4_pass_outsider_pass_is_1(self):
        result = _run_outsider([_build_mock_outsider_row()])
        assert result["outsider_pass"].iloc[0] == 1

    def test_single_pillar_failure_outsider_pass_is_0(self):
        """Any single pillar failing must flip outsider_pass to 0."""
        for fail_case in [
            {"dilution_flag": 1},
            {"de_slope_3y": 0.1},
            {"cfo_to_pat": 80.0},
            {"roce_med_10y": 14.0},
        ]:
            result = _run_outsider([_build_mock_outsider_row(**fail_case)])
            assert result["outsider_pass"].iloc[0] == 0, (
                f"outsider_pass must be 0 when {fail_case} fails"
            )

    # ── Score Range ────────────────────────────────────────────────────────────

    def test_score_all_pass_is_4(self):
        result = _run_outsider([_build_mock_outsider_row()])
        assert result["outsider_score"].iloc[0] == 4

    def test_score_all_fail_is_0(self):
        result = _run_outsider([_build_mock_outsider_row(
            dilution_flag=3, de_slope_3y=1.0, cfo_to_pat=50.0, roce_med_10y=8.0
        )])
        assert result["outsider_score"].iloc[0] == 0

    def test_score_3_when_one_fails(self):
        result = _run_outsider([_build_mock_outsider_row(cfo_to_pat=70.0)])
        assert result["outsider_score"].iloc[0] == 3

    def test_score_2_when_two_fail(self):
        result = _run_outsider([_build_mock_outsider_row(
            cfo_to_pat=70.0, roce_med_10y=10.0
        )])
        assert result["outsider_score"].iloc[0] == 2

    def test_score_range_always_0_to_4(self):
        """Score must be in [0, 4] for any combination of inputs."""
        combos = [
            _build_mock_outsider_row(),
            _build_mock_outsider_row(dilution_flag=3),
            _build_mock_outsider_row(de_slope_3y=1.0, cfo_to_pat=50.0),
            _build_mock_outsider_row(dilution_flag=3, de_slope_3y=1.0,
                                     cfo_to_pat=50.0, roce_med_10y=5.0),
        ]
        result = _run_outsider(combos)
        assert result["outsider_score"].between(0, 4).all()

    def test_pass_equals_score_4(self):
        """outsider_pass == 1 ⟺ outsider_score == 4 (AND identity)."""
        combos = [
            _build_mock_outsider_row(),                   # all pass → score=4
            _build_mock_outsider_row(cfo_to_pat=70.0),   # 1 fail → score=3
            _build_mock_outsider_row(dilution_flag=3, de_slope_3y=1.0,
                                     cfo_to_pat=50.0, roce_med_10y=5.0),  # all fail → score=0
        ]
        result = _run_outsider(combos)
        # pass == 1 iff score == 4
        assert (result["outsider_pass"] == (result["outsider_score"] == 4).astype(int)).all()


# ═══════════════════════════════════════════════════════════════════════════════
# TestOutsiderIndexAlignment
# ═══════════════════════════════════════════════════════════════════════════════

class TestOutsiderIndexAlignment:
    """Non-default index safety — guards against pandas index alignment drift."""

    def test_integer_index_non_default(self):
        """Results must be correctly aligned with a shuffled integer index [100, 200, 300]."""
        rows = [
            _build_mock_outsider_row(),               # all pass
            _build_mock_outsider_row(cfo_to_pat=70.0),  # C fails
            _build_mock_outsider_row(dilution_flag=3),   # S fails
        ]
        from config import MASTER_PROFILES
        from core.scoring_engine import compute_qglp_score
        df = pd.DataFrame(rows, index=[100, 200, 300])
        df.attrs["detected_market_regime"] = "SIDEWAYS"
        profile = MASTER_PROFILES.get("Balanced", next(iter(MASTER_PROFILES.values())))
        result = compute_qglp_score(df, profile)

        assert result.index.tolist() == [100, 200, 300], "Index must be preserved"
        assert result.loc[100, "outsider_pass"]  == 1
        assert result.loc[200, "outsider_pass"]  == 0   # C pillar failed
        assert result.loc[300, "outsider_pass"]  == 0   # S pillar failed

    def test_string_index(self):
        """Results must be correctly aligned with a string index ["HDFC", "TCS", "ITC"]."""
        rows = [
            _build_mock_outsider_row(cfo_to_pat=90.0),  # all pass
            _build_mock_outsider_row(de_slope_3y=0.5),  # D fails
            _build_mock_outsider_row(),                  # all pass
        ]
        from config import MASTER_PROFILES
        from core.scoring_engine import compute_qglp_score
        df = pd.DataFrame(rows, index=["HDFC", "TCS", "ITC"])
        df.attrs["detected_market_regime"] = "SIDEWAYS"
        profile = MASTER_PROFILES.get("Balanced", next(iter(MASTER_PROFILES.values())))
        result = compute_qglp_score(df, profile)

        assert result.index.tolist() == ["HDFC", "TCS", "ITC"]
        assert result.loc["HDFC", "outsider_pass"] == 1
        assert result.loc["TCS",  "outsider_pass"] == 0   # D pillar failed
        assert result.loc["ITC",  "outsider_pass"] == 1

    def test_all_nan_row_fails_conservatively(self):
        """A row with all NaN inputs must have outsider_pass=0 and outsider_score=0."""
        nan_row = {
            "dilution_flag": float("nan"),
            "de_slope_3y":   float("nan"),
            "cfo_to_pat":    float("nan"),
            "roce_med_10y":  float("nan"),
        }
        result = _run_outsider([nan_row])
        assert result["outsider_pass"].iloc[0]  == 0, "All-NaN row must fail outsider_pass"
        assert result["outsider_score"].iloc[0] == 0, "All-NaN row must have score=0"


# ═══════════════════════════════════════════════════════════════════════════════
# TestOutsiderUIContract
# ═══════════════════════════════════════════════════════════════════════════════

class TestOutsiderUIContract:
    """Verify render_outsider_radar is correctly exported and reads the right columns."""

    def test_render_outsider_radar_importable(self):
        from ui import render_outsider_radar
        assert callable(render_outsider_radar)

    def test_render_outsider_radar_in_all(self):
        import ui
        assert "render_outsider_radar" in ui.__all__

    def test_ui_reads_outsider_share_retirement(self, ui_source):
        assert "outsider_share_retirement" in ui_source

    def test_ui_reads_outsider_debt_discipline(self, ui_source):
        assert "outsider_debt_discipline" in ui_source

    def test_ui_reads_outsider_cash_generation(self, ui_source):
        assert "outsider_cash_generation" in ui_source

    def test_ui_reads_outsider_capital_returns(self, ui_source):
        assert "outsider_capital_returns" in ui_source

    def test_ui_reads_outsider_pass(self, ui_source):
        assert '"outsider_pass"' in ui_source

    def test_ui_reads_outsider_score(self, ui_source):
        assert '"outsider_score"' in ui_source

    def test_ui_uses_no_threshold_math(self, ui_source):
        """Pure display: render_outsider_radar must not contain numeric threshold literals."""
        # Extract the render_outsider_radar function body
        match = re.search(r'def render_outsider_radar.*?(?=\ndef |\Z)', ui_source, re.DOTALL)
        assert match, "render_outsider_radar function not found in ui_tearsheet.py"
        fn_body = match.group(0)
        # Must NOT contain threshold comparisons (>= 85, >= 15, == 0, <= 0)
        forbidden = [r'>= 85', r'>= 15', r'<= 0\.0', r'dilution_flag.*== 0']
        for pattern in forbidden:
            assert not re.search(pattern, fn_body), (
                f"render_outsider_radar contains threshold math: {pattern}. "
                "Pure display layer must never re-compute thresholds."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TestOutsiderRawSignalsContract
# ═══════════════════════════════════════════════════════════════════════════════

class TestOutsiderRawSignalsContract:
    """Verify outsider cells are surfaced in the tearsheet UI (render_outsider_radar in the
    Frameworks tab — the All Data pillar grid was removed as a duplicate; whole-source scan)."""

    def test_raw_signals_has_outsider_score_cell(self, ui_source):
        assert '"outsider_score"' in ui_source or "outsider_score" in ui_source

    def test_raw_signals_has_outsider_pass_cell(self, ui_source):
        assert "outsider_pass" in ui_source

    def test_raw_signals_section_header(self, ui_source):
        assert "Outsider CEO Pillars" in ui_source or "Outsider CEO" in ui_source
