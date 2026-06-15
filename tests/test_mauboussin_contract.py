"""
Contract Tests — Mauboussin Expectations Investing Framework (Framework 34)
============================================================================
Automated verification that docs/mauboussin_expectations_specs.json,
core/scoring_engine.py, ui/ui_tearsheet.py, and app.py are in perfect alignment.

Framework architecture: 3-layer implied-CAP inversion (v1.1-mauboussin-nopat-precision)
  Layer 1 — Implied CAP proxy: pe * (nopat_margin/100) * reinvestment_rate
  Layer 2 — CAP trap flag:     implied_cap > 15.0 AND (roce - roce_med_3y)/2 < -1.0
  Layer 3 — Interactive Reverse DCF calculator (UI only, single-stock)

Key unit conventions (v1.1):
  • pe                  — FLOAT (e.g. 35.0 = 35×)
  • ebit, revenue, pbt, pat — raw financials for NOPAT margin computation
  • mauboussin_nopat_margin — PERCENTAGE output (e.g. 20.0 = 20%); capital-structure-neutral
  • reinvestment_rate   — DECIMAL [0,1] (e.g. 0.50 = 50% earnings retained)
  • implied_cap         — dimensionless proxy (pe × nopat_margin/100 × reinvestment_rate)
  • sell_alert_treadmill — binary int (0 = safe, 1 = alert firing)
  • operating_leverage   — binary int (1 = functioning, 0 = broken)
  • roce, roce_med_3y   — FLOAT; structural slope = (roce - roce_med_3y) / 2.0
  • Score range: 0–3 (bidirectional: score==3 ↔ pass==1, no asymmetric veto)

Structure:
    TestMauboussinSpecLedger       — JSON schema completeness and meta keys
    TestMauboussinEngineContract   — regex source-code verification
    TestMauboussinPillarArithmetic — boundary conditions, AND invariant, score 0-3
    TestMauboussinNaNConservative  — NaN handling (conservative gate failure)
    TestMauboussinIndexAlignment   — non-default integer and string index safety
    TestMauboussinUIContract       — render_mauboussin_radar import + pure-display contract
    TestMauboussinRawSignalsContract — Mauboussin cells in render_raw_signals
    TestMauboussinAppWiring        — app.py import + call + ui/__init__.py export
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

SPEC_PATH   = os.path.join(REPO_ROOT, "docs", "mauboussin_expectations_specs.json")
SE_PATH     = os.path.join(REPO_ROOT, "core", "scoring_engine.py")
UI_PATH     = os.path.join(REPO_ROOT, "ui",   "ui_tearsheet.py")
APP_PATH    = os.path.join(REPO_ROOT, "app.py")
INIT_PATH   = os.path.join(REPO_ROOT, "ui",   "__init__.py")


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


@pytest.fixture(scope="module")
def app_source() -> str:
    with open(APP_PATH, encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module")
def init_source() -> str:
    with open(INIT_PATH, encoding="utf-8") as f:
        return f.read()


# ── Mock data helpers ─────────────────────────────────────────────────────────

def _build_mock_maub_row(**overrides) -> dict:
    """Build a fully-passing Mauboussin row (all 3 gates green) — v1.1 decimal units.

    Default values satisfy every gate:
      T — sell_alert_treadmill = 0   (treadmill not firing → breach = 1)
      O — operating_leverage   = 1   (efficient conversion → oplev = 1)
      C — implied_cap = 20 * 0.20 * 0.50 = 2.0  (< 15 threshold → cap_trap = 0)
            nopat_margin computed as: ebit=250, pbt=250, pat=200, revenue=1000
            → eff_tax=(250-200)/250=0.20, nopat=250*0.80=200, margin=200/1000=20%
            reinvestment_rate = 0.50 (decimal [0,1] — NOT percent)
            roce=25, roce_med_3y=25 → structural slope=0 → no ROCE decay
    """
    base = {
        "pe":                    20.0,   # → implied_cap = 20 * 0.20 * 0.50 = 2.0
        # NOPAT margin inputs (v1.1: replaces npm)
        "ebit":                 250.0,   # NOPAT margin = ebit*(1-eff_tax)/revenue = 20%
        "pbt":                  250.0,   # eff_tax = (pbt-pat)/pbt = 0.20
        "pat":                  200.0,
        "revenue":             1000.0,
        "reinvestment_rate":      0.50,  # decimal [0,1] — no /100 applied in v1.1
        "sell_alert_treadmill":    0,    # T: not firing → treadmill_breach = 1
        "operating_leverage":      1,    # O: functioning → oplev_drift = 1
        # CAP trap slope inputs (v1.1: replaces d35_roce_trend)
        "roce":                  25.0,   # structural slope = (25-25)/2 = 0 → no decay
        "roce_med_3y":           25.0,
        # scaffold
        "market_cap":          2500.0,
        "close_price":          250.0,
        "name":                "TestExpectations",
        "sector":              "Consumer",
    }
    base.update(overrides)
    return base


def _run_maub(rows: list, regime: str = "SIDEWAYS") -> pd.DataFrame:
    """Execute compute_qglp_score on a list of row dicts; return result df."""
    from config import MASTER_PROFILES
    from core.scoring_engine import compute_qglp_score
    df = pd.DataFrame(rows)
    df.attrs["detected_market_regime"] = regime
    profile = MASTER_PROFILES.get("Balanced", next(iter(MASTER_PROFILES.values())))
    return compute_qglp_score(df, profile)


# ═══════════════════════════════════════════════════════════════════════════════
# TestMauboussinSpecLedger
# ═══════════════════════════════════════════════════════════════════════════════

class TestMauboussinSpecLedger:
    """Verify docs/mauboussin_expectations_specs.json is complete and correct."""

    def test_spec_file_exists(self):
        assert os.path.exists(SPEC_PATH), f"Spec not found: {SPEC_PATH}"

    def test_spec_is_valid_json(self, spec):
        assert isinstance(spec, dict)

    def test_meta_section_present(self, spec):
        assert "_meta" in spec

    def test_meta_required_keys(self, spec):
        required = [
            "title", "framework_variable", "pass_column", "score_column",
            "frameworks_passed_label", "implementation_file", "version",
            "comment_anchor", "framework_number_in_code",
        ]
        for key in required:
            assert key in spec["_meta"], f"_meta missing: {key}"

    def test_pass_column_name(self, spec):
        assert spec["_meta"]["pass_column"] == "mauboussin_pass"

    def test_score_column_name(self, spec):
        assert spec["_meta"]["score_column"] == "mauboussin_score"

    def test_framework_variable_name(self, spec):
        assert spec["_meta"]["framework_variable"] == "fw_mauboussin"

    def test_fw_str_label(self, spec):
        assert spec["_meta"]["frameworks_passed_label"] == "Expectations Matrix"

    def test_framework_number_is_34(self, spec):
        assert spec["_meta"]["framework_number_in_code"] == 34, (
            f"Mauboussin is Framework 34. Got: {spec['_meta']['framework_number_in_code']}"
        )

    def test_version_references_mauboussin(self, spec):
        v = spec["_meta"]["version"].lower()
        assert "mauboussin" in v or "nopat" in v, (
            f"Version must reference mauboussin or nopat precision; got: {spec['_meta']['version']}"
        )

    def test_comment_anchor_references_mauboussin(self, spec):
        anchor = spec["_meta"]["comment_anchor"]
        assert "Mauboussin" in anchor or "mauboussin" in anchor, (
            f"comment_anchor must reference Mauboussin; got: {anchor}"
        )

    def test_three_layer_design_present(self, spec):
        assert "three_layer_design" in spec

    def test_layer_1_output_column(self, spec):
        assert spec["three_layer_design"]["layer_1"]["output_column"] == "mauboussin_implied_cap"

    def test_layer_2_output_column(self, spec):
        assert spec["three_layer_design"]["layer_2"]["output_column"] == "mauboussin_cap_trap"

    def test_implied_cap_threshold_is_15(self, spec):
        t = spec["three_layer_design"]["layer_1"]["threshold_for_trap"]
        assert abs(t - 15.0) < 1e-9, f"Implied CAP threshold must be 15.0; got: {t}"

    def test_pillar_t_section_present(self, spec):
        assert "pillar_t_treadmill" in spec

    def test_pillar_o_section_present(self, spec):
        assert "pillar_o_operating_leverage" in spec

    def test_pillar_c_section_present(self, spec):
        assert "pillar_c_cap_trap" in spec

    def test_pillar_t_column_materialized(self, spec):
        assert spec["pillar_t_treadmill"]["_column_materialized"] == "mauboussin_treadmill_breach"

    def test_pillar_o_column_materialized(self, spec):
        assert spec["pillar_o_operating_leverage"]["_column_materialized"] == "mauboussin_oplev_drift"

    def test_pillar_c_column_materialized(self, spec):
        assert spec["pillar_c_cap_trap"]["_column_materialized"] == "mauboussin_cap_trap"

    def test_output_registry_present(self, spec):
        assert "output_registry" in spec

    def test_output_registry_all_6_columns(self, spec):
        expected = [
            "mauboussin_implied_cap", "mauboussin_treadmill_breach",
            "mauboussin_oplev_drift", "mauboussin_cap_trap",
            "mauboussin_pass", "mauboussin_score",
        ]
        for col in expected:
            assert col in spec["output_registry"], f"output_registry missing: {col}"

    def test_score_range_in_meta(self, spec):
        assert spec["_meta"]["score_range"] == "0-3"

    def test_bidirectional_flag_true(self, spec):
        assert spec["score_column_logic"]["score_3_implies_pass_1"] is True
        assert spec["score_column_logic"]["pass_1_implies_score_3"] is True

    def test_nan_handling_section_present(self, spec):
        assert "nan_handling" in spec

    def test_nan_handling_all_inputs_covered(self, spec):
        for col in ["pe", "reinvestment_rate",
                    "sell_alert_treadmill", "operating_leverage"]:
            assert col in spec["nan_handling"], f"nan_handling missing: {col}"
        # v1.1: nopat_margin and roce_med_3y replace npm and d35_roce_trend
        assert "mauboussin_nopat_margin" in spec["nan_handling"] or \
               "roce_med_3y" in spec["nan_handling"], (
                   "nan_handling must cover v1.1 inputs (nopat_margin or roce_med_3y)"
               )

    def test_vectorization_matrix_present(self, spec):
        assert "vectorization_matrix" in spec
        assert "approach" in spec["vectorization_matrix"]


# ═══════════════════════════════════════════════════════════════════════════════
# TestMauboussinEngineContract
# ═══════════════════════════════════════════════════════════════════════════════

class TestMauboussinEngineContract:
    """Verify core/scoring_engine.py source code matches the spec exactly."""

    def test_comment_anchor_present(self, se_source):
        assert "# 34. Mauboussin Expectations Investing Framework" in se_source

    def test_spec_reference_comment_present(self, se_source):
        assert "mauboussin_expectations_specs.json" in se_source

    def test_implied_cap_column_assigned(self, se_source):
        assert 'df["mauboussin_implied_cap"]' in se_source

    def test_implied_cap_formula_pe_times_nopat_times_rr(self, se_source):
        assert re.search(
            r'df\["mauboussin_implied_cap"\]\s*=\s*_pe_ly\s*\*\s*_nopat_m_ly\s*\*\s*_rr_ly',
            se_source
        ), "implied_cap formula must be _pe_ly * _nopat_m_ly * _rr_ly (v1.1 NOPAT precision)"

    def test_nopat_margin_column_computed(self, se_source):
        assert re.search(
            r'df\["mauboussin_nopat_margin"\]',
            se_source
        ), "mauboussin_nopat_margin must be computed and materialized in scoring_engine"

    def test_nopat_m_ly_divided_by_100(self, se_source):
        assert re.search(
            r'_nopat_m_ly\s*=.*fillna\(0\.0\)\s*/\s*100\.0',
            se_source
        ), "_nopat_m_ly must divide by 100.0 to convert percent to decimal"

    def test_rr_not_divided_by_100(self, se_source):
        # v1.1 fix: reinvestment_rate is already decimal [0,1], no /100
        assert re.search(
            r'_rr_ly\s*=\s*df\.get\("reinvestment_rate".*\)\.fillna\(0\.0\)\s*\n',
            se_source
        ) or re.search(
            r'_rr_ly\s*=\s*df\.get\("reinvestment_rate"',
            se_source
        ), "_rr_ly must read reinvestment_rate without /100 division (already decimal)"

    def test_treadmill_breach_column_assigned(self, se_source):
        assert 'df["mauboussin_treadmill_breach"]' in se_source

    def test_treadmill_breach_checks_sell_alert(self, se_source):
        assert re.search(
            r'sell_alert_treadmill.*==\s*1.*0.*1|'
            r'mauboussin_treadmill_breach.*sell_alert_treadmill',
            se_source
        ), "treadmill_breach must reference sell_alert_treadmill"

    def test_oplev_drift_column_assigned(self, se_source):
        assert 'df["mauboussin_oplev_drift"]' in se_source

    def test_oplev_drift_checks_operating_leverage(self, se_source):
        assert "operating_leverage" in se_source
        assert "mauboussin_oplev_drift" in se_source

    def test_cap_trap_column_assigned(self, se_source):
        assert 'df["mauboussin_cap_trap"]' in se_source

    def test_cap_trap_threshold_15(self, se_source):
        assert re.search(
            r'mauboussin_implied_cap.*>\s*15\.0|15\.0.*mauboussin_implied_cap',
            se_source
        ), "cap_trap must use threshold 15.0"

    def test_cap_trap_uses_structural_roce_slope(self, se_source):
        assert "roce_med_3y" in se_source, (
            "v1.1: cap_trap must use structural 3-year ROCE slope (roce_med_3y)"
        )

    def test_cap_trap_slope_threshold_minus_1(self, se_source):
        assert re.search(
            r'_roce_slope_3y\s*<\s*-1\.0',
            se_source
        ), "cap_trap must use structural slope < -1.0 (v1.1 noise-resistant condition)"

    def test_mauboussin_pass_column_assigned(self, se_source):
        assert 'df["mauboussin_pass"]' in se_source

    def test_mauboussin_score_column_assigned(self, se_source):
        assert 'df["mauboussin_score"]' in se_source

    def test_fw_mauboussin_variable_defined(self, se_source):
        assert "fw_mauboussin" in se_source

    def test_fw_str_includes_expectations_matrix(self, se_source):
        assert "Expectations Matrix" in se_source

    def test_fw_str_uses_fw_mauboussin(self, se_source):
        assert re.search(
            r'np\.where\(fw_mauboussin.*Expectations Matrix',
            se_source
        ), "fw_str must include np.where(fw_mauboussin, 'Expectations Matrix|', '')"

    def test_pe_fillna_zero(self, se_source):
        assert re.search(
            r'_pe_ly\s*=.*fillna\(0\.0\)',
            se_source
        ), "_pe_ly must fillna(0.0)"

    def test_sell_alert_treadmill_fillna_zero(self, se_source):
        assert re.search(
            r'sell_alert_treadmill.*fillna\(0\)',
            se_source
        ), "sell_alert_treadmill must fillna(0)"

    def test_operating_leverage_fillna_one(self, se_source):
        assert re.search(
            r'operating_leverage.*fillna\(1\)',
            se_source
        ), "operating_leverage must fillna(1)"

    def test_roce_med3y_fillna_defaults_to_roce(self, se_source):
        # v1.1: roce_med_3y NaN → fillna chain (roce_2yb → roce) → slope=0 (no assumed decay)
        # Multi-line fallback: roce_med_3y and fillna may be on separate lines
        assert re.search(
            r'roce_med_3y.*fillna|fillna.*roce_med_3y',
            se_source, re.DOTALL
        ), "roce_med_3y must have fillna guard (v1.1 NaN-safe structural slope)"

    def test_score_uses_cap_trap_inverted(self, se_source):
        assert re.search(
            r'mauboussin_cap_trap.*==\s*0.*astype\(int\)',
            se_source
        ), "mauboussin_score must use (cap_trap == 0).astype(int)"


# ═══════════════════════════════════════════════════════════════════════════════
# TestMauboussinPillarArithmetic
# ═══════════════════════════════════════════════════════════════════════════════

class TestMauboussinPillarArithmetic:
    """Boundary conditions, AND invariant, score 0-3."""

    # ── Full pass ─────────────────────────────────────────────────────────────

    def test_all_gates_pass_gives_pass_1_score_3(self):
        res = _run_maub([_build_mock_maub_row()])
        assert res.loc[0, "mauboussin_pass"]  == 1
        assert res.loc[0, "mauboussin_score"] == 3

    def test_frameworks_passed_contains_expectations_matrix_when_pass(self):
        res = _run_maub([_build_mock_maub_row()])
        assert "Expectations Matrix" in res.loc[0, "frameworks_passed"]

    # ── Treadmill pillar ──────────────────────────────────────────────────────

    def test_treadmill_alert_firing_kills_pass(self):
        res = _run_maub([_build_mock_maub_row(sell_alert_treadmill=1)])
        assert res.loc[0, "mauboussin_treadmill_breach"] == 0
        assert res.loc[0, "mauboussin_pass"] == 0

    def test_treadmill_alert_firing_reduces_score(self):
        res = _run_maub([_build_mock_maub_row(sell_alert_treadmill=1)])
        assert res.loc[0, "mauboussin_score"] == 2

    def test_treadmill_safe_gives_breach_1(self):
        res = _run_maub([_build_mock_maub_row(sell_alert_treadmill=0)])
        assert res.loc[0, "mauboussin_treadmill_breach"] == 1

    # ── Operating leverage pillar ─────────────────────────────────────────────

    def test_oplev_broken_kills_pass(self):
        res = _run_maub([_build_mock_maub_row(operating_leverage=0)])
        assert res.loc[0, "mauboussin_oplev_drift"] == 0
        assert res.loc[0, "mauboussin_pass"] == 0

    def test_oplev_broken_reduces_score(self):
        res = _run_maub([_build_mock_maub_row(operating_leverage=0)])
        assert res.loc[0, "mauboussin_score"] == 2

    def test_oplev_functioning_gives_drift_1(self):
        res = _run_maub([_build_mock_maub_row(operating_leverage=1)])
        assert res.loc[0, "mauboussin_oplev_drift"] == 1

    # ── CAP trap pillar ───────────────────────────────────────────────────────

    def test_high_cap_plus_roce_decline_triggers_trap(self):
        # v1.1: pe=60, nopat_margin=30%, rr=1.0 → implied_cap = 18.0 > 15
        # slope = (roce - roce_med_3y) / 2 = (15-20)/2 = -2.5 < -1.0 → trap fires
        res = _run_maub([_build_mock_maub_row(
            pe=60.0, ebit=375.0, revenue=1000.0, pbt=375.0, pat=300.0,
            reinvestment_rate=1.0,
            roce=15.0, roce_med_3y=20.0,
        )])
        assert res.loc[0, "mauboussin_implied_cap"] > 15.0
        assert res.loc[0, "mauboussin_cap_trap"] == 1
        assert res.loc[0, "mauboussin_pass"] == 0

    def test_cap_trap_reduces_score_to_2(self):
        res = _run_maub([_build_mock_maub_row(
            pe=60.0, ebit=375.0, revenue=1000.0, pbt=375.0, pat=300.0,
            reinvestment_rate=1.0,
            roce=15.0, roce_med_3y=20.0,
        )])
        assert res.loc[0, "mauboussin_score"] == 2

    def test_cap_exactly_15_does_not_trigger_trap(self):
        # pe=50, nopat_margin=30%, rr=1.0 → 50 * 0.30 * 1.0 = 15.0 — STRICTLY > 15 required
        res = _run_maub([_build_mock_maub_row(
            pe=50.0, ebit=375.0, revenue=1000.0, pbt=375.0, pat=300.0,
            reinvestment_rate=1.0,
            roce=15.0, roce_med_3y=20.0,   # declining slope, but cap exactly at threshold
        )])
        assert abs(res.loc[0, "mauboussin_implied_cap"] - 15.0) < 1e-6
        assert res.loc[0, "mauboussin_cap_trap"] == 0, (
            "implied_cap == 15.0 must NOT trigger trap (threshold is strictly > 15)"
        )

    def test_cap_above_15_but_roce_stable_no_trap(self):
        # High CAP but structural ROCE slope not declining → no trap
        res = _run_maub([_build_mock_maub_row(
            pe=60.0, ebit=375.0, revenue=1000.0, pbt=375.0, pat=300.0,
            reinvestment_rate=1.0,
            roce=25.0, roce_med_3y=25.0,   # stable slope = 0 > -1 → no trap
        )])
        assert res.loc[0, "mauboussin_implied_cap"] > 15.0
        assert res.loc[0, "mauboussin_cap_trap"] == 0

    def test_roce_declining_but_low_cap_no_trap(self):
        # Declining structural ROCE slope but low implied CAP → no trap
        res = _run_maub([_build_mock_maub_row(
            pe=10.0, ebit=50.0, revenue=1000.0, pbt=50.0, pat=40.0,
            reinvestment_rate=0.20,
            roce=10.0, roce_med_3y=20.0,   # slope = -5 << -1.0 but cap too low
        )])
        assert res.loc[0, "mauboussin_cap_trap"] == 0
        assert res.loc[0, "mauboussin_pass"] == 1

    # ── Implied CAP formula ───────────────────────────────────────────────────

    def test_implied_cap_formula_correct(self):
        # v1.1: pe=40, nopat_margin=25%, rr=0.80 → 40 * 0.25 * 0.80 = 8.0
        # ebit=312.5, pbt=312.5, pat=250, revenue=1000 → eff_tax=0.20, nopat=250, margin=25%
        res = _run_maub([_build_mock_maub_row(
            pe=40.0, ebit=312.5, revenue=1000.0, pbt=312.5, pat=250.0,
            reinvestment_rate=0.80,
        )])
        expected = 40.0 * 0.25 * 0.80
        assert abs(res.loc[0, "mauboussin_implied_cap"] - expected) < 1e-4

    def test_implied_cap_zero_when_pe_is_zero(self):
        res = _run_maub([_build_mock_maub_row(pe=0.0)])
        assert res.loc[0, "mauboussin_implied_cap"] == 0.0

    def test_implied_cap_zero_when_ebit_is_zero(self):
        # v1.1: ebit=0 → nopat_margin=0 → implied_cap=0
        res = _run_maub([_build_mock_maub_row(ebit=0.0)])
        assert res.loc[0, "mauboussin_implied_cap"] == 0.0

    # ── AND invariant ─────────────────────────────────────────────────────────

    def test_each_single_failure_independently_kills_pass(self):
        c_fail = {
            "pe": 60.0, "ebit": 375.0, "revenue": 1000.0, "pbt": 375.0, "pat": 300.0,
            "reinvestment_rate": 1.0, "roce": 15.0, "roce_med_3y": 20.0,
        }
        failures = [
            {"sell_alert_treadmill": 1},   # T fails
            {"operating_leverage": 0},     # O fails
            c_fail,                        # C fails: implied_cap=18>15, slope=-2.5<-1
        ]
        for overrides in failures:
            res = _run_maub([_build_mock_maub_row(**overrides)])
            assert res.loc[0, "mauboussin_pass"] == 0, (
                f"Expected pass=0 with overrides={overrides}"
            )

    def test_two_failures_score_1(self):
        # T and O both fail
        res = _run_maub([_build_mock_maub_row(
            sell_alert_treadmill=1, operating_leverage=0
        )])
        assert res.loc[0, "mauboussin_score"] == 1

    def test_all_failures_score_0(self):
        # All three fail — v1.1 C-fail requires ebit/revenue + declining slope
        res = _run_maub([_build_mock_maub_row(
            sell_alert_treadmill=1,
            operating_leverage=0,
            pe=60.0, ebit=375.0, revenue=1000.0, pbt=375.0, pat=300.0,
            reinvestment_rate=1.0, roce=15.0, roce_med_3y=20.0,
        )])
        assert res.loc[0, "mauboussin_score"] == 0
        assert res.loc[0, "mauboussin_pass"]  == 0

    # ── Score / pass bidirectionality ─────────────────────────────────────────

    def test_pass_1_implies_score_3(self):
        res = _run_maub([_build_mock_maub_row()])
        if res.loc[0, "mauboussin_pass"] == 1:
            assert res.loc[0, "mauboussin_score"] == 3

    def test_score_3_implies_pass_1(self):
        res = _run_maub([_build_mock_maub_row()])
        if res.loc[0, "mauboussin_score"] == 3:
            assert res.loc[0, "mauboussin_pass"] == 1

    def test_score_range_0_to_3(self):
        rows = [
            _build_mock_maub_row(),
            _build_mock_maub_row(sell_alert_treadmill=1),
            _build_mock_maub_row(sell_alert_treadmill=1, operating_leverage=0),
            _build_mock_maub_row(
                sell_alert_treadmill=1, operating_leverage=0,
                pe=60.0, ebit=375.0, revenue=1000.0, pbt=375.0, pat=300.0,
                reinvestment_rate=1.0, roce=15.0, roce_med_3y=20.0,
            ),
        ]
        res = _run_maub(rows)
        for i, expected_score in enumerate([3, 2, 1, 0]):
            assert res.loc[i, "mauboussin_score"] == expected_score

    def test_no_asymmetric_disqualifier_unlike_lynch(self):
        # Mauboussin has no inventory-surge-style veto; score==3 always means pass==1
        res = _run_maub([_build_mock_maub_row()])
        assert res.loc[0, "mauboussin_score"] == 3
        assert res.loc[0, "mauboussin_pass"]  == 1


# ═══════════════════════════════════════════════════════════════════════════════
# TestMauboussinNaNConservative
# ═══════════════════════════════════════════════════════════════════════════════

class TestMauboussinNaNConservative:
    """NaN handling — all missing inputs must default conservatively."""

    def test_nan_pe_implies_cap_zero_no_trap(self):
        row = _build_mock_maub_row()
        row["pe"] = float("nan")
        res = _run_maub([row])
        assert res.loc[0, "mauboussin_implied_cap"] == 0.0
        assert res.loc[0, "mauboussin_cap_trap"] == 0

    def test_nan_revenue_implies_nopat_nan_implies_cap_zero(self):
        # v1.1: if revenue is NaN/0, nopat_margin = NaN → fillna(0) → implied_cap = 0
        row = _build_mock_maub_row()
        row["revenue"] = float("nan")
        res = _run_maub([row])
        assert res.loc[0, "mauboussin_implied_cap"] == 0.0

    def test_nan_reinvestment_rate_implies_cap_zero(self):
        row = _build_mock_maub_row()
        row["reinvestment_rate"] = float("nan")
        res = _run_maub([row])
        assert res.loc[0, "mauboussin_implied_cap"] == 0.0

    def test_nan_sell_alert_treadmill_defaults_to_safe(self):
        # fillna(0) → treadmill not firing → breach = 1 (conservative pass)
        row = _build_mock_maub_row()
        row["sell_alert_treadmill"] = float("nan")
        res = _run_maub([row])
        assert res.loc[0, "mauboussin_treadmill_breach"] == 1

    def test_nan_operating_leverage_defaults_to_functioning(self):
        # fillna(1) → operating leverage assumed functioning → oplev_drift = 1
        row = _build_mock_maub_row()
        row["operating_leverage"] = float("nan")
        res = _run_maub([row])
        assert res.loc[0, "mauboussin_oplev_drift"] == 1

    def test_nan_roce_med3y_defaults_to_no_slope(self):
        # v1.1: roce_med_3y NaN → fillna(roce) → slope = 0 → no cap trap even with high cap
        row = _build_mock_maub_row(
            pe=60.0, ebit=375.0, revenue=1000.0, pbt=375.0, pat=300.0,
            reinvestment_rate=1.0,
        )
        row["roce_med_3y"] = float("nan")
        res = _run_maub([row])
        assert res.loc[0, "mauboussin_cap_trap"] == 0

    def test_all_inputs_nan_still_produces_columns(self):
        row = _build_mock_maub_row()
        for col in ["pe", "ebit", "revenue", "reinvestment_rate",
                    "sell_alert_treadmill", "operating_leverage", "roce", "roce_med_3y"]:
            row[col] = float("nan")
        res = _run_maub([row])
        for col in ["mauboussin_implied_cap", "mauboussin_treadmill_breach",
                    "mauboussin_oplev_drift", "mauboussin_cap_trap",
                    "mauboussin_pass", "mauboussin_score"]:
            assert col in res.columns, f"Column missing: {col}"

    def test_all_inputs_nan_gives_pass_1(self):
        # All NaN → implied_cap=0 (no trap), treadmill safe, oplev functioning → all pass
        row = _build_mock_maub_row()
        for col in ["pe", "ebit", "revenue", "reinvestment_rate",
                    "sell_alert_treadmill", "operating_leverage", "roce", "roce_med_3y"]:
            row[col] = float("nan")
        res = _run_maub([row])
        assert res.loc[0, "mauboussin_pass"] == 1
        assert res.loc[0, "mauboussin_score"] == 3


# ═══════════════════════════════════════════════════════════════════════════════
# TestMauboussinIndexAlignment
# ═══════════════════════════════════════════════════════════════════════════════

class TestMauboussinIndexAlignment:
    """Non-default integer and string index safety."""

    def test_non_zero_start_index(self):
        rows = [_build_mock_maub_row(), _build_mock_maub_row(sell_alert_treadmill=1)]
        df = pd.DataFrame(rows, index=[100, 200])
        from config import MASTER_PROFILES
        from core.scoring_engine import compute_qglp_score
        df.attrs["detected_market_regime"] = "SIDEWAYS"
        profile = MASTER_PROFILES.get("Balanced", next(iter(MASTER_PROFILES.values())))
        res = compute_qglp_score(df, profile)
        assert res.loc[100, "mauboussin_pass"] == 1
        assert res.loc[200, "mauboussin_treadmill_breach"] == 0

    def test_string_index(self):
        rows = [_build_mock_maub_row(), _build_mock_maub_row(operating_leverage=0)]
        df = pd.DataFrame(rows, index=["TATA", "INFY"])
        from config import MASTER_PROFILES
        from core.scoring_engine import compute_qglp_score
        df.attrs["detected_market_regime"] = "SIDEWAYS"
        profile = MASTER_PROFILES.get("Balanced", next(iter(MASTER_PROFILES.values())))
        res = compute_qglp_score(df, profile)
        assert res.loc["TATA", "mauboussin_pass"]  == 1
        assert res.loc["INFY", "mauboussin_oplev_drift"] == 0

    def test_shuffled_index(self):
        rows = [
            _build_mock_maub_row(),
            _build_mock_maub_row(sell_alert_treadmill=1),
            _build_mock_maub_row(),
        ]
        df = pd.DataFrame(rows, index=[7, 3, 11])
        from config import MASTER_PROFILES
        from core.scoring_engine import compute_qglp_score
        df.attrs["detected_market_regime"] = "SIDEWAYS"
        profile = MASTER_PROFILES.get("Balanced", next(iter(MASTER_PROFILES.values())))
        res = compute_qglp_score(df, profile)
        assert res.loc[7,  "mauboussin_pass"] == 1
        assert res.loc[3,  "mauboussin_pass"] == 0
        assert res.loc[11, "mauboussin_pass"] == 1

    def test_single_row_df(self):
        res = _run_maub([_build_mock_maub_row()])
        assert "mauboussin_pass" in res.columns
        assert len(res) == 1

    def test_two_contrasting_rows_independent(self):
        rows = [
            _build_mock_maub_row(),
            _build_mock_maub_row(sell_alert_treadmill=1),
        ]
        res = _run_maub(rows)
        assert res.loc[0, "mauboussin_pass"] == 1
        assert res.loc[1, "mauboussin_pass"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# TestMauboussinUIContract
# ═══════════════════════════════════════════════════════════════════════════════

class TestMauboussinUIContract:
    """render_mauboussin_radar import + pure-display contract."""

    def test_render_function_importable(self):
        from ui.ui_tearsheet import render_mauboussin_radar
        assert callable(render_mauboussin_radar)

    def test_render_function_defined_in_source(self, ui_source):
        assert "def render_mauboussin_radar" in ui_source

    def test_render_function_docstring_pure_display(self, ui_source):
        assert re.search(
            r'def render_mauboussin_radar.*?PURE DISPLAY',
            ui_source, re.DOTALL
        ), "render_mauboussin_radar must declare PURE DISPLAY in docstring"

    def test_render_reads_treadmill_breach_column(self, ui_source):
        assert "mauboussin_treadmill_breach" in ui_source

    def test_render_reads_oplev_drift_column(self, ui_source):
        assert "mauboussin_oplev_drift" in ui_source

    def test_render_reads_cap_trap_column(self, ui_source):
        assert "mauboussin_cap_trap" in ui_source

    def test_render_reads_implied_cap_column(self, ui_source):
        assert "mauboussin_implied_cap" in ui_source

    def test_render_reads_score_column(self, ui_source):
        assert "mauboussin_score" in ui_source

    def test_render_reads_pass_column(self, ui_source):
        assert "mauboussin_pass" in ui_source

    def test_render_contains_pillar_letters_T_O_C(self, ui_source):
        render_block = ui_source[ui_source.find("def render_mauboussin_radar"):]
        # Find function body until next def
        next_def = render_block.find("\ndef ", 10)
        body = render_block[:next_def] if next_def > 0 else render_block
        assert '"T"' in body or "'T'" in body, "Pillar T must appear in render function"
        assert '"O"' in body or "'O'" in body, "Pillar O must appear in render function"
        assert '"C"' in body or "'C'" in body, "Pillar C must appear in render function"

    def test_render_contains_expected_value_calculator(self, ui_source):
        assert "Expected Value" in ui_source or "expected_value" in ui_source.lower()
        assert "p_upside" in ui_source or "p_up" in ui_source

    def test_render_amethyst_theme_color(self, ui_source):
        assert "#8b5cf6" in ui_source

    def test_render_contains_score_out_of_3(self, ui_source):
        assert "/ 3" in ui_source or "/3" in ui_source

    def test_render_no_threshold_recomputation(self, ui_source):
        # Pure display: no raw threshold math in render function
        render_start = ui_source.find("def render_mauboussin_radar")
        render_end_match = re.search(r'\ndef \w', ui_source[render_start + 10:])
        body = ui_source[render_start:render_start + render_end_match.start() + 10] if render_end_match else ui_source[render_start:]
        assert "15.0" not in body, (
            "render_mauboussin_radar must not recompute implied_cap threshold — PURE DISPLAY"
        )

    def test_docstring_version_references_spec(self, ui_source):
        assert "mauboussin_expectations_specs.json" in ui_source


# ═══════════════════════════════════════════════════════════════════════════════
# TestMauboussinRawSignalsContract
# ═══════════════════════════════════════════════════════════════════════════════

class TestMauboussinRawSignalsContract:
    """Mauboussin cells must be surfaced in render_mauboussin_radar.

    The All Data tab's Mauboussin pillar grid was removed (it duplicated the Frameworks-tab
    radar, which shows the pillars with labels + thresholds; every column still ships in the
    Export). The radar is now the single on-screen home for these cells.
    """

    @staticmethod
    def _radar_body(ui_source):
        start = ui_source.find("def render_mauboussin_radar")
        assert start != -1, "render_mauboussin_radar not defined in ui_tearsheet.py"
        m = re.search(r'\ndef \w', ui_source[start + 10:])
        return ui_source[start:start + m.start() + 10] if m else ui_source[start:]

    def test_radar_has_mauboussin_score_cell(self, ui_source):
        assert "mauboussin_score" in self._radar_body(ui_source), \
            "render_mauboussin_radar must display mauboussin_score"

    def test_radar_has_mauboussin_pass_cell(self, ui_source):
        assert "mauboussin_pass" in self._radar_body(ui_source), \
            "render_mauboussin_radar must display mauboussin_pass"

    def test_radar_has_implied_cap_cell(self, ui_source):
        assert "mauboussin_implied_cap" in self._radar_body(ui_source), \
            "render_mauboussin_radar must display mauboussin_implied_cap"

    def test_radar_mauboussin_section_has_amethyst_color(self, ui_source):
        assert "#8b5cf6" in self._radar_body(ui_source), \
            "render_mauboussin_radar must use amethyst color #8b5cf6"


# ═══════════════════════════════════════════════════════════════════════════════
# TestMauboussinAppWiring
# ═══════════════════════════════════════════════════════════════════════════════

class TestMauboussinAppWiring:
    """app.py correctly imports and calls render_mauboussin_radar."""

    def test_app_imports_render_mauboussin_radar(self, app_source):
        assert "render_mauboussin_radar" in app_source

    def test_app_calls_render_mauboussin_radar(self, app_source):
        assert re.search(
            r'render_mauboussin_radar\s*\(\s*stock\s*\)',
            app_source
        ), "app.py must call render_mauboussin_radar(stock)"

    def test_mauboussin_call_after_lynch_call(self, app_source):
        lynch_pos  = app_source.find("render_lynch_radar(stock)")
        maub_pos   = app_source.find("render_mauboussin_radar(stock)")
        assert lynch_pos > 0,  "render_lynch_radar(stock) not found in app.py"
        assert maub_pos  > 0,  "render_mauboussin_radar(stock) not found in app.py"
        assert maub_pos  > lynch_pos, (
            "render_mauboussin_radar must be called AFTER render_lynch_radar"
        )

    def test_init_exports_render_mauboussin_radar(self, init_source):
        assert "render_mauboussin_radar" in init_source

    def test_init_imports_from_tearsheet(self, init_source):
        assert re.search(
            r'from\s+\.ui_tearsheet\s+import.*render_mauboussin_radar',
            init_source, re.DOTALL
        ), "ui/__init__.py must import render_mauboussin_radar from .ui_tearsheet"

    def test_init_all_includes_render_mauboussin_radar(self, init_source):
        assert '"render_mauboussin_radar"' in init_source or \
               "'render_mauboussin_radar'" in init_source, (
                   "__all__ must include render_mauboussin_radar"
               )

    def test_init_stub_covers_render_mauboussin_radar(self, init_source):
        stub_block_start = init_source.find("except ImportError")
        assert stub_block_start > 0, "No except ImportError block found in __init__.py"
        stub_block = init_source[stub_block_start:]
        assert "render_mauboussin_radar" in stub_block, (
            "Stub fallback must cover render_mauboussin_radar in except ImportError block"
        )
