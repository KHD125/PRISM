"""
Contract Tests — Mauboussin Expectations Investing Framework (Framework 34)
============================================================================
Automated verification that docs/mauboussin_expectations_specs.json,
core/scoring_engine.py, ui/ui_tearsheet.py, and app.py are in perfect alignment.

Framework architecture v2.0 (REPOINTED 2026-08-28): gap-certification + trap-disqualifiers
  Pillar G — Expectations Discount: expectations_gap <= -5.0 with g_implied/g_star/roe ALL
             present (the book's ch.7 buy standard — the CERTIFICATION leg, strict inputs)
  Pillar T — Treadmill Safety:      sell_alert_treadmill not firing   (DISQUALIFIER)
  Pillar C — CAP Trap Clear:        implied_cap > 15 + ROCE 3Y slope < -1  (DISQUALIFIER)
  Layer 3 — Interactive Reverse DCF calculator (UI only, single-stock)

WHY v2.0 EXISTS. The v1.x gate (T & oplev & ~C) certified on operational checks alone —
census: T-clear 95.3%, C-clear 99.0%, oplev-intact 48.3%, whose product is the 45.8% that
was "EXPECTATIONS MATRIX CERTIFIED": the gate was effectively the oplev flag renamed. Worse, its NaN contract PINNED that a stock
with EVERY input missing passes (the old test_all_inputs_nan_gives_pass_1) — certification
on zero evidence, the consistency_champion defect class enshrined in a test. Meanwhile the
book's actual method (price-implied growth vs deliverable growth) sat computed in data_engine
(expectations_gap) and unread by the framework carrying the book's name.

THE TWO §5 POLARITIES THIS FILE PINS:
  • G certifies → requires evidence (all three inputs present; roe explicitly, because g_star
    silently fabricates a 0 deliverable when roe is missing). Unverifiable is not passed.
  • T and C disqualify → fire only on positive evidence (a missing alert is not an alert;
    absent evidence of the trap is not the trap). Unverifiable is equally not condemned.
  • Oplev left the gate (its fillna(1) was benefit-of-doubt certification; latent — 0 live
    NaN rows) but the COLUMN stays materialized (snapshot-schema stability).

GAP_MIN = 5.0 is PRISM's documented quant choice, not the book's (ch.7 prescribes no
universal cutoff — Table 7.7 maps price/EV × years-to-convergence). Census at ship:
gate fires 13.8% (293 of 2,117), cohort median quality 64.4 vs universe 33.8.

Key unit conventions:
  • expectations_gap    — PERCENT POINTS (g_implied − g_star, both % growth)
  • pe                  — FLOAT (e.g. 35.0 = 35×)
  • mauboussin_nopat_margin — PERCENTAGE output; capital-structure-neutral
  • reinvestment_rate   — DECIMAL [0,1]
  • Score range: 0–3 (bidirectional: score==3 ↔ pass==1, no asymmetric veto)
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
    """Build a fully-passing v2.0 Mauboussin row (G certified, T safe, C clear).

    Default values satisfy every gate:
      G — g_implied=5, g_star=15, expectations_gap=-10 (≤ -5), roe=25 → certified
          (these are data_engine Pass-1 columns; compute_qglp_score reads them via .get(),
           so the mock must CARRY them — they are not derived here)
      T — sell_alert_treadmill = 0   (treadmill not firing → breach = 1)
      C — implied_cap = 20 * 0.20 * 0.50 = 2.0  (< 15 threshold → cap_trap = 0)
            nopat_margin computed as: ebit=250, pbt=250, pat=200, revenue=1000
            → eff_tax=(250-200)/250=0.20, nopat=250*0.80=200, margin=200/1000=20%
            roce=25, roce_med_3y=25 → structural slope=0 → no ROCE decay
    """
    base = {
        # G pillar inputs (v2.0 certification leg)
        "g_implied":              5.0,
        "g_star":                15.0,
        "expectations_gap":     -10.0,
        "roe":                   25.0,
        # C pillar inputs
        "pe":                    20.0,   # → implied_cap = 20 * 0.20 * 0.50 = 2.0
        "ebit":                 250.0,
        "pbt":                  250.0,
        "pat":                  200.0,
        "revenue":             1000.0,
        "reinvestment_rate":      0.50,
        "roce":                  25.0,
        "roce_med_3y":           25.0,
        # T pillar input
        "sell_alert_treadmill":    0,
        # retired-from-gate column input (still materialized)
        "operating_leverage":      1,
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

    def test_version_is_v2_expectations_gap(self, spec):
        v = spec["_meta"]["version"].lower()
        assert "2.0" in v and "expectations" in v, (
            f"Version must be the v2.0 expectations-gap architecture; got: {spec['_meta']['version']}"
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

    def test_pillar_g_section_present(self, spec):
        assert "pillar_g_expectations_gap" in spec, "v2.0 spec must document the G pillar"

    def test_pillar_g_column_materialized(self, spec):
        assert spec["pillar_g_expectations_gap"]["_column_materialized"] == "mauboussin_gap_opportunity"

    def test_pillar_g_threshold_is_minus_5(self, spec):
        assert abs(spec["pillar_g_expectations_gap"]["pass_value"] - (-5.0)) < 1e-9

    def test_pillar_g_documents_threshold_provenance(self, spec):
        """GAP_MIN is OURS, not the book's — the spec must say so (book-fidelity rule)."""
        prov = spec["pillar_g_expectations_gap"]["threshold_provenance"].lower()
        assert "not the book" in prov and "census" in prov, (
            "the G threshold must be documented as PRISM's census-calibrated quant choice"
        )

    def test_pillar_t_section_present(self, spec):
        assert "pillar_t_treadmill" in spec

    def test_pillar_o_section_marked_retired(self, spec):
        """The oplev pillar section stays (the column exists) but must say it left the gate."""
        assert "pillar_o_operating_leverage" in spec
        assert spec["pillar_o_operating_leverage"]["in_gate"] is False

    def test_pillar_c_section_present(self, spec):
        assert "pillar_c_cap_trap" in spec

    def test_pillar_t_column_materialized(self, spec):
        assert spec["pillar_t_treadmill"]["_column_materialized"] == "mauboussin_treadmill_breach"

    def test_pillar_c_column_materialized(self, spec):
        assert spec["pillar_c_cap_trap"]["_column_materialized"] == "mauboussin_cap_trap"

    def test_output_registry_present(self, spec):
        assert "output_registry" in spec

    def test_output_registry_all_7_columns(self, spec):
        expected = [
            "mauboussin_implied_cap", "mauboussin_gap_opportunity",
            "mauboussin_treadmill_breach", "mauboussin_oplev_drift",
            "mauboussin_cap_trap", "mauboussin_pass", "mauboussin_score",
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
        for col in ["g_implied", "g_star", "roe", "pe", "reinvestment_rate",
                    "sell_alert_treadmill", "operating_leverage"]:
            assert col in spec["nan_handling"], f"nan_handling missing: {col}"

    def test_nan_handling_pins_the_all_missing_inversion(self, spec):
        """The headline v2.0 fix: all inputs missing must FAIL, and the spec must say the
        v1.x pin (pass on zero evidence) was deliberately inverted."""
        assert "all_inputs_missing" in spec["nan_handling"]
        assert "FAILS" in spec["nan_handling"]["all_inputs_missing"]

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
        ), "implied_cap formula must be _pe_ly * _nopat_m_ly * _rr_ly"

    def test_nopat_margin_column_computed(self, se_source):
        assert re.search(r'df\["mauboussin_nopat_margin"\]', se_source)

    def test_gap_opportunity_column_assigned(self, se_source):
        assert 'df["mauboussin_gap_opportunity"]' in se_source

    def test_gap_pillar_requires_all_three_inputs(self, se_source):
        """The strictness that inverts the old all-NaN pass: g_implied, g_star AND roe must
        all be .notna() before the gap can certify."""
        i = se_source.index('df["mauboussin_gap_opportunity"]')
        block = se_source[max(0, i - 1200):i + 200]
        assert re.search(r'_gi_v2\.notna\(\)\s*&\s*_gs_v2\.notna\(\)\s*&\s*_roe_v2\.notna\(\)',
                         block), "the G pillar no longer requires all three inputs present"

    def test_gap_threshold_is_minus_5(self, se_source):
        assert re.search(r'_eg_v2\s*<=\s*-5\.0', se_source), (
            "the G pillar must use expectations_gap <= -5.0 (spec pass_value)"
        )

    def test_gate_is_gap_and_treadmill_and_not_captrap(self, se_source):
        i = se_source.index("fw_mauboussin = (")
        block = se_source[i:i + 400]
        assert '"mauboussin_gap_opportunity"' in block, "the gate lost its certification leg"
        assert '"mauboussin_treadmill_breach"' in block
        assert '"mauboussin_cap_trap"' in block
        assert '"mauboussin_oplev_drift"' not in block, (
            "oplev is back in the gate — its fillna(1) certifies on absent evidence (retired v2.0)"
        )

    def test_treadmill_breach_column_assigned(self, se_source):
        assert 'df["mauboussin_treadmill_breach"]' in se_source

    def test_oplev_drift_column_still_materialized(self, se_source):
        """Retired from the GATE, retained as a COLUMN (snapshot-schema stability)."""
        assert 'df["mauboussin_oplev_drift"]' in se_source

    def test_cap_trap_column_assigned(self, se_source):
        assert 'df["mauboussin_cap_trap"]' in se_source

    def test_cap_trap_threshold_15(self, se_source):
        assert re.search(
            r'mauboussin_implied_cap.*>\s*15\.0|15\.0.*mauboussin_implied_cap', se_source)

    def test_cap_trap_slope_threshold_minus_1(self, se_source):
        assert re.search(r'_roce_slope_3y\s*<\s*-1\.0', se_source)

    def test_mauboussin_pass_column_assigned(self, se_source):
        assert 'df["mauboussin_pass"]' in se_source

    def test_mauboussin_score_column_assigned(self, se_source):
        assert 'df["mauboussin_score"]' in se_source

    def test_score_sums_gap_treadmill_captrap(self, se_source):
        i = se_source.index('df["mauboussin_score"]')
        block = se_source[i:i + 300]
        assert '"mauboussin_gap_opportunity"' in block
        assert '"mauboussin_treadmill_breach"' in block
        assert re.search(r'mauboussin_cap_trap.*==\s*0.*astype\(int\)', block)
        assert '"mauboussin_oplev_drift"' not in block

    def test_fw_str_includes_expectations_matrix(self, se_source):
        assert "Expectations Matrix" in se_source

    def test_fw_str_uses_fw_mauboussin(self, se_source):
        assert re.search(r'np\.where\(fw_mauboussin.*Expectations Matrix', se_source)

    def test_threshold_provenance_documented_in_code(self, se_source):
        """Book-fidelity rule: GAP_MIN is ours; the code comment must say the book prescribes
        no universal cutoff and that 5.0 came from the liveness census."""
        i = se_source.index('df["mauboussin_gap_opportunity"]')
        block = se_source[max(0, i - 3000):i]
        assert "GAP_MIN" in block and "census" in block, (
            "the census-calibrated threshold provenance comment is gone"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TestMauboussinPillarArithmetic
# ═══════════════════════════════════════════════════════════════════════════════

class TestMauboussinPillarArithmetic:
    """Boundary conditions, AND invariant, score 0-3 — v2.0 semantics."""

    # ── Full pass ─────────────────────────────────────────────────────────────

    def test_all_gates_pass_gives_pass_1_score_3(self):
        res = _run_maub([_build_mock_maub_row()])
        assert res.loc[0, "mauboussin_pass"]  == 1
        assert res.loc[0, "mauboussin_score"] == 3

    def test_frameworks_passed_contains_expectations_matrix_when_pass(self):
        res = _run_maub([_build_mock_maub_row()])
        assert "Expectations Matrix" in res.loc[0, "frameworks_passed"]

    # ── G pillar (the certification leg) ──────────────────────────────────────

    def test_insufficient_gap_kills_pass(self):
        """gap −4.9 misses the −5.0 bar → not certified."""
        res = _run_maub([_build_mock_maub_row(expectations_gap=-4.9)])
        assert res.loc[0, "mauboussin_gap_opportunity"] == 0
        assert res.loc[0, "mauboussin_pass"] == 0
        assert res.loc[0, "mauboussin_score"] == 2

    def test_gap_exactly_minus_5_certifies(self):
        """Boundary is inclusive: <= −5.0."""
        res = _run_maub([_build_mock_maub_row(expectations_gap=-5.0)])
        assert res.loc[0, "mauboussin_gap_opportunity"] == 1
        assert res.loc[0, "mauboussin_pass"] == 1

    def test_positive_gap_priced_above_deliverable_fails(self):
        res = _run_maub([_build_mock_maub_row(expectations_gap=12.0)])
        assert res.loc[0, "mauboussin_gap_opportunity"] == 0
        assert res.loc[0, "mauboussin_pass"] == 0

    # ── T pillar (disqualifier) ───────────────────────────────────────────────

    def test_treadmill_alert_firing_kills_pass(self):
        res = _run_maub([_build_mock_maub_row(sell_alert_treadmill=1)])
        assert res.loc[0, "mauboussin_treadmill_breach"] == 0
        assert res.loc[0, "mauboussin_pass"] == 0
        assert res.loc[0, "mauboussin_score"] == 2

    def test_treadmill_safe_gives_breach_1(self):
        res = _run_maub([_build_mock_maub_row(sell_alert_treadmill=0)])
        assert res.loc[0, "mauboussin_treadmill_breach"] == 1

    # ── Oplev is OUT of the gate (v2.0) ───────────────────────────────────────

    def test_oplev_broken_no_longer_kills_pass(self):
        """v2.0: the column still materializes 0, but the gate ignores it — its fillna(1)
        benefit-of-doubt made it a certification leg that passed on absent evidence."""
        res = _run_maub([_build_mock_maub_row(operating_leverage=0)])
        assert res.loc[0, "mauboussin_oplev_drift"] == 0, "column must still materialize"
        assert res.loc[0, "mauboussin_pass"] == 1, "oplev must no longer gate the pass"
        assert res.loc[0, "mauboussin_score"] == 3

    # ── C pillar (disqualifier) ───────────────────────────────────────────────

    def test_high_cap_plus_roce_decline_triggers_trap(self):
        res = _run_maub([_build_mock_maub_row(
            pe=60.0, ebit=375.0, revenue=1000.0, pbt=375.0, pat=300.0,
            reinvestment_rate=1.0,
            roce=15.0, roce_med_3y=20.0,
        )])
        assert res.loc[0, "mauboussin_implied_cap"] > 15.0
        assert res.loc[0, "mauboussin_cap_trap"] == 1
        assert res.loc[0, "mauboussin_pass"] == 0
        assert res.loc[0, "mauboussin_score"] == 2

    def test_cap_exactly_15_does_not_trigger_trap(self):
        res = _run_maub([_build_mock_maub_row(
            pe=50.0, ebit=375.0, revenue=1000.0, pbt=375.0, pat=300.0,
            reinvestment_rate=1.0,
            roce=15.0, roce_med_3y=20.0,
        )])
        assert abs(res.loc[0, "mauboussin_implied_cap"] - 15.0) < 1e-6
        assert res.loc[0, "mauboussin_cap_trap"] == 0, (
            "implied_cap == 15.0 must NOT trigger trap (threshold is strictly > 15)"
        )

    def test_cap_above_15_but_roce_stable_no_trap(self):
        res = _run_maub([_build_mock_maub_row(
            pe=60.0, ebit=375.0, revenue=1000.0, pbt=375.0, pat=300.0,
            reinvestment_rate=1.0,
            roce=25.0, roce_med_3y=25.0,
        )])
        assert res.loc[0, "mauboussin_implied_cap"] > 15.0
        assert res.loc[0, "mauboussin_cap_trap"] == 0

    # ── Implied CAP formula (unchanged v1.1 arithmetic) ───────────────────────

    def test_implied_cap_formula_correct(self):
        res = _run_maub([_build_mock_maub_row(
            pe=40.0, ebit=312.5, revenue=1000.0, pbt=312.5, pat=250.0,
            reinvestment_rate=0.80,
        )])
        expected = 40.0 * 0.25 * 0.80
        assert abs(res.loc[0, "mauboussin_implied_cap"] - expected) < 1e-4

    def test_implied_cap_zero_when_pe_is_zero(self):
        res = _run_maub([_build_mock_maub_row(pe=0.0)])
        assert res.loc[0, "mauboussin_implied_cap"] == 0.0

    # ── AND invariant ─────────────────────────────────────────────────────────

    def test_each_single_failure_independently_kills_pass(self):
        c_fail = {
            "pe": 60.0, "ebit": 375.0, "revenue": 1000.0, "pbt": 375.0, "pat": 300.0,
            "reinvestment_rate": 1.0, "roce": 15.0, "roce_med_3y": 20.0,
        }
        failures = [
            {"expectations_gap": +3.0},    # G fails: priced above deliverable
            {"sell_alert_treadmill": 1},   # T fails
            c_fail,                        # C fails: implied_cap=18>15, slope=-2.5<-1
        ]
        for overrides in failures:
            res = _run_maub([_build_mock_maub_row(**overrides)])
            assert res.loc[0, "mauboussin_pass"] == 0, (
                f"Expected pass=0 with overrides={overrides}"
            )

    def test_two_failures_score_1(self):
        res = _run_maub([_build_mock_maub_row(
            expectations_gap=+3.0, sell_alert_treadmill=1
        )])
        assert res.loc[0, "mauboussin_score"] == 1

    def test_all_failures_score_0(self):
        res = _run_maub([_build_mock_maub_row(
            expectations_gap=+3.0,
            sell_alert_treadmill=1,
            pe=60.0, ebit=375.0, revenue=1000.0, pbt=375.0, pat=300.0,
            reinvestment_rate=1.0, roce=15.0, roce_med_3y=20.0,
        )])
        assert res.loc[0, "mauboussin_score"] == 0
        assert res.loc[0, "mauboussin_pass"]  == 0

    # ── Score / pass bidirectionality ─────────────────────────────────────────

    def test_score_3_iff_pass_1(self):
        rows = [
            _build_mock_maub_row(),
            _build_mock_maub_row(expectations_gap=-4.9),
            _build_mock_maub_row(sell_alert_treadmill=1),
        ]
        res = _run_maub(rows)
        for i in range(len(rows)):
            assert (res.loc[i, "mauboussin_score"] == 3) == (res.loc[i, "mauboussin_pass"] == 1)

    def test_score_range_0_to_3(self):
        rows = [
            _build_mock_maub_row(),
            _build_mock_maub_row(expectations_gap=+3.0),
            _build_mock_maub_row(expectations_gap=+3.0, sell_alert_treadmill=1),
            _build_mock_maub_row(
                expectations_gap=+3.0, sell_alert_treadmill=1,
                pe=60.0, ebit=375.0, revenue=1000.0, pbt=375.0, pat=300.0,
                reinvestment_rate=1.0, roce=15.0, roce_med_3y=20.0,
            ),
        ]
        res = _run_maub(rows)
        for i, expected_score in enumerate([3, 2, 1, 0]):
            assert res.loc[i, "mauboussin_score"] == expected_score


# ═══════════════════════════════════════════════════════════════════════════════
# TestMauboussinNaNStrictness
# ═══════════════════════════════════════════════════════════════════════════════

class TestMauboussinNaNStrictness:
    """The two §5 polarities: certification requires evidence; disqualifiers require evidence.

    THE HEADLINE INVERSION: v1.x pinned test_all_inputs_nan_gives_pass_1 — a stock with every
    input missing was CERTIFIED. v2.0 pins the opposite: it fails, because the certification
    leg (G) demands its inputs. The disqualifier defaults keep their old polarity — a missing
    alert cannot condemn — so the all-NaN score is exactly 2 (T=1, C-clear=1, G=0)."""

    def test_missing_g_implied_is_not_certified(self):
        res = _run_maub([_build_mock_maub_row(g_implied=float("nan"))])
        assert res.loc[0, "mauboussin_gap_opportunity"] == 0
        assert res.loc[0, "mauboussin_pass"] == 0

    def test_missing_g_star_is_not_certified(self):
        res = _run_maub([_build_mock_maub_row(g_star=float("nan"))])
        assert res.loc[0, "mauboussin_gap_opportunity"] == 0
        assert res.loc[0, "mauboussin_pass"] == 0

    def test_missing_roe_is_not_certified_even_with_a_deep_gap(self):
        """g_star fabricates a 0 deliverable when roe is missing (roe.fillna(0) inside its
        min), which would make the gap look deeply negative. The explicit roe requirement
        closes that hole."""
        res = _run_maub([_build_mock_maub_row(roe=float("nan"), expectations_gap=-40.0)])
        assert res.loc[0, "mauboussin_gap_opportunity"] == 0
        assert res.loc[0, "mauboussin_pass"] == 0

    def test_nan_sell_alert_treadmill_cannot_disqualify(self):
        res = _run_maub([_build_mock_maub_row(sell_alert_treadmill=float("nan"))])
        assert res.loc[0, "mauboussin_treadmill_breach"] == 1

    def test_nan_roce_med3y_cannot_fabricate_the_trap(self):
        row = _build_mock_maub_row(
            pe=60.0, ebit=375.0, revenue=1000.0, pbt=375.0, pat=300.0,
            reinvestment_rate=1.0,
        )
        row["roce_med_3y"] = float("nan")
        res = _run_maub([row])
        assert res.loc[0, "mauboussin_cap_trap"] == 0

    def test_all_inputs_nan_still_produces_columns(self):
        row = _build_mock_maub_row()
        for col in ["g_implied", "g_star", "expectations_gap", "roe",
                    "pe", "ebit", "revenue", "reinvestment_rate",
                    "sell_alert_treadmill", "operating_leverage", "roce_med_3y"]:
            row[col] = float("nan")
        res = _run_maub([row])
        for col in ["mauboussin_implied_cap", "mauboussin_gap_opportunity",
                    "mauboussin_treadmill_breach", "mauboussin_oplev_drift",
                    "mauboussin_cap_trap", "mauboussin_pass", "mauboussin_score"]:
            assert col in res.columns, f"Column missing: {col}"

    def test_all_inputs_nan_FAILS(self):
        """THE INVERSION. v1.x: all NaN → pass=1 score=3 ('EXPECTATIONS MATRIX CERTIFIED' on
        zero evidence — pinned by the old contract). v2.0: the certification leg is dead
        without its inputs → pass=0, score exactly 2 (the disqualifiers stay silent, as they
        must — unverifiable is neither passed nor condemned)."""
        row = _build_mock_maub_row()
        for col in ["g_implied", "g_star", "expectations_gap", "roe",
                    "pe", "ebit", "revenue", "reinvestment_rate",
                    "sell_alert_treadmill", "operating_leverage", "roce_med_3y"]:
            row[col] = float("nan")
        res = _run_maub([row])
        assert res.loc[0, "mauboussin_pass"] == 0, "a stock with zero evidence was certified"
        assert res.loc[0, "mauboussin_score"] == 2


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
        rows = [_build_mock_maub_row(), _build_mock_maub_row(expectations_gap=+3.0)]
        df = pd.DataFrame(rows, index=["TATA", "INFY"])
        from config import MASTER_PROFILES
        from core.scoring_engine import compute_qglp_score
        df.attrs["detected_market_regime"] = "SIDEWAYS"
        profile = MASTER_PROFILES.get("Balanced", next(iter(MASTER_PROFILES.values())))
        res = compute_qglp_score(df, profile)
        assert res.loc["TATA", "mauboussin_pass"] == 1
        assert res.loc["INFY", "mauboussin_gap_opportunity"] == 0

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

    def test_two_contrasting_rows_independent(self):
        rows = [
            _build_mock_maub_row(),
            _build_mock_maub_row(expectations_gap=+3.0),
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

    def test_render_function_docstring_pure_display(self, ui_source):
        assert re.search(
            r'def render_mauboussin_radar.*?PURE DISPLAY',
            ui_source, re.DOTALL
        )

    @staticmethod
    def _radar_body(ui_source):
        start = ui_source.find("def render_mauboussin_radar")
        assert start != -1
        m = re.search(r'\ndef \w', ui_source[start + 10:])
        return ui_source[start:start + m.start() + 10] if m else ui_source[start:]

    def test_render_reads_gap_opportunity_column(self, ui_source):
        assert "mauboussin_gap_opportunity" in self._radar_body(ui_source), (
            "the radar must display the v2.0 certification pillar"
        )

    def test_render_reads_treadmill_breach_column(self, ui_source):
        assert "mauboussin_treadmill_breach" in self._radar_body(ui_source)

    def test_render_reads_cap_trap_column(self, ui_source):
        assert "mauboussin_cap_trap" in self._radar_body(ui_source)

    def test_render_reads_implied_cap_column(self, ui_source):
        assert "mauboussin_implied_cap" in self._radar_body(ui_source)

    def test_render_shows_the_gap_value(self, ui_source):
        """The header quotes expectations_gap so the pillar's evidence is on screen."""
        assert "expectations_gap" in self._radar_body(ui_source)

    def test_render_contains_pillar_letters_G_T_C(self, ui_source):
        body = self._radar_body(ui_source)
        for letter in ("G", "T", "C"):
            assert f'"{letter}"' in body or f"'{letter}'" in body, (
                f"Pillar {letter} must appear in render function"
            )

    def test_render_no_threshold_recomputation(self, ui_source):
        """Pure display: the radar mirrors mauboussin_gap_opportunity — it must never
        recompute the −5 threshold (the Fisher module↔engine drift lesson)."""
        body = self._radar_body(ui_source)
        assert "15.0" not in body
        assert "-5.0" not in body and "<= -5" not in body, (
            "the radar is recomputing the gap threshold instead of mirroring the engine pill"
        )

    def test_render_amethyst_theme_color(self, ui_source):
        assert "#8b5cf6" in self._radar_body(ui_source)

    def test_render_contains_expected_value_calculator(self, ui_source):
        assert "p_upside" in ui_source or "p_up" in ui_source

    def test_docstring_version_references_spec(self, ui_source):
        assert "mauboussin_expectations_specs.json" in ui_source


# ═══════════════════════════════════════════════════════════════════════════════
# TestMauboussinAppWiring
# ═══════════════════════════════════════════════════════════════════════════════

class TestMauboussinAppWiring:
    """app.py correctly imports and calls render_mauboussin_radar."""

    def test_app_imports_render_mauboussin_radar(self, app_source):
        assert "render_mauboussin_radar" in app_source

    def test_app_calls_render_mauboussin_radar(self, app_source):
        assert re.search(r'render_mauboussin_radar\s*\(\s*stock\s*\)', app_source)

    def test_mauboussin_call_after_lynch_call(self, app_source):
        lynch_pos  = app_source.find("render_lynch_radar(stock)")
        maub_pos   = app_source.find("render_mauboussin_radar(stock)")
        assert lynch_pos > 0 and maub_pos > 0
        assert maub_pos > lynch_pos

    def test_init_exports_render_mauboussin_radar(self, init_source):
        assert "render_mauboussin_radar" in init_source

    def test_init_imports_from_tearsheet(self, init_source):
        assert re.search(
            r'from\s+\.ui_tearsheet\s+import.*render_mauboussin_radar',
            init_source, re.DOTALL
        )

    def test_init_all_includes_render_mauboussin_radar(self, init_source):
        assert '"render_mauboussin_radar"' in init_source or \
               "'render_mauboussin_radar'" in init_source

    def test_init_stub_covers_render_mauboussin_radar(self, init_source):
        stub_block_start = init_source.find("except ImportError")
        assert stub_block_start > 0
        assert "render_mauboussin_radar" in init_source[stub_block_start:]
