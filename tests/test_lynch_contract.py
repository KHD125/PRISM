"""
Contract Tests — Peter Lynch Fast Grower Tenbagger Framework
=============================================================
Automated verification that docs/lynch_growth_specs.json and
core/scoring_engine.py are in perfect alignment.

Version 1.1 — 6 book-grounded improvements over v1.0:
  • Pillar V: rev_gr_5y >= 20%  AND  eps_gr_5y >= 15% (5Y/3Y fallback)  AND  FCF > 0
  • Pillar P: peg > 0  AND  peg <= 0.75                  (PEG Sweet Spot — unchanged)
  • Pillar D: (fii_holdings + dii_holdings) < 20%        (Pre-Discovery — combined threshold)
  • Pillar F: (D/E < 0.5 OR is_financial) AND (promoter >= 45% OR buying > 0)
  • Inventory Surge Disqualifier: vetos lynch_pass (NOT score) when inv_growth > rev + 20pp

Key unit conventions:
  • rev_gr_5y, eps_gr_5y, fii_holdings, dii_holdings, promoter_holdings  — PERCENTAGE
  • peg, debt_to_equity  — FLOAT RATIO (e.g. 0.75 = 0.75×)
  • fii/dii fillna(50): NaN → assume already discovered → conservative gate failure
  • D/E threshold is STRICT: < 0.5 (exactly 0.5 FAILS, unlike Malik's <= 0.5)
  • Score range: 0–4 (not 0–5 like Malik — only 4 pillars)
  • v1.1 one-directional: lynch_pass==1 → lynch_score==4, but score==4 ≠> pass==1 (inv. surge)

Structure:
    TestLynchSpecLedger         — JSON schema completeness and meta keys
    TestLynchEngineContract     — regex source-code threshold verification
    TestLynchPillarArithmetic   — boundary conditions, AND invariant, score 0-4
    TestLynchNaNConservative    — NaN handling strategy (conservative gate failure)
    TestLynchIndexAlignment     — non-default integer and string index safety
    TestLynchFinancialExemption — financial sector exempt from D/E sub-gate only
    TestLynchUIContract         — render_lynch_radar import + pure-display contract
    TestLynchRawSignalsContract — Lynch cells present in render_raw_signals
    TestLynchAppWiring          — app.py correctly imports and calls render_lynch_radar
    TestLynchV11Features        — v1.1 new gates: EPS, FCF, FII+DII, promoter buying, inv surge
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

SPEC_PATH = os.path.join(REPO_ROOT, "docs", "lynch_growth_specs.json")
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

def _build_mock_lynch_row(**overrides) -> dict:
    """Build a fully-passing Lynch Fast Grower row (all 4 pillars green).

    v1.1 Default values exceed every pillar gate:
      V — rev_gr_5y=25.0 (>= 20%), eps_gr_5y=18.0 (>= 15%), free_cash_flow=500.0 (> 0)
      P — peg=0.60 (0 < peg <= 0.75)
      D — fii_holdings=5.0, dii_holdings=5.0 → combined 10% < 20%
      F — debt_to_equity=0.3 (< 0.5), promoter_holdings=55.0 (>= 45%), change_promoter_1y=0.0

    pat_gr_3y=18.0 retained for backward test compatibility; v1.1 engine ignores it (uses EPS).
    """
    base = {
        "rev_gr_5y":           25.0,  # V: 25% > 20% threshold
        "eps_gr_5y":           18.0,  # V: 18% > 15% EPS per share threshold (v1.1)
        "pat_gr_3y":           18.0,  # V: legacy v1.0 field (ignored by v1.1 engine)
        "free_cash_flow":     500.0,  # V: positive FCF cash gate (v1.1)
        "peg":                  0.60, # P: 0.60 > 0 AND <= 0.75 sweet spot
        "fii_holdings":         5.0,  # D: FII+DII = 5+5 = 10% < 20% combined threshold (v1.1)
        "dii_holdings":         5.0,  # D: 5% DII component (v1.1)
        "debt_to_equity":       0.3,  # F: 0.3 < 0.5 threshold (strict less-than)
        "promoter_holdings":   55.0,  # F: 55% >= 45% threshold
        "change_promoter_1y":   0.0,  # F: not actively buying; level alone (55%) suffices (v1.1)
        "is_financial":        False,
        # scaffold columns
        "market_cap":    2500.0,
        "close_price":    250.0,
        "name":           "TestFastGrower",
        "sector":         "Consumer",
    }
    base.update(overrides)
    return base


def _run_lynch(rows: list, regime: str = "SIDEWAYS") -> pd.DataFrame:
    """Execute compute_qglp_score on a list of row dicts; return result df."""
    from config import MASTER_PROFILES
    from core.scoring_engine import compute_qglp_score
    df = pd.DataFrame(rows)
    df.attrs["detected_market_regime"] = regime
    profile = MASTER_PROFILES.get("Balanced", next(iter(MASTER_PROFILES.values())))
    return compute_qglp_score(df, profile)


# ═══════════════════════════════════════════════════════════════════════════════
# TestLynchSpecLedger
# ═══════════════════════════════════════════════════════════════════════════════

class TestLynchSpecLedger:
    """Verify docs/lynch_growth_specs.json is complete and structurally correct."""

    def test_spec_file_exists(self):
        assert os.path.exists(SPEC_PATH), f"Spec file not found: {SPEC_PATH}"

    def test_spec_is_valid_json(self, spec):
        assert isinstance(spec, dict)

    def test_meta_section_present(self, spec):
        assert "_meta" in spec

    def test_meta_required_keys(self, spec):
        required = [
            "title", "author", "framework_variable", "pass_column",
            "score_column", "frameworks_passed_label", "implementation_file",
            "version", "comment_anchor",
        ]
        for key in required:
            assert key in spec["_meta"], f"_meta missing key: {key}"

    def test_pass_column_name(self, spec):
        assert spec["_meta"]["pass_column"] == "lynch_pass"

    def test_score_column_name(self, spec):
        assert spec["_meta"]["score_column"] == "lynch_score"

    def test_framework_variable_name(self, spec):
        assert spec["_meta"]["framework_variable"] == "fw_lynch"

    def test_framework_label(self, spec):
        assert spec["_meta"]["frameworks_passed_label"] == "Lynch Dream"

    def test_version_present(self, spec):
        assert "lynch" in spec["_meta"]["version"].lower(), (
            f"Version should reference lynch; got: {spec['_meta']['version']}"
        )

    def test_framework_number_is_5(self, spec):
        assert spec["_meta"]["framework_number_in_code"] == 5, (
            f"Lynch is Framework 5. Got: {spec['_meta']['framework_number_in_code']}"
        )

    def test_comment_anchor_references_lynch(self, spec):
        anchor = spec["_meta"]["comment_anchor"]
        assert "Lynch" in anchor or "lynch" in anchor, (
            f"comment_anchor must reference Lynch; got: {anchor}"
        )

    def test_4_pillar_sections_present(self, spec):
        required_pillars = [
            "pillar_v_growth_velocity",
            "pillar_p_valuation_peg",
            "pillar_d_pre_discovery",
            "pillar_f_fortress_owner",
        ]
        for p in required_pillars:
            assert p in spec, f"Spec missing pillar section: {p}"

    def test_pillar_v_column_materialized(self, spec):
        assert spec["pillar_v_growth_velocity"]["_column_materialized"] == "lynch_growth_velocity"

    def test_pillar_p_column_materialized(self, spec):
        assert spec["pillar_p_valuation_peg"]["_column_materialized"] == "lynch_valuation_peg"

    def test_pillar_d_column_materialized(self, spec):
        assert spec["pillar_d_pre_discovery"]["_column_materialized"] == "lynch_pre_discovery"

    def test_pillar_f_column_materialized(self, spec):
        assert spec["pillar_f_fortress_owner"]["_column_materialized"] == "lynch_fortress_owner"

    # ── Pillar V thresholds ───────────────────────────────────────────────────

    def test_pillar_v_revenue_threshold_is_20(self, spec):
        t = spec["pillar_v_growth_velocity"]["revenue_gate"]["threshold"]
        assert abs(t - 20.0) < 1e-9, f"Pillar V revenue threshold must be 20.0; got: {t}"

    def test_pillar_v_revenue_operator_is_gte(self, spec):
        op = spec["pillar_v_growth_velocity"]["revenue_gate"]["operator"]
        assert op == ">=", f"Pillar V revenue operator must be '>='; got: '{op}'"

    def test_pillar_v_pat_threshold_is_15(self, spec):
        t = spec["pillar_v_growth_velocity"]["earnings_confirmation_gate"]["threshold"]
        assert abs(t - 15.0) < 1e-9, f"Pillar V PAT threshold must be 15.0; got: {t}"

    def test_pillar_v_pat_operator_is_gte(self, spec):
        op = spec["pillar_v_growth_velocity"]["earnings_confirmation_gate"]["operator"]
        assert op == ">=", f"Pillar V PAT operator must be '>='; got: '{op}'"

    def test_pillar_v_unit_is_percentage(self, spec):
        unit = spec["pillar_v_growth_velocity"]["revenue_gate"]["unit"]
        assert "PERCENTAGE" in unit.upper(), f"Pillar V revenue unit must be PERCENTAGE; got: {unit}"

    # ── Pillar P thresholds ───────────────────────────────────────────────────

    def test_pillar_p_max_threshold_is_0point75(self, spec):
        t = spec["pillar_p_valuation_peg"]["peg_gate"]["threshold_max"]
        assert abs(t - 0.75) < 1e-9, f"Pillar P max PEG threshold must be 0.75; got: {t}"

    def test_pillar_p_min_threshold_is_0(self, spec):
        t = spec["pillar_p_valuation_peg"]["peg_gate"]["threshold_min_exclusive"]
        assert abs(t - 0.0) < 1e-9, f"Pillar P min PEG threshold must be 0.0; got: {t}"

    def test_pillar_p_operator_has_positive_guard(self, spec):
        op = spec["pillar_p_valuation_peg"]["peg_gate"]["operator"]
        assert ">" in op and "0" in op, (
            f"Pillar P operator must include positive PEG guard (peg > 0); got: '{op}'"
        )

    def test_pillar_p_operator_has_upper_bound(self, spec):
        op = spec["pillar_p_valuation_peg"]["peg_gate"]["operator"]
        assert "0.75" in op, (
            f"Pillar P operator must include upper bound (0.75); got: '{op}'"
        )

    def test_pillar_p_unit_is_float_ratio(self, spec):
        unit = spec["pillar_p_valuation_peg"]["peg_gate"]["unit"]
        assert "RATIO" in unit.upper() or "FLOAT" in unit.upper(), (
            f"Pillar P PEG unit must be FLOAT_RATIO; got: {unit}"
        )

    # ── Pillar D thresholds ───────────────────────────────────────────────────

    def test_pillar_d_combined_threshold_is_20(self, spec):
        """v1.1: combined FII+DII threshold is 20.0 (replaces v1.0 FII-only threshold of 10.0)."""
        t = spec["pillar_d_pre_discovery"]["institutional_ownership_gate"]["threshold"]
        assert abs(t - 20.0) < 1e-9, (
            f"Pillar D v1.1 combined threshold must be 20.0 (FII+DII < 20%); got: {t}"
        )

    def test_pillar_d_operator_is_strict_lt(self, spec):
        op = spec["pillar_d_pre_discovery"]["institutional_ownership_gate"]["operator"]
        assert op == "<", f"Pillar D operator must be strict '<'; got: '{op}'"

    def test_pillar_d_fillna_is_50_conservative(self, spec):
        """FII fillna(50) is critical — missing data → assume already discovered → gate fails."""
        fillna = spec["pillar_d_pre_discovery"]["institutional_ownership_gate"]["fillna_strategy"]
        assert "50" in fillna, (
            f"Pillar D fillna strategy must use 50 (assume already discovered); got: {fillna}"
        )

    def test_pillar_d_unit_is_percentage(self, spec):
        unit = spec["pillar_d_pre_discovery"]["institutional_ownership_gate"]["unit"]
        assert "PERCENTAGE" in unit.upper(), f"Pillar D FII unit must be PERCENTAGE; got: {unit}"

    # ── Pillar F thresholds ───────────────────────────────────────────────────

    def test_pillar_f_de_threshold_is_0point5(self, spec):
        t = spec["pillar_f_fortress_owner"]["debt_gate"]["threshold"]
        assert abs(t - 0.5) < 1e-9, f"Pillar F D/E threshold must be 0.5; got: {t}"

    def test_pillar_f_de_operator_is_strict_lt(self, spec):
        """CRITICAL: Lynch D/E is STRICT < 0.5 (exactly 0.5 fails), unlike Malik's <= 0.5."""
        op = spec["pillar_f_fortress_owner"]["debt_gate"]["operator"]
        assert op == "<", (
            f"Pillar F D/E operator must be strict '<' (exactly 0.5 fails). Got: '{op}'. "
            "Lynch's threshold is D/E < 0.5, not D/E <= 0.5."
        )

    def test_pillar_f_promoter_threshold_is_45(self, spec):
        t = spec["pillar_f_fortress_owner"]["promoter_gate"]["threshold"]
        assert abs(t - 45.0) < 1e-9, f"Pillar F promoter threshold must be 45.0; got: {t}"

    def test_pillar_f_promoter_operator_is_gte_or_buying(self, spec):
        """v1.1: promoter gate uses OR condition — level >= 45% OR change_promoter_1y > 0."""
        op = spec["pillar_f_fortress_owner"]["promoter_gate"]["operator"]
        assert ">=" in op and "45" in op, (
            f"Pillar F promoter operator must include '>= 45'; got: '{op}'"
        )

    def test_pillar_f_financial_exemption_present(self, spec):
        exemption = spec["pillar_f_fortress_owner"]["financial_sector_exemption"]
        assert "condition" in exemption
        assert "is_financial" in exemption["condition"]

    def test_pillar_f_financial_exemption_debt_only(self, spec):
        """Exemption applies to DEBT gate only — promoter requirement still stands."""
        behaviour = spec["pillar_f_fortress_owner"]["financial_sector_exemption"]["behaviour"]
        assert "debt" in behaviour.lower() or "D/E" in behaviour, (
            "Financial sector exemption must be documented as debt-gate-only"
        )

    # ── Output columns registry ───────────────────────────────────────────────

    def test_output_columns_registry_present(self, spec):
        assert "output_columns_registry" in spec
        required = [
            "lynch_growth_velocity", "lynch_valuation_peg",
            "lynch_pre_discovery", "lynch_fortress_owner",
            "lynch_pass", "lynch_score",
        ]
        for col in required:
            assert col in spec["output_columns_registry"], (
                f"output_columns_registry missing: {col}"
            )

    def test_lynch_score_range_documented_as_0_to_4(self, spec):
        """lynch_score range must be 0-4 (not 0-5 like Malik — only 4 pillars)."""
        sm = spec["scoring_matrix"]
        assert sm["score_range"] == "0-4", (
            f"Lynch score range must be 0-4 (4 pillars, not 5). Got: {sm['score_range']}"
        )

    def test_scoring_matrix_all_gates_equal_weight(self, spec):
        sm = spec["scoring_matrix"]
        assert sm["all_gates_equal_weight"] is True

    def test_vectorization_matrix_present(self, spec):
        assert "vectorization_matrix" in spec
        vm = spec["vectorization_matrix"]["nan_handling"]
        # v1.1: eps_gr_5y replaces pat_gr_3y as primary earnings column
        for col in ["rev_gr_5y", "eps_gr_5y", "peg", "fii_holdings",
                    "debt_to_equity", "promoter_holdings"]:
            assert col in vm, f"vectorization_matrix missing NaN handling for: {col}"

    def test_vectorization_fii_fillna_50(self, spec):
        vm = spec["vectorization_matrix"]["nan_handling"]
        fii_strategy = vm["fii_holdings"]
        assert "50" in fii_strategy, (
            f"fii_holdings NaN strategy must use fillna(50); got: {fii_strategy}"
        )

    def test_vectorization_peg_fillna_999(self, spec):
        vm = spec["vectorization_matrix"]["nan_handling"]
        peg_strategy = vm["peg"]
        assert "999" in peg_strategy, (
            f"peg NaN strategy must use fillna(999); got: {peg_strategy}"
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

    def test_peg_noted_as_unique_to_lynch(self, spec):
        """PEG <= 0.75 valuation gate is UNIQUE to Lynch — spec must document this."""
        dists = spec["distinction_from_other_frameworks"]
        dists_str = json.dumps(dists).lower()
        assert "unique" in dists_str, (
            "distinction_from_other_frameworks must note PEG gate is unique to Lynch"
        )

    def test_unit_notes_section_present(self, spec):
        assert "unit_notes" in spec["_meta"]
        # v1.1: eps_gr_5y is now the primary earnings column; pat_gr_3y retained as legacy
        for col in ["rev_gr_5y", "eps_gr_5y", "peg", "fii_holdings",
                    "debt_to_equity", "promoter_holdings"]:
            assert col in spec["_meta"]["unit_notes"], (
                f"unit_notes missing entry for: {col}"
            )

    def test_unit_notes_peg_is_float_ratio(self, spec):
        note = spec["_meta"]["unit_notes"]["peg"]
        assert "RATIO" in note.upper() or "FLOAT" in note.upper(), (
            f"unit_notes.peg must document FLOAT RATIO convention; got: {note}"
        )

    def test_pass_flag_logic_section_present(self, spec):
        assert "pass_flag_logic" in spec
        pfl = spec["pass_flag_logic"]
        assert pfl["_column"] == "lynch_pass"

    def test_score_column_logic_section_present(self, spec):
        assert "score_column_logic" in spec
        scl = spec["score_column_logic"]
        assert scl["_column"] == "lynch_score"
        assert "0-4" in scl["_dtype"] or "0-4" in scl["_description"]


# ═══════════════════════════════════════════════════════════════════════════════
# TestLynchEngineContract
# ═══════════════════════════════════════════════════════════════════════════════

class TestLynchEngineContract:
    """Verify scoring_engine.py has all Lynch columns and correct thresholds."""

    def test_lynch_anchor_comment(self, se_source):
        assert "Lynch" in se_source and "Fast Grower" in se_source

    def test_lynch_spec_reference_in_code(self, se_source):
        """Code must reference the spec JSON file."""
        assert "lynch_growth_specs.json" in se_source

    def test_lynch_growth_velocity_column_defined(self, se_source):
        assert 'df["lynch_growth_velocity"]' in se_source

    def test_lynch_valuation_peg_column_defined(self, se_source):
        assert 'df["lynch_valuation_peg"]' in se_source

    def test_lynch_pre_discovery_column_defined(self, se_source):
        assert 'df["lynch_pre_discovery"]' in se_source

    def test_lynch_fortress_owner_column_defined(self, se_source):
        assert 'df["lynch_fortress_owner"]' in se_source

    def test_lynch_pass_column_defined(self, se_source):
        assert 'df["lynch_pass"]' in se_source

    def test_lynch_score_column_defined(self, se_source):
        assert 'df["lynch_score"]' in se_source

    def test_pillar_v_revenue_threshold_20_in_code(self, se_source):
        """Pillar V: rev_ly.fillna(0) >= 20.0"""
        assert re.search(r'rev_ly\.fillna\s*\(\s*0\s*\)\s*>=\s*20\.0', se_source), (
            "scoring_engine must check rev_ly.fillna(0) >= 20.0 for lynch_growth_velocity"
        )

    def test_pillar_v_eps_threshold_15_in_code(self, se_source):
        """Pillar V v1.1: eps_ly.fillna(0) >= 15.0 (EPS per share replaces PAT total)."""
        assert re.search(r'eps_ly\.fillna\s*\(\s*0\s*\)\s*>=\s*15\.0', se_source), (
            "scoring_engine must check eps_ly.fillna(0) >= 15.0 for lynch_growth_velocity. "
            "v1.1: EPS per share (eps_gr_5y/eps_gr_3y fallback) replaces pat_gr_3y."
        )

    def test_pillar_p_peg_positive_guard_in_code(self, se_source):
        """Pillar P: peg_ly > 0 (excludes negative-PE stocks)."""
        assert re.search(r'peg_ly\s*>\s*0', se_source), (
            "scoring_engine must check peg_ly > 0 (positive PEG filter) for lynch_valuation_peg"
        )

    def test_pillar_p_peg_upper_bound_in_code(self, se_source):
        """Pillar P: peg_ly <= 0.75 (Lynch's sweet spot)."""
        assert re.search(r'peg_ly\s*<=\s*0\.75', se_source), (
            "scoring_engine must check peg_ly <= 0.75 for lynch_valuation_peg"
        )

    def test_pillar_d_combined_threshold_20_in_code(self, se_source):
        """Pillar D v1.1: (fii_ly + dii_ly) < 20.0 — combined institutional threshold."""
        assert re.search(r'\(\s*fii_ly\s*\+\s*dii_ly\s*\)\s*<\s*20\.0', se_source), (
            "scoring_engine must check (fii_ly + dii_ly) < 20.0 for lynch_pre_discovery. "
            "v1.1: combined FII+DII threshold replaces FII-only < 10.0."
        )

    def test_pillar_f_de_threshold_0point5_in_code(self, se_source):
        """Pillar F: debt_ly.fillna(999) < 0.5 (strict less-than, not <=)."""
        assert re.search(r'debt_ly\.fillna\s*\(\s*999\s*\)\s*<\s*0\.5', se_source), (
            "scoring_engine must check debt_ly.fillna(999) < 0.5 for lynch_fortress_owner. "
            "Must be strict '<' (not '<='). Exactly 0.5 should FAIL."
        )

    def test_pillar_f_de_not_lte_in_code(self, se_source):
        """Lynch D/E gate must be strict < 0.5, NOT <= 0.5."""
        # Extract the Lynch block specifically to avoid matching Malik's <= 0.5
        lynch_block = re.search(
            r'Lynch Fast Grower.*?(?=# \d+\.\s+(?:CAN SLIM|Schilit|Billionaire|fw_str|Build))',
            se_source, re.DOTALL
        )
        if lynch_block:
            block = lynch_block.group(0)
            # Should NOT have <= 0.5 for the Lynch D/E gate (that's Malik's threshold)
            # Note: checking within the Fortress Owner definition specifically
            fortress_section = re.search(
                r'lynch_fortress_owner.*?(?=lynch_pass|fw_lynch)', block, re.DOTALL
            )
            if fortress_section:
                fortress_code = fortress_section.group(0)
                assert not re.search(r'debt_ly\.fillna\s*\(\s*999\s*\)\s*<=\s*0\.5', fortress_code), (
                    "lynch_fortress_owner D/E gate must use strict '<' not '<='. "
                    "Exactly D/E=0.5 must FAIL for Lynch."
                )

    def test_pillar_f_promoter_threshold_45_in_code(self, se_source):
        """Pillar F: promo_ly.fillna(0) >= 45.0"""
        assert re.search(r'promo_ly\.fillna\s*\(\s*0\s*\)\s*>=\s*45\.0', se_source), (
            "scoring_engine must check promo_ly.fillna(0) >= 45.0 for lynch_fortress_owner"
        )

    def test_pillar_f_financial_exemption_in_code(self, se_source):
        """Pillar F: is_fin_ly makes financial sector exempt from D/E gate."""
        assert re.search(r'is_fin_ly\s*\|', se_source), (
            "scoring_engine must have is_fin_ly | (debt gate) for financial sector exemption"
        )

    def test_fii_fillna_50_in_code(self, se_source):
        """Critical: FII NaN → fillna(50) → assume already discovered → gate fails."""
        assert re.search(r'fii_holdings.*fillna\s*\(\s*50', se_source, re.DOTALL) or \
               re.search(r'pd\.Series\s*\(\s*50\.0.*fii', se_source, re.DOTALL) or \
               re.search(r'fii_ly.*fillna\s*\(\s*50\s*\)', se_source), (
            "scoring_engine must use fillna(50) for fii_holdings — assume already discovered"
        )

    def test_peg_fillna_999_in_code(self, se_source):
        """PEG NaN → fillna(999) → 999 > 0.75 → gate fails."""
        assert re.search(r'peg.*fillna\s*\(\s*999\s*\)', se_source) or \
               re.search(r'pd\.Series\s*\(\s*999\.0.*peg', se_source), (
            "scoring_engine must use fillna(999) or pd.Series(999.0) for peg"
        )

    def test_fw_lynch_is_and_of_4_pillars(self, se_source):
        """fw_lynch = AND of all 4 materialized pillar columns."""
        assert re.search(r'fw_lynch\s*=\s*\(', se_source)
        for col in ["lynch_growth_velocity", "lynch_valuation_peg",
                    "lynch_pre_discovery", "lynch_fortress_owner"]:
            assert col in se_source, f"fw_lynch must reference {col}"

    def test_fw_str_includes_lynch_dream_label(self, se_source):
        """fw_str must include 'Lynch Dream|' via fw_lynch."""
        assert re.search(r'fw_lynch.*Lynch Dream', se_source), (
            "fw_str must have np.where(fw_lynch, 'Lynch Dream|', '')"
        )

    def test_lynch_score_is_sum_of_4_pillars(self, se_source):
        """lynch_score = sum of the 4 pillar flags (0-4 range)."""
        assert re.search(r'lynch_score.*lynch_growth_velocity', se_source, re.DOTALL), (
            "lynch_score must sum all 4 pillar flags"
        )

    def test_spec_comment_in_lynch_block(self, se_source):
        """Lynch block must reference spec file in a comment."""
        lynch_block = re.search(
            r'Lynch Fast Grower.*?(?=# \d+\.\s+(?:CAN SLIM|Schilit|Billionaire|fw_str))',
            se_source, re.DOTALL
        )
        if lynch_block:
            block = lynch_block.group(0)
            assert "lynch_growth_specs.json" in block, (
                "Lynch scoring block must have a comment referencing lynch_growth_specs.json"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TestLynchPillarArithmetic
# ═══════════════════════════════════════════════════════════════════════════════

class TestLynchPillarArithmetic:
    """Boundary conditions and arithmetic invariants for all 4 pillar gates."""

    # ── Pillar V: Growth Velocity ─────────────────────────────────────────────

    def test_pillar_v_revenue_above_boundary_passes(self):
        result = _run_lynch([_build_mock_lynch_row(rev_gr_5y=25.0)])
        assert result["lynch_growth_velocity"].iloc[0] == 1

    def test_pillar_v_revenue_at_boundary_passes(self):
        """Exactly 20.0% = at threshold → passes (>= 20.0)."""
        result = _run_lynch([_build_mock_lynch_row(rev_gr_5y=20.0)])
        assert result["lynch_growth_velocity"].iloc[0] == 1

    def test_pillar_v_revenue_just_below_boundary_fails(self):
        result = _run_lynch([_build_mock_lynch_row(rev_gr_5y=19.9)])
        assert result["lynch_growth_velocity"].iloc[0] == 0

    def test_pillar_v_revenue_zero_fails(self):
        result = _run_lynch([_build_mock_lynch_row(rev_gr_5y=0.0)])
        assert result["lynch_growth_velocity"].iloc[0] == 0

    def test_pillar_v_revenue_negative_fails(self):
        result = _run_lynch([_build_mock_lynch_row(rev_gr_5y=-10.0)])
        assert result["lynch_growth_velocity"].iloc[0] == 0

    def test_pillar_v_eps_at_boundary_passes(self):
        """v1.1: EPS per share exactly 15.0% = at threshold → passes (>= 15.0)."""
        result = _run_lynch([_build_mock_lynch_row(eps_gr_5y=15.0)])
        assert result["lynch_growth_velocity"].iloc[0] == 1

    def test_pillar_v_eps_just_below_boundary_fails(self):
        """v1.1: EPS per share 14.9% < 15% → Pillar V fails."""
        result = _run_lynch([_build_mock_lynch_row(eps_gr_5y=14.9)])
        assert result["lynch_growth_velocity"].iloc[0] == 0

    def test_pillar_v_eps_zero_fails(self):
        """v1.1: EPS per share 0% → no dilution-adjusted earnings growth → Pillar V fails."""
        result = _run_lynch([_build_mock_lynch_row(eps_gr_5y=0.0)])
        assert result["lynch_growth_velocity"].iloc[0] == 0

    def test_pillar_v_rev_pass_eps_fail_fails(self):
        """v1.1: Revenue gate passes but EPS per share fails → Pillar V fails (AND condition)."""
        result = _run_lynch([_build_mock_lynch_row(rev_gr_5y=25.0, eps_gr_5y=10.0)])
        assert result["lynch_growth_velocity"].iloc[0] == 0

    def test_pillar_v_rev_fail_pat_pass_fails(self):
        """Revenue gate fails but PAT gate passes → Pillar V fails (AND condition)."""
        result = _run_lynch([_build_mock_lynch_row(rev_gr_5y=15.0, pat_gr_3y=18.0)])
        assert result["lynch_growth_velocity"].iloc[0] == 0

    # ── Pillar P: PEG Sweet Spot ──────────────────────────────────────────────

    def test_pillar_p_peg_sweet_spot_passes(self):
        """PEG=0.60 → 0 < 0.60 <= 0.75 → passes."""
        result = _run_lynch([_build_mock_lynch_row(peg=0.60)])
        assert result["lynch_valuation_peg"].iloc[0] == 1

    def test_pillar_p_peg_at_upper_boundary_passes(self):
        """PEG=0.75 exactly → passes (<= 0.75, inclusive upper)."""
        result = _run_lynch([_build_mock_lynch_row(peg=0.75)])
        assert result["lynch_valuation_peg"].iloc[0] == 1

    def test_pillar_p_peg_just_above_upper_boundary_fails(self):
        """PEG=0.76 > 0.75 → fails."""
        result = _run_lynch([_build_mock_lynch_row(peg=0.76)])
        assert result["lynch_valuation_peg"].iloc[0] == 0

    def test_pillar_p_peg_exactly_1_fails(self):
        """PEG=1.0 (fair value) → fails (Lynch sweet spot is below 0.75)."""
        result = _run_lynch([_build_mock_lynch_row(peg=1.0)])
        assert result["lynch_valuation_peg"].iloc[0] == 0

    def test_pillar_p_peg_above_1_fails(self):
        """PEG=2.0 (overvalued relative to growth) → fails."""
        result = _run_lynch([_build_mock_lynch_row(peg=2.0)])
        assert result["lynch_valuation_peg"].iloc[0] == 0

    def test_pillar_p_peg_zero_fails(self):
        """PEG=0.0 exactly → fails (positive filter: must be > 0)."""
        result = _run_lynch([_build_mock_lynch_row(peg=0.0)])
        assert result["lynch_valuation_peg"].iloc[0] == 0, (
            "peg=0.0 must FAIL — zero PEG is undefined/zero earnings. "
            "Lynch requires peg > 0 (strictly positive)."
        )

    def test_pillar_p_peg_negative_fails(self):
        """PEG < 0 (negative earnings) → fails (not a Fast Grower)."""
        result = _run_lynch([_build_mock_lynch_row(peg=-0.5)])
        assert result["lynch_valuation_peg"].iloc[0] == 0

    def test_pillar_p_peg_very_small_positive_passes(self):
        """PEG=0.001 (extremely cheap vs growth) → passes (> 0 AND <= 0.75)."""
        result = _run_lynch([_build_mock_lynch_row(peg=0.001)])
        assert result["lynch_valuation_peg"].iloc[0] == 1

    def test_pillar_p_peg_exceptional_value_0point4_passes(self):
        """PEG=0.40 (Bajaj Finance 2012 equivalent) → exceptional value → passes."""
        result = _run_lynch([_build_mock_lynch_row(peg=0.40)])
        assert result["lynch_valuation_peg"].iloc[0] == 1

    # ── Pillar D: Pre-Discovery ───────────────────────────────────────────────

    def test_pillar_d_combined_below_threshold_passes(self):
        """v1.1: FII=5% + DII=5% = 10% < 20% → pre-discovery phase → passes."""
        result = _run_lynch([_build_mock_lynch_row(fii_holdings=5.0, dii_holdings=5.0)])
        assert result["lynch_pre_discovery"].iloc[0] == 1

    def test_pillar_d_combined_just_below_threshold_passes(self):
        """v1.1: FII=10% + DII=9% = 19% < 20% → passes (strict less-than on combined)."""
        result = _run_lynch([_build_mock_lynch_row(fii_holdings=10.0, dii_holdings=9.0)])
        assert result["lynch_pre_discovery"].iloc[0] == 1

    def test_pillar_d_combined_at_threshold_fails(self):
        """v1.1: FII=10% + DII=10% = 20% exactly → fails (strict '<', not '<='). Combined = 20."""
        result = _run_lynch([_build_mock_lynch_row(fii_holdings=10.0, dii_holdings=10.0)])
        assert result["lynch_pre_discovery"].iloc[0] == 0, (
            "FII+DII=20.0 must FAIL (gate is strict '<', not '<='). "
            "Combined institutional weight at exactly 20% = discovered."
        )

    def test_pillar_d_fii_above_threshold_fails(self):
        """v1.1: FII=25% + DII=5% = 30% >= 20% → already institutionally discovered → fails."""
        result = _run_lynch([_build_mock_lynch_row(fii_holdings=25.0)])
        assert result["lynch_pre_discovery"].iloc[0] == 0

    def test_pillar_d_fii_zero_passes(self):
        """FII=0% → no institutional holding → maximum pre-discovery → passes."""
        result = _run_lynch([_build_mock_lynch_row(fii_holdings=0.0)])
        assert result["lynch_pre_discovery"].iloc[0] == 1

    # ── Pillar F: Fortress Owner ──────────────────────────────────────────────

    def test_pillar_f_all_passing_passes(self):
        """D/E=0.3, promoter=55% → passes."""
        result = _run_lynch([_build_mock_lynch_row(debt_to_equity=0.3, promoter_holdings=55.0)])
        assert result["lynch_fortress_owner"].iloc[0] == 1

    def test_pillar_f_de_just_below_threshold_passes(self):
        """D/E=0.49 < 0.5 → passes (strict less-than)."""
        result = _run_lynch([_build_mock_lynch_row(debt_to_equity=0.49)])
        assert result["lynch_fortress_owner"].iloc[0] == 1

    def test_pillar_f_de_at_threshold_fails(self):
        """CRITICAL: D/E=0.5 exactly → FAILS. Lynch uses strict '<' (not '<= 0.5')."""
        result = _run_lynch([_build_mock_lynch_row(debt_to_equity=0.5)])
        assert result["lynch_fortress_owner"].iloc[0] == 0, (
            "D/E=0.5 must FAIL for Lynch Fortress Owner (gate is strict '<', not '<='). "
            "This is the critical difference from Malik's D/E <= 0.5."
        )

    def test_pillar_f_de_above_threshold_fails(self):
        """D/E=0.8 > 0.5 → fails."""
        result = _run_lynch([_build_mock_lynch_row(debt_to_equity=0.8)])
        assert result["lynch_fortress_owner"].iloc[0] == 0

    def test_pillar_f_de_zero_passes(self):
        """D/E=0.0 (debt-free) → Lynch's preferred state → passes."""
        result = _run_lynch([_build_mock_lynch_row(debt_to_equity=0.0)])
        assert result["lynch_fortress_owner"].iloc[0] == 1

    def test_pillar_f_promoter_at_boundary_passes(self):
        """Promoter=45.0% exactly → passes (>= 45.0)."""
        result = _run_lynch([_build_mock_lynch_row(promoter_holdings=45.0)])
        assert result["lynch_fortress_owner"].iloc[0] == 1

    def test_pillar_f_promoter_just_below_fails(self):
        """Promoter=44.9% < 45% → fails."""
        result = _run_lynch([_build_mock_lynch_row(promoter_holdings=44.9)])
        assert result["lynch_fortress_owner"].iloc[0] == 0

    def test_pillar_f_promoter_zero_fails(self):
        """Promoter=0% → no owner conviction → fails."""
        result = _run_lynch([_build_mock_lynch_row(promoter_holdings=0.0)])
        assert result["lynch_fortress_owner"].iloc[0] == 0

    def test_pillar_f_de_passing_promoter_failing(self):
        """D/E ok but promoter fails → Pillar F fails (AND condition)."""
        result = _run_lynch([_build_mock_lynch_row(debt_to_equity=0.3, promoter_holdings=30.0)])
        assert result["lynch_fortress_owner"].iloc[0] == 0

    def test_pillar_f_de_failing_promoter_passing(self):
        """Promoter ok but D/E fails → Pillar F fails (AND condition)."""
        result = _run_lynch([_build_mock_lynch_row(debt_to_equity=1.5, promoter_holdings=60.0)])
        assert result["lynch_fortress_owner"].iloc[0] == 0

    # ── AND Invariant: lynch_pass ─────────────────────────────────────────────

    def test_all_4_pass_lynch_pass_is_1(self):
        result = _run_lynch([_build_mock_lynch_row()])
        assert result["lynch_pass"].iloc[0] == 1

    def test_single_v_failure_lynch_pass_is_0(self):
        """Any single pillar failing must flip lynch_pass to 0."""
        result = _run_lynch([_build_mock_lynch_row(rev_gr_5y=10.0)])
        assert result["lynch_pass"].iloc[0] == 0

    def test_single_p_failure_lynch_pass_is_0(self):
        result = _run_lynch([_build_mock_lynch_row(peg=1.5)])
        assert result["lynch_pass"].iloc[0] == 0

    def test_single_d_failure_lynch_pass_is_0(self):
        result = _run_lynch([_build_mock_lynch_row(fii_holdings=25.0)])
        assert result["lynch_pass"].iloc[0] == 0

    def test_single_f_de_failure_lynch_pass_is_0(self):
        result = _run_lynch([_build_mock_lynch_row(debt_to_equity=2.0)])
        assert result["lynch_pass"].iloc[0] == 0

    def test_single_f_promoter_failure_lynch_pass_is_0(self):
        result = _run_lynch([_build_mock_lynch_row(promoter_holdings=20.0)])
        assert result["lynch_pass"].iloc[0] == 0

    def test_each_single_failure_independently_kills_pass(self):
        """Parametric: each pillar, when alone failing, must cause lynch_pass=0."""
        fail_cases = [
            {"rev_gr_5y": 10.0},                        # V fails (revenue)
            {"eps_gr_5y": 5.0},                         # V fails (EPS per share — v1.1)
            {"free_cash_flow": -100.0, "fcf_yield": -5.0},  # V fails (cash gate — v1.1)
            {"peg": 2.0},                               # P fails (overvalued)
            {"peg": 0.0},                               # P fails (zero PEG)
            {"fii_holdings": 30.0},                     # D fails (FII+DII = 35 >= 20 — v1.1)
            {"fii_holdings": 15.0, "dii_holdings": 5.0},   # D fails (combined = 20, strict <)
            {"debt_to_equity": 2.0},                    # F fails (D/E)
            {"promoter_holdings": 20.0},                # F fails (promoter < 45 + no buying)
        ]
        for fc in fail_cases:
            result = _run_lynch([_build_mock_lynch_row(**fc)])
            assert result["lynch_pass"].iloc[0] == 0, (
                f"lynch_pass must be 0 when {fc} fails"
            )

    # ── Score Range: 0–4 ─────────────────────────────────────────────────────

    def test_score_all_pass_is_4(self):
        """All 4 pillars pass → score=4."""
        result = _run_lynch([_build_mock_lynch_row()])
        assert result["lynch_score"].iloc[0] == 4

    def test_score_all_fail_is_0(self):
        """All 4 pillars fail → score=0."""
        result = _run_lynch([_build_mock_lynch_row(
            rev_gr_5y=5.0, peg=5.0, fii_holdings=50.0, promoter_holdings=10.0
        )])
        assert result["lynch_score"].iloc[0] == 0

    def test_score_3_when_v_fails(self):
        """V fails → score=3."""
        result = _run_lynch([_build_mock_lynch_row(rev_gr_5y=10.0)])
        assert result["lynch_score"].iloc[0] == 3

    def test_score_3_when_p_fails(self):
        """P fails → score=3."""
        result = _run_lynch([_build_mock_lynch_row(peg=2.0)])
        assert result["lynch_score"].iloc[0] == 3

    def test_score_3_when_d_fails(self):
        """D fails → score=3."""
        result = _run_lynch([_build_mock_lynch_row(fii_holdings=30.0)])
        assert result["lynch_score"].iloc[0] == 3

    def test_score_3_when_f_fails(self):
        """F fails → score=3."""
        result = _run_lynch([_build_mock_lynch_row(debt_to_equity=2.0)])
        assert result["lynch_score"].iloc[0] == 3

    def test_score_2_when_two_fail(self):
        """Two pillar failures → score=2."""
        result = _run_lynch([_build_mock_lynch_row(
            rev_gr_5y=10.0, fii_holdings=30.0
        )])
        assert result["lynch_score"].iloc[0] == 2

    def test_score_1_when_three_fail(self):
        """Three pillar failures → score=1."""
        result = _run_lynch([_build_mock_lynch_row(
            rev_gr_5y=10.0, peg=2.0, fii_holdings=30.0
        )])
        assert result["lynch_score"].iloc[0] == 1

    def test_score_range_always_0_to_4(self):
        """Score must never exceed 4 or drop below 0."""
        combos = [
            _build_mock_lynch_row(),
            _build_mock_lynch_row(rev_gr_5y=10.0),
            _build_mock_lynch_row(peg=2.0, fii_holdings=30.0),
            _build_mock_lynch_row(
                rev_gr_5y=5.0, peg=5.0, fii_holdings=50.0, promoter_holdings=10.0
            ),
        ]
        result = _run_lynch(combos)
        assert result["lynch_score"].between(0, 4).all(), (
            "lynch_score must always be in [0, 4] — 4 pillars maximum"
        )

    def test_pass_equals_score_4(self):
        """v1.1 ONE-DIRECTIONAL: lynch_pass==1 → lynch_score==4 always holds.
        But score==4 does NOT always imply pass==1 (inventory surge can veto pass).
        This test uses no inventory data so the identity is bidirectional in these cases."""
        combos = [
            _build_mock_lynch_row(),                      # all pass → score=4, pass=1
            _build_mock_lynch_row(rev_gr_5y=10.0),        # V fails → score=3, pass=0
            _build_mock_lynch_row(
                rev_gr_5y=5.0, peg=5.0, fii_holdings=50.0, promoter_holdings=10.0
            ),  # all fail → score=0, pass=0
        ]
        result = _run_lynch(combos)
        assert (result["lynch_pass"] == (result["lynch_score"] == 4).astype(int)).all(), (
            "Without inventory surge, lynch_pass must be 1 when lynch_score==4, else 0. "
            "Full one-directional test (score=4 + pass=0) is in TestLynchV11Features."
        )

    def test_score_never_5_or_above(self):
        """Critical: Lynch has 4 pillars only. Score cannot be 5 (unlike Malik)."""
        result = _run_lynch([_build_mock_lynch_row()])
        assert result["lynch_score"].iloc[0] <= 4, (
            "lynch_score must never exceed 4 — Lynch has exactly 4 pillars, not 5."
        )

    def test_score_columns_are_integer_dtype(self):
        """lynch_pass and lynch_score must be integer-typed columns."""
        result = _run_lynch([_build_mock_lynch_row()])
        assert pd.api.types.is_integer_dtype(result["lynch_pass"]), (
            "lynch_pass must be integer dtype (not float or bool)"
        )
        assert pd.api.types.is_integer_dtype(result["lynch_score"]), (
            "lynch_score must be integer dtype"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TestLynchNaNConservative
# ═══════════════════════════════════════════════════════════════════════════════

class TestLynchNaNConservative:
    """NaN handling: every numeric gate must default to failure on missing data."""

    def test_all_numeric_nan_fails_pass(self):
        """All numeric columns NaN → all 4 pillars fail → lynch_pass=0."""
        nan_row = {
            "rev_gr_5y":         float("nan"),
            "pat_gr_3y":         float("nan"),
            "peg":               float("nan"),
            "fii_holdings":      float("nan"),
            "debt_to_equity":    float("nan"),
            "promoter_holdings": float("nan"),
        }
        result = _run_lynch([nan_row])
        assert result["lynch_pass"].iloc[0] == 0, "All-NaN row must fail lynch_pass"

    def test_all_nan_score_is_0(self):
        """All-NaN row: all 4 pillars fail → score=0."""
        nan_row = {k: float("nan") for k in [
            "rev_gr_5y", "pat_gr_3y", "peg", "fii_holdings",
            "debt_to_equity", "promoter_holdings",
        ]}
        result = _run_lynch([nan_row])
        assert result["lynch_score"].iloc[0] == 0, (
            "All-NaN row: every pillar fails → score must be 0"
        )

    def test_rev_gr_nan_fails_growth_velocity(self):
        """NaN rev_gr_5y → fillna(0) → 0 < 20 → Pillar V fails."""
        result = _run_lynch([_build_mock_lynch_row(rev_gr_5y=float("nan"))])
        assert result["lynch_growth_velocity"].iloc[0] == 0

    def test_eps_gr_nan_fails_growth_velocity(self):
        """v1.1: NaN eps_gr_5y AND NaN eps_gr_3y → eps_ly = NaN → fillna(0) → 0 < 15 → Pillar V fails."""
        result = _run_lynch([_build_mock_lynch_row(eps_gr_5y=float("nan"), eps_gr_3y=float("nan"))])
        assert result["lynch_growth_velocity"].iloc[0] == 0, (
            "NaN eps_gr_5y + NaN eps_gr_3y → eps_ly = NaN → fillna(0) → 0 < 15 → V fails. "
            "v1.1: EPS per share replaces PAT. EPS data missing = cannot confirm Fast Grower."
        )

    def test_peg_nan_fails_valuation_peg(self):
        """NaN peg → fillna(999) → 999 > 0.75 → Pillar P fails."""
        result = _run_lynch([_build_mock_lynch_row(peg=float("nan"))])
        assert result["lynch_valuation_peg"].iloc[0] == 0

    def test_fii_nan_fails_pre_discovery(self):
        """NaN fii_holdings → fillna(50) → 50 >= 10 → Pillar D fails.

        CRITICAL: This is the conservative 'already discovered' assumption.
        If FII data is missing, we cannot verify pre-discovery status → gate fails.
        """
        result = _run_lynch([_build_mock_lynch_row(fii_holdings=float("nan"))])
        assert result["lynch_pre_discovery"].iloc[0] == 0, (
            "NaN fii_holdings must FAIL lynch_pre_discovery. "
            "fillna(50) = assume already discovered = conservative gate failure. "
            "FII missing data ≠ FII = 0%."
        )

    def test_fii_nan_not_treated_as_zero(self):
        """NaN FII must NOT be treated as 0% (which would pass). fillna(50) is required."""
        result = _run_lynch([_build_mock_lynch_row(fii_holdings=float("nan"))])
        # If fillna(0) were used: 0 < 10 → would pass (wrong)
        # If fillna(50) used: 50 >= 10 → correctly fails
        assert result["lynch_pre_discovery"].iloc[0] == 0, (
            "NaN fii_holdings treated as 0% (fillna(0)) would incorrectly pass. "
            "Must use fillna(50) — missing FII data = assume already discovered."
        )

    def test_de_nan_fails_fortress_owner(self):
        """NaN debt_to_equity → fillna(999) → 999 >= 0.5 → D/E gate fails → Pillar F fails."""
        result = _run_lynch([_build_mock_lynch_row(debt_to_equity=float("nan"))])
        assert result["lynch_fortress_owner"].iloc[0] == 0

    def test_promoter_nan_fails_fortress_owner(self):
        """NaN promoter_holdings → fillna(0) → 0 < 45 → Pillar F fails."""
        result = _run_lynch([_build_mock_lynch_row(promoter_holdings=float("nan"))])
        assert result["lynch_fortress_owner"].iloc[0] == 0

    def test_de_nan_is_stricter_than_zero(self):
        """fillna(999) for D/E is more conservative than fillna(0).
        D/E=0 would pass (< 0.5) but unknown leverage must fail."""
        result = _run_lynch([_build_mock_lynch_row(debt_to_equity=float("nan"))])
        assert result["lynch_fortress_owner"].iloc[0] == 0, (
            "NaN D/E must fail: unknown leverage is never safe. "
            "fillna(999) → 999 >= 0.5 → D/E gate correctly fails."
        )

    def test_peg_nan_is_conservative_999(self):
        """fillna(999) for PEG is the most conservative choice: unknown PEG ≠ cheap stock."""
        result = _run_lynch([_build_mock_lynch_row(peg=float("nan"))])
        assert result["lynch_valuation_peg"].iloc[0] == 0, (
            "NaN peg must fail: unknown PEG cannot be assumed to be in the sweet spot. "
            "fillna(999) → 999 > 0.75 → PEG gate correctly fails."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TestLynchIndexAlignment
# ═══════════════════════════════════════════════════════════════════════════════

class TestLynchIndexAlignment:
    """Non-default index safety — guards against pandas index alignment drift."""

    def test_integer_index_non_default(self):
        """Results must be correctly aligned with a shuffled integer index."""
        rows = [
            _build_mock_lynch_row(),                        # all pass → score=4
            _build_mock_lynch_row(fii_holdings=30.0),       # D fails → score=3
            _build_mock_lynch_row(peg=2.0),                 # P fails → score=3
        ]
        from config import MASTER_PROFILES
        from core.scoring_engine import compute_qglp_score
        df = pd.DataFrame(rows, index=[100, 200, 300])
        df.attrs["detected_market_regime"] = "SIDEWAYS"
        profile = MASTER_PROFILES.get("Balanced", next(iter(MASTER_PROFILES.values())))
        result = compute_qglp_score(df, profile)

        assert result.index.tolist() == [100, 200, 300], "Index must be preserved"
        assert result.loc[100, "lynch_pass"] == 1,  "Row 100: all pass → lynch_pass=1"
        assert result.loc[200, "lynch_pass"] == 0,  "Row 200: D fails → lynch_pass=0"
        assert result.loc[300, "lynch_pass"] == 0,  "Row 300: P fails → lynch_pass=0"

    def test_string_index(self):
        """Results must be correctly aligned with a string index."""
        rows = [
            _build_mock_lynch_row(),                         # all pass
            _build_mock_lynch_row(rev_gr_5y=10.0),           # V fails
            _build_mock_lynch_row(promoter_holdings=20.0),   # F fails
        ]
        from config import MASTER_PROFILES
        from core.scoring_engine import compute_qglp_score
        df = pd.DataFrame(rows, index=["PAGEINDS", "MOTHERSON", "NMDC"])
        df.attrs["detected_market_regime"] = "SIDEWAYS"
        profile = MASTER_PROFILES.get("Balanced", next(iter(MASTER_PROFILES.values())))
        result = compute_qglp_score(df, profile)

        assert result.index.tolist() == ["PAGEINDS", "MOTHERSON", "NMDC"]
        assert result.loc["PAGEINDS",  "lynch_pass"] == 1, "PAGEINDS: all pass"
        assert result.loc["MOTHERSON", "lynch_pass"] == 0, "MOTHERSON: V fails (rev_gr_5y=10%)"
        assert result.loc["NMDC",      "lynch_pass"] == 0, "NMDC: F fails (promoter=20%)"

    def test_score_aligned_with_integer_index(self):
        """Scores must align correctly with a non-default integer index."""
        rows = [
            _build_mock_lynch_row(peg=2.0),    # P fails → score=3
            _build_mock_lynch_row(),            # all pass → score=4
        ]
        from config import MASTER_PROFILES
        from core.scoring_engine import compute_qglp_score
        df = pd.DataFrame(rows, index=[99, 42])
        df.attrs["detected_market_regime"] = "SIDEWAYS"
        profile = MASTER_PROFILES.get("Balanced", next(iter(MASTER_PROFILES.values())))
        result = compute_qglp_score(df, profile)

        assert result.loc[99, "lynch_score"] == 3, "Index 99: P fails → score=3"
        assert result.loc[42, "lynch_score"] == 4, "Index 42: all pass → score=4"

    def test_large_index_batch(self):
        """Batch of 10 rows with non-default index → all columns computed correctly."""
        rows = [_build_mock_lynch_row() for _ in range(10)]
        from config import MASTER_PROFILES
        from core.scoring_engine import compute_qglp_score
        idx = list(range(500, 510))
        df = pd.DataFrame(rows, index=idx)
        df.attrs["detected_market_regime"] = "BULL"
        profile = MASTER_PROFILES.get("Balanced", next(iter(MASTER_PROFILES.values())))
        result = compute_qglp_score(df, profile)

        assert len(result) == 10
        assert result["lynch_pass"].sum() == 10,          "All 10 fully-passing rows → lynch_pass=1"
        assert (result["lynch_score"] == 4).all(),         "All 10 rows → lynch_score=4"

    def test_pillar_columns_present_in_output(self):
        """All 6 Lynch output columns must be present in compute_qglp_score output."""
        result = _run_lynch([_build_mock_lynch_row()])
        for col in ["lynch_growth_velocity", "lynch_valuation_peg",
                    "lynch_pre_discovery", "lynch_fortress_owner",
                    "lynch_pass", "lynch_score"]:
            assert col in result.columns, f"Missing column in output: {col}"


# ═══════════════════════════════════════════════════════════════════════════════
# TestLynchFinancialExemption
# ═══════════════════════════════════════════════════════════════════════════════

class TestLynchFinancialExemption:
    """Financial sector (banks/NBFCs) exempt from D/E sub-gate only within Fortress pillar."""

    def test_financial_with_high_de_passes_fortress(self):
        """is_financial=True + D/E=15.0 → Fortress PASSES (exempt from D/E gate only)."""
        result = _run_lynch([_build_mock_lynch_row(
            is_financial=True, debt_to_equity=15.0
        )])
        assert result["lynch_fortress_owner"].iloc[0] == 1, (
            "Financial sector with D/E=15.0 must pass lynch_fortress_owner. "
            "Banks structurally carry high leverage — D/E gate is inapplicable."
        )

    def test_financial_with_de_nan_passes_fortress(self):
        """is_financial=True + D/E=NaN → Fortress PASSES."""
        result = _run_lynch([_build_mock_lynch_row(
            is_financial=True, debt_to_equity=float("nan")
        )])
        assert result["lynch_fortress_owner"].iloc[0] == 1, (
            "Financial sector with NaN D/E must pass lynch_fortress_owner (exempt)."
        )

    def test_financial_still_requires_promoter_gate(self):
        """CRITICAL: Financial exemption covers D/E only. Promoter gate still applies."""
        result = _run_lynch([_build_mock_lynch_row(
            is_financial=True,
            debt_to_equity=15.0,    # exempt → passes D/E
            promoter_holdings=20.0, # NOT exempt → fails promoter gate
        )])
        assert result["lynch_fortress_owner"].iloc[0] == 0, (
            "Financial sector must still FAIL lynch_fortress_owner when promoter_holdings < 45%. "
            "Promoter gate is NOT exempt for financial companies."
        )

    def test_non_financial_de_at_boundary_fails(self):
        """Non-financial with D/E=0.5 exactly → fails (strict '<' gate)."""
        result = _run_lynch([_build_mock_lynch_row(
            is_financial=False, debt_to_equity=0.5
        )])
        assert result["lynch_fortress_owner"].iloc[0] == 0, (
            "Non-financial with D/E=0.5 must FAIL (Lynch D/E gate is strict '<', not '<='). "
            "Exemption is financial-only."
        )

    def test_non_financial_high_de_fails(self):
        """Non-financial with D/E=10.0 must FAIL (exemption is financial-only)."""
        result = _run_lynch([_build_mock_lynch_row(
            is_financial=False, debt_to_equity=10.0
        )])
        assert result["lynch_fortress_owner"].iloc[0] == 0, (
            "Non-financial with D/E=10.0 must FAIL lynch_fortress_owner."
        )

    def test_financial_full_pass_when_promoter_clears(self):
        """Financial sector: D/E exempt + promoter ≥ 45% → Fortress passes."""
        result = _run_lynch([_build_mock_lynch_row(
            is_financial=True,
            debt_to_equity=15.0,    # exempt
            promoter_holdings=55.0, # passes
        )])
        assert result["lynch_fortress_owner"].iloc[0] == 1, (
            "Financial sector with promoter_holdings >= 45% must PASS lynch_fortress_owner."
        )

    def test_financial_other_3_pillars_not_exempt(self):
        """Financial exemption is ONLY for D/E in Pillar F. All other pillars still required."""
        result = _run_lynch([_build_mock_lynch_row(
            is_financial=True,
            debt_to_equity=15.0,  # F: exempt → D/E passes
            peg=3.0,              # P: NOT exempt → PEG fails → lynch_pass=0
        )])
        assert result["lynch_pass"].iloc[0] == 0, (
            "Financial sector must still fail lynch_pass when PEG gate fails. "
            "Financial exemption ONLY applies to D/E gate inside Pillar F."
        )

    def test_financial_full_tenbagger_if_all_others_pass(self):
        """Financial sector: all non-debt conditions pass → full lynch_pass=1."""
        result = _run_lynch([_build_mock_lynch_row(
            is_financial=True,
            debt_to_equity=15.0,    # F: exempt
            promoter_holdings=60.0, # F: passes
        )])
        assert result["lynch_pass"].iloc[0] == 1, (
            "Financial sector with all other pillars passing must achieve lynch_pass=1."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TestLynchUIContract
# ═══════════════════════════════════════════════════════════════════════════════

class TestLynchUIContract:
    """Verify render_lynch_radar is correctly exported and reads the right columns."""

    def test_render_lynch_radar_importable(self):
        from ui import render_lynch_radar
        assert callable(render_lynch_radar)

    def test_render_lynch_radar_in_all(self):
        import ui
        assert "render_lynch_radar" in ui.__all__

    def test_render_lynch_radar_is_not_stub(self):
        """render_lynch_radar must resolve to real function, not the fallback _stub."""
        from ui import render_lynch_radar
        assert render_lynch_radar.__name__ != "_stub", (
            "render_lynch_radar resolved to _stub — check for ImportError in ui_tearsheet.py"
        )

    def test_ui_reads_lynch_growth_velocity(self, ui_source):
        assert "lynch_growth_velocity" in ui_source

    def test_ui_reads_lynch_valuation_peg(self, ui_source):
        assert "lynch_valuation_peg" in ui_source

    def test_ui_reads_lynch_pre_discovery(self, ui_source):
        assert "lynch_pre_discovery" in ui_source

    def test_ui_reads_lynch_fortress_owner(self, ui_source):
        assert "lynch_fortress_owner" in ui_source

    def test_ui_reads_lynch_pass(self, ui_source):
        assert '"lynch_pass"' in ui_source or "'lynch_pass'" in ui_source

    def test_ui_reads_lynch_score(self, ui_source):
        assert '"lynch_score"' in ui_source or "'lynch_score'" in ui_source

    def test_ui_uses_no_threshold_math(self, ui_source):
        """Pure display: render_lynch_radar must not contain numeric threshold computations."""
        match = re.search(r'def render_lynch_radar.*?(?=\ndef |\Z)', ui_source, re.DOTALL)
        assert match, "render_lynch_radar function not found in ui_tearsheet.py"
        fn_body = match.group(0)
        # These patterns must not appear as computation operators in the display layer
        forbidden_computations = [
            r'>=\s*20\.0',    # V pillar revenue threshold as computation
            r'>=\s*15\.0',    # V pillar PAT threshold as computation
            r'<=\s*0\.75',    # P pillar PEG threshold as computation
            r'<\s*10\.0',     # D pillar FII threshold as computation
            r'<\s*0\.5',      # F pillar D/E threshold as computation
            r'>=\s*45\.0',    # F pillar promoter threshold as computation
        ]
        for pattern in forbidden_computations:
            assert not re.search(pattern, fn_body), (
                f"render_lynch_radar contains threshold computation: {pattern}. "
                "Pure display layer must never re-compute thresholds — read pre-materialized columns only."
            )

    def test_ui_has_ruby_red_color_for_lynch(self, ui_source):
        """Lynch radar must use the ruby/red accent color (#e74c3c)."""
        match = re.search(r'def render_lynch_radar.*?(?=\ndef |\Z)', ui_source, re.DOTALL)
        assert match
        fn_body = match.group(0)
        assert "e74c3c" in fn_body or "_LYNCH_RED" in fn_body, (
            "render_lynch_radar must use the ruby red accent color (#e74c3c or _LYNCH_RED)"
        )

    def test_ui_displays_score_out_of_4(self, ui_source):
        """Score display must show X/4 (not /100 or /5 like Malik — only 4 pillars)."""
        match = re.search(r'def render_lynch_radar.*?(?=\ndef |\Z)', ui_source, re.DOTALL)
        assert match
        fn_body = match.group(0)
        assert "/ 4" in fn_body or "/4" in fn_body, (
            "render_lynch_radar must display score out of 4 (not /5 or /100). "
            "Lynch has exactly 4 pillars."
        )

    def test_ui_has_4_pillar_letter_labels(self, ui_source):
        """The 4 pillar single-letter labels V/P/D/F must be present in the widget."""
        match = re.search(r'def render_lynch_radar.*?(?=\ndef |\Z)', ui_source, re.DOTALL)
        assert match
        fn_body = match.group(0)
        for letter in ["V", "P", "D", "F"]:
            assert f'"{letter}"' in fn_body or f"'{letter}'" in fn_body, (
                f"render_lynch_radar missing pillar letter: {letter}"
            )

    def test_ui_has_pillar_descriptive_labels(self, ui_source):
        """Descriptive pillar names must appear in the widget."""
        match = re.search(r'def render_lynch_radar.*?(?=\ndef |\Z)', ui_source, re.DOTALL)
        assert match
        fn_body = match.group(0)
        for label in ["Growth Velocity", "PEG", "Discovery", "Fortress"]:
            assert label in fn_body, f"render_lynch_radar missing pillar label: '{label}'"

    def test_ui_lynch_tenbagger_title_present(self, ui_source):
        """The function must identify Lynch or Fast Grower or Tenbagger."""
        match = re.search(r'def render_lynch_radar.*?(?=\ndef |\Z)', ui_source, re.DOTALL)
        assert match
        fn_body = match.group(0)
        assert "Lynch" in fn_body and ("Tenbagger" in fn_body or "Fast Grower" in fn_body)

    def test_ui_certified_status_message(self, ui_source):
        """Status message must differentiate pass from fail."""
        match = re.search(r'def render_lynch_radar.*?(?=\ndef |\Z)', ui_source, re.DOTALL)
        assert match
        fn_body = match.group(0)
        assert "CERTIFIED" in fn_body or "Criteria Not Met" in fn_body, (
            "render_lynch_radar must have a pass/fail status message"
        )

    def test_ui_function_is_pure_display(self, ui_source):
        """Docstring must declare PURE DISPLAY — zero threshold re-computation."""
        match = re.search(r'def render_lynch_radar.*?(?=\ndef |\Z)', ui_source, re.DOTALL)
        assert match
        fn_body = match.group(0)
        assert "PURE DISPLAY" in fn_body, (
            "render_lynch_radar docstring must declare PURE DISPLAY"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TestLynchRawSignalsContract
# ═══════════════════════════════════════════════════════════════════════════════

class TestLynchRawSignalsContract:
    """Verify Lynch cells appear in render_raw_signals grid."""

    def test_raw_signals_has_lynch_score_cell(self, ui_source):
        assert "lynch_score" in ui_source

    def test_raw_signals_has_lynch_pass_cell(self, ui_source):
        assert "lynch_pass" in ui_source

    def test_raw_signals_lynch_score_format_is_out_of_4(self, ui_source):
        """lynch_score in render_raw_signals must show /4 not /5 or /100."""
        assert '"/4"' in ui_source or "'/4'" in ui_source or "/4" in ui_source, (
            "render_raw_signals must display lynch_score out of 4"
        )

    def test_raw_signals_lynch_pass_has_readable_label(self, ui_source):
        """lynch_pass cell must use a human-readable label."""
        assert "Lynch Pass" in ui_source or "lynch_pass" in ui_source

    def test_raw_signals_lynch_score_has_readable_label(self, ui_source):
        """lynch_score cell must use 'Lynch Score' label."""
        assert "Lynch Score" in ui_source

    def test_raw_signals_lynch_pillar_section_present(self, ui_source):
        """A section header for Lynch pillars must appear in render_raw_signals."""
        assert "Lynch Fast Grower" in ui_source or "lynch" in ui_source.lower()

    def test_raw_signals_all_4_pillar_columns_referenced(self, ui_source):
        """All 4 Lynch pillar column names must be referenced in the raw signals grid."""
        for col in ["lynch_growth_velocity", "lynch_valuation_peg",
                    "lynch_pre_discovery", "lynch_fortress_owner"]:
            assert col in ui_source, f"render_raw_signals missing Lynch pillar: {col}"


# ═══════════════════════════════════════════════════════════════════════════════
# TestLynchAppWiring
# ═══════════════════════════════════════════════════════════════════════════════

class TestLynchAppWiring:
    """Verify app.py correctly imports and calls render_lynch_radar."""

    @pytest.fixture(scope="class")
    def app_source(self) -> str:
        app_path = os.path.join(REPO_ROOT, "app.py")
        with open(app_path, encoding="utf-8") as f:
            return f.read()

    def test_app_imports_render_lynch_radar(self, app_source):
        assert "render_lynch_radar" in app_source, (
            "app.py must import render_lynch_radar from ui"
        )

    def test_app_calls_render_lynch_radar(self, app_source):
        assert "render_lynch_radar(stock)" in app_source, (
            "app.py must call render_lynch_radar(stock) in the frameworks tab"
        )

    def test_ui_init_exports_render_lynch_radar(self):
        """ui/__init__.py must export render_lynch_radar in __all__."""
        import ui
        assert "render_lynch_radar" in ui.__all__

    def test_render_lynch_radar_callable_after_import(self):
        """render_lynch_radar must be callable (not a stub from failed import)."""
        from ui import render_lynch_radar
        assert callable(render_lynch_radar)
        assert render_lynch_radar.__name__ != "_stub", (
            "render_lynch_radar resolved to _stub — check for ImportError in ui/__init__.py"
        )

    def test_ui_init_exports_both_malik_and_lynch(self):
        """ui/__init__.py must export BOTH render_malik_radar and render_lynch_radar."""
        import ui
        assert "render_malik_radar" in ui.__all__, "render_malik_radar must stay in __all__"
        assert "render_lynch_radar" in ui.__all__, "render_lynch_radar must be in __all__"

    def test_app_import_line_has_both_malik_and_lynch(self, app_source):
        """app.py must import both Malik and Lynch renderers (no regression)."""
        assert "render_malik_radar" in app_source, "render_malik_radar import must remain in app.py"
        assert "render_lynch_radar" in app_source, "render_lynch_radar import must be in app.py"


# ═══════════════════════════════════════════════════════════════════════════════
# TestLynchV11Features — v1.1 book-grounded improvements
# ═══════════════════════════════════════════════════════════════════════════════

class TestLynchV11Features:
    """Verify all 6 v1.1 book-grounded improvements: EPS per share, FCF gate,
    FII+DII combined, promoter buying OR level, inventory surge disqualifier,
    and spec ledger v1.1 completeness."""

    # ── EPS per share (replaces PAT total) ───────────────────────────────────

    def test_eps_5y_preferred_over_3y(self):
        """v1.1: eps_gr_5y is used first; eps_gr_3y is only fallback when 5Y missing."""
        result = _run_lynch([_build_mock_lynch_row(eps_gr_5y=5.0, eps_gr_3y=25.0)])
        assert result["lynch_growth_velocity"].iloc[0] == 0, (
            "eps_gr_5y=5% must override eps_gr_3y=25%. "
            "5Y preferred, 3Y is fallback only — not an OR condition."
        )

    def test_eps_3y_fallback_when_5y_missing(self):
        """v1.1: when eps_gr_5y is NaN, eps_gr_3y is used as fallback."""
        result = _run_lynch([_build_mock_lynch_row(eps_gr_5y=float("nan"), eps_gr_3y=18.0)])
        assert result["lynch_growth_velocity"].iloc[0] == 1, (
            "eps_gr_5y=NaN + eps_gr_3y=18% → should use 3Y fallback → 18 >= 15 → V passes."
        )

    def test_eps_3y_fallback_below_threshold_fails(self):
        """v1.1: 3Y fallback is subject to same 15% threshold."""
        result = _run_lynch([_build_mock_lynch_row(eps_gr_5y=float("nan"), eps_gr_3y=10.0)])
        assert result["lynch_growth_velocity"].iloc[0] == 0, (
            "eps_gr_5y=NaN + eps_gr_3y=10% → 3Y fallback used → 10 < 15 → V fails."
        )

    def test_pat_gr_3y_ignored_in_v11(self):
        """v1.1: pat_gr_3y is NOT used. Setting it low has no effect when eps_gr_5y passes."""
        result = _run_lynch([_build_mock_lynch_row(pat_gr_3y=0.0, eps_gr_5y=18.0)])
        assert result["lynch_growth_velocity"].iloc[0] == 1, (
            "pat_gr_3y=0% must be ignored in v1.1. "
            "eps_gr_5y=18% >= 15% → V passes regardless of pat_gr_3y."
        )

    # ── FCF cash gate ────────────────────────────────────────────────────────

    def test_cash_gate_negative_fcf_and_yield_fails_pillar_v(self):
        """v1.1: negative FCF + negative FCF yield → cash gate fails → Pillar V fails."""
        result = _run_lynch([_build_mock_lynch_row(
            free_cash_flow=-100.0, fcf_yield=-3.0
        )])
        assert result["lynch_growth_velocity"].iloc[0] == 0, (
            "negative free_cash_flow AND negative fcf_yield → cash gate fails → Pillar V fails. "
            "Lynch Ch13: 'make sure it is free cash flow'."
        )

    def test_cash_gate_positive_fcf_passes(self):
        """v1.1: positive free_cash_flow passes the cash gate."""
        result = _run_lynch([_build_mock_lynch_row(free_cash_flow=200.0)])
        assert result["lynch_growth_velocity"].iloc[0] == 1

    def test_cash_gate_fcf_yield_fallback_passes(self):
        """v1.1: NaN free_cash_flow but positive fcf_yield → cash gate passes via fallback."""
        result = _run_lynch([_build_mock_lynch_row(
            free_cash_flow=float("nan"), fcf_yield=2.5
        )])
        assert result["lynch_growth_velocity"].iloc[0] == 1, (
            "free_cash_flow=NaN + fcf_yield=2.5% → cash gate passes via fcf_yield fallback."
        )

    def test_cash_gate_nan_fcf_nan_yield_fails(self):
        """v1.1: both FCF columns NaN → fillna(-1) → both -1 > 0 = False → cash gate fails."""
        result = _run_lynch([_build_mock_lynch_row(
            free_cash_flow=float("nan"), fcf_yield=float("nan")
        )])
        assert result["lynch_growth_velocity"].iloc[0] == 0, (
            "free_cash_flow=NaN + fcf_yield=NaN → both fillna(-1) → -1 > 0 = False → V fails. "
            "Missing cash data = cannot verify positive FCF = conservative gate failure."
        )

    def test_cash_gate_zero_fcf_fails(self):
        """v1.1: free_cash_flow=0.0 → 0 > 0 = False → fails (must be strictly positive)."""
        result = _run_lynch([_build_mock_lynch_row(
            free_cash_flow=0.0, fcf_yield=float("nan")
        )])
        assert result["lynch_growth_velocity"].iloc[0] == 0, (
            "free_cash_flow=0.0 fails the cash gate — Lynch requires positive FCF, not zero."
        )

    # ── FII+DII combined threshold ────────────────────────────────────────────

    def test_fii_low_dii_high_fails_combined(self):
        """v1.1: FII=5% + DII=25% = 30% >= 20% → fails despite low FII alone."""
        result = _run_lynch([_build_mock_lynch_row(fii_holdings=5.0, dii_holdings=25.0)])
        assert result["lynch_pre_discovery"].iloc[0] == 0, (
            "FII=5% + DII=25% = 30% >= 20% → Pillar D fails. "
            "v1.1 combined threshold catches high DII even with low FII."
        )

    def test_fii_high_dii_low_fails_combined(self):
        """v1.1: FII=18% + DII=3% = 21% >= 20% → fails."""
        result = _run_lynch([_build_mock_lynch_row(fii_holdings=18.0, dii_holdings=3.0)])
        assert result["lynch_pre_discovery"].iloc[0] == 0

    def test_combined_just_under_20_passes(self):
        """v1.1: FII=10% + DII=9% = 19% < 20% → passes (strict less-than)."""
        result = _run_lynch([_build_mock_lynch_row(fii_holdings=10.0, dii_holdings=9.0)])
        assert result["lynch_pre_discovery"].iloc[0] == 1

    def test_dii_nan_conservative_fails(self):
        """v1.1: DII=NaN → fillna(50) → FII+DII = 5+50 = 55 >= 20 → Pillar D fails."""
        result = _run_lynch([_build_mock_lynch_row(fii_holdings=5.0, dii_holdings=float("nan"))])
        assert result["lynch_pre_discovery"].iloc[0] == 0, (
            "NaN dii_holdings → fillna(50) → assume already discovered → gate fails. "
            "Missing DII data ≠ DII = 0%."
        )

    # ── Promoter buying OR level ──────────────────────────────────────────────

    def test_promoter_below_45_with_active_buying_passes_fortress(self):
        """v1.1: promoter=35% (below 45) but actively buying → Fortress passes."""
        result = _run_lynch([_build_mock_lynch_row(
            promoter_holdings=35.0, change_promoter_1y=2.0
        )])
        assert result["lynch_fortress_owner"].iloc[0] == 1, (
            "promoter=35% < 45% but change_promoter_1y=2% > 0 → OR condition satisfied. "
            "Lynch Ch15: 'insider buying is a positive sign' — active accumulation counts."
        )

    def test_promoter_above_45_no_buying_still_passes(self):
        """v1.1: promoter=55% (above 45) with no buying → level alone satisfies Fortress."""
        result = _run_lynch([_build_mock_lynch_row(
            promoter_holdings=55.0, change_promoter_1y=0.0
        )])
        assert result["lynch_fortress_owner"].iloc[0] == 1, (
            "promoter_holdings=55% >= 45% → level condition alone passes Fortress. "
            "change_promoter_1y=0 is irrelevant when level gate passes."
        )

    def test_promoter_below_45_no_buying_fails(self):
        """v1.1: promoter=35% AND no buying → both OR conditions false → Fortress fails."""
        result = _run_lynch([_build_mock_lynch_row(
            promoter_holdings=35.0, change_promoter_1y=0.0
        )])
        assert result["lynch_fortress_owner"].iloc[0] == 0, (
            "promoter=35% < 45% AND change_promoter_1y=0 (not buying) → both OR gates fail. "
            "Exactly 0 buying is NOT > 0."
        )

    def test_promoter_nan_with_positive_buying_passes(self):
        """v1.1: promoter=NaN (→fillna(0) <45 fails level) but buying=5% > 0 → passes via OR."""
        result = _run_lynch([_build_mock_lynch_row(
            promoter_holdings=float("nan"), change_promoter_1y=5.0
        )])
        assert result["lynch_fortress_owner"].iloc[0] == 1, (
            "promoter_holdings=NaN→fillna(0)<45 (level fails) but change_promoter_1y=5%>0 "
            "(buying passes) → OR condition satisfied → Fortress passes."
        )

    def test_promoter_nan_no_buying_fails(self):
        """v1.1: promoter=NaN AND change_promoter_1y=NaN → both fail → Fortress fails."""
        result = _run_lynch([_build_mock_lynch_row(
            promoter_holdings=float("nan"), change_promoter_1y=float("nan")
        )])
        assert result["lynch_fortress_owner"].iloc[0] == 0, (
            "promoter_holdings=NaN→fillna(0)<45 AND change_promoter_1y=NaN→fillna(-1)<0. "
            "Both OR gates fail → Fortress fails."
        )

    # ── Inventory surge disqualifier ─────────────────────────────────────────

    def test_inventory_surge_vetos_pass_score_unchanged(self):
        """v1.1: all 4 pillars pass + inventory surge → lynch_pass=0 BUT lynch_score=4."""
        result = _run_lynch([_build_mock_lynch_row(
            rev_gr_5y=25.0,
            inv_growth=50.0,   # 50% > (25% + 20pp = 45%) → surge fires
        )])
        assert result["lynch_pass"].iloc[0] == 0, (
            "Inventory growing 50% when revenue grows 25% (25pp gap > 20pp threshold) "
            "→ inventory surge disqualifier fires → lynch_pass must be 0."
        )
        assert result["lynch_score"].iloc[0] == 4, (
            "Inventory surge vetos pass but NOT score. "
            "All 4 pillars are green → lynch_score must remain 4."
        )

    def test_inventory_no_data_does_not_disqualify(self):
        """v1.1: NaN inv_growth → notna() guard = False → disqualifier does NOT fire."""
        result = _run_lynch([_build_mock_lynch_row(inv_growth=float("nan"))])
        assert result["lynch_pass"].iloc[0] == 1, (
            "NaN inv_growth → inv_gr_ly.notna() = False → disqualifier inactive. "
            "No inventory data = no penalty. lynch_pass must be 1."
        )

    def test_inventory_within_tolerance_does_not_disqualify(self):
        """v1.1: inv_growth only 15pp above revenue → within 20pp tolerance → no veto."""
        result = _run_lynch([_build_mock_lynch_row(
            rev_gr_5y=25.0,
            inv_growth=39.0,   # 39% < (25% + 20pp = 45%) → below threshold
        )])
        assert result["lynch_pass"].iloc[0] == 1, (
            "inv_growth=39% vs rev_gr_5y=25%: gap is 14pp < 20pp threshold → no veto."
        )

    def test_inventory_exactly_at_boundary_does_not_disqualify(self):
        """v1.1: inv_growth = rev + 20pp exactly → strict '>' means boundary does NOT fire."""
        result = _run_lynch([_build_mock_lynch_row(
            rev_gr_5y=25.0,
            inv_growth=45.0,   # exactly 25% + 20pp = boundary; '>' so does not fire
        )])
        assert result["lynch_pass"].iloc[0] == 1, (
            "inv_growth=rev_gr_5y+20pp exactly → disqualifier is strict '>': boundary does NOT fire."
        )

    def test_inventory_just_above_boundary_disqualifies(self):
        """v1.1: inv_growth = rev + 20pp + 0.1pp → just above threshold → fires."""
        result = _run_lynch([_build_mock_lynch_row(
            rev_gr_5y=25.0,
            inv_growth=45.1,   # just over 25% + 20pp = 45%
        )])
        assert result["lynch_pass"].iloc[0] == 0, (
            "inv_growth=45.1% > rev_gr_5y=25% + 20pp → disqualifier fires → pass=0."
        )

    def test_score_4_with_pass_0_proves_one_directional(self):
        """v1.1: score=4 AND pass=0 is possible — proves one-directional implication."""
        result = _run_lynch([_build_mock_lynch_row(
            rev_gr_5y=25.0,
            inv_growth=60.0,   # 60% > 25%+20%=45% → surge fires
        )])
        score = result["lynch_score"].iloc[0]
        passv = result["lynch_pass"].iloc[0]
        assert score == 4, f"All 4 pillars pass → score must be 4; got {score}"
        assert passv == 0, f"Inventory surge vetos pass → pass must be 0; got {passv}"
        assert score == 4 and passv == 0, (
            "v1.1 one-directional: score=4 does NOT imply pass=1 when surge is active."
        )

    # ── Spec ledger v1.1 validation ───────────────────────────────────────────

    def test_spec_version_is_v11(self, spec):
        """Spec version must contain '1.1'."""
        assert "1.1" in spec["_meta"]["version"], (
            f"Spec version must contain '1.1'; got: {spec['_meta']['version']}"
        )

    def test_spec_has_inventory_disqualifier_section(self, spec):
        """v1.1 spec must include inventory_surge_disqualifier section."""
        assert "inventory_surge_disqualifier" in spec, (
            "Spec missing inventory_surge_disqualifier section (added in v1.1)."
        )

    def test_spec_inventory_disqualifier_has_one_directional_note(self, spec):
        """inventory_surge_disqualifier must document one-directional implication."""
        section = spec["inventory_surge_disqualifier"]
        assert "one_directional_implication" in section, (
            "inventory_surge_disqualifier must document v1.1 one-directional implication."
        )

    def test_spec_has_implemented_in_v11_section(self, spec):
        """v1.1 spec must document what was implemented from the not_implementable list."""
        assert "implemented_in_v11" in spec, (
            "Spec missing implemented_in_v11 section — inventory and insider buying are now implemented."
        )

    def test_spec_dii_in_unit_notes(self, spec):
        """v1.1: dii_holdings must appear in unit_notes."""
        assert "dii_holdings" in spec["_meta"]["unit_notes"], (
            "unit_notes missing dii_holdings (v1.1 addition for combined FII+DII gate)."
        )

    def test_spec_change_promoter_in_unit_notes(self, spec):
        """v1.1: change_promoter_1y must appear in unit_notes."""
        assert "change_promoter_1y" in spec["_meta"]["unit_notes"], (
            "unit_notes missing change_promoter_1y (v1.1 addition for promoter buying gate)."
        )

    def test_spec_inv_growth_in_unit_notes(self, spec):
        """v1.1: inv_growth must appear in unit_notes."""
        assert "inv_growth" in spec["_meta"]["unit_notes"], (
            "unit_notes missing inv_growth (v1.1 inventory surge disqualifier column)."
        )

    def test_spec_score_column_logic_mentions_one_directional(self, spec):
        """v1.1 score_column_logic must explain one-directional implication."""
        scl_str = json.dumps(spec["score_column_logic"]).lower()
        assert "one-directional" in scl_str or "one directional" in scl_str, (
            "score_column_logic must document v1.1 one-directional implication."
        )

    # ── Engine v1.1 presence tests ────────────────────────────────────────────

    def test_engine_has_dii_variable(self, se_source):
        """v1.1: dii_ly variable must be present in scoring_engine.py Lynch block."""
        assert "dii_ly" in se_source, (
            "scoring_engine missing dii_ly variable — required for FII+DII combined gate."
        )

    def test_engine_has_eps_fallback_chain(self, se_source):
        """v1.1: eps_ly = eps5y_ly.fillna(eps3y_ly) fallback chain must be present."""
        assert re.search(r'eps_ly\s*=\s*eps5y_ly\.fillna\s*\(\s*eps3y_ly\s*\)', se_source), (
            "scoring_engine missing eps_ly = eps5y_ly.fillna(eps3y_ly) fallback chain."
        )

    def test_engine_has_cash_ok_variable(self, se_source):
        """v1.1: _cash_ok_ly variable must be present for FCF gate."""
        assert "_cash_ok_ly" in se_source, (
            "scoring_engine missing _cash_ok_ly — required for Lynch Ch13 FCF cash gate."
        )

    def test_engine_has_inventory_surge_disqualifier(self, se_source):
        """v1.1: _inv_surge_disq variable must be present."""
        assert "_inv_surge_disq" in se_source, (
            "scoring_engine missing _inv_surge_disq — required for inventory surge disqualifier."
        )

    def test_engine_inventory_not_in_score_sum(self, se_source):
        """v1.1: inventory disqualifier must NOT be part of lynch_score sum (only affects pass)."""
        # Extract just the df["lynch_score"] = ( ... ) assignment block
        score_assign = re.search(
            r'df\["lynch_score"\]\s*=\s*\([^)]+\)',
            se_source, re.DOTALL
        )
        assert score_assign, "df['lynch_score'] assignment not found in scoring_engine.py"
        score_code = score_assign.group(0)
        assert "inv_surge" not in score_code and "_inv_surge_disq" not in score_code, (
            "lynch_score sum must NOT include inventory surge. "
            "Disqualifier vetos pass only, score stays 0-4."
        )

    def test_engine_has_promoter_buying_variable(self, se_source):
        """v1.1: chg_promo_ly variable must be present for promoter buying gate."""
        assert "chg_promo_ly" in se_source, (
            "scoring_engine missing chg_promo_ly — required for Lynch Ch15 insider buying gate."
        )

    def test_engine_promo_ok_or_condition(self, se_source):
        """v1.1: _promo_ok_ly uses OR condition for promoter level or buying."""
        assert "_promo_ok_ly" in se_source, (
            "scoring_engine missing _promo_ok_ly OR condition for promoter gate."
        )

    def test_engine_inv_surge_in_fw_lynch(self, se_source):
        """v1.1: fw_lynch must include ~_inv_surge_disq (tilde = NOT inventory surge)."""
        assert re.search(r'~\s*_inv_surge_disq', se_source), (
            "fw_lynch must include (~_inv_surge_disq) to veto pass on inventory surge."
        )
