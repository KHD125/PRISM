"""
tests/test_dorsey_contract.py
═════════════════════════════
Pat Dorsey Wide Moat Framework — Closed-Loop Contract Test Suite

Validates that:
  1. docs/dorsey_moat_specs.json is well-formed and complete
  2. All 5 materialized pillar column names exist in scoring_engine.py source text
  3. All numeric thresholds in the spec precisely match the source code
  4. The 5-pillar pass/score logic is arithmetically correct on a controlled DataFrame
  5. Non-default pandas index alignment is handled correctly (no index drift)
  6. The dorsey_pass flag is the exact AND of the 5 materialized pillars
  7. dorsey_score is the exact sum of the 5 materialized pillars (0-5)
  8. All 5 column names are exported from ui/ui_tearsheet.py
  9. render_dorsey_radar is importable from the ui package
 10. docs/dorsey_moat_specs.json lives at the expected path
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPEC_PATH   = ROOT / "docs" / "dorsey_moat_specs.json"
ENGINE_PATH = ROOT / "core" / "scoring_engine.py"
UI_PATH     = ROOT / "ui"  / "ui_tearsheet.py"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_spec() -> dict:
    with SPEC_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _engine_src() -> str:
    return ENGINE_PATH.read_text(encoding="utf-8")


def _ui_src() -> str:
    return UI_PATH.read_text(encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Part 1 — Spec ledger structural integrity
# ─────────────────────────────────────────────────────────────────────────────

class TestDorseySpecLedger:
    """Verifies docs/dorsey_moat_specs.json schema completeness."""

    def test_spec_file_exists(self):
        assert SPEC_PATH.exists(), f"Spec file missing: {SPEC_PATH}"

    def test_spec_is_valid_json(self):
        spec = _load_spec()
        assert isinstance(spec, dict)

    def test_meta_section_present(self):
        spec = _load_spec()
        assert "_meta" in spec
        meta = spec["_meta"]
        for required_key in ("title", "framework_variable", "pass_column",
                             "score_column", "frameworks_passed_label",
                             "framework_number_in_code"):
            assert required_key in meta, f"_meta missing key: {required_key}"

    def test_all_five_pillar_sections_present(self):
        spec = _load_spec()
        expected = [
            "pillar_m_moat_return_level",
            "pillar_d_moat_direction",
            "pillar_v_fcf_valuation",
            "pillar_q_cash_quality",
            "pillar_c_capital_structure",
        ]
        for section in expected:
            assert section in spec, f"Pillar section missing: {section}"

    def test_each_pillar_has_materialized_column_name(self):
        spec = _load_spec()
        pillar_sections = [
            "pillar_m_moat_return_level",
            "pillar_d_moat_direction",
            "pillar_v_fcf_valuation",
            "pillar_q_cash_quality",
            "pillar_c_capital_structure",
        ]
        for section in pillar_sections:
            p = spec[section]
            assert "_column_materialized" in p, (
                f"{section} missing _column_materialized key"
            )
            col = p["_column_materialized"]
            assert isinstance(col, str) and col.startswith("dorsey_"), (
                f"{section} _column_materialized '{col}' must start with 'dorsey_'"
            )

    def test_pass_and_score_columns_documented(self):
        spec = _load_spec()
        assert "pass_flag_logic"   in spec
        assert "score_column_logic" in spec
        assert spec["_meta"]["pass_column"]  == "dorsey_pass"
        assert spec["_meta"]["score_column"] == "dorsey_score"

    def test_framework_label_is_wide_moat(self):
        spec = _load_spec()
        assert spec["_meta"]["frameworks_passed_label"] == "Wide Moat"

    def test_framework_number_is_14(self):
        spec = _load_spec()
        assert spec["_meta"]["framework_number_in_code"] == 14

    def test_vectorization_matrix_covers_all_inputs(self):
        spec = _load_spec()
        assert "vectorization_matrix" in spec
        nan_keys = spec["vectorization_matrix"]["nan_handling"]
        required_inputs = [
            "roce_med_10y", "roce_med_5y", "d35_roce_trend",
            "fcf_yield", "cfo_to_pat", "debt_to_equity",
        ]
        for k in required_inputs:
            assert k in nan_keys, f"vectorization_matrix.nan_handling missing: {k}"


# ─────────────────────────────────────────────────────────────────────────────
# Part 2 — Source code contract verification (regex-based)
# ─────────────────────────────────────────────────────────────────────────────

class TestDorseyEngineContract:
    """Verifies scoring_engine.py materializes columns that match the spec ledger."""

    def test_five_pillar_columns_materialized_in_engine(self):
        src = _engine_src()
        expected_cols = [
            "dorsey_moat_level",
            "dorsey_moat_direction",
            "dorsey_fcf_valuation",
            "dorsey_cash_quality",
            "dorsey_cap_structure",
        ]
        for col in expected_cols:
            pattern = rf'df\[.{col}.\]\s*='
            assert re.search(pattern, src), (
                f"scoring_engine.py does not materialize df['{col}']"
            )

    def test_dorsey_pass_materialized_from_five_pillars(self):
        src = _engine_src()
        # dorsey_pass must be assigned from fw_dorsey
        assert re.search(r'df\["dorsey_pass"\]\s*=\s*fw_dorsey\.astype', src), (
            "dorsey_pass not derived from fw_dorsey boolean mask"
        )

    def test_dorsey_score_materialized_as_sum(self):
        src = _engine_src()
        assert re.search(r'df\["dorsey_score"\]\s*=\s*\(', src), (
            "dorsey_score not materialized as a sum expression"
        )

    def test_threshold_moat_level_20pct_in_engine(self):
        src = _engine_src()
        # ROCE >= 20.0 for both 10Y and 5Y windows
        assert re.search(r'roce_10y_dw\.fillna\(0\)\s*>=\s*20', src), (
            "ROCE 10Y threshold 20 not found in engine"
        )
        assert re.search(r'roce_5y_dw\.fillna\(0\)\s*>=\s*20', src), (
            "ROCE 5Y threshold 20 not found in engine"
        )

    def test_threshold_moat_direction_in_engine(self):
        src = _engine_src()
        assert re.search(r'roce_dir_dw\.fillna\(-1\)\s*>=\s*0', src), (
            "ROCE direction threshold (fillna(-1) >= 0) not found in engine"
        )

    def test_threshold_fcf_yield_5pct_in_engine(self):
        src = _engine_src()
        assert re.search(r'fcf_yield_dw\.fillna\(0\)\s*>=\s*5\.0', src), (
            "FCF yield threshold 5.0 not found in engine"
        )

    def test_threshold_cash_quality_80pct_in_engine(self):
        src = _engine_src()
        assert re.search(r'cfo_pat_dw\.fillna\(0\)\s*>=\s*80\.0', src), (
            "CFO/PAT threshold 80.0 not found in engine"
        )

    def test_threshold_cap_structure_1pt0_in_engine(self):
        src = _engine_src()
        # de_dw.fillna(999) < 1.0 with is_fin_dw OR
        assert re.search(r'de_dw\.fillna\(999\)\s*<\s*1\.0', src), (
            "D/E threshold < 1.0 not found in engine"
        )

    def test_fw_str_uses_fw_dorsey_for_wide_moat_label(self):
        src = _engine_src()
        assert re.search(r'np\.where\(\s*fw_dorsey\s*,\s*.Wide Moat\|', src), (
            "fw_str does not tag fw_dorsey rows with 'Wide Moat|'"
        )

    def test_spec_thresholds_match_engine_values(self):
        """Cross-validates spec JSON thresholds against engine regex captures."""
        src  = _engine_src()
        spec = _load_spec()

        # Pillar M — ROCE 20%
        m10_spec = spec["pillar_m_moat_return_level"]["roce_10y_floor"]["threshold"]
        assert m10_spec == 20.0
        assert re.search(rf'roce_10y_dw\.fillna\(0\)\s*>=\s*{int(m10_spec)}', src)

        # Pillar V — FCF yield 5%
        fcf_spec = spec["pillar_v_fcf_valuation"]["fcf_yield_floor"]["threshold"]
        assert fcf_spec == 5.0
        assert re.search(rf'fcf_yield_dw\.fillna\(0\)\s*>=\s*{fcf_spec}', src)

        # Pillar Q — CFO/PAT 80%
        cfo_spec = spec["pillar_q_cash_quality"]["cfo_pat_floor"]["threshold"]
        assert cfo_spec == 80.0
        assert re.search(rf'cfo_pat_dw\.fillna\(0\)\s*>=\s*{cfo_spec}', src)

        # Pillar C — D/E 1.0
        de_spec = spec["pillar_c_capital_structure"]["de_ceiling"]["threshold"]
        assert de_spec == 1.0
        assert re.search(rf'de_dw\.fillna\(999\)\s*<\s*{de_spec}', src)


# ─────────────────────────────────────────────────────────────────────────────
# Part 3 — Vector arithmetic: controlled DataFrame logic tests
# ─────────────────────────────────────────────────────────────────────────────

def _build_mock_dorsey_row(**overrides) -> dict:
    """All-pass base row; override any field to flip individual gates."""
    base = {
        "roce_med_10y":   22.0,   # M: >= 20 ✅
        "roce_med_5y":    21.0,   # M: >= 20 ✅
        "d35_roce_trend":  1.0,   # D: >= 0  ✅
        "fcf_yield":       6.0,   # V: >= 5  ✅
        "cfo_to_pat":     85.0,   # Q: >= 80 ✅
        "debt_to_equity":  0.5,   # C: < 1.0 ✅
        "is_financial":  False,
        # fill columns used elsewhere in compute_qglp_score
        "roce":           22.0,
        "roe":            18.0,
        "pat_gr_5y":      20.0,
        "eps_gr_5y":      18.0,
        "peg":             1.0,
        "forensic_score": 90.0,
        "forensic_multiplier": 1.0,
        "piotroski_fscore": 7,
    }
    base.update(overrides)
    return base


def _run_dorsey(rows: list[dict], regime: str = "SIDEWAYS") -> pd.DataFrame:
    from config import MASTER_PROFILES
    from core.scoring_engine import compute_qglp_score

    df = pd.DataFrame(rows)
    df.attrs["detected_market_regime"] = regime
    profile = MASTER_PROFILES.get("Balanced", next(iter(MASTER_PROFILES.values())))
    return compute_qglp_score(df, profile)


class TestDorseyPillarArithmetic:
    """Validates the materialized column logic using controlled DataFrames."""

    def test_all_pillars_pass_all_pass_row(self):
        result = _run_dorsey([_build_mock_dorsey_row()])
        assert result["dorsey_moat_level"].iloc[0]    == 1
        assert result["dorsey_moat_direction"].iloc[0] == 1
        assert result["dorsey_fcf_valuation"].iloc[0]  == 1
        assert result["dorsey_cash_quality"].iloc[0]   == 1
        assert result["dorsey_cap_structure"].iloc[0]  == 1
        assert result["dorsey_pass"].iloc[0]  == 1
        assert result["dorsey_score"].iloc[0] == 5

    def test_all_pillars_fail_all_fail_row(self):
        row = _build_mock_dorsey_row(
            roce_med_10y=10.0, roce_med_5y=10.0,
            d35_roce_trend=-1.0,
            fcf_yield=1.0,
            cfo_to_pat=50.0,
            debt_to_equity=2.0,
            is_financial=False,
        )
        result = _run_dorsey([row])
        for col in ["dorsey_moat_level", "dorsey_moat_direction",
                    "dorsey_fcf_valuation", "dorsey_cash_quality",
                    "dorsey_cap_structure"]:
            assert result[col].iloc[0] == 0, f"{col} should be 0"
        assert result["dorsey_pass"].iloc[0]  == 0
        assert result["dorsey_score"].iloc[0] == 0

    def test_pillar_m_boundary_exact_20(self):
        """ROCE exactly at 20.0 must pass (>= not >)."""
        row = _build_mock_dorsey_row(roce_med_10y=20.0, roce_med_5y=20.0)
        result = _run_dorsey([row])
        assert result["dorsey_moat_level"].iloc[0] == 1

    def test_pillar_m_fails_when_5y_below_threshold(self):
        """10Y passes but 5Y < 20 → M pillar fails."""
        row = _build_mock_dorsey_row(roce_med_10y=25.0, roce_med_5y=18.0)
        result = _run_dorsey([row])
        assert result["dorsey_moat_level"].iloc[0] == 0

    def test_pillar_d_boundary_exact_zero(self):
        """ROCE trend exactly 0.0 must pass (>= 0)."""
        row = _build_mock_dorsey_row(d35_roce_trend=0.0)
        result = _run_dorsey([row])
        assert result["dorsey_moat_direction"].iloc[0] == 1

    def test_pillar_d_fails_negative_trend(self):
        row = _build_mock_dorsey_row(d35_roce_trend=-0.01)
        result = _run_dorsey([row])
        assert result["dorsey_moat_direction"].iloc[0] == 0

    def test_pillar_v_boundary_exact_5pct(self):
        """FCF yield exactly 5.0 must pass."""
        row = _build_mock_dorsey_row(fcf_yield=5.0)
        result = _run_dorsey([row])
        assert result["dorsey_fcf_valuation"].iloc[0] == 1

    def test_pillar_v_fails_below_5pct(self):
        row = _build_mock_dorsey_row(fcf_yield=4.99)
        result = _run_dorsey([row])
        assert result["dorsey_fcf_valuation"].iloc[0] == 0

    def test_pillar_q_boundary_exact_80pct(self):
        """CFO/PAT exactly 80.0 must pass (PERCENTAGE unit, not 0.8)."""
        row = _build_mock_dorsey_row(cfo_to_pat=80.0)
        result = _run_dorsey([row])
        assert result["dorsey_cash_quality"].iloc[0] == 1

    def test_pillar_q_fails_below_80pct(self):
        """CFO/PAT 79.9 must fail — confirms PERCENTAGE unit not ratio unit."""
        row = _build_mock_dorsey_row(cfo_to_pat=79.9)
        result = _run_dorsey([row])
        assert result["dorsey_cash_quality"].iloc[0] == 0

    def test_pillar_c_financial_sector_exempt(self):
        """Financial sector stocks with D/E >> 1.0 still pass pillar C."""
        row = _build_mock_dorsey_row(debt_to_equity=5.0, is_financial=True)
        result = _run_dorsey([row])
        assert result["dorsey_cap_structure"].iloc[0] == 1

    def test_pillar_c_non_financial_de_boundary(self):
        """Non-financial D/E exactly 0.99 must pass; 1.00 must fail."""
        row_pass = _build_mock_dorsey_row(debt_to_equity=0.99, is_financial=False)
        row_fail = _build_mock_dorsey_row(debt_to_equity=1.00, is_financial=False)
        r_pass = _run_dorsey([row_pass])
        r_fail = _run_dorsey([row_fail])
        assert r_pass["dorsey_cap_structure"].iloc[0] == 1
        assert r_fail["dorsey_cap_structure"].iloc[0] == 0

    def test_dorsey_pass_is_and_of_pillars(self):
        """dorsey_pass must equal the AND of all 5 pillar columns."""
        rows = [_build_mock_dorsey_row()]   # full pass
        for pillar_override in [
            {"roce_med_10y": 5.0},           # M fails
            {"d35_roce_trend": -5.0},         # D fails
            {"fcf_yield": 1.0},               # V fails
            {"cfo_to_pat": 40.0},             # Q fails
            {"debt_to_equity": 3.0},          # C fails
        ]:
            rows.append(_build_mock_dorsey_row(**pillar_override))

        result = _run_dorsey(rows)
        for i, row in result.iterrows():
            pillars = [
                row["dorsey_moat_level"],
                row["dorsey_moat_direction"],
                row["dorsey_fcf_valuation"],
                row["dorsey_cash_quality"],
                row["dorsey_cap_structure"],
            ]
            expected_pass  = int(all(p == 1 for p in pillars))
            expected_score = int(sum(pillars))
            assert row["dorsey_pass"]  == expected_pass,  f"Row {i}: dorsey_pass mismatch"
            assert row["dorsey_score"] == expected_score, f"Row {i}: dorsey_score mismatch"

    def test_score_range_is_zero_to_five(self):
        """dorsey_score must always be in [0, 5]."""
        import random
        random.seed(42)
        rows = []
        for _ in range(20):
            rows.append(_build_mock_dorsey_row(
                roce_med_10y=random.choice([10.0, 22.0]),
                roce_med_5y=random.choice([10.0, 22.0]),
                d35_roce_trend=random.choice([-1.0, 1.0]),
                fcf_yield=random.choice([2.0, 6.0]),
                cfo_to_pat=random.choice([60.0, 85.0]),
                debt_to_equity=random.choice([0.5, 1.5]),
                is_financial=random.choice([True, False]),
            ))
        result = _run_dorsey(rows)
        assert result["dorsey_score"].between(0, 5).all(), (
            f"dorsey_score out of [0,5]: {result['dorsey_score'].tolist()}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Part 4 — Non-default index alignment (guard against pandas index drift)
# ─────────────────────────────────────────────────────────────────────────────

class TestDorseyIndexAlignment:
    """Verifies pillar logic is correct when DataFrame has non-default index."""

    def test_non_default_index_does_not_drift(self):
        """Create DataFrame with index [100, 200, 300]; verify pillar values align."""
        from config import MASTER_PROFILES
        from core.scoring_engine import compute_qglp_score

        idx = pd.Index([100, 200, 300])
        rows = [
            _build_mock_dorsey_row(),                              # all pass
            _build_mock_dorsey_row(roce_med_10y=5.0),              # M fails
            _build_mock_dorsey_row(fcf_yield=1.0, cfo_to_pat=40.0),  # V,Q fail
        ]
        df = pd.DataFrame(rows, index=idx)
        df.attrs["detected_market_regime"] = "SIDEWAYS"
        profile = MASTER_PROFILES.get("Balanced", next(iter(MASTER_PROFILES.values())))
        result = compute_qglp_score(df, profile)

        # Index must be preserved
        assert list(result.index) == [100, 200, 300]

        # Row 100: all pass
        assert result.loc[100, "dorsey_pass"]  == 1
        assert result.loc[100, "dorsey_score"] == 5

        # Row 200: M fails (ROCE < 20)
        assert result.loc[200, "dorsey_moat_level"] == 0
        assert result.loc[200, "dorsey_pass"]       == 0
        assert result.loc[200, "dorsey_score"]      == 4

        # Row 300: V and Q fail
        assert result.loc[300, "dorsey_fcf_valuation"] == 0
        assert result.loc[300, "dorsey_cash_quality"]  == 0
        assert result.loc[300, "dorsey_pass"]          == 0
        assert result.loc[300, "dorsey_score"]         == 3

    def test_mixed_index_types_no_keyerror(self):
        """String index — pillar reads must not raise KeyError."""
        from config import MASTER_PROFILES
        from core.scoring_engine import compute_qglp_score

        idx = pd.Index(["HDFC", "INFY", "TCS"])
        rows = [_build_mock_dorsey_row() for _ in range(3)]
        df = pd.DataFrame(rows, index=idx)
        df.attrs["detected_market_regime"] = "BULL"
        profile = MASTER_PROFILES.get("Balanced", next(iter(MASTER_PROFILES.values())))
        result = compute_qglp_score(df, profile)

        for col in ["dorsey_moat_level", "dorsey_moat_direction",
                    "dorsey_fcf_valuation", "dorsey_cash_quality",
                    "dorsey_cap_structure", "dorsey_pass", "dorsey_score"]:
            assert col in result.columns
            assert result[col].notna().all(), f"{col} has NaN with string index"

    def test_nan_inputs_conservative_failure(self):
        """NaN in every numeric input → all numeric pillars fail conservatively."""
        from config import MASTER_PROFILES
        from core.scoring_engine import compute_qglp_score

        row = {k: np.nan for k in [
            "roce_med_10y", "roce_med_5y", "d35_roce_trend",
            "fcf_yield", "cfo_to_pat", "debt_to_equity",
        ]}
        row["is_financial"] = False
        # provide minimum columns to avoid KeyError in score functions
        row.update({
            "roce": np.nan, "roe": np.nan,
            "pat_gr_5y": np.nan, "eps_gr_5y": np.nan,
            "peg": np.nan, "forensic_score": np.nan,
            "forensic_multiplier": np.nan, "piotroski_fscore": np.nan,
        })
        df = pd.DataFrame([row])
        df.attrs["detected_market_regime"] = "SIDEWAYS"
        profile = MASTER_PROFILES.get("Balanced", next(iter(MASTER_PROFILES.values())))
        result = compute_qglp_score(df, profile)

        # Numeric pillars (M, D, V, Q) must fail; C might pass for financials but row is non-fin
        assert result["dorsey_moat_level"].iloc[0]    == 0, "M should fail on NaN ROCE"
        assert result["dorsey_moat_direction"].iloc[0] == 0, "D should fail on NaN trend"
        assert result["dorsey_fcf_valuation"].iloc[0]  == 0, "V should fail on NaN FCF"
        assert result["dorsey_cash_quality"].iloc[0]   == 0, "Q should fail on NaN CFO/PAT"
        # C: de_dw.fillna(999) < 1.0 → 999 < 1.0 = False → fails
        assert result["dorsey_cap_structure"].iloc[0]  == 0, "C should fail on NaN D/E"
        assert result["dorsey_pass"].iloc[0]  == 0
        assert result["dorsey_score"].iloc[0] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Part 5 — UI surface contract
# ─────────────────────────────────────────────────────────────────────────────

class TestDorseyUIContract:
    """Validates the UI module exposes the correct function and reads correct columns."""

    def test_render_dorsey_radar_importable(self):
        from ui import render_dorsey_radar
        assert callable(render_dorsey_radar)

    def test_render_dorsey_radar_in_ui_all(self):
        import ui
        assert "render_dorsey_radar" in ui.__all__

    def test_ui_tearsheet_reads_all_five_pillar_columns(self):
        src = _ui_src()
        expected_cols = [
            "dorsey_moat_level",
            "dorsey_moat_direction",
            "dorsey_fcf_valuation",
            "dorsey_cash_quality",
            "dorsey_cap_structure",
        ]
        for col in expected_cols:
            assert col in src, (
                f"ui_tearsheet.py does not reference column '{col}'"
            )

    def test_ui_tearsheet_reads_dorsey_pass_and_score(self):
        src = _ui_src()
        assert "dorsey_pass"  in src, "ui_tearsheet.py does not reference dorsey_pass"
        assert "dorsey_score" in src, "ui_tearsheet.py does not reference dorsey_score"

    def test_render_dorsey_radar_defined_in_tearsheet(self):
        src = _ui_src()
        assert re.search(r'def render_dorsey_radar\s*\(', src), (
            "render_dorsey_radar function not defined in ui_tearsheet.py"
        )

    def test_raw_signals_includes_dorsey_cells(self):
        """render_raw_signals must include dorsey_score and dorsey_pass cells."""
        src = _ui_src()
        assert re.search(r'_cell\(.*Dorsey Score.*dorsey_score', src, re.DOTALL), (
            "render_raw_signals does not include dorsey_score cell"
        )
        assert re.search(r'_cell\(.*Dorsey Pass.*dorsey_pass', src, re.DOTALL), (
            "render_raw_signals does not include dorsey_pass cell"
        )
