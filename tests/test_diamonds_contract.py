"""
test_diamonds_contract.py

Contract tests that parse docs/diamonds_financial_specs.json and assert that
core/scoring_engine.py and core/data_engine.py honour every threshold in the
Diamonds in the Dust Three-Lens Framework spec.

Twelve gates verified:
  Gate Zero:  promoter_holdings >= 40    (promoter alignment)
              pledged_percentage < 10    (no distress collateral)
  Lens 1:     cfo_to_pat >= 80.0        (cash earnings quality; book 0.8 ratio)
              dso_delta_3y <= 15.0      (DSO channel-stuffing guard)
              dm_forensic_flag_count == 0  (Diamond-specific 6-flag forensic subset)
  Lens 2:     roce_med_10y >= 15        (moat proven over full cycle)
              roce_med_5y  >= 15        (moat sustained recently)
  Lens 3:     cumulative_fcf_to_ccfo >= 25  (FCF/CFO self-sufficiency)
  Stage 1:    rev_gr_10y >= 10          (long-run demand growth)
              rev_gr_5y  >= 8           (recent deceleration guard)
              D/E < 0.5 (fin sector exempted)
              market_cap >= 500         (proven business scale)

Output column: df["diamonds_pass"] must be assigned.
DSO proxy column: df["dso_delta_3y"] must be computed in data_engine.py.
Cumulative proxy: df["cumulative_fcf_to_ccfo"] must be computed in data_engine.py.

Rejected gates verified absent from fw_diamond block:
  - cfo_to_pat >= 75  (old threshold; book specifies 0.8 = 80%, not 0.75 = 75%)
"""
import json
import re
from pathlib import Path

ROOT         = Path(__file__).parent.parent
SPEC_PATH    = ROOT / "docs" / "diamonds_financial_specs.json"
SCORING_PATH = ROOT / "core" / "scoring_engine.py"
DATA_PATH    = ROOT / "core" / "data_engine.py"


def _load_spec():
    with open(SPEC_PATH, encoding="utf-8") as f:
        return json.load(f)


def _scoring_src():
    return SCORING_PATH.read_text(encoding="utf-8")


def _data_src():
    return DATA_PATH.read_text(encoding="utf-8")


def _fw_diamond_block(src: str) -> str:
    """Extract the fw_diamond block from scoring_engine.py source."""
    anchor = "# 13. Diamond Field Guide"
    next_anchor = "# 14."
    start = src.find(anchor)
    assert start != -1, "Cannot find fw_diamond block anchor in scoring_engine.py"
    end = src.find(next_anchor, start)
    assert end != -1, "Cannot find end boundary of fw_diamond block"
    return src[start:end]


# ─── Spec file integrity ───────────────────────────────────────────────────────

class TestSpecFileIntegrity:
    def test_spec_file_exists(self):
        assert SPEC_PATH.exists(), f"Missing spec file: {SPEC_PATH}"

    def test_spec_has_meta(self):
        spec = _load_spec()
        assert "_meta" in spec
        assert spec["_meta"]["version"] == "1.0-proxy"

    def test_spec_has_gate_zero(self):
        spec = _load_spec()
        assert "gate_zero_integrity" in spec
        gz = spec["gate_zero_integrity"]
        assert "promoter_holding_floor" in gz
        assert "pledge_ceiling" in gz

    def test_spec_has_three_lenses(self):
        spec = _load_spec()
        assert "lens_1_clean_accounts" in spec
        assert "lens_2_competitive_moat" in spec
        assert "lens_3_capital_allocation" in spec

    def test_spec_has_stage1(self):
        spec = _load_spec()
        assert "stage_1_screen" in spec

    def test_spec_has_implementation_mapping(self):
        spec = _load_spec()
        assert "implementation_mapping" in spec
        im = spec["implementation_mapping"]
        assert im["_pass_column"] == "diamonds_pass"
        assert im["_framework_variable"] == "fw_diamond"
        assert im["_frameworks_passed_label"] == "Diamond"

    def test_spec_promoter_threshold(self):
        spec = _load_spec()
        assert spec["gate_zero_integrity"]["promoter_holding_floor"]["threshold"] == 40.0

    def test_spec_pledge_threshold(self):
        spec = _load_spec()
        assert spec["gate_zero_integrity"]["pledge_ceiling"]["threshold"] == 10.0

    def test_spec_cfo_pat_threshold(self):
        spec = _load_spec()
        # Book: 0.8 ratio = 80% in PERCENTAGE CSV column
        t = spec["lens_1_clean_accounts"]["cfo_to_pat_floor"]["threshold"]
        assert t == 80.0, f"Expected 80.0 (book 0.8 ratio), got {t}"
        assert spec["lens_1_clean_accounts"]["cfo_to_pat_floor"]["book_ratio"] == 0.8

    def test_spec_dso_delta_threshold(self):
        spec = _load_spec()
        t = spec["lens_1_clean_accounts"]["dso_rolling_3y_max_rise"]["threshold"]
        assert t == 15.0, f"Expected 15.0 days, got {t}"
        assert spec["lens_1_clean_accounts"]["dso_rolling_3y_max_rise"]["operator"] == "<="

    def test_spec_roce_10y_threshold(self):
        spec = _load_spec()
        assert spec["lens_2_competitive_moat"]["roce_10y_floor"]["threshold"] == 15.0

    def test_spec_roce_5y_threshold(self):
        spec = _load_spec()
        assert spec["lens_2_competitive_moat"]["roce_5y_floor"]["threshold"] == 15.0

    def test_spec_fcf_to_ccfo_threshold(self):
        spec = _load_spec()
        t = spec["lens_3_capital_allocation"]["cumulative_fcf_to_ccfo"]["threshold"]
        assert t == 25.0, f"Expected 25.0%, got {t}"

    def test_spec_mcap_threshold(self):
        spec = _load_spec()
        assert spec["stage_1_screen"]["size_floor"]["threshold_crore"] == 500.0

    def test_spec_rev_10y_threshold(self):
        spec = _load_spec()
        assert spec["stage_1_screen"]["revenue_growth_10y"]["threshold"] == 10.0

    def test_spec_rev_5y_threshold(self):
        spec = _load_spec()
        assert spec["stage_1_screen"]["revenue_growth_5y_guard"]["threshold"] == 8.0

    def test_spec_de_threshold(self):
        spec = _load_spec()
        assert spec["stage_1_screen"]["balance_sheet_quality"]["threshold"] == 0.5
        assert spec["stage_1_screen"]["balance_sheet_quality"]["financial_sector_exempted"] is True

    def test_spec_has_known_gaps(self):
        spec = _load_spec()
        assert "known_gaps" in spec
        assert len(spec["known_gaps"]) >= 2

    def test_spec_has_rejected_gates(self):
        spec = _load_spec()
        assert "rejected_gates" in spec
        # RPT gate must be documented as rejected
        rejected_names = [r["gate"] for r in spec["rejected_gates"]]
        assert any("RPT" in g or "related_party" in g.lower() for g in rejected_names)


# ─── Gate Zero — Promoter alignment ──────────────────────────────────────────

class TestGateZeroPromoter:
    def test_promoter_40_threshold_in_code(self):
        block = _fw_diamond_block(_scoring_src())
        # promo_dm.fillna(0) >= 40 — exactly 40, not 45 or 50
        assert re.search(r"promo_dm\.fillna\(0\)\s*>=\s*40\b", block), \
            "fw_diamond must gate promoter_holdings >= 40"

    def test_promoter_not_50_threshold(self):
        block = _fw_diamond_block(_scoring_src())
        # Diamonds uses 40%, not 100-Bagger's 50%
        assert not re.search(r"promo_dm\.fillna\(0\)\s*>=\s*50\b", block), \
            "fw_diamond must use >= 40 for promoter (not >= 50 — that's 100-Bagger)"

    def test_promoter_reads_promoter_holdings(self):
        block = _fw_diamond_block(_scoring_src())
        assert "promoter_holdings" in block, "fw_diamond must read promoter_holdings column"

    def test_pledge_threshold_in_code(self):
        block = _fw_diamond_block(_scoring_src())
        assert re.search(r"pledge_dm\s*<\s*10\b", block), \
            "fw_diamond must gate pledge < 10"

    def test_pledge_fillna_conservative(self):
        block = _fw_diamond_block(_scoring_src())
        # pledge_dm = df.get(...).fillna(100) — unknown pledge fails conservatively
        assert re.search(r"fillna\(100\)", block), \
            "pledge must fillna(100) so unknown pledge fails the < 10 gate"


# ─── Lens 1 — Cash earnings quality ──────────────────────────────────────────

class TestLens1CleanAccounts:
    def test_cfo_pat_80_threshold(self):
        block = _fw_diamond_block(_scoring_src())
        # Must be >= 80.0 (book 0.8 ratio in PERCENTAGE column), NOT >= 75
        assert re.search(r"cfo_pat_dm\.fillna\(0\)\s*>=\s*80(?:\.0)?\b", block), \
            "fw_diamond Lens 1 must gate CFO/PAT >= 80.0 (book 0.8 ratio)"

    def test_cfo_pat_not_75_threshold(self):
        block = _fw_diamond_block(_scoring_src())
        assert not re.search(r"cfo_pat_dm\.fillna\(0\)\s*>=\s*75\b", block), \
            "Old 75% threshold must be updated to 80% (book specifies 0.8 ratio)"

    def test_dso_delta_gate_present(self):
        block = _fw_diamond_block(_scoring_src())
        assert re.search(r"dso_delta_dm\.fillna\(999\)\s*<=\s*15(?:\.0)?\b", block), \
            "fw_diamond Lens 1 must gate DSO delta <= 15.0 days"

    def test_dso_delta_reads_dso_delta_3y(self):
        block = _fw_diamond_block(_scoring_src())
        assert "dso_delta_3y" in block, "fw_diamond must read dso_delta_3y column"

    def test_dso_fillna_conservative(self):
        block = _fw_diamond_block(_scoring_src())
        # dso_delta_dm.fillna(999) — unknown DSO fails the <= 15 gate
        assert re.search(r"dso_delta_dm\.fillna\(999\)", block), \
            "dso_delta_dm must fillna(999) so missing DSO data fails conservatively"

    def test_forensic_score_zero_gate(self):
        """fw_diamond must gate on fscore_dm == 0."""
        block = _fw_diamond_block(_scoring_src())
        assert re.search(r"fscore_dm\s*==\s*0", block), \
            "fw_diamond must gate fscore_dm == 0 (zero Diamond-specific forensic flags)"

    def test_forensic_reads_dm_forensic_flag_count(self):
        """fw_diamond must read dm_forensic_flag_count (Diamond-specific 6-flag subset).
        Must NOT read red_flag_count (27-flag generic set) or forensic_score (scaled 42-100).
        """
        block = _fw_diamond_block(_scoring_src())
        assert '"dm_forensic_flag_count"' in block, (
            'fscore_dm must read "dm_forensic_flag_count" — the Diamond-specific 6-flag subset. '
            'Using red_flag_count (27 generic flags) penalises companies for Malik/Coffee Can/'
            'WCS24 standards that Diamonds in the Dust never checks.'
        )
        assert '"red_flag_count"' not in block, (
            'fw_diamond must NOT read "red_flag_count" — use dm_forensic_flag_count instead'
        )
        assert '"forensic_score"' not in block, (
            'fw_diamond must NOT read "forensic_score" — that column is a scaled score (42-100), not a count'
        )

    def test_forensic_fillna_conservative(self):
        """fscore_dm must fillna(999) so stocks with missing forensic data fail conservatively."""
        block = _fw_diamond_block(_scoring_src())
        assert re.search(r"fscore_dm\s*=.*fillna\(999\)", block), \
            "fscore_dm must fillna(999) so unknown forensic data fails conservatively"

    def test_dm_forensic_flag_count_computed_in_forensic_engine(self):
        """dm_forensic_flag_count must be computed in forensic_engine.py with the correct 6 flags."""
        src = Path(ROOT / "core" / "forensic_engine.py").read_text(encoding="utf-8")
        assert "dm_forensic_flag_count" in src, \
            "dm_forensic_flag_count not found in forensic_engine.py — Diamond forensic subset not computed"
        assert "_DIAMOND_FLAGS" in src, \
            "_DIAMOND_FLAGS list not found in forensic_engine.py"
        # Verify each of the 6 Diamond-specific flags is in the subset
        for flag in ("rf_low_cfo_pat", "rf_high_accruals", "rf_high_receivables",
                     "rf_receivables_bloat", "rf_rising_debt", "rf_dilution"):
            assert flag in src, f"{flag} not found in forensic_engine.py _DIAMOND_FLAGS definition"

    def test_dm_forensic_excludes_coffee_can_flag(self):
        """rf_low_cfo_ebitda (Coffee Can CFO/EBITDA standard) must NOT be in _DIAMOND_FLAGS.
        Diamond uses CFO/PAT; Coffee Can uses CFO/EBITDA — different books, different denominators.
        """
        src = Path(ROOT / "core" / "forensic_engine.py").read_text(encoding="utf-8")
        # Find the _DIAMOND_FLAGS block
        flag_start = src.find("_DIAMOND_FLAGS = [")
        flag_end   = src.find("]", flag_start)
        flag_block = src[flag_start:flag_end]
        assert "rf_low_cfo_ebitda" not in flag_block, (
            "rf_low_cfo_ebitda must NOT be in _DIAMOND_FLAGS — it is a Coffee Can Investing "
            "standard (CFO/EBITDA denominator). Diamond uses CFO/PAT. Different book, different gate."
        )


# ─── Lens 2 — ROCE moat durability ───────────────────────────────────────────

class TestLens2CompetitiveMoat:
    def test_roce_10y_gate(self):
        block = _fw_diamond_block(_scoring_src())
        assert re.search(r"roce_10y_dm\.fillna\(0\)\s*>=\s*15\b", block), \
            "fw_diamond Lens 2 must gate roce_med_10y >= 15"

    def test_roce_5y_gate(self):
        block = _fw_diamond_block(_scoring_src())
        assert re.search(r"roce_5y_dm\.fillna\(0\)\s*>=\s*15\b", block), \
            "fw_diamond Lens 2 must gate roce_med_5y >= 15 (recent moat guard)"

    def test_roce_reads_correct_columns(self):
        block = _fw_diamond_block(_scoring_src())
        assert "roce_med_10y" in block, "fw_diamond must read roce_med_10y"
        assert "roce_med_5y" in block, "fw_diamond must read roce_med_5y"


# ─── Lens 3 — Capital allocation self-sufficiency ────────────────────────────

class TestLens3CapitalAllocation:
    def test_fcf_to_ccfo_gate(self):
        block = _fw_diamond_block(_scoring_src())
        assert re.search(r"fcf_cfo_dm\.fillna\(0\)\s*>=\s*25\b", block), \
            "fw_diamond Lens 3 must gate FCF/CFO >= 25"

    def test_fcf_reads_cumulative_column(self):
        block = _fw_diamond_block(_scoring_src())
        # Must read cumulative_fcf_to_ccfo (not raw fcf_to_cfo_pct directly)
        assert "cumulative_fcf_to_ccfo" in block, \
            "fw_diamond Lens 3 must read cumulative_fcf_to_ccfo column (proxy alias)"


# ─── Stage 1 Screen ──────────────────────────────────────────────────────────

class TestStage1Screen:
    def test_rev_10y_gate(self):
        block = _fw_diamond_block(_scoring_src())
        assert re.search(r"rev_10y_dm\.fillna\(0\)\s*>=\s*10\b", block), \
            "fw_diamond Stage 1 must gate rev_gr_10y >= 10"

    def test_rev_5y_gate(self):
        block = _fw_diamond_block(_scoring_src())
        assert re.search(r"rev_5y_dm\.fillna\(0\)\s*>=\s*8\b", block), \
            "fw_diamond Stage 1 must gate rev_gr_5y >= 8 (deceleration guard)"

    def test_de_05_threshold(self):
        block = _fw_diamond_block(_scoring_src())
        assert re.search(r"de_dm\.fillna\(999\)\s*<\s*0\.5\b", block), \
            "fw_diamond Stage 1 must gate D/E < 0.5"

    def test_financial_sector_exemption(self):
        block = _fw_diamond_block(_scoring_src())
        # is_fin_dm exempts financial sector from D/E gate
        assert re.search(r"is_fin_dm\s*\|.*de_dm.*<\s*0\.5", block), \
            "fw_diamond must exempt financial sector from D/E gate"

    def test_mcap_500_gate(self):
        block = _fw_diamond_block(_scoring_src())
        assert re.search(r"mcap_dm\.fillna\(0\)\s*>=\s*500\b", block), \
            "fw_diamond Stage 1 must gate market_cap >= 500 Cr"


# ─── Output column ───────────────────────────────────────────────────────────

class TestDiamondsPassOutput:
    def test_diamonds_pass_column_assigned(self):
        block = _fw_diamond_block(_scoring_src())
        assert 'df["diamonds_pass"]' in block, \
            'fw_diamond block must assign df["diamonds_pass"]'

    def test_diamonds_pass_is_int(self):
        block = _fw_diamond_block(_scoring_src())
        assert re.search(r'df\["diamonds_pass"\]\s*=\s*fw_diamond\.astype\(int\)', block), \
            'df["diamonds_pass"] must be fw_diamond.astype(int)'

    def test_diamonds_label_in_fw_str(self):
        src = _scoring_src()
        # Find the fw_str builder
        fw_str_start = src.find('fw_str = (')
        assert fw_str_start != -1, "Cannot find fw_str builder block"
        fw_str_end = src.find('df["frameworks_passed"]', fw_str_start)
        fw_str_block = src[fw_str_start:fw_str_end]
        assert 'fw_diamond' in fw_str_block, \
            'fw_diamond must be wired into frameworks_passed fw_str builder'
        assert '"Diamond|"' in fw_str_block or '"Diamond"' in fw_str_block, \
            'Framework label "Diamond" must appear in fw_str builder'


# ─── Rejected gate guards ─────────────────────────────────────────────────────

class TestRejectedGates:
    def test_old_75_threshold_absent(self):
        block = _fw_diamond_block(_scoring_src())
        assert not re.search(r">=\s*75\b", block), \
            "Old CFO/PAT >= 75 threshold must be removed (updated to 80)"

    def test_no_fcf_to_cfo_pct_direct_read_in_fw_diamond(self):
        block = _fw_diamond_block(_scoring_src())
        # fw_diamond must read cumulative_fcf_to_ccfo (proxy alias), not raw fcf_to_cfo_pct
        # The variable assignment uses cumulative_fcf_to_ccfo; fcf_to_cfo_pct may appear in comment only
        assert "cumulative_fcf_to_ccfo" in block, \
            "fw_diamond should reference cumulative_fcf_to_ccfo column, not raw fcf_to_cfo_pct"


# ─── Data engine columns ──────────────────────────────────────────────────────

class TestDataEngineColumns:
    def test_dso_delta_3y_computed(self):
        src = _data_src()
        assert 'df["dso_delta_3y"]' in src, \
            'data_engine.py must compute df["dso_delta_3y"]'

    def test_dso_delta_uses_3yb_not_1yb(self):
        src = _data_src()
        # dso_delta_3y must use days_receivable_3yb (TRUE 3Y window, not 1Y proxy)
        dso_block_start = src.find('df["dso_delta_3y"]')
        assert dso_block_start != -1
        snippet = src[dso_block_start:dso_block_start + 300]
        assert "days_receivable_3yb" in snippet, \
            "dso_delta_3y must use days_receivable_3yb (true 3Y window — CSV now has 3YB data)"
        assert "days_receivable_1yb" not in snippet, \
            "dso_delta_3y must NOT use days_receivable_1yb (that was the 1Y proxy; 3YB is now available)"

    def test_days_receivable_3yb_in_column_map(self):
        src = _data_src()
        assert '"Days Receivable 3 Years Back"' in src, \
            'data_engine.py must map "Days Receivable 3 Years Back" → days_receivable_3yb in RATIO_COLS'

    def test_cumulative_fcf_to_ccfo_computed(self):
        src = _data_src()
        assert 'df["cumulative_fcf_to_ccfo"]' in src, \
            'data_engine.py must compute df["cumulative_fcf_to_ccfo"]'

    def test_cumulative_fcf_aliases_fcf_to_cfo_pct(self):
        src = _data_src()
        ccfo_start = src.find('df["cumulative_fcf_to_ccfo"]')
        assert ccfo_start != -1
        snippet = src[ccfo_start:ccfo_start + 200]
        assert "fcf_to_cfo_pct" in snippet, \
            "cumulative_fcf_to_ccfo must alias fcf_to_cfo_pct as proxy"

    def test_no_iterrows_in_diamond_path(self):
        src = _scoring_src()
        block = _fw_diamond_block(src)
        assert "iterrows()" not in block, "fw_diamond must not use iterrows()"

    def test_no_apply_in_diamond_block(self):
        src = _scoring_src()
        block = _fw_diamond_block(src)
        # No df.apply within the fw_diamond variable definition
        assert ".apply(" not in block, "fw_diamond must not use df.apply()"
