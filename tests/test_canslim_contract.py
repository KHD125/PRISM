"""
test_canslim_contract.py

Contract tests that parse docs/canslim_technical_specs.json and assert that
core/scoring_engine.py honours every threshold in the CAN SLIM Tactical Momentum
Engine specification (William O'Neil — How to Make Money in Stocks, 4th Edition).

Seven criteria verified:
  C: Quarterly EPS per share >= 25% YoY (q_eps_yoy)  AND  quarterly sales >= 25% YoY (q_rev_yoy)
  A: EPS 5Y CAGR >= 25%  +  ROE >= 17%  +  eps_gr_3y >= 0  +  eps_gr_yoy >= 0
  N: dist_52wh <= 15% (near 52-week high)
  S: vol_ratio >= 1.5 (institutional surge threshold)
  L: d47_rs_composite percentile rank >= 80 (top 20% of universe)
  I: change_fii_lq > 0 OR change_dii_lq > 0
  M: detect_market_regime != "BEAR"  (internal breadth consensus)

Output column: df["can_slim_pass"] must be assigned immediately after fw_can_slim.
Spec: docs/canslim_technical_specs.json
"""
import json
import re
from pathlib import Path

ROOT         = Path(__file__).parent.parent
SPEC_PATH    = ROOT / "docs" / "canslim_technical_specs.json"
SCORING_PATH = ROOT / "core" / "scoring_engine.py"


def _load_spec():
    with open(SPEC_PATH, encoding="utf-8") as f:
        return json.load(f)


def _scoring_src():
    return SCORING_PATH.read_text(encoding="utf-8")


def _fw_can_slim_block(src: str) -> str:
    """Extract the fw_can_slim block from scoring_engine.py source."""
    anchor     = "# 6. CAN SLIM (William O'Neill)"
    next_anchor = "# 7. Bruised Blue Chip"
    start = src.find(anchor)
    assert start != -1, "Cannot find fw_can_slim block anchor '# 6. CAN SLIM' in scoring_engine.py"
    end = src.find(next_anchor, start)
    assert end != -1, "Cannot find end boundary '# 7. Bruised Blue Chip' of fw_can_slim block"
    return src[start:end]


# ─── Spec file integrity ───────────────────────────────────────────────────────

class TestSpecFileIntegrity:
    def test_spec_file_exists(self):
        assert SPEC_PATH.exists(), f"Missing spec file: {SPEC_PATH}"

    def test_spec_has_meta(self):
        spec = _load_spec()
        assert "_meta" in spec
        assert "CAN SLIM" in spec["_meta"]["title"]

    def test_spec_has_all_seven_criteria(self):
        spec = _load_spec()
        required = [
            "c_current_quarterly_growth",
            "a_annual_earnings_increases",
            "n_new_highs_and_catalysts",
            "s_supply_and_demand",
            "l_leader_vs_laggard",
            "i_institutional_sponsorship",
            "m_market_direction_integration",
        ]
        for key in required:
            assert key in spec, f"Spec missing '{key}' section"

    def test_spec_has_implementation_mapping(self):
        spec = _load_spec()
        assert "implementation_mapping" in spec
        im = spec["implementation_mapping"]
        assert im["_pass_column"]  == "can_slim_pass"
        assert im["_framework_variable"] == "fw_can_slim"
        assert im["_score_column"] == "can_slim_score"
        assert im["_frameworks_passed_label"] == "CAN SLIM"

    def test_spec_c_eps_threshold(self):
        spec = _load_spec()
        t = spec["c_current_quarterly_growth"]["quarterly_eps_growth"]["threshold"]
        assert t == 25.0, f"C: quarterly EPS threshold must be 25.0%, got {t}"

    def test_spec_c_sales_threshold(self):
        spec = _load_spec()
        t = spec["c_current_quarterly_growth"]["quarterly_sales_growth"]["threshold"]
        assert t == 25.0, f"C: quarterly sales threshold must be 25.0%, got {t}"
        assert spec["c_current_quarterly_growth"]["quarterly_sales_growth"]["metric"] == "q_rev_yoy"

    def test_spec_a_cagr_threshold(self):
        spec = _load_spec()
        t = spec["a_annual_earnings_increases"]["annual_eps_cagr_5y"]["threshold"]
        assert t == 25.0, f"A: 5Y CAGR threshold must be 25.0%, got {t}"

    def test_spec_a_roe_threshold(self):
        spec = _load_spec()
        t = spec["a_annual_earnings_increases"]["return_on_equity"]["threshold"]
        assert t == 17.0, f"A: ROE threshold must be 17.0%, got {t}"

    def test_spec_a_consistency_threshold(self):
        spec = _load_spec()
        t = spec["a_annual_earnings_increases"]["annual_consistency_eps_no_decline"]["threshold"]
        assert t == 0.0, f"A: consistency no-decline threshold must be 0.0, got {t}"

    def test_spec_n_distance_threshold(self):
        spec = _load_spec()
        t = spec["n_new_highs_and_catalysts"]["distance_from_52w_high_pct"]["threshold"]
        assert t == 15.0, f"N: 52WH distance threshold must be 15.0%, got {t}"

    def test_spec_s_volume_threshold(self):
        spec = _load_spec()
        t = spec["s_supply_and_demand"]["volume_ratio_sma20"]["threshold"]
        assert t == 1.5, f"S: volume ratio threshold must be 1.5, got {t}"

    def test_spec_l_rs_percentile_threshold(self):
        spec = _load_spec()
        t = spec["l_leader_vs_laggard"]["relative_strength_percentile"]["threshold"]
        assert t == 80.0, f"L: RS percentile threshold must be 80.0, got {t}"

    def test_spec_i_requires_institutional_accumulation(self):
        spec = _load_spec()
        assert spec["i_institutional_sponsorship"]["institutional_accumulation"]["requires_institutional_accumulation"] is True

    def test_spec_m_disables_on_bear(self):
        spec = _load_spec()
        assert spec["m_market_direction_integration"]["market_regime_gate"]["regime_block"] == "BEAR"
        assert spec["m_market_direction_integration"]["market_regime_gate"]["internal_breadth_source"] == "detect_market_regime"

    def test_spec_has_known_gaps(self):
        spec = _load_spec()
        assert "known_gaps" in spec
        assert len(spec["known_gaps"]) >= 1


# ─── C Criterion — Current Quarterly Earnings ─────────────────────────────────

class TestCriterionC:
    def test_c_quarterly_eps_gate_present(self):
        """C criterion: quarterly EPS-per-share growth >= 25% YoY via q_eps_yoy.
        O'Neil Ch.3 specifies EPS (per share), not total PAT. Using the precomputed
        q_eps_yoy series from data_engine.py eliminates inline ratio noise."""
        block = _fw_can_slim_block(_scoring_src())
        assert re.search(r"q_eps_cs\s*>=\s*25(?:\.0)?\b", block), \
            "fw_can_slim C criterion must gate q_eps_yoy (EPS per share) >= 25.0 — O'Neil uses EPS not PAT"

    def test_c_reads_q_eps_yoy(self):
        """q_eps_yoy must be loaded into the CAN SLIM block (precomputed in data_engine.py)."""
        block = _fw_can_slim_block(_scoring_src())
        assert "q_eps_yoy" in block, \
            "fw_can_slim must read q_eps_yoy for C criterion (EPS per share quarterly YoY)"

    def test_c_no_inline_pat_ratio(self):
        """C criterion must NOT inline-compute pat_lq / pat_pyq ratio — use precomputed q_eps_yoy.
        Inlining raw PAT quarterly values introduces point-in-time noise excluded by the
        noise-exclusion principle: all ratio computations belong in data_engine.py."""
        block = _fw_can_slim_block(_scoring_src())
        assert not re.search(r"pat_lq_cs\s*/\s*pat_pyq_cs", block), \
            "C criterion must not inline pat_lq_cs / pat_pyq_cs — use precomputed q_eps_yoy"

    def test_c_quarterly_sales_gate_present(self):
        """C criterion: quarterly sales YoY >= 25% — top-line validation required."""
        block = _fw_can_slim_block(_scoring_src())
        assert re.search(r"q_rev_cs\s*>=\s*25(?:\.0)?\b", block), \
            "fw_can_slim C criterion must gate q_rev_yoy (quarterly sales) >= 25.0"

    def test_c_reads_q_rev_yoy(self):
        """q_rev_yoy must be loaded into the CAN SLIM block."""
        block = _fw_can_slim_block(_scoring_src())
        assert "q_rev_yoy" in block, \
            "fw_can_slim must read q_rev_yoy for quarterly sales confirmation"

    def test_c_sales_threshold_not_20(self):
        """C criterion sales gate must be 25%, not the looser 20%."""
        block = _fw_can_slim_block(_scoring_src())
        assert not re.search(r"q_rev_cs\s*>=\s*20\b", block), \
            "C criterion sales threshold must be 25% (not 20%)"

    def test_c_eps_zero_base_guarded_in_data_engine(self):
        """q_eps_yoy zero-base guard must exist in data_engine.py (not duplicated in scoring).
        The guard (eps_pyq.abs() > 0) is applied at source — scoring reads a clean series."""
        src = Path(ROOT / "core" / "data_engine.py").read_text(encoding="utf-8")
        assert re.search(r"eps_pyq.*abs\(\).*>\s*0|eps_pyq.*notna\(\)", src), \
            "data_engine.py must guard q_eps_yoy against zero/missing eps_pyq base"

    def test_c_eps_missing_data_fails_conservatively(self):
        """Missing EPS history must fail C criterion — fillna(0) < 25 = fail."""
        block = _fw_can_slim_block(_scoring_src())
        # q_eps_cs = df.get("q_eps_yoy", ...).fillna(0) → 0 < 25 → fails
        assert re.search(r"q_eps.*fillna\(0\)|fillna\(0\).*q_eps", block), \
            "q_eps_yoy must fillna(0) so missing EPS data fails the >= 25 gate conservatively"


# ─── A Criterion — Annual Earnings Increases ──────────────────────────────────

class TestCriterionA:
    def test_a_eps_cagr_25_threshold(self):
        """A criterion: EPS 5Y CAGR >= 25%."""
        block = _fw_can_slim_block(_scoring_src())
        assert re.search(r"eps_gr_cs\s*>=\s*25\b", block), \
            "fw_can_slim A criterion must gate eps_gr_5y >= 25"

    def test_a_eps_cagr_not_20(self):
        """A criterion CAGR threshold must be 25%, not the old (wrong) 20%."""
        block = _fw_can_slim_block(_scoring_src())
        assert not re.search(r"eps_gr_cs\s*>=\s*20\b", block), \
            "EPS CAGR threshold must be 25 (not 20) — book specifies 25%"

    def test_a_roe_threshold(self):
        """A criterion: ROE >= 17% (O'Neil's explicit capital efficiency gate)."""
        block = _fw_can_slim_block(_scoring_src())
        assert re.search(r"roe_cs\s*>=\s*17\b", block), \
            "fw_can_slim A criterion must gate ROE >= 17"

    def test_a_eps_3y_consistency(self):
        """A criterion: eps_gr_3y >= 0 — 3Y CAGR not negative (net upward trajectory)."""
        block = _fw_can_slim_block(_scoring_src())
        assert re.search(r"eps_gr_3y_cs\s*>=\s*0\b", block), \
            "fw_can_slim A criterion must gate eps_gr_3y >= 0 for annual consistency"

    def test_a_eps_yoy_consistency(self):
        """A criterion: eps_gr_yoy >= 0 — not currently contracting."""
        block = _fw_can_slim_block(_scoring_src())
        assert re.search(r"eps_yoy_cs\s*>=\s*0\b", block), \
            "fw_can_slim A criterion must gate eps_gr_yoy >= 0 (no current year decline)"

    def test_a_reads_eps_gr_3y(self):
        """eps_gr_3y must be loaded into the CAN SLIM block."""
        block = _fw_can_slim_block(_scoring_src())
        assert "eps_gr_3y" in block, \
            "fw_can_slim must read eps_gr_3y for annual consistency check"

    def test_a_reads_eps_gr_yoy(self):
        """eps_gr_yoy must be loaded into the CAN SLIM block."""
        block = _fw_can_slim_block(_scoring_src())
        assert "eps_gr_yoy" in block, \
            "fw_can_slim must read eps_gr_yoy for no-current-year-decline check"

    def test_a_consistency_fillna_conservative(self):
        """Missing eps history must fail the consistency gate (not pass)."""
        block = _fw_can_slim_block(_scoring_src())
        # fillna(-1) — unknown history treated as decline (conservative)
        assert re.search(r"eps_gr_3y.*fillna\(-1\)|fillna\(-1\).*eps_gr_3y", block), \
            "eps_gr_3y must fillna(-1) so missing history fails the >= 0 gate conservatively"


# ─── N Criterion — Near New High ──────────────────────────────────────────────

class TestCriterionN:
    def test_n_distance_threshold(self):
        """N criterion: within 15% of 52-week high."""
        block = _fw_can_slim_block(_scoring_src())
        assert re.search(r"dist_wh_cs\s*<=\s*15\b", block), \
            "fw_can_slim N criterion must gate dist_52wh <= 15"

    def test_n_not_25_threshold(self):
        """N criterion must be 15%, not the looser 25% (that's SEPA)."""
        block = _fw_can_slim_block(_scoring_src())
        assert not re.search(r"dist_wh_cs\s*<=\s*25\b", block), \
            "CAN SLIM N criterion uses <= 15% (not 25% — that's SEPA's threshold)"

    def test_n_fillna_conservative(self):
        """Missing 52WH distance must fail N criterion (not pass)."""
        block = _fw_can_slim_block(_scoring_src())
        assert re.search(r"dist_wh_cs\s*=.*fillna\(999", block) or \
               re.search(r"fillna\(999.*dist_52wh", block) or \
               re.search(r"dist_52wh.*fillna\(999\)", block), \
            "dist_52wh must fillna(999) so missing price history fails the <= 15 gate"


# ─── S Criterion — Supply and Demand ──────────────────────────────────────────

class TestCriterionS:
    def test_s_volume_threshold(self):
        """S criterion: volume ratio >= 1.5 (50%+ above average)."""
        block = _fw_can_slim_block(_scoring_src())
        assert re.search(r"vol_r_cs\s*>=\s*1\.5\b", block), \
            "fw_can_slim S criterion must gate vol_ratio >= 1.5"

    def test_s_not_1_4_threshold(self):
        """S criterion must be 1.5, not the looser 1.4."""
        block = _fw_can_slim_block(_scoring_src())
        assert not re.search(r"vol_r_cs\s*>=\s*1\.4\b", block), \
            "S criterion must use >= 1.5 (not 1.4)"

    def test_s_reads_vol_ratio(self):
        """vol_ratio must be loaded into the CAN SLIM block."""
        block = _fw_can_slim_block(_scoring_src())
        assert "vol_ratio" in block, \
            "fw_can_slim must read vol_ratio for S criterion"


# ─── L Criterion — Leader vs Laggard ─────────────────────────────────────────

class TestCriterionL:
    def test_l_rs_percentile_threshold(self):
        """L criterion: RS percentile rank >= 80 (top 20% of universe)."""
        block = _fw_can_slim_block(_scoring_src())
        assert re.search(r"rs_pctrank_cs\s*>=\s*80\b", block), \
            "fw_can_slim L criterion must gate rs_pctrank_cs >= 80"

    def test_l_reads_d47_rs_composite(self):
        """d47_rs_composite must be loaded into the CAN SLIM block."""
        block = _fw_can_slim_block(_scoring_src())
        assert "d47_rs_composite" in block, \
            "fw_can_slim L criterion must read d47_rs_composite for RS Rating"

    def test_l_uses_pct_rank(self):
        """L criterion must compute percentile rank (not raw value)."""
        block = _fw_can_slim_block(_scoring_src())
        assert "_pct_rank" in block, \
            "fw_can_slim L criterion must compute _pct_rank() for RS composite"

    def test_l_ascending_true(self):
        """L pct_rank must be ascending=True (higher RS → higher rank)."""
        block = _fw_can_slim_block(_scoring_src())
        assert re.search(r"_pct_rank.*ascending\s*=\s*True", block), \
            "RS pct_rank must use ascending=True (higher RS composite = higher rank)"


# ─── I Criterion — Institutional Sponsorship ──────────────────────────────────

class TestCriterionI:
    def test_i_fii_or_dii_gate(self):
        """I criterion: FII or DII buying in latest quarter."""
        block = _fw_can_slim_block(_scoring_src())
        assert re.search(r"fii_cs\s*>\s*0.*dii_cs\s*>\s*0|dii_cs\s*>\s*0.*fii_cs\s*>\s*0", block) or \
               re.search(r"\(fii_cs\s*>\s*0\)\s*\|\s*\(dii_cs\s*>\s*0\)", block), \
            "fw_can_slim I criterion must gate (change_fii_lq > 0) OR (change_dii_lq > 0)"

    def test_i_reads_change_fii_lq(self):
        """change_fii_lq must be loaded into the CAN SLIM block."""
        block = _fw_can_slim_block(_scoring_src())
        assert "change_fii_lq" in block, \
            "fw_can_slim must read change_fii_lq for institutional sponsorship"

    def test_i_reads_change_dii_lq(self):
        """change_dii_lq must be loaded into the CAN SLIM block."""
        block = _fw_can_slim_block(_scoring_src())
        assert "change_dii_lq" in block, \
            "fw_can_slim must read change_dii_lq for institutional sponsorship"


# ─── M Criterion — Market Direction ──────────────────────────────────────────

class TestCriterionM:
    def test_m_market_regime_gate_present(self):
        """M criterion: fw_can_slim must reference detected_market_regime."""
        block = _fw_can_slim_block(_scoring_src())
        assert "detected_market_regime" in block, \
            "fw_can_slim M criterion must read df.attrs['detected_market_regime'] — internal breadth consensus"

    def test_m_blocks_on_bear(self):
        """M criterion must block (return False) when regime is BEAR."""
        block = _fw_can_slim_block(_scoring_src())
        assert re.search(r"['\"]BEAR['\"]", block), \
            "fw_can_slim M criterion must specifically check for BEAR regime"
        # The BEAR condition must result in blocking (not in BEAR = passes)
        assert re.search(r"!=\s*['\"]BEAR['\"]", block), \
            "fw_can_slim must block if regime == BEAR via '!= BEAR' condition"

    def test_m_fallback_to_sideways(self):
        """M criterion must default to SIDEWAYS (pass) when called outside run_full_scoring."""
        block = _fw_can_slim_block(_scoring_src())
        assert re.search(r"attrs\.get\(.*['\"]SIDEWAYS['\"]", block), \
            "M criterion must fallback to SIDEWAYS (not BEAR) when detected_market_regime is absent"

    def test_m_not_commented_out(self):
        """M criterion comment must say it IS implemented, not 'NOT IMPLEMENTABLE'."""
        block = _fw_can_slim_block(_scoring_src())
        assert "NOT IMPLEMENTABLE" not in block, \
            "Old comment 'NOT IMPLEMENTABLE' must be removed — M criterion is now wired to detect_market_regime"

    def test_m_uses_df_attrs(self):
        """M criterion must read regime from df.attrs (set by run_full_scoring)."""
        block = _fw_can_slim_block(_scoring_src())
        assert "df.attrs" in block, \
            "M criterion must access regime via df.attrs — not a function call inside compute_qglp_score"


# ─── Output columns ───────────────────────────────────────────────────────────

class TestOutputColumns:
    def test_can_slim_pass_column_assigned(self):
        """df['can_slim_pass'] must be assigned immediately after fw_can_slim."""
        block = _fw_can_slim_block(_scoring_src())
        assert re.search(r'df\["can_slim_pass"\]\s*=\s*fw_can_slim\.astype\(int\)', block), \
            "fw_can_slim block must assign df['can_slim_pass'] = fw_can_slim.astype(int)"

    def test_can_slim_score_column_assigned(self):
        """df['can_slim_score'] must be assigned inside the fw_can_slim block."""
        block = _fw_can_slim_block(_scoring_src())
        assert 'df["can_slim_score"]' in block or "df['can_slim_score']" in block, \
            "fw_can_slim block must assign df['can_slim_score'] (criteria count)"

    def test_fw_str_includes_can_slim_label(self):
        """frameworks_passed string builder must use 'CAN SLIM|' label."""
        src = _scoring_src()
        assert re.search(r'fw_can_slim.*"CAN SLIM\|"', src, re.DOTALL), \
            "fw_str builder must map fw_can_slim → 'CAN SLIM|' label in frameworks_passed"

    def test_can_slim_pass_uses_astype_int(self):
        """can_slim_pass must be integer (0 or 1), not boolean."""
        block = _fw_can_slim_block(_scoring_src())
        assert "fw_can_slim.astype(int)" in block, \
            "can_slim_pass must be .astype(int) — not a boolean Series"


# ─── Vectorization contract ───────────────────────────────────────────────────

class TestVectorizationContract:
    def test_no_iterrows_in_can_slim_block(self):
        """fw_can_slim must use zero row-iteration."""
        block = _fw_can_slim_block(_scoring_src())
        assert "iterrows" not in block, \
            "fw_can_slim must not use iterrows() — pure vectorized NumPy/Pandas operations only"

    def test_no_apply_in_can_slim_block(self):
        """fw_can_slim must use zero df.apply(axis=1)."""
        block = _fw_can_slim_block(_scoring_src())
        assert "apply(" not in block, \
            "fw_can_slim must not use df.apply() — zero apply() contract for all scoring paths"

    def test_fw_can_slim_is_boolean_mask(self):
        """fw_can_slim must be defined as a chained boolean expression, not a loop."""
        block = _fw_can_slim_block(_scoring_src())
        assert "fw_can_slim = (" in block, \
            "fw_can_slim must be a single vectorized boolean mask expression"


# ─── Spec–code threshold alignment ───────────────────────────────────────────

class TestSpecCodeAlignment:
    """Parse spec JSON thresholds and assert they exactly match code literals."""

    def test_c_eps_threshold_matches_spec(self):
        spec = _load_spec()
        t = spec["c_current_quarterly_growth"]["quarterly_eps_growth"]["threshold"]
        block = _fw_can_slim_block(_scoring_src())
        pattern = rf"q_eps_cs\s*>=\s*{re.escape(str(t))}"
        assert re.search(pattern, block), \
            f"C EPS threshold in code must match spec ({t})"

    def test_c_eps_uses_q_eps_yoy_precomputed(self):
        """C criterion spec must reference q_eps_yoy as the precomputed metric."""
        spec = _load_spec()
        metric = spec["c_current_quarterly_growth"]["quarterly_eps_growth"]["metric"]
        assert metric == "q_eps_yoy", \
            f"C criterion spec metric must be 'q_eps_yoy' (precomputed EPS per share), got '{metric}'"

    def test_c_sales_threshold_matches_spec(self):
        spec = _load_spec()
        t = spec["c_current_quarterly_growth"]["quarterly_sales_growth"]["threshold"]
        block = _fw_can_slim_block(_scoring_src())
        pattern = rf"q_rev_cs\s*>=\s*{re.escape(str(t))}"
        assert re.search(pattern, block), \
            f"C sales threshold in code must match spec ({t})"

    def test_a_cagr_threshold_matches_spec(self):
        spec = _load_spec()
        t = spec["a_annual_earnings_increases"]["annual_eps_cagr_5y"]["threshold"]
        block = _fw_can_slim_block(_scoring_src())
        assert re.search(rf"eps_gr_cs\s*>=\s*{re.escape(str(int(t)))}\b", block), \
            f"A CAGR threshold in code must match spec ({t})"

    def test_a_roe_threshold_matches_spec(self):
        spec = _load_spec()
        t = spec["a_annual_earnings_increases"]["return_on_equity"]["threshold"]
        block = _fw_can_slim_block(_scoring_src())
        assert re.search(rf"roe_cs\s*>=\s*{re.escape(str(int(t)))}\b", block), \
            f"A ROE threshold in code must match spec ({t})"

    def test_n_distance_threshold_matches_spec(self):
        spec = _load_spec()
        t = spec["n_new_highs_and_catalysts"]["distance_from_52w_high_pct"]["threshold"]
        block = _fw_can_slim_block(_scoring_src())
        assert re.search(rf"dist_wh_cs\s*<=\s*{re.escape(str(int(t)))}\b", block), \
            f"N distance threshold in code must match spec ({t})"

    def test_s_volume_threshold_matches_spec(self):
        spec = _load_spec()
        t = spec["s_supply_and_demand"]["volume_ratio_sma20"]["threshold"]
        block = _fw_can_slim_block(_scoring_src())
        assert re.search(rf"vol_r_cs\s*>=\s*{re.escape(str(t))}\b", block), \
            f"S volume threshold in code must match spec ({t})"

    def test_l_rs_percentile_matches_spec(self):
        spec = _load_spec()
        t = spec["l_leader_vs_laggard"]["relative_strength_percentile"]["threshold"]
        block = _fw_can_slim_block(_scoring_src())
        assert re.search(rf"rs_pctrank_cs\s*>=\s*{re.escape(str(int(t)))}\b", block), \
            f"L RS percentile in code must match spec ({t})"


# ─── A Criterion Upgrade — PAT Step-Growth ────────────────────────────────────

class TestCriterionAPatStepGrowth:
    def test_a_pat_step_growth_in_fw_can_slim(self):
        """A criterion: pat_step_ok must be a hard gate in fw_can_slim (O'Neil Ch.4 direct step verification)."""
        block = _fw_can_slim_block(_scoring_src())
        assert "pat_step_ok" in block, \
            "fw_can_slim A criterion must include pat_step_ok (3-year PAT step-growth hard gate)"

    def test_a_pat_step_growth_chain(self):
        """PAT step-growth must verify all 3 year-over-year steps (current > 1YB > 2YB > 3YB)."""
        block = _fw_can_slim_block(_scoring_src())
        assert re.search(r"pat_cs\s*>\s*pat_1yb_cs", block), \
            "pat_step_ok must check pat_cs > pat_1yb_cs (current year > 1 year back)"
        assert re.search(r"pat_1yb_cs\s*>\s*pat_2yb_cs", block), \
            "pat_step_ok must check pat_1yb_cs > pat_2yb_cs (1 year back > 2 years back)"
        assert re.search(r"pat_2yb_cs\s*>\s*pat_3yb_cs", block), \
            "pat_step_ok must check pat_2yb_cs > pat_3yb_cs (2 years back > 3 years back)"

    def test_a_pat_reads_all_back_years(self):
        """All four PAT series (current + 3 back-years) must be loaded from df."""
        block = _fw_can_slim_block(_scoring_src())
        for col in ["pat", "pat_1yb", "pat_2yb", "pat_3yb"]:
            assert col in block, \
                f"fw_can_slim A criterion must load df['{col}'] for PAT step-growth check"

    def test_a_pat_2yb_3yb_in_data_engine_column_map(self):
        """PAT 2 Years Back and PAT 3 Years Back must be mapped in data_engine.py COLUMN_MAP."""
        src = Path(ROOT / "core" / "data_engine.py").read_text(encoding="utf-8")
        assert "PAT 2 Years Back" in src and "pat_2yb" in src, \
            "data_engine.py must map 'PAT 2 Years Back' -> 'pat_2yb' in COLUMN_MAP"
        assert "PAT 3 Years Back" in src and "pat_3yb" in src, \
            "data_engine.py must map 'PAT 3 Years Back' -> 'pat_3yb' in COLUMN_MAP"

    def test_a_pat_step_nan_fails_conservatively(self):
        """Missing back-year PAT data must fail conservatively — NaN comparisons return False."""
        block = _fw_can_slim_block(_scoring_src())
        # No fillna(0) on the PAT series — NaN comparisons produce False natively in pandas
        # Confirm no fillna(0) on pat_cs, pat_1yb_cs, pat_2yb_cs, pat_3yb_cs
        assert not re.search(r"pat_cs\s*=.*fillna\(0\)", block), \
            "pat_cs must NOT fillna(0) — NaN should fail step-growth gate conservatively"

    def test_a_pat_step_growth_in_spec(self):
        """canslim_technical_specs.json must document the PAT step-growth gate."""
        spec = _load_spec()
        assert "pat_step_growth_3y" in spec["a_annual_earnings_increases"], \
            "Spec must document pat_step_growth_3y in a_annual_earnings_increases section"


# ─── S Criterion Upgrade — Float Retraction / Share Buyback ──────────────────

class TestCriterionSBuyback:
    def test_s_buyback_in_can_slim_score(self):
        """S criterion bonus: buyback_cs must be included in can_slim_score."""
        block = _fw_can_slim_block(_scoring_src())
        assert "buyback_cs" in block, \
            "can_slim_score must include buyback_cs (float retraction / share buyback bonus)"

    def test_s_buyback_not_in_fw_can_slim_hard_gate(self):
        """buyback_cs must be a score bonus only — NOT a hard gate in fw_can_slim boolean chain.
        Primary S gate is volume surge; buyback is O'Neil's secondary supply-constraint signal."""
        block = _fw_can_slim_block(_scoring_src())
        # fw_can_slim = (...) must not contain buyback_cs inside the boolean chain
        fw_def = re.search(r"fw_can_slim\s*=\s*\((.*?)\)", block, re.DOTALL)
        assert fw_def is not None, "fw_can_slim = (...) boolean chain not found"
        assert "buyback_cs" not in fw_def.group(1), \
            "buyback_cs must NOT appear inside fw_can_slim = (...) — it is a score bonus, not a hard gate"

    def test_s_buyback_reads_equity_shares(self):
        """equity_shares and equity_shares_1yb must be loaded in the CAN SLIM block."""
        block = _fw_can_slim_block(_scoring_src())
        assert "equity_shares" in block, \
            "fw_can_slim must read equity_shares for buyback check"
        assert "equity_shares_1yb" in block, \
            "fw_can_slim must read equity_shares_1yb for buyback check"

    def test_s_buyback_direction_correct(self):
        """Float retraction check: current shares <= prior year shares (declining float)."""
        block = _fw_can_slim_block(_scoring_src())
        assert re.search(r"eq_shr_cs\s*<=\s*eq_shr_1yb_cs", block), \
            "buyback check must be eq_shr_cs <= eq_shr_1yb_cs (current float <= prior year float)"

    def test_s_buyback_missing_data_fails_conservatively(self):
        """Both equity_shares values must be present — notna() guard required."""
        block = _fw_can_slim_block(_scoring_src())
        assert re.search(r"eq_shr_cs\.notna\(\)", block), \
            "buyback check must guard eq_shr_cs.notna() — missing data fails conservatively"
        assert re.search(r"eq_shr_1yb_cs\.notna\(\)", block), \
            "buyback check must guard eq_shr_1yb_cs.notna() — missing data fails conservatively"

    def test_s_buyback_in_spec(self):
        """canslim_technical_specs.json must document float_retraction_buyback in s_supply_and_demand."""
        spec = _load_spec()
        assert "float_retraction_buyback" in spec["s_supply_and_demand"], \
            "Spec must document float_retraction_buyback in s_supply_and_demand section"


# ─── L Criterion Upgrade — IBD-Weighted RS Formula ───────────────────────────

class TestCriterionLIBDWeighted:
    def test_l_ibd_weighted_formula_in_data_engine(self):
        """d47_rs_composite in data_engine.py must use IBD weights (not simple equal average)."""
        src = Path(ROOT / "core" / "data_engine.py").read_text(encoding="utf-8")
        assert re.search(r"crs_50d.*\*\s*0\.40|0\.40\s*\*.*crs_50d", src), \
            "data_engine.py d47_rs_composite must weight crs_50d at 0.40 (IBD recent-quarter double weight)"
        assert re.search(r"crs_26w.*\*\s*0\.30|0\.30\s*\*.*crs_26w", src), \
            "data_engine.py d47_rs_composite must weight crs_26w at 0.30"
        assert re.search(r"crs_52w.*\*\s*0\.30|0\.30\s*\*.*crs_52w", src), \
            "data_engine.py d47_rs_composite must weight crs_52w at 0.30"

    def test_l_ibd_weights_sum_to_one(self):
        """IBD RS weights must sum to 1.0 (0.40 + 0.30 + 0.30)."""
        assert abs(0.40 + 0.30 + 0.30 - 1.0) < 1e-9, \
            "IBD RS weights 0.40 + 0.30 + 0.30 must sum to exactly 1.0"

    def test_l_not_simple_average(self):
        """d47_rs_composite must NOT use the old simple /3.0 equal average.
        Checks only non-comment code lines to avoid false positives from '/ 30%' in comments."""
        src = Path(ROOT / "core" / "data_engine.py").read_text(encoding="utf-8")
        d47_start = src.find("D47")
        d47_end = src.find("D48", d47_start)
        d47_block = src[d47_start:d47_end]
        # Strip comment lines before checking — avoids "40% recent / 30% mid" false positive
        code_lines = [
            ln for ln in d47_block.splitlines()
            if not ln.lstrip().startswith("#")
        ]
        code_only = "\n".join(code_lines)
        assert not re.search(r"\)\s*/\s*3(?:\.0)?\b", code_only), \
            "d47_rs_composite must not use simple ) / 3.0 equal average — IBD weighted formula required"

    def test_l_ibd_formula_in_spec(self):
        """Spec must document the IBD weighting formula for L criterion."""
        spec = _load_spec()
        ibd_formula = spec["l_leader_vs_laggard"]["relative_strength_percentile"].get("ibd_formula", "")
        assert "0.40" in ibd_formula and "0.30" in ibd_formula, \
            "Spec L criterion must document ibd_formula with 0.40 and 0.30 weights"


# ─── Spec Version ─────────────────────────────────────────────────────────────

class TestSpecVersion:
    def test_spec_version_is_2_2(self):
        """Spec version must be 2.2 (VCP volume + RS uptrend additions)."""
        spec = _load_spec()
        ver = spec["_meta"]["version"]
        assert "2.2" in ver, \
            f"canslim_technical_specs.json must be version 2.2 (vcp-volume-rs-uptrend), got '{ver}'"

    def test_spec_score_max_components_is_17(self):
        """implementation_mapping must document 17 max can_slim_score components (S3 + L2 bonuses added)."""
        spec = _load_spec()
        n = spec["implementation_mapping"]["_score_max_components"]
        assert n == 17, \
            f"implementation_mapping._score_max_components must be 17 after v2.2 additions, got {n}"


# ─── C+A Criterion — EPS Annual Acceleration (v2.1) ──────────────────────────

class TestCriterionCEPSAcceleration:
    """O'Neil Ch.3+4: 'The rate of earnings increase should be getting larger in each
    of the past few quarters or years.' eps_gr_yoy > eps_gr_3y = most recent annual
    growth beating the 3-year baseline = acceleration confirmed."""

    def test_eps_accel_in_can_slim_score(self):
        """EPS acceleration bonus (eps_accel_cs) must appear in can_slim_score."""
        block = _fw_can_slim_block(_scoring_src())
        assert "eps_accel_cs" in block, \
            "can_slim_score must include eps_accel_cs (EPS annual acceleration bonus, O'Neil Ch.3+4)"

    def test_eps_accel_uses_yoy_vs_3y_cagr(self):
        """EPS acceleration must compare eps_yoy_cs > eps_gr_3y_cs (recent beats 3Y baseline)."""
        block = _fw_can_slim_block(_scoring_src())
        assert re.search(r"eps_yoy_cs\s*>\s*eps_gr_3y_cs", block), \
            "eps_accel_cs must be defined as (eps_yoy_cs > eps_gr_3y_cs) — recent YoY beating 3Y CAGR"

    def test_eps_accel_not_a_hard_gate_in_fw_can_slim(self):
        """EPS acceleration must be a score bonus only — NOT inside the fw_can_slim hard gate chain."""
        block = _fw_can_slim_block(_scoring_src())
        fw_def = re.search(r"fw_can_slim\s*=\s*\((.*?)\)", block, re.DOTALL)
        assert fw_def is not None, "fw_can_slim = (...) boolean chain not found"
        assert "eps_accel_cs" not in fw_def.group(1), \
            "eps_accel_cs must NOT be inside fw_can_slim = (...) — it is a score bonus, not a hard gate"

    def test_eps_accel_in_spec(self):
        """Spec must document eps_annual_acceleration in c_current_quarterly_growth."""
        spec = _load_spec()
        assert "eps_annual_acceleration" in spec["c_current_quarterly_growth"], \
            "Spec must document eps_annual_acceleration in c_current_quarterly_growth section"

    def test_eps_accel_spec_type_is_score_bonus(self):
        """Spec eps_annual_acceleration must have type SCORE_BONUS (not CANONICAL_LIMIT)."""
        spec = _load_spec()
        t = spec["c_current_quarterly_growth"]["eps_annual_acceleration"]["type"]
        assert t == "SCORE_BONUS", \
            f"eps_annual_acceleration must be type SCORE_BONUS (not a hard gate), got '{t}'"


# ─── A Criterion — OPM Expansion (v2.1) ──────────────────────────────────────

class TestCriterionAOPMExpansion:
    """O'Neil Ch.4: 'Look for annual pre-tax profit margins at new peak levels — this
    confirms pricing power and cost discipline.' opm > opm_1yb = margin expanding YoY."""

    def test_opm_expand_in_can_slim_score(self):
        """OPM expansion bonus (opm_expand_cs) must appear in can_slim_score."""
        block = _fw_can_slim_block(_scoring_src())
        assert "opm_expand_cs" in block, \
            "can_slim_score must include opm_expand_cs (OPM expansion bonus, O'Neil Ch.4)"

    def test_opm_expand_compares_opm_to_opm_1yb(self):
        """OPM expansion must compare current opm > prior year opm_1yb."""
        block = _fw_can_slim_block(_scoring_src())
        assert re.search(r"opm_cs\s*>\s*opm_1yb_cs", block), \
            "opm_expand_cs must be (opm_cs > opm_1yb_cs) — current OPM exceeds prior year OPM"

    def test_opm_expand_reads_opm_and_opm_1yb(self):
        """Both opm and opm_1yb must be loaded in the CAN SLIM block."""
        block = _fw_can_slim_block(_scoring_src())
        assert "opm" in block, \
            "fw_can_slim block must load opm for OPM expansion check"
        assert "opm_1yb" in block, \
            "fw_can_slim block must load opm_1yb for OPM expansion check"

    def test_opm_expand_not_a_hard_gate_in_fw_can_slim(self):
        """OPM expansion must be a score bonus only — NOT inside the fw_can_slim hard gate chain."""
        block = _fw_can_slim_block(_scoring_src())
        fw_def = re.search(r"fw_can_slim\s*=\s*\((.*?)\)", block, re.DOTALL)
        assert fw_def is not None, "fw_can_slim = (...) boolean chain not found"
        assert "opm_expand_cs" not in fw_def.group(1), \
            "opm_expand_cs must NOT be inside fw_can_slim = (...) — it is a score bonus, not a hard gate"

    def test_opm_expand_in_spec(self):
        """Spec must document operating_margin_expansion in a_annual_earnings_increases."""
        spec = _load_spec()
        assert "operating_margin_expansion" in spec["a_annual_earnings_increases"], \
            "Spec must document operating_margin_expansion in a_annual_earnings_increases section"

    def test_opm_expand_spec_type_is_score_bonus(self):
        """Spec operating_margin_expansion must have type SCORE_BONUS (not CANONICAL_LIMIT)."""
        spec = _load_spec()
        t = spec["a_annual_earnings_increases"]["operating_margin_expansion"]["type"]
        assert t == "SCORE_BONUS", \
            f"operating_margin_expansion must be type SCORE_BONUS (not a hard gate), got '{t}'"

    def test_opm_mapped_in_data_engine(self):
        """opm and opm_1yb must be mapped in data_engine.py RATIO_COLS."""
        src = Path(ROOT / "core" / "data_engine.py").read_text(encoding="utf-8")
        assert '"OPM"' in src and '"opm"' in src, \
            "data_engine.py must map 'OPM' -> 'opm' for OPM expansion check"
        assert '"OPM 1 Year Back"' in src and '"opm_1yb"' in src, \
            "data_engine.py must map 'OPM 1 Year Back' -> 'opm_1yb' for OPM expansion check"


# ─── Missing CSV Columns Gap ─────────────────────────────────────────────────

class TestMissingCSVColumnsGap:
    """Verifies that the spec correctly documents the quarterly back-series columns
    that are listed in ALL_INDICATOR_AVAILABLES but absent from the user's 6 CSV exports."""

    def test_spec_has_missing_csv_columns_gap(self):
        """Spec must have a missing_csv_columns_gap section."""
        spec = _load_spec()
        assert "missing_csv_columns_gap" in spec, \
            "canslim_technical_specs.json must document missing_csv_columns_gap"

    def test_spec_gap_documents_quarterly_eps_back_series(self):
        """Gap section must document missing EPS quarterly back-series columns."""
        spec = _load_spec()
        gap = spec["missing_csv_columns_gap"]
        assert "quarterly_eps_back_series" in gap, \
            "missing_csv_columns_gap must document quarterly_eps_back_series absence"
        cols = gap["quarterly_eps_back_series"]["screener_columns"]
        assert "EPS 1 Quarter Back" in cols, \
            "quarterly_eps_back_series must list 'EPS 1 Quarter Back' as absent from exports"

    def test_spec_gap_acceleration_in_known_gaps(self):
        """known_gaps must include the C criterion quarterly acceleration limitation."""
        spec = _load_spec()
        gap_texts = [g["gap"] for g in spec["known_gaps"]]
        assert any("quarterly" in g.lower() and "acceleration" in g.lower() for g in gap_texts), \
            "known_gaps must include C criterion quarterly acceleration gap (requires back-quarter EPS series)"


# ─── S3-Bonus: Minervini VCP Volume Contraction Pattern (v2.2) ───────────────

class TestSBonusVCPVolume:
    """O'Neil Ch.6 + Minervini SEPA: highest-conviction breakouts follow volume drying
    up in the base (supply absorbed) then exploding on the pivot. vcp_vol_cs captures
    both conditions simultaneously with zero row-iteration."""

    def test_vcp_vol_in_can_slim_score(self):
        """S3-bonus: vcp_vol_cs must appear in can_slim_score."""
        block = _fw_can_slim_block(_scoring_src())
        assert "vcp_vol_cs" in block, \
            "can_slim_score must include vcp_vol_cs (Minervini VCP volume bonus, O'Neil Ch.6)"

    def test_vcp_vol_not_in_fw_can_slim_hard_gate(self):
        """VCP bonus must be score-only — NOT inside the fw_can_slim boolean chain."""
        block = _fw_can_slim_block(_scoring_src())
        fw_def = re.search(r"fw_can_slim\s*=\s*\((.*?)\)", block, re.DOTALL)
        assert fw_def is not None, "fw_can_slim = (...) boolean chain not found"
        assert "vcp_vol_cs" not in fw_def.group(1), \
            "vcp_vol_cs must NOT appear inside fw_can_slim = (...) — it is a score bonus, not a hard gate"

    def test_vcp_vol_uses_5d_below_20d(self):
        """VCP dryup condition must check vol_sma_5d < vol_sma_20d (recent volume contracting)."""
        block = _fw_can_slim_block(_scoring_src())
        assert re.search(r"_vcp_5d\s*<\s*_vcp_20d", block), \
            "vcp_vol_cs must gate _vcp_5d < _vcp_20d (5D avg below 20D avg = volume drying up)"

    def test_vcp_vol_combines_dryup_and_surge(self):
        """VCP must require BOTH dryup (5D<20D) AND current surge (vol_r_cs >= 1.5)."""
        block = _fw_can_slim_block(_scoring_src())
        assert re.search(r"_vcp_5d\s*<\s*_vcp_20d", block), \
            "VCP must include dryup condition (_vcp_5d < _vcp_20d)"
        assert re.search(r"vol_r_cs\s*>=\s*1\.5", block), \
            "VCP must include surge condition (vol_r_cs >= 1.5)"

    def test_vcp_vol_guards_notna_and_positive(self):
        """VCP must guard both SMA series as notna() and 20D > 0 before comparing."""
        block = _fw_can_slim_block(_scoring_src())
        assert re.search(r"_vcp_5d\.notna\(\)", block), \
            "vcp_vol_cs must guard _vcp_5d.notna() — missing 5D vol fails conservatively"
        assert re.search(r"_vcp_20d\.notna\(\)", block), \
            "vcp_vol_cs must guard _vcp_20d.notna() — missing 20D vol fails conservatively"
        assert re.search(r"_vcp_20d\s*>\s*0", block), \
            "vcp_vol_cs must guard _vcp_20d > 0 to prevent zero-baseline false signals"

    def test_vcp_vol_in_spec(self):
        """Spec must document vcp_volume_contraction_pattern in s_supply_and_demand."""
        spec = _load_spec()
        assert "vcp_volume_contraction_pattern" in spec["s_supply_and_demand"], \
            "Spec must document vcp_volume_contraction_pattern in s_supply_and_demand"

    def test_vcp_spec_type_is_score_bonus(self):
        """Spec vcp_volume_contraction_pattern must be SCORE_BONUS (not a hard gate)."""
        spec = _load_spec()
        t = spec["s_supply_and_demand"]["vcp_volume_contraction_pattern"]["type"]
        assert t == "SCORE_BONUS", \
            f"vcp_volume_contraction_pattern type must be SCORE_BONUS, got '{t}'"

    def test_vcp_data_source_in_spec(self):
        """Spec must document the data_source for VCP (vol_sma_5d and vol_sma_20d)."""
        spec = _load_spec()
        src = spec["s_supply_and_demand"]["vcp_volume_contraction_pattern"].get("data_source", "")
        assert "vol_sma_5d" in src and "vol_sma_20d" in src, \
            "Spec VCP data_source must reference vol_sma_5d and vol_sma_20d"


# ─── L2-Bonus: RS Line Multi-Timeframe Uptrend (v2.2) ────────────────────────

class TestLBonusRSUptrend:
    """O'Neil Ch.7: buy stocks whose RS LINE is trending upward — ideally making new
    highs before price breaks out. rs_uptrend_cs measures DIRECTION (ascending) while
    rs_pctrank_cs >= 80 measures LEVEL (top 20%). Together they confirm level + direction."""

    def test_rs_uptrend_in_can_slim_score(self):
        """L2-bonus: rs_uptrend_cs must appear in can_slim_score."""
        block = _fw_can_slim_block(_scoring_src())
        assert "rs_uptrend_cs" in block, \
            "can_slim_score must include rs_uptrend_cs (RS line uptrend bonus, O'Neil Ch.7)"

    def test_rs_uptrend_not_in_fw_can_slim_hard_gate(self):
        """RS uptrend bonus must be score-only — NOT inside the fw_can_slim boolean chain."""
        block = _fw_can_slim_block(_scoring_src())
        fw_def = re.search(r"fw_can_slim\s*=\s*\((.*?)\)", block, re.DOTALL)
        assert fw_def is not None, "fw_can_slim = (...) boolean chain not found"
        assert "rs_uptrend_cs" not in fw_def.group(1), \
            "rs_uptrend_cs must NOT appear inside fw_can_slim = (...) — it is a score bonus, not a hard gate"

    def test_rs_uptrend_checks_50d_above_26w(self):
        """RS uptrend must check _rs_50d > _rs_26w (short-term RS above medium-term)."""
        block = _fw_can_slim_block(_scoring_src())
        assert re.search(r"_rs_50d\s*>\s*_rs_26w", block), \
            "rs_uptrend_cs must check _rs_50d > _rs_26w (50D RS above 26W RS)"

    def test_rs_uptrend_checks_26w_above_52w(self):
        """RS uptrend must check _rs_26w > _rs_52w (medium-term RS above long-term)."""
        block = _fw_can_slim_block(_scoring_src())
        assert re.search(r"_rs_26w\s*>\s*_rs_52w", block), \
            "rs_uptrend_cs must check _rs_26w > _rs_52w (26W RS above 52W RS)"

    def test_rs_uptrend_guards_all_three_notna(self):
        """All three CRS timeframes must be guarded as notna() — direction unconfirmed if any missing."""
        block = _fw_can_slim_block(_scoring_src())
        assert re.search(r"_rs_50d\.notna\(\)", block), \
            "rs_uptrend_cs must guard _rs_50d.notna()"
        assert re.search(r"_rs_26w\.notna\(\)", block), \
            "rs_uptrend_cs must guard _rs_26w.notna()"
        assert re.search(r"_rs_52w\.notna\(\)", block), \
            "rs_uptrend_cs must guard _rs_52w.notna()"

    def test_rs_uptrend_reads_crs_columns(self):
        """crs_50d, crs_26w, crs_52w must be loaded as local variables in the block."""
        block = _fw_can_slim_block(_scoring_src())
        assert "crs_50d" in block, \
            "fw_can_slim block must access crs_50d for RS uptrend check"
        assert "crs_26w" in block, \
            "fw_can_slim block must access crs_26w for RS uptrend check"
        assert "crs_52w" in block, \
            "fw_can_slim block must access crs_52w for RS uptrend check"

    def test_rs_uptrend_in_spec(self):
        """Spec must document rs_line_uptrend_multiframe in l_leader_vs_laggard."""
        spec = _load_spec()
        assert "rs_line_uptrend_multiframe" in spec["l_leader_vs_laggard"], \
            "Spec must document rs_line_uptrend_multiframe in l_leader_vs_laggard"

    def test_rs_uptrend_spec_type_is_score_bonus(self):
        """Spec rs_line_uptrend_multiframe must be SCORE_BONUS (not a hard gate)."""
        spec = _load_spec()
        t = spec["l_leader_vs_laggard"]["rs_line_uptrend_multiframe"]["type"]
        assert t == "SCORE_BONUS", \
            f"rs_line_uptrend_multiframe type must be SCORE_BONUS, got '{t}'"
