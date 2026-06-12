"""
test_coffee_can_contract.py

Contract tests that parse docs/coffee_can_financial_specs.json and assert that
core/scoring_engine.py and core/config.py honour every quantitative threshold
declared in the ledger. If the ledger says ROCE ≥ 15%, the code must say >= 15.
Drift from the spec causes a test failure — not a silent bug.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
SPEC_PATH = ROOT / "docs" / "coffee_can_financial_specs.json"
SCORING_PATH = ROOT / "core" / "scoring_engine.py"
DATA_ENGINE_PATH = ROOT / "core" / "data_engine.py"
CONFIG_PATH = ROOT / "config.py"


def _load_spec():
    with open(SPEC_PATH, encoding="utf-8") as f:
        return json.load(f)


def _scoring_src():
    return SCORING_PATH.read_text(encoding="utf-8")


def _data_engine_src():
    return DATA_ENGINE_PATH.read_text(encoding="utf-8")


# ── Spec file integrity ────────────────────────────────────────────────────────

def test_spec_file_exists():
    """coffee_can_financial_specs.json must exist in docs/."""
    assert SPEC_PATH.exists(), f"Spec file not found: {SPEC_PATH}"


def test_spec_required_top_level_keys():
    spec = _load_spec()
    for key in ("twin_filters_core", "clean_accounts_firewall",
                "leverage_governance_gates", "implementation_mapping"):
        assert key in spec, f"Missing top-level key '{key}' in spec"


def test_spec_non_fin_roce_threshold():
    spec = _load_spec()
    roce = spec["twin_filters_core"]["non_financial_companies"]["roce_median_10y_floor_pct"]
    assert roce == 15.0, f"Spec ROCE 10Y floor should be 15.0, got {roce}"


def test_spec_financial_roe_threshold():
    spec = _load_spec()
    roe = spec["twin_filters_core"]["financial_companies_bfsi"]["roe_median_10y_floor_pct"]
    assert roe == 15.0, f"Spec ROE 10Y floor should be 15.0, got {roe}"


def test_spec_revenue_10y_threshold():
    spec = _load_spec()
    rev = spec["twin_filters_core"]["non_financial_companies"]["revenue_growth_10y_cagr_floor_pct"]
    assert rev == 10.0, f"Spec revenue 10Y CAGR floor should be 10.0, got {rev}"


def test_spec_individual_year_floor():
    spec = _load_spec()
    floor = spec["twin_filters_core"]["non_financial_companies"]["individual_year_floor_pct"]
    assert floor == 5.0, f"Spec individual year floor should be 5.0 (pandemic-adjusted), got {floor}"


def test_spec_cfo_ebitda_threshold():
    spec = _load_spec()
    threshold = spec["clean_accounts_firewall"]["engineering_proxy_used_in_code"]["clean_threshold"]
    assert threshold == 90.0, f"Spec CFO/EBITDA clean threshold should be 90.0, got {threshold}"


def test_spec_cfo_ebitda_scale_is_percentage():
    spec = _load_spec()
    scale = spec["clean_accounts_firewall"]["engineering_proxy_used_in_code"]["scale_unit"]
    assert scale == "PERCENTAGE", f"CFO/EBITDA scale should be PERCENTAGE, got {scale}"


def test_spec_de_ceiling():
    spec = _load_spec()
    de = spec["leverage_governance_gates"]["debt_ceiling_non_financial"]["threshold"]
    assert de == 1.0, f"Spec D/E ceiling should be 1.0, got {de}"


def test_spec_pledge_ceiling():
    spec = _load_spec()
    pledge = spec["leverage_governance_gates"]["promoter_pledge_ceiling"]["threshold"]
    assert pledge == 10.0, f"Spec pledge ceiling should be 10.0, got {pledge}"


# ── Scoring engine code alignment ─────────────────────────────────────────────

def test_fw_coffee_can_block_exists():
    src = _scoring_src()
    assert "fw_coffee_can" in src, "fw_coffee_can block not found in scoring_engine.py"


def test_roce_10y_threshold_in_code_matches_spec():
    """ROCE 10Y median gate must be >= 15 (exactly 15, not 12 or 20)."""
    spec = _load_spec()
    floor = spec["twin_filters_core"]["non_financial_companies"]["roce_median_10y_floor_pct"]
    src = _scoring_src()
    # Find the cc_roce_10y >= N pattern inside fw_coffee_can area
    pattern = r"_cc_roce_10y\.fillna\(0\)\s*>=\s*(\d+(?:\.\d+)?)"
    matches = re.findall(pattern, src)
    assert matches, "Pattern _cc_roce_10y.fillna(0) >= N not found in scoring_engine.py"
    for m in matches:
        assert float(m) == floor, f"ROCE 10Y threshold in code ({m}) != spec ({floor})"


def test_roe_financial_recent_threshold_is_15_not_12():
    """Financial ROE recent threshold must be >= 15 (not the old relaxed 12)."""
    spec = _load_spec()
    floor = spec["twin_filters_core"]["financial_companies_bfsi"]["roe_median_5y_floor_pct"]
    src = _scoring_src()
    # Ensure there is no >= 12 for _cc_roe_rec anywhere
    pattern_12 = r"_cc_roe_rec\.fillna\(0\)\s*>=\s*12"
    assert not re.search(pattern_12, src), (
        "_cc_roe_rec >= 12 found in code — must be >= 15 per spec"
    )
    pattern_correct = r"_cc_roe_rec\.fillna\(0\)\s*>=\s*15"
    assert re.search(pattern_correct, src), (
        "_cc_roe_rec >= 15 not found in scoring_engine.py"
    )


def test_revenue_10y_threshold_in_code_matches_spec():
    spec = _load_spec()
    floor = spec["twin_filters_core"]["non_financial_companies"]["revenue_growth_10y_cagr_floor_pct"]
    src = _scoring_src()
    pattern = r"rev_10y_cc\.fillna\(0\)\s*>=\s*(\d+(?:\.\d+)?)"
    matches = re.findall(pattern, src)
    assert matches, "Pattern rev_10y_cc.fillna(0) >= N not found in scoring_engine.py"
    for m in matches:
        assert float(m) == floor, f"Revenue 10Y threshold in code ({m}) != spec ({floor})"


def test_cc_each_year_checks_10pct_floor():
    """_cc_each_year must check rev_gr_y2..y5 each >= individual_year_floor (10.0)."""
    spec = _load_spec()
    floor = spec["twin_filters_core"]["non_financial_companies"]["individual_year_floor_pct"]
    src = _scoring_src()
    # Each of y2..y5 must appear with >= floor
    for yr in (2, 3, 4, 5):
        pattern = rf"_cc_y{yr}\s*>=\s*(\d+(?:\.\d+)?)"
        matches = re.findall(pattern, src)
        assert matches, f"_cc_y{yr} >= N not found in scoring_engine.py"
        for m in matches:
            assert float(m) == floor, (
                f"_cc_y{yr} threshold ({m}) != spec individual year floor ({floor})"
            )


def test_cc_each_year_wired_into_fw_coffee_can():
    src = _scoring_src()
    assert "_cc_each_year" in src, "_cc_each_year not found in scoring_engine.py"
    # Must appear inside fw_coffee_can expression (after its definition)
    cc_block_start = src.find("fw_coffee_can = (")
    assert cc_block_start > 0, "fw_coffee_can = ( not found"
    cc_block = src[cc_block_start: cc_block_start + 800]
    assert "_cc_each_year" in cc_block, (
        "_cc_each_year not wired into fw_coffee_can expression"
    )


def test_cfo_ebitda_threshold_in_code_matches_spec():
    spec = _load_spec()
    threshold = spec["clean_accounts_firewall"]["engineering_proxy_used_in_code"]["clean_threshold"]
    src = _scoring_src()
    pattern = r"cfo_ebitda_cc\.fillna\(0\)\s*>=\s*(\d+(?:\.\d+)?)"
    matches = re.findall(pattern, src)
    assert matches, "Pattern cfo_ebitda_cc.fillna(0) >= N not found in scoring_engine.py"
    for m in matches:
        assert float(m) == threshold, (
            f"CFO/EBITDA threshold in code ({m}) != spec ({threshold})"
        )


def test_debt_to_equity_ceiling_in_code_matches_spec():
    spec = _load_spec()
    ceiling = spec["leverage_governance_gates"]["debt_ceiling_non_financial"]["threshold"]
    src = _scoring_src()
    # de_cc < 1.0 inside fw_coffee_can
    pattern = r"de_cc\.fillna\(999\)\s*<\s*(\d+(?:\.\d+)?)"
    matches = re.findall(pattern, src)
    assert matches, "Pattern de_cc.fillna(999) < N not found in scoring_engine.py"
    for m in matches:
        assert float(m) == ceiling, f"D/E ceiling in code ({m}) != spec ({ceiling})"


def test_pledge_ceiling_in_code_matches_spec():
    spec = _load_spec()
    ceiling = spec["leverage_governance_gates"]["promoter_pledge_ceiling"]["threshold"]
    src = _scoring_src()
    # pledge_cc < 10 inside fw_coffee_can
    pattern = r"pledge_cc\.fillna\(0\)\s*<\s*(\d+(?:\.\d+)?)"
    matches = re.findall(pattern, src)
    assert matches, "Pattern pledge_cc.fillna(0) < N not found in scoring_engine.py"
    for m in matches:
        assert float(m) == ceiling, f"Pledge ceiling in code ({m}) != spec ({ceiling})"


# ── Data engine alignment ──────────────────────────────────────────────────────

def test_revenue_historical_columns_in_column_map():
    """Revenue 2-5 Years Back must be in data_engine COLUMN_MAP."""
    src = _data_engine_src()
    for yr in (2, 3, 4, 5):
        label = f"Revenue {yr} Years Back"
        assert label in src, f"'{label}' not found in data_engine.py COLUMN_MAP"


def test_rev_gr_y2_to_y5_computed():
    """rev_gr_y2 through rev_gr_y5 must be computed in data_engine.
    Code uses f"rev_gr_y{_yr}" in a loop — test checks for the f-string template."""
    src = _data_engine_src()
    # The loop template pattern: f"rev_gr_y{_yr}" generates rev_gr_y2..y5
    assert 'f"rev_gr_y{_yr}"' in src or "f'rev_gr_y{_yr}'" in src, (
        "rev_gr_y{_yr} f-string loop not found in data_engine.py — "
        "year-by-year revenue growth columns (rev_gr_y2..y5) are not being generated"
    )


# ── Sector pct rank infinity guard ────────────────────────────────────────────

def test_sector_pct_rank_infinity_purge_before_groupby():
    """replace([np.inf, -np.inf], np.nan) must appear before .groupby() in _sector_pct_rank."""
    src = _scoring_src()
    fn_start = src.find("def _sector_pct_rank(")
    assert fn_start >= 0, "_sector_pct_rank function not found"
    # Use 1200 chars to ensure we capture the full function body (including return block)
    fn_body = src[fn_start: fn_start + 1200]
    replace_pos = fn_body.find("replace([np.inf, -np.inf], np.nan)")
    groupby_pos = fn_body.find(".groupby(")
    assert replace_pos >= 0, (
        "replace([np.inf, -np.inf], np.nan) not found in _sector_pct_rank — infinity guard missing"
    )
    assert groupby_pos >= 0, ".groupby() not found in _sector_pct_rank"
    assert replace_pos < groupby_pos, (
        f"Infinity purge (pos={replace_pos}) must occur BEFORE groupby (pos={groupby_pos})"
    )


# ── Vectorization contract ─────────────────────────────────────────────────────

def test_no_apply_in_scoring_engine():
    src = _scoring_src()
    apply_hits = [
        line.strip() for line in src.splitlines()
        if ".apply(" in line and not line.strip().startswith("#")
    ]
    assert not apply_hits, (
        f"df.apply() found in scoring_engine.py (violates vectorization contract):\n"
        + "\n".join(apply_hits)
    )


def test_no_iterrows_in_scoring_engine():
    src = _scoring_src()
    iter_hits = [
        line.strip() for line in src.splitlines()
        if "iterrows()" in line and not line.strip().startswith("#")
    ]
    assert not iter_hits, (
        f"iterrows() found in scoring_engine.py:\n" + "\n".join(iter_hits)
    )


# ── Financial sector routing ───────────────────────────────────────────────────

def test_ub_efficiency_gate_routes_roe_for_financials():
    """fw_unusual_billionaires must use ub_efficiency_gate (sector-routed), not raw ROCE."""
    src = _scoring_src()
    # Should have ub_efficiency_gate defined
    assert "ub_efficiency_gate" in src, "ub_efficiency_gate not found in scoring_engine.py"
    # Should use it inside fw_unusual_billionaires
    ub_start = src.find("fw_unusual_billionaires = (")
    assert ub_start > 0, "fw_unusual_billionaires = ( not found"
    ub_block = src[ub_start: ub_start + 600]
    assert "ub_efficiency_gate" in ub_block, (
        "ub_efficiency_gate not wired into fw_unusual_billionaires — financials still use raw ROCE"
    )


def test_cc_efficiency_routes_roe_for_financials():
    """fw_coffee_can must use _cc_efficiency (sector-routed), not raw ROCE directly."""
    src = _scoring_src()
    assert "_cc_efficiency" in src, "_cc_efficiency not found in scoring_engine.py"
    cc_start = src.find("fw_coffee_can = (")
    assert cc_start > 0, "fw_coffee_can = ( not found"
    cc_block = src[cc_start: cc_start + 600]
    assert "_cc_efficiency" in cc_block, (
        "_cc_efficiency not wired into fw_coffee_can"
    )
