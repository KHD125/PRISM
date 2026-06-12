"""
test_baid_contract.py

Contract tests that parse docs/baid_financial_specs.json and assert that
core/scoring_engine.py honours every engineering proxy threshold declared
in the ledger. If the ledger says roce_med_7y >= 15.0, the code must say >= 15.

Also guards against Gemini's fabricated chapter citations re-entering the code
(the old wrong Chapter 4 / Chapter 6 / Chapter 15 labels are banished).
"""
import json
import re
from pathlib import Path

ROOT          = Path(__file__).parent.parent
SPEC_PATH     = ROOT / "docs" / "baid_financial_specs.json"
SCORING_PATH  = ROOT / "core" / "scoring_engine.py"


def _load_spec():
    with open(SPEC_PATH, encoding="utf-8") as f:
        return json.load(f)


def _scoring_src():
    return SCORING_PATH.read_text(encoding="utf-8")


# ── Spec file integrity ────────────────────────────────────────────────────────

def test_spec_file_exists():
    assert SPEC_PATH.exists(), f"baid_financial_specs.json not found: {SPEC_PATH}"


def test_spec_required_top_level_keys():
    spec = _load_spec()
    for key in ("engineering_proxies", "book_qualitative_principles",
                "implementation_mapping", "actual_chapter_titles", "gemini_fabrications"):
        assert key in spec, f"Missing top-level key '{key}' in baid spec"


def test_spec_roce_7y_threshold():
    spec = _load_spec()
    t = spec["engineering_proxies"]["capital_efficiency_gate"]["threshold"]
    assert t == 15.0, f"Spec ROCE 7Y threshold should be 15.0, got {t}"


def test_spec_revenue_10y_threshold():
    spec = _load_spec()
    t = spec["engineering_proxies"]["revenue_growth_10y"]["threshold"]
    assert t == 12.0, f"Spec revenue 10Y threshold should be 12.0, got {t}"


def test_spec_annual_velocity_floor():
    spec = _load_spec()
    t = spec["engineering_proxies"]["annual_velocity_floor"]["threshold_each_year"]
    assert t == 0.0, f"Spec annual velocity floor should be 0.0 (anti-contraction standard), got {t}"


def test_spec_fcf_yield_large_cap():
    spec = _load_spec()
    t = spec["engineering_proxies"]["fcf_yield_size_aware"]["threshold_large_cap"]
    assert t == 3.0, f"Spec FCF yield large cap should be 3.0, got {t}"


def test_spec_fcf_yield_mid_small():
    spec = _load_spec()
    t = spec["engineering_proxies"]["fcf_yield_size_aware"]["threshold_mid_small"]
    assert t == 4.0, f"Spec FCF yield mid/small should be 4.0, got {t}"


def test_spec_peg_upper_threshold():
    spec = _load_spec()
    t = spec["engineering_proxies"]["peg_entry_corridor"]["threshold_upper"]
    assert t == 2.0, f"Spec PEG upper threshold should be 2.0 (expanded GARP regime), got {t}"


def test_spec_cfo_to_pat_threshold():
    spec = _load_spec()
    t = spec["engineering_proxies"]["cfo_to_pat_quality"]["threshold"]
    assert t == 80.0, f"Spec CFO/PAT threshold should be 80.0, got {t}"


def test_spec_fabrication_log_present():
    """The ledger must contain the Gemini fabrication audit record."""
    spec = _load_spec()
    fab = spec["gemini_fabrications"]
    assert "fabricated_chapter_4_title" in fab, "Gemini fabrication log missing ch4 entry"
    assert "fabricated_7y_roce_quote" in fab, "Gemini fabrication log missing ROCE entry"


# ── Scoring engine code alignment ─────────────────────────────────────────────

def test_fw_baid_block_exists():
    src = _scoring_src()
    assert "fw_baid" in src, "fw_baid not found in scoring_engine.py"


def test_roce_7y_in_fw_baid():
    """fw_baid must use roce_med_7y (7Y window), not roce_med_5y."""
    src = _scoring_src()
    # Must NOT use the old 5Y variable inside fw_baid context
    baid_start = src.find("# 19. Baid Compounder")
    baid_end   = src.find("# 20.", baid_start)
    baid_block = src[baid_start:baid_end]
    assert "roce_med_7y" in baid_block, (
        "roce_med_7y not found in fw_baid block — still using old 5Y window"
    )
    assert "roce_med_5y" not in baid_block, (
        "roce_med_5y found in fw_baid block — must use 7Y window"
    )


def test_roce_7y_threshold_matches_spec():
    spec = _load_spec()
    floor = spec["engineering_proxies"]["capital_efficiency_gate"]["threshold"]
    src   = _scoring_src()
    pattern = r"roce_7y_bd\.fillna\(0\)\s*>=\s*(\d+(?:\.\d+)?)"
    matches = re.findall(pattern, src)
    assert matches, "Pattern roce_7y_bd.fillna(0) >= N not found in scoring_engine.py"
    for m in matches:
        assert float(m) == floor, f"ROCE 7Y threshold in code ({m}) != spec ({floor})"


def test_rev_10y_in_fw_baid():
    """fw_baid must use rev_gr_10y (10Y CAGR), not rev_gr_5y."""
    src = _scoring_src()
    baid_start = src.find("# 19. Baid Compounder")
    baid_end   = src.find("# 20.", baid_start)
    baid_block = src[baid_start:baid_end]
    assert "rev_gr_10y" in baid_block, (
        "rev_gr_10y not found in fw_baid block — still using old 5Y window"
    )
    assert "rev_gr_5y" not in baid_block, (
        "rev_gr_5y found in fw_baid block — must use 10Y window"
    )


def test_rev_10y_threshold_matches_spec():
    spec  = _load_spec()
    floor = spec["engineering_proxies"]["revenue_growth_10y"]["threshold"]
    src   = _scoring_src()
    pattern = r"rev_10y_bd\.fillna\(0\)\s*>=\s*(\d+(?:\.\d+)?)"
    matches = re.findall(pattern, src)
    assert matches, "Pattern rev_10y_bd.fillna(0) >= N not found in scoring_engine.py"
    for m in matches:
        assert float(m) == floor, f"Revenue 10Y threshold in code ({m}) != spec ({floor})"


def test_bd_each_year_defined():
    src = _scoring_src()
    assert "_bd_each_year" in src, "_bd_each_year not found in scoring_engine.py"


def test_bd_each_year_checks_5pct_floor():
    """Each of _bd_y2.._bd_y5 must check >= 5 (the annual velocity floor)."""
    spec  = _load_spec()
    floor = spec["engineering_proxies"]["annual_velocity_floor"]["threshold_each_year"]
    src   = _scoring_src()
    for yr in (2, 3, 4, 5):
        pattern = rf"_bd_y{yr}\s*>=\s*(\d+(?:\.\d+)?)"
        matches = re.findall(pattern, src)
        assert matches, f"_bd_y{yr} >= N not found in scoring_engine.py"
        for m in matches:
            assert float(m) == floor, (
                f"_bd_y{yr} threshold ({m}) != spec annual velocity floor ({floor})"
            )


def test_bd_each_year_wired_into_fw_baid():
    src = _scoring_src()
    fw_start = src.find("fw_baid = (")
    assert fw_start > 0, "fw_baid = ( not found in scoring_engine.py"
    fw_block = src[fw_start: fw_start + 800]
    assert "_bd_each_year" in fw_block, (
        "_bd_each_year not wired into fw_baid expression"
    )


def test_size_aware_fcf_hurdle_in_code():
    """baid_fcf_yield_hurdle must use np.where for size-aware FCF gate."""
    src = _scoring_src()
    assert "baid_fcf_yield_hurdle" in src, (
        "baid_fcf_yield_hurdle not found in scoring_engine.py"
    )
    assert "np.where(is_large_bd" in src, (
        "np.where(is_large_bd not found — size-aware FCF hurdle not vectorized"
    )


def test_fcf_large_cap_threshold_matches_spec():
    spec  = _load_spec()
    large = spec["engineering_proxies"]["fcf_yield_size_aware"]["threshold_large_cap"]
    src   = _scoring_src()
    pattern = r"np\.where\(is_large_bd,\s*(\d+(?:\.\d+)?)"
    matches = re.findall(pattern, src)
    assert matches, "np.where(is_large_bd, N, ...) not found in scoring_engine.py"
    for m in matches:
        assert float(m) == large, f"Large cap FCF threshold ({m}) != spec ({large})"


def test_fcf_mid_small_threshold_matches_spec():
    spec     = _load_spec()
    mid_small = spec["engineering_proxies"]["fcf_yield_size_aware"]["threshold_mid_small"]
    src      = _scoring_src()
    pattern  = r"np\.where\(is_large_bd,\s*\d+(?:\.\d+)?,\s*(\d+(?:\.\d+)?)\)"
    matches  = re.findall(pattern, src)
    assert matches, "np.where(is_large_bd, N, M) — M not found in scoring_engine.py"
    for m in matches:
        assert float(m) == mid_small, f"Mid/Small FCF threshold ({m}) != spec ({mid_small})"


def test_peg_ceiling_matches_spec():
    spec  = _load_spec()
    upper = spec["engineering_proxies"]["peg_entry_corridor"]["threshold_upper"]
    src   = _scoring_src()
    pattern = r"peg_bd\.fillna\(999\)\s*<=\s*(\d+(?:\.\d+)?)"
    matches = re.findall(pattern, src)
    assert matches, "Pattern peg_bd.fillna(999) <= N not found in scoring_engine.py"
    for m in matches:
        assert float(m) == upper, f"PEG ceiling in code ({m}) != spec ({upper})"


def test_cfo_pat_threshold_matches_spec():
    spec  = _load_spec()
    floor = spec["engineering_proxies"]["cfo_to_pat_quality"]["threshold"]
    src   = _scoring_src()
    pattern = r"cfo_pat_bd\.fillna\(0\)\s*>=\s*(\d+(?:\.\d+)?)"
    matches = re.findall(pattern, src)
    assert matches, "Pattern cfo_pat_bd.fillna(0) >= N not found in scoring_engine.py"
    for m in matches:
        assert float(m) == floor, f"CFO/PAT threshold in code ({m}) != spec ({floor})"


# ── Fabrication guard — banish wrong chapter citations ────────────────────────

def test_no_fabricated_chapter_4_identifying_compounders():
    """The fabricated 'Chapter 4 (Identifying Compounders)' label must not appear."""
    src = _scoring_src()
    baid_start = src.find("# 19. Baid Compounder")
    baid_end   = src.find("# 20.", baid_start)
    baid_block = src[baid_start:baid_end]
    assert "Identifying Compounders" not in baid_block, (
        "Fabricated chapter label 'Identifying Compounders' still in fw_baid — "
        "actual Ch.4 = 'Harnessing the Power of Passion and Focus'"
    )


def test_no_fabricated_chapter_6_valuation_discipline():
    """The fabricated 'Chapter 6 (Valuation Discipline)' label must not appear."""
    src = _scoring_src()
    baid_start = src.find("# 19. Baid Compounder")
    baid_end   = src.find("# 20.", baid_start)
    baid_block = src[baid_start:baid_end]
    assert "Valuation Discipline" not in baid_block, (
        "Fabricated chapter label 'Valuation Discipline' still in fw_baid — "
        "actual Ch.6 = 'Humility is the Gateway to Attaining Wisdom'"
    )


def test_no_fabricated_chapter_15_sell():
    """The fabricated 'Chapter 15 (Sell)' label must not appear."""
    src = _scoring_src()
    baid_start = src.find("# 19. Baid Compounder")
    baid_end   = src.find("# 20.", baid_start)
    baid_block = src[baid_start:baid_end]
    assert "Chapter 15 (Sell)" not in baid_block, (
        "Fabricated chapter label 'Chapter 15 (Sell)' still in fw_baid — "
        "actual Ch.15 = 'Journaling Is a Powerful Tool for Self-Reflection'"
    )


# ── Vectorization contract ─────────────────────────────────────────────────────

def test_no_apply_in_scoring_engine():
    src = _scoring_src()
    apply_hits = [
        line.strip() for line in src.splitlines()
        if ".apply(" in line and not line.strip().startswith("#")
    ]
    assert not apply_hits, (
        f"df.apply() found in scoring_engine.py:\n" + "\n".join(apply_hits)
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
