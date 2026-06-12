"""
test_hundred_bagger_contract.py

Contract tests that parse docs/hundred_bagger_specs.json and assert that
core/scoring_engine.py and core/data_engine.py honour every threshold in the
Hybrid 100-Bagger Hunter spec (v2.0).

Ten gates verified:
  S (Size):          mcap >= 200  AND  mcap <= 3000
  Q (Quality):       roce_med_7y  >= 15
  G (Growth):        rev_gr_5y    >= 15  AND  pat_gr_3y >= 20
  B (Balance sheet): D/E < 0.5 (financial sector exempted)
  O (Owner-Op):      promoter_holdings >= 50
  C (Cash):          cfo_to_pat > 0  (anti-fraud, not strict 80% gate)
  P (Price):         peg > 0  AND  peg <= 2.0

Rejected gates verified absent:
  - _hb_each_year stumble check (belongs to Baid, not Mayer)
  - retention_rate gate (replaced by pat_gr_3y)
  - rev_gr_10y (replaced by rev_gr_5y for data availability)
"""
import json
import re
from pathlib import Path

ROOT         = Path(__file__).parent.parent
SPEC_PATH    = ROOT / "docs" / "hundred_bagger_specs.json"
SCORING_PATH = ROOT / "core" / "scoring_engine.py"
DATA_PATH    = ROOT / "core" / "data_engine.py"


def _load_spec():
    with open(SPEC_PATH, encoding="utf-8") as f:
        return json.load(f)


def _scoring_src():
    return SCORING_PATH.read_text(encoding="utf-8")


def _data_src():
    return DATA_PATH.read_text(encoding="utf-8")


def _hb_block(src):
    """Extract the fw_100_bagger block from scoring_engine.py source."""
    start = src.find("# 12. 100-Bagger Hunter")
    end   = src.find("# 13.", start)
    assert start > 0, "100-Bagger Hunter block anchor not found in scoring_engine.py"
    return src[start:end]


# ── Spec file integrity ────────────────────────────────────────────────────────

def test_spec_file_exists():
    assert SPEC_PATH.exists(), f"hundred_bagger_specs.json not found: {SPEC_PATH}"


def test_spec_required_top_level_keys():
    spec = _load_spec()
    for key in ("hybrid_sieve", "rejected_gates", "implementation_mapping", "known_gaps"):
        assert key in spec, f"Missing top-level key '{key}' in hundred_bagger spec"


def test_spec_mcap_floor():
    spec = _load_spec()
    t = spec["hybrid_sieve"]["size_corridor_floor"]["threshold_crore"]
    assert t == 200.0, f"Spec mcap floor should be 200.0, got {t}"


def test_spec_mcap_ceiling():
    spec = _load_spec()
    t = spec["hybrid_sieve"]["size_corridor_ceiling"]["threshold_crore"]
    assert t == 3000.0, f"Spec mcap ceiling should be 3000.0, got {t}"


def test_spec_roce_7y_threshold():
    spec = _load_spec()
    t = spec["hybrid_sieve"]["capital_quality_7y"]["threshold"]
    assert t == 15.0, f"Spec ROCE 7Y threshold should be 15.0, got {t}"


def test_spec_rev_5y_threshold():
    spec = _load_spec()
    t = spec["hybrid_sieve"]["revenue_growth_5y"]["threshold"]
    assert t == 15.0, f"Spec rev_gr_5y threshold should be 15.0, got {t}"


def test_spec_pat_3y_threshold():
    spec = _load_spec()
    t = spec["hybrid_sieve"]["pat_growth_3y"]["threshold"]
    assert t == 20.0, f"Spec pat_gr_3y threshold should be 20.0, got {t}"


def test_spec_de_threshold():
    spec = _load_spec()
    t = spec["hybrid_sieve"]["balance_sheet_fortress"]["threshold"]
    assert t == 0.5, f"Spec D/E threshold should be 0.5, got {t}"


def test_spec_de_financial_sector_exempted():
    spec = _load_spec()
    exempt = spec["hybrid_sieve"]["balance_sheet_fortress"]["financial_sector_exempted"]
    assert exempt is True, "Spec must flag financial sector exemption for D/E gate"


def test_spec_promoter_threshold():
    spec = _load_spec()
    t = spec["hybrid_sieve"]["owner_operator_gate"]["threshold"]
    assert t == 50.0, f"Spec promoter threshold should be 50.0 (majority control), got {t}"


def test_spec_cfo_pat_unit_is_percentage():
    spec = _load_spec()
    unit = spec["hybrid_sieve"]["cash_quality_anti_fraud"]["unit"]
    assert unit == "PERCENTAGE", f"cfo_to_pat unit must be PERCENTAGE, got {unit}"


def test_spec_peg_upper_threshold():
    spec = _load_spec()
    t = spec["hybrid_sieve"]["valuation_peg_upper"]["threshold"]
    assert t == 2.0, f"Spec PEG upper threshold should be 2.0, got {t}"


def test_spec_rejected_gates_documented():
    spec = _load_spec()
    rejected = [g["gate"] for g in spec["rejected_gates"]]
    assert any("each_year" in g for g in rejected), (
        "Rejected gates must document why _hb_each_year was excluded"
    )
    assert any("retention_rate" in g for g in rejected), (
        "Rejected gates must document why retention_rate was excluded"
    )
    assert any("10y" in g.lower() or "rev_gr_10y" in g for g in rejected), (
        "Rejected gates must document why rev_gr_10y was excluded"
    )


# ── scoring_engine.py — framework definition ──────────────────────────────────

def test_fw_100_bagger_block_exists():
    src = _scoring_src()
    assert "fw_100_bagger" in src, "fw_100_bagger not found in scoring_engine.py"


def test_hundred_bagger_pass_column_set():
    src = _scoring_src()
    assert 'df["hundred_bagger_pass"]' in src, (
        'df["hundred_bagger_pass"] not assigned in scoring_engine.py'
    )


def test_100_bagger_wired_into_frameworks_passed():
    src = _scoring_src()
    assert '100-Bagger|' in src, '"100-Bagger|" label not found in fw_str builder'
    fw_str_start = src.find("fw_str = (")
    assert fw_str_start > 0, "fw_str = ( not found in scoring_engine.py"
    fw_str_end = src.find('df["frameworks_passed"]', fw_str_start)
    fw_str_block = src[fw_str_start:fw_str_end]
    assert 'fw_100_bagger' in fw_str_block, (
        "fw_100_bagger not in fw_str builder — 100-Bagger not wired into frameworks_passed"
    )


# ── S: Size gates ─────────────────────────────────────────────────────────────

def test_mcap_floor_matches_spec():
    spec  = _load_spec()
    floor = spec["hybrid_sieve"]["size_corridor_floor"]["threshold_crore"]
    src   = _scoring_src()
    pattern = r"mcap_hb\.fillna\(0\)\s*>=\s*(\d+(?:\.\d+)?)"
    matches = re.findall(pattern, src)
    assert matches, "mcap_hb.fillna(0) >= N not found in scoring_engine.py"
    assert any(float(m) == floor for m in matches), (
        f"mcap floor in code {matches} != spec {floor}"
    )


def test_mcap_ceiling_matches_spec():
    spec    = _load_spec()
    ceiling = spec["hybrid_sieve"]["size_corridor_ceiling"]["threshold_crore"]
    src     = _scoring_src()
    pattern = r"mcap_hb\.fillna\(\d+\)\s*<=\s*(\d+(?:\.\d+)?)"
    matches = re.findall(pattern, src)
    assert matches, "mcap_hb.fillna(N) <= N not found in scoring_engine.py"
    assert any(float(m) == ceiling for m in matches), (
        f"mcap ceiling in code {matches} != spec {ceiling}"
    )


# ── Q: Quality gate ───────────────────────────────────────────────────────────

def test_roce_7y_used_not_5y():
    block = _hb_block(_scoring_src())
    assert "roce_med_7y" in block, "roce_med_7y not found in fw_100_bagger block"
    assert "roce_med_5y" not in block, (
        "roce_med_5y found in fw_100_bagger — must use 7Y window"
    )


def test_roce_7y_threshold_matches_spec():
    spec  = _load_spec()
    floor = spec["hybrid_sieve"]["capital_quality_7y"]["threshold"]
    src   = _scoring_src()
    pattern = r"roce_7y_hb\.fillna\(0\)\s*>=\s*(\d+(?:\.\d+)?)"
    matches = re.findall(pattern, src)
    assert matches, "roce_7y_hb.fillna(0) >= N not found in scoring_engine.py"
    for m in matches:
        assert float(m) == floor, f"ROCE 7Y threshold in code ({m}) != spec ({floor})"


# ── G: Growth gates ───────────────────────────────────────────────────────────

def test_rev_5y_used_not_10y_or_3y():
    block = _hb_block(_scoring_src())
    assert "rev_gr_5y" in block, "rev_gr_5y not found in fw_100_bagger block"
    assert "rev_gr_10y" not in block, (
        "rev_gr_10y found in fw_100_bagger — must use 5Y (data availability)"
    )
    assert "rev_gr_3y" not in block, (
        "rev_gr_3y found in fw_100_bagger — must use 5Y (covers more history)"
    )


def test_rev_5y_threshold_matches_spec():
    spec  = _load_spec()
    floor = spec["hybrid_sieve"]["revenue_growth_5y"]["threshold"]
    src   = _scoring_src()
    pattern = r"rev_5y_hb\.fillna\(0\)\s*>=\s*(\d+(?:\.\d+)?)"
    matches = re.findall(pattern, src)
    assert matches, "rev_5y_hb.fillna(0) >= N not found in scoring_engine.py"
    for m in matches:
        assert float(m) == floor, f"Revenue 5Y threshold in code ({m}) != spec ({floor})"


def test_pat_3y_gate_in_fw_100_bagger():
    block = _hb_block(_scoring_src())
    assert "pat_gr_3y" in block, (
        "pat_gr_3y not found in fw_100_bagger — earnings engine-fire signal missing"
    )


def test_pat_3y_threshold_matches_spec():
    spec  = _load_spec()
    floor = spec["hybrid_sieve"]["pat_growth_3y"]["threshold"]
    src   = _scoring_src()
    pattern = r"pat_3y_hb\.fillna\(0\)\s*>=\s*(\d+(?:\.\d+)?)"
    matches = re.findall(pattern, src)
    assert matches, "pat_3y_hb.fillna(0) >= N not found in scoring_engine.py"
    for m in matches:
        assert float(m) == floor, f"PAT 3Y threshold in code ({m}) != spec ({floor})"


# ── B: Balance sheet gate ─────────────────────────────────────────────────────

def test_de_gate_in_fw_100_bagger():
    block = _hb_block(_scoring_src())
    assert "de_hb" in block, "de_hb (D/E) not found in fw_100_bagger block"
    assert "< 0.5" in block, "D/E < 0.5 threshold not found in fw_100_bagger block"


def test_de_gate_has_financial_sector_exemption():
    block = _hb_block(_scoring_src())
    assert "is_fin_hb" in block, (
        "is_fin_hb financial sector exemption missing from fw_100_bagger D/E gate"
    )


# ── O: Owner-Operator gate ────────────────────────────────────────────────────

def test_promoter_threshold_matches_spec():
    spec  = _load_spec()
    floor = spec["hybrid_sieve"]["owner_operator_gate"]["threshold"]
    src   = _scoring_src()
    pattern = r"promo_hb\.fillna\(0\)\s*>=\s*(\d+(?:\.\d+)?)"
    matches = re.findall(pattern, src)
    assert matches, "promo_hb.fillna(0) >= N not found in scoring_engine.py"
    for m in matches:
        assert float(m) == floor, f"Promoter threshold in code ({m}) != spec ({floor})"


def test_promoter_is_50_not_45():
    src = _scoring_src()
    pattern = r"promo_hb\.fillna\(0\)\s*>=\s*(\d+(?:\.\d+)?)"
    matches = re.findall(pattern, src)
    assert matches, "promo_hb.fillna(0) >= N not found in scoring_engine.py"
    for m in matches:
        assert float(m) == 50.0, (
            f"Promoter gate is {m}% — must be 50.0% (majority control), not 45%"
        )


# ── C: Cash quality (anti-fraud) gate ────────────────────────────────────────

def test_cfo_pat_positive_gate_in_fw_100_bagger():
    block = _hb_block(_scoring_src())
    assert "cfo_pat_hb" in block, "cfo_pat_hb not found in fw_100_bagger block"
    assert ">  0.0" in block or "> 0.0" in block or ">  0" in block or "> 0" in block, (
        "CFO/PAT > 0 anti-fraud gate not found in fw_100_bagger block"
    )


def test_cfo_pat_gate_not_strict_80pct():
    block = _hb_block(_scoring_src())
    assert ">= 80" not in block and ">= 80.0" not in block, (
        "CFO/PAT >= 80 strict gate found in fw_100_bagger — must be > 0 (anti-fraud only)"
    )


# ── P: Valuation gates ────────────────────────────────────────────────────────

def test_peg_upper_threshold_matches_spec():
    spec  = _load_spec()
    upper = spec["hybrid_sieve"]["valuation_peg_upper"]["threshold"]
    src   = _scoring_src()
    pattern = r"peg_hb\.fillna\(999\)\s*<=\s*(\d+(?:\.\d+)?)"
    matches = re.findall(pattern, src)
    assert matches, "peg_hb.fillna(999) <= N not found in scoring_engine.py"
    for m in matches:
        assert float(m) == upper, f"PEG upper threshold in code ({m}) != spec ({upper})"


def test_peg_lower_positive_earnings():
    block = _hb_block(_scoring_src())
    assert "peg_hb" in block, "peg_hb not found in fw_100_bagger block"
    assert ">  0.0" in block or "> 0.0" in block or ">  0" in block, (
        "peg_hb > 0 (positive earnings check) not found in fw_100_bagger block"
    )


# ── Rejected gate guards ──────────────────────────────────────────────────────

def test_no_hb_each_year_stumble_check():
    block = _hb_block(_scoring_src())
    assert "_hb_each_year" not in block, (
        "_hb_each_year found in fw_100_bagger — stumble check belongs to Baid, not Mayer"
    )


def test_no_retention_rate_gate_in_fw_100_bagger():
    block = _hb_block(_scoring_src())
    assert "reten_hb" not in block, (
        "reten_hb found in fw_100_bagger — retention_rate gate rejected in hybrid spec"
    )


def test_no_rev_10y_in_fw_100_bagger():
    block = _hb_block(_scoring_src())
    assert "rev_10y_hb" not in block, (
        "rev_10y_hb found in fw_100_bagger — rejected due to data availability gaps"
    )


# ── data_engine.py — retention_rate still computed (tearsheet use) ────────────

def test_retention_rate_still_in_data_engine():
    src = _data_src()
    assert 'df["retention_rate"]' in src, (
        'df["retention_rate"] must remain in data_engine.py for tearsheet display, '
        'even though it is no longer a fw_100_bagger gate'
    )


def test_retention_rate_formula_correct():
    src = _data_src()
    assert "100.0 - " in src and "dividend_payout_ratio" in src, (
        "retention_rate formula (100 - DPR) not found in data_engine.py"
    )


# ── Vectorization contract ─────────────────────────────────────────────────────

def test_no_apply_in_scoring_engine():
    src = _scoring_src()
    apply_hits = [
        line.strip() for line in src.splitlines()
        if ".apply(" in line and not line.strip().startswith("#")
    ]
    assert not apply_hits, (
        "df.apply() found in scoring_engine.py:\n" + "\n".join(apply_hits)
    )


def test_no_iterrows_in_scoring_engine():
    src = _scoring_src()
    iter_hits = [
        line.strip() for line in src.splitlines()
        if "iterrows()" in line and not line.strip().startswith("#")
    ]
    assert not iter_hits, (
        "iterrows() found in scoring_engine.py:\n" + "\n".join(iter_hits)
    )
