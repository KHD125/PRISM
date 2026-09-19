"""
test_dividend_yield_and_opm_ladder.py
=====================================
Contract for the three vendor columns added to the Ratio tab on 2026-09-19:
`Dividend Yield`, `OPM 3 Years Back`, `OPM 5 Years Back`.

WHY THESE THREE (measured before anything was wired):

  * `Dividend Payout Ratio` is null on 752 rows (27.7%) of the live vintage — Coal India, the
    largest dividend payer in the country, among them — and every consumer of it read that null
    through `.fillna(0)` as "pays nothing, retains everything". The vendor's `Dividend Yield`
    is 100% populated and Coal India reads 6.43 in it. It is a TRAILING-12-MONTH yield while the
    payout ratio is FISCAL-YEAR, so the two are NOT the same basis: DY×PE reproduces DPR within
    ±10% on only 31.6% of the rows where both exist. That measurement decided the wiring below.

  * `moat_tau` — leg C of the wealth tier — was a Kendall tau over FOUR points, one of which was
    a 5-year MEDIAN standing in for the missing 5-years-back level. Four points is 6 pairs and 13
    attainable values; 588 stocks sat at exactly ±1/3, one OPM observation from a tier flip.
    The 2026-09-11 audit said the coarseness must not be "fixed" ONLY because no more OPM level
    columns existed. Now two do. Both were verified to be what their names claim (lag decay
    0.951 → 0.716 → 0.577 against `opm`; each correlates best with the derived NPM of its OWN
    year; ≤0.1% identical to any level sibling).

THE PAYOUT RULE (data_engine `dpr_effective`): the vendor's observed payout ratio ALWAYS wins.
Only where it is null does the yield speak, and it speaks in two different voices:
    yield == 0        → payout 0 — an EVIDENCED non-payer (RR = 1.0 is now a statement, not a fill)
    yield  > 0, PE>0  → payout ≈ yield × PE — an ESTIMATE on the TTM basis, kept in its own column
                        (`dpr_from_yield`) so the vendor column is never contaminated
    yield  > 0, no PE → nothing (a payer whose payout cannot be estimated is not guessed at)
    yield  absent     → the legacy path, unchanged (archived vintages carry no yield column)
The estimate is admitted on a pre-declared bar: on the 1,736 rows where BOTH routes exist they
land on the same side of every consumer threshold ≥93% of the time (RR>0.5 / <0.30 / ≥0.60),
median |ΔRR| 0.009. If the vendor ever changes the basis of either column, the fitness pin below
fails before a fabricated retention rate reaches a flag.

Run with: pytest tests/test_dividend_yield_and_opm_ladder.py -v
"""

import ast
import contextlib
import io
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import pytest

from data_engine import RATIO_COLS, compute_derived_signals
from test_data_quality_fixes import _frame

_ROOT = Path(__file__).resolve().parent.parent
_ENGINE = _ROOT / "core" / "data_engine.py"
_APP = _ROOT / "app.py"


@pytest.fixture(scope="module")
def live():
    from core import fetch_and_clean_data, run_scoring_pipeline
    with contextlib.redirect_stdout(io.StringIO()):
        return run_scoring_pipeline(fetch_and_clean_data("local"))


def _num(df, c):
    return pd.to_numeric(df[c], errors="coerce")


# ═══════════════════════════════════════════════════════════════════════════════════════
# 1. Mapping — each vendor header lands under its OWN name
# ═══════════════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("header,name", [
    ("Dividend Yield",   "dividend_yield"),
    ("OPM 3 Years Back", "opm_3yb"),
    ("OPM 5 Years Back", "opm_5yb"),
])
def test_vendor_header_maps_to_its_own_internal_name(header, name):
    assert RATIO_COLS.get(header) == name, (
        f"{header!r} must map to {name!r} in RATIO_COLS (it lives on the vendor's Ratio tab)"
    )


def test_the_vendor_yield_is_never_mapped_onto_the_engine_s_synthetic_names():
    """`dividend_yield_ratio` is yield ÷ G-Sec and `dividend_yield_synthetic` is DPR × earnings
    yield. Mapping the vendor's raw yield onto either name would silently change what the
    column MEANS (docs/known-issues.md, dividend entry). It gets its own name."""
    src = _ENGINE.read_text(encoding="utf-8")
    assert not re.search(r':\s*"dividend_yield_(ratio|synthetic)"', src)


# ═══════════════════════════════════════════════════════════════════════════════════════
# 2. The payout rule, on a frame small enough to check by hand
# ═══════════════════════════════════════════════════════════════════════════════════════
def _payout_frame(drop_yield: bool = False):
    #            A: obs wins   B: non-payer  C: estimate   D: no PE     E: no yield   F: clip    G: NEGATIVE PE
    f = _frame(
        n=7,
        dividend_payout_ratio=[40.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        dividend_yield=       [9.0,  0.0,    2.0,    2.0,    np.nan, 5.0,    2.0],
        pe=                   [20.0, 20.0,   20.0,   np.nan, 20.0,   30.0,   -20.0],
    )
    if drop_yield:
        f = f.drop(columns=["dividend_yield"])
    return compute_derived_signals(f)


def test_the_observed_payout_always_wins_over_the_yield_route():
    out = _payout_frame()
    # A: DY×PE = 180 (would clip to 100) but the vendor's 40 is present, so 40 it is.
    assert out["dpr_effective"].iloc[0] == 40.0
    assert np.isclose(out["reinvestment_rate"].iloc[0], 0.60)


def test_a_zero_yield_is_evidence_of_a_non_payer_not_a_fill():
    out = _payout_frame()
    assert out["dpr_effective"].iloc[1] == 0.0, "yield 0 must resolve the payout to an evidenced 0"
    assert out["reinvestment_rate"].iloc[1] == 1.0
    assert pd.notna(out["dpr_effective"].iloc[1]), "the 0 must be a VALUE, not the legacy NaN→0 path"


def test_a_positive_yield_estimates_the_payout_as_yield_times_pe_in_its_own_column():
    out = _payout_frame()
    assert np.isclose(out["dpr_from_yield"].iloc[2], 40.0)          # 2.0 × 20
    assert np.isclose(out["dpr_effective"].iloc[2], 40.0)
    assert np.isclose(out["reinvestment_rate"].iloc[2], 0.60)
    # the vendor column itself is never back-filled — two bases never share one column
    assert pd.isna(out["dividend_payout_ratio"].iloc[2])


def test_the_estimate_is_clipped_to_a_full_payout():
    out = _payout_frame()
    assert out["dpr_from_yield"].iloc[5] == 100.0                     # 5.0 × 30 = 150 → 100
    assert out["reinvestment_rate"].iloc[5] == 0.0


def test_no_estimate_is_fabricated_without_a_positive_pe():
    out = _payout_frame()
    assert pd.isna(out["dpr_from_yield"].iloc[3])
    assert pd.isna(out["dpr_effective"].iloc[3])
    assert out["reinvestment_rate"].iloc[3] == 1.0, "a payer with no PE takes the legacy path, not a guess"
    # G: a loss-maker that PAYS (yield 2, PE −20). Without the PE>0 guard, DY×PE = −40 clips to 0 and
    # the row is certified an evidenced NON-payer — the opposite of the evidence.
    assert pd.isna(out["dpr_from_yield"].iloc[6]), "a negative PE must never yield a payout estimate"
    assert pd.isna(out["dpr_effective"].iloc[6])


def test_an_absent_or_null_yield_is_the_legacy_path_exactly():
    """Archived vintages (Movers re-scores them with the live engine) carry no yield column at
    all; the engine must not raise and must produce what it always produced."""
    out = _payout_frame()
    assert pd.isna(out["dpr_effective"].iloc[4]) and out["reinvestment_rate"].iloc[4] == 1.0
    legacy = _payout_frame(drop_yield=True)
    assert legacy["dpr_effective"].iloc[0] == 40.0
    assert legacy["dpr_effective"].iloc[1:].isna().all()
    assert (legacy["reinvestment_rate"].iloc[1:] == 1.0).all()
    assert legacy["dpr_from_yield"].isna().all()


def test_retention_rate_and_reinvestment_rate_share_one_payout_basis():
    out = _payout_frame()
    eff0 = out["dpr_effective"].fillna(0.0)
    assert np.allclose(out["retention_rate"], (100.0 - eff0).clip(0, 100))
    assert np.allclose(out["reinvestment_rate"] * 100.0, out["retention_rate"])


def test_no_consumer_reads_the_raw_payout_column_with_its_own_fallback():
    """Every payout consumer (SSGR, Blue Chip, GAPR, RR, retention) goes through dpr_effective.
    A new consumer that reaches for the raw vendor column with a private .fillna re-creates the
    fabricated-zero defect one site at a time."""
    src = _ENGINE.read_text(encoding="utf-8")
    body = src[src.index("def compute_derived_signals"):]
    # exactly ONE read of the raw vendor column is allowed: the _dpr_obs line that DEFINES dpr_effective
    reads = [ln for ln in body.splitlines() if 'df.get("dividend_payout_ratio"' in ln]
    assert reads == [ln for ln in reads if "_dpr_obs" in ln] and len(reads) == 1, (
        f"a consumer bypasses dpr_effective (df.get): {reads}"
    )
    assert 'df["dividend_payout_ratio"].fillna' not in body, "a consumer bypasses dpr_effective (.fillna)"
    assert body.count('df["dpr_effective"]') >= 4, "the effective series is not the shared input it must be"


# ═══════════════════════════════════════════════════════════════════════════════════════
# 3. Live — the spot-check that gated this family, and the split it produces
# ═══════════════════════════════════════════════════════════════════════════════════════
def test_coal_india_reads_a_yield_and_a_real_retention_rate(live):
    row = live[live["name"].astype(str).str.contains("Coal India", case=False, na=False)]
    assert len(row) == 1, "Coal India missing from the universe"
    r = row.iloc[0]
    assert float(r["dividend_yield"]) > 5.0, "the spot-check that gated this family — a real yield"
    assert pd.isna(r["dividend_payout_ratio"]), (
        "the vendor's DPR is populated for Coal India now — the DPR half of known-issues is resolved; "
        "re-read that entry, and this test's premise, before editing"
    )
    assert 30.0 <= float(r["dpr_effective"]) <= 80.0
    assert float(r["reinvestment_rate"]) < 0.75, "Coal India no longer reads 'retains everything'"


def test_no_dpr_rows_split_into_evidenced_non_payers_and_estimated_payers(live):
    dpr, dy, pe = _num(live, "dividend_payout_ratio"), _num(live, "dividend_yield"), _num(live, "pe")
    rr, eff = _num(live, "reinvestment_rate"), _num(live, "dpr_effective")
    non_payer = dpr.isna() & (dy == 0)
    payer = dpr.isna() & (dy > 0) & (pe > 0)
    assert non_payer.sum() > 100 and payer.sum() > 50, "the split this vintage showed has vanished"
    assert (eff[non_payer] == 0.0).all() and (rr[non_payer] == 1.0).all()
    expected = 1.0 - (dy[payer] * pe[payer]).clip(0, 100) / 100.0
    assert np.allclose(rr[payer], expected)
    assert (rr[payer] < 1.0).all(), "a paying company still reads 'retains everything'"


def test_the_yield_route_agrees_with_the_observation_at_every_consumer_threshold(live):
    """The fitness bar the estimate was admitted on. Both routes exist on ~1,700 rows; if they
    stop agreeing on which SIDE of a consumer threshold a stock sits, the vendor changed a basis
    and the estimate must be withdrawn — before it manufactures a flag."""
    dpr, dy, pe = _num(live, "dividend_payout_ratio"), _num(live, "dividend_yield"), _num(live, "pe")
    both = dpr.notna() & (dpr > 0) & (dy > 0) & (pe > 0)
    assert both.sum() > 800
    rr_obs = (1 - dpr / 100).clip(0, 1)[both]
    rr_est = (1 - (dy * pe).clip(0, 100) / 100)[both]
    for thr in (0.5, 0.30, 0.60):
        agree = ((rr_obs > thr) == (rr_est > thr)).mean()
        assert agree >= 0.85, f"routes disagree on RR>{thr} for {1-agree:.1%} of rows (bar 15%)"
    assert (rr_obs - rr_est).abs().median() < 0.05


def test_flags_resting_on_no_payout_evidence_at_all_are_now_rare(live):
    """Before: 344 misallocation flags stood on a fillna. After: a flag can only rest on nothing
    where the yield itself is missing or a payer has no PE — a handful of rows."""
    flagged = _num(live, "capital_misallocation_risk") == 1
    assert flagged.sum() > 0
    blind = flagged & _num(live, "dpr_effective").isna()
    assert blind.sum() / flagged.sum() < 0.02, f"{blind.sum()} flags rest on no payout evidence"


def test_the_study_16_ratio_reads_the_observation_wherever_it_exists(live):
    """dividend_yield_ratio is the yield ÷ G-Sec (MOSL Study 16). With the vendor's yield present it
    must be THAT yield ÷ G-Sec on every such row — the synthetic (DPR × earnings yield) is only
    the fallback for vintages that carry no yield column."""
    from config import INDIA_GSEC_YIELD
    dy, ratio = _num(live, "dividend_yield"), _num(live, "dividend_yield_ratio")
    have = dy.notna()
    assert have.mean() > 0.95
    assert np.allclose(ratio[have], dy[have] / INDIA_GSEC_YIELD)


def test_the_rr_sentinel_share_fell_and_the_column_got_richer(live):
    rr = _num(live, "reinvestment_rate")
    share = (rr == 1.0).mean()
    assert 0.40 < share < 0.58, f"RR==1.0 on {share:.1%} (was 60.6% before the yield arrived)"
    assert rr.nunique() > 950


# ═══════════════════════════════════════════════════════════════════════════════════════
# 4. The OPM ladder — partial ladders keep the NaN-conservative rule
#    (the full quantization contract lives in tests/test_moat_tau_quantization.py)
# ═══════════════════════════════════════════════════════════════════════════════════════
def _ladder(**over):
    base = dict(n=1, opm_5yb=[7.0], opm_3yb=[8.0], opm_1yb=[9.0], opm=[10.0], opm_latest_q=[11.0])
    base.update(over)
    return compute_derived_signals(_frame(**base))["moat_tau"].iloc[0]


def test_a_monotone_five_point_ladder_is_tau_one():
    assert np.isclose(_ladder(), 1.0)


def test_a_row_missing_only_the_oldest_level_still_gets_a_tau_from_six_pairs():
    """5 columns = 10 pairs, min_pairs = 6. Without opm_5yb (19% of the universe) the other four
    give exactly 6 comparable pairs — enough. The step for those rows is 1/6, as before."""
    assert np.isclose(_ladder(opm_5yb=[np.nan]), 1.0)
    # latest_q 9.5 sits below opm (10) but above opm_1yb (9) and opm_3yb (8): ONE discordant pair of six
    assert np.isclose(_ladder(opm_5yb=[np.nan], opm_latest_q=[9.5]), (5 - 1) / 6)


def test_two_missing_levels_is_not_enough_history_for_a_trend():
    assert pd.isna(_ladder(opm_5yb=[np.nan], opm_3yb=[np.nan]))


def test_the_five_year_median_is_no_longer_a_point_on_the_ladder():
    """A median is centred ~2.5 years back; it was only ever a stand-in for the missing oldest
    level. With opm_5yb present it must not be able to move the tau."""
    assert np.isclose(_ladder(opm_med_5y=[50.0]), _ladder(opm_med_5y=[-50.0]))


def test_ep_approaching_flag_is_alive_on_the_new_ladder(live):
    rate = _num(live, "ep_approaching_flag").mean()
    assert 0.0 < rate < 0.10, f"ep_approaching_flag fires {rate:.1%}"


# ═══════════════════════════════════════════════════════════════════════════════════════
# 5. Surfaces — the yield is on screen where the known-issues entry said it should land
# ═══════════════════════════════════════════════════════════════════════════════════════
def _app_dict(name: str):
    tree = ast.parse(_APP.read_text(encoding="utf-8"), filename="app.py")
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
                and isinstance(node.value, ast.Dict)):
            return {ast.literal_eval(k): ast.literal_eval(v) for k, v in zip(node.value.keys, node.value.values)}
    raise AssertionError(f"{name} not found as a dict literal in app.py")


def test_dividend_yield_sits_directly_after_earnings_yield_in_the_valuation_view():
    cols = _app_dict("_DS_VIEWS")["💰 Valuation"]
    assert "dividend_yield" in cols, "the yield is not in the 💰 Valuation preset"
    assert cols.index("dividend_yield") == cols.index("earnings_yield") + 1, (
        "the two yields must be adjacent — a reader compares them in one glance"
    )


def test_dividend_yield_is_titled_and_formatted_as_a_percent():
    fmt = _app_dict("_num_fmt")
    assert fmt.get("dividend_yield") == ("Div Yield", "%.2f%%")


def test_the_scanner_tip_resolves_from_the_glossary_by_value_and_states_the_basis():
    from ui.ui_components import _RAW_GLOSSARY
    from ui.ui_scanner import _SCANNER_HEADER_TIPS
    text = _RAW_GLOSSARY["Dividend Yield"]
    assert _SCANNER_HEADER_TIPS["dividend_yield"] == text
    low = text.lower()
    assert "12 months" in low or "trailing" in low or "ttm" in low, "the TTM basis must be stated"
    assert "0" in text or "zero" in low, "the entry must say what a zero means (does not pay)"


def _rendered(fn, row) -> str:
    import ui.ui_tearsheet as T
    out = []

    class _Rec:
        def markdown(self, *a, **k):
            if a:
                out.append(str(a[0]))

        def __getattr__(self, _n):
            return lambda *a, **k: None

    real = T.st
    try:
        T.st = _Rec()
        fn(row)
    finally:
        T.st = real
    return re.sub(r"<[^>]+>", " ", " ".join(out))


def test_the_all_data_grid_prints_the_yield_and_an_honest_zero(live):
    import ui.ui_tearsheet as T
    coal = live[live["name"].astype(str).str.contains("Coal India", case=False, na=False)].iloc[0]
    txt = _rendered(T.render_raw_signals, coal)
    assert re.search(r"Dividend Yield.{0,60}?6\.\d\d%", txt, re.S), txt[:400]
    zero = live[_num(live, "dividend_yield") == 0].iloc[0]
    txt0 = _rendered(T.render_raw_signals, zero)
    assert re.search(r"Dividend Yield.{0,60}?0\.00%", txt0, re.S), (
        "a zero yield is a STATEMENT (does not pay) and must print as 0.00%, never N/A"
    )
    gone = coal.copy(); gone["dividend_yield"] = np.nan          # an archived vintage without the column
    txtn = _rendered(T.render_raw_signals, gone)
    assert re.search(r"Dividend Yield.{0,60}?N/A", txtn, re.S) and not re.search(r"Dividend Yield.{0,60}?0\.00%", txtn, re.S), (
        "an ABSENT yield must read N/A — the grid's g() defaults NaN to 0, which would fabricate 0.00%"
    )


def test_the_insights_panel_flags_only_the_rare_above_gsec_yield():
    """MOSL Study 16's buy signal is dividend yield ≥ the G-Sec yield — rare (p99 of the live
    universe is 7.0%). So the row is ✅ only there and ⚪ everywhere else: not paying a dividend is
    not a failure, and a 2% yield is not a ❌. Absent yield → no row (archived vintages)."""
    from config import INDIA_GSEC_YIELD
    from test_financial_insights_display import _base_stock, _one_row
    mark, rest = _one_row(_base_stock(dividend_yield=INDIA_GSEC_YIELD + 0.5), "Dividend Yield")
    assert mark == "✅" and f"{INDIA_GSEC_YIELD + 0.5:.2f}%" in rest, (mark, rest)
    mark, rest = _one_row(_base_stock(dividend_yield=2.0), "Dividend Yield")
    assert mark == "⚪" and "2.00%" in rest, (mark, rest)
    mark, rest = _one_row(_base_stock(dividend_yield=0.0), "Dividend Yield")
    assert mark == "⚪" and "0.00%" in rest, (mark, rest)
    mark, _ = _one_row(_base_stock(), "Dividend Yield")          # no yield in the row at all
    assert mark is None, "an absent yield must not render a row"
