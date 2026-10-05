"""
test_cash_flow_year_basis.py
============================
Contract: wherever PRISM compares PROFIT with CASH FLOW, both figures come from the SAME fiscal year.

THE DEFECT (audited 2026-10-05). The data vendor rolls the P&L to a new fiscal year months before the
cash-flow statement and the balance sheet. Proven by the vendor's own ratios, which are each computed
inside one year: in June its CFO/PAT equalled 100 x OCF / PAT on 97.8% of stocks; by October it equals
100 x OCF / PAT-1-YEAR-BACK on 94%, while OCF itself has not moved in any snapshot (97-99% unchanged).
So on the live data, for ~80% of stocks, PAT and EBITDA are FY26 while operating cash flow, free cash
flow and the balance sheet are FY25.

Eight places compared the two directly: the accruals ratio (and its clean/warning flags, which feed
the Schilit checker), rf_high_accruals, Piotroski F4 (CFO > PAT), rf_low_fcf_ebitda,
rf_lease_inflation, FCF/PAT (fcf_quality and d28_fcf_to_pat_pct), and the depreciation leg of the
capex estimate that fills missing free cash flow. A company whose profit GREW looked accrual-heavy
purely because this year's bigger profit was set against last year's cash: the accruals red flag fired
on 50.3% of stocks with profit up >50% against 45.0% on matched years, and on 27.3% of decliners
against a true 32.8%. One mismatched flag can cross the forensic penalty cliff (5 flags -> 0.50x,
4 -> 0.75x): NINtec Systems 53.5 -> 77.8, Arrow Greentech 42.7 -> 64.0 when the years are matched.

THE FIX. `cash_flow_year_lag` decides, per stock, whether the cash-flow statement is a year behind the
P&L, from the vendor's own same-year ratios (CFO/PAT, then CFO/EBITDA, then ROA — CFO/PAT and
CFO/EBITDA agree 100% wherever both decide; cash flow and balance sheet roll together on 98%). Where it
is, those eight sites use last year's PAT / EBITDA / depreciation. Where the years match — or the data
cannot tell — they use the current figures, exactly as before. Nothing is hard-coded to a year, so the
fix switches itself off the day the vendor rolls the cash-flow statement, and archived vintages (June:
same year) are scored as they always were.

NOT IN SCOPE, deliberately (see docs/known-issues.md): profit divided by the balance sheet (NFAT, SSGR,
debt/EBITDA) is profit over OPENING capital — a recognised definition, kept; and comparisons of two
CHANGES across different years (Schilit's turnover-vs-margin and depreciation-rate signals) cannot be
aligned from the columns that exist.

Run with: pytest tests/test_cash_flow_year_basis.py -v
"""

import contextlib
import io as _io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

import numpy as np
import pandas as pd
import pytest

from core.data_engine import (BALANCE_COLS, CASHFLOW_COLS, COMMON_COLS, INCOME_COLS, RATIO_COLS,
                              SHAREHOLDING_COLS, TECHNICAL_COLS, cash_flow_year_lag,
                              coerce_numeric_columns, compute_derived_signals, load_all_csvs,
                              merge_datasets)
from core.forensic_engine import compute_forensic_signals

_ROOT = os.path.join(os.path.dirname(__file__), "..")
_MAPPED = set()
for _m in (COMMON_COLS, RATIO_COLS, INCOME_COLS, BALANCE_COLS, CASHFLOW_COLS, SHAREHOLDING_COLS,
           TECHNICAL_COLS):
    _MAPPED.update(_m.values())


# ── 1. The detector ─────────────────────────────────────────────────────────────────────────────
def _d(**cols):
    """A one-row frame carrying only what the detector reads."""
    return pd.DataFrame({k: [v] for k, v in cols.items()})


def _lag(**cols):
    v = cash_flow_year_lag(_d(**cols)).iloc[0]
    return None if pd.isna(v) else int(v)


def test_cash_flow_a_year_behind_is_detected_from_cfo_to_pat():
    # FY26 PAT 200, FY25 PAT 100, FY25 OCF 110 -> the vendor's same-year CFO/PAT is 110
    assert _lag(operating_cash_flow=110.0, pat=200.0, pat_1yb=100.0, cfo_to_pat=110.0) == 1


def test_same_year_is_detected_from_cfo_to_pat():
    assert _lag(operating_cash_flow=110.0, pat=100.0, pat_1yb=80.0, cfo_to_pat=110.0) == 0


def test_no_evidence_is_undecided_never_assumed():
    assert _lag(operating_cash_flow=110.0, pat=200.0, pat_1yb=100.0) is None


def test_cfo_to_ebitda_decides_when_cfo_to_pat_is_missing():
    assert _lag(operating_cash_flow=150.0, ebitda=300.0, ebitda_1yb=200.0, cfo_to_ebitda=75.0,
                pat=200.0, pat_1yb=100.0) == 1


def test_roa_decides_when_both_cash_ratios_are_missing():
    """Archived vintages need this: in the June snapshot the vendor's CFO/PAT was blank on 24%."""
    assert _lag(pat=200.0, pat_1yb=100.0, total_assets=1000.0, roa=10.0) == 1
    assert _lag(pat=200.0, pat_1yb=100.0, total_assets=1000.0, roa=20.0) == 0


def test_a_ratio_rounded_to_two_decimals_still_matches():
    # 100 x 110 / 97 = 113.4020... ; the vendor prints 113.40
    assert _lag(operating_cash_flow=110.0, pat=150.0, pat_1yb=97.0, cfo_to_pat=113.40) == 1
    assert _lag(operating_cash_flow=110.0, pat=97.0, pat_1yb=150.0, cfo_to_pat=113.40) == 0


def test_a_loss_year_is_decided_by_the_balance_sheet():
    """A loss cannot be a denominator (CLAUDE.md §5: every division guards denom > 0), so CFO/PAT
    carries no evidence for a loss year. ROA divides by total assets, keeps the loss in the
    numerator, and decides instead: FY25 PAT -50 on assets 1000 is ROA -5.0."""
    assert _lag(operating_cash_flow=30.0, pat=20.0, pat_1yb=-50.0, cfo_to_pat=-60.0,
                total_assets=1000.0, roa=-5.0) == 1
    assert _lag(operating_cash_flow=30.0, pat=20.0, pat_1yb=-50.0, cfo_to_pat=-60.0) is None


def test_years_that_cannot_be_told_apart_are_left_undecided():
    # PAT moved 0.2%: both years reproduce the vendor ratio, so the ratio carries no evidence
    assert _lag(operating_cash_flow=110.0, pat=100.2, pat_1yb=100.0, cfo_to_pat=109.89) is None


def test_years_under_one_percent_apart_are_still_told_apart():
    """The vendor ratio reproduces its own year to well under 0.5% (98% of stocks), so a 0.8% gap
    between the two candidate years is real evidence. A looser tolerance would throw it away."""
    # 100 x 110 / 100 = 110.00 (last year) vs 100 x 110 / 100.8 = 109.13 (this year)
    assert _lag(operating_cash_flow=110.0, pat=100.8, pat_1yb=100.0, cfo_to_pat=110.0) == 1


def test_a_decision_needs_the_two_years_to_be_apart_by_more_than_the_noise():
    """This year reproduces the ratio (110.3 vs 110.0) and last year narrowly misses (110.7) — but the
    two candidates are only 0.4 apart, inside the 0.55 noise band, so which one 'matched' is rounding,
    not evidence. No decision."""
    assert _lag(operating_cash_flow=110.0, pat=11000.0 / 110.3, pat_1yb=11000.0 / 110.7,
                cfo_to_pat=110.0) is None


def test_zero_cash_flow_falls_through_to_the_balance_sheet():
    # OCF 0 makes every cash ratio 0 whichever year it was taken from; ROA still discriminates
    assert _lag(operating_cash_flow=0.0, pat=200.0, pat_1yb=100.0, cfo_to_pat=0.0,
                ebitda=300.0, ebitda_1yb=150.0, cfo_to_ebitda=0.0,
                total_assets=1000.0, roa=10.0) == 1


def test_cash_flow_evidence_outranks_balance_sheet_evidence():
    """The question is which year the CASH FLOW belongs to, so the cash ratios answer it first."""
    assert _lag(operating_cash_flow=110.0, pat=100.0, pat_1yb=80.0, cfo_to_pat=110.0,
                total_assets=1000.0, roa=8.0) == 0


# ── 2. Every site judges profit against cash on the same year ───────────────────────────────────
def _frame(rows):
    """compute_derived_signals frame: every mapped column materialised (NaN where not given)."""
    n = len(rows)
    base = {"company_id": [f"NSE:Y{i}" for i in range(n)], "name": [r.pop("name") for r in rows],
            "industry": ["Test Industry"] * n, "market_category": ["Mid Cap"] * n,
            "market_cap": [5000.0] * n, "close_price": [100.0] * n}
    keys = sorted({k for r in rows for k in r})
    for k in keys:
        base[k] = [r.get(k, np.nan) for r in rows]
    df = pd.DataFrame(base)
    missing = sorted(c for c in _MAPPED if c not in df.columns)
    if missing:
        df = pd.concat([df, pd.DataFrame(np.nan, index=df.index, columns=missing)], axis=1)
    return df


def _common(**over):
    r = dict(total_assets=1000.0, total_assets_1yb=900.0, revenue=2000.0, revenue_1yb=1500.0,
             reserves=500.0, reserves_1yb=450.0, equity_shares=10.0, debt=100.0, debt_1yb=100.0)
    r.update(over)
    return r


ROWS = [
    # GROWER whose cash flow lags: FY25 PAT 100 / EBITDA 150 / dep 30 sit with FY25 OCF 110, FCF 60.
    # FY26 (the P&L the vendor shows as current) is PAT 200 / EBITDA 300 / dep 50.
    _common(name="Grower", sector="Retail", pat=200.0, pat_1yb=100.0, pat_2yb=90.0,
            ebitda=300.0, ebitda_1yb=150.0, ebit=250.0, ebit_1yb=120.0,
            operating_cash_flow=110.0, ocf_1yb=95.0, free_cash_flow=60.0, fcf_1yb=50.0,
            cfo_to_pat=110.0, cfo_to_ebitda=73.33, roa=10.0),
    # Same lag, vendor FCF missing -> PRISM rebuilds FY25 FCF = OCF - (net fixed-asset additions + dep)
    _common(name="Rebuilt", sector="Chemicals", pat=200.0, pat_1yb=100.0,
            ebitda=300.0, ebitda_1yb=150.0, ebit=250.0, ebit_1yb=120.0,
            operating_cash_flow=110.0, ocf_1yb=95.0, fixed_assets=500.0, fixed_assets_1yb=450.0,
            cwip=10.0, cwip_1yb=10.0, cfo_to_pat=110.0, cfo_to_ebitda=73.33, roa=10.0),
    # SAME YEAR (the P&L has not rolled): current figures are already the cash-flow year. Built as the
    # Grower's MIRROR — last year was the bigger year (PAT 200, EBITDA 300) — so a fix that reached for
    # last year's figures unconditionally would flip every one of the nine sites below.
    _common(name="SameYear", sector="Retail", pat=100.0, pat_1yb=200.0,
            ebitda=150.0, ebitda_1yb=300.0, ebit=120.0, ebit_1yb=250.0,
            operating_cash_flow=110.0, ocf_1yb=90.0, free_cash_flow=60.0, fcf_1yb=50.0,
            cfo_to_pat=110.0, cfo_to_ebitda=73.33, roa=10.0),
    # ACCRUAL-HEAVY with a lag: FY25 PAT 150 vs OCF 50 -> 10.5% of average assets, fires on its own year
    _common(name="Accrual", sector="Chemicals", pat=120.0, pat_1yb=150.0,
            ebitda=260.0, ebitda_1yb=200.0, ebit=220.0, ebit_1yb=170.0,
            operating_cash_flow=50.0, ocf_1yb=60.0, free_cash_flow=20.0, fcf_1yb=25.0,
            cfo_to_pat=33.33, cfo_to_ebitda=25.0, roa=15.0),
]


@pytest.fixture(scope="module")
def scored():
    with contextlib.redirect_stdout(_io.StringIO()):
        d = compute_forensic_signals(compute_derived_signals(_frame([dict(r) for r in ROWS])))
    return d.set_index("name")


# (column, Grower's value on matched years, its value on the mixed years it used to read)
SITES = [
    ("accruals_ratio",     -0.01, 0.09),    # (100 - 110) / 1000      vs (200 - 110) / 1000
    ("accruals_warning",   0,     1),
    ("accruals_clean",     1,     0),
    ("rf_high_accruals",   0,     1),       # -10 / 950                vs 90 / 950 = 9.5%
    ("f_accrual_quality",  1,     0),       # OCF 110 > PAT 100        vs 110 > 200
    ("rf_low_fcf_ebitda",  0,     1),       # 60 / 150 = 0.40          vs 60 / 300 = 0.20
    ("rf_lease_inflation", 0,     1),       # 150 - 110 = 40 <= 45     vs 300 - 110 = 190 > 90
    ("fcf_quality",        0.60,  0.30),    # 60 / 100                 vs 60 / 200
    ("d28_fcf_to_pat_pct", 60.0,  30.0),
]


@pytest.mark.parametrize("col,matched,mixed", SITES, ids=[s[0] for s in SITES])
def test_a_lagging_grower_is_judged_on_matched_years(scored, col, matched, mixed):
    got = float(scored.loc["Grower", col])
    assert np.isclose(got, matched), (
        f"{col} = {got}: profit is still compared with cash flow from a different year "
        f"(the mixed-year value is {mixed}, the same-year value {matched})")


def test_the_rebuilt_free_cash_flow_uses_the_same_years_depreciation(scored):
    """FCF the vendor did not report is rebuilt as OCF - (net fixed-asset additions + depreciation).
    The additions are FY25 (balance sheet), the OCF is FY25, so the depreciation must be FY25 too:
    110 - (50 + 30) = 30, not 110 - (50 + 50) = 10."""
    assert np.isclose(float(scored.loc["Rebuilt", "fcf_reconstructed"]), 30.0)
    assert np.isclose(float(scored.loc["Rebuilt", "free_cash_flow"]), 30.0)


@pytest.mark.parametrize("col,matched,mixed", SITES, ids=[s[0] for s in SITES])
def test_a_same_year_company_is_unchanged(scored, col, matched, mixed):
    """Its current figures already ARE the cash-flow year, so it must read exactly what the Grower
    reads on matched years — the fix may not reach for last year's profit unconditionally."""
    assert np.isclose(float(scored.loc["SameYear", col]), matched)


def test_the_same_year_figures_are_materialised_for_display(scored):
    g, s = scored.loc["Grower"], scored.loc["SameYear"]
    assert (g["cf_year_lag"], g["pat_cf_year"], g["ebitda_cf_year"], g["depreciation_cf_year"]) == \
        (1.0, 100.0, 150.0, 30.0)
    assert (s["cf_year_lag"], s["pat_cf_year"], s["ebitda_cf_year"], s["depreciation_cf_year"]) == \
        (0.0, 100.0, 150.0, 30.0)


# ── 3. It switches itself off when the vendor rolls the cash-flow statement ─────────────────────
def test_once_cash_flow_rolls_the_current_year_is_used_again():
    rolled = _common(name="Rolled", sector="Chemicals", pat=200.0, pat_1yb=100.0,
                     ebitda=300.0, ebitda_1yb=150.0, ebit=250.0, ebit_1yb=120.0,
                     operating_cash_flow=220.0, ocf_1yb=110.0, free_cash_flow=120.0,
                     total_assets=1200.0, total_assets_1yb=1000.0,
                     cfo_to_pat=110.0, cfo_to_ebitda=73.33, roa=16.67)
    with contextlib.redirect_stdout(_io.StringIO()):
        d = compute_forensic_signals(compute_derived_signals(_frame([rolled])))
    r = d.iloc[0]
    assert r["cf_year_lag"] == 0.0 and r["pat_cf_year"] == 200.0 and r["ebitda_cf_year"] == 300.0
    assert np.isclose(r["accruals_ratio"], (200.0 - 220.0) / 1200.0)


# ── 4. The tear-sheet shows the numbers the engine judged ───────────────────────────────────────
def test_the_accruals_evidence_shows_the_same_years_profit(scored):
    from ui.ui_tearsheet import _get_flag_context
    row = scored.loc["Accrual"]
    assert row["rf_high_accruals"] == 1, "fixture: the flag must fire on its own year (10.5%)"
    ctx = _get_flag_context(row, "rf_high_accruals")
    assert "10.5%" in ctx and "PAT ₹150cr" in ctx, (
        f"evidence {ctx!r} does not show the FY25 profit the flag was computed from")


def test_the_fcf_to_ebitda_evidence_shows_the_same_years_ebitda(scored):
    from ui.ui_tearsheet import _get_flag_context
    row = scored.loc["Accrual"]
    assert row["rf_low_fcf_ebitda"] == 1, "fixture: 20 / 200 = 10% fires"
    ctx = _get_flag_context(row, "rf_low_fcf_ebitda")
    assert "FCF/EBITDA: 10%" in ctx and "EBITDA ₹200cr" in ctx, ctx


# ── 5. Live data ────────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def raw():
    with contextlib.redirect_stdout(_io.StringIO()):
        return coerce_numeric_columns(merge_datasets(load_all_csvs("local")))


def test_the_detector_decides_almost_every_stock_that_reports_cash_flow(raw):
    lag = cash_flow_year_lag(raw)
    has_cf = pd.to_numeric(raw["operating_cash_flow"], errors="coerce").notna()
    decided = lag[has_cf].notna().mean()
    assert decided >= 0.95, f"only {decided:.1%} decided — the vendor ratios may have changed basis"


def test_cash_flow_and_balance_sheet_roll_together(raw):
    """One indicator serves both statements (the capex estimate mixes balance-sheet additions with
    depreciation). That is only sound while they move in lockstep — measured 98% on 2026-10-05."""
    cash_only = cash_flow_year_lag(raw.assign(roa=np.nan))
    bs_only = cash_flow_year_lag(raw.assign(cfo_to_pat=np.nan, cfo_to_ebitda=np.nan))
    both = cash_only.notna() & bs_only.notna()
    assert both.sum() > 1000, "too few stocks decided by both — the check went blind"
    agree = (cash_only[both] == bs_only[both]).mean()
    assert agree >= 0.95, f"cash flow and balance sheet agree on only {agree:.1%} — re-examine"


def test_the_june_vintage_reads_as_same_year():
    """Before the vendor rolled the P&L (June 2026) the statements matched. Movers re-scores archived
    vintages with this engine, so they must read 'same year' and score exactly as they did."""
    p = os.path.join(_ROOT, "Other Resources", "snapshots", "prism_snapshot_2026-06-17.csv")
    if not os.path.exists(p):
        pytest.skip("June snapshot not present")
    lag = cash_flow_year_lag(pd.read_csv(p, low_memory=False))
    assert (lag == 0).mean() >= 0.90 and (lag == 1).mean() <= 0.05, (
        f"June: same {(lag == 0).mean():.1%}, lag {(lag == 1).mean():.1%}")


def test_no_site_reads_this_years_profit_for_a_lagging_stock(raw):
    """Implementation-independent: scale the FY26 P&L of every lagging stock and re-score. A site that
    still reads this year's profit against last year's cash moves; a correct one cannot."""
    with contextlib.redirect_stdout(_io.StringIO()):
        base = compute_forensic_signals(compute_derived_signals(raw.copy()))
    lag = base["cf_year_lag"].eq(1.0).to_numpy()
    assert lag.sum() > 500, "too few lagging stocks to test — the vendor may have rolled"
    bumped = raw.copy()
    for c in ("pat", "ebitda", "ebit"):
        bumped[c] = np.where(lag, pd.to_numeric(bumped[c], errors="coerce") * 1.37, bumped[c])
    with contextlib.redirect_stdout(_io.StringIO()):
        moved = compute_forensic_signals(compute_derived_signals(bumped))
    held = moved["cf_year_lag"].eq(1.0).to_numpy() & lag   # still read as lagging after the bump
    assert held.mean() / lag.mean() > 0.97
    bad = []
    for col, _, _ in SITES + [("fcf_reconstructed", None, None)]:
        a, b = base.loc[held, col].astype(float), moved.loc[held, col].astype(float)
        n = int((~np.isclose(a, b, equal_nan=True)).sum())
        if n:
            bad.append(f"{col} ({n} stocks)")
    assert not bad, "these still read this year's profit against last year's cash: " + ", ".join(bad)


# ── 6. The reference says so ────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("key", ["Accruals Ratio", "FCF/PAT"])
def test_the_glossary_states_the_same_year_rule(key):
    from ui.ui_components import _RAW_GLOSSARY
    assert "same year" in _RAW_GLOSSARY[key].lower(), _RAW_GLOSSARY[key]
