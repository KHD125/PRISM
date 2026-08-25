"""Contract: the tearsheet's verdict ICONS tell the truth about the numbers beside them.

THE BUG THIS PINS (found 2026-08-24 from a user's pasted panel). `_row`/`_ds` decided ✅/❌/⚪ with
IDENTITY checks — `if passed is True`. But every threshold in these panels compares a NumPy float,
and `np.float64(18.4) >= 15` returns np.bool_, for which `x is True` is **False**. Both branches
missed, so 13 of 17 financial rows and all 5 Deep-Signals chips fell to the neutral ⚪ branch on
EVERY stock — and ⚪ is this codebase's honest-blank signal, so the panel claimed "no data" about
data it had computed correctly. EPack Prefab rendered ROCE 18.4% (≥15), CFO/PAT 146.5% (≥70),
PEG 0.62 (≤1.0) and Pledge 0.0% all as grey shrugs.

Invisible to everything: nothing crashes, no NaN leaks, the render sweeps passed — the output is
WELL-FORMED BUT WRONG. Hence the differential test at the bottom, which reads each row's OWN stated
threshold out of its context string and asserts the icon agrees.
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

import ui.ui_tearsheet as ts

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Other Resources", "CSV Data")


class _StubST:
    """Capture markdown instead of rendering (the module is stateless — 6 st.* calls total)."""
    def __init__(self): self.out = []
    def markdown(self, x=None, **k): self.out.append(str(x))
    def write(self, x=None, **k): pass
    def info(self, x=None, **k): pass
    def warning(self, x=None, **k): pass
    def success(self, x=None, **k): pass
    def plotly_chart(self, fig=None, **k): pass


def _render(fn, stock):
    stub = _StubST()
    real, ts.st = ts.st, stub
    try:
        fn(stock)
    finally:
        ts.st = real
    return " ".join(stub.out)


def _text(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))


# ── 1. the root cause, in isolation ───────────────────────────────────────────
def test_numpy_bool_must_not_read_as_unknown():
    """The exact trap: np.bool_ is neither `is True` nor `is False`. Any icon helper that decides
    on identity MUST coerce first, or a computed verdict renders as the unknown state."""
    np_true, np_false = (np.float64(18.4) >= 15), (np.float64(4.8) >= 10)
    assert np_true is not True and np_false is not False, (
        "NumPy comparison returned a real Python bool — the trap this test guards is gone; "
        "re-read the helpers before relaxing anything."
    )
    assert bool(np_true) is True and bool(np_false) is False, "bool() coercion is the fix"


@pytest.mark.parametrize("passed,icon", [
    (np.float64(18.4) >= 15, "✅"),      # NumPy True  -> pass
    (np.float64(4.8) >= 10, "❌"),       # NumPy False -> fail
    (True, "✅"), (False, "❌"),          # Python bools still work
    (None, "⚪"),                        # genuine unknown PRESERVED
])
def test_row_icon_for_every_input_shape(passed, icon):
    """_row must render the right icon for NumPy bools, Python bools AND None. Extracted from the
    live module so it can never drift from what ships."""
    import ast
    import inspect
    import textwrap

    # AST-extract the nested _row exactly. A substring slice swallowed the following helper and
    # died on indentation — same lesson as the discovery-filter contracts: parse, don't slice.
    src = textwrap.dedent(inspect.getsource(ts.render_financial_insights))
    node = next((n for n in ast.walk(ast.parse(src))
                 if isinstance(n, ast.FunctionDef) and n.name == "_row"), None)
    assert node is not None, "_row not found inside render_financial_insights"
    ns = {"COLORS": ts.COLORS, "_esc": ts._esc}
    exec(textwrap.dedent(ast.get_source_segment(src, node)), ns)
    assert icon in ns["_row"]("L", passed, "V", "ctx")


# ── 2. the SSGR row's three honest states ─────────────────────────────────────
def _one_row(stock, label):
    txt = _text(_render(ts.render_financial_insights, stock))
    m = re.search(r"(✅|❌|⚪)\s*" + re.escape(label) + r"\s*([^✅❌⚪]{0,120})", txt)
    return (m.group(1), m.group(2).strip()) if m else (None, "")


def _base_stock(**over):
    row = {"ssgr": 25.0, "ssgr_cushion": 4.0, "sales_profit_conversion": 18.2,
           "roce_med_10y": 18.4, "roce": 18.3, "pat_gr_5y": 37.2, "rev_gr_5y": 44.6,
           "npm": 5.9, "npm_med_5y": 4.8, "cfo_to_pat": 146.5, "debt_to_equity": 0.15,
           "interest_coverage": 4.9, "tax_rate_est": 24.4, "pe_discount": -12.9,
           "fcf_yield": 0.9, "earnings_yield": 4.3, "peg": 0.62, "peg_zone": "🟢 Fair PEG",
           "promoter_holdings": 65.0, "pledged_percentage": 0.0, "fii_holdings": 11.1,
           "dii_holdings": 3.0, "change_promoter_lq": 0.0, "smart_money_flow": "⚪ Neutral"}
    row.update(over)
    return pd.Series(row)


def test_ssgr_unknown_is_neutral_never_an_accusation():
    """428 live rows (20.2%) have a NaN SSGR. They must read ⚪ 'unavailable' — NOT a red ✗
    'External Capital — actual growth exceeds SSGR 0.0%', which fabricated a verdict from a hole
    on stocks ranked #1/#3/#4/#6."""
    icon, body = _one_row(_base_stock(ssgr=np.nan, ssgr_cushion=np.nan), "Growth Funding")
    assert icon == "⚪", f"unknown SSGR must be neutral, got {icon}"
    assert "unavailable" in body.lower()
    assert "0.0%" not in body and "External Capital" not in body


def test_ssgr_states_the_gap_not_just_the_level():
    """The old fail text said only the level in a sentence parsed as the gap ('exceeds SSGR 13.5%'
    reads as 'exceeds BY 13.5%'). Sarda: SSGR 13.5%, shortfall −7.4%. Both branches must state the
    level AND the gap."""
    icon, body = _one_row(_base_stock(ssgr=13.5, ssgr_cushion=-7.4), "Growth Funding")
    assert icon == "❌"
    assert "13.5" in body and "7.4" in body, f"must state level AND gap: {body!r}"
    icon2, body2 = _one_row(_base_stock(ssgr=25.0, ssgr_cushion=4.0), "Growth Funding")
    assert icon2 == "✅"
    assert "25.0" in body2 and "4.0" in body2


def test_ssgr_keeps_the_short_value_long_context_shape():
    """LAYOUT REGRESSION GUARD. _row renders [icon][label][value nowrap][context ellipsis]: a
    sentence in the VALUE slot consumes the row and pushes the context off the screen entirely —
    which the first version of the SSGR fix did (user-reported from a live screenshot). The value
    must stay a bare number like every sibling row; the prose belongs in the context."""
    html = _render(ts.render_financial_insights, _base_stock(ssgr=13.5, ssgr_cushion=-7.4))
    m = re.search(r"Growth Funding \(SSGR\)</span>\s*<span[^>]*>([^<]*)</span>", html)
    assert m, "SSGR value span not found — the row structure changed"
    value = m.group(1).strip()
    assert value == "13.5%", f"value slot must hold the bare level, got {value!r}"
    assert len(value) <= 8, "a sentence in the value slot squeezes the context off the row"


def test_sales_profit_conversion_names_its_real_basis():
    """The help text claimed 'PAT CAGR > Revenue CAGR' — only the FALLBACK. The engine prefers
    EBIT 3Y − Revenue 3Y, so EPack showed +18.2pp here while the row above showed 5Y PAT trailing
    revenue: two true statements reading as a contradiction."""
    icon, body = _one_row(_base_stock(), "Sales→Profit Conversion")
    assert icon == "✅" and "18.2" in body, f"must surface the computed value: {body!r}"
    assert "EBIT 3Y" in body, "row must name the basis the engine actually used"
    icon2, _ = _one_row(_base_stock(sales_profit_conversion=np.nan), "Sales→Profit Conversion")
    assert icon2 == "⚪", "unknown conversion must be neutral, not 'Negative'"


# ── 3. THE DIFFERENTIAL NET — each row judged against its OWN stated threshold ──
@pytest.mark.skipif(not os.path.isdir(_DATA_DIR), reason="local CSV data absent")
def test_icons_agree_with_the_thresholds_the_rows_themselves_state():
    """The generalized detector for this whole bug family. Each row prints its own rule in the
    context text ('≥70%: real cash', '≥4% justifies equity risk'). Parse the rule, parse the
    value, and assert the icon agrees. This is what makes a well-formed-but-wrong panel fail
    LOUDLY — the class no crash test, NaN test or render sweep can see."""
    from core import run_scoring_pipeline
    from core.data_engine import (coerce_numeric_columns, compute_derived_signals,
                                  load_all_csvs, merge_datasets)
    import contextlib
    import io as _io

    with contextlib.redirect_stdout(_io.StringIO()):
        df = run_scoring_pipeline(compute_derived_signals(coerce_numeric_columns(
            merge_datasets(load_all_csvs("local")))))

    # (row label, value regex, rule) — rules taken verbatim from the rows' own context strings
    CHECKS = [
        ("Cash Earnings", r"([\d.]+)%", lambda v: v >= 70),
        ("Earnings Yield", r"([\d.]+)%", lambda v: v >= 4),
        ("Promoter Pledge", r"([\d.]+)%", lambda v: v <= 10),
        ("ROCE — 10Y Median", r"([\d.]+)%", lambda v: v >= 15),
    ]
    failures = []
    for i in range(0, len(df), 350):            # stratified sample across the ranked universe
        stock = df.iloc[i]
        txt = _text(_render(ts.render_financial_insights, stock))
        for label, val_re, rule in CHECKS:
            m = re.search(r"(✅|❌|⚪)\s*" + re.escape(label) + r"\s*" + val_re, txt)
            if not m:
                continue
            icon, val = m.group(1), float(m.group(2))
            want = "✅" if rule(val) else "❌"
            if icon != want:
                failures.append(f"{stock['name']}: {label}={val} -> {icon}, expected {want}")
    assert not failures, (
        "icon contradicts the threshold the row itself states (the NumPy-bool class):\n  "
        + "\n  ".join(failures[:10])
    )


# ══════════════════════════════════════════════════════════════════════════════
# FORENSIC FLAG EVIDENCE — every fired flag must show NUMBERS from the metric it
# actually tested, and the Clean verdict must respect severity.
# ══════════════════════════════════════════════════════════════════════════════
# Two classes found from user screenshots (2026-08-24), both invisible to crash/NaN tests:
#   * WRONG METRIC — the evidence line printed a DIFFERENT column than the flag fired on, so it
#     appeared to REFUTE its own accusation. rf_ssgr_deficit showed pat_gr_yoy 9.4% under the
#     title "growth exceeds SSGR" while the flag used revenue growth 20.9% vs SSGR 13.5%.
#     rf_opm_volatile showed the CURRENT opm while the engine tests opm_1yb (52/474 rows, 11%,
#     displayed a pair that does not breach the stated 30% threshold).
#   * NO EVIDENCE — 5 of 28 flags fell back to prose ("Beneish TATA forensic signal") or, for
#     rf_dilution, rendered a title above a BLANK line (626 firing stocks).


@pytest.mark.skipif(not os.path.isdir(_DATA_DIR), reason="local CSV data absent")
def test_every_fired_flag_shows_numeric_evidence():
    """No fired forensic flag may fall back to a slogan or a blank line: the reader must see the
    measured value behind the accusation. Sampled across every flag that fires on live data."""
    import contextlib
    import io as _io

    from core import run_scoring_pipeline
    from core.data_engine import (coerce_numeric_columns, compute_derived_signals,
                                  load_all_csvs, merge_datasets)
    from ui.ui_tearsheet import _FLAG_DISPLAY, _get_flag_context

    with contextlib.redirect_stdout(_io.StringIO()):
        df = run_scoring_pipeline(compute_derived_signals(coerce_numeric_columns(
            merge_datasets(load_all_csvs("local")))))

    offenders = []
    for col in _FLAG_DISPLAY:
        if col not in df.columns:
            continue
        fired = df[df[col].fillna(0) == 1]
        if fired.empty:
            continue
        # a flag passes if it yields numeric evidence on the majority of its firing rows
        sample = min(20, len(fired))
        with_numbers = sum(bool(re.search(r"\d", _get_flag_context(fired.iloc[i], col) or ""))
                           for i in range(sample))
        if with_numbers < sample * 0.5:
            offenders.append(f"{col} ({with_numbers}/{sample} rows show numbers, fires {len(fired)}x)")
    assert not offenders, (
        "forensic flags falling back to prose or a blank line instead of showing the measured "
        f"value: {offenders}. Add a handler in _get_flag_context mirroring the engine's condition."
    )


@pytest.mark.skipif(not os.path.isdir(_DATA_DIR), reason="local CSV data absent")
def test_flag_evidence_never_refutes_its_own_flag():
    """WRONG-METRIC GUARD for the two flags whose evidence states an explicit threshold: the
    numbers shown must actually breach that threshold. This is what would have caught the
    rf_ssgr_deficit ('9.4% exceeds 13.5%'?) and rf_opm_volatile mismatches."""
    import contextlib
    import io as _io

    from core import run_scoring_pipeline
    from core.data_engine import (coerce_numeric_columns, compute_derived_signals,
                                  load_all_csvs, merge_datasets)
    from ui.ui_tearsheet import _get_flag_context

    with contextlib.redirect_stdout(_io.StringIO()):
        df = run_scoring_pipeline(compute_derived_signals(coerce_numeric_columns(
            merge_datasets(load_all_csvs("local")))))

    bad = []
    ssgr_fired = df[df["rf_ssgr_deficit"].fillna(0) == 1]
    for i in range(0, len(ssgr_fired), max(1, len(ssgr_fired) // 40)):
        ctx = _get_flag_context(ssgr_fired.iloc[i], "rf_ssgr_deficit")
        m = re.search(r"growth:\s*([\d.-]+)%.*?SSGR:\s*([\d.-]+)%", ctx or "")
        if m and float(m.group(1)) <= float(m.group(2)):
            bad.append(f"ssgr {ssgr_fired.iloc[i]['name']}: growth {m.group(1)} !> SSGR {m.group(2)}")

    opm_fired = df[df["rf_opm_volatile"].fillna(0) == 1]
    for i in range(0, len(opm_fired), max(1, len(opm_fired) // 40)):
        ctx = _get_flag_context(opm_fired.iloc[i], "rf_opm_volatile")
        m = re.search(r"deviation\s*(\d+)%\s*·\s*threshold:\s*>(\d+)%", ctx or "")
        if m and int(m.group(1)) <= int(m.group(2)):
            bad.append(f"opm {opm_fired.iloc[i]['name']}: deviation {m.group(1)}% !> {m.group(2)}%")

    assert not bad, ("flag evidence contradicts the flag that fired:\n  " + "\n  ".join(bad[:10]))


def test_clean_verdict_respects_severity():
    """'No Material Red Flags' must not print above a CRITICAL flag. 201 of 639 Clean labels
    (31.5%) did exactly that before the severity gate was added."""
    from ui.ui_tearsheet import _CRITICAL_FLAG_COLS, _forensic_status

    assert len(_CRITICAL_FLAG_COLS) == 7, "critical severity set changed — re-verify _FLAG_DISPLAY"
    clean_txt, _, is_clean = _forensic_status(89.0, 3, has_critical=False)
    assert is_clean and "Clean" in clean_txt
    elev_txt, _, is_clean2 = _forensic_status(89.0, 3, has_critical=True)
    assert not is_clean2 and "Elevated" in elev_txt, "a critical flag must block the Clean verdict"
    # a genuinely bad stock still reads Sharp Practices regardless of severity flag
    assert "Sharp" in _forensic_status(39.0, 17, has_critical=True)[0]


@pytest.mark.skipif(not os.path.isdir(_DATA_DIR), reason="local CSV data absent")
def test_every_stated_threshold_is_actually_breached():
    """GENERAL wrong-metric net (2026-08-24). Supersedes the two hand-picked flags above.

    Four flags were caught displaying a column their engine condition never tested — evidence that
    APPEARS TO REFUTE its own accusation, which is worse than showing nothing because it teaches
    the reader to distrust a flag that is actually correct:
      * rf_ssgr_deficit   showed pat_gr_yoy (profit) — engine uses revenue growth
      * rf_opm_volatile   showed current opm       — engine uses opm_1yb
      * rf_high_cash_debt showed D/E               — engine tests cash vs debt×0.3 (443/878 rows
                                                     had D/E<0.1, i.e. "high debt · D/E: 0.01")
      * rf_inventory_bloat showed rev_gr_yoy       — engine tests inv_vs_rev_gap

    Every evidence string that states "threshold: <op><number>" now states the TRIGGER (what makes
    the flag fire), so this test can mechanically verify the leading value actually satisfies it.
    Any future handler that shows the wrong metric will almost always fail this.
    """
    import contextlib
    import io as _io

    from core import run_scoring_pipeline
    from core.data_engine import (coerce_numeric_columns, compute_derived_signals,
                                  load_all_csvs, merge_datasets)
    from ui.ui_tearsheet import _FLAG_DISPLAY, _get_flag_context

    with contextlib.redirect_stdout(_io.StringIO()):
        df = run_scoring_pipeline(compute_derived_signals(coerce_numeric_columns(
            merge_datasets(load_all_csvs("local")))))

    # "…: <value><unit> … threshold: <op><number>" — take the LAST number before 'threshold'
    THRESH = re.compile(r"threshold:\s*([<>])\s*([\d.]+)")
    VALUE = re.compile(r"(-?[\d,]+\.?\d*)\s*(?:%|×|d|pp|cr)?(?=[^0-9]*$)")

    violations, checked = [], 0
    for col in _FLAG_DISPLAY:
        if col not in df.columns:
            continue
        fired = df[df[col].fillna(0) == 1]
        if fired.empty:
            continue
        # EVERY fired row, not a sample. This net previously walked 25 rows per flag, which made
        # it a coin flip against a sparse offender: rf_high_receivables stated a stale ">75d" that
        # was wrong for just 14 of its 664 firing stocks (~2%), so a 25-row sample missed it for
        # months and only surfaced when an unrelated scoring change reshuffled `rank` — i.e. row
        # order, i.e. which rows the sample happened to land on. A correctness net must not depend
        # on that. Full scan costs a few seconds; a wrong-metric bug shipped costs a user's trust.
        for i in range(len(fired)):
            ctx = _get_flag_context(fired.iloc[i], col) or ""
            tm = THRESH.search(ctx)
            if not tm:
                continue
            op, thr = tm.group(1), float(tm.group(2))
            head = ctx[: tm.start()]
            nums = re.findall(r"(-?[\d,]+\.?\d*)", head.replace(",", ""))
            if not nums:
                continue
            val = float(nums[-1])          # the measured quantity sits immediately before it
            # EPS absorbs display rounding only (NFAT 1.497 renders "1.50"; a 30.04% deviation
            # renders "30.0"). Wrong-metric bugs miss by whole units, never by 0.05.
            _EPS = 0.05
            ok = (val > thr - _EPS) if op == ">" else (val < thr + _EPS)
            checked += 1
            if not ok:
                violations.append(
                    f"{col} [{fired.iloc[i]['name'][:26]}]: shows {val} but claims "
                    f"'threshold: {op}{thr}' — evidence does not breach its own threshold "
                    f"| ctx={ctx[:90]!r}")
                break                      # one example per flag is enough

    assert checked >= 8, f"only {checked} threshold statements found — the net went blind"
    assert not violations, (
        f"{len(violations)} flag(s) display evidence that does not satisfy the threshold they "
        "state — the wrong-metric class:\n  " + "\n  ".join(violations))


# ══════════════════════════════════════════════════════════════════════════════
# BOOK + ENGINE FIDELITY (2026-08-25) — the display must not contradict either
# ══════════════════════════════════════════════════════════════════════════════

def test_earnings_yield_uses_the_live_gsec_constant_not_a_literal():
    """FALSE-GREEN GUARD. The bar was a hardcoded 4%, so a 6.0% yield printed '✅ justifies equity
    risk' while risk-free G-Secs paid 7% — the panel endorsing a sub-risk-free return. Malik's rule
    is RELATIVE ('greater than long-term government bond yields') and config already carries
    INDIA_GSEC_YIELD. Pinned so the literal can never come back."""
    import inspect

    from config import INDIA_GSEC_YIELD

    src = inspect.getsource(ts.render_financial_insights)
    assert "ey >= INDIA_GSEC_YIELD" in src, "Earnings Yield must compare against the live constant"
    assert "ey >= 4" not in src, "the hardcoded 4% bar is back — it endorses sub-risk-free returns"
    # behaviour: a yield BELOW the G-Sec must fail, one above must pass
    below, _ = _one_row(_base_stock(earnings_yield=INDIA_GSEC_YIELD - 1.0), "Earnings Yield")
    above, _ = _one_row(_base_stock(earnings_yield=INDIA_GSEC_YIELD + 1.0), "Earnings Yield")
    assert below == "❌", "a yield under the risk-free rate must not read as a pass"
    assert above == "✅"


def test_tax_band_matches_the_post_2019_indian_regime():
    """Malik's '>30%' was the statutory rate's value when he wrote it ('In India, the corporate tax
    rate is 30%', p.45). India cut it in 2019 to ~25.17%; the live median effective rate is 25.4%.
    The old 30–55% band failed 88.1% of the universe — 1,496 stocks flagged for paying the current
    legal rate. 20–40% spans both regimes and restores the book's actual (relative) rule."""
    import inspect

    src = inspect.getsource(ts.render_financial_insights)
    assert "(20 <= tax <= 40)" in src, "tax band must span both post-2019 regimes"
    assert "(30 <= tax <= 55)" not in src, "the pre-2019 band is back"
    assert _one_row(_base_stock(tax_rate_est=25.4), "Tax Rate")[0] == "✅", \
        "the median Indian company (25.4%) pays the statutory rate and must pass"
    assert _one_row(_base_stock(tax_rate_est=34.9), "Tax Rate")[0] == "✅", "old regime must pass"
    assert _one_row(_base_stock(tax_rate_est=8.0), "Tax Rate")[0] == "❌", \
        "a sharp-practices-level rate must still fail"


def test_net_margin_bar_matches_the_engines_own_malik_pillar():
    """The row demanded NPM ≥10 while the engine's malik_profit_stability uses ≥8 (the book's
    number) — the panel was stricter than the engine beside it."""
    import inspect

    assert "npm_5y >= 8," in inspect.getsource(ts.render_financial_insights)
    assert _one_row(_base_stock(npm_med_5y=8.5), "Net Margin")[0] == "✅", \
        "8.5% clears Malik's >8% bar and must pass"
    assert _one_row(_base_stock(npm_med_5y=6.0), "Net Margin")[0] == "❌"
