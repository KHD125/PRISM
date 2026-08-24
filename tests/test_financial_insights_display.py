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
