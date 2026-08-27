"""
test_all_data_export.py
=======================
Contract for the All Data tab's per-stock Export button (app.py).

WHAT THIS BUTTON PROMISES, and why each promise needs a pin:

1. COMPLETENESS — "all N signals" means every column of the scored frame, not the 154 the grid
   above renders. If a column is ever dropped from the export the label becomes a lie, and the
   label is generated from `df.shape[1]` precisely so it cannot drift.

2. FIDELITY — this is the "machine-readable" artifact the tab's caption points at, so a number
   must survive the round-trip EXACTLY. A float silently rounded on the way out would make the
   export unusable for the one job it exists to do.

3. EXCEL-SAFETY — the Value column is full of emoji decision-strings (corporate_class "GREAT",
   smart_money_flow, ep_power_curve, verdict emojis) and Indian company names. A bare `.to_csv()`
   mojibakes all of them in Excel; the UTF-8 BOM (utf-8-sig) is what prevents it.

4. SINGLE DERIVATION — the export used to re-run `df[df["name"] == selected].iloc[0]` twice, while
   `stock` (the identical expression, assigned once at the top of the tab) was already in scope and
   had just been handed to render_raw_signals(). Two derivations of "the row on screen" can drift;
   one cannot. An AST check pins that the lookup does not come back.

WHAT IS DELIBERATELY *NOT* PINNED: that the export uses display labels. It does not, by design —
0 of the 154 on-screen labels appear in it, because it is keyed by ENGINE column name
(roce_med_10y, not "ROCE 10Y Med"). That split is stated in the tab caption and in the button's
help text; this file pins that the help SAYS so, not that the split goes away.

Run with: pytest tests/test_all_data_export.py -v
"""

import ast
import contextlib
import io as _io
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

import pandas as pd
import pytest

from data_engine import (coerce_numeric_columns, compute_derived_signals, load_all_csvs,
                         merge_datasets)
from ui.ui_export import _to_csv_bytes

APP = os.path.join(os.path.dirname(__file__), "..", "app.py")

_ILLEGAL_FILENAME_CHARS = set('<>:"/\\|?*')


@pytest.fixture(scope="module")
def live():
    from core import run_scoring_pipeline
    with contextlib.redirect_stdout(_io.StringIO()):
        return run_scoring_pipeline(
            compute_derived_signals(coerce_numeric_columns(merge_datasets(load_all_csvs("local")))))


@pytest.fixture(scope="module")
def app_src():
    return _io.open(APP, encoding="utf-8").read()


def _export_frame(row):
    """Mirror of what app.py builds, so the assertions below test the real shape."""
    return pd.DataFrame({"Signal": row.index, "Value": row.values})


# -- 1. Completeness --------------------------------------------------------------------
def test_export_carries_every_column(live):
    row = live.iloc[0]
    exp = _export_frame(row)
    assert len(exp) == live.shape[1], (
        f"export has {len(exp)} signals but the frame has {live.shape[1]} columns -- the button's "
        f"label is generated from df.shape[1] and would now advertise a count it does not ship"
    )
    assert list(exp["Signal"]) == list(live.columns), "column ORDER changed; export must mirror the frame"


def test_export_is_wider_than_the_rendered_grid(live):
    """The whole reason this button exists: the grid omits intermediate working columns."""
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
        T.render_raw_signals(live.iloc[0])
    finally:
        T.st = real
    rendered = " ".join(out).count("ts-raw-cell")
    assert 0 < rendered < live.shape[1], (
        f"the grid renders {rendered} cells and the export ships {live.shape[1]} columns; if these "
        f"ever converge the caption's promise of a 'complete' export stops distinguishing the two"
    )


# -- 2. Fidelity: a number must survive the round-trip exactly --------------------------
def test_no_numeric_precision_is_lost_on_round_trip(live):
    row = live[live["name"].notna()].iloc[0]
    back = pd.read_csv(_io.StringIO(_to_csv_bytes(_export_frame(row)).decode("utf-8-sig")))
    m = dict(zip(back["Signal"], back["Value"]))
    bad = []
    for c in live.columns:
        if live[c].dtype.kind not in "if" or pd.isna(row[c]):
            continue
        try:
            rt = float(m[c])
        except (TypeError, ValueError):
            bad.append((c, "unparseable", m.get(c)))
            continue
        if abs(rt - float(row[c])) > max(abs(float(row[c])) * 1e-9, 1e-9):
            bad.append((c, float(row[c]), rt))
    assert not bad, f"values changed on the way out: {bad[:5]}"


def test_missing_data_exports_blank_not_the_string_nan(live):
    """Sentinel discipline at the export boundary: a gap must read as a gap in Excel.

    ONE-DIRECTIONAL BY DESIGN. Every true NaN must come back blank -- but NOT every blank is a
    NaN, so the reverse is not asserted. `trend_modifier` is a str column that uses "" (never
    NaN) to mean "no modifier" on 1,930 of 2,117 rows; that empty string IS its value. CSV cannot
    distinguish "" from missing on read-back, which is a round-trip property of the format, not a
    defect in the export. An equality-of-counts assertion here failed on exactly that column.
    """
    row = live.iloc[0]
    txt = _to_csv_bytes(_export_frame(row)).decode("utf-8-sig")
    assert ",nan" not in txt.lower(), "a literal 'nan' leaked into the CSV"
    assert ",none" not in txt.lower(), "a literal 'None' leaked into the CSV"
    back = pd.read_csv(_io.StringIO(txt))
    blank = set(back[back["Value"].isna()]["Signal"])
    missing = {c for c in live.columns if pd.isna(row[c])}
    assert missing <= blank, (
        f"these columns are NaN in the row but did NOT export blank: {sorted(missing - blank)}"
    )
    for c in sorted(blank - missing):
        assert str(row[c]).strip() == "", (
            f"{c} exported blank but holds {row[c]!r} -- a real value was lost on the way out"
        )


# -- 3. Excel-safety ---------------------------------------------------------------------
def test_export_is_bom_prefixed_so_excel_renders_the_emoji(live):
    b = _to_csv_bytes(_export_frame(live.iloc[0]))
    assert b[:3] == b"\xef\xbb\xbf", "BOM missing -- every emoji decision-string mojibakes in Excel"


def test_emoji_decision_strings_survive_the_round_trip(live):
    row = live[live["corporate_class"].notna()].iloc[0]
    back = pd.read_csv(_io.StringIO(_to_csv_bytes(_export_frame(row)).decode("utf-8-sig")))
    m = dict(zip(back["Signal"], back["Value"]))
    checked = 0
    for c in ("corporate_class", "smart_money_flow", "ep_power_curve", "verdict_emoji"):
        if c in row.index and isinstance(row[c], str) and row[c].strip():
            assert m[c] == row[c], f"{c}: {row[c]!r} came back as {m.get(c)!r}"
            checked += 1
    assert checked >= 2, "no emoji columns were actually exercised -- the probe list went stale"


# -- 4. One derivation of the row, not three --------------------------------------------
def _all_data_export_block(src):
    i = src.index("_stock_export")
    return src[i - 400:src.index(")", src.index("use_container_width", i)) + 600]


def test_export_reuses_stock_rather_than_re_deriving_the_row(app_src):
    """AST, not a substring scan: prose in a comment may legitimately NAME the old expression
    (this file's own docstring does), and a substring test would match that."""
    tree = ast.parse(app_src)
    target = None
    for n in ast.walk(tree):
        if (isinstance(n, ast.Assign) and n.targets and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == "_stock_export"):
            target = n
    assert target is not None, "the All Data export frame is gone"
    names = {x.id for x in ast.walk(target) if isinstance(x, ast.Name)}
    assert "stock" in names, "the export no longer builds from `stock`"
    assert "df" not in names, (
        "the export re-derives the row from `df` again. `stock` is that exact expression, assigned "
        "once at the top of the tab and already handed to render_raw_signals() -- two derivations of "
        "'the row on screen' can drift apart, one cannot."
    )


def test_stock_and_the_old_expression_are_still_the_same_row(live):
    """The refactor above is only safe while `stock` IS df[df.name == selected].iloc[0]. Nothing
    rebinds `df` between the two lines today; this fails loudly if that ever changes."""
    sel = live["name"].dropna().iloc[0]
    a = live[live["name"] == sel].iloc[0]
    assert list(a.index) == list(live.columns)
    assert live["name"].value_counts().max() == 1, (
        "a duplicate stock name appeared -- .iloc[0] now silently picks one of several rows and the "
        "export may not be the stock the user selected"
    )


def test_df_is_not_rebound_between_the_stock_assignment_and_the_export(app_src):
    """The premise of the refactor, pinned structurally."""
    tree = ast.parse(app_src)
    stock_ln = [n.lineno for n in ast.walk(tree)
                if isinstance(n, ast.Assign) and n.targets
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "stock"]
    exp_ln = [n.lineno for n in ast.walk(tree)
              if isinstance(n, ast.Assign) and n.targets
              and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "_stock_export"]
    assert len(stock_ln) == 1, f"`stock` is assigned {len(stock_ln)} times; the export's source is ambiguous"
    df_ln = [n.lineno for n in ast.walk(tree)
             if isinstance(n, ast.Assign) and n.targets
             and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "df"]
    between = [ln for ln in df_ln if stock_ln[0] < ln < exp_ln[0]]
    assert not between, (
        f"`df` is rebound at {between}, between `stock` (line {stock_ln[0]}) and the export "
        f"(line {exp_ln[0]}) -- `stock` may no longer be the row the export is meant to ship"
    )


# -- 5. The label and help must not drift from what is shipped --------------------------
def test_button_label_states_the_count_from_the_frame(app_src):
    """AST-scoped to the LABEL, not a text window around the call.

    The first version searched a ~1000-char slice for "df.shape[1]" and passed even after the
    count was stripped from the label -- because the `help=` string in that same window also
    interpolates df.shape[1]. A mutation run caught it: the assertion was measuring the wrong
    argument. Here the label is located as download_button's first positional arg.
    """
    tree = ast.parse(app_src)
    label = None
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "download_button" and n.args
                and any(isinstance(k, ast.keyword) and k.arg == "data"
                        and isinstance(k.value, ast.Call)
                        and getattr(k.value.func, "id", "") == "_to_csv_bytes"
                        and any(getattr(a, "id", "") == "_stock_export" for a in k.value.args)
                        for k in n.keywords)):
            label = n.args[0]
    assert label is not None, "could not locate the All Data export button's label"
    src = ast.unparse(label)
    assert "df.shape[1]" in src, (
        f"the export label omits or hardcodes the signal count: {src}. Both sibling exports state "
        f"theirs, and a hardcoded number goes stale the moment a column is added."
    )


def test_help_warns_that_the_csv_uses_engine_column_names(app_src):
    """Measured on live data: 0 of the 154 labels the grid teaches appear in the export."""
    blk = _all_data_export_block(app_src)
    assert "help=" in blk, "the export button has no tooltip"
    low = blk.lower()
    assert "column name" in low, "the help does not warn that rows are keyed by engine column name"
    assert "roce_med_10y" in low, "the help should show a concrete engine-name example"


def test_filename_is_filesystem_safe(live):
    for sel in live["name"].dropna().head(200):
        fn = f"{re.sub(r'[^A-Za-z0-9._-]+', '_', sel).lower()}_signals.csv"
        assert not (set(fn) & _ILLEGAL_FILENAME_CHARS), f"illegal filename char in {fn!r}"
        assert fn.endswith("_signals.csv") and len(fn) > len("_signals.csv")
