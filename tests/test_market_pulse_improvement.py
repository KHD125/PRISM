"""
test_market_pulse_improvement.py
================================
Contract for the 🚀 Improvement column in Market Pulse → 📈 Sectors and 🏭 Industry (2026-10-03).

WHY THE COLUMN EXISTS. `improvement_score` (20% of quality since 2026-10-03) asks whether a business
got BETTER over the last year: profit growth, revenue growth, operating-margin change. Averaged over
a GROUP it was the strongest group-level signal measured: 3-year rank-IC +0.440 for sectors and
+0.319 for industries (>= 5 stocks; top-minus-bottom fifth +11.5 / +12.7 pp a year), and still
+0.400 / +0.284 after controlling for the group's prior price run; 1-year +0.159 / +0.195. PRISM's
own two short windows are MIXED — 66 days +0.029 / -0.028, 28 days +0.267 / +0.232 (sectors /
industries throughout) — which is why it is a COLUMN and never the sort.

WHY IT IS NOT ALREADY ON SCREEN. The tables already show average Quality, and improvement is a
fifth of quality — but at group level the two are unrelated: Spearman -0.10 … +0.20 across all four
saved vintages. Against average Momentum it is +0.19 … +0.52: related, not the same. So it sits
BESIDE Momentum, where the pair separates "the businesses are getting better and the price has
noticed" from "getting better, the price has not" — and from the opposite shape, which the live
frame shows in Sugar at build: Momentum 54.6 over an Improvement of 33.4, a price running on
businesses that are getting worse.

WHAT IS PINNED — each one is invisible on screen when it breaks:
  1. it is averaged in the SAME .agg call as the table's other averages, so every tab filter and the
     🎯 scope reach it — EXECUTED on a synthetic frame, never a source scan (`if False and ...`
     satisfies any presence scan; this suite has paid for that more than once);
  2. it is displayed, beside Momentum — the displayed order is EVALUATED, filter included;
  3. it is a bar on the same scale as its neighbours, headed "Improvement", carrying ONE tooltip
     shared by both tables;
  4. that tooltip's definition IS the glossary entry every other surface shows (single source);
  5. it is a column, not the ranking — neither table's sort reads it;
  6. the Sectors caption, which lists the averaged columns by name, names it (caption drift);
  7. at group level it is not a restatement of Quality or Momentum (re-measured on live data).

Run with: pytest tests/test_market_pulse_improvement.py -v
"""

import ast
import contextlib
import io as _io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

import numpy as np
import pandas as pd
import pytest

_APP = os.path.join(os.path.dirname(__file__), "..", "app.py")
SRC = _io.open(_APP, encoding="utf-8").read()
TREE = ast.parse(SRC)

# table -> (groupby key, the frame its averages are taken over, display-order var, stats frame var)
TABLES = {
    "sectors":  ("sector",   "_sec_src", "_sec_order", "_sec_stats"),
    "industry": ("industry", "_ind_src", "_ind_order", "_ind_stats"),
}
# Where each table's own code lives — the inner-tab headers. A check on one table must not pass on
# the other's near-identical code.
_BLOCKS = {"sectors": ("# ══ Sectors ══", "# ══ Industry ══"),
           "industry": ("# ══ Industry ══", "# ══ Movers ══")}


def _lines(table):
    """1-based [first, last] source lines of a table's block."""
    a, b = _BLOCKS[table]
    return SRC[:SRC.index(a)].count("\n") + 1, SRC[:SRC.index(b)].count("\n") + 1


def _in_block(node, table):
    lo, hi = _lines(table)
    return lo <= node.lineno <= hi


def _frame():
    """Two groups. Group A's improvement mean (40) differs from its median (20) and from every
    other column's group value, so the wrong column OR the wrong statistic gives a wrong answer."""
    return pd.DataFrame({
        "sector":            ["A", "A", "A", "B", "B"],
        "industry":          ["X", "X", "X", "Y", "Y"],
        "name":              ["a1", "a2", "a3", "b1", "b2"],
        "gate_pass":         [1, 0, 1, 0, 0],
        "quality_score":     [70.0, 70.0, 70.0, 30.0, 30.0],
        "momentum_score":    [55.0, 55.0, 55.0, 45.0, 45.0],
        "valuation_score":   [50.0, 50.0, 50.0, 50.0, 50.0],
        "composite_score":   [60.0, 60.0, 60.0, 40.0, 40.0],
        "improvement_score": [10.0, 20.0, 90.0, 40.0, 80.0],
    })


def _agg_call(table):
    """The table's averaging call: `<src>.groupby("<key>").agg(..., avg_quality=..., ...)`."""
    key, src_name, _, _ = TABLES[table]
    found = []
    for n in ast.walk(TREE):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "agg" and isinstance(n.func.value, ast.Call)):
            continue
        g = n.func.value
        if (isinstance(g.func, ast.Attribute) and g.func.attr == "groupby"
                and isinstance(g.func.value, ast.Name) and g.func.value.id == src_name
                and g.args and isinstance(g.args[0], ast.Constant) and g.args[0].value == key
                and any(k.arg == "avg_quality" for k in n.keywords)):
            found.append(n)
    assert len(found) == 1, f"{table}: expected ONE averaging call over {src_name}, found {len(found)}"
    return found[0]


def _run(node, ns):
    expr = ast.fix_missing_locations(ast.Expression(body=node))
    return eval(compile(expr, "<app.py>", "eval"), {"pd": pd, "np": np, **ns})


def _source(table):
    """The frame the table averages over, built by EXECUTING the app's own first assignment of it
    from `_mp_df` — including any column projection it reads (🏭 Industry projects to _IND_KEEP).

    FOUND IN THE BROWSER, 2026-10-03: the first build of this file ran the shipped aggregation on a
    frame the TEST built, which proved the arithmetic and nothing about the path the data takes to
    it. The Industry tab projects its stocks to a fixed column list, improvement_score was not on
    it, and the tab died with a KeyError while all 12 of these tests passed."""
    _, src_name, _, _ = TABLES[table]
    first = min((n for n in ast.walk(TREE) if isinstance(n, ast.Assign) and _in_block(n, table)
                 and any(isinstance(t, ast.Name) and t.id == src_name for t in n.targets)),
                key=lambda n: n.lineno)
    ns = {"_mp_df": _frame()}
    for name in sorted({x.id for x in ast.walk(first.value) if isinstance(x, ast.Name)} - {"_mp_df"}):
        ns[name] = _run(_assigned(name).value, ns)
    return _run(first.value, ns)


def _assigned(name):
    hits = [n for n in ast.walk(TREE) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    assert len(hits) == 1, f"expected ONE assignment to {name}, found {len(hits)}"
    return hits[0]


def _column_config(table):
    """key -> value node of the column_config dict passed to the table's st.dataframe call."""
    _, _, order, stats = TABLES[table]
    want = f"{stats}[{order}].reset_index()"
    for n in ast.walk(TREE):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "dataframe" and n.args and ast.unparse(n.args[0]) == want):
            cfg = next(k.value for k in n.keywords if k.arg == "column_config")
            return {k.value: v for k, v in zip(cfg.keys, cfg.values) if isinstance(k, ast.Constant)}
    raise AssertionError(f"no st.dataframe({want}, ...) call — the {table} table moved")


def _kw(call, name):
    return next((k.value for k in call.keywords if k.arg == name), None)


def _displayed(table):
    """The columns the table ACTUALLY shows: the order comprehension evaluated, its filter included,
    against the columns the shipped aggregation really produces plus the ones added after it."""
    key, src_name, order, stats = TABLES[table]
    agg = _run(_agg_call(table), {src_name: _source(table)})
    extra = ["pct_tier", "delta_vs_sector", "dom_sector"]
    cols = pd.DataFrame(columns=list(agg.columns) + extra)
    return _run(_assigned(order).value, {stats: cols})


# ── 1. Averaged over the same stocks as everything else in the row ────────────────────────────
@pytest.mark.parametrize("table", sorted(TABLES))
def test_it_is_averaged_over_the_same_stocks_as_the_other_columns(table):
    """In the SAME .agg call as avg_quality, so the cap / cyclicality / wealth filters and the 🎯
    scope — all applied to the frame BEFORE this call — reach it with no extra wiring. A separate
    `df.groupby(...)` would silently average the whole universe beside columns that do not. The
    call runs on the frame the APP builds (_source), so a projection that drops the score fails."""
    key, src_name, _, _ = TABLES[table]
    call = _agg_call(table)
    assert "avg_improvement" in {k.arg for k in call.keywords}, (
        f"{table}: avg_improvement is not averaged alongside the table's other columns")
    src = _source(table)
    assert "improvement_score" in src.columns, (
        f"{table}: the frame the tab builds from _mp_df drops improvement_score — the tab would "
        f"raise KeyError on render")
    got = _run(call, {src_name: src})["avg_improvement"]
    want = _frame().groupby(key)["improvement_score"].mean()
    pd.testing.assert_series_equal(got, want, check_names=False)


# ── 2. Displayed, beside Momentum ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("table", sorted(TABLES))
def test_it_is_displayed_beside_momentum(table):
    """Its tooltip tells the reader to read it beside Momentum, so that must be literally true:
    the pair is the point (getting better AND the price has noticed, or not yet)."""
    cols = _displayed(table)
    assert "avg_improvement" in cols, f"{table}: computed but not displayed — {cols}"
    assert abs(cols.index("avg_improvement") - cols.index("avg_momentum")) == 1, (
        f"{table}: Improvement is not beside Momentum — {cols}")


# ── 3. A bar like its neighbours, one shared tooltip ───────────────────────────────────────────
@pytest.mark.parametrize("table", sorted(TABLES))
def test_it_is_a_bar_on_its_neighbours_scale_with_the_shared_tooltip(table):
    cfg = _column_config(table)
    assert "avg_improvement" in cfg, f"{table}: no column_config — it would render as 'avg_improvement'"
    col, ref = cfg["avg_improvement"], cfg["avg_quality"]
    assert ast.unparse(col.func) == "st.column_config.ProgressColumn", f"{table}: not a bar"
    assert isinstance(col.args[0], ast.Constant) and col.args[0].value == "Improvement", (
        f"{table}: headed {ast.unparse(col.args[0])!r}, not 'Improvement' — the header every other "
        f"surface uses for this score")
    for k in ("min_value", "max_value", "format"):
        assert ast.unparse(_kw(col, k)) == ast.unparse(_kw(ref, k)), (
            f"{table}: {k} differs from Quality's — the bars would not be comparable side by side")
    h = _kw(col, "help")
    assert isinstance(h, ast.Name) and h.id == "_MP_IMPROVEMENT_HELP", (
        f"{table}: the tooltip is not the shared definition — two copies drift")


# ── 4. The tooltip IS the glossary definition, plus how to read a group ────────────────────────
def test_the_tooltip_is_the_glossary_definition_plus_the_group_reading():
    """One definition on every surface: the Deep Scanner header and the Reference tab already show
    the glossary entry, and tests/test_improvement_facet.py pins that entry to the shipped design.
    Only the group-level reading is written here."""
    from ui.ui_components import _RAW_GLOSSARY
    text = _run(_assigned("_MP_IMPROVEMENT_HELP").value, {"_RAW_GLOSSARY": _RAW_GLOSSARY})
    assert _RAW_GLOSSARY["Improvement Score"] in text, "the tooltip no longer carries the glossary definition"
    assert "average" in text.lower(), "it never says the number is an average of the row's stocks"
    assert "Momentum" in text and "Count" in text, (
        "the group reading must point to Momentum (the price) and Count (the sample)")


# ── 5. A column, not the ranking ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("table", sorted(TABLES))
def test_it_is_a_column_not_the_ranking(table):
    """Sectors keeps % Qualify → Score and Industry keeps Δ vs Sector → Score. The group evidence is
    strong over years and mixed over PRISM's two short windows; re-ranking a tab on it is a separate
    decision, not a side effect of adding a column."""
    sorts = [n for n in ast.walk(TREE) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "sort_values" and _in_block(n, table)]
    assert sorts, f"{table}: found no sort in the block — the slice is wrong"
    for s in sorts:
        consts = {c.value for c in ast.walk(s) if isinstance(c, ast.Constant)}
        assert "avg_improvement" not in consts, f"{table}: Improvement became a sort key"


# ── 6. The caption that lists the averaged columns names it ────────────────────────────────────
def test_the_sectors_caption_names_every_averaged_column():
    """The caption says which columns are averaged across ALL the sector's stocks. A list on screen
    that omits a column is the caption-drift class this project has shipped before."""
    caps = [n for n in ast.walk(TREE) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "markdown" and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "_sec_cap_ph"]
    assert len(caps) == 1, f"expected ONE Sectors caption, found {len(caps)}"
    text = "".join(v.value for v in caps[0].args[0].values
                   if isinstance(v, ast.Constant) and isinstance(v.value, str))
    cfg = _column_config("sectors")
    shown = [c for c in _displayed("sectors") if c.startswith("avg_")]
    assert len(shown) >= 4, f"the averaged columns vanished: {shown}"
    for c in shown:
        label = cfg[c].args[0].value
        assert label in text, f"the Sectors caption lists the averaged columns but omits {label!r}"


# ── 7. Not a restatement of the columns beside it (live) ───────────────────────────────────────
@pytest.fixture(scope="module")
def live():
    from core import run_scoring_pipeline
    from data_engine import (coerce_numeric_columns, compute_derived_signals, load_all_csvs,
                             merge_datasets)
    with contextlib.redirect_stdout(_io.StringIO()):
        return run_scoring_pipeline(
            compute_derived_signals(coerce_numeric_columns(merge_datasets(load_all_csvs("local")))))


@pytest.mark.parametrize("level,floor", [("sector", 5), ("industry", 3)])
def test_the_group_average_is_not_a_restatement_of_quality_or_momentum(live, level, floor):
    """SELF-VERIFYING: the column earns its place only while it shows something its neighbours do
    not. Measured across the four saved vintages: vs Quality -0.10 … +0.20, vs Momentum +0.19 …
    +0.52, spread (SD) 7.6 … 11.8. The bounds sit well clear of all of it; a fail means the facet or
    the data changed shape and the column should be re-argued, not that the bound needs nudging."""
    g = live.groupby(level).agg(n=("name", "count"), imp=("improvement_score", "mean"),
                                qual=("quality_score", "mean"), mom=("momentum_score", "mean"))
    g = g[g["n"] >= floor]
    assert len(g) >= 30, f"only {len(g)} {level}s with >= {floor} stocks — too few to judge"
    rho = lambda a, b: g[a].rank().corr(g[b].rank())
    assert g["imp"].std() >= 4.0, f"{level} averages have collapsed (SD {g['imp'].std():.1f}) — a flat bar"
    assert abs(rho("imp", "qual")) < 0.5, f"{level}: tracks Quality (rho {rho('imp', 'qual'):+.3f})"
    assert rho("imp", "mom") < 0.8, f"{level}: has become Momentum (rho {rho('imp', 'mom'):+.3f})"
