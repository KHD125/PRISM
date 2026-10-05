"""
test_strong_quarter.py
======================
The earnings-season lens (2026-10-05): SOIC's "students of earnings" idea, built from data PRISM has.

ONE DEFINITION — core.scoring_engine.strong_quarter: the latest reported quarter against the SAME
quarter last year, read from the vendor's quarterly LEVELS (rev_lq/rev_pyq, pat_lq/pat_pyq, ebitda_lq/
ebitda_pyq): revenue up more than 15%, profit up more than 20%, EBITDA margin wider. Growth is only
measured off a POSITIVE base (§5): a company that lost money a year ago CANNOT BE JUDGED — NaN, never
"strong" — which is ~12% of stocks. The vendor's own growth % (rev_gr_yoy / pat_gr_yoy) matches the
levels on 97% but is computed even off a loss, which is why the levels are read instead.

TWO SURFACES
  - Stock: the 🔥 Strong Quarter catalyst (cat_strong_quarter; cannot-be-judged = not strong) on every
    catalyst surface — the Discovery filter, the Market Pulse lens rows, the tear-sheet pills, the
    Discovery card and the Reference. DISPLAY-ONLY: no score reads it (December rule 12).
  - Group: "Strong qtr" on 📈 Sectors Core — the share of a sector's judgeable companies that had a
    strong quarter. Ishmohit Arora (SOIC) reads a sector where "most players are reporting strong
    earnings growth" as one starting to do well.

MEASURED BEFORE BUILDING (four snapshots, two independent results seasons):
  - stock: flagged beat the median by +2.4 to +4.7pp in every window, +2.0 to +3.6pp with momentum held
    fixed; fires on 23%. Stronger than the existing 🔥 OpLev Inflection (-0.3 to +2.4pp), which it
    overlaps 89% — kept separate: sales-led growth is not costs lagging revenue.
  - sector breadth: rank-IC with sector returns +0.27 to +0.39, BEYOND Momentum +0.15 to +0.31 — it
    passed the pre-declared bar (>= +0.05 in both seasons). INDUSTRY breadth FAILED it (-0.02 / +0.03 in
    the Jan-Mar season): too noisy in small groups, so it is NOT on 🏭 Industry (December re-tests).

Run with: pytest tests/test_strong_quarter.py -v
"""

import ast
import contextlib
import io as _io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
import pandas as pd
import pytest

from core.scoring_engine import detect_catalysts_and_tsunami, strong_quarter
from ui.ui_group_views import MEASURES, group_measures

APP = (ROOT / "app.py").read_text(encoding="utf-8")

# One row per edge — each row moves exactly ONE thing away from "strong":
#   0 strong · 1 revenue exactly +15% · 2 profit exactly +20% · 3 margin flat · 4 loss a year ago
#   5 no revenue a year ago · 6 EBITDA missing · 7 profit x10 off a POSITIVE base (still strong)
_TOY = pd.DataFrame({
    "rev_lq":     [120.0, 115.0, 130.0, 120.0, 130.0, 50.0, 130.0, 130.0],
    "rev_pyq":    [100.0, 100.0, 100.0, 100.0, 100.0, 0.0, 100.0, 100.0],
    "pat_lq":     [13.0, 15.0, 12.0, 15.0, 10.0, 5.0, 15.0, 10.0],
    "pat_pyq":    [10.0, 10.0, 10.0, 10.0, -5.0, 1.0, 10.0, 1.0],
    "ebitda_lq":  [20.0, 20.0, 25.0, 18.0, 25.0, 8.0, np.nan, 25.0],
    "ebitda_pyq": [15.0, 15.0, 15.0, 15.0, 15.0, 1.0, 15.0, 15.0],
})
_WANT = [1.0, 0.0, 0.0, 0.0, np.nan, np.nan, np.nan, 1.0]


def _dict_literal(src, name):
    node = next(n.value for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    return {k.value: v for k, v in zip(node.keys, node.values)}


# ── 1. The definition ───────────────────────────────────────────────────────────────────────────
def test_each_edge_moves_one_condition():
    got = strong_quarter(_TOY).tolist()
    assert np.allclose(got, _WANT, equal_nan=True), (
        f"{got}: thresholds are STRICT (>15%, >20%, margin wider), and a missing or non-positive base "
        f"cannot be judged — NaN, never strong")


def test_it_reads_the_levels_not_the_vendor_growth_percent():
    tree = ast.parse((ROOT / "core" / "scoring_engine.py").read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "strong_quarter")
    names = {c.value for c in ast.walk(fn) if isinstance(c, ast.Constant) and isinstance(c.value, str)}
    assert {"rev_lq", "rev_pyq", "pat_lq", "pat_pyq", "ebitda_lq", "ebitda_pyq"} <= names
    assert not names & {"rev_gr_yoy", "pat_gr_yoy", "q_pat_yoy"}, (
        "the vendor's growth % is computed even off a loss; the levels carry the base")


# ── 2. The stock surface ────────────────────────────────────────────────────────────────────────
def test_the_catalyst_is_the_definition_and_an_unjudgeable_stock_is_not_strong():
    from core.data_engine import compute_derived_signals
    from tests.test_data_quality_fixes import _frame
    f = _frame(len(_TOY), **{c: _TOY[c].tolist() for c in _TOY.columns})
    with contextlib.redirect_stdout(_io.StringIO()):
        d = compute_derived_signals(f).assign(gate_pass=0, quality_score=50.0)
        out = detect_catalysts_and_tsunami(d)
    assert out["cat_strong_quarter"].tolist() == [1, 0, 0, 0, 0, 0, 0, 1]
    assert (out["catalyst_count"] - out[["cat_capacity", "cat_oplev", "cat_inst_discovery",
                                          "cat_deleveraging", "cat_lynch_dream"]].sum(axis=1)
            == out["cat_strong_quarter"]).all(), "the catalyst count must include it"


def test_no_score_reads_the_catalyst():
    """Display-only until December (rule 12): the only reads allowed are its own assignment and the
    catalyst count — which nothing scores (it feeds a console line)."""
    for mod in ("scoring_engine.py", "forensic_engine.py", "verdict_engine.py", "data_engine.py"):
        src = (ROOT / "core" / mod).read_text(encoding="utf-8")
        uses = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Constant) and n.value == "cat_strong_quarter"]
        allowed = 2 if mod == "scoring_engine.py" else 0
        assert len(uses) == allowed, f"{mod} reads cat_strong_quarter {len(uses)}x — only the assignment and the count may"
    sc = (ROOT / "core" / "scoring_engine.py").read_text(encoding="utf-8")
    reads = [n for n in ast.walk(ast.parse(sc)) if isinstance(n, ast.Subscript)
             and isinstance(n.slice, ast.Constant) and n.slice.value == "catalyst_count" and isinstance(n.ctx, ast.Load)]
    assert len(reads) == 1, "something new reads catalyst_count — check it is not a score before shipping"


def test_it_is_on_every_catalyst_surface():
    import io
    disc = io.open(ROOT / "ui" / "ui_discovery.py", encoding="utf-8").read()
    tear = io.open(ROOT / "ui" / "ui_tearsheet.py", encoding="utf-8").read()
    comp = io.open(ROOT / "ui" / "ui_components.py", encoding="utf-8").read()
    label = "🔥 Strong Quarter"
    assert ast.literal_eval(_dict_literal(disc, "_CATALYSTS")[label]) == "cat_strong_quarter"
    assert ast.literal_eval(_dict_literal(APP, "_MP_CATALYSTS")[label]) == "cat_strong_quarter"
    pills = next(n.value for n in ast.walk(ast.parse(tear)) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "_CAT_PILLS" for t in n.targets))
    assert any(e.elts[0].value == "cat_strong_quarter" and e.elts[2].value == label for e in pills.elts)
    assert '"cat_strong_quarter"' in comp and label in comp, "the Discovery card shows no Strong Quarter pill"
    from ui.ui_reference_data import CONCEPT_REFERENCE
    entry = dict(CONCEPT_REFERENCE["🔥 Catalysts"]).get(label, "")
    assert "same quarter last year" in entry and "15%" in entry and "20%" in entry and "loss" in entry, (
        "the Reference must state the rule, its base and that a loss a year ago cannot be judged")


# ── 3. The group surface ────────────────────────────────────────────────────────────────────────
def test_the_sector_share_counts_only_companies_that_can_be_judged():
    toy = pd.concat([_TOY.iloc[[0, 1, 4, 6]]]).assign(sector="A")      # strong, not, loss base, missing
    assert group_measures(toy, "sector").loc["A", "grp_strong_quarter"] == 50.0, (
        "1 of the 2 judgeable companies — 25 would count the unjudgeable as 'not strong'")
    m = MEASURES["grp_strong_quarter"]
    assert m["stat"] == "share" and m["bar"] and "most players" in m["help"]


def test_sectors_shows_it_and_industry_does_not_yet():
    def order(var):
        node = next(n.value for n in ast.walk(ast.parse(APP)) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == var for t in n.targets))
        return [c.value for c in ast.walk(node) if isinstance(c, ast.Constant) and isinstance(c.value, str)]
    sec, ind = order("_sec_order"), order("_ind_order")
    assert "grp_strong_quarter" in sec and sec.index("grp_strong_quarter") < sec.index("avg_improvement"), (
        "Sectors Core shows it, before Improvement — Improvement stays beside Momentum")
    assert "grp_strong_quarter" not in ind, (
        "industry breadth failed the pre-declared bar (-0.02 / +0.03 beyond Momentum in Jan-Mar); "
        "December re-tests it — do not add it before that")


# ── 4. December's instrument ────────────────────────────────────────────────────────────────────
def test_validate_registers_the_strong_quarter_cohort():
    if not (ROOT / "tools" / "validate.py").exists():
        pytest.skip("tools/ is gitignored; the harness is local-only")
    with contextlib.redirect_stdout(_io.StringIO()):
        import validate
    sets = dict(validate._registered_cohorts(_TOY.copy()))
    title = next((t for t in sets if t.lower().startswith("strong quarter")), None)
    assert title, "no Strong Quarter cohort registered in tools/validate.py"
    masks = dict(sets[title])
    strong = next(m for k, m in masks.items() if k.lower().startswith("strong"))
    assert strong.tolist() == [True, False, False, False, False, False, False, True]


# ── 5. The data the definition stands on ────────────────────────────────────────────────────────
_DATA = ROOT / "Other Resources" / "CSV Data"


@pytest.mark.skipif(not _DATA.is_dir(), reason="local data not present (code-only checkout)")
def test_the_levels_agree_with_the_vendor_growth_and_the_flag_is_alive():
    from data_engine import coerce_numeric_columns, load_all_csvs, merge_datasets
    with contextlib.redirect_stdout(_io.StringIO()):
        live = coerce_numeric_columns(merge_datasets(load_all_csvs("local")))
    n = lambda c: pd.to_numeric(live[c], errors="coerce")
    pg = np.where(n("pat_pyq") > 0, (n("pat_lq") / n("pat_pyq") - 1) * 100, np.nan)
    k = ~np.isnan(pg) & n("pat_gr_yoy").notna()
    agree = (np.abs(pg[k] - n("pat_gr_yoy")[k]) <= np.maximum(0.6, 0.01 * np.abs(pg[k]))).mean()
    assert agree >= 0.90, f"profit levels match the vendor's growth on only {agree:.0%} — the basis moved"
    s = strong_quarter(live)
    assert 0.70 <= s.notna().mean() and 0.08 <= (s == 1).mean() <= 0.45, (
        f"judgeable {s.notna().mean():.0%}, strong {(s == 1).mean():.0%} — dead, saturated or unjudgeable")
