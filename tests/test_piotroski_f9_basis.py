"""Contract: Piotroski F9 measures ASSET TURNOVER, the thing its label says.

THE DEFECT (fixed 2026-10-03). `f_efficiency_improving` computed `roce > roce_1yb` as a "proxy",
while the tearsheet checklist labelled it "revenue/assets up" — a claim about a number the code never
read. The proxy was unnecessary: `asset_turnover` / `asset_turnover_1yb` are mapped at 92.1% joint
coverage (93.2% / 95.2% on the two older snapshots, so archived vintages score too). Piotroski's F9
IS the change in asset turnover, so this is book fidelity, not a deviation.

THE PAIR WAS VERIFIED BEFORE IT WAS TRUSTED (names lie in this data):
  * asset_turnover IS revenue / total_assets — but for the PREVIOUS fiscal year: on the 2026-09-19
    snapshot it equals the June snapshot's revenue/total_assets on 95.7% of rows, and
    asset_turnover_1yb equals June's revenue_1yb/total_assets_1yb on 91.6%. So the pair is one
    construction, one year apart — a valid like-for-like delta (CLAUDE.md §5 cross-year basis rule).
  * WHY the lag: the vendor rolled the P&L to the new fiscal year (revenue / pat / opm rolled on
    81-89% of rows between June and September) but NOT the balance sheet (total_assets, roa, roe,
    reserves unchanged on 81-94%). So F9 now compares the same two years as F3 (ROA improving).
  * Recomputing revenue / total_assets in the engine was REJECTED: it would divide this year's
    revenue by last year's assets — the mixed-period error being removed.

MEASURED, bar pre-declared before looking (ship unless the F-Score's forward rank-IC falls by more
than 0.02 in BOTH independent windows): +0.1464 -> +0.1341 and +0.0795 -> +0.0710 (Δ −0.012 / −0.009).
The cost is stated, not hidden: F9 ALONE fell +0.096 -> +0.025, because the old ROCE leg moved with
F3 at +0.71 — it was counting profit improvement twice, and short-term profit improvement is what
66-day and 28-day windows reward. The turnover leg correlates +0.34 with F3: a genuinely separate
check, as Piotroski designed it. `piotroski_fscore` is display-only (no scoring, gate or verdict code
reads it), so no composite moves.
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd

import ui.ui_tearsheet as T
from core.forensic_engine import compute_piotroski_fscore

_ROOT = Path(__file__).resolve().parent.parent


def _f9(**cols):
    return compute_piotroski_fscore(pd.DataFrame(cols))["f_efficiency_improving"].tolist()


def test_rising_returns_on_a_shrinking_turnover_is_not_credited():
    # ROCE up, but revenue per rupee of assets FELL -> no point (the old proxy gave one)
    assert _f9(roce=[25.0], roce_1yb=[20.0], asset_turnover=[0.9], asset_turnover_1yb=[1.1]) == [0]


def test_rising_turnover_counts_even_when_roce_fell():
    assert _f9(roce=[15.0], roce_1yb=[20.0], asset_turnover=[1.3], asset_turnover_1yb=[1.1]) == [1]


def test_missing_either_year_is_not_credited():
    assert _f9(asset_turnover=[np.nan, 1.3], asset_turnover_1yb=[1.1, np.nan]) == [0, 0]


def test_an_unchanged_turnover_is_not_improvement():
    assert _f9(asset_turnover=[1.1], asset_turnover_1yb=[1.1]) == [0]


def test_older_vintages_without_roce_still_score():
    assert _f9(asset_turnover=[1.3, 0.8], asset_turnover_1yb=[1.1, 1.1]) == [1, 0]


def _f9_inputs():
    """Every column the F9 assignment reads, traced through the local `_x = df.get(...)` aliases."""
    src = (_ROOT / "core" / "forensic_engine.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "compute_piotroski_fscore")
    alias = {}
    for n in ast.walk(fn):
        if (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                and isinstance(n.value, ast.Call) and getattr(n.value.func, "attr", "") == "get"
                and n.value.args and isinstance(n.value.args[0], ast.Constant)):
            alias[n.targets[0].id] = n.value.args[0].value
    rhs = [n.value for n in ast.walk(fn)
           if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Subscript)
           and getattr(n.targets[0].slice, "value", None) == "f_efficiency_improving"]
    assert len(rhs) == 1
    return {alias[n.id] for n in ast.walk(rhs[0]) if isinstance(n, ast.Name) and n.id in alias}


def test_f9_reads_exactly_the_vendor_turnover_pair():
    assert _f9_inputs() == {"asset_turnover", "asset_turnover_1yb"}


def test_the_checklist_label_matches_what_the_code_reads():
    row = next(r for r in T._PIOTROSKI_9 if r[0] == "f_efficiency_improving")
    assert "revenue/assets" in row[2], row
    assert "roce" not in " ".join(row[1:]).lower(), row
