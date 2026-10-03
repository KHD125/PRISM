"""Contract: Piotroski F8 compares like with like — this year's full-year margin vs last year's.

THE DEFECT (fixed 2026-10-03). `f_margin_improving` computed `opm_latest_q > opm_1yb`: the latest
QUARTER against the full-year figure one year back. That is the quarter-vs-annual basis mismatch
removed from the `*_acceleration` columns on 2026-09-18 — a seasonal quarter reads as improvement
when nothing changed. The tearsheet also labelled it "gross margin up" while the code compared
OPERATING margin.

THE FIX IS `opm > opm_1yb`, FULL YEAR vs FULL YEAR, and the like-for-like QUARTERLY form
(`opm_latest_q > opm_pyq`) was rejected on two measured grounds:
  1. Every other F-Score leg (F3 ROA, F5 leverage, F6 liquidity, F9 efficiency) compares a full-year
     figure with its 1-year-back twin, and Piotroski's own F8 is annual. One quarterly leg inside an
     annual score compares different periods within one number.
  2. `opm_pyq` exists only from the 2026-09-18 vintage. 🔁 Movers re-scores ARCHIVED vintages with
     this engine, so a quarterly F8 would read 0 for every stock on any older copy and manufacture a
     market-wide F-Score drop. `opm` / `opm_1yb` exist in every snapshot. Coverage 97.9% vs 93.1%.

BOOK DEVIATION, stated: Piotroski's F8 is GROSS margin. No annual gross-margin column exists (gross
margin arrives quarterly only, 83% coverage), so the nearest annual margin, OPM, stands in — the same
kind of proxy F9 already uses. The label says OPERATING margin so the screen does not claim otherwise.

MEASURED before shipping (bar pre-declared: ship unless the F-Score's forward rank-IC falls by more
than 0.02 in BOTH independent windows): F-Score IC old → new +0.1464 → +0.1430 and +0.0795 → +0.0729
(Δ −0.003 / −0.007), well inside the bar. The old form's small edge is the FRESHER quarter, not a
correct comparison. F8 agrees with the old form on 81.2% of stocks; the full-year form agrees with
the seasonally-clean quarterly form on 88.8% vs the old form's 77.3%. `piotroski_fscore` is not read
by any scoring, gate or verdict code (display, sidebar filter and scanner only), so no composite moves.
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd

import ui.ui_tearsheet as T
from core.forensic_engine import compute_piotroski_fscore

_ROOT = Path(__file__).resolve().parent.parent


def _f8(**cols):
    return compute_piotroski_fscore(pd.DataFrame(cols))["f_margin_improving"].tolist()


def test_a_seasonal_quarter_no_longer_reads_as_improvement():
    # quarter spikes above last year's annual margin, but the full year FELL -> no credit
    assert _f8(opm=[14.0], opm_1yb=[15.0], opm_latest_q=[19.0]) == [0]


def test_a_full_year_improvement_counts_even_after_a_weak_quarter():
    # full year ROSE, latest quarter dipped below last year's annual -> credit
    assert _f8(opm=[17.0], opm_1yb=[15.0], opm_latest_q=[12.0]) == [1]


def test_missing_either_year_is_not_credited():
    # unverifiable is not passed (CLAUDE.md §5): a missing year earns no point
    assert _f8(opm=[np.nan, 17.0], opm_1yb=[15.0, np.nan]) == [0, 0]


def test_older_vintages_without_the_quarterly_columns_still_score():
    # an archived copy carries no opm_latest_q / opm_pyq; F8 must still be evaluated, not zeroed
    assert _f8(opm=[17.0, 13.0], opm_1yb=[15.0, 15.0]) == [1, 0]


def _f8_inputs():
    """Every column the F8 assignment reads, traced through the local aliases (`_opm = df.get(...)`)."""
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
           and getattr(n.targets[0].slice, "value", None) == "f_margin_improving"]
    assert len(rhs) == 1
    return {alias[n.id] for n in ast.walk(rhs[0]) if isinstance(n, ast.Name) and n.id in alias}


def test_f8_reads_exactly_the_two_full_year_margins():
    assert _f8_inputs() == {"opm", "opm_1yb"}


def test_the_checklist_label_says_what_the_code_compares():
    row = next(r for r in T._PIOTROSKI_9 if r[0] == "f_margin_improving")
    text = " ".join(row[1:]).lower()
    assert "gross" not in row[2].lower(), row          # the short criterion must not claim gross margin
    assert "operating" in text or "op." in text, row
    assert "full-year" in text or "full year" in text or "annual" in text, row
