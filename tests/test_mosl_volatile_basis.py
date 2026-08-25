"""
test_mosl_volatile_basis.py
===========================
Contract for `mosl_volatile_flag`'s MEASUREMENT BASIS — the 27th WCS Volatile label.

THE BUG THIS FILE PINS (fixed 2026-08-25). Leg A read `pat_gr_yoy`, a column whose NAME says
annual but whose VALUES are quarterly. Measured on the raw Income Statement sheet:

    pat_gr_yoy vs (pat_lq - pat_pyq)/pat_pyq   -> median gap  0.00pp, 99.2% agree   <- what it IS
    pat_gr_yoy vs (pat - pat_1yb)/pat_1yb      -> median gap 31.95pp,  1.9% agree   <- what it was READ as

So one soft QUARTER branded a company structurally Volatile — a classification the study defines
over 10-15 years of ANNUAL PAT. On live data that cost 193 stocks a -10 quality penalty for a
quarterly wobble, and put 81 stocks in the impossible state of being certified Consistency Champion
AND Volatile simultaneously, with both cells rendered side by side in the tearsheet.
Triveni Turbine was the clean illustration: annual PAT -3.8%, quarter -20.6% -> "Volatile".

Leg A now reads the SAME annual PAT level series `consistency_champion` uses. That shared basis is
the whole point: the study calls a Volatile "the structural opposite of a Consistent", which is only
meaningful if both labels measure the same quantity.

LEG C is a regression guard, not a feature. Leg A is a ratio, so it is undefined when the prior year
was a loss — and dropping those rows silently un-flagged 18 of the most volatile names in the
universe (IndiGo -4,808cr from -2,394cr; Aequs, loss deepening). A loss year IS a >100% fall: the
evidence exists, it just cannot be written as a ratio, so leg C asserts it directly.

Run with: pytest tests/test_mosl_volatile_basis.py -v
"""

import ast
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import pytest

from data_engine import compute_derived_signals
from test_data_quality_fixes import _frame

_ENGINE = os.path.join(os.path.dirname(__file__), "..", "core", "data_engine.py")

# A healthy 6-point PAT window: steady compounding, no fall anywhere.
_HEALTHY = dict(pat_5yb=60.0, pat_4yb=70.0, pat_3yb=80.0, pat_2yb=90.0, pat_1yb=100.0, pat=110.0)


def _run(n=2, **over):
    """Build a frame whose PAT window is healthy unless a test overrides it."""
    cols = {k: [v] * n for k, v in _HEALTHY.items()}
    cols.update({k: (v if isinstance(v, list) else [v] * n) for k, v in over.items()})
    return compute_derived_signals(_frame(n=n, **cols))


# ── 1. THE BUG: a quarterly wobble must not create a structural Volatile ─────────────
def test_soft_quarter_does_not_make_a_company_volatile():
    """Triveni Turbine's shape: annual PAT barely moved, one quarter fell hard."""
    out = _run(pat=100.0, pat_1yb=103.0, pat_gr_5y=15.0, pat_gr_yoy=-50.0)
    assert out["mosl_volatile_flag"].iloc[0] == 0, (
        "a -50% QUARTER with a -2.9% ANNUAL year was labelled structurally Volatile — "
        "leg A has been re-pointed at the quarterly pat_gr_yoy column"
    )


def test_quarterly_column_cannot_rescue_a_bad_annual_year():
    """Mirror direction: a flattering quarter must not hide a real annual collapse."""
    out = _run(pat=50.0, pat_1yb=100.0, pat_gr_5y=10.0, pat_gr_yoy=80.0)
    assert out["mosl_volatile_flag"].iloc[0] == 1, "a -50% ANNUAL fall must fire leg A"


# ── 2. LEG C: losses stay flagged (the 18-stock regression) ──────────────────────────
def test_prior_year_loss_stays_volatile():
    """IndiGo/Aequs shape: leg A is undefined on a negative base — leg C must catch it."""
    out = _run(pat=10.0, pat_1yb=-50.0, pat_gr_5y=20.0, pat_gr_yoy=5.0)
    assert out["mosl_volatile_flag"].iloc[0] == 1, (
        "a prior-year LOSS left the stock unflagged — an undefined denominator was allowed "
        "to suppress a known >100% fall"
    )


def test_current_year_loss_stays_volatile_even_with_no_prior_figure():
    """John Cockerill shape: current loss, prior year absent -> nothing else can catch it."""
    out = _run(pat=-29.0, pat_1yb=np.nan, pat_gr_5y=np.nan, pat_gr_yoy=np.nan)
    assert out["mosl_volatile_flag"].iloc[0] == 1


def test_zero_prior_pat_does_not_raise_and_is_flagged():
    """Guarded denominator (§5): pat_1yb == 0 must yield NaN, not inf — and leg C fires."""
    out = _run(pat=10.0, pat_1yb=0.0, pat_gr_5y=10.0, pat_gr_yoy=0.0)
    assert out["mosl_volatile_flag"].iloc[0] == 1


# ── 3. Legs B and the healthy baseline still behave ──────────────────────────────────
def test_negative_five_year_cagr_still_fires_leg_b():
    out = _run(pat_gr_5y=-5.0, pat_gr_yoy=10.0)
    assert out["mosl_volatile_flag"].iloc[0] == 1


def test_healthy_compounder_is_not_volatile():
    out = _run(pat_gr_5y=18.0, pat_gr_yoy=12.0)
    assert out["mosl_volatile_flag"].iloc[0] == 0


# ── 4. The two labels are opposites, measured on one series ──────────────────────────
def test_leg_c_can_never_collide_with_consistency_champion():
    """consistency_champion requires all 6 PAT > 0, so a loss row can never hold both labels."""
    out = _run(n=4,
               pat=[110.0, 10.0, -5.0, 110.0],
               pat_1yb=[100.0, -50.0, 100.0, 100.0],
               pat_gr_5y=[15.0, 20.0, 12.0, 15.0],
               pat_gr_yoy=[10.0, 5.0, -3.0, 10.0])
    loss_rows = (out["pat"] <= 0) | (out["pat_1yb"] <= 0)
    assert (out.loc[loss_rows, "mosl_volatile_flag"] == 1).all()
    assert (out.loc[loss_rows, "consistency_champion"] == 0).all(), (
        "a stock with a loss year was certified a Consistency Champion"
    )


def test_healthy_row_is_champion_and_not_volatile():
    out = _run(pat_gr_5y=18.0, pat_gr_yoy=12.0, pat_gr_10y=16.0)
    assert out["mosl_volatile_flag"].iloc[0] == 0
    assert out["consistency_champion"].iloc[0] == 1


# ── 5. STATIC PIN: the quarterly column must not reappear inside the flag ────────────
def _volatile_assignment_node():
    tree = ast.parse(io.open(_ENGINE, encoding="utf-8").read())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if (isinstance(tgt, ast.Subscript) and isinstance(tgt.slice, ast.Constant)
                    and tgt.slice.value == "mosl_volatile_flag"):
                return node.value
    pytest.fail('df["mosl_volatile_flag"] assignment not found in data_engine.py')


def test_volatile_flag_does_not_read_any_quarterly_growth_column():
    """AST, not substring: the prose above legitimately names pat_gr_yoy many times."""
    names = {n.value for n in ast.walk(_volatile_assignment_node())
             if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    banned = names & {"pat_gr_yoy", "eps_gr_yoy", "rev_gr_yoy", "pat_lq", "pat_pyq"}
    assert not banned, (
        f"mosl_volatile_flag reads quarterly-basis column(s) {sorted(banned)} — the 27th WCS "
        f"Volatile label is defined over ANNUAL PAT (see this file's docstring)"
    )


def test_volatile_flag_tracks_the_annual_pat_level_series():
    """Behavioural half of the pin: identical quarters, opposite annual years -> opposite labels."""
    out = _run(n=2, pat=[110.0, 50.0], pat_1yb=[100.0, 100.0],
               pat_gr_5y=[15.0, 15.0], pat_gr_yoy=[0.0, 0.0])
    assert out["mosl_volatile_flag"].tolist() == [0, 1], (
        "identical quarterly inputs, opposite ANNUAL years -> the flag must track the annual series"
    )
