"""
test_moat_tau_quantization.py
=============================
Contract: `moat_tau` is COARSELY QUANTIZED, and the wealth tier's C/FADE gates depend on it.

WHY THIS FILE EXISTS (2026-09-11 label-threshold audit). `wealth_tier` leg C is written as
`tau >= WEALTH_TAU_CONF` with WEALTH_TAU_CONF = 0.25. But 0.25 is NOT AN ATTAINABLE VALUE.
moat_tau is a pairwise-sign Kendall tau over FOUR ordered OPM level columns, so a row that gets a
tau at all has exactly 6 comparable pairs and the result is net/6 -- thirteen values, the multiples
of 1/6 from -1 to +1. The nearest attainable values to the 0.25 cut are 1/6 and 1/3.

So the gate that READS 0.25 actually MEANS `tau >= 1/3`, and the one that reads -0.25 means
`tau <= -1/3`. Nothing in the suite knew that.

THE FAILURE THIS GUARDS. The threshold's meaning is a function of the attainable value set, and
that set is a function of HOW MANY OPM LEVEL COLUMNS EXIST. Add one -- if the vendor ever ships an
opm_med_10y or an opm_2yb -- and the ladder grows to 5 columns, 10 pairs, a step of 1/10, and
`>= 0.25` silently starts meaning `>= 0.3` instead of `>= 1/3`. A different gate, with no failing
test, no error, and no visible symptom except a churn spike somebody spends a day chasing.

WHAT THIS FILE DOES NOT CLAIM. The coarseness is NOT a defect and must not be "fixed": moat_tau
already consumes every OPM LEVEL column that exists (opm_acceleration / opm_stability / opm_stable
are derived signals, not points on a time ladder, and belong nowhere near a trajectory tau). Its
siblings roce_tau / revenue_tau / pat_tau reach 41 / 33 / 32 distinct values because ROCE, revenue
and PAT history is deeper in the source -- not because they are built better.

MEASURED CONSEQUENCE, recorded so the next reader does not re-derive it: 692 stocks (25.5%) sit ONE
QUANTUM from flipping C or FADE -- 299 at exactly +1/3 (the minimum value that passes C) and 290 at
exactly -1/3 (the minimum that triggers FADE). In the audit's one-quarter simulation a single
adverse quantum moved 279 wealth tiers, 12.1% of comparable stocks: the largest single source of
label churn found, and larger than the EP and velocity legs combined.

Run with: pytest tests/test_moat_tau_quantization.py -v
"""

import contextlib
import io as _io
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

import numpy as np
import pandas as pd
import pytest

_ENGINE = os.path.join(os.path.dirname(__file__), "..", "core", "data_engine.py")

# The ordered OPM level ladder moat_tau is built from. Changing this list changes the step size,
# and therefore changes what WEALTH_TAU_CONF means -- which is the whole point of this file.
TAU_COLS = ["opm_med_5y", "opm_1yb", "opm", "opm_latest_q"]


@pytest.fixture(scope="module")
def live():
    from core import fetch_and_clean_data
    with contextlib.redirect_stdout(_io.StringIO()):
        return fetch_and_clean_data("local")


# -- 1. The ladder, pinned at the source ------------------------------------------------
def test_moat_tau_is_built_from_exactly_these_four_level_columns():
    """The step size is 1/pairs, and pairs is a function of THIS list. If it grows, every
    assertion below changes meaning -- so the list itself is the contract."""
    src = _io.open(_ENGINE, encoding="utf-8").read()
    i = src.index('df["moat_tau"] = _kendall_tau_cols(')
    call = src[i:src.index(")", src.index("[", i))]
    for c in TAU_COLS:
        assert f'"{c}"' in call, f"{c} is no longer in the moat_tau ladder: {call}"
    assert call.count('"opm') == len(TAU_COLS), (
        f"the moat_tau ladder is no longer {len(TAU_COLS)} columns. The quantum is 1/pairs, so a "
        f"5th column changes the step from 1/6 to 1/10 and silently redefines WEALTH_TAU_CONF. "
        f"Re-derive every number in this file before changing it.\n{call}"
    )


# -- 2. The attainable value set ---------------------------------------------------------
def test_tau_takes_exactly_the_thirteen_sixths(live):
    """4 columns -> 6 pairs; a row needs MORE than half its pairs comparable (min_pairs = 4), so a
    row with only 3 columns present yields 3 pairs and is NaN. Every row that gets a value therefore
    has all 6 pairs, and tau is net/6."""
    t = pd.to_numeric(live["moat_tau"], errors="coerce").dropna()
    vals = sorted(t.unique())
    assert len(vals) == 13, f"moat_tau has {len(vals)} distinct values, expected 13: {vals}"
    for v in vals:
        f = Fraction(v).limit_denominator(20)
        assert f.denominator in (1, 2, 3, 6), f"{v} is not a multiple of 1/6 (got {f})"
    assert np.isclose(min(np.diff(vals)), 1 / 6), "the quantum is no longer 1/6"


def test_the_quantum_is_larger_than_the_distance_to_the_gate(live):
    """The fragility, stated as a number: one step is 0.167 while the gap from 1/6 to the 0.25 cut
    is only 0.083. A single OPM observation changing can therefore cross the gate."""
    t = pd.to_numeric(live["moat_tau"], errors="coerce").dropna()
    step = float(min(np.diff(sorted(t.unique()))))
    from verdict_engine import WEALTH_TAU_CONF
    below = max(v for v in t.unique() if v < WEALTH_TAU_CONF)
    assert step > (WEALTH_TAU_CONF - below), (
        "one quantum no longer exceeds the distance from the nearest value below the gate; the "
        "fragility this file documents has changed character -- re-run the audit"
    )


# -- 3. What the gate actually means ------------------------------------------------------
def test_the_0_25_gate_is_exactly_the_one_third_gate(live):
    """WEALTH_TAU_CONF reads 0.25 but 0.25 is unreachable, so the gate IS `tau >= 1/3`. Anyone
    reasoning about 'a quarter' is reasoning about a number no stock can hold."""
    from verdict_engine import WEALTH_TAU_CONF, WEALTH_TAU_FADE
    t = pd.to_numeric(live["moat_tau"], errors="coerce")
    # NOTE: compare against the FLOAT 1/3, never Fraction(1, 3). The exact rational is strictly
    # greater than the double moat_tau actually holds, so a Fraction comparison silently excludes
    # every stock sitting on the boundary -- which is exactly the population this file is about.
    assert ((t >= WEALTH_TAU_CONF) == (t >= 1 / 3)).all()
    assert ((t <= WEALTH_TAU_FADE) == (t <= -1 / 3)).all()


def test_no_stock_sits_marginally_across_either_cut(live):
    """The one GOOD consequence of the quantization, pinned so it is not 'fixed' away: both cuts
    sit in an empty gap, so no stock is ever a hair from the line."""
    from verdict_engine import WEALTH_TAU_CONF, WEALTH_TAU_FADE
    t = pd.to_numeric(live["moat_tau"], errors="coerce").dropna()
    for cut in (WEALTH_TAU_CONF, WEALTH_TAU_FADE):
        assert ((t - cut).abs() <= 0.05).sum() == 0, (
            f"stocks now sit within 0.05 of the {cut} cut; the gate has moved out of its gap"
        )


# -- 4. The exposure, so December has a baseline -----------------------------------------
def test_the_one_quantum_population_is_still_a_quarter_of_the_universe(live):
    """299 at +1/3 (minimum passing C) and 290 at -1/3 (minimum FADE), plus the 1/6 rows one step
    below each gate: 692 stocks, 25.5%. This is the audit's headline exposure figure; if it moves a
    lot, the churn picture has changed and the analysis needs re-running rather than re-citing."""
    t = pd.to_numeric(live["moat_tau"], errors="coerce")
    # isclose, not isin: these are computed doubles, and exact-rational membership never matches.
    tv = t.to_numpy(dtype=float)
    one_quantum = np.zeros(len(tv), dtype=bool)
    for q in (1 / 3, -1 / 3, 1 / 6, -1 / 6):
        one_quantum |= np.isclose(tv, q, rtol=0, atol=1e-9)
    share = one_quantum.sum() / t.notna().sum()
    assert 0.15 < share < 0.40, (
        f"{share:.1%} of rated stocks are one quantum from flipping C or FADE (audit measured "
        f"25.5%). A large move means the OPM distribution shifted -- re-measure before trusting "
        f"the churn analysis in docs/known-issues.md"
    )


def test_tau_is_not_dead_and_does_not_pile_into_one_value(live):
    """Liveness: a trajectory statistic stuck at 0 would make leg C a constant."""
    t = pd.to_numeric(live["moat_tau"], errors="coerce")
    assert t.isna().mean() < 0.10, f"moat_tau is {t.isna().mean():.1%} NaN"
    top = t.value_counts(normalize=True).iloc[0]
    assert top < 0.40, f"a single tau value holds {top:.1%} of the universe"
