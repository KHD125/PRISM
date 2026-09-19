"""
test_fisher_pricing_sentinel.py
===============================
Contract: the Fisher Scalability PRICING sub-gate may only be credited on a REAL margin
observation. Found 2026-09-19 while auditing the margin rebasing.

THE DEFECT. `_fs_pricing = (opm_acc.fillna(0) >= 0) | (npm_acc.fillna(0) >= 0)`. A NaN becomes 0,
and 0 >= 0 is True — so a MISSING margin PASSES the gate. The spec ledger even documents it:
"missing OPM accel -> OR gate passes (neutral; npm guards this)". That guard is fictional: the two
accelerations come from the same vendor row (`*_pyq`) and go missing TOGETHER. Measured on the
2026-09-18 vintage:

    both accelerations NaN AND credited the pricing point:   87 rows on the old basis
                                                             187 rows on the rebased basis

So the rebasing WIDENED a pre-existing hole. fisher_pass itself was never reached this way (the
other three gates filtered every one of them) but fisher_score — displayed as "Fisher Scal. Score
N/4" in All Data — was inflated by +1 on 187 rows for a margin nobody observed. CLAUDE.md §5,
"unverifiable is not passed": require the inputs, or return not-certified. A gate that skips a
criterion it cannot evaluate fabricates a GOOD verdict, which is the harder-to-spot mirror of the
sentinel bug.

THE RULE: a NaN never contributes a pass. `(opm_acc >= 0) | (npm_acc >= 0)` with NO fillna — pandas
comparisons against NaN are False — passes if ANY real observation is non-declining, and returns
not-certified when there is nothing to observe. This also closes the one-NaN case: previously a
row with opm NaN and npm at -8pp passed via the sentinel; the single real observation said
declining.

Run with: pytest tests/test_fisher_pricing_sentinel.py -v
"""
from __future__ import annotations

import contextlib
import io as _io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def live():
    from core import fetch_and_clean_data, run_scoring_pipeline
    with contextlib.redirect_stdout(_io.StringIO()):
        return run_scoring_pipeline(fetch_and_clean_data("local").copy())


def _num(df, c):
    return pd.to_numeric(df[c], errors="coerce")


def _other_three(live):
    """The three Fisher Scalability sub-gates that are NOT under test, mirrored from the engine's
    own expressions (pinned in docs/fisher_quality_specs.json). Isolating them lets the test read
    the engine's pricing contribution as fisher_score minus these."""
    rev3, revy = _num(live, "rev_gr_3y").fillna(0), _num(live, "rev_gr_yoy").fillna(0)
    oplev = _num(live, "d05_rev_minus_exp_gr").fillna(0)
    dil = _num(live, "dilution_pct").fillna(999)
    return ((rev3 >= 12.0) & (revy >= 10.0)).astype(int) + (oplev >= 2.0).astype(int) + (dil <= 1.0).astype(int)


def test_a_missing_margin_never_earns_the_pricing_point(live):
    """THE PIN. With both accelerations NaN there is nothing to certify."""
    opm, npm = _num(live, "opm_acceleration"), _num(live, "npm_acceleration")
    both_nan = opm.isna() & npm.isna()
    assert both_nan.sum() > 50, f"only {int(both_nan.sum())} both-NaN rows — the case is not exercised"
    credited = _num(live, "fisher_score") - _other_three(live)
    bad = both_nan & (credited >= 1)
    assert not bad.any(), (
        f"{int(bad.sum())} rows carry the Fisher pricing-power point with BOTH margin accelerations "
        f"missing. fillna(0) >= 0 is certifying pricing power on a margin nobody observed — the "
        f"'npm guards this' rationale in the spec is false because both go NaN together."
    )


def test_the_pricing_point_matches_the_no_fillna_rule_on_every_row(live):
    """Stronger than the sentinel case: the engine's pricing contribution must equal the rule
    'any REAL observation non-declining' on ALL rows — which also fixes one-NaN-one-negative."""
    opm, npm = _num(live, "opm_acceleration"), _num(live, "npm_acceleration")
    expected = ((opm >= 0) | (npm >= 0)).astype(int)          # NaN compares False: never a pass
    credited = (_num(live, "fisher_score") - _other_three(live)).astype(int)
    mismatch = credited != expected
    assert not mismatch.any(), (
        f"pricing sub-gate disagrees with the no-sentinel rule on {int(mismatch.sum())} rows "
        f"(engine credited {int((mismatch & (credited==1)).sum())} that should not be, withheld "
        f"{int((mismatch & (credited==0)).sum())} that should be)."
    )


def test_one_real_declining_observation_is_not_rescued_by_a_missing_one(live):
    """The one-NaN case, stated on its own so the reason survives: opm NaN + npm at -8pp used to
    pass via the sentinel although the only real observation said declining."""
    opm, npm = _num(live, "opm_acceleration"), _num(live, "npm_acceleration")
    one_nan_neg = (opm.isna() & (npm < 0)) | (npm.isna() & (opm < 0))
    if one_nan_neg.sum() == 0:
        pytest.skip("no one-NaN-one-negative rows on this vintage")
    credited = _num(live, "fisher_score") - _other_three(live)
    assert (credited[one_nan_neg] == 0).all(), (
        f"{int((credited[one_nan_neg] >= 1).sum())} rows passed pricing power on a MISSING margin "
        f"while their only real margin observation was declining"
    )
