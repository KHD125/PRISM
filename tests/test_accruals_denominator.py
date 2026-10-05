"""KI-8: the accruals red flag halved its denominator when one year's total assets were missing.

forensic_engine's rf_high_accruals divides (PAT - OCF) by the AVERAGE of this year's and last year's
total assets. When this year's assets were blank it averaged last year's with 0, so the denominator
halved and the accrual ratio doubled; the reverse case (last year blank) already fell back to this
year IN FULL. Measured on the local 2026-09-18 vintage (2026-10-05): 128 stocks have last year's
assets but not this year's, and 12 carried an accrual red flag only because of the halving, each one
feeding red_flag_count, the forensic penalty multiplier and the composite. A recorded asset figure of
0 did the same (two rows carry it last year): no listed company has zero assets, so 0 is a missing
figure, not a measurement.

THE RULE, one for both years and both surfaces: use the years that exist (and are above 0), averaging
only when both do. The tear-sheet's evidence line follows it, so the screen shows the number the flag
was computed from (it printed nothing for exactly these stocks); the Market Pulse share mirrors it.
"""
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

import numpy as np
import pandas as pd
import pytest

from core.data_engine import compute_derived_signals
from core.forensic_engine import compute_forensic_signals

NAN = np.nan
# (name, total_assets, total_assets_1yb, pat, operating_cash_flow, flag the rule gives, % shown)
CASES = [
    ("this year blank",          NAN,    1000.0, 140.0, 100.0, 0, "4.0%"),   # 40 / 1000 (was 40 / 500)
    ("last year blank",          1000.0, NAN,    140.0, 100.0, 0, "4.0%"),   # already fell back in full
    ("both years",               1000.0, 1000.0, 160.0, 100.0, 1, "6.0%"),   # real accruals still flag
    ("this year blank, real",    NAN,    1000.0, 170.0, 100.0, 1, "7.0%"),   # the fallback still flags
    ("last year recorded as 0",  1000.0, 0.0,    140.0, 100.0, 0, "4.0%"),   # 0 is missing (was 40 / 500)
    ("both blank",               NAN,    NAN,    140.0, 100.0, 0, ""),       # cannot judge
]


@pytest.fixture(scope="module")
def scored():
    from tests.test_data_quality_fixes import _frame
    f = _frame(len(CASES), total_assets=[c[1] for c in CASES], total_assets_1yb=[c[2] for c in CASES],
               pat=[c[3] for c in CASES], operating_cash_flow=[c[4] for c in CASES])
    f["name"] = [c[0] for c in CASES]
    with contextlib.redirect_stdout(io.StringIO()):
        return compute_forensic_signals(compute_derived_signals(f)).set_index("name")


def test_a_missing_year_of_assets_falls_back_in_full_never_to_half(scored):
    wrong = [f"{c[0]}: flag {int(scored.loc[c[0], 'rf_high_accruals'])}, should be {c[5]}"
             for c in CASES if int(scored.loc[c[0], "rf_high_accruals"]) != c[5]]
    assert not wrong, "accruals judged against the wrong denominator: " + "; ".join(wrong)


def test_the_tear_sheet_shows_the_number_the_flag_was_computed_from(scored):
    from ui.ui_tearsheet import _get_flag_context
    wrong = []
    for c in CASES:
        text = _get_flag_context(scored.loc[c[0]], "rf_high_accruals")
        if c[6] and f"accruals: {c[6]} of assets" not in text:
            wrong.append(f"{c[0]}: shows {text!r}, expected accruals {c[6]}")
        if not c[6] and text:
            wrong.append(f"{c[0]}: shows {text!r} with no assets on record")
    assert not wrong, "; ".join(wrong)


def test_the_market_pulse_share_mirrors_the_same_rule(scored):
    from ui.ui_group_views import group_measures
    with contextlib.redirect_stdout(io.StringIO()):
        g = group_measures(scored.reset_index(), "sector")
    # judged: every row with assets in either year (five of six); flagged: the two real accruals
    assert g["grp_high_accruals"].iloc[0] == pytest.approx(40.0)
