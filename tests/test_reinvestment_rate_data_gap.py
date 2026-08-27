"""
test_reinvestment_rate_data_gap.py
==================================
Keeps the reinvestment_rate DATA-GAP comment (core/data_engine.py, above the RR assignment)
honest against live data.

WHY THIS FILE EXISTS. The comment it guards is the SECOND documentation-truth failure of the same
shape found in one day:

  * cyclicality_tier was documented "INERT -- never touches a score". It reaches composite_score
    through the fair_pe_qglp cap (fixed in cbb0ddc).
  * the RR note said the DPR column was "96% empty", RR was "1.0 universe-wide", and that
    capital_misallocation_risk / flag_epoch2_compounder were "INERT/always-pass". The DPR column
    was repaired in part on 2026-08-22 and all four claims went false. Nothing failed.

Both notes were dangerous in the same specific way: they did not merely record a stale number,
they told a reader the signal was inert and therefore safe to ignore -- while it was live and
moving scores. A wrong fact is a nuisance; a wrong *all-clear* stops the next person looking.

HOW IT WORKS, and why it reads the comment instead of hardcoding. The expected rates are PARSED
OUT OF THE COMMENT ITSELF, then checked against a live pipeline run. So:

  * the comment is the single source -- test and prose cannot drift apart, because there is only
    one copy of each number;
  * updating the comment updates the contract, which is the behaviour you want from a note whose
    whole job is to be current;
  * when the DPR repair continues and the rates move, THIS FAILS and names the line to rewrite,
    instead of the note rotting silently for another two months.

TOLERANCE. +/- 8 percentage points. Wide enough that an ordinary data refresh does not cry wolf,
far tighter than the errors it exists to catch (the old note's "passes for ALL" was ~57 points
off the measured 42.5%).

Run with: pytest tests/test_reinvestment_rate_data_gap.py -v
"""

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

_ENGINE = os.path.join(os.path.dirname(__file__), "..", "core", "data_engine.py")
TOL_PP = 8.0          # percentage points
_WHERE = "the RR data-gap comment in core/data_engine.py (above df['reinvestment_rate'])"


@pytest.fixture(scope="module")
def live():
    from core import run_scoring_pipeline
    with contextlib.redirect_stdout(_io.StringIO()):
        return run_scoring_pipeline(
            compute_derived_signals(coerce_numeric_columns(merge_datasets(load_all_csvs("local")))))


@pytest.fixture(scope="module")
def note():
    """The data-gap comment block, sliced from the engine source."""
    src = _io.open(_ENGINE, encoding="utf-8").read()
    start = src.index("PARTIAL DATA GAP")
    end = src.index('df["reinvestment_rate"]', start)
    block = src[start:end]
    assert len(block) > 500, "the data-gap comment shrank unexpectedly -- was it deleted?"
    return block


def _num(note, pattern, label):
    m = re.search(pattern, note)
    assert m, (
        f"could not find the documented {label} in {_WHERE}. The test parses its expectations out "
        f"of that comment; if you reworded it, keep the number in a form this pattern matches."
    )
    return float(m.group(1))


def _close(actual, documented, label):
    assert abs(actual - documented) <= TOL_PP, (
        f"{label}: the comment documents {documented:.1f}% but live data measures {actual:.1f}% "
        f"({abs(actual - documented):.1f}pp apart, tolerance {TOL_PP:.0f}pp).\n\n"
        f"THE COMMENT IS NOW STALE. Remeasure and rewrite {_WHERE} -- that note is the reason a "
        f"future reader will or will not investigate these gates, and this is exactly the drift "
        f"that made its predecessor claim four things that were all false."
    )


# -- 1. The four numbers the comment states, checked against live data -------------------
def test_documented_dpr_coverage_is_current(live, note):
    documented = _num(note, r"DPR is ([\d.]+)% populated", "DPR coverage")
    _close(live["dividend_payout_ratio"].notna().mean() * 100, documented, "DPR coverage")


def test_documented_rr_sentinel_share_is_current(live, note):
    documented = _num(note, r"RR = 1\.0 on ([\d.]+)%", "RR==1.0 share")
    _close((live["reinvestment_rate"] == 1.0).mean() * 100, documented, "share of rows with RR==1.0")


def test_documented_misallocation_fire_rate_is_current(live, note):
    documented = _num(note, r"capital_misallocation_risk .*?fires ([\d.]+)%", "misallocation fire rate")
    _close(live["capital_misallocation_risk"].mean() * 100, documented, "capital_misallocation_risk")


def test_documented_epoch2_fire_rate_is_current(live, note):
    documented = _num(note, r"flag_epoch2_compounder .*?fires ([\d.]+)%", "epoch2 fire rate")
    _close(live["flag_epoch2_compounder"].mean() * 100, documented, "flag_epoch2_compounder")


# -- 2. The evidence-fabrication measurement --------------------------------------------
def test_documented_share_of_flags_resting_on_absent_dpr_is_current(live, note):
    """The load-bearing claim: most flagged stocks are condemned on a fillna, not on evidence."""
    documented = _num(note, r"601 of them \(([\d.]+)%\)", "share of flags with no DPR")
    flagged = live["capital_misallocation_risk"] == 1
    assert flagged.sum() > 0, "nothing is flagged -- the gate died; the comment needs a rewrite"
    actual = (flagged & live["dividend_payout_ratio"].isna()).sum() / flagged.sum() * 100
    _close(actual, documented, "share of capital_misallocation_risk flags with NO DPR data")


def test_the_fillna_is_what_manufactures_those_flags(live):
    """Mechanism, not just correlation: every no-DPR row must sit at RR exactly 1.0, which is the
    fillna(0) result. If this stops holding, the comment's explanation is wrong even if its
    percentages still match."""
    no_dpr = live["dividend_payout_ratio"].isna()
    assert no_dpr.sum() > 0, "DPR is fully populated now -- the whole data-gap note is obsolete"
    assert (live.loc[no_dpr, "reinvestment_rate"] == 1.0).all(), (
        "a row with missing DPR no longer resolves to RR==1.0; the fillna(0) path changed and the "
        "comment's account of HOW the flags are manufactured is now wrong"
    )


# -- 3. The structural all-clear that must never come back -------------------------------
def test_the_rr_gates_are_not_inert(live):
    """The predecessor comment's actual harm was the word INERT. These gates discriminate; if one
    ever stops, that is a finding to document -- not a default to assume."""
    assert live["reinvestment_rate"].nunique() > 1, (
        "reinvestment_rate is constant again. The RR-gated signals are now decided entirely by "
        "their other legs -- update the comment before trusting any of them."
    )
    for col in ("capital_misallocation_risk", "flag_epoch2_compounder"):
        rate = live[col].mean()
        assert 0.0 < rate < 1.0, (
            f"{col} fires on {rate:.1%} of the universe -- it has become inert (all or nothing) and "
            f"is no longer a test of anything. Re-read {_WHERE}."
        )


def test_the_haircut_target_is_still_the_low_roce_cohort(live, note):
    """The comment sizes the harm as small BECAUSE flagged names already score badly. If flagged
    and unflagged ROCE ever converge, that argument collapses and the 10% haircut needs re-judging."""
    flagged = live["capital_misallocation_risk"] == 1
    f_roce = live.loc[flagged, "roce_med_5y"].median()
    u_roce = live.loc[~flagged, "roce_med_5y"].median()
    assert f_roce < u_roce, (
        f"flagged 5Y ROCE median ({f_roce:.2f}) is no longer below unflagged ({u_roce:.2f}). The "
        f"comment argues the 10% haircut is harmless because it lands on already-weak businesses; "
        f"that argument no longer holds and the penalty needs re-judging."
    )


# -- 4. The standing decision must stay visible ------------------------------------------
def test_the_note_still_records_why_this_is_not_guarded(note):
    """A future reader who does not know about the 2026-06-14 ruling will 'helpfully' add the NaN
    guard that was explicitly rejected. The reason has to travel with the code."""
    assert "2026-06-14" in note, "the standing no-guard decision lost its date"
    low = note.lower()
    assert "rejected" in low or "standing" in low, (
        "the note no longer records that neutralising guards are a REJECTED approach here"
    )
    assert "census" in low, "the pointer to tools/census.py's stagnant_cash_cow triage is gone"
