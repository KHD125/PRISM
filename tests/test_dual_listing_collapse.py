"""
test_dual_listing_collapse.py
=============================
Contract: the universe holds ONE ROW PER COMPANY.

WHY THIS FILE EXISTS. On the 2026-09-09 refresh Brand Concepts Ltd arrived twice -- BSE:BCONCEPTS
and NSE:BCONCEPTS, the same company priced on two venues -- and it was the first duplicate `name`
this universe had ever carried. Two things broke at once:

  * REACHABILITY. app.py picks a stock by NAME and resolves `df[df.name == selected].iloc[0]`, so
    one row was unreachable and the Tear-Sheet export shipped whichever one .iloc[0] landed on.
  * DOUBLE-COUNTING, the worse one. Every sector/industry aggregate, every fire-rate census and
    every Movers diff counted that company twice -- in exactly the breadth numbers those surfaces
    exist to report.

`core.data_engine._collapse_dual_listings` keeps the NSE listing (the deeper book, so its market
cap and technicals are what a buyer transacts against). The tie-break is TOTAL -- NSE, then BSE,
then lowest company_id -- so a venue the rule has never seen still collapses deterministically
rather than by row order, which PYTHONHASHSEED and merge order can both permute.

The live-data case is the one that actually fired; the synthetic cases pin the rule's edges, which
live data does not currently exercise and therefore cannot defend.

Run with: pytest tests/test_dual_listing_collapse.py -v
"""

import contextlib
import io as _io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

import numpy as np
import pandas as pd
import pytest

from data_engine import _collapse_dual_listings, merge_datasets


def _frame(pairs):
    """(company_id, name) pairs -> a frame shaped like the merged master."""
    return pd.DataFrame({
        "company_id": [c for c, _ in pairs],
        "name": [n for _, n in pairs],
        "market_cap": np.arange(len(pairs), dtype=float),
    })


# -- 1. The rule ------------------------------------------------------------------------
def test_nse_wins_over_bse():
    out = _collapse_dual_listings(_frame([("BSE:X", "Acme Ltd"), ("NSE:X", "Acme Ltd")]))
    assert list(out["company_id"]) == ["NSE:X"], "the NSE listing is the one that survives"


def test_nse_wins_regardless_of_which_row_came_first():
    """Row order must not decide identity -- the merge that produces this frame does not promise
    a stable exchange ordering."""
    out = _collapse_dual_listings(_frame([("NSE:X", "Acme Ltd"), ("BSE:X", "Acme Ltd")]))
    assert list(out["company_id"]) == ["NSE:X"]


def test_an_unknown_venue_loses_to_both_known_ones():
    out = _collapse_dual_listings(_frame([("MSE:X", "Acme Ltd"), ("BSE:X", "Acme Ltd")]))
    assert list(out["company_id"]) == ["BSE:X"], "a known venue outranks an unrecognised one"


def test_two_unknown_venues_collapse_deterministically_by_id():
    """No NSE/BSE to arbitrate: the lowest company_id wins, so the answer cannot depend on row
    order. Both orderings must give the SAME survivor -- that is what determinism means here."""
    a = _collapse_dual_listings(_frame([("ZZZ:X", "Acme Ltd"), ("AAA:X", "Acme Ltd")]))
    b = _collapse_dual_listings(_frame([("AAA:X", "Acme Ltd"), ("ZZZ:X", "Acme Ltd")]))
    assert list(a["company_id"]) == list(b["company_id"]) == ["AAA:X"]


def test_a_triple_listing_still_leaves_exactly_one_row():
    out = _collapse_dual_listings(
        _frame([("BSE:X", "Acme Ltd"), ("MSE:X", "Acme Ltd"), ("NSE:X", "Acme Ltd")]))
    assert list(out["company_id"]) == ["NSE:X"]


# -- 2. What it must NOT touch ----------------------------------------------------------
def test_distinct_companies_are_never_collapsed():
    pairs = [("NSE:A", "Acme Ltd"), ("NSE:B", "Beta Ltd"), ("BSE:C", "Gamma Ltd")]
    out = _collapse_dual_listings(_frame(pairs))
    assert list(out["company_id"]) == [c for c, _ in pairs], "a clean frame passes through intact"


def test_missing_names_are_never_treated_as_the_same_company():
    """NaN is not an identity. Collapsing several unnamed rows into one would be the sentinel bug
    CLAUDE.md bans -- inventing a fact (these are the same issuer) out of absent evidence."""
    df = _frame([("NSE:A", np.nan), ("NSE:B", np.nan), ("NSE:C", "Acme Ltd")])
    out = _collapse_dual_listings(df)
    assert len(out) == 3, "two rows with no name are two unknown companies, not one company"


def test_original_row_order_survives():
    """Downstream code reads this frame positionally in places; collapsing must not reshuffle it."""
    pairs = [("NSE:A", "Acme Ltd"), ("BSE:X", "Zeta Ltd"), ("NSE:X", "Zeta Ltd"),
             ("NSE:B", "Beta Ltd")]
    out = _collapse_dual_listings(_frame(pairs))
    assert list(out["name"]) == ["Acme Ltd", "Zeta Ltd", "Beta Ltd"]


def test_a_frame_without_the_identity_columns_is_returned_unchanged():
    df = pd.DataFrame({"market_cap": [1.0, 2.0]})
    assert _collapse_dual_listings(df) is df


# -- 3. It is actually WIRED IN -- existence is not reachability ------------------------
def test_merge_datasets_applies_the_collapse():
    """A rule that is defined but never called is worth nothing. This runs the real merge over a
    dual-listed pair and asserts the duplicate is gone by the time the master frame is returned."""
    base = pd.DataFrame({
        "company_id": ["BSE:X", "NSE:X", "NSE:A"],
        "name": ["Acme Ltd", "Acme Ltd", "Beta Ltd"],
        "market_cap": [10.0, 11.0, 12.0],
    })
    others = {k: pd.DataFrame({"company_id": base["company_id"], f"{k}_col": [1, 2, 3]})
              for k in ("income", "balance", "cashflow", "shareholding", "technical")}
    with contextlib.redirect_stdout(_io.StringIO()):
        out = merge_datasets({"ratio": base, **others})
    assert len(out) == 2, "merge_datasets must return one row per company"
    assert list(out["company_id"]) == ["NSE:X", "NSE:A"]


# -- 4. Live data -----------------------------------------------------------------------
@pytest.fixture(scope="module")
def live():
    from core import fetch_and_clean_data
    with contextlib.redirect_stdout(_io.StringIO()):
        return fetch_and_clean_data("local")


def test_the_live_universe_has_no_duplicate_names(live):
    """The failure that started this: `.iloc[0]` silently picks one of several rows."""
    dupes = live.loc[live["name"].notna() & live["name"].duplicated(keep=False), "name"]
    assert dupes.empty, f"duplicate names are back: {sorted(set(dupes))}"


def test_the_live_universe_has_no_duplicate_company_ids(live):
    assert live["company_id"].duplicated().sum() == 0


def test_one_name_means_one_row(live):
    assert live["name"].nunique() == len(live)
