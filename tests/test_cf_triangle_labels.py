"""
test_cf_triangle_labels.py
==========================
Contract for the cash-flow-triangle labels (core/forensic_engine.cf_triangle).

THE BUG THIS CLOSES (2026-08-27, found by reading a live Igarashi Motors tearsheet).
cf_triangle's best value was named "✅ Perfect — Buy Zone". The column classifies a CASH-FLOW
SHAPE — operating cash positive, investing negative, financing negative — and says nothing about
whether to buy anything. But the wording read as a recommendation, and app.py renders the value
verbatim as a green badge a few pixels below the verdict header. Measured on the live universe:

    944 stocks carried the label
    835 of them (39.4% of ALL tearsheets) had a verdict of AVOID
    347 were classed "💀 GRUESOME"

So four in ten pages showed a green "Perfect — Buy Zone" tick beside "avoid regardless of how
cheap it looks". It also collided with buy_zone_label — the column that actually means entry
timing, whose top value is "🟢 Perfect Entry (Low Risk)". Igarashi displayed the badge while its
real buy_zone_label was "🟠 Loose Entry Zone".

WHY ONLY THE POSITIVE LABEL CHANGED. The asymmetry is measured, not assumed: "🚨 Debt Trap —
Avoid" fires on 287 stocks and EVERY one has an AVOID verdict (0 BUY), so its action word never
contradicts the page it sits on. A green badge making a claim the verdict denies is the whole
defect; a red badge agreeing with a red verdict is not. Renaming the negatives would be churn.

THE FAILURE MODE THIS FILE REALLY GUARDS. A rename like this touches four places — the engine,
the Discovery filter's canonical order, the glossary, and the handbook. Miss the glossary and
help_chip() silently loses the tooltip; miss the order list and the label still appears but sorts
to the end. Those are invisible in a screenshot. So the label set is derived from the ENGINE and
the other sites are checked against it.

Run with: pytest tests/test_cf_triangle_labels.py -v
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

_ROOT = os.path.join(os.path.dirname(__file__), "..")
_DISCOVERY = os.path.join(_ROOT, "ui", "ui_discovery.py")
_HANDBOOK = os.path.join(_ROOT, "docs", "handbook", "04-categories-and-labels.md")

# Words that turn a description of a cash-flow SHAPE into investment advice.
_ADVICE_WORDS = ["buy zone", "buy ", "accumulate", "entry zone", "sell "]


@pytest.fixture(scope="module")
def live():
    from core import run_scoring_pipeline
    with contextlib.redirect_stdout(_io.StringIO()):
        return run_scoring_pipeline(
            compute_derived_signals(coerce_numeric_columns(merge_datasets(load_all_csvs("local")))))


@pytest.fixture(scope="module")
def emitted(live):
    """The label set the ENGINE actually produces -- the single source everything else follows."""
    vals = sorted(set(live["cf_triangle"].dropna().astype(str)) - {""})
    assert len(vals) == 4, f"cf_triangle emits {len(vals)} labels, expected 4: {vals}"
    return vals


# -- 1. No cash-flow label may give investment advice --------------------------------------
def test_no_positive_label_reads_as_a_recommendation(emitted, live):
    """Only labels that can appear on a page whose verdict DISAGREES are constrained. A negative
    label that only ever co-occurs with AVOID is not making a false claim."""
    vd = live["verdict_direction"].astype(str)
    for lab in emitted:
        on_avoid = ((live["cf_triangle"].astype(str) == lab) & vd.str.contains("AVOID")).sum()
        if on_avoid == 0:
            continue                      # never shares a page with a contrary verdict
        low = lab.lower()
        hits = [w for w in _ADVICE_WORDS if w in low]
        assert not hits, (
            f"cf_triangle label {lab!r} contains advice wording {hits} and appears on {on_avoid} "
            f"tearsheets whose verdict is AVOID. cf_triangle describes a cash-flow SHAPE; it does "
            f"not know whether to buy. This is the exact defect the rename closed."
        )


def test_the_old_label_is_gone(emitted):
    assert not any("Perfect — Buy Zone" in l for l in emitted), (
        "the 'Perfect — Buy Zone' label is back; see this file's docstring for the 835-tearsheet "
        "contradiction it caused"
    )


def test_it_does_not_collide_with_the_real_entry_timing_vocabulary(emitted, live):
    """buy_zone_label is the column that MEANS entry timing. Two different columns must not both
    claim a 'Perfect ... Buy Zone' vocabulary, or a reader cannot tell which one a badge is."""
    bz = set(live["buy_zone_label"].dropna().astype(str))
    for lab in emitted:
        core = re.sub(r"[^a-z ]", "", lab.lower()).strip()
        for other in bz:
            other_core = re.sub(r"[^a-z ]", "", str(other).lower()).strip()
            assert core != other_core, f"cf_triangle {lab!r} duplicates a buy_zone_label value"
    assert any("perfect" in str(b).lower() for b in bz), (
        "buy_zone_label no longer has a 'Perfect' value -- re-check whether the collision this "
        "test guards still exists in the form described"
    )


# -- 2. The negative labels stay, and the reason is measured -------------------------------
def test_negative_labels_never_contradict_their_page(live):
    """'Debt Trap — Avoid' keeps its action word BECAUSE it never lands on a positive verdict.
    If that stops being true, its wording needs the same scrutiny the positive one got."""
    cf, vd = live["cf_triangle"].astype(str), live["verdict_direction"].astype(str)
    trap = cf.str.contains("Debt Trap", na=False)
    assert trap.sum() > 0, "the Debt Trap label vanished"
    on_buy = int((trap & vd.str.contains("BUY", na=False)).sum())
    assert on_buy == 0, (
        f"'Debt Trap — Avoid' now appears on {on_buy} tearsheets with a BUY verdict. It was left "
        f"unrenamed precisely because it never did; re-judge the wording."
    )


# -- 3. All four sites must agree (the invisible half of a rename) --------------------------
def test_glossary_covers_exactly_the_emitted_labels(emitted):
    """A glossary key that drifts from the emitted value silently kills the '?' tooltip."""
    from ui.ui_reference_data import CONCEPT_REFERENCE
    section = next((v for k, v in CONCEPT_REFERENCE.items() if "Cash-Flow Triangle" in k), None)
    assert section is not None, "the Cash-Flow Triangle glossary section is gone"
    documented = {lab for lab, _desc in section}
    assert documented == set(emitted), (
        f"glossary and engine disagree.\n  only in glossary: {sorted(documented - set(emitted))}\n"
        f"  only in engine:   {sorted(set(emitted) - documented)}"
    )


def test_discovery_filter_order_covers_exactly_the_emitted_labels(emitted):
    """_ordered_present() treats this list as the canonical ORDER and appends anything unknown
    LAST -- so a stale entry does not error, it just quietly sorts the label to the bottom."""
    src = _io.open(_DISCOVERY, encoding="utf-8").read()
    i = src.index('_ordered_present(_cf, "cf_triangle"')
    block = src[i:src.index("]", i) + 1]
    listed = set(re.findall(r'"([^"]+)"', block)) - {"cf_triangle"}
    assert listed == set(emitted), (
        f"Discovery's cf_triangle order list disagrees with the engine.\n"
        f"  only in list:   {sorted(listed - set(emitted))}\n"
        f"  only in engine: {sorted(set(emitted) - listed)}"
    )


def test_handbook_documents_the_current_label(emitted):
    """docs-as-code (CLAUDE.md §6). The handbook was the 4th site and the easiest to forget --
    an exhaustive grep found it after a narrower search had already missed it."""
    if not os.path.exists(_HANDBOOK):
        pytest.skip("handbook is local-only and not present in this checkout")
    text = _io.open(_HANDBOOK, encoding="utf-8").read()
    assert "Perfect — Buy Zone" not in text, "the handbook still documents the old label"
    positive = next(l for l in emitted if l.startswith("✅"))
    assert positive.replace("✅ ", "") in text, (
        f"the handbook does not mention {positive!r}; the rename stopped short of the docs"
    )


# -- 4. The contradiction is actually gone --------------------------------------------------
def test_no_green_badge_claims_a_buy_on_an_avoid_page(live):
    """The end-to-end property, measured the way the bug was found."""
    cf, vd = live["cf_triangle"].astype(str), live["verdict_direction"].astype(str)
    green = cf.str.startswith("✅")
    avoid = vd.str.contains("AVOID", na=False)
    overlap = int((green & avoid).sum())
    assert overlap > 0, "no green cf_triangle label co-occurs with AVOID -- probe has gone stale"
    offending = [l for l in set(cf[green & avoid])
                 if any(w in l.lower() for w in _ADVICE_WORDS)]
    assert not offending, (
        f"{overlap} tearsheets show a green cash-flow badge beside an AVOID verdict, and the "
        f"badge text makes a buy claim: {offending}"
    )
