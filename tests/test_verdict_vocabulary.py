"""
test_verdict_vocabulary.py
==========================
Contract for the engine verdict's vocabulary: 🟢 SOUND · 🟡 MIXED · 🔴 FLAWED.

RENAMED 2026-08-27 from BUY / WATCH / AVOID, for two reasons this file exists to keep true:

  1. NO WORD IN TWO LAYERS. The wealth tier deliberately speaks action words (BUY★ / BUY /
     WATCH★ / WATCH / AVOID / N/A — the user's "calculated verdict, my decision"). The engine
     verdict speaking the SAME words meant "AVOID" carried two different meanings in adjacent
     columns — the cf_triangle "Perfect — Buy Zone" defect class rebuilt at vocabulary level.
     The engine now speaks CONDITION words only.

  2. HONESTY. With 18 affirmatives against 93% of the universe failing, the engine verdict was
     never a buy call — it is a qualification gate (quality + forensics + valuation + timing).
     Condition words say what it measures; the action stays with the user.

THE COLLISION AUDIT THAT CHOSE THESE WORDS (each alternative rejected for a concrete reason):
     PASS / FAIL           conflates with gate_pass / schilit_pass / "Gate Passed"
     QUALIFIED / REJECTED  conflates with the Sectors tab's "% Qualify" (= gate-pass rate)
     STRONG / WEAK         already verdict_strength's vocabulary AND a conviction-tier label
     NEUTRAL               already smart_money_flow ⚪ and sector_capital_phase ⚖️
     UNSOUND               SOUND ⊂ UNSOUND — substring nesting, the bug class hit four times
                           on 2026-08-27 alone (QGLP⊂SQGLP, "sis ltd"⊂Mphasis, …)

THE THREE-LAYER ARCHITECTURE THIS PINS:
     Conviction Tier (1–5, Crown Jewel…)  how good, per the composite      — graded
     Verdict (SOUND/MIXED/FLAWED)         does it qualify, with the vetoes — gate
     Wealth Tier (BUY★…N/A)               which direction                  — change

Old snapshots keep the recorded BUY/WATCH/AVOID strings; tools/validate.py maps them on load so
longitudinal comparisons survive the rename.

Run with: pytest tests/test_verdict_vocabulary.py -v
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

from core.verdict_engine import compute_verdict

ENGINE_WORDS = {"SOUND", "MIXED", "FLAWED"}
WEALTH_WORDS = {"BUY★", "BUY", "WATCH★", "WATCH", "AVOID", "N/A"}
OLD_WORDS = {"BUY", "WATCH", "AVOID"}

_BASE = {
    "conviction_tier": 1, "composite_score": 90.0, "forensic_score": 90.0,
    "schilit_pass": 1, "red_flag_count": 1, "corporate_class": "🏆 GREAT",
    "quality_score": 90.0, "growth_score": 80.0, "expected_excess_return": 20.0,
    "pe": 20.0, "fair_pe_qglp": 30.0, "buy_zone_label": "Accumulate",
    "governance_risk_multiplier": 1.0, "data_coverage_pct": 90.0,
    "net_worth": 1000.0, "economic_profit": 100.0,
    "economic_profit_velocity": 50.0, "moat_tau": 1.0,
}


def _verdict(**o):
    return compute_verdict(pd.DataFrame([{**_BASE, **o}]))


# -- 1. The vocabulary itself ----------------------------------------------------------------
def test_the_three_states_and_their_emoji():
    for o, word, emoji in [({}, "SOUND", "🟢"),
                           ({"conviction_tier": 3}, "MIXED", "🟡"),
                           ({"conviction_tier": 5}, "FLAWED", "🔴")]:
        d = _verdict(**o)
        assert d["verdict_direction"].iloc[0] == word
        assert d["verdict_emoji"].iloc[0] == emoji, f"{word} lost its {emoji}"


def test_the_vetoes_speak_the_new_words():
    assert _verdict(forensic_score=40.0)["verdict_direction"].iloc[0] == "FLAWED"
    assert _verdict(schilit_pass=0)["verdict_direction"].iloc[0] == "MIXED"   # soft downgrade


def test_old_action_words_never_appear():
    for o in ({}, {"conviction_tier": 3}, {"conviction_tier": 5}, {"forensic_score": 40.0}):
        v = _verdict(**o)["verdict_direction"].iloc[0]
        assert v not in OLD_WORDS, f"the old vocabulary is back: {v!r}"
        assert v in ENGINE_WORDS


# -- 2. THE RULE: no word in two layers ------------------------------------------------------
def test_engine_and_wealth_vocabularies_are_disjoint():
    """The reason for the rename. If these sets ever intersect, one word means two things in
    adjacent columns again."""
    assert ENGINE_WORDS & WEALTH_WORDS == set(), (
        f"vocabulary collision between the verdict and the wealth tier: "
        f"{ENGINE_WORDS & WEALTH_WORDS}"
    )


def test_no_engine_word_nests_inside_another():
    """UNSOUND was rejected because SOUND ⊂ UNSOUND. Keep the set nesting-free so substring
    matching can never silently cross-match (the QGLP⊂SQGLP class)."""
    for a in ENGINE_WORDS:
        for b in ENGINE_WORDS:
            if a != b:
                assert a not in b, f"{a!r} is a substring of {b!r}"


# -- 3. On the live universe -----------------------------------------------------------------
@pytest.fixture(scope="module")
def live():
    from data_engine import (coerce_numeric_columns, compute_derived_signals, load_all_csvs,
                             merge_datasets)
    from core import run_scoring_pipeline
    with contextlib.redirect_stdout(_io.StringIO()):
        return run_scoring_pipeline(
            compute_derived_signals(coerce_numeric_columns(merge_datasets(load_all_csvs("local")))))


def test_live_verdicts_use_exactly_the_new_vocabulary(live):
    got = set(live["verdict_direction"].dropna().astype(str))
    assert got <= ENGINE_WORDS, f"unexpected verdict values on live data: {got - ENGINE_WORDS}"
    assert got == ENGINE_WORDS, f"a verdict state died on live data — only {got} present"


def test_the_gate_stays_a_gate(live):
    """The calibration that justified condition words: an affirmative-rare, reject-heavy gate.
    If SOUND ever became common, the 'this is a gate, not advice' framing needs revisiting."""
    share = live["verdict_direction"].value_counts(normalize=True)
    assert share.get("SOUND", 0) < 0.10, f"SOUND at {share.get('SOUND', 0):.1%} — no longer rare"
    assert share.get("FLAWED", 0) > 0.50, "FLAWED no longer dominates — the gate's shape changed"


def test_the_glossary_explains_all_three(live):
    from ui.ui_reference_data import CONCEPT_REFERENCE
    section = next((v for k, v in CONCEPT_REFERENCE.items() if "Verdict" in k), None)
    assert section is not None, "the Verdict glossary section is gone"
    documented = {lab for lab, _ in section}
    assert ENGINE_WORDS <= documented, f"glossary missing: {ENGINE_WORDS - documented}"
    assert not (OLD_WORDS & documented), "the glossary still documents the old action words"


# -- 4. The strength word stays off the screen -----------------------------------------------
def test_verdict_strength_is_never_rendered():
    """DISPLAY-RETIRED 2026-08-27. verdict_strength is a measured 1:1 map of conviction_tier --
    zero added information -- and the hero showed it beside the tier badge AND the score: the
    same fact three times on one screen ("MIXED / HIGH CONVICTION / Score 90/100"). The COLUMN
    remains (snapshot-schema stability; the orphan principle -- an unsurfaced column harms
    nobody); no UI surface may READ it for display again. A surface that wants a strength word
    should render the tier, which is the same fact with its own established vocabulary.

    Matches the READ pattern, not the word -- app.py's explanatory comment legitimately names
    the column, and prose that names a banned thing is not the banned thing (the substring-scan
    lesson, learned four times on 2026-08-27)."""
    import glob
    import re
    root = os.path.join(os.path.dirname(__file__), "..")
    read_pat = re.compile(r"""(?:\.get\(\s*|_sg\(\s*|_g\(\s*\w+,\s*)["']verdict_strength["']""")
    offenders = []
    for f in [os.path.join(root, "app.py")] + glob.glob(os.path.join(root, "ui", "*.py")):
        if read_pat.search(_io.open(f, encoding="utf-8").read()):
            offenders.append(os.path.basename(f))
    assert not offenders, (
        f"these UI files read verdict_strength for display again: {offenders} -- it is a 1:1 "
        f"duplicate of the tier; render the tier instead"
    )
