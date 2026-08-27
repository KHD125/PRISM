"""
test_display_threshold_consistency.py
=====================================
ONE INVARIANT, APPLIED EVERYWHERE:

    A cell must be classified on the number it PRINTS, never on the raw value behind it.

Break it and the page contradicts itself in a way nothing on screen can explain — two cells
showing the identical text in different colours. Measured across ui_tearsheet.py on 2026-08-27,
before the sweep:

    score strip  (moat/growth/cash/momentum)   237 renders
    EP velocity                                 99
    QGLP legs    (quality/growth/longevity)    128
    cockpit "cheaper than X% of peers"          42
    hero ring                                   34   (deliberately NOT fixed -- see below)
                                               ---
                                               540

Worked example: growth_score 69.6744 and 70.3481 both render "70". One was gold, the other green.

WHY THIS FILE EXISTS RATHER THAN FOUR MORE ONE-OFF TESTS. The same defect was found and patched
four separate times in one session -- "PAT YoY", the Trajectory card, the VCR chip, and then these.
Each patch pinned its own instance. That is whack-a-mole against something a script can enumerate,
so this file pins the RULE and adds a structural net that fails when an 11th site appears.

THE REMEDY IS NOT UNIFORM, AND CHOOSING WRONG DESTROYS SIGNAL:

  * DISPLAY BAND (Strong/Average/Weak at 70/40) -> round, THEN classify. The cut is a
    presentation convention, so moving a value into the band its printed form implies loses
    nothing. This is `_shown()`.
  * REAL ECONOMIC LINE -> show MORE PRECISION instead. VCR's 1.00 is Buffett's one-dollar
    premise; blurring its edge would erase the distinction the chip exists to make. That site
    went 1dp -> 2dp (tests/test_deep_signal_chips.py).
  * SIGN with a genuine "neither" state -> round, then treat zero as its own band. EP velocity
    that rounds to 0 Cr is not ascending or descending; it reads "Flat".

WHAT IS EXEMPT, AND WHY IT IS PRINCIPLED RATHER THAN CONVENIENT. A colour that carries its OWN
label is not ambiguous -- the reader is not decoding it from the number. The hero ring is coloured
by conviction_tier and the tier's name sits right beside it, so its 34 renders are not a defect and
changing the app's most prominent figure to "26.0" would be a regression. The Fisher gauge formats
a CSS width, not a number anyone reads, and takes its colour from an integer pass-count.

Run with: pytest tests/test_display_threshold_consistency.py -v
"""

import ast
import contextlib
import io as _io
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

import pandas as pd
import pytest

import ui.ui_tearsheet as T
from data_engine import (coerce_numeric_columns, compute_derived_signals, load_all_csvs,
                         merge_datasets)

_TEARSHEET = os.path.join(os.path.dirname(__file__), "..", "ui", "ui_tearsheet.py")

# (site, column, decimals shown, band cuts high->low). Cuts are the UI's own thresholds.
REGISTRY = [
    ("score strip",   "moat_score",               0, [70, 40]),
    ("score strip",   "growth_score",             0, [70, 40]),
    ("score strip",   "cash_score",               0, [70, 40]),
    ("score strip",   "momentum_score",           0, [70, 40]),
    ("QGLP radar",    "qglp_quality",             0, [70, 40]),
    ("QGLP radar",    "qglp_growth",              0, [70, 40]),
    ("QGLP radar",    "qglp_longevity",           0, [70, 40]),
    ("QGLP radar",    "qglp_price",               0, [70, 40]),
    ("forensic",      "forensic_score",           0, [80, 60]),
    ("EP velocity",   "economic_profit_velocity", 0, [0]),
    ("cockpit vcv",   "value_creation_velocity",  2, [0]),
    ("cockpit egap",  "expectations_gap",         2, [0]),
]

# Functions allowed to format-and-colour WITHOUT a rounding classifier, each for a structural
# reason -- not because the affected count happened to be small.
EXEMPT = {
    "render_stock_hero":
        "ring colour comes from conviction_tier (an ENGINE classification) and the tier's own "
        "name is displayed beside it, so the number is not what the reader decodes",
    "render_fisher_module":
        "the formatted value is a CSS bar width, not a figure anyone reads; the colour comes from "
        "an integer pass-count",
}

# Helpers that classify on the DISPLAYED value. Any format+colour function must use one.
APPROVED = ("_shown(", "_accel_move(")


@pytest.fixture(scope="module")
def live():
    from core import run_scoring_pipeline
    with contextlib.redirect_stdout(_io.StringIO()):
        return run_scoring_pipeline(
            compute_derived_signals(coerce_numeric_columns(merge_datasets(load_all_csvs("local")))))


@pytest.fixture(scope="module")
def src():
    return _io.open(_TEARSHEET, encoding="utf-8").read()


def _band(value, cuts):
    for c in cuts:
        if value >= c:
            return c
    return -10**9


# -- 1. The invariant, PROVEN BY RENDERING THE UI ------------------------------------------
# An earlier version of this section computed banding from the dataframe and compared it to
# itself. That is a tautology: it never called a render function, so it could not have failed no
# matter what the UI did. A mutation run exposed it -- reverting the score strip to raw
# classification left it fully green. These tests render the real component across hundreds of
# stocks and read the colours back out of the HTML.

def _render(fn, row):
    out = []

    class _Rec:
        def markdown(self, *a, **k):
            if a:
                out.append(str(a[0]))

        def __getattr__(self, _n):
            return lambda *a, **k: None

    real = T.st
    try:
        T.st = _Rec()
        fn(row)
    finally:
        T.st = real
    return " ".join(out)


def _score_cells(html):
    """(label, printed value, zone word, zone colour) for each score-strip cell."""
    for c in html.split('class="ts-score-cell"')[1:]:
        lbl = re.search(r'ts-score-cell-lbl">([^<]+)', c)
        val = re.search(r'ts-score-cell-val" style="color:[^"]*">([^<]+)', c)
        zone = re.search(r'font-size:0\.52rem;color:(#[0-9a-fA-F]{6});[^>]*>([A-Za-z]+)<', c)
        if lbl and val and zone:
            yield lbl.group(1).strip(), val.group(1).strip(), zone.group(2), zone.group(1).lower()


def test_score_strip_never_prints_one_number_in_two_bands(live):
    """THE REGRESSION TEST WITH TEETH. Verified to fail on the real defect: reverting the strip to
    raw classification produces 7 clashes, e.g. "🌱 Growth 70" rendering as both Average/gold and
    Strong/green (from growth_score 69.6744 and 70.3481)."""
    seen = {}
    cells = 0
    for _, row in live.head(600).iterrows():
        for lbl, val, word, clr in _score_cells(_render(T.render_score_strip, row)):
            cells += 1
            seen.setdefault((lbl, val), set()).add((word, clr))
    assert cells > 1000, f"only {cells} cells rendered -- the extractor's selectors went stale"
    clashes = {k: v for k, v in seen.items() if len(v) > 1}
    assert not clashes, (
        "these cells print the SAME number in different bands/colours, with nothing on screen to "
        "explain the difference: " +
        " | ".join(f"{lbl} {val!r} -> {sorted(v)}" for (lbl, val), v in list(clashes.items())[:6])
    )


def test_the_extractor_can_actually_see_bands(live):
    """A selector that matched nothing would make the test above pass forever."""
    seen = set()
    for _, row in live.head(60).iterrows():
        for lbl, val, word, clr in _score_cells(_render(T.render_score_strip, row)):
            seen.add(word)
    assert {"Strong", "Average", "Weak"} & seen, f"no zone words extracted, got {seen}"
    assert len(seen) >= 2, f"only one band ever rendered ({seen}) -- cannot detect a clash"


def _qglp_legs(html):
    """(leg letter, printed value, colour) for each QGLP leg row."""
    for m in re.finditer(
        r"font-weight:900;color:(#[0-9a-fA-F]{6});'>([QGLP])</div>"      # letter, coloured
        r".*?text-align:right;font-size:0\.85rem;font-weight:800;color:(#[0-9a-fA-F]{6});'>([^<]+)<",
        html, re.S,
    ):
        yield m.group(2), m.group(4).strip(), m.group(3).lower()


def test_qglp_legs_never_print_one_number_in_two_colours(live):
    """Second-largest site (128 renders before the sweep). Same teeth as the score-strip test."""
    seen, n = {}, 0
    for _, row in live.head(400).iterrows():
        for letter, val, clr in _qglp_legs(_render(T.render_qglp_radar, row)):
            if val == "—":
                continue
            n += 1
            seen.setdefault((letter, val), set()).add(clr)
    assert n > 500, f"only {n} QGLP legs extracted -- the selectors went stale"
    clashes = {k: v for k, v in seen.items() if len(v) > 1}
    assert not clashes, (
        "QGLP legs printing the same number in different colours: " +
        " | ".join(f"{k} -> {sorted(v)}" for k, v in list(clashes.items())[:6])
    )


# -- 2. The helper itself -------------------------------------------------------------------
def test_shown_rounds_to_the_displayed_precision():
    assert T._shown(69.6744, 0) == 70.0
    assert T._shown(70.3481, 0) == 70.0
    assert T._shown(69.4, 0) == 69.0
    assert T._shown(1.0433, 2) == 1.04
    assert T._shown(-0.4, 0) == 0.0          # normalises toward the printed "0"


def test_shown_makes_straddling_values_agree():
    """The property in one line: two values that PRINT alike must BAND alike."""
    for a, b, dp in ((69.6744, 70.3481, 0), (39.6, 40.4, 0), (74.6, 75.4, 0)):
        assert f"{a:.{dp}f}" == f"{b:.{dp}f}", "probe values do not print alike"
        assert T._shown(a, dp) == T._shown(b, dp)


# -- 3. The structural net: no NEW site may skip the rule -----------------------------------
def _format_and_colour_functions(src):
    """Functions that BOTH format a number and pick a colour from a numeric comparison."""
    fmt = re.compile(r"\{[^{}]*:[^{}]*?\.\df\}")
    colour_by_number = re.compile(r"COLORS\[[^\]]+\][^\n]{0,40}if\s+[^\n]{0,40}[<>]=?\s*-?\d")
    alt = re.compile(r"if\s+[\w_.\(\)\[\]\"' ]{0,30}[<>]=?\s*-?\d+(\.\d+)?[^\n]{0,60}COLORS\[")
    lines = src.splitlines()
    out = {}
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = "\n".join(lines[node.lineno - 1:node.end_lineno])
        if fmt.search(body) and (colour_by_number.search(body) or alt.search(body)):
            out[node.name] = body
    return out


def test_every_format_and_colour_site_classifies_on_the_shown_value(src):
    """THE GENERIC NET -- a tripwire for NEW sites, not a proof of correctness.

    It fails the moment someone adds a function that colours a number by a threshold without a
    rounding classifier, which is how an 11th site would otherwise slip in unnoticed.

    KNOWN LIMITATION, STATED SO IT IS NOT OVER-TRUSTED: this checks that an approved helper
    APPEARS in the function, not that every threshold in it uses one. A function that calls
    _shown() once and still bands a second value on the raw figure passes here. That is exactly
    what happened during the sweep -- reverting one line inside render_qglp_radar left this test
    green, and only the render-based detector above caught it. Treat this as the net for
    UNKNOWN sites and the render tests as the proof for known ones; when a site grows large
    enough to matter, give it a render detector rather than relying on this."""
    offenders = []
    for name, body in _format_and_colour_functions(src).items():
        if name in EXEMPT:
            continue
        if not any(h in body for h in APPROVED):
            offenders.append(name)
    assert not offenders, (
        f"these functions format a number AND colour it by a numeric threshold, but classify on "
        f"the RAW value: {sorted(offenders)}.\n\n"
        f"Use _shown(value, decimals) so the band follows what the cell prints -- unless the "
        f"threshold is a real economic line, in which case show more precision instead (see this "
        f"file's docstring), or add a STRUCTURAL reason to EXEMPT."
    )


def test_the_structural_net_actually_finds_sites(src):
    """A scanner that matches nothing would pass the test above forever."""
    found = _format_and_colour_functions(src)
    assert len(found) >= 8, f"the scanner only found {len(found)} sites; its patterns went stale"
    assert "render_score_strip" in found or "_cell" in found, (
        "the scanner no longer sees the score strip, which is the largest known site"
    )


def test_exemptions_are_justified_and_still_needed(src):
    """An exemption list rots into a dumping ground. Every entry must carry a reason and must
    still correspond to a real site."""
    found = _format_and_colour_functions(src)
    for name, reason in EXEMPT.items():
        assert len(reason) > 40, f"{name}'s exemption has no substantive reason"
        assert name in found, (
            f"{name} is exempted but the scanner no longer flags it -- drop the exemption so the "
            f"list does not accumulate dead entries"
        )
