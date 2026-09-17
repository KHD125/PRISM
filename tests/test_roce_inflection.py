"""
test_roce_inflection.py
=======================
Contract for `roce_inflection` — the SHORT-window companion to `roce_expansion`.

WHY THIS COLUMN EXISTS. `roce_expansion` (3Y median - 10Y median) reads STRUCTURAL re-rating and
is deliberately damped. It cannot see a business that turned last year. Measured Spearman between
the two is +0.051, and of the 697 stocks the short window flags only 357 also carry a positive
roce_expansion: half of what each finds is invisible to the other.

WHY IT IS NOT THE SOURCE'S FORMULA. The source is the published Screener.in screen of Ishmohit
Arora (SOIC), whose second condition reads "ROCE preceding year > average ROCE (3 years)".
Implemented literally against THESE columns it selects deterioration, because the vendor's 3Y
window is exactly {roce, roce_1yb, roce_2yb} -- verified, roce_med_3y equals the median of those
three on 100.0% of rows -- which puts roce_1yb inside its own baseline and reduces the test to
`roce_1yb > (roce + roce_2yb) / 2`, a local-peak test on the MIDDLE year. Measured live:

    literal form   fires on 586/586 (100.0%) PEAKED-and-falling, 262/482 (54.4%) steadily RISING
    corrected form fires on 149/586 ( 25.9%) PEAKED-and-falling, 482/482 (100.0%) steadily RISING

That is CLAUDE.md section 5 "book fidelity vs math": the intent is right, the literal translation
is backwards, so math wins and the deviation is documented in place. Section 2 below is the pin
that keeps it from being "corrected" back.

FOUR THINGS THIS FILE DEFENDS:

1. THE VALUE. Latest against its own prior base, both terms from the same yearly family
   (CLAUDE.md section 5 cross-year basis rule), units in percentage POINTS.
2. THE DIRECTION. Rising must read positive and peaked must read negative. This is the whole
   reason the formula deviates from its source, and it is the test a revert has to break.
3. NO FABRICATED ZERO. A missing operand is NaN, never 0 -- "no third year of history" is not
   "no inflection". Deliberately NO collapse guard: unlike roce_expansion's nested medians,
   these are three distinct yearly observations (0.00% carry all three equal).
4. IT MUST NOT REACH THE SCORE. Display + sort only; the forward test is the December vintage.

Run with: pytest tests/test_roce_inflection.py -v
"""

import contextlib
import io as _io
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

import numpy as np
import pandas as pd
import pytest

_CORE = os.path.join(os.path.dirname(__file__), "..", "core")


@pytest.fixture(scope="module")
def live():
    from core import fetch_and_clean_data, run_scoring_pipeline
    with contextlib.redirect_stdout(_io.StringIO()):
        return run_scoring_pipeline(fetch_and_clean_data("local").copy())


def _src(name):
    return _io.open(os.path.join(_CORE, name), encoding="utf-8").read()


def _num(df, c):
    return pd.to_numeric(df[c], errors="coerce")


def _spearman(a, b):
    """Spearman rho WITHOUT scipy -- Pearson on ranks, which is the definition, not an approximation.

    `Series.corr(method="spearman")` imports scipy, and scipy is NOT in requirements.txt (production
    never calls .corr() at all). It happened to be installed in the dev venv, so these three tests
    passed locally and died the moment the suite ran against the DEPLOYED dependency set -- the
    undeclared-dependency class, in the test layer this time.

    Verified equal on the exact three pairs below against the real 2,716-row frame: worst absolute
    difference 1.39e-17, i.e. float noise. Tie handling is load-bearing (roce_inflection carries 182
    duplicate values) and pandas .rank() defaults to 'average', the same convention scipy uses.
    """
    return a.rank().corr(b.rank())


def test_the_spearman_helper_is_a_real_spearman():
    """Pins the helper itself, on hand-computable cases — the three tests below trust it.

    Without this, the docstring's "identical to scipy" claim is an assertion nobody checks, and a
    future edit to plain `.corr()` (Pearson on VALUES, not ranks) would slide through: the three
    live thresholds are loose bounds, so they do not notice a changed correlation measure.
    """
    assert _spearman(pd.Series([1, 2, 3, 4, 5]), pd.Series([5, 4, 3, 2, 1])) == pytest.approx(-1.0)
    assert _spearman(pd.Series([1, 2, 3, 4, 5]), pd.Series([2, 4, 6, 8, 10])) == pytest.approx(1.0)
    # monotone-but-not-linear: Spearman is 1.0 where Pearson is not — the case that separates them
    assert _spearman(pd.Series([1, 2, 3, 4]), pd.Series([1, 2, 4, 800])) == pytest.approx(1.0)
    assert pd.Series([1, 2, 3, 4]).corr(pd.Series([1, 2, 4, 800])) < 0.95, (
        "fixture no longer discriminates rank correlation from value correlation")
    # ties must take AVERAGE ranks (scipy's convention): ranks [1, 2.5, 2.5, 4] vs [1, 2, 3, 4]
    assert _spearman(pd.Series([1, 2, 2, 3]), pd.Series([1, 2, 3, 4])) == pytest.approx(
        4.5 / (22.5 ** 0.5))


# -- 1. The value ------------------------------------------------------------------------

def test_the_column_exists_and_is_the_documented_difference(live):
    """latest ROCE minus the mean of its own two prior years, in percentage points."""
    assert "roce_inflection" in live.columns
    expected = _num(live, "roce") - (_num(live, "roce_1yb") + _num(live, "roce_2yb")) / 2.0
    got = _num(live, "roce_inflection")
    both = expected.notna() & got.notna()
    assert both.sum() > 1000, "too few comparable rows to verify the formula"
    assert np.allclose(got[both], expected[both], atol=1e-9)


def test_both_terms_come_from_the_same_yearly_family(live):
    """Cross-year basis rule: a delta whose terms are built differently is not a delta.

    The baseline is the mean of two YEARLY observations, never a median column -- mixing a
    yearly level against a median window is the Economic Profit defect.
    """
    src = _src("data_engine.py")
    block = src[src.index('df["roce_inflection"]') - 200: src.index('df["roce_inflection"]') + 200]
    assert "roce_1yb" in block and "roce_2yb" in block
    assert "roce_med" not in block, "baseline must not be a median window"


def test_units_are_percentage_points_not_a_ratio(live):
    """A ratio would sit around 1.0; percentage points straddle zero with a real spread."""
    v = _num(live, "roce_inflection").dropna()
    assert v.min() < -1.0 and v.max() > 1.0
    assert abs(v.median()) < 5.0, "a percentage-point delta should centre near zero"


# -- 2. The direction: the entire reason this deviates from its source --------------------

def test_a_steadily_rising_business_reads_positive(live):
    """roce_2yb < roce_1yb < roce must read positive. The literal source form misses 45.6%."""
    y0, y1, y2 = (_num(live, c) for c in ("roce", "roce_1yb", "roce_2yb"))
    rising = (y0 > y1) & (y1 > y2) & y0.notna() & y1.notna() & y2.notna()
    assert rising.sum() > 100, "fixture universe has too few rising businesses to judge"
    v = _num(live, "roce_inflection")[rising]
    # NaN here is the degenerate-ROCE guard doing its job, not a direction error — but it
    # must not have swallowed the population this test exists to check.
    assert v.notna().mean() > 0.90, "the guard removed too much of the rising cohort"
    assert (v.dropna() > 0).all(), (
        "a monotonically rising ROCE must never read as a negative inflection"
    )


def test_a_business_that_peaked_and_is_falling_is_not_rewarded(live):
    """The literal source form fires on 100% of these. The corrected form must not."""
    y0, y1, y2 = (_num(live, c) for c in ("roce", "roce_1yb", "roce_2yb"))
    peaked = (y1 > y2) & (y0 < y1) & y0.notna() & y1.notna() & y2.notna()
    assert peaked.sum() > 100
    share_positive = (_num(live, "roce_inflection")[peaked] > 0).mean()
    assert share_positive < 0.40, (
        f"{share_positive:.1%} of peaked-and-falling businesses read positive; the literal "
        f"source form scores 100% here and that is exactly the defect this column deviates from"
    )


def test_the_literal_source_form_is_not_what_shipped(live):
    """Pins the deviation itself, so a well-meaning 'fix' back to the source breaks a test."""
    y0, y1, y2 = (_num(live, c) for c in ("roce", "roce_1yb", "roce_2yb"))
    literal = y1 - (y0 + y1 + y2) / 3.0          # roce_1yb vs the 3Y average, self-included
    shipped = _num(live, "roce_inflection")
    both = literal.notna() & shipped.notna()
    assert not np.allclose(shipped[both], literal[both], atol=1e-6)
    rho = _spearman(shipped[both], literal[both])
    assert rho < 0.75, f"shipped column has drifted back toward the literal form (rho={rho:+.3f})"


# -- 3. No fabricated zero ----------------------------------------------------------------

def test_a_missing_operand_is_nan_never_zero(live):
    """'No third year of history' is not 'no inflection' -- the sentinel bug."""
    y0, y1, y2 = (_num(live, c) for c in ("roce", "roce_1yb", "roce_2yb"))
    missing = y0.isna() | y1.isna() | y2.isna()
    assert missing.sum() > 0, "fixture has no missing-operand rows to test"
    assert _num(live, "roce_inflection")[missing].isna().all()


def test_exact_zeros_are_rare_the_degeneracy_that_broke_roce_trajectory(live):
    """roce_trajectory shipped 34% exact zeros. Three distinct yearly values must not."""
    v = _num(live, "roce_inflection").dropna()
    assert (v == 0).mean() < 0.02, "a pile-up at exactly 0.00 means fabricated flatness"


def test_degenerate_roce_is_guarded_out_so_the_sort_opens_on_signal(live):
    """ROCE explodes as capital employed approaches zero. The first live census put Sharp
    India (roce 34,166.67) at the top of this column; a descending sort opened on garbage.
    Anything with |ROCE| > 100 in any of the three years is NaN, not clipped."""
    y0, y1, y2 = (_num(live, c) for c in ("roce", "roce_1yb", "roce_2yb"))
    degenerate = (y0.abs() > 100) | (y1.abs() > 100) | (y2.abs() > 100)
    assert degenerate.sum() > 10, "fixture has no degenerate-ROCE rows to test"
    assert _num(live, "roce_inflection")[degenerate].isna().all(), (
        "a degenerate ROCE must yield NaN; clipping would fabricate a bounded value"
    )
    top = _num(live, "roce_inflection").nlargest(10)
    assert top.max() < 200, (
        f"the top of the column is {top.max():.0f}pp — the sort is opening on degenerate ROCE"
    )


def test_no_collapse_guard_was_added_speculatively(live):
    """Measured: 0.00% of rows carry all three operands equal. A guard would be dead code."""
    y0, y1, y2 = (_num(live, c) for c in ("roce", "roce_1yb", "roce_2yb"))
    collapsed = (y0 == y1) & (y1 == y2) & y0.notna()
    assert collapsed.mean() < 0.01, (
        "all-three-equal is no longer negligible; the section 2 argument for omitting a "
        "collapse guard was measured and would need re-measuring"
    )


# -- 4. It must not reach the score --------------------------------------------------------

@pytest.mark.parametrize("module", ["scoring_engine.py", "forensic_engine.py", "verdict_engine.py"])
def test_no_scoring_layer_consumes_it(module):
    assert "roce_inflection" not in _src(module), (
        f"{module} reads roce_inflection; this is a display + sort column and the forward "
        f"test is the December vintage"
    )


def test_it_is_not_wired_into_the_composite(live):
    """Structural absence can be dodged; this catches the behaviour too."""
    sub = pd.DataFrame({
        "i": _num(live, "roce_inflection"),
        "c": _num(live, "composite_score"),
    }).dropna()
    rho = _spearman(sub["i"], sub["c"])
    assert abs(rho) < 0.45, f"roce_inflection tracks composite_score too closely (rho={rho:+.3f})"


# -- 5. Alive, and genuinely additive ------------------------------------------------------

def test_the_signal_is_alive_and_no_band_owns_the_universe(live):
    v = _num(live, "roce_inflection")
    assert 0.60 < v.notna().mean() < 0.98, "coverage outside the measured 88.1%"
    assert v.nunique() > 500, "too few distinct values to be a real measurement"
    assert 0.20 < (v > 0).mean() < 0.60, "a direction that owns the universe is not a signal"


def test_it_is_not_a_restatement_of_roce_expansion(live):
    """The companion it was built beside. Measured +0.051."""
    sub = pd.DataFrame({
        "i": _num(live, "roce_inflection"),
        "e": _num(live, "roce_expansion"),
    }).dropna()
    rho = _spearman(sub["i"], sub["e"])
    assert abs(rho) < 0.40, f"roce_inflection has collapsed into roce_expansion (rho={rho:+.3f})"


def test_it_still_discriminates_where_the_existing_level_signal_is_neutral(live):
    """roce_current_vs_med (roce - roce_med_10y) shares this column's numerator and correlates
    +0.635 on levels. What justifies a separate column is that it still separates rising from
    peaked INSIDE each band of that signal -- measured gap +3.24pp in the middle quintile."""
    y0, y1, y2 = (_num(live, c) for c in ("roce", "roce_1yb", "roce_2yb"))
    d = pd.DataFrame({
        "infl": _num(live, "roce_inflection"),
        "cvm": _num(live, "roce_current_vs_med"),
        "rising": (y0 > y1) & (y1 > y2),
        "peaked": (y1 > y2) & (y0 < y1),
    }).dropna(subset=["infl", "cvm"])
    d["q"] = pd.qcut(d["cvm"], 5, labels=False, duplicates="drop")
    mid = d[d["q"] == 2]
    gap = mid.loc[mid["rising"], "infl"].median() - mid.loc[mid["peaked"], "infl"].median()
    assert gap > 1.0, (
        f"inside the middle band of roce_current_vs_med the column no longer separates rising "
        f"from peaked (gap={gap:+.2f}pp); without that it is a restatement of the level"
    )


# -- 6. The deviation stays documented ------------------------------------------------------

def test_the_source_deviation_is_recorded_in_place():
    """A future reader must find WHY this is not the source's formula, with the numbers."""
    src = _src("data_engine.py")
    i = src.index('df["roce_inflection"]')
    block = src[max(0, i - 2600): i]
    for token in ("Screener", "100.0%", "54.4%", "25.9%", "DISPLAY"):
        assert token in block, f"provenance/deviation note is missing {token!r}"


# -- 7. It is actually on screen, beside the column it completes ---------------------------

_APP = os.path.join(os.path.dirname(__file__), "..", "app.py")
_UI = os.path.join(os.path.dirname(__file__), "..", "ui")


def test_it_is_in_the_quality_view_and_has_a_display_format():
    app = _io.open(_APP, encoding="utf-8").read()
    assert '"roce_inflection"' in app, "column never reaches the Deep Scanner"
    fmt = re.search(r'"roce_inflection":\s*\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)', app)
    assert fmt, "no display format registered"
    header, spec = fmt.group(1), fmt.group(2)
    assert "pp" in spec, f"units must read as percentage POINTS, got {spec!r}"
    assert not re.fullmatch(r"[a-z_]+", header), "header must not be a raw snake_case column name"


def test_it_sits_beside_roce_expansion_in_the_quality_view():
    """Adjacency is the product: the slow reading and the fast reading are read together,
    and both sit beside the raw roce level that they qualify."""
    app = _io.open(_APP, encoding="utf-8").read()
    quality = app[app.index('"📊 Quality"'): app.index('"💰 Valuation"')]
    for col in ("roce", "roce_expansion", "roce_inflection"):
        assert f'"{col}"' in quality, f"{col} missing from the Quality view"
    assert quality.index('"roce_expansion"') < quality.index('"roce_inflection"')


def test_both_roce_deltas_are_in_the_reference_glossary():
    """The Reference tab renders _RAW_GLOSSARY. roce_expansion shipped 2026-09-10 with a scanner
    tooltip but NO glossary entry, so 'ROCE' in Reference search returned three entries and
    neither delta column for five days. Both are now there."""
    from ui.ui_components import _RAW_GLOSSARY
    for label in ("ROCE Δ 10Y", "ROCE Δ 2Y"):
        assert label in _RAW_GLOSSARY, f"{label!r} missing from the Reference glossary"
        assert len(_RAW_GLOSSARY[label].strip()) >= 40, f"{label!r} entry too short to be real"


def test_the_grid_header_and_the_glossary_cannot_drift():
    """Same contract as test_scanner_tips_reuse_the_shared_glossary: ONE definition. The text is
    duplicated in ui_scanner (deliberately, by the user's call) so this pins them equal instead."""
    from ui.ui_components import _RAW_GLOSSARY
    from ui.ui_scanner import _SCANNER_HEADER_TIPS
    for col, label in (("roce_expansion", "ROCE Δ 10Y"), ("roce_inflection", "ROCE Δ 2Y")):
        assert _SCANNER_HEADER_TIPS[col] == _RAW_GLOSSARY[label], (
            f"{col!r} grid tooltip has drifted from the {label!r} glossary entry"
        )


def test_the_tooltip_explains_the_degenerate_roce_blank():
    """Ksolves (ROCE 148.9%) renders 'None' with all three years present — a tooltip that says
    blank only means missing years would leave the reader unable to explain what they see."""
    from ui.ui_components import _RAW_GLOSSARY
    text = _RAW_GLOSSARY["ROCE Δ 2Y"]
    assert "100%" in text, "the degenerate-ROCE guard is invisible to the reader"


def test_it_carries_a_header_tooltip():
    """Every Deep Scanner preset column must explain itself — the rule that caught the
    roce_expansion author shipping without one."""
    # Assert the RESOLVED value, never the source text: the tip is sourced from _GLOSSARY, so a
    # regex looking for a string literal reports "no tooltip" on a perfectly wired column. That
    # is the source-scan trap this repo has already paid for twice.
    from ui.ui_scanner import _SCANNER_HEADER_TIPS
    tip = _SCANNER_HEADER_TIPS.get("roce_inflection", "")
    assert len(tip.strip()) >= 40, "no usable header tooltip registered for roce_inflection"
    assert "ROCE" in tip
