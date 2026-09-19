"""
test_prelisting_dilution.py
===========================
Contract for the PRE-LISTING BASELINE arm of the dilution tiering (shipped 2026-09-18, Phase 2).

WHAT IT FIXES. `equity_shares_1yb` for a freshly listed company is its PRE-LISTING share count,
so (now - then)/then compares today's public float against a private shell. Unified Data-Tech
carried 5,010 prior shares -- an implied prior book value of Rs 155,289/share against a universe
median of Rs 112. The engine read that category error as predatory issuance and HARD-REJECTED on
it: 109 stocks sat at dilution_flag 3 with 0% gate_pass because they had recently IPO'd.

WHY A DAY THRESHOLD AT ALL, AND WHY 730. `listed_days` does not add an "IPO tier"; it identifies a
measurement that was never valid. The distortion is structural, not a curve fit: equity_shares_1yb
is a prior FISCAL-YEAR figure, and the gap between a vintage and the prior fiscal year end runs
~365-730 days depending on where in the year the vintage is taken. 730 is the maximum, so it is
the only bound that holds on EVERY vintage. On the 2026-09-18 vintage the observed boundary is
~537 days (FY2025 closed 31 Mar 2025) and the measured share-multiple medians land exactly there:

    0-90d 7.175 · 90-180d 12.042 · 180-270d 8.062 · 270-365d 1.370
    365-455d 1.370 · 455-545d 1.373 · 545-635d 1.023 · 635-730d 1.014 · >5y 1.000

test_the_threshold_still_covers_the_distortion below RE-MEASURES that every run, so a future
vintage that pushes the boundary past 730 fails here instead of silently under-covering.

TIER 1, NOT TIER 0. The corporate-action arm's own documented precedent: the share count genuinely
DID move, we just cannot interpret the move, so it must never be certified zero-dilution. Tier 1
clears rf_dilution (needs >= 2) and gate_no_dilution (needs 3) but still fails the
`dilution_flag == 0` pillars (Fisher P13 "zero equity dilution", Outsider CEO pillar S).

WHAT IT DELIBERATELY DOES NOT FIX, so docs/known-issues.md is not prematurely closed: this is the
false-REJECT half only. Vodafone Idea (listed 3998d, 1.5176x) is an established company riding
free on the >= 1.5x SIZE arm and is untouched. Separating a bonus from a QIP for a mature company
still needs the pro-rata shape test.

Run with: pytest tests/test_prelisting_dilution.py -v
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

from config import PRELISTING_BASELINE_DAYS


@pytest.fixture(scope="module")
def live():
    from core import fetch_and_clean_data, run_scoring_pipeline
    with contextlib.redirect_stdout(_io.StringIO()):
        return run_scoring_pipeline(fetch_and_clean_data("local").copy())


def _num(df, c):
    return pd.to_numeric(df[c], errors="coerce")


# ── 1. The rule ──────────────────────────────────────────────────────────────────────────

def test_the_threshold_is_the_prior_fiscal_year_bound_not_a_round_number():
    """730 = the maximum age a prior-FISCAL-YEAR figure can have. 365 would under-cover (the
    2026-09-18 vintage's own boundary is 537 days, and Belrise sits at 478)."""
    assert PRELISTING_BASELINE_DAYS == 730, (
        "the pre-listing bound moved. It is not a free parameter: equity_shares_1yb is a prior "
        "fiscal-year figure whose age runs ~365-730 days depending on the vintage, so 730 is the "
        "only value that holds on all of them. Changing it needs the measurement in the docstring "
        "redone plus a /census."
    )


def test_a_recent_listing_is_tier_1_not_tier_0_and_not_tier_3(live):
    """The whole fix, and the tier choice that makes it conservative."""
    flag = _num(live, "dilution_flag")
    pre = _num(live, "dilution_prelisting_baseline") == 1
    assert pre.sum() > 20, f"only {int(pre.sum())} pre-listing rows -- the arm looks dead"
    # Not tier 3: that was the hard reject this exists to stop.
    worst = flag[pre].max()
    assert worst <= 2, f"a pre-listing row is still at dilution_flag {worst:.0f}"
    # And never certified clean on a measurement we could not make.
    mild = pre & (_num(live, "dilution_pct") > 10.0)
    assert (flag[mild] == 1).all(), (
        "a pre-listing row with a material raw delta was certified dilution_flag 0 (zero "
        "dilution). The share count DID move; only its interpretation is unavailable."
    )


def test_it_only_ever_unblocks_never_newly_blocks(live):
    """Safety property: the arm may rescue a hard reject, it may never create one."""
    pre = _num(live, "dilution_prelisting_baseline") == 1
    assert (_num(live, "dilution_flag")[pre] != 3).all(), (
        "the pre-listing arm produced a Tier 3 — it is only ever allowed to relax one"
    )


def test_the_raw_percentage_is_still_reported(live):
    """The number is not hidden, only its interpretation is guarded — same as the >=1.5x arm."""
    pre = _num(live, "dilution_prelisting_baseline") == 1
    assert _num(live, "dilution_pct")[pre].notna().all(), "dilution_pct was blanked for IPOs"
    assert (_num(live, "dilution_pct")[pre] != 0).any(), "every pre-listing dilution_pct is 0 — suspicious"


# ── 2. The assumption behind the threshold, re-measured every run ────────────────────────

def test_the_threshold_still_covers_the_distortion(live):
    """THE SELF-POLICING PIN. If a future vintage moves the pre-listing boundary past 730, this
    fails and names it, instead of the arm silently under-covering and IPOs going back to Tier 3.
    """
    ld = _num(live, "listed_days")
    sh, sh1 = _num(live, "equity_shares"), _num(live, "equity_shares_1yb")
    mult = pd.Series(np.where(sh1 > 0, sh / sh1, np.nan), index=live.index)
    mature = mult[(ld >= 1825)].dropna()
    assert len(mature) > 500, "too few mature companies to form a baseline"
    baseline = float((mature >= 1.5).mean())
    # Just ABOVE the threshold the population must already look mature. A band still carrying a
    # big excess means the prior-year figure is older than 730 days on this vintage.
    just_above = mult[(ld >= PRELISTING_BASELINE_DAYS) & (ld < PRELISTING_BASELINE_DAYS + 365)].dropna()
    assert len(just_above) > 30, "too few companies just above the threshold to judge"
    excess = float((just_above >= 1.5).mean()) - baseline
    assert excess < 0.15, (
        f"companies just past the {PRELISTING_BASELINE_DAYS}-day bound still show a "
        f"{100*excess:.1f}pp excess of >=1.5x share multiples over the mature baseline "
        f"({100*baseline:.1f}%). The pre-listing distortion now extends BEYOND the threshold — "
        f"re-measure the fiscal-year gap for this vintage and widen PRELISTING_BASELINE_DAYS."
    )


def test_listed_days_is_censored_at_the_top_so_it_is_never_an_age_proxy(live):
    """Recorded, not worked around: a large pile-up at one value means listed_days cannot be read
    as company age. The arm only uses the YOUNG end, which is clean (min ~10 days, fine gradient).
    """
    ld = _num(live, "listed_days").dropna()
    top = ld.value_counts().iloc[0]
    assert top > 200, (
        "listed_days no longer piles up at a ceiling — if the vendor started supplying true "
        "listing dates, this note (and any code that avoided the upper range) can be revisited"
    )
    assert ld.min() < 200, "the young end lost its resolution; the arm depends on it"


# ── 3. What it must NOT have touched ─────────────────────────────────────────────────────

def test_an_established_company_on_the_size_arm_is_untouched(live):
    """Vodafone Idea is the false-EXONERATE half and is explicitly out of scope. If this starts
    failing, someone widened the arm beyond pre-listing and the known-issues entry needs rewriting.
    """
    ld = _num(live, "listed_days")
    pre = _num(live, "dilution_prelisting_baseline") == 1
    assert not (pre & (ld >= PRELISTING_BASELINE_DAYS)).any(), (
        "the pre-listing arm fired on a company listed longer ago than the threshold"
    )
    vi = live[live["name"].astype(str).str.contains("Vodafone Idea", case=False, na=False)]
    if len(vi):
        assert (_num(vi, "dilution_prelisting_baseline") == 0).all(), (
            "Vodafone Idea was caught by the pre-listing arm — it is 3998 days listed; this arm "
            "does not address the >=1.5x size-exoneration half of the bug"
        )


def test_missing_listed_days_does_not_trigger_the_arm(live):
    """Absent evidence is not evidence of a recent listing (the sentinel class, mirrored)."""
    from core.data_engine import compute_derived_signals
    from core import data_engine as de
    cols = set()
    for dct in (de.COMMON_COLS, de.RATIO_COLS, de.INCOME_COLS, de.BALANCE_COLS,
                de.CASHFLOW_COLS, de.SHAREHOLDING_COLS, de.TECHNICAL_COLS):
        cols |= set(dct.values())
    n = 6
    f = pd.DataFrame({c: [np.nan] * n for c in sorted(cols)})
    f["company_id"] = [f"NSE:T{i}" for i in range(n)]
    f["name"] = [f"Test Co {i}" for i in range(n)]
    f["equity_shares"], f["equity_shares_1yb"] = 150.0, 100.0   # +50% raw, would be Tier 3
    f["listed_days"] = np.nan                                    # unknown listing age
    with contextlib.redirect_stdout(_io.StringIO()):
        out = compute_derived_signals(f)
    assert (out["dilution_prelisting_baseline"] == 0).all(), (
        "a NaN listed_days triggered the pre-listing arm — unknown age must not buy an exemption"
    )


def test_missing_share_data_does_not_trigger_the_arm_either():
    """Found by mutation: dropping `shares_valid &` is INERT for dilution_flag, because
    `~shares_valid` matches first in the np.select cascade — but it silently mislabels the UI
    mirror column, claiming a pre-listing baseline for rows that have no share delta at all.
    A flag nobody can see going wrong is exactly what the mirror column exists to prevent.
    """
    from core.data_engine import compute_derived_signals
    from core import data_engine as de
    cols = set()
    for dct in (de.COMMON_COLS, de.RATIO_COLS, de.INCOME_COLS, de.BALANCE_COLS,
                de.CASHFLOW_COLS, de.SHAREHOLDING_COLS, de.TECHNICAL_COLS):
        cols |= set(dct.values())
    n = 6
    f = pd.DataFrame({c: [np.nan] * n for c in sorted(cols)})
    f["company_id"] = [f"NSE:T{i}" for i in range(n)]
    f["name"] = [f"Test Co {i}" for i in range(n)]
    f["listed_days"] = 30.0                       # genuinely just listed
    f["equity_shares"] = np.nan                   # but no share data to compare
    f["equity_shares_1yb"] = np.nan
    with contextlib.redirect_stdout(_io.StringIO()):
        out = compute_derived_signals(f)
    assert (out["dilution_prelisting_baseline"] == 0).all(), (
        "the arm claimed a pre-listing baseline on rows with NO share data — there is no delta "
        "to guard there, and the mirror column must not assert one"
    )
    assert (out["dilution_flag"] == 0).all(), "no share data must stay benefit-of-doubt"


# ── 4. The display must not contradict the engine ────────────────────────────────────────

def test_the_tearsheet_does_not_print_a_dilution_percentage_for_a_prelisting_row(live):
    """FOUND BY AUDIT, not by the suite. The engine classified Belrise as a pre-listing baseline
    (Tier 1, unmeasurable) while the tearsheet's _share_change_str went on printing
    "Dilution 48.6%", because that branch only fired for the >=1.5x corporate-action arm and
    Belrise is 1.4857x — under it. Engine and display said opposite things about the same row:
    the Fisher module/engine drift class. Both arms now route through the same branch.
    """
    import re
    import ui.ui_tearsheet as T

    def _render_text(row):
        """Drive the REAL renderer and read what it printed — a source scan would pass on a
        branch that exists but is unreachable (the trap this repo has paid for twice)."""
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
            T.render_verdict_scorecard(row)
        finally:
            T.st = real
        return re.sub(r"<[^>]+>", " ", " ".join(out))

    pre = _num(live, "dilution_prelisting_baseline") == 1
    assert pre.sum() > 0, "no pre-listing rows to render"
    row = live[pre].iloc[0]
    txt = _render_text(row)
    assert "pre-listing baseline" in txt.lower(), (
        f"a pre-listing row did not say so on the page; it rendered: {txt[:300]!r}"
    )
    assert not re.search(r"Dilution\s+\d", txt), (
        f"the tearsheet still prints a dilution PERCENTAGE for a row the engine declared "
        f"unmeasurable: {txt[:300]!r}"
    )
    # and an ordinary row must still get its percentage — the branch must not swallow everyone
    ordinary = live[(_num(live, "dilution_prelisting_baseline") == 0)
                    & (_num(live, "dilution_is_corporate_action") == 0)
                    & (_num(live, "dilution_pct") > 3.0)]
    assert len(ordinary), "no ordinary-dilution row to check against"
    t2 = _render_text(ordinary.iloc[0])
    assert re.search(r"Dilution\s+\d", t2), (
        f"an ordinary issuance row lost its dilution percentage: {t2[:300]!r}"
    )


def test_every_prelisting_row_has_the_counts_needed_to_explain_itself(live):
    """The replacement text shows raw share counts; if they are missing the row would fall back
    to a bare label, so check the data actually supports the explanation."""
    pre = _num(live, "dilution_prelisting_baseline") == 1
    have = _num(live, "equity_shares")[pre].notna() & (_num(live, "equity_shares_1yb")[pre] > 0)
    assert have.mean() > 0.95, (
        f"only {100*have.mean():.0f}% of pre-listing rows carry both share counts, so the "
        f"tearsheet would show a bare label instead of the numbers that justify it"
    )


def test_a_recently_listed_stock_shows_the_listed_badge(live):
    """The cause of "(pre-listing baseline)" must be visible, not just its effect. Driven through
    the REAL renderer, and checked in BOTH directions so the badge cannot fire universally."""
    import re
    import ui.ui_tearsheet as T

    def _render_text(row):
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
            T.render_stock_hero(row)          # the badge row lives on the hero, not the scorecard
        finally:
            T.st = real
        return re.sub(r"<[^>]+>", " ", " ".join(out))

    ld = _num(live, "listed_days")
    young = live[ld < PRELISTING_BASELINE_DAYS]
    old = live[ld >= PRELISTING_BASELINE_DAYS]
    assert len(young) > 50 and len(old) > 500, "not enough of both kinds to test"
    assert "Listed" in _render_text(young.iloc[0]), (
        "a recently listed stock does not say so on the page, so '(pre-listing baseline)' on the "
        "share row has no visible cause"
    )
    assert "Listed" not in _render_text(old.iloc[0]), (
        "an established company shows the recently-listed badge — listed_days is CENSORED at 3998 "
        "for 43% of the universe, so it must never be rendered as a general company-age figure"
    )


def test_the_badge_imports_the_bound_instead_of_retyping_it():
    """Re-deriving a threshold in the UI is the drift that produced the 'Dilution 48.6%'
    contradiction this card was fixed for."""
    src = _io.open(os.path.join(os.path.dirname(__file__), "..", "ui", "ui_tearsheet.py"),
                   encoding="utf-8").read()
    assert "PRELISTING_BASELINE_DAYS" in src, "the tearsheet no longer imports the shared bound"
    i = src.find("_listed_badge = \"\"")
    assert i > 0, "the recently-listed badge vanished"
    assert "730" not in src[i:i + 600], (
        "the pre-listing bound is hardcoded beside the badge instead of imported — one definition, "
        "or the UI and the engine drift apart again"
    )
