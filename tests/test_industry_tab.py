"""
test_industry_tab.py
====================
Contract for the Market Pulse → 🏭 Industry tab (added 2026-08-28).

WHY THE TAB EXISTS — the one number that justifies it. `sector` has 81 values; `industry` has 355.
Averaging to the sector hides real dispersion: the six sizeable industries inside Pharmaceuticals
run 18.1 → 51.3 on average composite, a 33-point spread that the Sectors tab reports as a single
number. Of the 76 industries holding ≥8 stocks, 20 sit more than 5 points from their parent
sector's average (Pharma - MNC bulk Drugs +22.7; Auto Ancillaries - Gears −12.9). That gap IS the
tab's subject, so it is a column — `delta_vs_sector` — and the table sorts by it.

THREE HAZARDS MEASUREMENT CAUGHT, each pinned below because each is invisible on screen:

  1. INDUSTRY IS NOT NESTED INSIDE SECTOR. 136 of 355 industries span more than one sector. A
     drill-down that assumes a tree is wrong. The tab shows the DOMINANT sector (modal, with a
     deterministic tie-break) and marks the impure ones with "~". Median purity is 0.91 but the
     minimum is 0.38, so the marker is not decorative.

  2. A DEGENERATE DELTA MUST BE NaN, NEVER 0.0. Three of the 76 industries are the only industry
     in their dominant sector, so industry-average and sector-average are the SAME NUMBER and the
     delta is exactly zero by construction. Rendering that as 0.0 says "perfectly average"; the
     truth is "there is nothing to compare against". This is CLAUDE.md's semantic-truth principle
     (never inject a sentinel where the honest answer is missing), and it is the single most
     likely thing a future edit breaks — `a - b` is the obvious implementation and it is wrong.

  3. `ret_vs_industry_1y` IS REJECTED, deliberately. It looks like the perfect column for this tab
     (an industry-relative return the engine already computes), but 7 of the 76 industries have
     ZERO non-null coverage of it and it ranges −1307…+585. Aggregating it would produce a
     confident-looking number backed by nothing. The test below re-measures the coverage hole so
     the rejection cannot quietly stop being true.

THERE IS NO SIZE FLOOR — every industry shows, down to n=1. It went dial (5..30) → fixed 8 → none,
each step on the user's explicit call, the last made twice after the cost was measured and put in
front of them. The cost is real: the median industry holds 3 stocks against 19 for sectors, and
sorting by Δ does not neutralise that — Δ is MORE exposed than % Qualify, being unbounded where
% Qualify is capped at 0–100. With no floor, 5 of the top 10 rows are single-stock industries and
the leader is Auto Ancillaries - Seats (n=1, +32). Count sits first in the table as the mitigation.
That trade is pinned by test_small_samples_lead_the_table_by_design, which RECORDS it rather than
objecting to it — a floor coming back is a reversal to argue for, not a fix.

BASIS RULE. When a re-aggregating filter narrows the stocks, the sector baseline must be recomputed
over the SAME survivors. Comparing a Smallcap-only industry average against an all-cap sector
average is the cross-year-basis defect wearing different clothes: both terms of a difference must
come from one source. Pinned behaviourally below.

Run with: pytest tests/test_industry_tab.py -v
"""

import ast
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

from data_engine import (coerce_numeric_columns, compute_derived_signals, load_all_csvs,
                         merge_datasets)

_APP = os.path.join(os.path.dirname(__file__), "..", "app.py")

# THE TAB HAS NO SIZE FLOOR — every industry shows, down to n=1. It went dial (5..30) → fixed 8 →
# none, each step on the user's explicit call. NO_FLOOR is what the table actually applies;
# SIZEABLE is only an analysis threshold these tests use to reason about well-sampled industries.
NO_FLOOR = 1
SIZEABLE = 8
# Below this share of its stocks sitting in one sector, an industry is labelled impure.
PURITY_MARK = 0.8


@pytest.fixture(scope="module")
def src():
    return _io.open(_APP, encoding="utf-8").read()


@pytest.fixture(scope="module")
def block(src):
    """Just the Industry tab's own source — so a check cannot accidentally pass on the Sectors
    tab's very similar code sitting directly above it."""
    start = src.index("# ══ Industry ══")
    end = src.index("# ━━━", start)
    return src[start:end]


@pytest.fixture(scope="module")
def live():
    from core import run_scoring_pipeline
    with contextlib.redirect_stdout(_io.StringIO()):
        return run_scoring_pipeline(
            compute_derived_signals(coerce_numeric_columns(merge_datasets(load_all_csvs("local")))))


def _agg(df, floor=NO_FLOOR):
    """An INDEPENDENT re-implementation of the tab's aggregation, written from the spec rather
    than copied from app.py — so it can disagree with the app and catch it."""
    d = df[["industry", "sector", "name", "composite_score", "gate_pass"]].copy()
    d["industry"] = d["industry"].astype(str).str.strip()
    g = d.groupby("industry").agg(stocks=("name", "count"),
                                  avg_composite=("composite_score", "mean"))
    g = g[g["stocks"] >= floor]
    pair = (d.groupby(["industry", "sector"]).size().rename("n").reset_index()
            .sort_values(["industry", "n", "sector"], ascending=[True, False, True]))
    dom = pair.drop_duplicates("industry").set_index("industry")
    g["dom_sector"] = dom["sector"].reindex(g.index)
    g["purity"] = dom["n"].reindex(g.index) / g["stocks"]
    base = d.groupby("sector")["composite_score"].mean()
    n_ind = d.groupby("sector")["industry"].nunique()
    comparable = g["dom_sector"].map(n_ind) > 1
    g["delta"] = np.where(comparable, g["avg_composite"] - g["dom_sector"].map(base), np.nan)
    return g


# ── 1. The tab is wired, and appended rather than inserted ──────────────────────────────────
def test_industry_tab_is_the_sixth_tab(src):
    """Appended AFTER Sectors on purpose: every `with _mp_tabs[i]` body binds by index, so
    inserting anywhere else silently renders existing content into the wrong tab."""
    tree = ast.parse(src)
    labels = None
    for n in ast.walk(tree):
        if (isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "_mp_tabs" for t in n.targets)
                and isinstance(n.value, ast.Call)
                and getattr(n.value.func, "attr", "") == "tabs"):
            labels = [e.value for e in n.value.args[0].elts if isinstance(e, ast.Constant)]
    assert labels is not None, "_mp_tabs assignment not found"
    assert labels[4] == "📈 Sectors", f"Sectors moved off index 4: {labels}"
    assert labels[5] == "🏭 Industry", f"Industry is not at index 5: {labels}"
    assert "with _mp_tabs[5]:" in src, "the Industry tab has no renderer body"


def test_industry_tab_reads_the_industry_column(block):
    assert '"industry"' in block, "the tab does not reference the industry column at all"


# ── 2. The degenerate delta: NaN, never 0.0 ─────────────────────────────────────────────────
def test_a_degenerate_delta_case_actually_exists_live(live):
    """SELF-VERIFYING. If no industry is alone in its sector any more, the guard below is dead
    weight and this test says so instead of silently passing forever."""
    g = _agg(live)
    d = live.copy()
    d["industry"] = d["industry"].astype(str).str.strip()
    n_ind = d.groupby("sector")["industry"].nunique()
    solo = g["dom_sector"].map(n_ind) == 1
    assert int(solo.sum()) >= 1, (
        "no industry is the sole industry of its sector any more — re-measure before deleting the "
        "guard, and delete the guard's justification with it"
    )


def test_degenerate_delta_is_nan_not_zero(live):
    """The defect this pins: `avg_industry - avg_sector` is the obvious implementation, and for a
    sector holding exactly one industry it returns 0.0 — which reads as 'perfectly average' when
    the truth is 'no comparison exists'. A sentinel standing in for missing evidence."""
    g = _agg(live)
    d = live.copy()
    d["industry"] = d["industry"].astype(str).str.strip()
    n_ind = d.groupby("sector")["industry"].nunique()
    solo = g["dom_sector"].map(n_ind) == 1
    assert g.loc[solo, "delta"].isna().all(), (
        f"these industries are alone in their sector and must show no delta at all: "
        f"{sorted(g.index[solo & g['delta'].notna()])}"
    )
    assert not (g.loc[solo, "delta"] == 0.0).any(), "a 0.0 sentinel leaked into a degenerate row"


def test_the_app_guards_the_degenerate_case(block):
    """The behavioural tests above run against this file's own re-implementation. This one checks
    app.py actually carries the guard, so the two cannot drift apart."""
    assert "nunique" in block, (
        "app.py computes no per-sector industry count, so it cannot know which deltas are "
        "degenerate — the 0.0 sentinel is back"
    )
    assert "np.nan" in block, "no NaN is ever emitted for the incomparable rows"


def test_delta_is_not_uniformly_nan(live):
    """The mirror failure: guarding so hard that every row loses its delta."""
    g = _agg(live)
    assert g["delta"].notna().sum() >= 20, "the delta column is mostly empty — the guard is too broad"


# ── 3. The basis rule: baseline recomputed over the same survivors ──────────────────────────
def test_delta_baseline_uses_the_filtered_subset(live):
    """Both terms of the difference must come from one population. Filtering the stocks to one
    cap tier and comparing against an ALL-cap sector average is the cross-year-basis defect in a
    different costume: the gap it reports is mostly the cap effect, not the industry."""
    sub = live[live["market_category"] == "Small Cap"] if "market_category" in live.columns else live
    assert len(sub) >= 200, f"only {len(sub)} Small Cap rows — the cap tier labels changed"
    g_sub = _agg(sub, floor=5)
    d = sub.copy()
    d["industry"] = d["industry"].astype(str).str.strip()
    base_sub = d.groupby("sector")["composite_score"].mean()
    base_all = live.groupby("sector")["composite_score"].mean()
    shared = [s for s in base_sub.index if s in base_all.index]
    assert (base_sub[shared] - base_all[shared]).abs().max() > 0.5, (
        "the filtered and unfiltered sector baselines are identical, so this test proves nothing"
    )
    row = g_sub[g_sub["delta"].notna()].index[0]
    expect = g_sub.loc[row, "avg_composite"] - base_sub[g_sub.loc[row, "dom_sector"]]
    assert abs(g_sub.loc[row, "delta"] - expect) < 1e-9


def test_app_recomputes_the_baseline_after_filtering(block):
    """Source-level: the sector baseline must be grouped off the FILTERED frame, not off `df`."""
    m = re.search(r'(\w+)\.groupby\("sector"\)\["composite_score"\]', block)
    assert m, "no per-sector composite baseline is computed in the Industry tab"
    assert m.group(1) != "df", (
        "the sector baseline is computed from the unfiltered `df` while the industry averages come "
        "from the filtered frame — the two terms of the delta no longer share a population"
    )


# ── 4. Non-nesting: dominant sector, deterministic, marked when impure ──────────────────────
def test_industries_really_do_span_multiple_sectors(live):
    """SELF-VERIFYING: the whole dominant-sector apparatus exists because of this fact."""
    d = live.copy()
    d["industry"] = d["industry"].astype(str).str.strip()
    spanning = (d.groupby("industry")["sector"].nunique() > 1).sum()
    assert spanning >= 50, (
        f"only {spanning} industries span >1 sector; if industry has become a clean child of "
        f"sector, the '~' marker and the dominant-sector language should be simplified away"
    )


def test_impure_industries_exist_and_are_marked(live, block):
    g = _agg(live)
    impure = g[g["purity"] < PURITY_MARK]
    assert len(impure) >= 5, (
        f"only {len(impure)} industries fall below {PURITY_MARK} purity — re-measure before "
        f"keeping the marker"
    )
    assert '"~' in block or "'~" in block, (
        "no impurity marker in the source: an industry whose stocks sit 38% in its labelled sector "
        "would be displayed as though it belonged there cleanly"
    )


def test_dominant_sector_tie_break_is_deterministic(block):
    """CLAUDE.md determinism mandate. A modal pick with an unsorted tie-break reorders between
    processes (PYTHONHASHSEED), so the displayed parent sector would flicker."""
    assert "drop_duplicates" in block, "no single-row-per-industry reduction found"
    assert "ascending=[True, False, True]" in block, (
        "the dominant-sector pick has no deterministic tie-break — sort by (industry, count desc, "
        "sector asc) so an exact tie always resolves the same way"
    )


# ── 5. The rejected column ─────────────────────────────────────────────────────────────────
def test_ret_vs_industry_is_not_aggregated(block):
    assert "ret_vs_industry" not in block, (
        "ret_vs_industry_* was aggregated into the industry table. It is the most tempting column "
        "for this tab and the least trustworthy: see the coverage test below."
    )


def test_the_reason_ret_vs_industry_was_rejected_is_still_true(live):
    """SELF-VERIFYING: if the coverage hole is ever filled, this fails and the column becomes fair
    game — the rejection is evidence-based, not a taste."""
    d = live.copy()
    d["industry"] = d["industry"].astype(str).str.strip()
    big = d["industry"].value_counts()
    big = big[big >= SIZEABLE].index
    cov = d[d["industry"].isin(big)].groupby("industry")["ret_vs_industry_1y"].apply(
        lambda s: s.notna().mean())
    assert int((cov == 0).sum()) >= 1, (
        "every sizeable industry now has some ret_vs_industry_1y coverage — remeasure; the column "
        "may now be safe to aggregate"
    )


# ── 6. The size floor ──────────────────────────────────────────────────────────────────────
def test_there_is_no_size_floor(block):
    """WHAT THE TAB DOES NOW, and the history so nobody re-derives it. It shipped with a
    5..30 selectbox mirroring the Sectors tab's; that became a fixed 8; the floor then went
    entirely. Every step was the user's explicit call, the last one made twice.

    So a floor reappearing is not a bug fix — it is a reversal, and it should have to argue with
    test_small_samples_lead_the_table_by_design below, which records exactly what the floor bought
    and what it cost."""
    assert "_IND_MIN_N" not in block, "a size-floor constant is back"
    assert "Min stocks / industry" not in block, "the retired dial is back"
    assert "mp_ind_minn" not in block, "the retired dial's widget key is back"
    assert not re.search(r'\["stocks"\]\s*>=', block), "a >= stocks row filter is back"


def test_every_industry_survives_to_the_table(live):
    """No floor means no industry is hidden -- including the single-stock ones."""
    d = live.copy()
    d["industry"] = d["industry"].astype(str).str.strip()
    d = d[~d["industry"].isin(["", "nan", "None"])]
    assert len(_agg(live)) == d["industry"].nunique(), "some industries are being dropped"
    assert int((_agg(live)["stocks"] == 1).sum()) >= 1, (
        "no single-stock industry survives -- something is still filtering by size"
    )


def test_small_samples_lead_the_table_by_design(live):
    """THE ACCEPTED COST, pinned so it stays a known trade rather than becoming a surprise.

    Sorting by Δ does NOT neutralise small samples -- an earlier comment in app.py claimed it did
    and measurement refuted it. Δ is MORE exposed than % Qualify, because % Qualify is bounded
    0-100 while Δ is unbounded. With no floor, the leaders are one- and two-stock industries whose
    Δ is one company's score minus a sector average.

    This test does not object to that. It records it, and it fails if the shape changes enough
    that the reasoning behind the decision no longer holds -- at which point the choice deserves
    re-making rather than inheriting."""
    t = _agg(live).sort_values("delta", ascending=False, na_position="last")
    top = t.head(10)
    assert int((top["stocks"] <= 2).sum()) >= 3, (
        f"only {int((top['stocks'] <= 2).sum())} of the top 10 are 1-2 stock industries. The "
        f"small-sample exposure this tab knowingly accepts has changed shape -- remeasure before "
        f"trusting either this comment or app.py's."
    )
    floored_top = set(_agg(live, floor=SIZEABLE)
                      .sort_values("delta", ascending=False, na_position="last").head(10).index)
    assert len(set(top.index) & floored_top) <= 3, (
        "the unfloored and well-sampled leaders now largely agree, so the no-floor choice costs "
        "little -- worth recording, but check the numbers in app.py's comment still hold"
    )


def test_count_is_the_mitigation_and_sits_up_front(src):
    """With no floor, Count is the ONLY thing standing between a reader and a one-stock Δ. It has
    to be visible without scrolling, which means near the left edge."""
    i = src.index("_ind_order = [c for c in [")
    cols = re.findall(r'"([a-z_]+)"', src[i:src.index("]", src.index("[c for c in [", i) + 14) + 1])
    assert cols.index("stocks") == 0, f"Count moved to position {cols.index('stocks')}"


def test_industries_are_smaller_than_sectors(live):
    """SELF-VERIFYING: the fact underneath every small-sample caveat on this tab."""
    d = live.copy()
    d["industry"] = d["industry"].astype(str).str.strip()
    assert d["industry"].value_counts().median() < d["sector"].value_counts().median(), (
        "industries are no longer smaller than sectors -- the caveats need re-justifying"
    )
    assert len(_agg(live)) >= 25, "too few industries for a useful tab"


# ── 7. Layout + house rules ────────────────────────────────────────────────────────────────
def test_the_wide_sector_column_comes_last(src):
    """Same invariant tests/test_market_pulse_columns.py pins for the other tables: the columns
    that make the tab's point precede the wide context one. Sector names here run to
    'Infrastructure Developers & Operators'."""
    i = src.index("_ind_order = [c for c in [")
    cols = re.findall(r'"([a-z_]+)"', src[i:src.index("]", src.index("[c for c in [", i) + 14) + 1])
    assert "delta_vs_sector" in cols, "the tab's headline column is not in the table"
    assert cols.index("delta_vs_sector") <= 3, (
        f"Δ vs Sector sits at position {cols.index('delta_vs_sector')} and will be clipped at the "
        f"right edge — it is the reason the tab exists"
    )
    assert cols.index("dom_sector") == len(cols) - 1, "the wide sector column must come last"


def test_table_sorts_by_delta_with_missing_last(block):
    """The user's choice: rank by distance from the parent sector. Rows with no comparison carry
    no signal, so they sink rather than floating to the top of a descending sort."""
    calls = re.findall(r"sort_values\((.{0,200}?)\)", block, re.S)
    assert calls, "no sort_values call in the Industry tab"
    ranked = [c for c in calls if "delta_vs_sector" in c]
    assert ranked, f"nothing sorts by the delta; sorts found: {calls}"
    assert all("na_position=\"last\"" in c for c in ranked), (
        "incomparable rows must sort LAST. In a descending sort pandas puts NaN first by default, "
        "so the rows with no comparison would head the table."
    )


def test_no_rowwise_iteration(block):
    """CLAUDE.md absolute vectorization mandate."""
    for banned in ("apply(axis=1)", "iterrows", "itertuples", "for _, row in"):
        assert banned not in block, f"row-wise iteration ({banned}) in the Industry tab"


def test_no_state_mutating_widgets_beyond_selectboxes(block):
    """Market Pulse owns no session state beyond its own widget keys. st.button/slider would
    write state app.py owns."""
    for banned in ("st.button(", "st.slider(", "st.number_input("):
        assert banned not in block, f"{banned} in the Industry tab"


def test_widget_keys_are_unique_to_this_tab(src, block):
    """A key collision with the Sectors tab would make one tab's control silently drive the
    other's — both tabs render on every run."""
    keys = re.findall(r'key="(\w+)"', block)
    assert len(keys) == 2, (
        f"expected exactly the two RE-AGGREGATING filters (market-cap, wealth tier), found {keys}. "
        f"The min-stocks dial was retired 2026-08-28; a third widget means it came back or a new "
        f"one arrived unexamined."
    )
    for k in keys:
        assert k.startswith("mp_ind_"), f"key {k!r} is not namespaced to the Industry tab"
        assert src.count(f'key="{k}"') == 1, f"key {k!r} is used more than once in app.py"


def test_missing_industry_column_is_handled(block):
    """Every other Market Pulse section degrades rather than crashing when a column is absent."""
    assert 'if "industry" not in' in block or '"industry" in df.columns' in block, (
        "no guard for a frame without an industry column"
    )
