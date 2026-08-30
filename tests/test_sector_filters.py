"""
test_sector_filters.py
======================
Contract for the Market Pulse → Sectors filter row.

FIVE CONTROLS, TWO KINDS — and the difference is invisible on screen, which is why it is pinned
here rather than left to whoever edits this next:

    RE-AGGREGATING   Market-cap tier, Wealth Tier, Cyclicality tier (returned 2026-08-28 —
                     measured: 42 of 81 sectors hold >1 tier, so only a stock filter can answer
                     "sector rankings among Defensive stocks"; the 2026-08-27 swap had quietly
                     lost that view)
                     filter the STOCKS, so every average and % Qualify is recomputed
    ROW FILTERS      Capital phase, Min stocks/sector
                     hide whole sectors and leave the survivors' numbers untouched

REPLACED 2026-08-27 (user request): the re-aggregating slot held Cyclicality tier; it now holds
the Wealth Tier. Same semantics, same slot, so every behavioural contract below carries over.
Cyclicality filtering still exists in the Discovery sidebar (sb_cyc) — no capability was lost.

Four identical-looking selectboxes that behave in two different ways is a trap. The tests below
assert the behaviour, not the label.

WHY THE SIZE DIAL EXISTS, and why it is the important one. The floor was hardcoded at 5, and that
is precisely why the ranking was dominated by tiny sectors: an extreme % Qualify is easy at n=7
and near-impossible at n=96. Measured 2026-08-27 — 8 of the top 10 sectors held fewer than 12
stocks (median 9 against 19 universe-wide), and Glass & Glass Products ranked #1 at 88% on 8
stocks while carrying a below-average composite of 27.9. Raising the floor to 15 changes the top
six COMPLETELY: 0 of 6 in common. The tab's own tooltip already warned about this; now it can be
acted on.

DEFAULT STAYS 5, so the tab is byte-identical for anyone who does not touch the control.

Run with: pytest tests/test_sector_filters.py -v
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

_APP = os.path.join(os.path.dirname(__file__), "..", "app.py")


@pytest.fixture(scope="module")
def src():
    return _io.open(_APP, encoding="utf-8").read()


@pytest.fixture(scope="module")
def live():
    from core import run_scoring_pipeline
    with contextlib.redirect_stdout(_io.StringIO()):
        return run_scoring_pipeline(
            compute_derived_signals(coerce_numeric_columns(merge_datasets(load_all_csvs("local")))))


def _agg(d, min_n=5):
    """Mirror of the tab's aggregation, so these tests measure what it measures."""
    g = d.groupby("sector").agg(
        stocks=("name", "count"),
        pct_qualify=("gate_pass", lambda s: 100.0 * s.mean()),
        avg_composite=("composite_score", "mean"),
    )
    return (g[g["stocks"] >= min_n]
            .sort_values(["pct_qualify", "avg_composite"], ascending=False))


# -- 1. The controls exist, and the default changes nothing ---------------------------------
@pytest.mark.parametrize("key", ["mp_sec_cap", "mp_sec_wealth", "mp_sec_cyc", "mp_sec_phase", "mp_sec_minn"])
def test_each_control_is_present_with_its_own_key(src, key):
    assert f'key="{key}"' in src, f"the {key} control is gone"


def test_the_size_dial_defaults_to_the_old_hardcoded_floor(src):
    """Anyone who never touches it must see exactly the tab they saw before."""
    i = src.index('key="mp_sec_minn"')
    block = src[i - 400:i + 200]
    # 1 ADDED 2026-08-30 (user request: a show-everything floor). The REAL invariant is that the
    # DEFAULT still resolves to 5 -- anyone who never touches the dial sees the tab unchanged.
    assert "[1, 5, 10, 15, 20, 30]" in block, f"the size options changed: {block[:200]}"
    assert "index=1" in block, "the default no longer resolves to 5 -- untouched behaviour changed"
    assert [1, 5, 10, 15, 20, 30][1] == 5   # the index-1 option IS 5 (self-check of this pin)


def test_the_hardcoded_floor_is_gone(src):
    assert '_sec_stats["stocks"] >= _min_n' in src, "the floor is not driven by the control"
    assert '_sec_stats["stocks"] >= 5]' not in src, "the hardcoded 5 is still there"


# -- 2. THE SEMANTICS: which filters move the numbers, and which only hide rows --------------
def test_wealth_tier_is_a_stock_filter_that_re_aggregates(live):
    """The premise for the slot it inherited from cyclicality: wealth_tier varies WITHIN sectors,
    so slicing by it genuinely changes what a sector's average is computed over."""
    v = live.groupby("sector")["wealth_tier"].nunique()
    assert (v > 1).sum() > 10, (
        f"wealth_tier now varies within only {int((v > 1).sum())} sectors -- if it became a "
        f"sector-constant attribute it should be a ROW filter like capital phase, not a stock one"
    )
    base = _agg(live)
    sliced = _agg(live[live["wealth_tier"] == "BUY★"])
    common = base.index.intersection(sliced.index)
    assert len(common) > 5, "not enough overlap to compare"
    moved = (base.loc[common, "pct_qualify"] - sliced.loc[common, "pct_qualify"]).abs() > 0.01
    assert moved.any(), "slicing by wealth tier changed no sector's % Qualify -- it is not re-aggregating"


def test_the_wealth_filter_is_actually_applied_before_aggregation(src):
    """STRUCTURAL, because the data-property test above is blind to app.py: it slices the live
    frame itself, so gutting the filter application in the app leaves it green (a mutation run
    proved exactly that). This pins that the selection reaches _sec_src BEFORE the groupby."""
    i = src.index('key="mp_sec_wealth"')
    j = src.index("_sec_stats = _sec_src.groupby", i)
    between = src[i:j]
    assert '_sec_src[_sec_src["wealth_tier"] == _wt_sec]' in between, (
        "the Wealth Tier selection is no longer applied to _sec_src before aggregation -- the "
        "control renders but filters nothing"
    )


def test_capital_phase_is_a_row_filter_that_moves_no_number(live):
    """It is applied AFTER aggregation, so a surviving sector's figures must be identical."""
    v = live.groupby("sector")["sector_capital_phase"].nunique()
    assert (v > 1).sum() == 0, (
        f"sector_capital_phase now varies within {int((v > 1).sum())} sectors. The tab applies it "
        f"after aggregation precisely so this stays correct -- but the help text calling it a "
        f"sector attribute needs revisiting."
    )
    base = _agg(live)
    keep = set(live.loc[live["sector_capital_phase"].astype(str).str.contains("Hot"), "sector"])
    filtered = base[base.index.isin(keep)]
    assert len(filtered) > 0, "no Hot sector survives -- probe is stale"
    pd.testing.assert_frame_equal(filtered, base.loc[filtered.index], check_like=True)


def test_the_phase_filter_is_applied_after_aggregation(src):
    """Structural: filtering the stocks instead would part-filter a sector and skew its averages
    the moment phase stops being sector-constant."""
    i = src.index("_sec_stats = (_sec_stats[")
    block = src[i:i + 900]
    assert "sector_capital_phase" in block, "the phase filter moved out of the post-aggregation block"
    assert "_sec_stats.index.isin" in block, "the phase filter no longer selects whole sectors by index"


# -- 3. The size dial must actually change the answer ----------------------------------------
def test_raising_the_minimum_changes_which_sectors_lead(live):
    """The whole reason the control exists. If this stops being true the small-sample bias has
    gone and the dial is merely decorative."""
    top5 = list(_agg(live, 5).head(6).index)
    top15 = list(_agg(live, 15).head(6).index)
    overlap = len(set(top5) & set(top15))
    assert overlap <= 3, (
        f"raising the floor from 5 to 15 leaves {overlap} of the top 6 unchanged; the small-sample "
        f"dominance this control addresses appears to have gone -- re-measure before trusting the "
        f"help text, which claims the ranking changes completely"
    )


def test_small_sectors_really_do_dominate_the_default_view(live):
    """The claim in the help text, checked against live data."""
    top10 = _agg(live, 5).head(10)
    assert top10["stocks"].median() < _agg(live, 5)["stocks"].median(), (
        "the top-ranked sectors are no longer smaller than typical -- the help text's premise is stale"
    )


def test_the_floor_never_admits_a_sector_below_it(live):
    for n in (1, 5, 10, 15, 20, 30):
        g = _agg(live, n)
        if g.empty:
            continue
        assert g["stocks"].min() >= n, f"a sector below {n} stocks survived the {n} floor"


# -- 4. The two kinds are explained to the reader --------------------------------------------
def test_the_help_text_distinguishes_the_two_behaviours(src):
    """Four identical-looking selectboxes behaving in two ways needs saying, not guessing."""
    i = src.index('key="mp_sec_cap"')
    block = src[i - 900:i + 5200]      # widened 2026-08-28: the cyclicality control joined the row
    assert block.lower().count("re-aggregat") >= 3, (
        "the three re-aggregating controls no longer all say that they recompute the averages"
    )
    assert block.lower().count("hides rows") >= 2, (
        "the row filters no longer say that they only hide rows"
    )

def test_the_caption_follows_the_size_dial(src):
    """The caption hardcoded "≥5 stocks" while the floor became a control, so it contradicted its
    own dial the moment that moved to 15. Verified in the browser before it was fixed."""
    # SCOPED TO THE RENDERED STRING, not the whole file. The first version scanned all of app.py
    # for the literal and matched the CODE COMMENT that explains this very bug -- the fourth
    # substring-scan-over-source mistake of the session. Prose naming a banned string is not the
    # banned string.
    i = src.index("_sec_cap_ph.markdown(")
    block = src[i:src.index("unsafe_allow_html=True,", i)]
    assert "{_min_n}" in block, "the caption no longer interpolates the chosen minimum"
    assert "≥5" not in block, f"the caption hardcodes the old floor: {block[:160]}"


# ── 5. 💹 tier-share column (2026-08-28) — one column, follows the filter ────────────────────
def test_tier_share_base_is_captured_before_the_wealth_filter(src):
    """THE 100% TRAP, pinned structurally: the share column's denominator must be the
    pre-wealth-filter roster. Captured after the filter, selecting any tier makes every
    surviving row read a meaningless 100% — the design review caught this before it shipped."""
    i = src.index("_sec_share_base = _sec_src")
    j = src.index('_sec_src = _sec_src[_sec_src["wealth_tier"] == _wt_sec]')
    assert i < j, "the share base is captured AFTER the wealth filter — the 100% trap is live"
    assert '_sec_share_base.groupby("sector")["wealth_tier"]' in src, (
        "the share aggregation no longer reads the pre-filter base — capturing it means nothing "
        "if the groupby runs on the filtered frame"
    )


def test_the_100_percent_trap_is_real(live):
    """Self-verifying: the naive post-filter share IS 100% everywhere, and the pre-filter one
    genuinely varies — so the structural pin above defends a measured hazard, not a style."""
    filt = live[live["wealth_tier"] == "AVOID"]
    trap = filt.groupby("sector")["wealth_tier"].apply(lambda s: 100.0 * (s == "AVOID").mean())
    assert (trap == 100.0).all(), "the trap probe went stale — post-filter share is no longer trivial"
    full = live.groupby("sector")["wealth_tier"].apply(lambda s: 100.0 * (s == "AVOID").mean())
    assert full.std() > 1.0, "the full-roster share no longer varies across sectors"


def test_tier_share_is_exact_match_never_contains(src, live):
    """'BUY' ⊂ 'BUY★' (the QGLP⊂SQGLP class). Structural: exact equality in the app. Live: the
    two matchers genuinely disagree, so the pin has teeth."""
    i = src.index("_sec_share_tier")
    seg = src[i:i + 2500]
    assert "(s == _sec_share_tier)" in seg, "the share is no longer an exact-equality match"
    assert ".str.contains" not in seg, "a contains-match crept into the tier share"
    star = live.groupby("sector")["wealth_tier"].apply(lambda s: 100.0 * (s == "BUY★").mean())
    loose = live.groupby("sector")["wealth_tier"].apply(
        lambda s: 100.0 * s.astype(str).str.contains("BUY", regex=False).mean())
    assert (loose - star).max() > 1.0, "BUY and BUY★ no longer diverge — re-check before relaxing"


def test_tier_share_defaults_to_buy_star_and_the_label_follows(src):
    """All → BUY★ (top of the forward-validated monotonic ladder); a selected tier renames the
    column so its semantics are self-announcing."""
    assert '_sec_share_tier = _wt_sec if _wt_sec != "All" else "BUY★"' in src
    assert 'f"💹 {_sec_share_tier} %"' in src, "the column label no longer follows the filter"


def test_tier_share_is_not_a_sort_key(src):
    """A column, not the ranking: Sectors keeps % Qualify → Score."""
    i = src.index('_sec_stats = (_sec_stats[_sec_stats["stocks"] >= _min_n]')
    assert "pct_tier" not in src[i:i + 300], "pct_tier became a sort key"


# ── 6. Cyclicality tier — the returned third re-aggregator (2026-08-28) ──────────────────────
def test_cyclicality_really_does_vary_within_sectors(live):
    """The premise that makes this a STOCK filter (and made the 2026-08-27 swap a silent
    capability loss): tiers cross sector lines. Measured at build: 42 of 81. If this collapses,
    the filter should become a row filter and this file rewritten."""
    v = live.groupby("sector")["cyclicality_tier"].nunique()
    assert (v > 1).sum() > 10, (
        f"cyclicality varies within only {int((v > 1).sum())} sectors — the stock-filter premise died"
    )


def test_cyclicality_filter_is_applied_before_the_share_base_and_the_groupby(src):
    """Ordering is the whole contract: cap → cyclicality → [💹 share base] → wealth → groupby.
    Applied after the share base, Defensive × BUY★ would silently show the ALL-stock share."""
    i = src.index('_sec_src[_sec_src["cyclicality_tier"] == _cyc_sec]')
    j = src.index("_sec_share_base = _sec_src")
    k = src.index('_sec_src = _sec_src[_sec_src["wealth_tier"] == _wt_sec]')
    m = src.index("_sec_stats = _sec_src.groupby")
    assert i < j < k < m, "the cyclicality filter is out of order in the filter chain"


def test_cyclicality_slice_re_aggregates(live):
    """Behavioural: slicing to one tier genuinely changes sector averages (it is not a row hide)."""
    base = _agg(live)
    sliced = _agg(live[live["cyclicality_tier"] == "Defensive"])
    common = base.index.intersection(sliced.index)
    assert len(common) > 5, "not enough overlap to compare"
    moved = (base.loc[common, "avg_composite"] - sliced.loc[common, "avg_composite"]).abs() > 0.01
    assert moved.any(), "slicing by cyclicality changed no sector average — it is not re-aggregating"


def test_the_caption_points_to_the_industry_tab(src):
    """The dispersion pointer (measured: up to 50 points of industry spread inside one sector):
    the Sectors caption must hand the reader to 🏭 Industry."""
    i = src.index("_sec_cap_ph.markdown(")
    block = src[i:src.index("unsafe_allow_html=True,", i)]
    assert "🏭 Industry" in block, "the Sectors caption no longer points to the Industry tab"
    assert "50 points" in block, "the dispersion magnitude vanished from the caption"
