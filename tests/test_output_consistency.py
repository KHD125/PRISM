"""
test_output_consistency.py
==========================
Output-consistency contract — the "rank-bug class". The rank bug (rank disagreeing with the
displayed composite_score) was an OUTPUT-consistency defect that CLAUDE.md invariants and the
mechanical contract tests did not cover. A live audit confirmed no other members of the class
exist today; this test LOCKS the whole class so none can silently regress.

Each test pins one cross-field relationship that must hold on the final, post-penalty,
post-verdict frame. Expectations for tier/label/emoji are derived from config.CONVICTION_TIERS
(the SAME source the engine uses) so the test can never drift from the engine; the forensic
multiplier is checked by RELATIONSHIP (valid set + monotonic), not by copying its thresholds.

Data-gated like test_ui_smoke / test_fisher_contract: skips cleanly on a code-only checkout.
Complements (does not duplicate) test_verdict_engine.test_rank_tracks_post_penalty_composite —
that one is synthetic; this pins the class on the real ~2,107-stock universe.
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

from config import CONVICTION_TIERS

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Other Resources", "CSV Data")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(_DATA_DIR),
    reason="Local CSV data not present (code-only checkout) — output-consistency audit needs real data",
)

_DISPLAYED = ["rank", "composite_score", "quality_score", "momentum_score",
              "conviction_tier", "tier_label", "tier_emoji"]
_DEFAULT_TIER = min(CONVICTION_TIERS, key=lambda t: t["min"])["tier"]   # config-derived (== 5)


@pytest.fixture(scope="module")
def scored() -> pd.DataFrame:
    """Final frame through the canonical locked pipeline (forensic → scoring → penalty → verdict)."""
    from core.data_engine import (load_all_csvs, merge_datasets,
                                  coerce_numeric_columns, compute_derived_signals)
    from core import run_scoring_pipeline

    with contextlib.redirect_stdout(io.StringIO()):
        df = load_all_csvs("local")
        df = merge_datasets(df)
        df = coerce_numeric_columns(df)
        df = compute_derived_signals(df)
        df = run_scoring_pipeline(df)   # compute_forensic_signals → run_full_scoring → penalty → verdict
    return df


def test_no_nan_in_displayed_fields(scored):
    """Every per-stock displayed field must be populated — a NaN here is a visible UI bug."""
    nan_counts = {c: int(scored[c].isna().sum()) for c in _DISPLAYED if c in scored.columns}
    assert all(v == 0 for v in nan_counts.values()), f"NaN leaked into displayed fields: {nan_counts}"


def test_core_scores_within_0_100(scored):
    for c in ("composite_score", "quality_score", "momentum_score"):
        lo, hi = scored[c].min(), scored[c].max()
        assert lo >= -1e-6 and hi <= 100 + 1e-6, f"{c} outside [0,100]: min={lo}, max={hi}"


def test_rank_is_permutation_and_tracks_composite(scored):
    """rank must be a 1..N permutation AND sort-by-rank must give non-increasing composite_score
    (this is the exact rank bug, pinned on the full real universe)."""
    n = len(scored)
    assert sorted(scored["rank"].tolist()) == list(range(1, n + 1)), "rank is not a 1..N permutation"
    by_rank = scored.sort_values("rank")
    assert (by_rank["composite_score"].diff().dropna() <= 1e-9).all(), (
        "composite_score INCREASES as rank worsens — rank no longer tracks the displayed score"
    )


def test_conviction_tier_matches_composite_band(scored):
    """conviction_tier must equal the CONVICTION_TIERS band the POST-penalty composite falls into."""
    conds = [scored["composite_score"] >= t["min"] for t in CONVICTION_TIERS]
    expected = np.select(conds, [t["tier"] for t in CONVICTION_TIERS], default=_DEFAULT_TIER)
    mismatch = int((scored["conviction_tier"].values != expected).sum())
    assert mismatch == 0, f"{mismatch} stocks: conviction_tier disagrees with its composite band"


def test_conviction_tiers_are_descending_by_min():
    """np.select (in scoring_engine + forensic_engine) returns the FIRST matching band, so
    CONVICTION_TIERS MUST be ordered by descending `min` (85,70,55,40,0). Reorder it and a score of
    90 would match the first `>=0` band → mislabeled Tier 5 — and test_conviction_tier_matches_
    composite_band would NOT catch it, because it replays the SAME np.select over config order
    (engine and test would be wrong together). This pins the ordering ORDER-INDEPENDENTLY."""
    mins = [t["min"] for t in CONVICTION_TIERS]
    assert mins == sorted(mins, reverse=True), (
        f"CONVICTION_TIERS must be ordered by descending 'min' for np.select first-match to be "
        f"correct; got {mins}"
    )


def test_tier_label_and_emoji_match_tier(scored):
    label_map = {t["tier"]: f"{t['emoji']} {t['label']}" for t in CONVICTION_TIERS}
    emoji_map = {t["tier"]: t["emoji"] for t in CONVICTION_TIERS}
    exp_label = scored["conviction_tier"].map(label_map).astype(str)
    exp_emoji = scored["conviction_tier"].map(emoji_map).astype(str)
    assert (scored["tier_label"].astype(str).values == exp_label.values).all(), "tier_label != conviction_tier mapping"
    assert (scored["tier_emoji"].astype(str).values == exp_emoji.values).all(), "tier_emoji != conviction_tier mapping"


def test_forensic_multiplier_monotonic_in_red_flags(scored):
    """Asymmetric-penalty relationship: the multiplier is one of the valid steps and NEVER rises
    as red_flag_count rises (checked by behavior, not by copying the engine's thresholds)."""
    vals = set(np.round(scored["forensic_multiplier"].dropna().unique(), 6))
    assert vals <= {1.0, 0.9, 0.75, 0.5}, f"unexpected forensic_multiplier values: {vals}"
    by_flags = scored.groupby("red_flag_count")["forensic_multiplier"].max().sort_index()
    assert (by_flags.diff().dropna() <= 1e-9).all(), (
        f"forensic_multiplier increases with red_flag_count (non-monotonic): {by_flags.to_dict()}"
    )


def test_verdict_aligns_with_score(scored):
    """Aggregate sanity (verdict has its own logic): median composite must order BUY > WATCH > AVOID."""
    med = scored.groupby("verdict_direction")["composite_score"].median()
    assert {"BUY", "WATCH", "AVOID"} <= set(med.index), f"missing a verdict bucket: {set(med.index)}"
    assert med["BUY"] > med["WATCH"] > med["AVOID"], f"verdict medians not ordered BUY>WATCH>AVOID: {med.to_dict()}"
