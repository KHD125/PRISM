"""Contract tests for the verdict synthesis layer (core/verdict_engine.py).

Guards: display-only (no scoring mutation), valid 3-state direction, asymmetric vetoes
(forensic / gruesome / schilit / governance / timing), confidence separation, and
crash-proof on minimal frames. Calibration is asserted against the live pipeline.
"""
import numpy as np
import pandas as pd
import pytest

from core.verdict_engine import compute_verdict


_DEFAULTS = {
    "conviction_tier": 1, "composite_score": 90.0,
    "forensic_score": 90.0, "schilit_pass": 1, "red_flag_count": 1,
    "corporate_class": "🏆 GREAT",
    "quality_score": 90.0, "growth_score": 80.0,
    "expected_excess_return": 20.0, "pe": 20.0, "fair_pe_qglp": 30.0,
    "buy_zone_label": "Accumulate", "marks_score": 70.0,
    "governance_risk_multiplier": 1.0, "data_coverage_pct": 90.0,
}


def _frame(**overrides):
    """One-row frame with all verdict inputs present; override any field per-test."""
    row = {**_DEFAULTS, **overrides}
    return pd.DataFrame([row])


# ── Direction validity + the happy path ──
def test_direction_is_always_valid():
    df = compute_verdict(_frame())
    assert df["verdict_direction"].iloc[0] in {"BUY", "WATCH", "AVOID"}


def test_clean_elite_is_buy():
    df = compute_verdict(_frame())
    assert df["verdict_direction"].iloc[0] == "BUY"
    assert df["verdict_strength"].iloc[0] == "HIGH CONVICTION"
    assert "🟢" in df["verdict_axis_forensics"].iloc[0]


# ── Asymmetric vetoes ──
def test_severe_forensic_vetoes_to_avoid():
    # elite tier-1 but forensic_score < 50 → must NOT be BUY
    df = compute_verdict(_frame(forensic_score=40.0))
    assert df["verdict_direction"].iloc[0] == "AVOID"
    assert "forensic" in df["verdict_top_risk"].iloc[0].lower()


def test_extreme_flag_count_vetoes_to_avoid():
    df = compute_verdict(_frame(red_flag_count=11))
    assert df["verdict_direction"].iloc[0] == "AVOID"


def test_gruesome_vetoes_to_avoid():
    df = compute_verdict(_frame(corporate_class="💀 GRUESOME"))
    assert df["verdict_direction"].iloc[0] == "AVOID"


def test_schilit_fail_softens_buy_to_watch():
    # elite + clean accounting score, but Schilit checker fails → BUY downgraded to WATCH (not AVOID)
    df = compute_verdict(_frame(schilit_pass=0))
    assert df["verdict_direction"].iloc[0] == "WATCH"
    assert "schilit" in df["verdict_narrative"].iloc[0].lower()


def test_poor_timing_softens_buy_to_watch():
    df = compute_verdict(_frame(buy_zone_label="Below Stop"))
    assert df["verdict_direction"].iloc[0] == "WATCH"
    assert "timing" in df["verdict_top_risk"].iloc[0].lower()


def test_low_tier_is_avoid():
    df = compute_verdict(_frame(conviction_tier=5, composite_score=20.0))
    assert df["verdict_direction"].iloc[0] == "AVOID"


def test_avoid_narrative_is_never_buy_toned():
    """The Sarda Energy value-trap pattern: AVOID (tier 4 / composite 42) yet cheap + high
    quality/growth + clean. The narrative reads the fundamental sub-scores, but it must NEVER
    contradict the direction with a buy-toned line ('high-conviction core holding' etc.)."""
    df = compute_verdict(_frame(
        conviction_tier=4, composite_score=42.0, quality_score=71.0, growth_score=66.0,
        pe=16.0, fair_pe_qglp=27.0, expected_excess_return=33.0,
    ))
    assert df["verdict_direction"].iloc[0] == "AVOID"
    narr = df["verdict_narrative"].iloc[0].lower()
    for phrase in ("high-conviction", "core holding", "elite compounder", "solid compounder"):
        assert phrase not in narr, f"AVOID narrative must not be buy-toned — got: {narr!r}"
    assert "avoid" in narr or "value-trap" in narr or "pass" in narr


# ── Display-only guarantee ──
def test_compute_verdict_does_not_mutate_scoring_columns():
    df_in = _frame(composite_score=77.0, conviction_tier=2)
    snap = df_in[["composite_score", "conviction_tier", "quality_score"]].copy()
    df_out = compute_verdict(df_in)
    pd.testing.assert_frame_equal(
        df_out[["composite_score", "conviction_tier", "quality_score"]], snap
    )


def test_all_verdict_columns_materialized():
    df = compute_verdict(_frame())
    expected = {
        "verdict_direction", "verdict_emoji", "verdict_strength", "verdict_confidence",
        # 6 orthogonal axes (Moat·Growth·Valuation·Balance·Governance·Forensics — no double-count)
        "verdict_axis_moat", "verdict_axis_growth", "verdict_axis_valuation",
        "verdict_axis_balance", "verdict_axis_governance", "verdict_axis_forensics",
        "verdict_top_risk", "verdict_narrative",
    }
    assert expected.issubset(set(df.columns))
    for c in expected:
        assert df[c].notna().all(), f"{c} has NaN"


# ── Confidence is separate from the score ──
def test_thin_data_lowers_confidence_only():
    df = compute_verdict(_frame(data_coverage_pct=30.0))
    assert df["verdict_confidence"].iloc[0] == "Very Low"


# ── Crash-proof on a minimal frame (defensive _col) ──
def test_minimal_frame_does_not_crash():
    df = compute_verdict(pd.DataFrame({"conviction_tier": [3]}))
    assert df["verdict_direction"].iloc[0] in {"BUY", "WATCH", "AVOID"}
    assert df["verdict_narrative"].notna().all()


# ── Live calibration: distribution must not collapse to one value ──
@pytest.mark.slow
def test_live_distribution_is_not_degenerate():
    import io, contextlib
    from core.data_engine import fetch_and_clean_data
    from core.scoring_engine import run_full_scoring
    from core.forensic_engine import compute_forensic_signals, apply_forensic_penalty
    with contextlib.redirect_stdout(io.StringIO()):
        df = fetch_and_clean_data("local")
        df = compute_forensic_signals(df)
        df = run_full_scoring(df)
        df = apply_forensic_penalty(df)
        df = compute_verdict(df)
    counts = df["verdict_direction"].value_counts(normalize=True)
    # no single bucket may swallow the whole universe (the 99% AVOID calibration bug guard)
    assert counts.max() < 0.97, f"degenerate verdict distribution: {dict(counts)}"
    assert (df["verdict_direction"] == "BUY").sum() >= 5


@pytest.mark.slow
def test_rank_tracks_post_penalty_composite():
    """`rank` must reflect the FINAL post-forensic-penalty composite_score, not the pre-penalty
    order it was first assigned in run_full_scoring. Regression guard for the bug where the top
    stock (composite 100) showed rank 2 while a composite-90 stock showed rank 1 (2,098/2,107
    misranked). Tie-safe: assert rank is a 1..N permutation and sorting by rank yields a
    non-increasing composite (ties get adjacent ranks)."""
    import io, contextlib
    from core.data_engine import fetch_and_clean_data
    from core.scoring_engine import run_full_scoring
    from core.forensic_engine import compute_forensic_signals, apply_forensic_penalty
    with contextlib.redirect_stdout(io.StringIO()):
        df = fetch_and_clean_data("local")
        df = compute_forensic_signals(df)
        df = run_full_scoring(df)
        df = apply_forensic_penalty(df)      # <-- the step that finalizes composite_score
    assert sorted(df["rank"].tolist()) == list(range(1, len(df) + 1)), "rank must be a 1..N permutation"
    assert df["rank"].isna().sum() == 0
    assert df.sort_values("rank")["composite_score"].is_monotonic_decreasing, \
        "sorting by rank must yield non-increasing composite_score (rank must track the penalized score)"
    assert int(df.loc[df["composite_score"].idxmax(), "rank"]) == 1, "the top composite stock must be rank 1"
