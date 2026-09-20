"""Contract: the scoring controls after the Command Center removal (2026-08-24).

The six-mandate Command Center was REMOVED as a measured false promise: the three Hybrid
mandates (QGLP Balanced / Lynch GARP / Deep Value) produced BIT-IDENTICAL composite_score,
rank, conviction_tier, gate_pass and quality_score — the profile feeds ONLY the QGLP screen
(qglp_score / qglp_pass), never the composite — while the Q/G/L/P weights strip implied
engine re-weighting that never happened. What remains: ONE plain selectbox in ⚙️ Config
with a widget-owned key (cfg_mode) and NO callbacks — the canonical/mirror machinery it
replaces caused the 2026-08-24 production KeyError crash.

The profile selectbox beside it was REMOVED 2026-09-20, carrying the same measurement one step
further: it re-ranked nothing AND rewrote itself on a mode change. Its pin below inverted into a
tombstone rather than dying (§6). Full account: tests/test_qglp_profile_fixed.py.

These tests pin the NEW architecture (this file previously pinned the mandate↔override sync;
updated per §6: stale tests follow the architecture, they don't die).
"""
import re
from pathlib import Path

_APP = Path(__file__).resolve().parent.parent / "app.py"


def test_command_center_stays_removed():
    """Tombstone: no zombie resurrection of the mandate machinery (e.g. from an old branch or a
    half-revert). Every one of these names was part of the removed false-promise layer."""
    src = _APP.read_text(encoding="utf-8")
    for name in ["_MANDATES", "_MANDATE_BY_COMBO", "_pick_mandate", "_sel_mandate",
                 "_sync_mode", "_sync_profile", "adv_mode", "adv_profile", "_w_mode", "_w_profile"]:
        assert name not in src, (
            f"{name!r} is back in app.py — the Command Center was removed 2026-08-24 because "
            "three of six mandates were ranking-identical (profile feeds only the QGLP screen). "
            "Reintroducing it requires the evidence-gated composite rewire, not a revert."
        )


def test_scoring_controls_are_plain_config_widgets():
    """The ONE live knob must be a plain widget-owned selectbox (key=cfg_mode), initialized via
    setdefault and read from session_state at the top — no on_change callback, no canonical/mirror
    key (the pattern that produced the prod KeyError)."""
    src = _APP.read_text(encoding="utf-8")
    # the default moved to a CONSTANT on 2026-09-19 (config.DEFAULT_ANALYSIS_MODE) so the app,
    # the engine, snapshots, /census and /verify cannot drift apart — the literal is gone ON PURPOSE
    assert 'st.session_state.setdefault("cfg_mode", DEFAULT_ANALYSIS_MODE)' in src
    assert 'key="cfg_mode"' in src
    # plain widget: the control may not wire a callback
    for m in re.finditer(r'key="cfg_(?:mode|profile)"[^)]*', src):
        assert "on_change" not in m.group(0), "cfg_* selectboxes must stay callback-free"


def test_the_profile_snap_stays_removed():
    """TOMBSTONE (2026-09-20), inverted from the pin that used to REQUIRE this block.

    It used to read: a mode change can orphan the profile, so snap cfg_profile into
    ANALYSIS_MODES[mode]['allowed_profiles'] before anything reads it. That guard WAS the defect.
    "Technical Only" permitted only [Momentum, Turnaround], the snap was one-way, and the round
    trip Breakout -> Technical Only -> Breakout left the QGLP screen on Momentum permanently:
    qglp_pass 413 -> 631, the MOSL convergence table 709 -> 789, 218 stocks' lens count moved —
    while composite_score, rank and every header tile stayed identical, so nothing on screen
    contradicted it and no test could see it.

    The profile is now a constant. Reintroducing any of these lines reopens the path."""
    src = _APP.read_text(encoding="utf-8")
    for gone in ['_allowed_profiles',
                 'st.session_state["cfg_profile"]',
                 'options=_allowed_profiles']:
        assert gone not in src, (
            f"{gone!r} is back in app.py — that is the silent-rewrite path. The QGLP screen is "
            "a constant (config.MASTER_PROFILES); see tests/test_qglp_profile_fixed.py."
        )


def test_marks_gauge_stays_removed():
    """Tombstone (2026-08-28): the 5-slider Marks Cycle Temperature Gauge was REMOVED — the only
    hand-set subjective input in an evidence-first system: its sliders never persisted (session
    state only), its output was sealed off from everything by its own display-only guardrail, and
    the objective macro layer (detect_market_regime) already owns adaptation. Removing it also
    closed the tab audit's two findings (the missing §5 pin and the hardcoded 25%/65% gradient).
    NAME-COLLISION WARNING: the "Marks Cycle Shield" FRAMEWORK is a separate, engine-computed
    feature and stays — these exact tombstone names deliberately do not match it."""
    src = _APP.read_text(encoding="utf-8")
    cfg = (_APP.parent / "config.py").read_text(encoding="utf-8")
    for name in ['key="ct_val"', 'key="ct_credit"', 'key="ct_psych"', 'key="ct_cap"',
                 'key="ct_qual"', "cycle_total", "MARKS_CYCLE", "DEFAULT_CYCLE_TEMPERATURE",
                 "Cycle Temperature Gauge"]:
        assert name not in src, f"{name!r} is back in app.py — the gauge was removed 2026-08-28"
    for name in ["MARKS_CYCLE", "DEFAULT_CYCLE_TEMPERATURE"]:
        assert name not in cfg, f"{name!r} is back in config.py — the gauge constants were removed"


def test_data_health_card_computes_live_never_hardcodes():
    """The 🩺 card's whole value is that it SELF-RESOLVES: every figure must be computed from
    this run's frame, so fixing the source sheet turns rows green with zero code changes. A
    hardcoded coverage percentage would freeze the diagnosis forever."""
    src = _APP.read_text(encoding="utf-8")
    i = src.index("🩺 DATA HEALTH")
    block = src[i:src.index('_cfg_card("Source-Sheet Gaps', i)]
    # live computations present
    assert '"dividend_payout_ratio"' in block and ".notna().mean()" in block
    assert '"current_ratio_1yb"' in block and "==" in block
    assert '"data_coverage_pct"' in block
    assert "get_data_freshness(" in block, "the vintage row no longer reads the sheet's own name"
    # no frozen diagnosis: the known figures at build time must NOT appear as literals
    for frozen in ("41%", "58.8", "59%"):
        assert frozen not in block, (
            f"{frozen!r} is hardcoded in the Data Health card — the card must measure, not remember"
        )


def test_data_health_vintage_row_has_both_states():
    """The vintage row replaced the snapshot nag (2026-08-30). Both states must exist —
    a dated sheet AND an undated one — because a card that only renders the happy path
    goes blank exactly when something is wrong. Graded in sessions, never calendar days:
    Friday's data read on a Monday is current, and day-counting would call it stale."""
    src = _APP.read_text(encoding="utf-8")
    i = src.index("🩺 DATA HEALTH")
    block = src[i:src.index('_cfg_card("Source-Sheet Gaps', i)]
    assert "_vin.is_known" in block, "no fallback for a sheet whose name carries no date"
    assert '"Data as of"' in block
    assert "_freshness_color(" in block, "the row must be colour-graded by staleness"
    assert "snapshot" not in block.lower(), "the retired snapshot nag is back"


def test_cr_1yb_copy_premise_still_true():
    """PREMISE PIN (the test_reinvestment_rate_data_gap precedent): the CR-1YB row's red state
    assumes the sheet still carries a copy of the current CR. Measured at build: ~100%% identical.
    IF THIS FAILS, IT IS GOOD NEWS — the sheet was fixed; delete this test and watch the card's
    row turn green on its own."""
    import contextlib, io as _io2, sys as _sys
    root = str(_APP.parent)
    for p in (root, root + "/core"):
        if p not in _sys.path:
            _sys.path.insert(0, p)
    from data_engine import (coerce_numeric_columns, compute_derived_signals, load_all_csvs,
                             merge_datasets)
    with contextlib.redirect_stdout(_io2.StringIO()):
        d = compute_derived_signals(coerce_numeric_columns(merge_datasets(load_all_csvs("local"))))
    both = d["current_ratio"].notna() & d["current_ratio_1yb"].notna()
    same = float((d.loc[both, "current_ratio"] == d.loc[both, "current_ratio_1yb"]).mean())
    assert same > 0.90, (
        f"CR-1YB now differs from CR on {(1-same):.0%} of rows — the source sheet appears FIXED. "
        f"Good news: delete this premise test; the Data Health row self-resolves."
    )
