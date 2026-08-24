"""Contract: the scoring controls after the Command Center removal (2026-08-24).

The six-mandate Command Center was REMOVED as a measured false promise: the three Hybrid
mandates (QGLP Balanced / Lynch GARP / Deep Value) produced BIT-IDENTICAL composite_score,
rank, conviction_tier, gate_pass and quality_score — the profile feeds ONLY the QGLP screen
(qglp_score / qglp_pass), never the composite — while the Q/G/L/P weights strip implied
engine re-weighting that never happened. What remains: two plain selectboxes in ⚙️ Config
with widget-owned keys (cfg_mode / cfg_profile) and NO callbacks — the canonical/mirror
machinery they replace caused the 2026-08-24 production KeyError crash.

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
    """The two live knobs must be plain widget-owned selectboxes (key=cfg_mode / cfg_profile),
    initialized via setdefault and read from session_state at the top — no on_change callbacks,
    no canonical/mirror keys (the pattern that produced the prod KeyError)."""
    src = _APP.read_text(encoding="utf-8")
    assert 'st.session_state.setdefault("cfg_mode", "Hybrid")' in src
    assert 'st.session_state.setdefault("cfg_profile", "Balanced")' in src
    assert 'key="cfg_mode"' in src and 'key="cfg_profile"' in src
    # plain widgets: neither control may wire a callback
    for m in re.finditer(r'key="cfg_(?:mode|profile)"[^)]*', src):
        assert "on_change" not in m.group(0), "cfg_* selectboxes must stay callback-free"


def test_profile_snaps_into_the_active_modes_allowed_set():
    """A mode change can orphan the profile (e.g. Technical allows only Momentum/Turnaround).
    The top-of-script guard must snap cfg_profile into ANALYSIS_MODES[mode]['allowed_profiles']
    BEFORE anything reads it — and the Config selectbox must offer exactly that allowed list."""
    src = _APP.read_text(encoding="utf-8")
    assert '_allowed_profiles = ANALYSIS_MODES[st.session_state["cfg_mode"]]["allowed_profiles"]' in src
    assert 'if st.session_state["cfg_profile"] not in _allowed_profiles:' in src
    assert 'st.session_state["cfg_profile"] = _allowed_profiles[0]' in src
    assert "options=_allowed_profiles" in src, "profile selectbox must offer the mode's allowed set"
