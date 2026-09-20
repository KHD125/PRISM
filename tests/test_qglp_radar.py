"""
test_qglp_radar.py
==================
Contract for the 👑 QGLP deep-dive radar — the Frameworks tab's audit card for the
system's namesake methodology (Raamdeo's Q·G·L·P), which until 2026-08-22 was the only
major framework WITHOUT a deep-dive despite its four 0-100 sub-scores, its three
profile-driven hard gates, and the 19th-WCS SQGLP letter screen all being pre-computed.

PURE DISPLAY contract: the card reads qglp_quality/growth/longevity/price, qglp_score,
qglp_pass, the S·Q·G·L·P letter columns and the raw gate inputs (roce / pat_gr_5y /
peg); thresholds come from MASTER_PROFILES[profile] — the same source the engine's
qglp_pass used — so the card can never drift from the gate (the Fisher lesson).
Missing data renders honest blanks, never fabricated zeros.

Run with: pytest tests/test_qglp_radar.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd


def _render(stock: pd.Series, profile: str = "Balanced") -> str:
    from streamlit.testing.v1 import AppTest

    def _app():
        import streamlit as st
        from ui.ui_tearsheet import render_qglp_radar
        render_qglp_radar(st.session_state["stock_row"], st.session_state["profile"])

    at = AppTest.from_function(_app)
    at.session_state["stock_row"] = stock
    at.session_state["profile"] = profile
    at.run(timeout=60)
    assert not at.exception, at.exception
    return " ".join(str(md.value) for md in at.markdown)


def _stock(**over) -> pd.Series:
    base = {
        "name": "Compounder Ltd",
        "qglp_quality": 82.0, "qglp_growth": 74.0, "qglp_longevity": 68.0, "qglp_price": 55.0,
        "qglp_score": 72.0, "qglp_pass": 1,
        "roce": 24.0, "pat_gr_5y": 21.0, "peg": 1.1,
        "sqglp_s": 1, "sqglp_q": 1, "sqglp_g": 1, "sqglp_l": 0, "sqglp_p": 0,
        "sqglp_score": 3, "century_stock_flag": 0,
    }
    base.update(over)
    return pd.Series(base)


def test_qglp_radar_is_exported_from_ui():
    """The moved-but-unexported-symbol crash class (help_chip precedent): app.py imports
    from `ui`, so the function must resolve there, not only in ui_tearsheet."""
    from ui import render_qglp_radar
    assert callable(render_qglp_radar)


def test_all_four_legs_render_with_their_scores():
    html = _render(_stock())
    for leg in ("Quality", "Growth", "Longevity", "Price"):
        assert leg in html, f"missing leg: {leg}"
    assert "82" in html and "55" in html


def test_pass_state_and_gates_show_actuals_against_profile_thresholds():
    html = _render(_stock())
    assert "QGLP" in html and ("COMPLIANT" in html or "PASS" in html.upper())
    from config import MASTER_PROFILES
    g = MASTER_PROFILES["Balanced"]
    assert f"{g['roce_gate']:.0f}" in html      # threshold shown, not hardcoded prose
    assert "24" in html                          # actual ROCE shown beside it


def test_failing_gate_reads_as_pending_not_compliant():
    html = _render(_stock(qglp_pass=0, peg=2.6))
    assert "COMPLIANT" not in html


def test_sqglp_letter_strip_shows_all_five_letters():
    html = _render(_stock())
    assert "Century" in html or "SQGLP" in html
    assert html.count("✅") >= 3 and html.count("❌") >= 2     # S,Q,G pass · L,P fail


def test_the_card_discloses_the_standard_it_judged_by_not_a_profile_name():
    """UPDATED 2026-09-20 (§6). This test used to read: "Gates move with the scoring profile — the
    card must say which one it's judging by", and asserted "Balanced" appeared in the HTML.

    That premise died with the selector. Gates no longer move with a profile — only with the market
    regime — so printing "Profile: Balanced (QGLP)" named a control the reader cannot see or change,
    and "Profile-Weighted QGLP Score" implied a weighting they could pick. Both are gone.

    The DISCLOSURE invariant survives and gets stronger: the profile name was only ever a proxy for
    "which thresholds judged this stock", so the card must show the thresholds themselves. It does,
    on each gate card ("vs >= 15"), which is also what keeps the label honest when a BEAR regime
    lifts the ROCE gate to 20 — a name could not have told you that.
    """
    from config import MASTER_PROFILES
    html = _render(_stock(), profile="Balanced")
    g = MASTER_PROFILES["Balanced"]
    for gate in ("roce_gate", "growth_gate"):
        assert f"{g[gate]:.0f}" in html, f"the card must disclose its {gate} ({g[gate]})"
    assert f"{g['peg_gate']:.1f}" in html, "the card must disclose its PEG gate"
    # and it must NOT advertise a profile the reader has no way to change
    for gone in ("Profile:", "Profile-Weighted", "profile:"):
        assert gone not in html, (
            f"the radar prints {gone!r} — the QGLP screen is a constant since 2026-09-20, so naming "
            f"it points the reader at a control that does not exist "
            f"(see tests/test_qglp_profile_fixed.py)"
        )


def test_missing_subscores_render_blanks_not_fabricated_values():
    bare = pd.Series({"name": "No Data Ltd"})
    html = _render(bare)
    assert "—" in html
    assert "COMPLIANT" not in html
