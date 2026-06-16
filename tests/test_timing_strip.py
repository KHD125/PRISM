"""
test_timing_strip.py
====================
Contract for the ⏱️ ENTRY TIMING strip in `render_verdict_scorecard` (ui_tearsheet).

WHY this strip exists: the 6 verdict axes (moat/growth/valuation/balance/governance/forensics)
weigh SELECTION (the WHAT) and are deliberately blind to momentum. Timing is the WHEN — a
separate, entry-only read. This strip surfaces 4 orthogonal momentum orphans (verified alive +
max pairwise corr 0.17 on live data) WITHOUT implying they drove the BUY/WATCH/AVOID verdict.

These are pure-display assertions driven through Streamlit's AppTest harness (same pattern as
test_ui_smoke). Synthetic Series → no CSV data folder required, runs everywhere.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from config import COLORS


def _stock(**over) -> pd.Series:
    """Minimal stock row: only the 4 timing columns matter; the rest default to NaN→'—'."""
    base = {
        "rs_score": 85.0,
        "trajectory_score": 0.70,
        "eps_acceleration": 50.0,
        "volume_score": 80.0,
    }
    base.update(over)
    return pd.Series(base)


def _scorecard_md(stock: pd.Series):
    """Render render_verdict_scorecard in-process; return (all_text, timing_strip_html)."""
    from streamlit.testing.v1 import AppTest

    def _app():
        import streamlit as st
        from ui.ui_tearsheet import render_verdict_scorecard
        render_verdict_scorecard(st.session_state["stock_row"])

    at = AppTest.from_function(_app)
    at.session_state["stock_row"] = stock
    at.run(timeout=30)
    assert not at.exception, f"scorecard raised: {[str(e.value) for e in at.exception]}"
    blocks = [str(md.value) for md in at.markdown]
    timing = next((b for b in blocks if "ENTRY TIMING" in b), None)
    return " ".join(blocks), timing


def test_timing_strip_renders_with_all_four_chips():
    """The strip exists, is labelled ENTRY TIMING, and shows all 4 orthogonal chips + values."""
    _, timing = _scorecard_md(_stock())
    assert timing is not None, "⏱️ ENTRY TIMING strip not rendered"
    for label in ("RS", "Traj", "EPS-Accel", "Vol"):
        assert label in timing, f"timing chip '{label}' missing"
    assert "85" in timing, "RS value not surfaced"
    assert "+0.70" in timing, "Trajectory value not surfaced"
    assert "80" in timing, "Volume value not surfaced"


def test_timing_strip_semantic_colours_strong_vs_weak():
    """Color law: a strong momentum profile is green; a weak one is red — never inverted."""
    _, strong = _scorecard_md(_stock())  # rs85/traj0.70/accel50/vol80 → all strong
    assert COLORS["green"] in strong, "strong timing profile not green"

    _, weak = _scorecard_md(_stock(rs_score=10.0, trajectory_score=-0.30,
                                   eps_acceleration=-20.0, volume_score=20.0))
    assert COLORS["red"] in weak, "weak timing profile not red"


def test_timing_strip_nan_safe():
    """Missing momentum data → '—', no 'nan' leak, no exception (the +nan% bug class)."""
    all_text, timing = _scorecard_md(_stock(rs_score=np.nan, trajectory_score=np.nan,
                                            eps_acceleration=np.nan, volume_score=np.nan))
    assert timing is not None, "strip must still render with NaN inputs"
    assert "—" in timing, "NaN momentum should render as em-dash"
    import re
    assert not re.findall(r"[+\-₹ >]nan[%< ]", all_text.lower()), "NaN leaked into timing strip"
