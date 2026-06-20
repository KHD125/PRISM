"""STANDALONE PROTOTYPE — inline "Open Analysis" on a Discovery card.

PURPOSE: show the proposed UX *before* touching any main project file. This script does NOT modify
app.py or ui/ — it only READS the real, already-shipped render functions (render_stock_card,
render_verdict_scorecard) + the real injected CSS, so this preview looks exactly like production.

What it demonstrates: clicking "Open Analysis" expands the verdict scorecard INLINE under the card —
no cross-tab handoff (today's friction: you set the stock, then must manually click The Tear-Sheet
tab). State is a PLAIN session_state key — no widget-key sharing (the anti-pattern fixed for the mandate).

Run:  STOCKSCAN_SHEET_ID=<id> PYTHONUTF8=1 streamlit run prototype_inline_card.py --server.port 8530
"""
import contextlib
import io

import streamlit as st

from config import COLORS
from ui import inject_css, render_stock_card, render_verdict_scorecard

st.set_page_config(layout="wide", page_title="Prototype · Inline Card", page_icon="🔬")
inject_css()


@st.cache_data(show_spinner="Loading the real scored universe (~40s, once)…")
def _load():
    with contextlib.redirect_stdout(io.StringIO()):
        from core import fetch_and_clean_data, run_scoring_pipeline
        return run_scoring_pipeline(fetch_and_clean_data("local"))


df = _load()
top = df.sort_values("composite_score", ascending=False).head(8).reset_index(drop=True)

st.markdown(
    f"<div style='font-size:1.15rem;font-weight:800;color:{COLORS['text_primary']};'>"
    f"🔬 Prototype — inline analysis on a Discovery card</div>"
    f"<div style='font-size:0.8rem;color:{COLORS['text_secondary']};margin:4px 0 16px;'>"
    f"Click <b>Open Analysis</b> on any card → the verdict band + 6-axis scorecard expand <b>right "
    f"here</b>, no tab switch. Standalone preview using the REAL components — the main app is untouched."
    f"</div>",
    unsafe_allow_html=True,
)

if "proto_open" not in st.session_state:
    st.session_state["proto_open"] = None


def _verdict_band(row) -> str:
    _dir = str(row.get("verdict_direction", "") or "—")
    _emoji = str(row.get("verdict_emoji", "") or "")
    _str = str(row.get("verdict_strength", "") or "")
    _narr = str(row.get("verdict_narrative", "") or "")
    _clr = {"BUY": COLORS["green"], "WATCH": COLORS["gold"]}.get(_dir, COLORS["text_muted"])
    _sc = float(row.get("composite_score", 0) or 0)
    return (
        f"<div style='border:1px solid {_clr}55;border-left:3px solid {_clr};border-radius:10px;"
        f"background:{_clr}0f;padding:10px 14px;margin:2px 0 12px;'>"
        f"<span style='font-size:0.98rem;font-weight:800;color:{_clr};'>{_emoji} {_dir}</span>"
        f"<span style='font-size:0.72rem;color:{COLORS['text_muted']};'> &nbsp;·&nbsp; {_str} · "
        f"Score {_sc:.0f}/100</span>"
        f"<div style='font-size:0.75rem;color:{COLORS['text_secondary']};margin-top:3px;'>{_narr}</div>"
        f"</div>"
    )


for i in range(len(top)):
    row = top.iloc[i]
    render_stock_card(row, show_scores=True)

    _is_open = st.session_state["proto_open"] == row["name"]
    _, _bc = st.columns([7, 3])
    with _bc:
        if st.button(
            "🔬 Close Analysis ▴" if _is_open else "🔬 Open Analysis ▾",
            key=f"proto_{i}", use_container_width=True,
            type="primary" if _is_open else "secondary",
        ):
            # PLAIN state key — set in the button handler, no widget-key sharing, no st.rerun crash.
            st.session_state["proto_open"] = None if _is_open else row["name"]
            st.rerun()

    if _is_open:
        with st.container(border=True):
            st.markdown(_verdict_band(row), unsafe_allow_html=True)
            try:
                render_verdict_scorecard(row)
            except Exception as e:  # the prototype must never crash on a render-context gap
                st.warning(f"(scorecard needs fuller tearsheet context here: {e})")
            st.caption("↑ Inline quick-look. The full Tear-Sheet tab (radars · frameworks · forensics) "
                       "stays the deep dive — but you never leave Discovery to read the verdict.")
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
