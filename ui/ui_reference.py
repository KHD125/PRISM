"""In-app searchable glossary for the 📖 Reference tab.

Pure render helper — returns inline dark-theme HTML and makes ZERO Streamlit calls.
The st.text_input search widget lives in app.py (which owns session_state); this module
only formats. The glossary (the 173-term `_RAW_GLOSSARY` single source of truth in
ui_components) is INJECTED so the coverage test can assert on the returned string.

Framework browser is a deliberate follow-up: `_FW_META` is function-local in the
parallel-session-owned ui_tearsheet.py and not importable — see the plan's follow-up.
"""
import html

from config import COLORS


def render_reference(glossary: dict, query: str = "") -> str:
    """Inline dark-theme HTML for the Reference tab: a short stable intro + the glossary
    as definition rows, filtered case-insensitively by `query` over term AND definition.
    Pure: no Streamlit, no I/O, no global state. Terms sorted case-insensitively (deterministic),
    so acronyms (CFO, CRS) interleave naturally with words (Capital, Cash) instead of ASCII-clumping."""
    q = (query or "").strip().lower()
    # Plain substring match — NOT regex — so query chars like '(' can never raise.
    items = [
        (term, defn) for term, defn in sorted(glossary.items(), key=lambda kv: kv[0].lower())
        if q in term.lower() or q in str(defn).lower()
    ]

    intro = (
        f'<div style="color:{COLORS["text_secondary"]};font-size:0.8rem;line-height:1.5;'
        f'margin:4px 0 14px 0;">PRISM scores ~2,100 stocks through many framework lenses and '
        f'synthesises them into one verdict per stock. This is the searchable glossary — type to '
        f'find any term you see on screen. Each entry explains what the term <em>means</em>; it '
        f'never says whether a value is good or bad.</div>'
    )

    if not items:
        return (
            intro
            + f'<div style="color:{COLORS["text_muted"]};font-size:0.85rem;padding:10px 0;">'
            + f'No terms match “{html.escape(query or "")}”.</div>'
        )

    count = (
        f'<div style="font-size:0.66rem;color:{COLORS["text_muted"]};text-transform:uppercase;'
        f'letter-spacing:0.5px;margin-bottom:6px;">{len(items)} of {len(glossary)} terms</div>'
    )
    rows = "".join(
        f'<div style="padding:9px 0;border-bottom:1px solid {COLORS["border"]};">'
        f'<div style="font-size:0.82rem;font-weight:700;color:{COLORS["text_primary"]};">'
        f'{html.escape(term)}</div>'
        f'<div style="font-size:0.74rem;color:{COLORS["text_secondary"]};line-height:1.45;'
        f'margin-top:2px;">{html.escape(str(defn))}</div></div>'
        for term, defn in items
    )
    return intro + count + rows


def render_concepts(concept_ref: dict, query: str = "") -> str:
    """Inline dark-theme HTML for the categorical VALUE-labels (Wealth Creator, Deep Value, Stage 2…),
    grouped by category, filtered case-insensitively by `query` over label AND explanation. Pure: no
    Streamlit, no I/O. Returns "" when nothing matches (the glossary section shows its own state)."""
    q = (query or "").strip().lower()
    blocks = []
    for category, entries in concept_ref.items():
        rows = [(lbl, exp) for lbl, exp in entries
                if q in lbl.lower() or q in str(exp).lower()]
        if not rows:
            continue
        head = (f'<div style="font-size:0.7rem;font-weight:800;color:{COLORS["purple"]};'
                f'text-transform:uppercase;letter-spacing:0.8px;margin:16px 0 6px 0;">'
                f'{html.escape(category)}</div>')
        body = "".join(
            f'<div style="padding:7px 0;border-bottom:1px solid {COLORS["border"]};">'
            f'<div style="font-size:0.8rem;font-weight:700;color:{COLORS["text_primary"]};">{html.escape(lbl)}</div>'
            f'<div style="font-size:0.74rem;color:{COLORS["text_secondary"]};line-height:1.45;margin-top:2px;">'
            f'{html.escape(str(exp))}</div></div>'
            for lbl, exp in rows
        )
        blocks.append(head + body)
    return "".join(blocks)
