"""In-app searchable glossary for the 📖 Reference tab.

Pure render helper — returns inline dark-theme HTML and makes ZERO Streamlit calls.
The st.text_input search widget lives in app.py (which owns session_state); this module
only formats. The glossary (the `_RAW_GLOSSARY` single source of truth in
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


def render_flags(flag_display: dict, query: str = "") -> str:
    """Inline dark-theme HTML for the forensic RED FLAGS — each shown as the engine's OWN display
    description + severity dot, severity-sorted (🔴→🟠→🟡). Pure: no Streamlit. Single source =
    `_FLAG_DISPLAY` in ui_tearsheet (rf_col → (description, severity)); rendered, never duplicated."""
    q = (query or "").strip().lower()
    _sev = {"🔴": 0, "🟠": 1, "🟡": 2}
    rows = [(desc, sev) for rf, (desc, sev) in flag_display.items()
            if q in desc.lower() or q in rf.lower()]
    if not rows:
        return ""
    rows.sort(key=lambda r: (_sev.get(r[1], 9), r[0].lower()))
    return "".join(
        f'<div style="padding:7px 0;border-bottom:1px solid {COLORS["border"]};">'
        f'<span style="font-size:0.8rem;">{html.escape(sev)} </span>'
        f'<span style="font-size:0.78rem;color:{COLORS["text_secondary"]};line-height:1.45;">'
        f'{html.escape(desc)}</span></div>'
        for desc, sev in rows
    )


def build_reference_markdown(glossary: dict, concept_ref: dict, flag_display: dict,
                             frameworks: dict = None, as_of: str = None) -> str:
    """The FULL Reference as one Markdown string, assembled from the SAME single-source dicts the
    render_* functions consume (NOT scraped from their HTML) — so the download, the on-screen
    Reference, and any future docs/reference.md can never drift. Exports EVERYTHING (no query filter).
    Deterministic ordering, matching the renderers. PURE: no Streamlit, no I/O.

    `frameworks` is optional (frameworks-ready): pass _FW_META once it's importable to emit a
    Frameworks section; None -> the section is omitted (no crash). Each value may be a {emoji,name}
    dict or a bare string."""
    import datetime as _dt
    as_of = as_of or _dt.date.today().isoformat()

    def _clean(s):
        # One line, whitespace-collapsed. Bullets (not tables) need no MD-escaping; keeping the text
        # verbatim lets the coverage test match terms exactly after the same collapse.
        return " ".join(str(s).split())

    n_concepts = sum(len(v) for v in concept_ref.values())
    out = [
        "# PRISM Reference", "",
        f"_Generated {as_of} · {len(glossary)} glossary terms · {n_concepts} concepts · "
        f"{len(flag_display)} red flags_", "",
    ]

    out += ["## Glossary", ""]
    for term, defn in sorted(glossary.items(), key=lambda kv: kv[0].lower()):
        out.append(f"- **{_clean(term)}** — {_clean(defn)}")
    out.append("")

    out += ["## Concepts", ""]
    for category, entries in concept_ref.items():            # dict order = deterministic
        out += [f"### {_clean(category)}", ""]
        for lbl, exp in entries:
            out.append(f"- **{_clean(lbl)}** — {_clean(exp)}")
        out.append("")

    out += ["## Forensic Red Flags", ""]
    _sev = {"🔴": 0, "🟠": 1, "🟡": 2}
    for desc, sev in sorted(((d, s) for d, s in flag_display.values()),
                            key=lambda r: (_sev.get(r[1], 9), r[0].lower())):
        out.append(f"- {sev} {_clean(desc)}")
    out.append("")

    if frameworks:
        out += ["## Frameworks", ""]
        for _key, meta in frameworks.items():
            emoji = meta.get("emoji", "") if isinstance(meta, dict) else ""
            name = meta.get("name", _key) if isinstance(meta, dict) else str(meta)
            out.append(f"- {emoji} **{_clean(name)}**")
        out.append("")

    return "\n".join(out).rstrip() + "\n"
