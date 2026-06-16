"""
test_tearsheet_stateless_contract.py
====================================
Guards the CLAUDE.md §5 "Stateless UI/presentation layer" invariant for
``ui/ui_tearsheet.py`` — the ONE hard rule in this repo that previously had no
contract test and therefore silently drifted (a metric cockpit reintroduced
``st.columns``/``st.metric`` padding the module is supposed to ban).

Banned calls inside ui_tearsheet.py and WHY:
  • st.button / st.number_input / st.slider — state-mutating widgets that write to
    st.session_state, which app.py owns (stock selection, scoring profile). A widget
    here corrupts that global state on rerun.
  • st.columns / st.metric — layout/padding bloat. The module's contract (and its own
    inline comments at lines ~265 / ~1245: "no st.columns/st.metric padding") is to
    render with compact inline HTML/CSS flex/grid containers instead.

This is a STATIC AST check (not a regex): it ignores the strings/comments that
mention these names and flags only real ``st.<name>(...)`` calls.
"""
import ast
import os

_TEARSHEET = os.path.join(os.path.dirname(__file__), "..", "ui", "ui_tearsheet.py")

# State-mutating widgets (always banned) + layout-bloat primitives (use inline flex/grid).
_BANNED = {"button", "number_input", "slider", "columns", "metric"}


def _banned_st_calls(path):
    """Return [(lineno, attr)] for every literal ``st.<banned>(...)`` call in the file."""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)

    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr in _BANNED
            and isinstance(func.value, ast.Name)
            and func.value.id == "st"
        ):
            hits.append((node.lineno, func.attr))
    return hits


def test_tearsheet_uses_no_banned_streamlit_widgets():
    """ui_tearsheet.py must stay 100% stateless + inline — no banned st.* calls."""
    hits = _banned_st_calls(_TEARSHEET)
    assert hits == [], (
        "ui/ui_tearsheet.py must contain no state-mutating widgets or st.columns/"
        "st.metric layout bloat (CLAUDE.md §5). Offending calls:\n  "
        + "\n  ".join(f"line {ln}: st.{attr}(...)" for ln, attr in hits)
    )
