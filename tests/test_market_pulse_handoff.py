"""Contract: the cross-tab "open in Tear-Sheet" handoff never violates Streamlit's
set-after-instantiation rule.

`st.session_state["xray_stock"]` is a WIDGET key (the Tear-Sheet selectbox, tab2). Streamlit raises
StreamlitAPIException if that key is assigned AFTER the widget is instantiated in the same run. Tabs
that render BEFORE the widget (Discovery tab0, Scanner tab1) may set it directly; tabs that render
AFTER it (Market Pulse tab3) MUST stage a transient `_pending_xray` + st.rerun(), consumed at the top
of the Tear-Sheet before the selectbox. This crashed live from Market Pulse until 2026-06-18; the
static guard below makes the regression impossible to reintroduce silently.
"""
import ast
from pathlib import Path

_APP = Path(__file__).resolve().parent.parent / "app.py"


def _tree():
    return ast.parse(_APP.read_text(encoding="utf-8"))


def _selectbox_lineno(tree):
    """Line of the st.selectbox(..., key='xray_stock') — the widget instantiation point."""
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "selectbox"
                and any(kw.arg == "key" and isinstance(kw.value, ast.Constant)
                        and kw.value.value == "xray_stock" for kw in n.keywords)):
            return n.lineno
    return None


def _widget_key_assignments(tree, key):
    """Linenos of every `st.session_state["<key>"] = ...` assignment."""
    out = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Assign):
            continue
        for t in n.targets:
            if (isinstance(t, ast.Subscript)
                    and isinstance(t.value, ast.Attribute) and t.value.attr == "session_state"
                    and isinstance(t.slice, ast.Constant) and t.slice.value == key):
                out.append(n.lineno)
    return out


def test_xray_stock_never_set_after_its_widget():
    """No direct write to the xray_stock widget key may occur after the selectbox is instantiated."""
    tree = _tree()
    box = _selectbox_lineno(tree)
    assert box is not None, "could not locate the st.selectbox(key='xray_stock') widget"
    late = [ln for ln in _widget_key_assignments(tree, "xray_stock") if ln > box]
    assert not late, (
        f"st.session_state['xray_stock'] assigned at line(s) {late}, AFTER the widget at line {box} "
        f"— this throws StreamlitAPIException. Stage _pending_xray + st.rerun() instead.")


def test_cross_tab_handoff_uses_pending_key():
    """The transient handoff must exist: _pending_xray is staged (by a later tab) AND consumed
    BEFORE the selectbox (so the regression fix isn't silently deleted)."""
    tree = _tree()
    box = _selectbox_lineno(tree)
    staged = _widget_key_assignments(tree, "_pending_xray")
    assert staged, "no st.session_state['_pending_xray'] staging found — the cross-tab handoff is gone"
    src = _APP.read_text(encoding="utf-8")
    consume = 'st.session_state["xray_stock"] = st.session_state.pop("_pending_xray")'
    assert consume in src, "the _pending_xray consume line is missing"
    consume_lineno = src[:src.index(consume)].count("\n") + 1
    assert consume_lineno < box, "the _pending_xray consume must run BEFORE the selectbox is instantiated"


def test_guard_present_alongside_rerun():
    """Each _pending_xray staging is paired with a change-guard + st.rerun(): st.dataframe selections
    persist across reruns, so an unguarded stage+rerun would loop forever. Pins the loop-safety."""
    src = _APP.read_text(encoding="utf-8")
    n_stage  = src.count('st.session_state["_pending_xray"]')
    n_rerun  = src.count("st.rerun()")
    n_guard  = src.count('!= st.session_state.get("xray_stock")')
    assert n_stage >= 2, "expected the Tsunami + QGLP stagings"
    assert n_rerun >= n_stage and n_guard >= n_stage, (
        f"each staging ({n_stage}) needs a change-guard ({n_guard}) + st.rerun() ({n_rerun}) "
        "to avoid an infinite rerun loop")
