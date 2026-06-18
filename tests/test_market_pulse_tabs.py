"""Contract: the Market Pulse inner-tab set is exactly {Tsunami, QGLP, Sectors} after Stage 3
(2026-06-18) dropped the dead 💙 Blue Chips (0% fires) and the brittle 🚀 Tipping Points (folded into
an enhanced Sectors view). Static AST/string check over app.py — pins the structure so a tab can't be
re-added or a removed renderer re-wired without a conscious change.
"""
import ast
from pathlib import Path

_APP = Path(__file__).resolve().parent.parent / "app.py"


def _mp_tab_labels():
    """The string-literal labels of the `_mp_tabs = st.tabs([...])` assignment."""
    tree = ast.parse(_APP.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if (isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "_mp_tabs" for t in n.targets)
                and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "tabs"
                and n.value.args and isinstance(n.value.args[0], ast.List)):
            return [e.value for e in n.value.args[0].elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    return None


def test_market_pulse_has_exactly_three_tabs():
    labels = _mp_tab_labels()
    assert labels == ["🌊 Tsunami", "🏛️ QGLP", "📈 Sectors"], labels


def test_removed_renderers_are_not_called():
    src = _APP.read_text(encoding="utf-8")
    assert "render_bruised_blue_chips(" not in src, "Blue Chips list renderer must be gone"
    assert "render_multi_trillion_tipping_points(" not in src, "Tipping Points renderer must be gone"


def test_removed_renderers_are_not_imported():
    """No dangling import of the deleted list renderers (would crash app boot)."""
    src = _APP.read_text(encoding="utf-8")
    assert "render_bruised_blue_chips," not in src and "render_multi_trillion_tipping_points," not in src


def test_tab_extractor_has_teeth():
    labels = _mp_tab_labels()
    assert labels is not None and len(labels) == 3
