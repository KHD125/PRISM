import ast
import html
import os

from ui.ui_components import _RAW_GLOSSARY
from ui.ui_reference import render_reference

_REF = os.path.join(os.path.dirname(__file__), "..", "ui", "ui_reference.py")


def test_renders_every_glossary_term():
    """Coverage guard: an empty query renders ALL 173 terms (escaping-aware)."""
    out = render_reference(_RAW_GLOSSARY, "")
    missing = [t for t in _RAW_GLOSSARY if html.escape(t) not in out]
    assert not missing, f"Reference dropped {len(missing)} terms: {missing[:5]}"


def test_query_filters_to_strict_subset():
    """A distinctive query narrows to its term and excludes unrelated ones."""
    out = render_reference(_RAW_GLOSSARY, "Dilution Vampire")
    assert html.escape("Dilution Vampire") in out
    assert html.escape("ROCE Current") not in out          # unrelated term filtered out
    assert out != render_reference(_RAW_GLOSSARY, "")        # strictly fewer than all


def test_no_match_is_graceful():
    """A query that matches nothing returns a friendly message, not broken/empty HTML."""
    out = render_reference(_RAW_GLOSSARY, "zzzznotarealterm")
    assert "No terms match" in out


def test_special_chars_are_escaped():
    """Definitions contain '<' — they must be HTML-escaped, not injected raw."""
    out = render_reference({"X": "Nano (<₹100 Cr)"}, "")
    assert "&lt;₹100" in out and "(<₹100" not in out


def test_render_is_pure_no_streamlit_calls():
    """Structural purity: ui_reference.py makes ZERO st.* calls (widgets live in app.py)."""
    tree = ast.parse(open(_REF, encoding="utf-8").read(), filename=_REF)
    st_calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name) and n.func.value.id == "st"
    ]
    assert not st_calls, "ui_reference.py must be pure HTML — no st.* calls"
    assert isinstance(render_reference({"Foo": "bar"}, ""), str)
