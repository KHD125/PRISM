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


# ── Framework registry ON SCREEN (2026-08-28) ────────────────────────────────────────────────
def test_frameworks_are_searchable_on_screen():
    """Before 2026-08-28 the Markdown export documented all 37 frameworks while on-screen search
    could find only 6 — the answer to "what is Dhandho Asymmetry?" lived in the download but not
    the search box. The on-screen section renders from the SAME _FW_META-derived dict the export
    consumes, so the two can never disagree."""
    from ui.ui_reference import render_frameworks
    from ui.ui_tearsheet import _FW_META
    fw = {name: {"emoji": meta[1], "name": name, "desc": meta[2]} for name, meta in _FW_META.items()}
    html_all = render_frameworks(fw, "")
    for name in ("Dhandho Asymmetry", "QGLP", "100-Bagger"):
        assert name in html_all, f"{name} missing from the on-screen framework registry"
    assert render_frameworks(fw, "dhandho") != ""
    assert render_frameworks(fw, "zzz-no-such-framework") == ""


def test_app_renders_the_framework_registry():
    import io as _io, os
    src = _io.open(os.path.join(os.path.dirname(__file__), "..", "app.py"), encoding="utf-8").read()
    assert "render_frameworks(_fw_md, _ref_q)" in src, (
        "app.py no longer renders the framework registry on screen — the export/search asymmetry is back"
    )
    assert "Framework Registry" in src
