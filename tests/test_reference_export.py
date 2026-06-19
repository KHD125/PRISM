"""
test_reference_export.py
========================
Coverage contract for the Reference -> Markdown export (ui.ui_reference.build_reference_markdown).
The generator must emit EVERY glossary term, concept label, and red-flag description from the SAME
single-source dicts the in-app renderers consume — so the download, the on-screen Reference, and any
future docs/reference.md can never drift. Sibling of test_tooltip_integrity's coverage discipline.
Pure: no Streamlit, no I/O.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ui.ui_reference import build_reference_markdown
from ui.ui_components import _RAW_GLOSSARY
from ui.ui_reference_data import CONCEPT_REFERENCE
from ui.ui_tearsheet import _FLAG_DISPLAY


def _norm(s):
    """Whitespace-normalized form — matches the generator's one-line collapse, so coverage checks
    are robust to multi-space / newline source text."""
    return " ".join(str(s).split())


def _md():
    return build_reference_markdown(_RAW_GLOSSARY, CONCEPT_REFERENCE, _FLAG_DISPLAY)


def test_every_glossary_term_in_markdown():
    md = _md()
    missing = [t for t in _RAW_GLOSSARY if _norm(t) not in md]
    assert not missing, f"{len(missing)} glossary terms missing from export: {missing[:8]}"


def test_every_flag_and_concept_in_markdown():
    md = _md()
    miss_flag = [desc for desc, _sev in _FLAG_DISPLAY.values() if _norm(desc) not in md]
    assert not miss_flag, f"flags missing from export: {miss_flag[:5]}"
    miss_concept = [lbl for entries in CONCEPT_REFERENCE.values() for lbl, _exp in entries
                    if _norm(lbl) not in md]
    assert not miss_concept, f"concept labels missing from export: {miss_concept[:5]}"


def test_section_headers_present():
    md = _md()
    for h in ["# PRISM Reference", "## Glossary", "## Concepts", "## Forensic Red Flags"]:
        assert h in md, f"missing section header: {h}"


def test_frameworks_ready_both_ways():
    base = build_reference_markdown(_RAW_GLOSSARY, CONCEPT_REFERENCE, _FLAG_DISPLAY)
    assert "## Frameworks" not in base   # None -> section omitted, no crash
    fw = build_reference_markdown(_RAW_GLOSSARY, CONCEPT_REFERENCE, _FLAG_DISPLAY,
                                  frameworks={"x": {"emoji": "🎯", "name": "Test Framework"}})
    assert "## Frameworks" in fw and "Test Framework" in fw and "🎯" in fw


def test_pure_nonempty_and_module_has_no_streamlit():
    md = _md()
    assert isinstance(md, str) and len(md) > 500
    # ui_reference.py must stay pure (the module's "ZERO Streamlit calls" contract).
    path = os.path.join(os.path.dirname(__file__), "..", "ui", "ui_reference.py")
    with open(path, encoding="utf-8") as f:
        assert "import streamlit" not in f.read(), "ui_reference.py must not import streamlit"
