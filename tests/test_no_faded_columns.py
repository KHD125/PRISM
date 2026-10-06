"""No table pins a column, because Streamlit draws every pinned column FADED.

Found 2026-10-06 after the user reported the Deep Scanner's Rank and Stock, and Market Pulse's
Sector / Industry name and Count, reading grey. Streamlit 1.54 hard-codes it in its own frontend:
every cell of a pinned column gets `style: isPinned ? "faded" : "normal"`. A side-by-side test page
showed that cell styling (a pandas Styler colour) cannot override it, and that the same column
unpinned renders in full white. Pinning was only there so a sideways scroll keeps the name in
view, and every view now fits a 1536px laptop screen (tests/test_deep_scanner_views.py), so the
name is on screen anyway. The most important column in a table should never be its dimmest.
"""
import ast
import glob
import io
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
SOURCES = [os.path.join(ROOT, "app.py")] + sorted(glob.glob(os.path.join(ROOT, "ui", "*.py")))


def test_no_table_pins_a_column():
    pinned = []
    for path in SOURCES:
        tree = ast.parse(io.open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and any(k.arg == "pinned" and not (isinstance(k.value, ast.Constant)
                                                                                and k.value.value in (False, None))
                                                  for k in node.keywords):
                pinned.append(f"{os.path.basename(path)}:{node.lineno}")
    assert not pinned, ("a pinned column is drawn faded (grey) by Streamlit, so the name a reader looks for "
                        f"first becomes the dimmest text in the table: {pinned}")


def test_streamlit_still_fades_pinned_columns():
    """Tripwire on the REASON. If a Streamlit upgrade stops fading pinned columns, this fails: pinning
    the name columns (so they stay in view on a narrow screen) becomes worth revisiting."""
    import streamlit
    js = glob.glob(os.path.join(os.path.dirname(streamlit.__file__), "static", "static", "js", "*.js"))
    assert js, "Streamlit's frontend bundle was not found"
    faded = any('isPinned?"faded"' in io.open(f, encoding="utf-8", errors="ignore").read() for f in js)
    assert faded, ("this Streamlit no longer draws pinned columns faded — pinning the name columns may be "
                   "worth bringing back (see this file's docstring)")
