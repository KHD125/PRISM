"""Contract for the Deep Scanner CSV export — the 📥 download button in app.py `with tabs[1]:`.
Pins two regressions the 2026-06-20 audit surfaced:

  1. The export must encode via ui_export._to_csv_bytes (UTF-8 BOM) so its emoji decision-columns
     (moat_growth_quad ⭐💀, smart_money_flow ⚪✅❌, weinstein_stage, buy_zone_label) render in Excel
     instead of mojibake — the sibling sidebar full-dump export already does exactly this.
  2. Every column the _DS_VIEWS presets surface (== the export's deduped union) must exist on the real
     scored frame, or it silently drops from BOTH the on-screen table and the export, with the button's
     "{n} columns" count quietly wrong and no error raised.

app.py's tab body is Streamlit runtime code (not unit-renderable), so finding #1 is pinned by a precise
AST check on the Deep Scanner block — the same house style as test_app_imports / test_tooltip_coverage.
"""
import ast
import contextlib
import io
from pathlib import Path

import pytest

_APP = Path(__file__).resolve().parent.parent / "app.py"
_DATA_DIR = Path(__file__).resolve().parent.parent / "Other Resources" / "CSV Data"


def _deep_scanner_block(tree):
    """The `with tabs[1]:` (Deep Scanner) With-node — scopes the search to this tab so it can't pick
    up a download_button from an unrelated tab (the Tear-Sheet and sidebar each have their own)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.With):
            for item in node.items:
                ctx = item.context_expr
                if (isinstance(ctx, ast.Subscript) and isinstance(ctx.value, ast.Name)
                        and ctx.value.id == "tabs"
                        and isinstance(ctx.slice, ast.Constant) and ctx.slice.value == 1):
                    return node
    return None


def _download_button_data_kw(block):
    """The `data=` keyword value-node of the Deep Scanner's st.download_button call (None if absent)."""
    for node in ast.walk(block):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "download_button"):
            for kw in node.keywords:
                if kw.arg == "data":
                    return kw.value
    return None


def test_export_encodes_through_bom_helper():
    """The Deep Scanner export must hand download_button bytes from _to_csv_bytes (UTF-8 BOM), not a
    bare DataFrame.to_csv() — otherwise its emoji decision-columns mojibake when opened in Excel."""
    block = _deep_scanner_block(ast.parse(_APP.read_text(encoding="utf-8"), filename="app.py"))
    assert block is not None, "could not locate the `with tabs[1]:` Deep Scanner block in app.py"
    data = _download_button_data_kw(block)
    assert data is not None, "Deep Scanner download_button has no data= keyword"
    assert (isinstance(data, ast.Call) and isinstance(data.func, ast.Name)
            and data.func.id == "_to_csv_bytes"), (
        "Deep Scanner export must encode via _to_csv_bytes(...) for an Excel-safe UTF-8 BOM "
        "(consistency with the sidebar full-dump export); found a different data= expression."
    )


def _ds_view_union(tree):
    """The deduped union of every _DS_VIEWS preset column — exactly the export's column set
    (`dict.fromkeys(... for _v in _DS_VIEWS.values() ...)` in app.py), AST-parsed (no execution)."""
    union: list = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "_DS_VIEWS" for t in node.targets)
                and isinstance(node.value, ast.Dict)):
            for preset in node.value.values:
                for elt in getattr(preset, "elts", []):
                    if isinstance(elt, ast.Constant) and elt.value not in union:
                        union.append(elt.value)
    return union


@pytest.mark.skipif(not _DATA_DIR.is_dir(),
                    reason="Local CSV data not present (code-only checkout) — needs the real scored frame")
def test_every_export_column_exists_on_scored_frame():
    """Every _DS_VIEWS preset column (== the export's deduped union) must resolve on the real scored
    frame. The export filters with `if _c in ds_df.columns`, so a renamed/typo'd engine column silently
    vanishes from BOTH the on-screen table and the CSV — the button's '{n} columns' count quietly drops
    and nothing raises. This locks the union against the live pipeline so that drift goes red."""
    union = _ds_view_union(ast.parse(_APP.read_text(encoding="utf-8"), filename="app.py"))
    assert len(union) >= 40, f"expected ~44 export columns parsed from _DS_VIEWS, got {len(union)}"

    from core.data_engine import (load_all_csvs, merge_datasets,
                                  coerce_numeric_columns, compute_derived_signals)
    from core import run_scoring_pipeline
    with contextlib.redirect_stdout(io.StringIO()):
        df = load_all_csvs("local")
        df = merge_datasets(df)
        df = coerce_numeric_columns(df)
        df = compute_derived_signals(df)
        df = run_scoring_pipeline(df)

    missing = [c for c in union if c not in df.columns]
    assert not missing, f"Deep Scanner export columns absent from the scored frame: {missing}"


def _download_button_by_label(tree, label_substr):
    """The st.download_button Call whose label (first positional arg, str or f-string) contains
    label_substr — targets one button by its visible text without depending on tab nesting."""
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "download_button" and node.args):
            label = node.args[0]
            if isinstance(label, ast.JoinedStr):
                text = "".join(v.value for v in label.values if isinstance(v, ast.Constant))
            elif isinstance(label, ast.Constant) and isinstance(label.value, str):
                text = label.value
            else:
                text = ""
            if label_substr in text:
                return node
    return None


def test_all_data_export_encodes_through_bom_helper():
    """The All Data tab's single-row export (📥 ... Full Data Row) must ALSO encode via _to_csv_bytes
    (UTF-8 BOM) — its Value column dumps every engine column, which includes emoji decision-strings
    (corporate_class 🏆, smart_money_flow ⚪/✅/❌, weinstein_stage, verdict emojis) + Indian names that
    mojibake in Excel under a bare DataFrame.to_csv(). Same regression class as the Deep Scanner export,
    on the third (previously unguarded) download button."""
    tree = ast.parse(_APP.read_text(encoding="utf-8"), filename="app.py")
    btn = _download_button_by_label(tree, "Full Data Row")
    assert btn is not None, "could not locate the All Data single-row export button in app.py"
    data = next((kw.value for kw in btn.keywords if kw.arg == "data"), None)
    assert data is not None, "All Data export download_button has no data= keyword"
    assert (isinstance(data, ast.Call) and isinstance(data.func, ast.Name)
            and data.func.id == "_to_csv_bytes"), (
        "All Data single-row export must encode via _to_csv_bytes(...) for an Excel-safe UTF-8 BOM "
        "(consistency with the Deep Scanner + sidebar exports); found a different data= expression."
    )
