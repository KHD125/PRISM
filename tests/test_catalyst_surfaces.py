"""Every catalyst the engine counts is shown, under ONE name, everywhere catalysts are shown.

Pinned 2026-10-05 (known-issues KI-10). The Discovery card never showed 🔥 Lynch Dream: the sidebar
could filter ~180 stocks by it while their cards carried no pill saying why they were there, and the
card called 🔥 Deleveraging "Deleveraging Cycle". The catalyst list is read from the engine's own
catalyst_count sum, so a new catalyst joins this contract the moment the engine counts it, and any
surface that misses it, or names it differently from the filter, fails here.
"""
import ast
import io
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _src(rel):
    return io.open(ROOT / rel, encoding="utf-8").read()


def _assigned(src, name):
    """The value node assigned to `name` anywhere in the module, function bodies included."""
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node.value
    raise AssertionError(f"{name} is no longer assigned")


def _engine_catalysts():
    """The columns summed into catalyst_count: the engine's own list of catalysts."""
    for node in ast.walk(ast.parse(_src("core/scoring_engine.py"))):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant)
                and t.slice.value == "catalyst_count" for t in node.targets):
            return sorted({n.slice.value for n in ast.walk(node.value)
                           if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant)})
    raise AssertionError("the engine no longer assigns catalyst_count")


def _names():
    """{column: label}. The sidebar filter's names are the canonical ones."""
    filt = ast.literal_eval(_assigned(_src("ui/ui_discovery.py"), "_CATALYSTS"))
    return {col: label for label, col in filt.items()}


def test_the_engine_still_counts_its_catalysts():
    """Teeth: the two tests below loop over this list, so an empty or truncated read would pass
    them while checking nothing."""
    cats = _engine_catalysts()
    assert len(cats) >= 6 and all(c.startswith("cat_") for c in cats), cats


def test_every_surface_shows_every_catalyst_under_the_filter_name():
    from ui.ui_reference_data import CONCEPT_REFERENCE
    names = _names()
    want = {c: names.get(c) for c in _engine_catalysts()}
    pills = _assigned(_src("ui/ui_tearsheet.py"), "_CAT_PILLS")
    surfaces = {
        "sidebar filter": names,
        "Market Pulse filters": {c: label for label, c in
                                 ast.literal_eval(_assigned(_src("app.py"), "_MP_CATALYSTS")).items()},
        "tear-sheet pills": {e.elts[0].value: e.elts[2].value for e in pills.elts},
    }
    problems = [f"{where}: {sorted(got.items())} but the engine counts {sorted(want.items(), key=str)}"
                for where, got in surfaces.items() if got != want]
    ref = [title for title, _ in CONCEPT_REFERENCE["🔥 Catalysts"]]
    if sorted(ref) != sorted(want.values(), key=str):
        problems.append(f"Reference: {ref} but the engine counts {sorted(want.values(), key=str)}")
    assert not problems, "\n".join(problems)


def test_the_discovery_card_shows_each_catalyst_under_the_filter_name(monkeypatch):
    """Behavioural, not a source scan: render the REAL card once with no catalyst and once per
    catalyst with only that flag on. Each flag must add exactly its own pill, named as the filter
    names it (matched as the whole pill text, so a longer label such as the old 'Deleveraging
    Cycle' does not pass as 'Deleveraging')."""
    import ui.ui_components as uc
    shown = []

    class _St:
        def markdown(self, html, **_):
            shown.append(html)

        def __getattr__(self, _):
            return lambda *a, **k: None

    monkeypatch.setattr(uc, "st", _St())
    cats, names = _engine_catalysts(), _names()
    base = {"name": "Test Ltd", "sector": "Sector", "industry": "Industry", **{c: 0 for c in cats}}

    def pills_on(**flags):
        shown.clear()
        uc.render_stock_card(pd.Series({**base, **flags}), show_scores=True)
        html = "".join(shown)
        return [names.get(c, c) for c in cats if f">{names.get(c, c)}<" in html]

    assert pills_on() == [], "a card with no catalyst shows a catalyst pill"
    problems = [f"{c}: the card shows {pills_on(**{c: 1})}, expected [{names.get(c, c)!r}]"
                for c in cats if pills_on(**{c: 1}) != [names.get(c, c)]]
    assert not problems, "\n".join(problems)
