"""Contract: every FIRST-PARTY name app.py imports actually resolves from its package/module.

WHY THIS EXISTS: app.py imports from the `ui` PACKAGE facade (`from ui import (...)`) and from
`config` / `core`. A facade that forgets to re-export a moved symbol crashes the app at STARTUP
(`ImportError: cannot import name '...' from 'ui'`) — yet this is INVISIBLE to the render tests,
because tests/test_ui_smoke.py drives the render functions via AppTest.from_function and never
executes app.py's module body or its `from ui import` line. (Exactly that gap let a green 1710-test
suite ship a broken app once — help_chip moved to ui_components but wasn't re-exported by ui/__init__.)

This test closes the gap cheaply: it STATICALLY parses app.py's first-party imports and asserts each
name is exported, with NO Streamlit runtime and NO CSV data. It never imports/executes app.py itself
(app.py runs st.* at module scope and autoloads data on import).
"""
import ast
import importlib
import re
from pathlib import Path

import pytest

_APP = Path(__file__).resolve().parent.parent / "app.py"


def _is_first_party(module: str) -> bool:
    return (
        module == "config"
        or module == "ui" or module.startswith("ui.")
        or module == "core" or module.startswith("core.")
    )


def _first_party_imports():
    """[(module, name), ...] for every `from <first-party> import name` in app.py (skips `*`)."""
    tree = ast.parse(_APP.read_text(encoding="utf-8"), filename="app.py")
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level:  # skip relative imports
            continue
        module = node.module or ""
        if not _is_first_party(module):
            continue
        for alias in node.names:
            if alias.name != "*":
                out.append((module, alias.name))
    return out


_IMPORTS = _first_party_imports()


def test_app_has_first_party_imports():
    """Guard against a silently-empty parse (wrong path / parser regression) that would make the
    parametrized contract below vacuously pass."""
    assert _IMPORTS, "parsed zero first-party imports from app.py — the path or parser is wrong"


def test_sel_mandate_never_dereferenced_unguarded():
    """`_sel_mandate` is None for the ⚙️ Custom mandate (and when switching to a profile that clears
    it), so a bare `_sel_mandate.<attr>` / `_sel_mandate[...]` raises AttributeError on the RUNNING
    app — invisible to render tests that never hit that branch. Shipped once: the Momentum profile
    cleared the mandate, and the Deep Scanner export's `_sel_mandate.replace(...)` crashed the deploy
    (2026-06-20). Every use must go through a None-safe form: `_sel_mandate or 'Custom'`,
    `_MANDATES.get(_sel_mandate, ...)`, or an `if _sel_mandate` guard."""
    src = _APP.read_text(encoding="utf-8")
    bad = re.findall(r"\b_sel_mandate\b\s*(?:\.\w+|\[)", src)
    assert not bad, (
        f"_sel_mandate dereferenced unguarded (it can be None): {bad}. "
        f"Use `_sel_mandate or 'Custom'` / `_MANDATES.get(_sel_mandate, {{}})` instead."
    )


@pytest.mark.parametrize("module,name", _IMPORTS, ids=[f"{m}.{n}" for m, n in _IMPORTS])
def test_app_first_party_import_resolves(module, name):
    """Every name app.py imports from a first-party package/module must actually be exported there —
    otherwise the app dies at startup with an ImportError that the render tests cannot see."""
    mod = importlib.import_module(module)
    assert hasattr(mod, name), (
        f"app.py does `from {module} import {name}`, but {module} does not export {name!r}. "
        f"This crashes the app at startup yet passes every render test — re-export it "
        f"(e.g. add it to {module}'s __init__ / module namespace)."
    )


def test_widget_callbacks_never_read_session_keys_unguarded():
    """PROD CRASH CLASS (KeyError '_w_mode', Streamlit Cloud 2026-08-24): a widget on_change/on_click
    callback runs BEFORE any script line of the rerun. When Cloud hibernates and restarts the app,
    the session is fresh and EMPTY while the user's browser still shows the old page — their first
    interaction fires the callback on a session_state holding nothing, so a raw
    st.session_state["key"] read crashes the whole app. Pytest cannot execute this path (it lives
    in the session lifecycle), so this static contract walks every function wired to on_change= /
    on_click= in app.py + ui/ and rejects any UNGUARDED subscript READ of st.session_state.
    Writes are safe (they run first and create the key); .get()/.pop(k, default) are safe; a read
    nested under an `if "key" in st.session_state` guard is safe."""
    import ast
    import os

    _root = os.path.join(os.path.dirname(__file__), "..")
    offenders = []
    for rel in ["app.py", os.path.join("ui", "ui_discovery.py"), os.path.join("ui", "ui_components.py")]:
        src = open(os.path.join(_root, rel), encoding="utf-8").read()
        tree = ast.parse(src)
        # names wired as callbacks: on_change=NAME / on_click=NAME keywords
        cb_names = {kw.value.id
                    for node in ast.walk(tree) if isinstance(node, ast.Call)
                    for kw in node.keywords
                    if kw.arg in ("on_change", "on_click") and isinstance(kw.value, ast.Name)}
        funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        for name in sorted(cb_names & set(funcs)):
            fn = funcs[name]
            guarded_spans = [
                (g.lineno, max(x.lineno for x in ast.walk(g) if hasattr(x, "lineno")))
                for g in ast.walk(fn)
                if isinstance(g, ast.If) and isinstance(g.test, ast.Compare)
                and any(isinstance(op, ast.In) for op in g.test.ops)
                and "session_state" in ast.dump(g.test)
            ]
            for node in ast.walk(fn):
                if (isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load)
                        and "session_state" in ast.dump(node.value)):
                    if not any(a <= node.lineno <= b for a, b in guarded_spans):
                        offenders.append(f"{rel}::{name} line {node.lineno}")
    assert not offenders, (
        "Unguarded st.session_state[...] READ inside a widget callback — this crashed prod "
        f"(KeyError on a hibernation-resurrected session): {offenders}. "
        'Wrap the read in `if "key" in st.session_state:` or use .get().'
    )
