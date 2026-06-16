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
