"""Guard: config.py is the single source of truth — it must contain NO dead constants.

Every module-level UPPER_CASE constant must be USED (referenced as a Name in Load context, or as an
attribute `.attr`) somewhere in the live code (core/ ui/ tools/ app.py / tests) OR internally by
another config constant. Crucially, merely being IMPORTED is NOT "used": `from config import X` is an
ast.ImportFrom node, not a Name reference — so an imported-but-never-read constant (the WAVE_DETECTION
class) is caught alongside the never-imported class (FRAMEWORK_TO_CATEGORY).

This converts the manual "grep each constant for usage" SSOT audit into a permanent invariant: the day
a future constant goes dead (a duplicate weights dict, an orphaned reverse-map), this test red-fails.
Companion to test_config_invariants.py — that pins config STRUCTURE; this pins config LIVENESS.
"""
import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_CONFIG = _ROOT / "config.py"


def _names_read(tree: ast.AST) -> set:
    """Every identifier READ in a tree: Name nodes in Load context + every Attribute's `.attr`.
    Deliberately excludes assignment targets (Store context) so a constant's own `X = …` definition
    does not count as a use of X — and excludes ImportFrom, so a bare import is not a use either."""
    used = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            used.add(n.id)
        elif isinstance(n, ast.Attribute):
            used.add(n.attr)
    return used


def _module_constants(tree: ast.AST) -> list:
    """Module-level UPPER_CASE assigned names — the policy constants config.py exists to hold."""
    out = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id.isupper() and len(t.id) > 2:
                    out.append(t.id)
    return out


def test_config_has_no_dead_constants():
    cfg_tree = ast.parse(_CONFIG.read_text(encoding="utf-8"))
    constants = _module_constants(cfg_tree)
    internal = _names_read(cfg_tree)   # a constant consumed by ANOTHER config constant is alive

    # Live-code corpus: every first-party .py EXCEPT config.py itself and this test file (so neither a
    # config self-reference nor a constant name typed here can mask a genuinely dead constant).
    files = [_ROOT / "app.py"]
    for sub in ("core", "ui", "tools", "tests"):
        d = _ROOT / sub
        if d.is_dir():
            files += list(d.rglob("*.py"))

    external = set()
    _skip = {_CONFIG.resolve(), Path(__file__).resolve()}
    for f in files:
        if f.resolve() in _skip:
            continue
        try:
            external |= _names_read(ast.parse(f.read_text(encoding="utf-8")))
        except (SyntaxError, UnicodeDecodeError):
            continue

    dead = sorted(c for c in constants if c not in external and c not in internal)
    assert not dead, (
        f"config.py defines {len(dead)} DEAD constant(s) — defined but never read in "
        f"core/ui/tools/app/tests (imported-but-unread counts as dead): {dead}. "
        f"Delete each (and any dead `from config import` of it) — config.py is the single source of truth."
    )
