"""
test_framework_emoji_registry.py
================================
Pins the framework emoji registry (`_FW_META` in `ui/ui_tearsheet.py`) — the single source
of truth for the 37 guru-framework emojis. CLAUDE.md §7's hard rule is "no two frameworks
share an emoji"; this test enforces it against the CODE instead of a hand-maintained prose
table (which duplicated the dict and could silently drift).

`_FW_META` is a function-local dict, so we read it statically via AST (same approach as
test_tearsheet_stateless_contract) — no import / no refactor of the 3k-line UI module.
"""
import ast
import os
from collections import Counter

_TEARSHEET = os.path.join(os.path.dirname(__file__), "..", "ui", "ui_tearsheet.py")
_EXPECTED_COUNT = 37


def _fw_meta_emojis():
    """Extract the emoji (2nd tuple element) of every `_FW_META` entry from source."""
    with open(_TEARSHEET, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=_TEARSHEET)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "_FW_META" for t in node.targets)
            and isinstance(node.value, ast.Dict)
        ):
            emojis = []
            for val in node.value.values:
                # each value is a tuple: (COLORS[...], "<emoji>", "<description>")
                assert isinstance(val, ast.Tuple) and len(val.elts) >= 2, "malformed _FW_META entry"
                emoji_node = val.elts[1]
                assert isinstance(emoji_node, ast.Constant) and isinstance(emoji_node.value, str), \
                    "_FW_META emoji must be a string literal"
                emojis.append(emoji_node.value)
            return emojis
    raise AssertionError("_FW_META dict literal not found in ui_tearsheet.py")


def test_framework_emojis_are_unique():
    """CLAUDE.md §7: no two frameworks may share an emoji."""
    emojis = _fw_meta_emojis()
    dups = {e: c for e, c in Counter(emojis).items() if c > 1}
    assert not dups, f"Duplicate framework emoji(s) in _FW_META: {dups}"


def test_framework_registry_has_expected_count():
    """Count lock (sibling to FORENSIC_MAX_FLAGS): the registry holds exactly 37 frameworks.
    If a framework is added/removed, update _EXPECTED_COUNT here and the §7 pointer in CLAUDE.md."""
    assert len(_fw_meta_emojis()) == _EXPECTED_COUNT
