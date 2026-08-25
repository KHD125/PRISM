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


# ══════════════════════════════════════════════════════════════════════════════
# COLOUR SEMANTICS — red means FAIL on every radar, never PASS (2026-08-25)
# ══════════════════════════════════════════════════════════════════════════════
def test_no_radar_uses_red_for_a_passing_pillar():
    """Found from a NALCO screenshot: the Lynch radar coloured a PASSING pillar with Lynch's brand
    red (#e74c3c) while six of the eight radars use red for FAILURE — so its two ✅ pillars
    rendered in red boxes and its two ❌ pillars in neutral grey. On a tab where the eye scans
    colour before icons, one panel spoke the opposite language.

    Pins the convention for every radar: the PASS branch of a pillar-colour ternary may never be a
    red, and the FAIL branch may never be the codebase's grey 'unknown' signal (which would make a
    failed gate read as missing data — the Mauboussin case, fixed in the same pass).

    CAN SLIM is exempt on the fail side by design: it deliberately reds only its critical pillars
    (C, A, M) and greys the rest, a documented severity distinction rather than an inversion.
    """
    import re
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "ui" / "ui_tearsheet.py").read_text(encoding="utf-8")
    REDS = ("#f85149", "#e74c3c", 'COLORS["red"]')
    GREY = ('COLORS["text_muted"]', "#8b949e")

    bad_pass, bad_fail = [], []
    for m in re.finditer(r"^\s*(clr\w*)\s*=\s*(.+?)\s+if passed else\s+(.+?)$", src, re.M):
        var, pass_expr, fail_expr = m.group(1), m.group(2), m.group(3)
        if any(r in pass_expr for r in REDS):
            bad_pass.append(f"{var}: PASS renders {pass_expr.strip()}")
        # allow CAN SLIM's documented conditional (critical letters red, others grey)
        if any(g == fail_expr.strip() for g in GREY):
            bad_fail.append(f"{var}: FAIL renders {fail_expr.strip()} (grey = 'unknown' elsewhere)")

    assert not bad_pass, (
        "a radar colours a PASSING pillar red — red must mean failure everywhere on the "
        f"Frameworks tab: {bad_pass}")
    assert not bad_fail, (
        "a radar colours a FAILED pillar with the grey 'unknown' signal, so it reads as missing "
        f"data rather than a failed gate: {bad_fail}")
