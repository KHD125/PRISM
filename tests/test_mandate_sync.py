"""Contract: the mandate↔override sync — sel_mandate is DERIVED from the active (mode, profile).

Before this fix, sel_mandate was set only by the mandate buttons and never reconciled with the
Advanced Override, so using the override to pick a profile that didn't match the clicked mandate left
the button highlight + the card showing a stale, contradictory mandate (e.g. "Coffee Can · Growth").
Now the highlighted mandate is whichever one's (mode, profile) matches the active combo, or None =
"⚙️ Custom". These tests pin the reverse-map (round-trip + uniqueness + Custom) and that app.py wires
it None-safely (no bare _MANDATES[None] KeyError path).
"""
import ast
import re
from pathlib import Path

_APP = Path(__file__).resolve().parent.parent / "app.py"


def _mandates() -> dict:
    """Extract the inline _MANDATES dict from app.py via AST (no Streamlit import / no app run)."""
    for node in ast.walk(ast.parse(_APP.read_text(encoding="utf-8"))):
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "_MANDATES" for t in node.targets)):
            return ast.literal_eval(node.value)
    raise AssertionError("_MANDATES not found in app.py")


def test_reverse_map_round_trips_and_is_unique():
    """Every mandate's (mode, profile) must resolve back to that mandate, and no two mandates may
    share a combo (else the derived highlight would be ambiguous)."""
    M = _mandates()
    combo = {(v["mode"], v["profile"]): k for k, v in M.items()}
    assert len(combo) == len(M), "two mandates share a (mode, profile) combo — reverse-map ambiguous"
    for k, v in M.items():
        assert combo[(v["mode"], v["profile"])] == k, f"{k} does not round-trip"


def test_override_only_combos_are_custom():
    """Combos no mandate uses (Growth, Defensive, or a cross-mode pair) must map to None → the card
    shows '⚙️ Custom', never a stale mandate name."""
    M = _mandates()
    combo = {(v["mode"], v["profile"]): k for k, v in M.items()}
    for c in [("Fundamental", "Growth"), ("Hybrid", "Defensive"), ("Technical", "Quality")]:
        assert combo.get(c) is None, f"{c} should be a Custom combo (None), got {combo.get(c)}"


def test_app_wires_derived_mandate_none_safely():
    """app.py must derive sel_mandate from the reverse-map, expose a '⚙️ Custom' label for None, and
    never bare-index _MANDATES[_sel_mandate] without a truthiness guard (None would KeyError)."""
    src = _APP.read_text(encoding="utf-8")
    assert "_MANDATE_BY_COMBO" in src, "app.py must build the (mode,profile)->mandate reverse-map"
    assert "_MANDATE_BY_COMBO.get(" in src, "sel_mandate must be DERIVED from the reverse-map"
    assert '_sel_mandate or "Custom"' in src, "must expose a Custom label for None (icon supplies the gear)"
    for m in re.finditer(r"_MANDATES\[_sel_mandate\]", src):
        window = src[m.start(): m.start() + 80]
        assert "if _sel_mandate" in window, (
            f"unguarded _MANDATES[_sel_mandate] (None would KeyError) near: {window!r}"
        )
