"""Contract: the Market Pulse inner-tab set, pinned so a tab cannot be re-added — or a removed
renderer re-wired — without a conscious change.

STAGE 3 (2026-06-18) cut the set to {Tsunami, QGLP, Sectors}, dropping 💙 Blue Chips (fired on 0%
of the universe — dead) and 🚀 Tipping Points (brittle; folded into an enhanced Sectors view).
Those two stay out, and the renderer checks below are what keep them out.

🔭 MOSL ADDED 2026-08-27, and this file did its job: the change failed here first and had to be
justified rather than slipped in. Neither Stage-3 removal reason applies to it —

    not dead      586 stocks clear 2+ of the 10 Wealth Creation lenses; a clean pyramid from
                  999 at zero down to 1 stock at eight; deepest agreement 8 of 10
    not brittle   a VIEW over frameworks already implemented and audited, with no new gate and no
                  engine change; exact-token parsed and cross-checked against the authoritative
                  qglp_pass column (both 328)
    not redundant corr(convergence, composite_score) = 0.479 — it carries information the score
                  does not

The behaviour of the tab itself is pinned by tests/test_mosl_convergence_tab.py. This file pins
only the SET, which is the thing a future edit is most likely to change carelessly.
"""
import ast
from pathlib import Path

_APP = Path(__file__).resolve().parent.parent / "app.py"


def _mp_tab_labels():
    """The string-literal labels of the `_mp_tabs = st.tabs([...])` assignment."""
    tree = ast.parse(_APP.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if (isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "_mp_tabs" for t in n.targets)
                and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "tabs"
                and n.value.args and isinstance(n.value.args[0], ast.List)):
            return [e.value for e in n.value.args[0].elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    return None


def test_market_pulse_tab_set_is_exact():
    """Order matters as much as membership: every `with _mp_tabs[i]` body is bound by index, so a
    reordering silently renders the wrong content into the wrong tab."""
    labels = _mp_tab_labels()
    assert labels == ["🌊 Tsunami", "🏛️ QGLP", "🔭 MOSL", "📈 Sectors"], labels


def test_removed_renderers_are_not_called():
    src = _APP.read_text(encoding="utf-8")
    assert "render_bruised_blue_chips(" not in src, "Blue Chips list renderer must be gone"
    assert "render_multi_trillion_tipping_points(" not in src, "Tipping Points renderer must be gone"


def test_removed_renderers_are_not_imported():
    """No dangling import of the deleted list renderers (would crash app boot)."""
    src = _APP.read_text(encoding="utf-8")
    assert "render_bruised_blue_chips," not in src and "render_multi_trillion_tipping_points," not in src


def test_tab_extractor_has_teeth():
    labels = _mp_tab_labels()
    assert labels is not None and len(labels) == 4


def test_the_stage_3_removals_were_not_quietly_restored():
    """The two tabs Stage 3 deleted must stay deleted -- adding MOSL is not licence to bring back
    a dead 0%-firing tab or the brittle one that was folded into Sectors."""
    labels = _mp_tab_labels() or []
    joined = " ".join(labels)
    assert "Blue Chip" not in joined, "💙 Blue Chips fired on 0% of the universe; it stays out"
    assert "Tipping" not in joined, "🚀 Tipping Points was folded into Sectors; it stays out"
