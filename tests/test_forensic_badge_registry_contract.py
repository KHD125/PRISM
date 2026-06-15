"""
Contract Test — Forensic Badge Registry ⇄ Engine Flag Set
=========================================================
Binds the UI's `_FLAG_DISPLAY` badge registry (ui/ui_tearsheet.py) to the engine's
`rf_*` flag set (core/forensic_engine.py). This is the drift guard that was MISSING when
rf_snoa (2026-06-13), rf_low_cfo_ebitda, and rf_wc_double_squeeze were added to the engine:
the engine counted/scored/penalised 28 flags while the UI could only render 25 as badges, so
788 stocks showed a "Red Flags / 28" KPI larger than their visible badges and 7 penalised
stocks rendered the cascade banner over an EMPTY severity section.

This source-scans both files (no import / no pipeline) and asserts:
  • every engine rf_ flag has a badge in _FLAG_DISPLAY (no flag is silently un-renderable)
  • _FLAG_DISPLAY has no ghost keys the engine never computes
  • the engine flag count == FORENSIC_MAX_FLAGS (the KPI denominator stays honest)

Isolation note: rf_ keys appear ELSEWHERE in ui_tearsheet.py (the `_get_flag_context`
`if rf_col == "rf_..."` branches end in a colon too), so a whole-file grep wrongly counts 26.
The UI regex is applied ONLY to the isolated `_FLAG_DISPLAY = { ... }` dict block.
"""

import os
import re
import sys

# ── Path bootstrap ────────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import FORENSIC_MAX_FLAGS

_FORENSIC_SRC = os.path.join(REPO_ROOT, "core", "forensic_engine.py")
_UI_SRC = os.path.join(REPO_ROOT, "ui", "ui_tearsheet.py")


def _engine_flags() -> set:
    """Every `df["rf_..."] =` flag assignment in the forensic engine."""
    src = open(_FORENSIC_SRC, encoding="utf-8").read()
    return set(re.findall(r'df\["(rf_[^"]+)"\]\s*=', src))


def _ui_flags() -> set:
    """Every `"rf_...":` key inside the isolated _FLAG_DISPLAY dict block only."""
    src = open(_UI_SRC, encoding="utf-8").read()
    m = re.search(r"_FLAG_DISPLAY\s*=\s*\{\n(.*?)\n\}", src, re.DOTALL)
    assert m, "Could not locate the _FLAG_DISPLAY = { ... } dict block in ui_tearsheet.py"
    return set(re.findall(r'"(rf_[a-z_]+)"\s*:', m.group(1)))


def test_badge_registry_matches_engine_flag_set():
    """The badge registry must render exactly the engine's rf_ flag set — no more, no less."""
    engine = _engine_flags()
    ui = _ui_flags()

    missing_in_ui = sorted(engine - ui)   # engine computes it, UI can't render it
    ghost_in_ui = sorted(ui - engine)     # UI badge for a flag the engine never computes

    assert engine == ui, (
        "Forensic badge registry has drifted from the engine flag set.\n"
        f"  Computed but NOT renderable (add to _FLAG_DISPLAY): {missing_in_ui}\n"
        f"  Renderable but NOT computed (remove ghost badge):   {ghost_in_ui}"
    )


def test_engine_flag_count_equals_forensic_max_flags():
    """FORENSIC_MAX_FLAGS is the KPI denominator — it must equal the live rf_ flag count."""
    engine = _engine_flags()
    assert len(engine) == FORENSIC_MAX_FLAGS, (
        f"Engine computes {len(engine)} rf_ flags but FORENSIC_MAX_FLAGS={FORENSIC_MAX_FLAGS}. "
        f"Flags: {sorted(engine)}"
    )
