"""Contract: the two new Discovery filters (Cyclicality Tier + Sector Capital Phase) are wired
correctly into the cascade. Static source check — no Streamlit runtime, no data dir.

Pins exactly the two genuinely-new risk points (everything else rides the battle-tested
_ms_cascade / _ordered_present machinery the other 11 categorical filters already use):

  1. REGISTRATION — each new sb_* key must appear in BOTH the 🏢 Universe _grp() call (the group
     badge) AND the bottom _active_n() total (the funnel count). Miss either and the active-filter
     count silently under-reports (no crash, a real bug).
  2. LABEL DRIFT — the filter's option labels must match what the engine actually emits. Cyclicality
     is import-pinned to TIER_LABELS (single source of truth). Sector-capital-phase has no exported
     constant (the 3 strings are inline literals in data_engine + the tearsheet tile), so we cross-file
     pin: the literals must appear in BOTH the filter AND the engine emitter — a one-sided rename fails.

NOT here by design: a column-existence test — cyclicality_tier is already runtime-guarded by
test_cyclicality.py and sector_capital_phase by test_capital_cycle_sector.py (re-testing violates §2).
"""
from pathlib import Path

from core.cyclicality_map import TIER_LABELS

_ROOT = Path(__file__).resolve().parent.parent
_DISC = (_ROOT / "ui" / "ui_discovery.py").read_text(encoding="utf-8")
_DENG = (_ROOT / "core" / "data_engine.py").read_text(encoding="utf-8")
_CAP_PHASES = ["🔥 Hot Capital (caution)", "❄️ Capital Starved (opportunity)", "⚖️ Neutral"]


def test_new_keys_registered_in_both_spots():
    grp_line = next(l for l in _DISC.splitlines() if '_grp("🏢 Universe"' in l)
    # rindex → the FINAL _active_n( call (the funnel TOTAL); .index would grab the `def _active_n(`
    # definition / the in-_grp `a = _active_n()` call, whose block holds only *keys, not the sb_* list.
    i = _DISC.rindex("_active_n(")
    active_block = _DISC[i:_DISC.index(")", i)]
    for k in ("sb_cyc", "sb_capphase"):
        assert k in grp_line,     f"{k} missing from the 🏢 Universe _grp() — group badge will undercount"
        assert k in active_block, f"{k} missing from _active_n() — the funnel total will undercount"


def test_filter_labels_match_the_engine_verbatim():
    # cyclicality: import-pinned to the engine's source of truth
    for v in TIER_LABELS.values():
        assert v in _DISC, f"cyclicality label drifted from TIER_LABELS: {v!r}"
    # capital-phase: no exported constant, so cross-file pin (emitter + filter must agree)
    for p in _CAP_PHASES:
        assert p in _DISC, f"capital-phase label missing from ui_discovery: {p!r}"
        assert p in _DENG, f"capital-phase label drifted in data_engine (emitter): {p!r}"
