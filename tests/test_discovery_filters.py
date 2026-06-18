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
from ui.ui_discovery import _CHIP_META, _compute_active_chips

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


def test_active_groups_stay_open_across_reruns():
    """st.expander has no persistent open-state (no key in Streamlit 1.54) → it re-renders at its
    static `expanded=` default every rerun, so selecting a filter / clicking Clear-all would snap
    the box shut. _grp must keep a group OPEN when it holds an active filter (expanded or a > 0).
    Pin it so the fix can't silently revert to a static `expanded=expanded`."""
    # Assert the actual st.expander KWARG (`expanded=(...)`) — unique to the call. The docstring's
    # prose says "(expanded or a > 0)" WITHOUT the `expanded=` prefix, so this can't be satisfied by
    # the comment alone: a revert to a static `expanded=expanded` genuinely RED-fails.
    assert "expanded=(expanded or a > 0)" in _DISC, (
        "_grp's st.expander() must pass expanded=(expanded or a > 0) — otherwise every rerun "
        "(filter select / Clear-all) collapses the box the user is working in"
    )


def test_filter_labels_match_the_engine_verbatim():
    # cyclicality: import-pinned to the engine's source of truth
    for v in TIER_LABELS.values():
        assert v in _DISC, f"cyclicality label drifted from TIER_LABELS: {v!r}"
    # capital-phase: no exported constant, so cross-file pin (emitter + filter must agree)
    for p in _CAP_PHASES:
        assert p in _DISC, f"capital-phase label missing from ui_discovery: {p!r}"
        assert p in _DENG, f"capital-phase label drifted in data_engine (emitter): {p!r}"


# ── applied-filter chip strip ─────────────────────────────────────────────────
def test_compute_active_chips_per_kind():
    """The pure chip detector handles every filter shape: ms (first value +N), ms_count (count, no
    raw cat_* leak), sel ('All'=off), slider-max (active when < ceiling), slider-min (active when >0),
    bool. Mirrors the funnel's active-logic so chips and badges agree."""
    state = {"sb_cyc": ["Defensive", "Financials"], "sb_sector": "Steel", "sb_industry": "All",
             "sb_maxrf": 3, "sb_mincov": 0, "sb_minscore": 70, "sb_gate": True,
             "sb_catalyst": ["cat_capacity", "cat_oplev"]}
    chips = dict(_compute_active_chips(state, rf_max=28))
    assert chips["sb_cyc"] == "Cyclicality: Defensive +1"   # ms: first value + N
    assert chips["sb_sector"] == "Sector: Steel"            # sel active
    assert "sb_industry" not in chips                       # "All" = off
    assert chips["sb_maxrf"] == "Max Red Flags: ≤3"         # slider-max active (3 < 28)
    assert "sb_mincov" not in chips                         # 0 = off
    assert chips["sb_minscore"] == "Min Score: ≥70"         # slider-min active
    assert chips["sb_gate"] == "Gate-passed"               # bool active
    assert chips["sb_catalyst"] == "Catalyst: 2 selected"  # ms_count — no raw cat_* leak


def test_no_active_filters_yields_no_chips():
    """All-default state → empty chip list (so the strip renders nothing)."""
    assert _compute_active_chips({}, rf_max=28) == []
    assert _compute_active_chips({"sb_cyc": [], "sb_sector": "All", "sb_maxrf": 28, "sb_minq": 0,
                                  "sb_gate": False}, rf_max=28) == []


def test_every_funnel_key_has_a_chip_entry():
    """Every sb_* key counted in the funnel total MUST have a _CHIP_META entry — else a filter can be
    active yet produce no removable chip (the registration-drift guard, like the _grp/_active_n pair)."""
    import re
    i = _DISC.rindex("_active_n(")
    total_keys = set(re.findall(r'"(sb_\w+)"', _DISC[i:_DISC.index(")", i)]))
    missing = total_keys - {k for k, _, _ in _CHIP_META}
    assert not missing, f"filters counted in the funnel total but with no chip entry: {missing}"
