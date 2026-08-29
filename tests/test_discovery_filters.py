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

from config import FRAMEWORK_CATEGORIES
from core.cyclicality_map import TIER_LABELS
from ui.ui_discovery import _CHIP_META, _compute_active_chips, _family_count

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
    # GENERALIZED 2026-08-28: this pin used to check only the two keys it was written for, which is
    # how sb_wealthtier shipped registered in the chips and the group badge but MISSING from the
    # funnel total — the headline read "No filters — full universe" over a filtered count. Every
    # chip-registered key must appear in the funnel total, so the NEXT filter cannot repeat it.
    from ui.ui_discovery import _CHIP_META
    missing = [k for k, _h, _kind in _CHIP_META if k not in active_block]
    assert not missing, (
        f"chip-registered filter keys missing from the funnel _active_n() total (the headline "
        f"will undercount active filters): {missing}"
    )


def test_clear_all_is_always_an_on_click_callback():
    """THE PARKED CLEAR-ALL BUG's call-site discipline (mechanised 2026-08-28): an inline
    `if st.button(...): clear_all_filters()` from the MAIN BODY runs after the sidebar widgets
    instantiated, and deleting an instantiated widget's key lets the frontend resurrect its value
    on the rerun — filters come back, intermittently, and AppTest (no frontend) can never catch
    it. So the discipline is pinned structurally: every clear button wires on_click, no call site
    invokes clear_all_filters() inline, and the function itself contains no st.rerun() (a no-op
    inside callbacks, and the tell-tale of the old inline pattern)."""
    app_src = (_ROOT / "app.py").read_text(encoding="utf-8")
    for src, where in ((_DISC, "ui_discovery.py"), (app_src, "app.py")):
        assert "clear_all_filters()" not in src.replace("def clear_all_filters()", ""), (
            f"an INLINE clear_all_filters() call is back in {where} — the resurrection bug returns"
        )
    assert app_src.count("on_click=clear_all_filters") == 2, "the two empty-state buttons must wire on_click"
    assert _DISC.count("on_click=clear_all_filters") == 1, "the sidebar button must wire on_click"
    fn = _DISC[_DISC.index("def clear_all_filters"):_DISC.index("def render_discovery_sidebar")]
    # Match the STATEMENT, not the docstring's prose mention of it (the substring-scan-over-source
    # lesson, learned repeatedly: prose naming a banned thing is not the banned thing).
    import re as _re
    assert not _re.search(r"^\s+st\.rerun\(\)\s*$", fn, _re.M), (
        "clear_all_filters must not call st.rerun() (no-op in callbacks)"
    )


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


# ── 🎭 Framework Family filter (min-count per family, AND across families) ──────
def test_family_filter_registered_in_group_and_funnel():
    """sb_fwfam must be wired into BOTH the 🧬 Frameworks _grp() (group badge) AND the final
    _active_n() funnel total — the same registration contract as every other filter. Its min-count
    PARAMETER (sb_fwfam_min) is NOT an independent filter and must NOT be counted in the funnel."""
    grp_line = next(l for l in _DISC.splitlines() if '_grp("🧬 Frameworks"' in l)
    i = _DISC.rindex("_active_n(")
    active_block = _DISC[i:_DISC.index(")", i)]
    assert "sb_fwfam" in grp_line, "sb_fwfam missing from the 🧬 Frameworks _grp() — group badge undercounts"
    assert "sb_fwfam" in active_block, "sb_fwfam missing from _active_n() — funnel total undercounts"
    assert "sb_fwfam_min" not in active_block, "sb_fwfam_min is a parameter, not a counted filter"


def test_family_filter_sourced_from_framework_categories():
    """The family options are built from the live FRAMEWORK_CATEGORIES taxonomy (single source), so a
    renamed/added family can never silently desync the filter from the cards' category chips."""
    assert "FRAMEWORK_CATEGORIES" in _DISC, "family filter must read FRAMEWORK_CATEGORIES, not a hardcoded list"


def test_family_count_boundary_safe_and_correct():
    """_family_count counts only EXACT framework tokens (boundary-safe) — 'Bruised Blue Chip' must
    NOT match 'Bruised Blue Chip 29' — and sums across the family's member frameworks."""
    import pandas as pd
    df = pd.DataFrame({"frameworks_passed": [
        "Coffee Can, Diamond, Quality Compounder",   # 3 Moats frameworks
        "QGLP, CAN SLIM",                            # 0 Moats
        "Bruised Blue Chip",                         # boundary: NOT the '…29' framework
    ]})
    moats = next(fws for (_e, lbl, _c, fws) in FRAMEWORK_CATEGORIES if lbl == "Moats")
    assert list(_family_count(df, moats)) == [3, 0, 0]
    mosl = next(fws for (_e, lbl, _c, fws) in FRAMEWORK_CATEGORIES if lbl == "MOSL")
    assert list(_family_count(df, mosl)) == [0, 1, 0]  # row C ('Bruised Blue Chip') must score 0


def test_family_count_handles_missing_and_empty():
    """No frameworks_passed column, an empty fw list, or NaN cells → all-zero (never crashes)."""
    import pandas as pd
    assert list(_family_count(pd.DataFrame({"x": [1, 2]}), ["Coffee Can"])) == [0, 0]
    df = pd.DataFrame({"frameworks_passed": ["Coffee Can", None]})
    assert list(_family_count(df, [])) == [0, 0]
    assert list(_family_count(df, ["Coffee Can"])) == [1, 0]


# ══════════════════════════════════════════════════════════════════════
# Blank labels must never become dropdown options (added 2026-08-23)
# ══════════════════════════════════════════════════════════════════════
# `_ordered_present` built options with frame[col].dropna() — which strips NaN but NOT the
# empty string. The "no verdict from a data hole" pass makes an unknown label "" instead of
# fabricating a verdict, so 7 dropdowns began offering an unlabelled, unusable checkbox
# (peg_zone, mef_label, moat_growth_quad, cash_machine_label, d49_momentum_quality, plus
# ep_power_curve / earnings_power_box which carried default="" all along). Selecting it
# silently narrowed the universe to rows the engine deliberately declined to judge.


def _ui_discovery_source() -> str:
    import io as _io
    import os as _os
    return _io.open(_os.path.join(_os.path.dirname(__file__), "..", "ui", "ui_discovery.py"),
                    encoding="utf-8").read()


def test_option_builder_excludes_blank_labels():
    src = _ui_discovery_source()
    i = src.find("def _ordered_present")
    assert i > 0, "helper not found"
    block = src[i:i + 900]
    assert "dropna()" in block, "helper no longer drops NaN"
    assert "if v.strip()" in block, "helper does not exclude blank labels"


def test_blank_never_offered_across_label_columns():
    """Exercise the REAL helper (module-level since the ❔ Unknown pass, so import it directly)."""
    import pandas as _pd

    from ui.ui_discovery import _ordered_present as fn

    frame = _pd.DataFrame({
        "peg_zone":  ["🟢 Fair PEG", "", "🔴 Overpriced", None],
        "mef_label": ["", "   ", "✅ Intact", "🔴 Degrading"],
    })
    for col, order in [("peg_zone",  ["🟢 Fair PEG", "🔴 Overpriced"]),
                       ("mef_label", ["✅ Intact", "🔴 Degrading"])]:
        opts = fn(frame, col, order)
        assert "" not in opts, f"{col}: blank option offered -> {opts}"
        assert not any(o.strip() == "" for o in opts), f"{col}: whitespace option -> {opts}"
        for real in order:
            assert real in opts, f"{col}: lost a real label {real!r}"


# ══════════════════════════════════════════════════════════════════════
# ❔ Unknown option + zero-results culprit (added 2026-08-23)
# ══════════════════════════════════════════════════════════════════════
# Two approved UX upgrades, one shared substrate:
#   1. The honest blanks the "no verdict from a data hole" pass produces (255 peg_zone,
#      177 mef_label, 133 ep_power_curve, …) must be SELECTABLE — a "❔ Unknown" option that
#      matches NaN/blank rows — not just hidden. Weinstein already emits a LITERAL "❔ Unknown"
#      label, so the sentinel must match by value AND union the holes (never replace value-match).
#   2. When the cascade hits 0 results the funnel must NAME the filter that emptied it, which
#      requires every one of the ~32 filter applications to route through one choke point.


def test_unknown_option_appended_when_blanks_present():
    """_ordered_present appends ❔ Unknown (last) iff the cascade frame holds NaN/blank rows —
    and never duplicates it when the engine itself emits the literal label (Weinstein)."""
    import pandas as _pd

    from ui.ui_discovery import _UNKNOWN, _ordered_present

    frame = _pd.DataFrame({
        "peg_zone":  ["🟢 Fair PEG", "", "🔴 Overpriced", None],       # holes -> sentinel offered
        "mef_label": ["✅ Intact", "🔴 Degrading", "✅ Intact", "✅ Intact"],  # fully labelled -> no sentinel
        "weinstein_stage": ["📈 Stage 2 Advancing", "❔ Unknown", "❔ Unknown", None],  # literal + hole
    })
    opts = _ordered_present(frame, "peg_zone", ["🟢 Fair PEG", "🔴 Overpriced"])
    assert opts[-1] == _UNKNOWN, f"sentinel missing/not last -> {opts}"
    assert _UNKNOWN not in _ordered_present(frame, "mef_label", ["✅ Intact", "🔴 Degrading"]), \
        "sentinel offered for a fully-labelled column"
    wein = _ordered_present(frame, "weinstein_stage", ["📈 Stage 2 Advancing", "❔ Unknown"])
    assert wein.count(_UNKNOWN) == 1, f"literal engine label duplicated the sentinel -> {wein}"


def test_label_mask_sentinel_matches_holes_and_literal():
    """_label_mask: real labels match by value; the ❔ Unknown sentinel additionally claims the
    honest holes (NaN / '' / whitespace) AND still matches a literal '❔ Unknown' the engine
    emits — so 'Unknown' always means the full 'engine declined to judge' set."""
    import pandas as _pd

    from ui.ui_discovery import _UNKNOWN, _label_mask

    s = _pd.Series(["🟢 Fair PEG", "", "   ", None, "❔ Unknown", "🔴 Overpriced"])
    assert list(_label_mask(s, ["🟢 Fair PEG"])) == [True, False, False, False, False, False]
    assert list(_label_mask(s, [_UNKNOWN])) == [False, True, True, True, True, False]
    assert list(_label_mask(s, ["🔴 Overpriced", _UNKNOWN])) == [False, True, True, True, True, True]
    assert list(_label_mask(s, [])) == [False] * 6, "empty selection must match nothing"


def test_first_zero_filter_names_first_culprit():
    """The choke point records the FIRST filter that empties a previously non-empty cascade —
    never a later filter applied to an already-empty frame, never a filter that leaves rows."""
    import pandas as _pd

    from ui.ui_discovery import _first_zero_filter

    df = _pd.DataFrame({"x": [1, 2, 3]})
    log = []
    out = _first_zero_filter(df, df["x"] > 1, "Keeps Rows", log)
    assert len(out) == 2 and log == [], "a filter that leaves rows must not be logged"
    out = _first_zero_filter(out, out["x"] > 99, "The Culprit", log)
    assert len(out) == 0 and log == ["The Culprit"]
    out = _first_zero_filter(out, out["x"] > 0, "Too Late", log)
    assert log == ["The Culprit"], "an already-empty frame must not overwrite the culprit"


def test_zero_culprit_reaches_the_main_panel_empty_state():
    """Two places report "no stocks match": the sidebar funnel and the Discovery main-panel card.
    They must name the SAME culprit — a funnel that says "PEG Zone removed the last stocks" beside
    a main panel that says only "loosen one" makes the user hunt for what the code already knows.

    The sidebar publishes the culprit on `filt.attrs` (the same channel the pipeline already uses
    for `detected_market_regime`); app.py's empty branch reads it. Set on `filt` AFTER the .copy(),
    so this never depends on pandas propagating attrs through an operation — which differs between
    the local pandas 3 and prod's pandas 2.
    """
    disc = _ui_discovery_source()
    app_src = (_ROOT / "app.py").read_text(encoding="utf-8")
    assert 'attrs["zero_culprit"]' in disc, \
        "render_discovery_sidebar must publish the culprit on filt.attrs"
    assert "zero_culprit" in app_src, \
        "the Discovery main-panel empty state must read filt.attrs['zero_culprit']"
    # the publish must happen on `filt` (post-copy), never on the cascade frame `_cf`
    assert '_cf.attrs["zero_culprit"]' not in disc, \
        "publish on filt (post-copy), not on _cf — attrs propagation through .copy() is not a contract"


def _zero_culprit_app():
    """Mini-app for AppTest: drives the REAL sidebar over a 2-row synthetic frame (no CSVs needed,
    fully deterministic) and prints what the caller receives."""
    import pandas as _pd
    import streamlit as _st

    from ui.ui_discovery import render_discovery_sidebar

    df = _pd.DataFrame({
        "name":             ["A", "B"],
        "sector":           ["X", "X"],
        "industry":         ["Y", "Y"],
        "conviction_tier":  [1, 2],
        "piotroski_fscore": [5, 6],
        "red_flag_count":   [0, 1],
        "quality_score":    [10.0, 12.0],
        "composite_score":  [10.0, 12.0],
    })
    filt = render_discovery_sidebar(df)
    _st.text(f"N={len(filt)}")
    _st.text(f"CULPRIT={filt.attrs.get('zero_culprit', '<MISSING>')}")


def test_sidebar_publishes_the_culprit_through_the_real_widget_machinery():
    """End-to-end on the ONE link the pure unit tests can't reach: that the culprit survives out of
    render_discovery_sidebar to the caller, through real Streamlit widgets. Both synthetic rows score
    10-12, so a Min-Composite of 50 empties the frame and 'Min Score' must be named."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_zero_culprit_app)
    at.session_state["sb_minscore"] = 50          # above both composites -> guaranteed zero
    at.run()
    assert not at.exception, f"sidebar raised: {at.exception}"
    out = [t.value for t in at.text]
    assert "N=0" in out, f"filter did not empty the frame: {out}"
    assert "CULPRIT=Min Score" in out, f"culprit not published to the caller: {out}"


def test_no_culprit_published_when_results_remain():
    """The inverse: with no filter active the frame is full, so the culprit must be EMPTY — never a
    stale name that would make the main panel accuse an innocent filter."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_zero_culprit_app)
    at.run()
    assert not at.exception, f"sidebar raised: {at.exception}"
    out = [t.value for t in at.text]
    assert "N=2" in out, f"unfiltered frame should keep both rows: {out}"
    assert "CULPRIT=" in out, f"culprit key must exist and be empty: {out}"


def test_all_filter_sites_route_through_choke_point():
    """Static pin: NO filter application may bypass the _narrow choke point (`_cf = _cf[` == 0),
    and the funnel's zero-state must actually read the culprit log."""
    src = _ui_discovery_source()
    n_direct = src.count("_cf = _cf[")
    assert n_direct == 0, (
        f"{n_direct} filter site(s) bypass the choke point — every application must be "
        "`_cf = _narrow(_cf, <mask>, <label>)` so the funnel can name a zero-results culprit"
    )
    assert "_first_zero_filter" in src, "choke-point helper not wired"
    assert src.count("_zero_at") >= 2, "funnel zero-state does not read the culprit log"


def test_every_non_multiselect_widget_seeds_its_key_before_instantiating():
    """THE STEEL REPRO (2026-08-29, user-reported the day after the on_click fix): deleting a
    widget key does NOT clear the widget's frontend state. A selectbox/slider/checkbox that
    renders with no session value asks the FRONTEND, which still remembers the old pick — so a
    cleared Sector=Steel resurrected on the next rerun. _ms_cascade multiselects were always
    immune because they unconditionally re-seed their key pre-instantiation; this pins the same
    discipline onto every non-multiselect sb_ widget, which is what makes delete-based Clear-All
    authoritative for ALL widget kinds. AppTest cannot catch a regression here (no frontend to
    resurrect from), so the seed is pinned structurally."""
    for key, seed in [
        ("sb_sector",    '"sb_sector" not in st.session_state'),
        ("sb_industry",  '"sb_industry" not in st.session_state'),
        ("sb_maxrf",     'st.session_state.setdefault("sb_maxrf"'),
        ("sb_mincov",    'st.session_state.setdefault("sb_mincov"'),
        ("sb_hidestale", 'st.session_state.setdefault("sb_hidestale"'),
        ("sb_gate",      'st.session_state.setdefault("sb_gate"'),
        ("sb_minq",      'st.session_state.setdefault("sb_minq"'),
        ("sb_minscore",  'st.session_state.setdefault("sb_minscore"'),
        ("sb_fwfam_min", '"sb_fwfam_min" not in st.session_state'),
    ]:
        assert seed in _DISC, (
            f"{key} lost its seed-before-instantiate guard — a cleared value will resurrect "
            f"from frontend widget state (the Steel repro)"
        )
        # the seed must appear BEFORE the widget that uses the key
        assert _DISC.index(seed) < _DISC.index(f'key="{key}"'), (
            f"{key}'s seed sits after its widget — it must run before instantiation"
        )
