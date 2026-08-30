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

💹 WEALTH ADDED 2026-08-27 (same session, later), and again this file forced the justification:
    not dead      every one of its six tiers is populated on live data (BUY★ 12% · BUY 5% ·
                  WATCH★ 9% · WATCH 32% · AVOID 34% · N/A 8%) — pinned by tests/test_wealth_tier.py
    not brittle   a pure READ of the wealth_* columns compute_verdict materializes; the tab holds
                  zero logic, so the tier a snapshot captures equals the tier displayed
    not redundant it answers the question no other surface does — "is it becoming more valuable?"
                  — and provably does not collapse into verdict_direction (the engine's complete
                  18-stock BUY list splits across four wealth tiers)

🏭 INDUSTRY ADDED 2026-08-28, and it clears the same three bars:
    not dead      76 industries hold ≥8 stocks (54 at ≥10) out of 355 — against 81 sectors, so the
                  view is 4.4× finer and still well populated at its default floor
    not brittle   a pure groupby over columns that already exist (industry, sector,
                  composite_score, gate_pass). No new gate, no engine change, nothing to calibrate
    not redundant THE STRONGEST CASE OF THE FOUR. Sector averaging destroys real dispersion: the
                  six sizeable industries inside Pharmaceuticals run 18.1 → 51.3 on average
                  composite — a 33-point spread the Sectors tab reports as ONE number (FMCG 22.9,
                  Auto Ancillaries 22.6). 20 of those 76 industries sit more than 5 points from
                  their parent sector's average, from Pharma - MNC bulk Drugs at +22.7 to Auto
                  Ancillaries - Gears at −12.9. The tab makes that gap its sort key.
                  Note it is NOT a drill-down: 136 of 355 industries span more than one sector, so
                  the hierarchy the name suggests does not exist — see tests/test_industry_tab.py.

The behaviour of the tabs themselves is pinned by tests/test_mosl_convergence_tab.py and
tests/test_wealth_tier.py. This file pins only the SET, which is the thing a future edit is most
likely to change carelessly.
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
    assert labels == ["🌊 Tsunami", "🏛️ QGLP", "🔭 MOSL", "💹 Wealth", "📈 Sectors",
                      "🏭 Industry"], labels


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
    assert labels is not None and len(labels) == 6


def test_the_stage_3_removals_were_not_quietly_restored():
    """The two tabs Stage 3 deleted must stay deleted -- adding MOSL is not licence to bring back
    a dead 0%-firing tab or the brittle one that was folded into Sectors."""
    labels = _mp_tab_labels() or []
    joined = " ".join(labels)
    assert "Blue Chip" not in joined, "💙 Blue Chips fired on 0% of the universe; it stays out"
    assert "Tipping" not in joined, "🚀 Tipping Points was folded into Sectors; it stays out"


# ── Fragment boundaries (2026-08-29) ─────────────────────────────────────────────────────────
def _app_src():
    import io as _io, os
    return _io.open(os.path.join(os.path.dirname(__file__), "..", "app.py"), encoding="utf-8").read()


def test_market_pulse_is_a_fragment_called_in_its_tab():
    """Market Pulse is market-wide by design (reads module `df` only — verified: zero filt/attrs
    reads, mp_* keys consumed nowhere else). Its in-tab controls cost 951 ms pre-fragment, ~97%
    of it re-rendering the OTHER tabs. The fragment scopes them to this tab."""
    src = _app_src()
    i = src.index("def _render_market_pulse():")
    deco = src[:i].rstrip().splitlines()[-1]
    assert deco.strip() == "@st.fragment", "Market Pulse lost its @st.fragment decorator"
    assert "with tabs[3]:" + chr(10) + "    _render_market_pulse()" in src, (
        "the fragment is no longer called in tab 3")


def test_reference_is_a_fragment_called_in_its_tab():
    src = _app_src()
    i = src.index("def _render_reference():")
    deco = src[:i].rstrip().splitlines()[-1]
    assert deco.strip() == "@st.fragment", "Reference lost its @st.fragment decorator"
    assert "with tabs[5]:" + chr(10) + "    _render_reference()" in src


def test_config_tab_is_never_fragmented():
    """THE CORRECTNESS TOMBSTONE. cfg_mode re-ranks the universe: the top of the script reads it
    to build the scored frame. Fragmented, an Analysis-Mode change would rerun only the fragment
    and every tab would show STALE RANKINGS. A 2026-08-29 audit proposed fragmenting Config as a
    speedup; rejected for exactly this reason — this pin keeps it rejected."""
    src = _app_src()
    cfg = src[src.index("# TAB 5: CONFIGURATION"):src.index("# TAB 6: REFERENCE")]
    assert "@st.fragment" not in cfg, (
        "Config was fragmented — cfg_mode changes will no longer recompute the scored frame"
    )
    assert "NEVER FRAGMENT THIS TAB" in cfg, "the tombstone comment explaining WHY is gone"
    assert 'key="cfg_mode"' in cfg, "cfg_mode moved out of Config — re-verify the fragment safety story"


# ── Lens-filter rows on QGLP / MOSL / Wealth (2026-08-30) ────────────────────────────────────
def test_lens_rows_wired_into_the_three_tabs():
    """QGLP and MOSL had ZERO controls over 328/586-row cohorts; the lens row (Sector · Wealth
    Tier · Market Cap · Catalyst · conditional Clear) fixes that. Wealth keeps its own Tier
    selectbox, so its row is with_tier=False — passing True there would render a SECOND tier
    control beside the first."""
    src = _app_src()
    assert '_mp_lens_row(_mp_qglp, "qglp")' in src, "QGLP tab lost its lens row"
    assert '_mp_lens_row(_mosl, "mosl")' in src, "MOSL tab lost its lens row"
    assert '_mp_lens_row(_wl, "w", with_tier=False)' in src, (
        "Wealth tab's lens row must be with_tier=False — its own Tier selectbox already exists")


def test_lens_clear_sets_all_and_never_deletes():
    """The reset callback must SET each key back to "All" — `del` on an instantiated widget's
    key lets the frontend resurrect the stale value on the next rerun (the Steel bug class).
    And the 🧹 button must be CONDITIONAL: rendered only when a filter is active, in a fixed
    slot (zero furniture idle, zero layout jump)."""
    src = _app_src()
    i = src.index("def _mp_clear_lens(")
    fn = src[i:src.index("def _mp_lens_row(")]
    assert "st.session_state[k] = v" in fn, "the reset no longer SETS keys to their defaults"
    assert "del " not in fn, "del in the lens reset — the resurrection class returns"
    j = src.index("if _n_active:")
    assert "st.button" in src[j:j + 250] and "_mp_clear_lens" in src[j:j + 250], (
        "the Clear button is no longer conditional on an active filter")


def test_lens_row_seeds_and_stale_guards_every_key():
    """Widget-state law, both halves: every key seeded BEFORE its selectbox instantiates, and
    re-seeded to "All" when a remembered value no longer exists in the cohort's options
    (a keyed selectbox whose state is missing from its options raises)."""
    src = _app_src()
    i = src.index("def _mp_lens_row(")
    fn = src[i:src.index("    # ── Inner navigation tabs")]
    assert 'st.session_state.setdefault(k, "All") not in _opts_for[k]' in fn
    assert fn.index("_opts_for[k]") < fn.index("st.selectbox"), "seeding must precede instantiation"


def test_mp_catalysts_mirrors_ui_discovery():
    """One catalyst vocabulary: app.py's _MP_CATALYSTS literal must equal ui_discovery's
    _CATALYSTS literal (label -> flag column). A drifted copy silently filters on flags the
    sidebar no longer means."""
    import ast, io as _io, os
    root = os.path.join(os.path.dirname(__file__), "..")

    def _dict_literal(path, name):
        for n in ast.walk(ast.parse(_io.open(path, encoding="utf-8").read())):
            if (isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
                    and isinstance(n.value, ast.Dict)):
                return {k.value: v.value for k, v in zip(n.value.keys, n.value.values)}
        return None

    app_map = _dict_literal(os.path.join(root, "app.py"), "_MP_CATALYSTS")
    disc_map = _dict_literal(os.path.join(root, "ui", "ui_discovery.py"), "_CATALYSTS")
    assert app_map and disc_map, "one of the catalyst dicts vanished"
    assert app_map == disc_map, f"catalyst vocabularies drifted: app={app_map} vs discovery={disc_map}"


def test_sectors_and_industry_clear_are_default_aware_and_complete():
    """The Sectors/Industry Clear resets each control to ITS OWN default — "All" is not
    universal (the size dial defaults to 5; resetting it to "All" would corrupt a numeric
    selectbox). COMPLETENESS is the contract: every mp_sec_*/mp_ind_* widget key must appear
    in its defaults map, so a future sixth control cannot ship without declaring its default."""
    import re
    src = _app_src()
    i = src.index("_SEC_DEFAULTS = {")
    sec_map = src[i:src.index("}", i)]
    for k in ("mp_sec_cap", "mp_sec_wealth", "mp_sec_cyc", "mp_sec_phase"):
        assert f'"{k}": "All"' in sec_map, f"{k} missing from _SEC_DEFAULTS"
    assert '"mp_sec_minn": 5' in sec_map, "the size dial must reset to 5, never 'All'"
    j = src.index("_IND_DEFAULTS = {")
    ind_map = src[j:src.index("}", j)]
    for k in ("mp_ind_cap", "mp_ind_wealth", "mp_ind_sec"):
        assert f'"{k}": "All"' in ind_map, f"{k} missing from _IND_DEFAULTS"
    # completeness: no mp_sec_/mp_ind_ WIDGET key exists outside its defaults map (clear buttons excluded)
    widget_keys = set(re.findall(r'key="(mp_(?:sec|ind)_[a-z_]+)"', src))
    declared = set(re.findall(r'"(mp_(?:sec|ind)_[a-z_]+)":', sec_map + ind_map))
    missing = sorted(widget_keys - declared - {"mp_sec_clear", "mp_ind_clear"})
    assert not missing, f"controls with NO declared reset default: {missing}"
    # both buttons are conditional + wired to the default-aware reset
    for anchor in ("_SEC_DEFAULTS.items())", "_IND_DEFAULTS.items())"):
        k = src.index(anchor)
        assert "st.button" in src[k:k + 300] and "_mp_clear_lens" in src[k:k + 300]
