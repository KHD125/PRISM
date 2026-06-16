"""PRISM — discovery filter cascade (the universe-level 'find stocks' UI).

STATEFUL BY DESIGN: this module OWNS the sidebar filter widgets and their session_state — it is
the deliberate counterpart to the STATELESS ui_tearsheet.py (which renders one stock and must
never hold widgets). The CLAUDE.md §5 stateless mandate binds ui_tearsheet, NOT this file. Do
not 'harmonize' this module to the stateless rule.
"""
import re

import numpy as np
import pandas as pd
import streamlit as st

from config import COLORS


def clear_all_filters() -> None:
    """Delete every `sb_*` filter selection so the cascade resets to its show-all defaults, then
    rerun. SINGLE SOURCE — called by the sidebar 'Clear all' button AND the Discovery empty-state
    button, so the two can never diverge. Only `sb_*` keys are removed, so any other widget key
    (e.g. the Discovery tab's `disc_clear` button) is untouched."""
    for _k in [k for k in st.session_state if k.startswith("sb_") and k != "sb_clear"]:
        del st.session_state[_k]
    st.rerun()


def render_discovery_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    """Build the cascade sidebar + the two Refine toggles, fill the live funnel, and return the
    fully-filtered frame `filt`. Self-contained: the funnel placeholder is created and filled
    inside this function (the old app.py main-body coupling is gone). The only input it reads is
    `df`; the only name it produces for the caller is the returned `filt`."""
    with st.sidebar:
        st.markdown(f"<div class='sec-head'>🎯 Filters</div>", unsafe_allow_html=True)

        # Live results funnel (filled in the main body once the final filtered count is known) +
        # one-click reset. The funnel is the heart of the "interconnected" feel — the count drops
        # in real time as filters narrow the universe.
        _funnel = st.empty()
        if st.button("🧹 Clear all filters", key="sb_clear", use_container_width=True):
            clear_all_filters()

        def _active_n(*keys):
            """Count filters in a group that ACTUALLY narrow. Every filter now defaults to its
            'show all' state — blank for multiselects, 'All' for the sector/industry selectboxes,
            unticked/0 for the toggles — so a filter is active iff it holds a non-default value."""
            n = 0
            for k in keys:
                v = st.session_state.get(k)
                if k in ("sb_sector", "sb_industry"):
                    n += 1 if (v and v != "All") else 0
                elif k == "sb_maxrf":
                    n += 1 if (v is not None and v < _RF_MAX) else 0   # show-all value is the max
                else:
                    n += 1 if v else 0          # blank multiselect / unticked / slider-0 = inactive
            return n

        def _grp(title, *keys, expanded=False):
            """An expander whose header carries a live active-filter count for the group."""
            a = _active_n(*keys)
            return st.expander(title + (f"  ·  {a}" if a else ""), expanded=expanded)

        # ── SMART INTERCONNECTED FILTER CASCADE ──────────────────────────────────
        # Every option-based filter narrows the OPTIONS of every filter BELOW it. A single
        # progressively-narrowed frame (_cf) drives all option lists AND the final filtered
        # dataframe — so what a dropdown shows is exactly what survives the filter (zero
        # drift between options and results).
        # Defensive: stored multiselect selections are pruned to the current valid options
        # before each widget renders, preventing Streamlit's "value not in options" crash.
        def _ms_cascade(label, options, key, default, help=None, format_func=None):
            """Cascade-safe multiselect. Fully manages session_state (no `default=` arg, which
            avoids Streamlit's default-plus-session-state warning) and prunes any stale stored
            selection down to the current options each run. Empty selection = no filter.
            format_func lets the OPTION VALUES stay stable (so pruning holds) while the DISPLAY
            can vary per run (e.g. a live count) — never bake volatile text into the values."""
            if key not in st.session_state:
                st.session_state[key] = [v for v in default if v in options]
            else:
                st.session_state[key] = [v for v in st.session_state[key] if v in options]
            _kw = {"key": key, "help": help}
            if format_func is not None:
                _kw["format_func"] = format_func
            return st.multiselect(label, options, **_kw)

        def _ordered_present(frame, col, order):
            """Labels present in frame[col], in canonical `order`; unknown labels appended last."""
            if col not in frame.columns:
                return []
            present = set(frame[col].dropna().astype(str).unique())
            opts = [v for v in order if v in present]
            opts += [v for v in sorted(present) if v not in opts]
            return opts

        _cf = df   # progressively-narrowed cascade frame — drives every option list below
        # Fixed slider ceiling for the 🛡️ Safety "Max red flags" dial (computed on the FULL df so the
        # slider range is stable across the cascade); _active_n treats this value as the show-all state.
        _RF_MAX = int(df["red_flag_count"].max()) if "red_flag_count" in df.columns else 28

        with _grp("🏢 Universe", "sb_mcap", "sb_sector", "sb_industry", expanded=False):
            # 1. Market Category — cascade root (only categories present in the data)
            _ALL_MCAPS = ["Mega Cap", "Large Cap", "Mid Cap", "Small Cap", "Micro Cap", "Nano Cap"]
            _mcap_opts = _ordered_present(_cf, "market_category", _ALL_MCAPS)
            sel_mcap = _ms_cascade("Market Category", _mcap_opts, "sb_mcap", default=[],
                                   help="Blank = all market-cap tiers. Pick one or more to narrow.")
            if sel_mcap:
                _cf = _cf[_cf["market_category"].isin(sel_mcap)]

            # 2. Sector — only sectors within the chosen market categories
            _sector_opts = ["All"] + sorted(_cf["sector"].dropna().unique().tolist())
            if st.session_state.get("sb_sector", "All") not in _sector_opts:
                st.session_state["sb_sector"] = "All"
            sel_sector = st.selectbox(
                "Sector", _sector_opts, key="sb_sector",
                help="Filter to one sector. 'All' = every sector; also narrows the Industry list below.",
            )
            if sel_sector != "All":
                _cf = _cf[_cf["sector"] == sel_sector]

            # 3. Industry — only industries within the chosen categories AND sector
            _industry_opts = ["All"] + sorted(_cf["industry"].dropna().unique().tolist())
            if st.session_state.get("sb_industry", "All") not in _industry_opts:
                st.session_state["sb_industry"] = "All"
            sel_industry = st.selectbox(
                "Industry", _industry_opts, key="sb_industry",
                help="Granular industry within the selected sector. Narrows with Sector above.",
            )
            if sel_industry != "All":
                _cf = _cf[_cf["industry"] == sel_industry]
            st.caption(f"→ {len(_cf):,} remaining")

        with _grp("🎯 Decision & Class", "sb_tier", "sb_verdict", "sb_corpclass", expanded=True):
            # 4. Conviction Tier — only tiers present in the remaining stocks
            _tier_opts = sorted(int(t) for t in _cf["conviction_tier"].dropna().unique())
            sel_tier = _ms_cascade("Conviction Tier", _tier_opts, "sb_tier", default=[],
                                   help="Blank = all tiers. 1 = Crown Jewels, 2 = Strong … pick to narrow.")
            if sel_tier:
                _cf = _cf[_cf["conviction_tier"].isin(sel_tier)]

            # 4b. Verdict — the engine's BUY/WATCH/AVOID decision (filter to the rare BUYs/WATCHes)
            _verdict_opts = _ordered_present(_cf, "verdict_direction", ["BUY", "WATCH", "AVOID"])
            sel_verdict = _ms_cascade("Verdict", _verdict_opts, "sb_verdict", default=[],
                                      help="The engine's top-line decision. Empty = all stocks.")
            if sel_verdict and "verdict_direction" in _cf.columns:
                _cf = _cf[_cf["verdict_direction"].isin(sel_verdict)]

            # 4c. Corporate Class — Motilal Oswal capital-allocation quality (Great / Good / Gruesome)
            _corp_opts = _ordered_present(_cf, "corporate_class", ["🏆 GREAT", "👍 GOOD", "💀 GRUESOME"])
            sel_corp = _ms_cascade("Corporate Class", _corp_opts, "sb_corpclass", default=[],
                                   help="Capital-allocation quality. 'Only Great', or exclude Gruesome.")
            if sel_corp and "corporate_class" in _cf.columns:
                _cf = _cf[_cf["corporate_class"].isin(sel_corp)]
            st.caption(f"→ {len(_cf):,} remaining")

        with _grp("🛡️ Safety", "sb_maxrf", "sb_piotier", "sb_mincov", "sb_hidestale", expanded=False):
            # Risk-control screen — the system's most defensible, validation-INDEPENDENT edge: avoiding
            # the zeros (Gensol et al.). Deliberately NOT here: a forensic_label dropdown (98.6% one value
            # → degenerate; red_flag_count is its live form) and an "exclude risk-flags" group
            # (pledge_rising / dilution_vampire / debt_restatement are 84–99% subsumed by red_flag_count≥3
            # — redundant per the orthogonality census).
            # 4d. Max red flags — cap forensic severity (0 = pristine; _RF_MAX = show all)
            sel_maxrf = st.slider("Max red flags", 0, _RF_MAX, _RF_MAX, key="sb_maxrf",
                                  help="Cap how many of the 28 forensic red flags a stock may carry. Max = all.")
            if sel_maxrf < _RF_MAX and "red_flag_count" in _cf.columns:
                _cf = _cf[_cf["red_flag_count"] <= sel_maxrf]

            # 4e. Piotroski Strength — financial-strength tier from the F-Score (derived inline, vectorized)
            _pf = _cf["piotroski_fscore"]
            _pio_tier = np.where(_pf >= 7, "💪 Strong (≥7)",
                                 np.where(_pf >= 4, "➖ Moderate (4–6)", "⚠️ Weak (≤3)"))
            _pio_opts = [t for t in ["💪 Strong (≥7)", "➖ Moderate (4–6)", "⚠️ Weak (≤3)"]
                         if t in set(_pio_tier)]
            sel_pio = _ms_cascade("Piotroski Strength", _pio_opts, "sb_piotier", default=[],
                                  help="Financial-strength tier (Piotroski F-Score). Empty = all.")
            if sel_pio:
                _cf = _cf[pd.Series(_pio_tier, index=_cf.index).isin(sel_pio)]

            # 4f. Min data coverage % — don't trust a high score built on thin data
            sel_mincov = st.slider("Min data coverage %", 0, 100, 0, key="sb_mincov",
                                   help="Hide stocks whose score rests on below-this-% evidence coverage.")
            if sel_mincov > 0 and "data_coverage_pct" in _cf.columns:
                _cf = _cf[_cf["data_coverage_pct"] >= sel_mincov]

            # 4g. Hide stale results — drop frozen filers (>120 days; catches Gensol-style filing freezes)
            if st.checkbox("Hide stale results (>120d)", value=False, key="sb_hidestale") \
                    and "result_stale_flag" in _cf.columns:
                _cf = _cf[_cf["result_stale_flag"] == 0]
            st.caption(f"→ {len(_cf):,} remaining")

        # ── 3-TIER FRAMEWORK FILTER ENGINE ───────────────────────────────────────
        # Set Algebra: (Universe − Excluded) ∩ (Included ∪ ∅) ∩ (∀ Combined)
        # Evaluation order: Exclude first → Include second → Combination last.
        # Each panel's options cascade from the panel above — zero impossible combos.
        #
        # Panel 1 — Exclude (NOT): remove stocks passing ANY selected framework
        # Panel 2 — Include (OR) : show stocks passing ANY selected framework (current behavior)
        # Panel 3 — Combination (AND): keep only stocks passing ALL selected simultaneously

        def _extract_frameworks(frame):
            """Extract sorted unique framework names from the cascade frame.
            Splitter uses ', ' — the exact separator written by scoring_engine fw_str builder."""
            if "frameworks_passed" not in frame.columns:
                return []
            return sorted(set(
                fw.strip()
                for cell in frame["frameworks_passed"].dropna()
                if cell != "None"
                for fw in cell.split(", ")
                if fw.strip()
            ))

        def _fw_match_mask(frame, fw_list, logic="or"):
            """Build a boolean mask for stocks matching framework list.
            logic='or'  → stock passes ANY of the selected frameworks
            logic='and' → stock passes ALL of the selected frameworks simultaneously
            Regex is boundary-safe: prevents 'Bruised Blue Chip' from substring-matching
            'Bruised Blue Chip 29' via anchored exact-token pattern."""
            if not fw_list or "frameworks_passed" not in frame.columns:
                return pd.Series(True, index=frame.index)
            if logic == "or":
                mask = pd.Series(False, index=frame.index)
                for fw in fw_list:
                    _pat = r"(?:^|, )" + re.escape(fw) + r"(?:,|$)"
                    mask = mask | frame["frameworks_passed"].str.contains(_pat, regex=True, na=False)
                return mask
            else:  # logic == "and"
                mask = pd.Series(True, index=frame.index)
                for fw in fw_list:
                    _pat = r"(?:^|, )" + re.escape(fw) + r"(?:,|$)"
                    mask = mask & frame["frameworks_passed"].str.contains(_pat, regex=True, na=False)
                return mask

        with _grp("🧬 Frameworks", "sb_fw_exclude", "sb_fw_include", "sb_fw_combine", expanded=False):
            st.caption("Set algebra: (Universe − Exclude) ∩ Include ∩ Combination")

            # 5a. EXCLUDE Framework — NOT logic (applied first, narrows universe)
            _all_fw_excl = _extract_frameworks(_cf)
            sel_fw_exclude = _ms_cascade(
                "🚫 Exclude Framework", _all_fw_excl, "sb_fw_exclude", default=[],
                help="Remove stocks passing ANY of these frameworks. Applied first.",
            )
            if sel_fw_exclude:
                _excl_mask = _fw_match_mask(_cf, sel_fw_exclude, logic="or")
                _cf = _cf[~_excl_mask]
            # Exclude badge
            _n_excl_fw = len(sel_fw_exclude)
            if _n_excl_fw > 0:
                st.markdown(
                    f'<div style="font-size:0.6rem;color:{COLORS["red"]};padding:0 0 6px 2px;"'
                    f'>⛔ {_n_excl_fw} excluded · {len(_cf)} stocks remaining</div>',
                    unsafe_allow_html=True,
                )

            # 5b. INCLUDE Framework — OR logic (show stocks passing ANY selected)
            _all_fw_incl = _extract_frameworks(_cf)
            sel_fw_include = _ms_cascade(
                "✅ Include Framework", _all_fw_incl, "sb_fw_include", default=[],
                help="Show stocks passing ANY of these. Empty = all remaining stocks.",
            )
            if sel_fw_include:
                _incl_mask = _fw_match_mask(_cf, sel_fw_include, logic="or")
                _cf = _cf[_incl_mask]
            # Include badge
            _n_incl_fw = len(sel_fw_include)
            if _n_incl_fw > 0:
                st.markdown(
                    f'<div style="font-size:0.6rem;color:{COLORS["green"]};padding:0 0 6px 2px;"'
                    f'>✅ {_n_incl_fw} included (OR) · {len(_cf)} stocks remaining</div>',
                    unsafe_allow_html=True,
                )

            # 5c. COMBINATION Framework — AND logic (stock must pass ALL selected)
            _all_fw_comb = _extract_frameworks(_cf)
            sel_fw_combine = _ms_cascade(
                "🔗 Combination Framework", _all_fw_comb, "sb_fw_combine", default=[],
                help="Stock must pass ALL of these simultaneously. AND logic.",
            )
            if sel_fw_combine:
                _comb_mask = _fw_match_mask(_cf, sel_fw_combine, logic="and")
                _cf = _cf[_comb_mask]
            # Combination badge
            _n_comb_fw = len(sel_fw_combine)
            if _n_comb_fw > 0:
                st.markdown(
                    f'<div style="font-size:0.6rem;color:{COLORS["blue"]};padding:0 0 6px 2px;"'
                    f'>🔗 {_n_comb_fw} required (AND) · {len(_cf)} stocks remaining</div>',
                    unsafe_allow_html=True,
                )
            st.caption(f"→ {len(_cf):,} remaining")

        with _grp("💰 Moat · Value · Entry", "sb_moat", "sb_peg_zone", "sb_buy_zone", expanded=False):
            # 6. Moat-Growth quadrant — only quadrants present in the remaining stocks
            _MOAT_ORDER = ["⭐ Wealth Creator", "🛡️ Quality Trap", "⚡ Growth Trap", "💀 Wealth Destroyer"]
            _moat_opts = _ordered_present(_cf, "moat_growth_quad", _MOAT_ORDER)
            sel_moat = _ms_cascade("Moat", _moat_opts, "sb_moat", default=[],
                                   help="Moat-growth quadrant: Wealth Creator / Quality Trap / "
                                        "Growth Trap / Wealth Destroyer. Empty = all stocks.")
            if sel_moat:
                _cf = _cf[_cf["moat_growth_quad"].isin(sel_moat)]

            # 7. PEG Zone — valuation tier (options present in remaining stocks, canonical order)
            _PEG_ZONE_ORDER = [
                "💎 Deep Value", "🟢 Fair PEG", "🟡 Stretched",
                "🟠 Expensive", "🔴 Overpriced", "🔴 Declining",
            ]
            _peg_opts = _ordered_present(_cf, "peg_zone", _PEG_ZONE_ORDER)
            sel_peg_zone = _ms_cascade("PEG Zone", _peg_opts, "sb_peg_zone", default=[],
                                       help="Valuation tier from the PEG ratio. Empty = all stocks.")
            if sel_peg_zone and "peg_zone" in _cf.columns:
                _cf = _cf[_cf["peg_zone"].isin(sel_peg_zone)]

            # 8. Buy Zone — entry timing vs Volatility Stop (options present in remaining stocks)
            _BUY_ZONE_ORDER = [
                "🟢 Perfect Entry (Low Risk)", "🟡 Standard Zone",
                "🔴 Extended (Wait for Pullback)", "🔻 Below Stop (Trend Broken)", "⚪ Uncharted",
            ]
            _buy_opts = _ordered_present(_cf, "buy_zone_label", _BUY_ZONE_ORDER)
            sel_buy_zone = _ms_cascade("Buy Zone", _buy_opts, "sb_buy_zone", default=[],
                                       help="Entry timing vs the Volatility Stop. Empty = all stocks.")
            if sel_buy_zone and "buy_zone_label" in _cf.columns:
                _cf = _cf[_cf["buy_zone_label"].isin(sel_buy_zone)]
            st.caption(f"→ {len(_cf):,} remaining")

        with _grp("📈 Trend · Style · Flow", "sb_weinstein", "sb_lynchcat", "sb_mef",
                  "sb_cftri", "sb_smartflow", expanded=False):
            # 8b. Weinstein Stage — the 30-week-trend stage (Stage 2 = uptrend buy, Stage 4 = avoid)
            _WEIN_ORDER = ["📈 Stage 2 Advancing", "🔄 Stage 1 Basing", "⚠️ Stage 3 Top",
                           "📉 Stage 4 Declining", "❔ Unknown"]
            _wein_opts = _ordered_present(_cf, "weinstein_stage", _WEIN_ORDER)
            sel_wein = _ms_cascade("Weinstein Stage", _wein_opts, "sb_weinstein", default=[],
                                   help="Long-term price trend stage. Empty = all stocks.")
            if sel_wein and "weinstein_stage" in _cf.columns:
                _cf = _cf[_cf["weinstein_stage"].isin(sel_wein)]

            # 8c. Lynch Type — Peter Lynch's stock archetype
            _lynch_opts = _ordered_present(_cf, "lynch_category",
                                           ["Fast Grower", "Stalwart", "Slow Grower", "Declining"])
            sel_lynch = _ms_cascade("Lynch Type", _lynch_opts, "sb_lynchcat", default=[],
                                    help="Lynch's classification (Fast Grower / Stalwart / …). Empty = all.")
            if sel_lynch and "lynch_category" in _cf.columns:
                _cf = _cf[_cf["lynch_category"].isin(sel_lynch)]

            # 8d. Moat Endurance — is the competitive advantage widening or eroding
            _mef_opts = _ordered_present(_cf, "mef_label",
                                         ["🟢 Expanding", "✅ Intact", "🟡 Eroding", "🔴 Degrading"])
            sel_mef = _ms_cascade("Moat Endurance", _mef_opts, "sb_mef", default=[],
                                  help="Whether the moat is widening or eroding over time. Empty = all.")
            if sel_mef and "mef_label" in _cf.columns:
                _cf = _cf[_cf["mef_label"].isin(sel_mef)]

            # 8e. Cash-Flow Triangle — cash-flow quality pattern
            _cftri_opts = _ordered_present(_cf, "cf_triangle",
                                           ["✅ Perfect — Buy Zone", "⚪ Mixed Pattern",
                                            "⚠️ Growth Phase — Watch D/E", "🚨 Debt Trap — Avoid"])
            sel_cftri = _ms_cascade("Cash-Flow Triangle", _cftri_opts, "sb_cftri", default=[],
                                    help="Operating/investing/financing cash-flow quality. Empty = all.")
            if sel_cftri and "cf_triangle" in _cf.columns:
                _cf = _cf[_cf["cf_triangle"].isin(sel_cftri)]

            # 8f. Smart-Money Flow — the 5-level institutional-flow read
            _smf_opts = _ordered_present(_cf, "smart_money_flow",
                                         ["🌊💎 Elite Accumulation", "🎯 Strong Accumulation",
                                          "✅ Moderate Interest", "⚪ Neutral", "❌ Distribution"])
            sel_smf = _ms_cascade("Smart-Money Flow", _smf_opts, "sb_smartflow", default=[],
                                  help="Institutional accumulation/distribution level. Empty = all.")
            if sel_smf and "smart_money_flow" in _cf.columns:
                _cf = _cf[_cf["smart_money_flow"].isin(sel_smf)]
            st.caption(f"→ {len(_cf):,} remaining")

        with _grp("🔥 Catalysts & Alerts", "sb_catalyst", "sb_sellalert", expanded=False):
            # 9. Catalyst — fast-moving EVENT triggers (debt repair, new capacity, margin inflection,
            # early institutional discovery, Lynch GARP). OR logic: show stocks with ANY selected catalyst.
            # Option VALUES are the stable cat_* column names (so the cascade pruning holds); the live
            # count is shown only via format_func. Only catalysts present in the remaining frame are listed.
            _CATALYSTS = {
                "🔥 Capacity Explosion": "cat_capacity",
                "🔥 OpLev Inflection":    "cat_oplev",
                "🔥 Deleveraging":         "cat_deleveraging",
                "🔥 Lynch Dream":          "cat_lynch_dream",
                "🔥 Inst Discovery":       "cat_inst_discovery",
            }
            _cat_name = {v: k for k, v in _CATALYSTS.items()}
            _cat_opts = [c for c in _CATALYSTS.values() if c in _cf.columns and int(_cf[c].sum()) > 0]
            sel_catalyst = _ms_cascade(
                "🔥 Catalyst", _cat_opts, "sb_catalyst", default=[],
                help="Show stocks where ANY selected catalyst (a fast-moving change) is firing. OR logic. "
                     "Empty = all stocks. Count = stocks with that catalyst in the current filtered set.",
                format_func=lambda c: f"{_cat_name[c]} ({int(_cf[c].sum())})",
            )
            if sel_catalyst:
                _cat_mask = pd.Series(False, index=_cf.index)
                for _c in sel_catalyst:
                    _cat_mask = _cat_mask | (_cf[_c] == 1)
                _cf = _cf[_cat_mask]
                st.markdown(
                    f'<div style="font-size:0.6rem;color:{COLORS["orange"]};padding:0 0 6px 2px;"'
                    f'>🔥 {len(sel_catalyst)} catalyst(s) (OR) · {len(_cf)} stocks remaining</div>',
                    unsafe_allow_html=True,
                )

            # 10. Sell Alerts — Baid sell triggers (separate 0/1 cols). OR logic: show stocks with ANY
            # selected alert (surface risk to review). Same stable-column-name + format_func pattern as 🔥.
            _SELL_ALERTS = {
                "🚨 Cash Collapse":      "sell_alert_cash_collapse",
                "🚨 Overvalued":         "sell_alert_overvalued",
                "🚨 Thesis Broken":      "sell_alert_thesis_broken",
                "🚨 Treadmill":          "sell_alert_treadmill",
                "🚨 Sequential Decline": "sell_alert_sequential_decline",
                "🚨 Mgmt Deteriorated":  "sell_alert_mgmt_deteriorated",
            }
            _sa_name = {v: k for k, v in _SELL_ALERTS.items()}
            _sa_opts = [c for c in _SELL_ALERTS.values() if c in _cf.columns and int(_cf[c].sum()) > 0]
            sel_alert = _ms_cascade(
                "🚨 Sell Alerts", _sa_opts, "sb_sellalert", default=[],
                help="Show stocks where ANY selected Baid sell trigger is active. OR logic. Empty = all.",
                format_func=lambda c: f"{_sa_name[c]} ({int(_cf[c].sum())})",
            )
            if sel_alert:
                _sa_mask = pd.Series(False, index=_cf.index)
                for _c in sel_alert:
                    _sa_mask = _sa_mask | (_cf[_c] == 1)
                _cf = _cf[_sa_mask]
                st.markdown(
                    f'<div style="font-size:0.6rem;color:{COLORS["red"]};padding:0 0 6px 2px;"'
                    f'>🚨 {len(sel_alert)} alert(s) (OR) · {len(_cf)} stocks remaining</div>',
                    unsafe_allow_html=True,
                )
            st.caption(f"→ {len(_cf):,} remaining")

        with _grp("🌊 Refine", "sb_gate", "sb_minq", "sb_minscore", expanded=False):
            # Final power-user knobs — now applied to the cascade frame (_cf) IN-GROUP, exactly like
            # every other group, so the funnel, this group's "·N" badge, and its "→ remaining"
            # caption all agree (no more last-caption-vs-funnel drift). All default OFF.
            gate_only = st.checkbox("Gate-passed only", value=False, key="sb_gate",
                                    help="Show only stocks that clear the engine's quality gate.")
            if gate_only and "gate_pass" in _cf.columns:
                _cf = _cf[_cf["gate_pass"] == 1]
            min_quality = st.slider("Min Quality Score", 0, 100, 0, key="sb_minq",
                                    help="Min fundamental quality score (PRE-forensic-penalty — moat + "
                                         "growth + cash + governance, before red-flag cuts).")
            if min_quality > 0 and "quality_score" in _cf.columns:
                _cf = _cf[_cf["quality_score"] >= min_quality]
            min_score = st.slider("Min Composite Score", 0, 100, 0, key="sb_minscore",
                                  help="Min headline composite_score (post-forensic-penalty — the score "
                                       "your tiers are built on; stronger than Min Quality, which is pre-penalty).")
            if min_score > 0 and "composite_score" in _cf.columns:
                _cf = _cf[_cf["composite_score"] >= min_score]
            st.caption(f"→ {len(_cf):,} remaining")

    # The cascade frame (_cf) now encodes EVERY filter — the categorical groups AND the Refine
    # thresholds — so the dropdown options and the final result are genuinely identical and `filt`
    # is just `_cf`. The .copy() is REQUIRED: _cf is `df` itself when nothing is filtered, so the
    # copy guarantees the caller can never mutate the source frame.
    filt = _cf.copy()

    # Fill the live results funnel (placeholder created at the TOP of the sidebar filter panel).
    # `filt` == `_cf` now, so the count reflects the WHOLE cascade — every group, Refine included.
    # The headline number sits above every filter while measuring all of them; the funnel is what
    # makes the panel feel like one interconnected system: change any filter, watch this number move.
    _uni_n, _fin_n = len(df), len(filt)
    _pct = (_fin_n / _uni_n) if _uni_n else 0.0
    _active_total = _active_n(
        "sb_mcap", "sb_sector", "sb_industry", "sb_tier", "sb_verdict", "sb_corpclass",
        "sb_maxrf", "sb_piotier", "sb_mincov", "sb_hidestale",
        "sb_fw_exclude", "sb_fw_include", "sb_fw_combine", "sb_moat", "sb_peg_zone", "sb_buy_zone",
        "sb_weinstein", "sb_lynchcat", "sb_mef", "sb_cftri", "sb_smartflow",
        "sb_catalyst", "sb_sellalert", "sb_gate", "sb_minq", "sb_minscore",
    )
    # Every filter now defaults to its show-all state, so zero active filters == the full universe.
    _flabel = (f"🎯 {_active_total} filter{'s' if _active_total != 1 else ''} active"
               if _active_total else "○ No filters — full universe")
    _funnel.markdown(
        f'''<div style="background:linear-gradient(135deg,{COLORS['bg_secondary']},{COLORS['bg_tertiary']});
             border:1px solid {COLORS['border']};border-radius:10px;padding:10px 12px;margin:2px 0 10px 0;">
          <div style="display:flex;align-items:baseline;justify-content:space-between;">
            <span style="font-size:1.5rem;font-weight:800;color:{COLORS['text_primary']};line-height:1;">{_fin_n:,}</span>
            <span style="font-size:0.62rem;color:{COLORS['text_muted']};">of {_uni_n:,} · {_pct:.0%}</span>
          </div>
          <div style="background:{COLORS['bg_tertiary']};border-radius:4px;height:5px;overflow:hidden;margin:6px 0;">
            <div style="width:{max(_pct * 100, 1):.0f}%;height:5px;border-radius:4px;background:{COLORS['purple']};"></div>
          </div>
          <div style="font-size:0.6rem;color:{COLORS['text_muted']};text-transform:uppercase;letter-spacing:0.6px;">
            {_flabel}
          </div>
        </div>''',
        unsafe_allow_html=True,
    )
    return filt
