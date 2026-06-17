"""Scanner column header-tip map.

This module once held an AgGrid `render_scanner_grid`; that grid was retired (superseded by the
`st.dataframe` Deep Scanner in app.py) and now lives only in git history. What remains is the live,
single-source tooltip map the Deep Scanner wires into its column_config `help=`.
"""

# Plain-language header tooltips for the scanner's machine-named columns. Reuse the SAME glossary
# the tearsheet "?" chip uses (single source of truth — a term can never drift between the grid
# header and the tearsheet); bespoke sentences only for scanner-only columns with no tearsheet cell.
from ui.ui_components import _RAW_GLOSSARY as _GLOSSARY

_SCANNER_HEADER_TIPS = {
    "rank":             "The stock's overall rank in the current screen (1 = highest conviction).",
    "verdict_direction":"The engine's one-word call (e.g. BUY / WATCH / AVOID) synthesised from all 6 axes after the forensic penalty.",
    "corporate_class":  _GLOSSARY["Corporate Class"],
    "composite_score":  _GLOSSARY["Composite Score"],
    "conviction_tier":  _GLOSSARY["Conviction Tier"],
    "moat_growth_quad": "Where the stock sits on the moat-versus-growth map (e.g. Wealth Creator, Quality Trap, Growth Trap).",
    "fisher_lifecycle_quadrant": "Phil Fisher's growth-versus-quality lifecycle quadrant for the business (e.g. Catalyst Play, Laggard).",
    "cash_score":       "A 0-100 score for how strongly the business converts reported profit into real operating cash.",
    "buy_zone_label":   _GLOSSARY["Buy Zone"],
    "forensic_score":   _GLOSSARY["Forensic Scr"],
    "red_flag_count":   _GLOSSARY["Red Flags"],
    "momentum_score":   _GLOSSARY["Momentum Scr"],
    # ── Deep Scanner view-preset columns (Quality / Valuation / Forensic / Technical) ──
    # Reuse the SAME glossary the tearsheet "?" chip uses (single source of truth); bespoke text only
    # where a column has no tearsheet-cell glossary entry. Coverage is pinned by
    # tests/test_tooltip_coverage.py::test_every_scanner_preset_column_has_header_tip.
    "smart_money_flow":       _GLOSSARY["Smart Money"],
    "quality_score":          "Overall fundamental quality sub-score (0-100): moat + growth + cash + governance, before the forensic penalty.",
    "moat_score":             _GLOSSARY["Moat Score"],
    "growth_score":           _GLOSSARY["Growth Score"],
    "governance_bonus":       _GLOSSARY["Governance Score"],
    "piotroski_fscore":       _GLOSSARY["Piotroski"],
    "roce":                   _GLOSSARY["ROCE Current"],
    "opm":                    _GLOSSARY["OPM"],
    "cfo_to_pat":             _GLOSSARY["CFO/PAT"],
    "valuation_score":        _GLOSSARY["Valuation Scr"],
    "expected_excess_return": "Mauboussin expected-value excess return (%): the probability-weighted upside-versus-downside payoff over the base case.",
    "pe":                     _GLOSSARY["PE"],
    "pb_ratio":               _GLOSSARY["P/B"],
    "peg":                    _GLOSSARY["PEG"],
    "earnings_yield":         _GLOSSARY["Earnings Yield"],
    "fcf_yield":              _GLOSSARY["FCF Yield"],
    "market_cap":             "Total market value of the company's equity (price × shares), in ₹ crore.",
    "accruals_ratio":         _GLOSSARY["Accruals Ratio"],
    "debt_to_equity":         _GLOSSARY["D/E Ratio"],
    "promoter_holdings":      _GLOSSARY["Promoter %"],
    "pledged_percentage":     _GLOSSARY["Pledge %"],
    "rsi_14d":                _GLOSSARY["RSI 14D"],
    "dist_52wh":              _GLOSSARY["Dist 52WH"],
    "crs_52w":                _GLOSSARY["CRS 52W"],
    "weinstein_stage":        _GLOSSARY["Weinstein Stage"],
    "breakout_score":         _GLOSSARY["Breakout Scr"],
    "vstop_green":            _GLOSSARY["VSTOP Green"],
    "gate_pass":              "Whether the stock clears ALL the engine's hard quality gates (e.g. ROCE, growth, positive-PAT floors).",
    "tsunami_signal":         "A rare confluence flag — a quality breakout meeting institutional accumulation; fires for only a handful of stocks by design.",
}
