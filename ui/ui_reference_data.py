"""CONCEPT_REFERENCE — plain-language meaning of every CATEGORICAL VALUE-LABEL PRISM shows.

The glossary (_RAW_GLOSSARY) explains the column NAMES (e.g. "Moat", "PEG Zone"); this explains the
VALUE OUTCOMES those columns take (Wealth Creator, Deep Value, Stage 2, …) — the labels you see on a
Tear-Sheet / All-Data cell that the glossary alone never defines. Grouped by category.

ACCURACY CONTRACT: every explanation is DERIVED FROM THE CODE that produces the label — each category
block cites its `file:line (np.select)` source. Do not edit a meaning without re-reading that source.
Coverage of the filterable sets is pinned by tests/test_concept_reference.py (cross-checked to the
live ui_discovery filters), and a 40-char quality floor guards against shallow entries.

Pure data — zero Streamlit. Rendered by ui_reference.render_concepts(); surfaced in the 📖 Reference tab.
"""

CONCEPT_REFERENCE = {
    # ── moat_growth_quad — core/data_engine.py:1725 (has_moat = ROCE 5y-med ≥15; has_growth = PAT 5y ≥15)
    "🧭 Moat × Growth Quadrant": [
        ("⭐ Wealth Creator", "High returns AND high growth — 5-year-median ROCE ≥15% and 5-year profit growth ≥15%. The best quadrant: a durable money-machine that is also compounding."),
        ("🛡️ Quality Trap", "High returns but low growth — ROCE ≥15% yet profit growth below 15%. A quality business whose engine has stopped compounding."),
        ("⚡ Growth Trap", "High growth but low returns — profit growing ≥15% while ROCE is below 15%. Growth that may DESTROY value because it earns below its cost of capital — expansion without economics."),
        ("💀 Wealth Destroyer", "Neither durable returns nor growth — ROCE below 15% and growth below 15%. The weakest position on the moat-vs-growth map."),
    ],
    # ── peg_zone — core/data_engine.py:1815 (PEG thresholds; negative PEG = falling earnings)
    "💰 Valuation — PEG Zone": [
        ("💎 Deep Value", "PEG of 0.5 or less — earnings growth is priced very cheaply relative to the growth rate."),
        ("🟢 Fair PEG", "PEG between 0.5 and 1.0 — Peter Lynch's sweet spot: paying roughly fairly for the growth."),
        ("🟡 Stretched", "PEG between 1.0 and 1.5 — paying a premium to the growth rate; valuation getting fuller."),
        ("🟠 Expensive", "PEG between 1.5 and 2.0 — the price runs well ahead of the earnings-growth rate."),
        ("🔴 Overpriced", "PEG above 2.0 — the valuation is far ahead of the earnings growth it rests on."),
        ("🔴 Declining", "PEG is negative because earnings are falling, so the ratio is meaningless — a warning, not a bargain."),
    ],
    # ── buy_zone_label — core/data_engine.py:1267 (distance above the Volatility Stop)
    "🎯 Entry — Buy Zone": [
        ("🟢 Perfect Entry (Low Risk)", "Price sits within 5% above its Volatility Stop — the tightest, lowest-risk entry: little downside to the stop if the trend breaks."),
        ("🟡 Standard Zone", "Price is 5–12% above the Volatility Stop — a normal volatility buffer; an ordinary entry, not extended."),
        ("🔴 Extended (Wait for Pullback)", "Price is more than 25% above the Volatility Stop — stretched far from support; waiting for a pullback lowers entry risk."),
        ("🔻 Below Stop (Trend Broken)", "Price has fallen BELOW its Volatility Stop — the trend is broken. The most dangerous state, not an entry however cheap it looks."),
        ("⚪ Uncharted", "No valid Volatility Stop (missing price/volatility data) — entry timing can't be judged."),
    ],
    # ── weinstein_stage — core/data_engine.py:2204 (price vs rising/falling 30-week MA + MA stacking)
    "📈 Trend — Weinstein Stage": [
        ("📈 Stage 2 Advancing", "Price above a rising 30-week moving average with the moving averages stacked up — Weinstein's confirmed uptrend, the buy stage."),
        ("🔄 Stage 1 Basing", "Bottoming/sideways after a decline — building a base before a possible advance; accumulate, don't chase."),
        ("⚠️ Stage 3 Top", "Topping after an advance — momentum stalling and distribution risk rising; tighten up."),
        ("📉 Stage 4 Declining", "Price below a falling 30-week moving average — Weinstein's downtrend, the avoid stage."),
        ("❔ Unknown", "Not enough price / moving-average history to place the stock in a Weinstein stage."),
    ],
    # ── lynch_category — core/data_engine.py:3005 (5-year revenue growth bands)
    "🚀 Style — Lynch Type": [
        ("Fast Grower", "Revenue growing 20%+ a year (5-year) — Lynch's high-growth archetype; judge it on whether the growth can last."),
        ("Stalwart", "Revenue growing 10–20% a year — large, steady compounders; reliable but rarely explosive."),
        ("Slow Grower", "Revenue growing 0–10% a year — mature and low-growth; usually held for dividends/value, not growth."),
        ("Declining", "Revenue shrinking — the 5-year top-line growth is below zero."),
    ],
    # ── mef_label — core/data_engine.py:3219 (moat_endurance_factor = current ÷ 10y-median ROCE)
    "🏰 Moat Endurance": [
        ("🟢 Expanding", "The moat is widening — current ROCE is 1.2× or more of its 10-year median; returns improving over time."),
        ("✅ Intact", "The moat is holding — current ROCE is at or above its 10-year median (about 1.0–1.2×)."),
        ("🟡 Eroding", "The moat is weakening — current ROCE has slipped to 0.80–1.0× its 10-year median."),
        ("🔴 Degrading", "The moat is breaking down — current ROCE is below 0.80× its 10-year median."),
    ],
    # ── smart_money_flow — core/data_engine.py:1234 (volume-quality + FII/DII flow + price confirmation)
    "🌊 Smart-Money Flow": [
        ("🌊💎 Elite Accumulation", "The strongest institutional-buying signal — top volume-quality (80+) with FII and DII flows converging in."),
        ("🎯 Strong Accumulation", "High volume-quality (60+) with FIIs or DIIs net buying — clear accumulation."),
        ("✅ Moderate Interest", "Decent volume-quality (40+) AND the price confirming (not lagging the market) — mild interest."),
        ("⚪ Neutral", "No clear institutional accumulation or distribution signal."),
        ("❌ Distribution", "Both FIIs and DIIs net selling while the price falls — Wyckoff distribution (institutions exiting), not interest."),
    ],
    # ── cf_triangle — core/forensic_engine.py:1006 (signs of operating/investing/financing cash flow)
    "💵 Cash-Flow Triangle": [
        ("✅ Perfect — Buy Zone", "Self-funding: operating cash positive, investing in the business, AND paying down financing — the healthiest cash pattern."),
        ("⚠️ Growth Phase — Watch D/E", "Operating cash is positive but the company is also BORROWING to invest — growth funded by debt; watch leverage."),
        ("🚨 Debt Trap — Avoid", "Burning operating cash, still spending, and borrowing to stay afloat — the dangerous cash pattern."),
        ("⚪ Mixed Pattern", "A cash-flow mix that doesn't fit the clean Perfect / Growth / Debt-trap patterns."),
    ],
    # ── corporate_class — core/scoring_engine.py:3142 (MOSL 13th-study Great/Good/Gruesome)
    "🏛️ Corporate Class (MOSL)": [
        ("🏆 GREAT", "MOSL's top capital-allocation class — 10-year-median ROCE ≥25%, strong free-cash conversion, and still high today. A proven, cash-generative compounder."),
        ("👍 GOOD", "Solid but not elite — 10-year-median ROCE ≥12% with weaker cash conversion. Decent economics, not best-in-class."),
        ("💀 GRUESOME", "Destroying economic value — 10-year-median ROCE below 12%, under the cost of capital; earns less than it costs to fund."),
    ],
    # ── capital_allocation_signal — core/data_engine.py:1977 (external_financing_to_assets)
    "🏦 Capital Allocation": [
        ("💰 Returning Capital", "Net capital FLOWS OUT to owners — buybacks, debt repayment and dividends exceed new raising (external financing below −5% of assets). Disciplined."),
        ("⚠️ Raising Capital", "Net capital is being RAISED — new equity/debt exceeds what's returned by more than 15% of assets; dilutive or leveraging."),
        ("⚖️ Neutral", "Capital raised and returned roughly balance (between −5% and +15% of assets)."),
    ],
    # ── cyclicality_tier — core/cyclicality_map.py TIER_LABELS (industry → behavioral tier)
    "🔄 Cyclicality Tier": [
        ("Deep Cyclical / Commodity", "A price-taking commodity business (metals, sugar, refining) — earnings swing hard with the commodity cycle; trade the cycle, don't marry it."),
        ("Cyclical", "Demand/capex/discretionary cyclical — sensitive to the economic cycle, but less extreme than a pure commodity."),
        ("Defensive", "Stable, non-cyclical demand (FMCG, pharma, utilities) — earnings hold up through the cycle; suited to holding."),
        ("Sensitive / Structural-Growth", "A secular, structural grower — driven more by a long-run growth theme than by the macro cycle."),
        ("Financials", "Banks, NBFCs and insurers — they ride their own credit and interest-rate cycle, judged differently from operating companies."),
        ("Catch-all", "Heterogeneous or hard-to-classify (conglomerates, trading, diversified) — no single cyclicality label fits."),
    ],
    # ── sector_capital_phase — core/data_engine.py:2004 (Chancellor capital cycle, sector asset growth)
    "❄️ Sector Capital Phase": [
        ("🔥 Hot Capital (caution)", "The sector is over-investing — capital is flooding in (high sector asset growth), which historically pressures future returns. Mean-reversion risk."),
        ("❄️ Capital Starved (opportunity)", "The sector is under-invested — little new capital coming in, which historically sets up supply tightness and recovery. Opportunity."),
        ("⚖️ Neutral", "Sector capital investment is neither unusually hot nor starved."),
    ],
    # ── verdict_direction — core/verdict_engine.py:72 (conviction tier + forensic/governance vetoes)
    "⚖️ Verdict": [
        ("BUY", "The engine's affirmative call — a high-conviction tier with no forensic or governance veto. Worth a deep look."),
        ("WATCH", "A qualified call — promising but with a caveat (mid conviction, or a soft forensic/timing flag). Monitor, don't act yet."),
        ("AVOID", "A negative call — low conviction, OR a hard veto from severe forensic red flags / governance risk. Stay away."),
    ],
}
