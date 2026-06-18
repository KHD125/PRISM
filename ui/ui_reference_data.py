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
    # ── conviction_tier / tier_label — config.CONVICTION_TIERS (composite-score bands)
    "🏆 Conviction Tier": [
        ("Crown Jewels", "Tier 1 (composite ≥85) — the highest-conviction compounders; deep-dive and build a position."),
        ("Strong Compounders", "Tier 2 (composite ≥70) — quality with momentum confirmation; watchlist priority."),
        ("Emerging Quality", "Tier 3 (composite ≥55) — quality building and momentum developing; monitor for an upgrade."),
        ("On Radar", "Tier 4 (composite ≥40) — on the radar but not yet qualified; keep watching."),
        ("Not Ready", "Tier 5 (composite below 40) — does not yet clear the bar; pass for now."),
    ],
    # ── mcap_tier / market_category — config.MCAP_TIERS (market-cap bands, ₹ Cr)
    "🏢 Market Cap Tier": [
        ("Mega Cap", "Market capitalisation of ₹2,00,000 Cr or more — the very largest, most-liquid companies."),
        ("Large Cap", "Market capitalisation of ₹20,000–2,00,000 Cr — large, well-established companies."),
        ("Mid Cap", "Market capitalisation of ₹5,000–20,000 Cr — mid-sized companies."),
        ("Small Cap", "Market capitalisation of ₹500–5,000 Cr — smaller companies, more volatile."),
        ("Micro Cap", "Market capitalisation of ₹100–500 Cr — very small companies, often thinly traded."),
        ("Nano Cap", "Market capitalisation below ₹100 Cr — the smallest and least-liquid companies."),
    ],
    # ── cash_machine_label — core/data_engine.py:1046 (cash_machine_score from CFO/PAT)
    "💵 Cash Machine": [
        ("💰 Cash Machine", "Profits are fully cash-backed — operating cash flow comfortably exceeds reported profit. The strongest cash-quality signal."),
        ("✅ Solid", "Operating cash flow is above 80% of reported profit — acceptable cash conversion."),
        ("📄 Paper Profits", "Operating cash flow lags reported profit — earnings aren't turning into cash. A quality warning."),
    ],
    # ── ep_power_curve — core/data_engine.py:1506 (economic profit level × its velocity)
    "📊 Economic-Profit Power Curve": [
        ("🚀 Hockey Stick", "Positive economic profit AND accelerating — earns above its cost of capital and the spread is widening. The best state."),
        ("✅ EP Positive", "Positive economic profit but not accelerating — earns above its cost of capital, holding steady."),
        ("📈 Improving", "Economic profit still negative but improving — below its cost of capital, yet the trend is turning up."),
        ("📉 Value Trap", "Negative economic profit and not improving — earns below its cost of capital with no upturn."),
    ],
    # ── earnings_power_box — core/data_engine.py:1547 (Heiserman defensive × enterprising)
    "📦 Earnings Power Box": [
        ("📦 Earnings Power", "Strong on BOTH Heiserman tests — defensive (stable, profitable) and enterprising (reinvesting for growth). The top box."),
        ("💰 Cash Cow", "Defensive but not enterprising — stable and profitable, but reinvesting little for growth."),
        ("🚀 Cash-Hungry Grower", "Enterprising but not defensive — reinvesting hard for growth, but the base economics aren't yet stable."),
        ("⚠️ Weakest", "Weak on both Heiserman tests — neither stable economics nor productive reinvestment."),
    ],
    # ── fisher_lifecycle_quadrant — core/scoring_engine.py:2787 (Fisher quality_pass × scalability pass)
    "🧬 Fisher Lifecycle Quadrant": [
        ("👑 Apex Winner", "Elite quality business at its operating-leverage peak — passes both the Fisher quality and scalability screens. Prime entry."),
        ("🐢 Steady Compounder", "Proven quality with no current inflection — a durable steady-state hold past its fastest phase."),
        ("⚡ Catalyst Play", "A scalability inflection firing but structural quality not yet proven — an earlier, trading-style candidate."),
        ("⚪ Laggard", "Neither the Fisher quality nor the scalability screen passes — no current edge on this map."),
    ],
    # ── malik_label / forensic_label — data_engine.py:1705 / forensic_engine.py:166 (checklist strength)
    "📋 Quality / Forensic Strength": [
        ("🟢 Strong", "A strong rating on the relevant checklist (Malik quality or forensic accounting-quality) — passes most or all of its tests."),
        ("🟡 Moderate", "A middling rating — passes some of the checklist's tests but not most."),
        ("🟠 Weak", "A weak rating — fails most of the Malik quality checklist's tests."),
        ("🔴 Poor", "The lowest Malik-quality rating — fails nearly all of the financial-strength tests."),
        ("🔴 Weak", "A weak forensic rating — the accounting-quality checks raise concern."),
    ],
    # ── Piotroski Strength — ui/ui_discovery.py np.where (F-Score band)
    "🛡️ Piotroski Strength": [
        ("💪 Strong (≥7)", "Piotroski F-Score of 7–9 — very healthy books across profitability, leverage and efficiency."),
        ("➖ Moderate (4–6)", "Piotroski F-Score of 4–6 — middling financial health on the nine-point checklist."),
        ("⚠️ Weak (≤3)", "Piotroski F-Score of 3 or below — weak financial health; treat with caution."),
    ],
    # ── trend_modifier — core/data_engine.py:2230 (Weinstein stage × Grimes path event)
    "↩️ Trend Modifier": [
        ("↩️ Pullback", "High-edge continuation: a Stage-2 dip below the 50-day line, still above the 30-week MA, on volume dry-up — a buy-the-dip setup."),
        ("🚀 Breakout", "With-trend edge: a Stage-2 stock within 3% of its 52-week high, not extended, on volume expansion — a breakout setup."),
        ("⚠️ Bounce", "Low-confidence counter-trend: a Stage-4 rally back up to the falling 30-week MA — Weinstein's 'don't chase' zone."),
        ("⚠️ Extended", "Low-confidence caution: stretched far above the 30-week MA and overbought (RSI > 70) — termination risk."),
    ],
    # ── d48_breakout_readiness — core/data_engine.py:2280 (distance to 52w/13w high)
    "🎯 Breakout Readiness": [
        ("IMMINENT", "Within 10% of the 52-week high AND within 5% of the 13-week high — a breakout looks imminent."),
        ("NEAR", "Within 20% of the 52-week high — approaching breakout territory."),
        ("FAR", "More than 20% below the 52-week high — not near a breakout."),
    ],
    # ── d49_momentum_quality — core/data_engine.py:2289 (RSI + ADX)
    "⚡ Momentum Quality": [
        ("OVERHEATED", "RSI above 70 — momentum is strong but overbought; pullback risk is elevated."),
        ("HIGH", "RSI in the healthy 50–70 zone with a strong trend (ADX > 20) — high-quality momentum."),
        ("WEAK", "Momentum is weak — neither overbought nor in a confirmed strong trend."),
    ],
    # ── verdict coverage confidence — core/verdict_engine.py:92 (evidence coverage %)
    "🔍 Verdict — Evidence Confidence": [
        ("High", "The verdict rests on 80%+ evidence coverage — most ranked inputs reported; trust it more."),
        ("Medium", "60–80% evidence coverage — a fair amount of the inputs reported."),
        ("Low", "40–60% evidence coverage — a meaningful share of inputs are missing; treat with care."),
        ("Very Low", "Under 40% evidence coverage — the verdict rests on thin data; tentative."),
    ],
    # ── verdict_axis_governance — core/verdict_engine.py:116 (governance multiplier)
    "🛡️ Verdict — Governance Axis": [
        ("Govern 🟢 Safe", "No governance penalty — promoter pledge, dilution and related-party signals are clean."),
        ("Govern 🟡 Caution", "A mild governance penalty — one or more governance signals warrant caution."),
        ("Govern 🔴 Risk", "A heavy governance penalty — serious pledge/dilution/related-party risk drags the score."),
    ],
    # ── verdict_top_risk — core/verdict_engine.py:121 (the single most important risk)
    "⚠️ Verdict — Top Risk": [
        ("🚨 Severe forensic / accounting-quality flags", "The dominant risk: severe forensic red flags veto the thesis — verify the accounts before anything."),
        ("💀 Value-destroying capital allocation", "The dominant risk: a Gruesome capital allocator earning below its cost of capital — destroys value."),
        ("🕵️ Schilit forensic checker flags", "The dominant risk: Schilit accounting-quality checkers fire — the reported numbers may be aggressive."),
        ("⚠️ Governance risk (pledge/dilution)", "The dominant risk: governance — meaningful promoter pledging or dilution."),
        ("⏳ Poor entry timing — wait for a base", "The dominant risk is timing, not quality — the chart is poorly placed; wait for a base."),
        ("🔍 Thin data — verdict tentative", "The dominant caveat: evidence coverage is thin, so the verdict is tentative until more inputs report."),
    ],
    # ── Catalysts — ui/ui_discovery.py _CATALYSTS (fast-moving change triggers)
    "🔥 Catalysts": [
        ("🔥 Capacity Explosion", "A capacity-expansion catalyst — fixed assets/CWIP converting to a step-up in productive capacity."),
        ("🔥 OpLev Inflection", "An operating-leverage inflection — profit growth pulling decisively ahead of revenue growth."),
        ("🔥 Deleveraging", "A deleveraging catalyst — debt being repaid materially, easing the balance sheet and interest drag."),
        ("🔥 Lynch Dream", "A Lynch GARP setup — fast growth available at a reasonable PEG; the classic Peter Lynch profile."),
        ("🔥 Inst Discovery", "Early institutional discovery — accumulation signs while the stock is still under-owned."),
    ],
    # ── Sell Alerts — ui/ui_discovery.py _SELL_ALERTS (Baid sell triggers)
    "🚨 Sell Alerts": [
        ("🚨 Cash Collapse", "Operating cash flow has collapsed relative to profit — the cash engine is breaking down."),
        ("🚨 Overvalued", "The valuation has run far ahead of the fundamentals — priced for perfection."),
        ("🚨 Thesis Broken", "A core pillar of the bull thesis (growth/returns) has broken — re-underwrite or exit."),
        ("🚨 Treadmill", "Running to stand still — heavy reinvestment producing little incremental return."),
        ("🚨 Sequential Decline", "Sequential (quarter-on-quarter) deterioration — the recent trend is turning down."),
        ("🚨 Mgmt Deteriorated", "Management-quality / governance signals have deteriorated — a red flag for owners."),
    ],
}
