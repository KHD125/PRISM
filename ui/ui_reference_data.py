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
        ("🟠 Loose Entry Zone", "12–25% above the Volatility Stop — still in the uptrend, but entering here means accepting a 12–25% loss if the stop is honored (~3× Minervini's max-loss rule). Fine to hold; mediocre to initiate."),
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
    # ── wealth_tier — core/verdict_engine.py (the change-lens verdict; price- and forensics-blind) ──
    "💹 Wealth Tier": [
        ("BUY★", "All three wealth clocks agree: earning above the cost of equity (EP% > 0), that excess return materially improving this year (Vel% ≥ +0.5), and the 5-year margin trend confirming it (tau ≥ +0.25). Price-blind — says the wealth engine is buy-grade, NOT that the price is right."),
        ("BUY", "Earning above the cost of equity and improving, but the 5-year margin trend is flat — the engine is running, its durability is unproven."),
        ("WATCH★", "The confirmed turnaround: NOT yet earning above the cost of equity, but the excess return is climbing and margins confirm the turn. The engine's AVOID pile is where these live."),
        ("WATCH", "One clock only — a margin spine without improvement, improvement without earnings, or a good business whose momentum just broke. Also caps anything whose margins are fading (tau ≤ −0.25)."),
        ("AVOID", "Nothing improving. The LEVEL may be excellent — a stock can earn far above its cost of equity and still land here because every change clock points down (the engine's own BUY list contains such names)."),
        ("N/A", "An input the tier needs (the reserves equity base, economic profit, its velocity, or the margin tau) is missing or unusable — unverifiable is neither passed nor condemned, so no tier is assigned. The row still shows whatever numbers exist."),
    ],
    # ── cf_triangle — core/forensic_engine.py:1006 (signs of operating/investing/financing cash flow)
    "💵 Cash-Flow Triangle": [
        ("✅ Self-Funding", "Operating cash positive, investing in the business, AND paying down financing — the healthiest cash pattern."),
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
        ("earn-DD (Earnings Drawdown)", "The number beside the tier badge: the deepest peak-to-trough fall in annual net profit over the last 6 years — a drawdown applied to PROFITS instead of price. ~0% = profits only ever rose (steady compounder); ~50% = profits once halved (cyclical signature); above 100% = the trough year was an outright loss (capped at 300% where the math turns to noise). The tier is the business's REPUTATION by industry; earn-DD is its RECORD — a 'Defensive' name carrying a big earn-DD is behaving cyclically despite its label. Needs 4 of 6 years reported; display-only, never scored."),
    ],
    # ── sector_capital_phase — core/data_engine.py:2004 (Chancellor capital cycle, sector asset growth)
    "❄️ Sector Capital Phase": [
        ("🔥 Hot Capital (caution)", "The sector is over-investing — capital is flooding in (high sector asset growth), which historically pressures future returns. Mean-reversion risk."),
        ("❄️ Capital Starved (opportunity)", "The sector is under-invested — little new capital coming in, which historically sets up supply tightness and recovery. Opportunity."),
        ("⚖️ Neutral", "Sector capital investment is neither unusually hot nor starved."),
    ],
    # ── verdict_direction — core/verdict_engine.py:72 (conviction tier + forensic/governance vetoes)
    "⚖️ Soundness": [
        ("SOUND", "The engine's affirmative gate — quality, clean books and a reasonable price all clear (tier 1–2, no forensic or governance veto). Renamed from BUY on 2026-08-27: with 18 passers against 93% of the universe failing, this was always a qualification gate, not a buy call — and the wealth tier now owns the action vocabulary. Condition words here, action words there, no word in both."),
        ("MIXED", "The middle of the gate — promising but with a caveat: mid conviction, or a soft forensic/timing/governance downgrade from SOUND. Renamed from WATCH (the wealth tier owns WATCH/WATCH★ now)."),
        ("FLAWED", "The gate found flaws — low conviction, OR a hard veto from severe forensic red flags / value-destroying capital allocation. Says nothing about direction: a FLAWED stock can be a confirmed turnaround on the wealth tier (WATCH★), and that pairing is exactly what the two lenses exist to surface. Renamed from AVOID."),
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
        ("🚀 Hockey Stick", "Earns above its cost of equity AND that economic profit is larger than a year ago — climbing the Power Curve. The best state."),
        ("➖ EP Positive, Not Rising", "Earns above its cost of equity, but the profit is not larger than last year — or the prior year isn't reported. Value-creating, not yet climbing."),
        ("📈 Improving", "Economic profit still negative but larger than a year ago — below its cost of equity, yet the trend is turning up."),
        ("📉 Value Trap", "Negative economic profit and no larger than a year ago — earns below its cost of equity with no upturn."),
    ],
    # ── earnings_power_box — core/data_engine.py:1547 (Heiserman defensive × enterprising)
    "📦 Earnings Power Box": [
        ("📦 Earnings Power", "Strong on BOTH Heiserman tests — defensive (stable, profitable) and enterprising (reinvesting for growth). The top box."),
        ("💰 Cash Cow", "Defensive but not enterprising — stable and profitable, but reinvesting little for growth."),
        ("🚀 Cash-Hungry Grower", "Enterprising but not defensive — reinvesting hard for growth, but the base economics aren't yet stable."),
        ("⚠️ Weakest", "Weak on both Heiserman tests — neither stable economics nor productive reinvestment."),
    ],
    # ── 🚀 Multibagger Setups (Candidate Flags) — mirrors the Discovery filter group's OR-list
    # (ui_discovery.py:644 _MULTIBAGGER). Exact dropdown labels so the filter and Reference read 1:1;
    # the EP Power Curve + Earnings Power Box dropdowns in the same group are the two categories above. ──
    "🚀 Multibagger Setups": [
        ("🐘 100x Candidate", "MOSL's Mouse-to-Elephant screen: an early, small-base business with high returns reinvested at scale and a long runway — the raw ingredients for a ~100× multibagger. The rarest, highest-ceiling setup (same flag as the 🐘 entry under MOSL Frameworks)."),
        ("🏅 Category Winner", "The sector leader on the WCS screen: top-30% capital efficiency (ROCE) within its OWN sector AND above-market 5-year revenue growth — winning its category on both quality and growth."),
        ("📈 Compound Growth", "Sustained compounding power — profit growth clears 15% (3Y), 12% (5Y) and 10% (10Y); earnings compound across every horizon, not just one good stretch."),
        ("🛡️ Consistency Champion", "Profits have grown steadily and durably — a 'consistent' compounder whose earnings rise smoothly rather than lumpily. Low-volatility compounding is the coffee-can ideal."),
        ("🔄 ROE Turnaround", "ROE is still below 15% but has turned UP above its 5-year trend — an early-innings quality inflection (a turnaround bargain), caught before the re-rating."),
        ("🧲 Value Migration", "The company is in the top quartile of its sector by revenue growth — a sign that value (demand and market share) is migrating TOWARD this business within its sector."),
        ("💎 Bruised Blue Chip", "A high-quality company trading unusually cheap — below 2× book value — after a setback. Quality on sale: a strong franchise the market has temporarily marked down."),
        ("🚀 Mid→Mega", "A mid-cap with the financial profile to grow into a mega-cap — the size-migration thesis (small/mid → large) that produces the biggest multibaggers."),
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
    "🔍 Soundness — Evidence Confidence": [
        ("High", "The soundness verdict rests on 80%+ evidence coverage — most ranked inputs reported; trust it more."),
        ("Medium", "60–80% evidence coverage — a fair amount of the inputs reported."),
        ("Low", "40–60% evidence coverage — a meaningful share of inputs are missing; treat with care."),
        ("Very Low", "Under 40% evidence coverage — the soundness verdict rests on thin data; tentative."),
    ],
    # ── result_age_days / result_stale_flag — core/data_engine.py:1566 (sign-flipped days_from_result;
    # stale at >120d). Shown as the tearsheet '⏳ Stale Nd' badge + the Discovery 'Hide Stale' filter.
    # A recency signal (how OLD the data is) — sibling to, and distinct from, the Evidence badge above. ──
    "⏳ Result Recency": [
        ("⏳ Stale", "The tearsheet badge shown when the company's most recent reported result is more than 120 days old — the financials may predate recent events, so treat the numbers as potentially out of date. The Discovery tab's 'Hide Stale' filter drops these."),
        ("Result Age (days)", "How many days since the company last reported financial results — higher means staler numbers. The recency sibling to the 🔍 Evidence badge: coverage measures how MUCH of the data reported, this measures how OLD it is."),
    ],
    # ── verdict_axis_governance — core/verdict_engine.py:116 (governance multiplier)
    "🛡️ Soundness — Governance Axis": [
        ("Govern 🟢 Safe", "No governance penalty — promoter pledge, dilution and related-party signals are clean."),
        ("Govern 🟡 Caution", "A mild governance penalty — one or more governance signals warrant caution."),
        ("Govern 🔴 Risk", "A heavy governance penalty — serious pledge/dilution/related-party risk drags the score."),
    ],
    # ── verdict_axis_forensics — core/verdict_engine.py:111 (nested np.where): the Forensics pill in
    # the 6-axis scorecard, parallel to the Governance axis above. np.where in an engine file isn't
    # enumerated by the categorical-label net, so these need explicit entries. ──
    "🔬 Soundness — Forensics Axis": [
        ("Forensics 🟢 Clean", "The scorecard's Forensics pill when accounting signals look clean — fewer than 5 red flags and no severe forensic or Schilit veto."),
        ("Forensics 🟡 Watch", "The Forensics pill when 5 or more red flags fire but no hard veto — some accounting-quality caution; read the specific flags before acting."),
        ("Forensics 🔴 Flagged", "The Forensics pill when a severe forensic veto fires — forensic score below 50, ten or more red flags, or a Schilit checker hard-fail. A serious accounting-quality concern."),
    ],
    # ── verdict_top_risk — core/verdict_engine.py:121 (the single most important risk)
    "⚠️ Soundness — Top Risk": [
        ("🚨 Severe forensic / accounting-quality flags", "The dominant risk: severe forensic red flags veto the thesis — verify the accounts before anything."),
        ("💀 Value-destroying capital allocation", "The dominant risk: a Gruesome capital allocator earning below its cost of capital — destroys value."),
        ("🕵️ Schilit forensic checker flags", "The dominant risk: Schilit accounting-quality checkers fire — the reported numbers may be aggressive."),
        ("⚠️ Governance risk (pledge/dilution)", "The dominant risk: governance — meaningful promoter pledging or dilution."),
        ("⏳ Poor entry timing — wait for a base", "The dominant risk is timing, not quality — the chart is poorly placed; wait for a base."),
        ("🔍 Thin data — verdict tentative", "The dominant caveat: evidence coverage is thin, so the soundness verdict is tentative until more inputs report."),
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
    # ── Frameworks — _FW_META (ui_tearsheet.py:853); explanations condensed from docs/handbook/08 ──
    # Each is a famous investor's/book's pass-fail screen; a Tear-Sheet shows which a stock passes.
    "🏛️ Frameworks — MOSL Wealth Creation": [
        ("🥇 QGLP", "Raamdeo/MOSL flagship — Quality + Growth + Longevity at a reasonable Price: ROCE ≥~15%, 5-yr profit growth ≥~15%, PEG ≤~1.5 with positive earnings."),
        ("🌟 MOSL Wealth Creator", "The proven long-run wealth-creator profile — profit growth consistent across 3/5/10-yr horizons, a low payback ratio, and a wide economic-profit spread."),
        ("👑 SQGLP Century Stock", "The strictest bar — QGLP plus Size: the full quality-growth-price test on a small/mid base that can still multiply many times. Passes are very rare."),
        ("🐘 100x Candidate", "MOSL Mouse-to-Elephant — an early, small-base business with high returns reinvested at scale and a long runway: the raw ingredients for a ~100× multibagger."),
        ("🩹 Fallen Quality", "A genuinely high-quality business temporarily beaten down — strong long-run quality plus a sharp recent price/valuation fall. Quality on sale."),
        ("📐 CAP-GAP Compounder", "MOSL 22nd study: RoE of 15%+ sustained across the decade, five years and today (a long Competitive Advantage Period) AND profit growth of 15%+ across all three windows (a long Growth Advantage Period). Longevity proven on both axes."),
        ("🏰 Economic Moat", "MOSL 17th study: return on equity above the SECTOR average in at least 4 of 5 time windows — a durable edge over direct peers. Sector-relative by design: a 20% RoE bank can have a moat a 50% RoE consumer company lacks."),
        ("💙 Blue Chip Quality", "An established, high-quality large-cap — large size with strong, stable quality metrics. A proven, lower-risk compounder."),
        ("🌪️ Consistent in Volatile", "A steady performer through volatile markets — low earnings/return volatility alongside solid quality. Holds up when the market doesn't."),
        ("🏒 EP Hockey Stick", "MOSL's 28th study: earns above its cost of equity, that profit is growing, AND the shares still trade at a P/E of 20 or less — the study's full TEMP setup (value creation bought cheap)."),
        ("🏛️ Bruised Blue Chip 29", "MOSL's 29th study — a quality large-cap fallen hard, trading at a steep discount (P/B ≤~2×) to its own history. Temporarily punished by the market."),
        ("🌐 Multi-Trillion Cap", "MOSL's 30th study — the very largest, most-proven compounders: mega-cap size with elite, durable quality. Battleship-grade."),
    ],
    "📚 Frameworks — Fundamental & Cash Quality": [
        ("☕ Coffee Can", "Mukherjea — own clean, consistent compounders and forget them: ROCE ≥15% over 10 & 5 yrs, revenue up every year, CFO/EBITDA ≥90%, D/E <1, pledge <10%."),
        ("💎 Diamond", "Mukherjea's 'Diamonds in the Dust' — forensic-verified compounders: high consistent returns, low debt, and clean accounting that survives the forensic screen."),
        ("🕊️ Peaceful Investing", "Vijay Malik's systematic forensic filter — margins, cash conversion, low debt (interest cover ≥3×), self-funded growth, clean forensics. Sleep-at-night investing."),
        ("💰 Unusual Billionaires", "Mukherjea's 'Greatness Formula' — sustained high returns on capital AND consistent revenue growth reinvested over a long period. A great franchise, both engines firing."),
        ("⏳ Long Game Quality", "Khandelwal's fort-like businesses — the strictest balance-sheet bar (interest cover ≥5×) plus strong free cash flow after capex (FCF ≥60% of profit). Funds its own future."),
        ("📚 Baid Compounder", "Gautam Baid's steady compounding — solid quality with balance-sheet strength and consistent growth. A dependable, disciplined compounder."),
        ("🏅 Basant 30% Club", "Basant Maheshwari — a fast grower at a reasonable price: high sustained earnings growth (book bar ~20%+) at a valuation that still leaves room."),
        ("⭐ Quality Compounder", "The 'Quality Investing' three-circle compounder — low capital intensity (asset turnover >4), a free-cash-flow floor (yield ≥2%), and high returns on capital. Asset-light, cash-generative."),
    ],
    "⚡ Frameworks — Momentum & Growth": [
        ("📡 CAN SLIM", "O'Neil — elite growth at a confirmed breakout: current EPS & sales each +25%, 5-yr EPS ≥25%, ROE ≥17%, within 15% of the 52-wk high, volume ≥1.5×, top-20% RS, institutions buying, market not bearish. Very rare."),
        ("⚡ SEPA Momentum", "Minervini — Specific Entry Point Analysis: the trend template (above rising MAs), a volatility contraction (VCP), relative strength, institutional support, and an earnings catalyst."),
        ("🚀 Quality Momentum", "Wesley Gray's Quantitative Momentum — top-20% relative strength plus a governance guard (pledge ≤30%) and quality. Durable price momentum in a quality name."),
        ("👓 Lynch Dream Framework", "Peter Lynch's growth-at-a-reasonable-price — strong EPS growth at PEG ≤~1, modest institutional ownership (room to be discovered), real free cash flow, and no inventory surge."),
        ("📈 EP Improver", "MOSL 28th study's turnaround stage: economic profit still NEGATIVE but climbing, with returns, capital efficiency and margins all turning at once. The study calls this stage speculative — completed turnarounds earned its best returns, at lower odds. The step before 🏒 EP Hockey Stick."),
        ("😊 SMILE", "Vijay Kedia — Small size, Medium experience, Large aspiration, Extra-large potential, with Integrity: a ₹100–2,000 Cr small-cap, 5-yr growth ≥20%, ROCE ≥20%, honest management."),
    ],
    "🛡️ Frameworks — Valuation & Capital Allocation": [
        ("🧮 Magic Formula", "Greenblatt — cheap AND good: high earnings yield (EBIT/EV ≥8%) and high return on capital (ROCE ≥20%), excluding financials/utilities. A high-return business at a cheap enterprise price."),
        ("🎲 Dhandho Asymmetry", "Pabrai — heads-I-win, tails-I-don't-lose-much: fallen ≥30% from the 52-wk high (perceived risk) yet a high FCF yield ≥8% (actual low risk), with clean forensics."),
        ("🔄 Parikh Contrarian", "Parag Parikh — out-of-favour but sound: low valuation (P/E <20), strong liquidity (current ratio >1.5), and decent returns (5-yr ROCE ≥12%). A sensible contrarian value candidate."),
        ("🌊 Wide Moat", "Pat Dorsey — structural, durable moats: high returns (ROCE ≥20% both windows), a healthy FCF yield ≥5%, and a moat that isn't eroding. Wide and holding."),
        ("🎯 Outsider CEO", "Thorndike's 'Outsiders' — elite capital allocators: disciplined buybacks without dilution, strong cash generation, debt discipline. Compounds per-share value."),
        ("🔮 Expectations Matrix", "Mauboussin — the price embeds expectations: judge the growth/returns the price implies versus what the business can deliver. A pass = the market implies less than the business can."),
        ("🕵️ Schilit Clean", "Howard Schilit's Financial Shenanigans screen, passed — the stock clears the four-checker accounting-manipulation perimeter (at most two checkers firing)."),
        ("🛡️ Marks Cycle Shield", "Howard Marks — respect the cycle: price-vs-value and cycle position are favourable, not late-cycle euphoric. A defensive overlay."),
    ],
    "🎣 Frameworks — Fisher & Mayer": [
        ("🎣 Fisher Quality", "Philip Fisher's 15 qualitative points, as automated proxies — margins, growth durability, R&D/efficiency, management quality. Scores like a Fisher 'uncommon' franchise."),
        ("📶 Fisher Scalability", "Does the business still have room to grow — a revenue runway, operating leverage, pricing power, and no dilution. The growth story isn't finished."),
        ("💯 100-Bagger", "Phelps/Mayer '100 Baggers' — the long-compounding, small-base setup: growth consistent across horizons, a low payback ratio, a wide economic-profit spread, a small base, and low pledging."),
    ],
    # ── Market regime (scoring_engine detect_market_regime) ──
    # Market-wide readings shown on the Market Pulse tab + the banner — never traits of one stock.
    "🌊 Market & Regime": [
        ("🐂 BULL", "The whole market is in a healthy uptrend — strong breadth (most stocks above their long-term averages). PRISM trusts momentum signals more in a Bull regime."),
        ("🐻 BEAR", "The whole market is weak — poor breadth. PRISM gets stricter and BLOCKS new momentum (CAN SLIM) entries, since most stocks fall in a falling market. Weight quality and caution."),
        ("➡️ SIDEWAYS", "The market is range-bound — breadth is mixed, neither clearly rising nor falling. The default, in-between regime."),
        ("🌊 Tsunami", "The rarest, highest-conviction setup — all SEVEN conviction conditions (quality + momentum + governance + technical) fire at once. Often only a handful exist, sometimes none."),
        ("🚀 Tipping Points", "A Market Pulse watch-list of stocks at a potential inflection — where a change in the business may be about to accelerate. Context, not a verdict."),
    ],
    # ── Sizing-cockpit cards — ui_tearsheet.render_valuation_inversion_and_sizing_cockpit (~2999) ──
    # Deep Layer-3 metric cards on the Tear-Sheet's Matrix & WCS tab; value + status only, so the
    # term itself needs defining here (read from the cockpit's labels/thresholds).
    "🔮 Returns & Mispricing Cockpit": [
        ("👑 Expected Return Estimate", "A three-part estimate of the annual return the business itself could deliver: sustainable growth + free-cash-flow yield + a gradual re-rating toward a quality-justified P/E. Each part is capped at sensible limits and the card shows the split, so one broken input can never fabricate a wild number. A model estimate from fundamentals — not a price forecast."),
        ("⏳ Margin Trend (5Y Tau)", "Whether OPERATING MARGINS have been trending up or fading across roughly the last five years (a rank-correlation over the margin history). Positive (above ~0.25) = margins strengthening, often a widening moat; negative = margin pressure. Direction only, not level."),
        ("📊 Price vs Fundamentals", "Where this stock's price sits versus every other stock AFTER adjusting for fundamentals (profitability, growth, size). 'Cheaper than 80%' means only one in five stocks is priced lower for similar quality. A relative-value rank — remember cheap can be cheap for a reason."),
        ("🚨 Hard Volatility Stop-Loss Level", "The price at which a volatility-based trailing stop would trigger — a risk-management exit reference for sizing a position, never a price target or a forecast."),
        ("🎯 Executable Capital Weight", "The position size the sizing engine can justify TODAY as a % of the portfolio — a fractional-Kelly heuristic on proxy odds, capped by the 1%-risk volatility-stop rule, per-stock (not portfolio-normalized). Read it beside the EV thesis band: a 0% here with a high band means 'target — but no entry right now'."),
        ("💰 Capital Deployment (10L Base)", "The rupee amount the Executable Capital Weight implies on an illustrative ₹10-lakh portfolio — the same weight expressed in rupees, to make the sizing concrete."),
        ("Value Creation Velocity", "Reinvestment rate × capital spread (the ROCE earned above the cost of capital) — how fast the company compounds intrinsic value by reinvesting at high returns. Higher = faster wealth creation."),
        ("Market-Implied Expectations Gap", "The gap between the growth the current share price already implies and the growth the business actually needs to justify it (Mauboussin). Positive = the market expects MORE than the fundamentals require — a high bar to clear."),
    ],
    # ── Mauboussin Expectations-Investing Radar — ui_tearsheet.render_mauboussin_radar (~2817): the
    # PIE audit (T/O/C pillars) + per-stock Payoff Framework. These are numeric / UI-composed card
    # signals (not np.select categoricals), so the label-coverage net doesn't enumerate them — grounded
    # in mauboussin_expectations_specs.json. The framework one-liner lives under "Frameworks" above. ──
    "🔮 Mauboussin — Expectations Investing": [
        ("Price-Implied Expectations (PIE) Audit", "Mauboussin & Rappaport's core move, read off the price: instead of guessing a fair value, judge the expectations the current price already bakes in — and score how many of three gates (Gap / Treadmill / CAP) the stock clears, shown out of 3. Repointed 2026-08-28 (v2.0): the certifying gate is now the book's own ch.7 buy standard — the expectations gap — where the old gate reduced to a single operational flag (the other two legs cleared 95%+ of stocks) and certified 45.8% of the universe."),
        ("G · Expectations Discount", "PIE pillar 1, the CERTIFYING gate — a green check means the growth the price implies (inverted from the P/B–Gordon identity) sits at least 5 points BELOW the growth the business can sustainably fund: a margin of safety in expectations, the book's ch.7 buy standard. Strict: if any input is missing the pillar fails — unverifiable is never certified. The 5-point bar is PRISM's census-calibrated choice; the book deliberately prescribes no universal cutoff."),
        ("T · Treadmill Safety", "PIE pillar 2, a DISQUALIFIER — a green check means the stock is NOT priced for indefinite perfection: it doesn't need a continuous stream of positive surprises just to hold today's price. Red means the price already assumes relentless out-performance."),
        ("C · CAP Trap Clear", "PIE pillar 3, a DISQUALIFIER — a green check means there's no dangerous pairing of a long competitive-advantage-period expectation with DECELERATING returns on capital. The trap: the price assumes a durable moat while ROCE is actually sliding."),
        ("Implied CAP Proxy", "A UNITLESS proxy (P/E × NOPAT margin × retention) for how long the price assumes the business keeps earning excess returns — higher = a longer implied competitive-advantage period, but NOT literal years. Very high values are a caution: the market is paying for a moat that must last improbably long."),
        ("NOPAT Margin", "Net Operating Profit After Tax as a share of sales — the clean, capital-structure-neutral operating profitability that Mauboussin's value-driver math runs on (it strips out financing effects)."),
        ("🧮 Payoff Framework — Expected Excess Return", "The per-stock expected value of the trade: P(Upside) × Upside% − P(Downside) × Downside%, where P(Upside) is the trajectory-calibrated win probability. The book's bar to act is a minimum 5% edge."),
        ("EV Upside %", "The reward leg of the payoff: how far the price could rise to reach the P/E its quality justifies (the gap to a quality-fair multiple)."),
        ("EV Downside %", "The risk leg of the payoff: the distance from today's price down to the volatility stop-loss level. Paired with EV Upside % to size the bet honestly."),
        ("EV Verdict & Position Sizing", "Translates the Expected Excess Return into PRISM's own THESIS sizing bands — roughly: 15%+ edge → High Conviction (8–12% weight); 10%+ → Moderate-High (5–8%); 5%+ → Moderate (3–5%); below 5% → Insufficient Edge, no position. Attribution corrected 2026-08-27: the expected-value method is Mauboussin & Rappaport (Expectations Investing ch.7, 'Buy, Sell, or Hold?'), but the BANDS ARE OURS — the book has 12 chapters (no ch.13) and prescribes no position sizes, no hurdle rate and no Kelly (0 occurrences of each). It gives price-to-expected-value and years-to-convergence, and leaves sizing to the investor. This is a thesis size, not an executable one — see the cockpit's thesis-vs-executable strip."),
    ],
    # ── Schilit Accounting Anomaly Shield — ui_tearsheet.render_schilit_shield (~2162); the four
    # Schilit checkers (schilit_ems/cfs/kms_* flags). Wording follows each checker's own description. ──
    "🛡️ Schilit Anomaly Shield": [
        ("EMS Anomaly Gimmick", "Schilit Earnings-Manipulation check — flags income-statement gimmicks such as aggressive revenue recognition or capitalising expenses to inflate reported profit."),
        ("CFS Cash Flow Trap", "Schilit Cash-Flow-Shenanigans check — flags operating-cash divergence and paper-profit shifts, where reported earnings aren't backed by real operating cash."),
        ("KMS Leverage Mirage", "Schilit Key-Metrics check (leverage) — flags off-balance-sheet guarantees and pledged-cash mismatches that disguise a company's true leverage."),
        ("KMS Operational Bloat", "Schilit Key-Metrics check (operations) — flags channel-stuffing and asset/inventory aging that bloats the balance sheet ahead of trouble."),
    ],
    # ── Economic-profit dynamics + tax — ui_tearsheet EP power-curve strip (~305) + tax_rate_est ──
    "📈 Economic-Profit Dynamics & Tax": [
        ("EP Velocity (YoY)", "The year-on-year change in economic profit (₹ Cr) — how fast the company's profit ABOVE its cost of capital is rising or falling. Rising velocity means value creation is accelerating."),
        ("EP Trajectory", "The company's position on Motilal Oswal's economic-profit power curve (28th study) — where it sits on the create → sustain → erode arc of economic value over time."),
        ("Tax Rate (Est.)", "The estimated effective tax rate, (PBT − PAT) ÷ PBT. A profitable company paying under ~10% can signal deferred-tax exhaustion, tax-holiday reliance, or opaque structuring — a forensic caution, not a positive."),
    ],
    # ── Systematic Fisher Proxy — ui_tearsheet.render_fisher_module (~1033); Fisher's 15 qualitative
    # points, the 7 quantifiable from CSV data. Each entry pairs the Fisher point with its proxy. ──
    "🧠 Systematic Fisher Proxy": [
        ("P1: Market Potential", "Fisher Point 1 — does the business have products or services with enough market room for years of sales growth? Proxy: 5-year revenue growth of 15% or more."),
        ("P4: Sales Org Efficiency", "Fisher Point 4 — an above-average sales and distribution organisation. Proxy: profit growing faster than sales (operating leverage is working)."),
        ("P5: Worthwhile Margins", "Fisher Point 5 — does the business earn a worthwhile profit margin? Proxy: a net profit margin above 10%."),
        ("P6: Margin Trajectory", "Fisher Point 6 — is the company doing what it needs to maintain or improve margins? Proxy: net margin at least as high as last year."),
        ("P10: Accounting Controls", "Fisher Point 10 — sound cost analysis and accounting controls. Proxy: operating cash flow at least 70% of reported profit, so earnings are backed by real cash."),
        ("P13: No Equity Dilution", "Fisher Point 13 — will growth force equity raises that dilute existing holders? Proxy: a stable share count, with no meaningful dilution."),
        ("P15: Accounting Integrity", "Fisher Point 15 — management of unquestionable integrity. Proxy: a clean forensic verdict — a high forensic score with few red flags."),
    ],
    # ── Hard gates + verdict-band states — config.HARD_GATES (descriptions verbatim) + the verdict
    # header's SYSTEM-REJECTED / SELL-ALERT branches in app.py. Every stock must pass ALL gates. ──
    "🚨 Hard Gates & Rejection (Pass ALL)": [
        ("Gate-Passed", "The stock cleared EVERY hard safety gate below — the universal floor a stock must pass before PRISM scores it seriously. 'Gate-passed' means safe and eligible, not 'buy'."),
        ("SYSTEM REJECTED", "The Tear-Sheet Soundness-band state when a stock FAILS any one hard gate — it is eliminated regardless of its other scores. The band names the gate that failed."),
        ("SELL ALERT", "The Soundness-band state when a Baid sell-trigger has fired (e.g. cash collapse, thesis broken) — a held or candidate stock flashing risk; review the Forensics tab before acting."),
        ("Debt Safety (gate)", "Hard gate: debt-to-equity ≤ 1.0 — caps balance-sheet risk before a stock can score (Baid prefers ≤ 0.5)."),
        ("Current Ratio (gate)", "Hard gate: current ratio ≥ 1.0 — a basic liquidity floor, so current assets at least cover current liabilities."),
        ("Pledge Safety (gate)", "Hard gate: promoter shares pledged ≤ 20% — limits the forced-selling risk that comes from promoters pledging stock as collateral."),
        ("Pledge Direction (gate)", "Hard gate: promoter pledging is NOT rising quarter-on-quarter — a rising pledge is an early governance warning."),
        ("Promoter Alignment (gate)", "Hard gate: promoter holding ≥ 30% — the founders must keep meaningful skin in the game."),
        ("Cash Quality (gate)", "Hard gate: operating cash flow ≥ 70% of reported profit — earnings must be backed by real cash, not just accounting accruals."),
        ("No Dilution (gate)", "Hard gate: no predatory equity raise — small ESOP-level dilution passes, but a >10% QIP that dilutes existing holders is rejected."),
        ("Positive OCF (gate)", "Hard gate: operating cash flow must be positive — the business has to actually generate cash from its operations."),
        ("Positive PAT (gate)", "Hard gate: annual profit after tax above zero — loss-making companies do not pass the screen."),
        ("Revenue Floor (gate)", "Hard gate: revenue growth of at least −20% year-on-year — excludes businesses in revenue freefall."),
        ("Mandate Screen (ROCE · Growth · PEG)", "On top of the universal safety gates, each mandate adds its own thesis screen — a minimum ROCE, a minimum growth rate and a PEG ceiling (shown in the banner). 'Mandate fit' = passes both the safety gates and this screen."),
    ],
    # ── Forensic integrity verdicts — forensic_engine.py forensic_label (np.where, ~656) + the
    # Schilit shield pass/fail banner (schilit_pass, score ≥ 70). Binary verdicts shown on the UI. ──
    "🕵️ Forensic Integrity (Schilit & Purity Gates)": [
        ("🟢 Clean", "The forensic-integrity state 'Clean' — the stock clears a strict four-part hard gate: operating cash flow ≥ 80% of profit, promoter pledge under 10%, no share dilution, AND zero red flags. The binary integrity stamp the SQGLP gate relies on."),
        ("⚠️ Integrity Gates Open", "The forensic-integrity state when any one of those four purity conditions is not met — most of the universe sits here, since a single red flag of 28 opens a gate. It means 'not surgically clean', NOT detected manipulation: a stock can hold a high forensic score and a Schilit pass and still show this. Treat it as a prompt to read WHICH flags fired, not as an accusation."),
        ("Perimeter Secure (Schilit)", "The Schilit Anomaly Shield's PASS state — at most two of the four Schilit checkers fired (a Schilit score of 70 or more). The accounting clears the manipulation screen."),
        ("Shenanigan Alert (Schilit)", "The Schilit Anomaly Shield's FAIL state — three or more of the four checkers fired (Schilit score below 70). The accounting raises manipulation concerns; investigate before trusting the reported numbers."),
    ],
    # ── Analysis Mode selector — config.ANALYSIS_MODES (label + description). ──
    "🎛️ Analysis Mode": [
        ("🔀 Hybrid (Quantamental)", "Analysis Mode — scores on BOTH fundamentals and technicals: a great business that institutions are also buying now. The all-round default."),
        ("📚 Fundamental Only", "Analysis Mode — pure business quality, setting price action aside. For long-term, buy-and-hold Coffee Can investors."),
        ("📈 Technical Only", "Analysis Mode — pure price action and institutional money flow (O'Neil rules), with fundamentals set aside."),
    ],
    # ── Scoring Profile selector — config.MASTER_PROFILES (label + description). ──
    "🎚️ Scoring Profile": [
        ("Balanced (QGLP)", "Scoring Profile — Raamdeo Agrawal's QGLP: a balanced weighting of Quality, Growth, Longevity and Price. The all-weather default."),
        ("Value (Marks / Kedia)", "Scoring Profile — beaten-down great businesses bought at a high margin of safety, betting on mean reversion (Howard Marks / Vijay Kedia)."),
        ("Growth (Fisher)", "Scoring Profile — rewards earnings acceleration and tolerates a higher P/E for 20%+ sustained growth (Philip Fisher)."),
        ("Quality (Coffee Can / Buffett)", "Scoring Profile — pure moat: a decade of consistent ROCE, strong free cash flow and minimal debt, ignoring market noise (Coffee Can / Buffett)."),
        ("GARP (Lynch)", "Scoring Profile — Growth at a Reasonable Price, with a mandatory PEG below 1.0 (Peter Lynch's golden rule)."),
        ("Defensive / Cash Cow", "Scoring Profile — capital-protection mode: a free-cash-flow fortress with zero debt."),
        ("Momentum (O'Neil CAN-SLIM)", "Scoring Profile — price and earnings momentum: buy what FII/DII are accumulating right now (O'Neil CAN-SLIM)."),
        ("Turnaround / Special Situation", "Scoring Profile — quarter-on-quarter earnings acceleration plus promoter buying and a volume surge. High risk, high reward."),
    ],
    # ── tier_label — config.CONVICTION_TIERS via apply_forensic_penalty (post-penalty score bands:
    # ≥85 / ≥70 / ≥55 / ≥40 / rest). Added 2026-08-28: the app's most visible labels were absent
    # from the searchable reference (found by the all-label coverage sweep). ──
    "🏆 Conviction Tier": [
        ("🏆 Crown Jewels", "Post-penalty composite score of 85 or more — the highest-conviction compounders; deep-dive and build a position. The rarest band (single digits of 2,100+ stocks)."),
        ("🥇 Strong Compounders", "Score 70–84 — quality with momentum confirmation; watchlist priority."),
        ("🥈 Emerging Quality", "Score 55–69 — quality building and momentum developing; monitor for an upgrade."),
        ("🥉 On Radar", "Score 40–54 — some quality signals but they need time; the early watchlist."),
        ("❌ Not Ready", "Score below 40 — insufficient quality or momentum for the engine to carry a thesis; ignore for now."),
    ],
    # ── atoms_to_bits_label — core/data_engine.py:1419 (sector-mapped, 26th WCS Atoms→Bits;
    # unmapped sectors default to Hybrid). ──
    "🌐 Business Design (Atoms → Bits)": [
        ("Bits", "An asset-light, digital-first business (26th Wealth Creation Study taxonomy) — scales through networks and software rather than plants; sector-mapped."),
        ("Atoms", "A capital-intensive physical business — growth needs plants, inventory and working capital, so scaling is linear with capital."),
        ("Hybrid", "A mixed physical-plus-digital model — also the default for sectors the Atoms→Bits map does not classify either way."),
    ],

}


# ═══════════════════════════════════════════════════════════════════════════════════════════
# WCS_STUDIES — Raamdeo Agrawal's Annual Wealth Creation Studies, explained simply.
#
# HONESTY CONTRACT: an entry ships ONLY after the study has been read COMPLETELY and its
# findings verified against the actual text (the early-era reading program, 2026-08-29 →).
# Nothing here is summarized from second-hand notes — every claim traces to the study itself.
# Entries are appended in the same change as each study's read completes (docs-as-code).
# Keys: study · years · pub · theme · says (plain-language findings) · prism (what the engine took).
# Pure data — zero Streamlit. Rendered by ui_reference.render_wcs_studies(); also emitted into the
# Markdown download. Pinned by tests/test_reference_tab.py.
# ═══════════════════════════════════════════════════════════════════════════════════════════

WCS_STUDIES = [
    {
        "study": "1st Study", "years": "1991–1996", "pub": "June 1996",
        "theme": "The Inquire 100 — where it all began",
        "says": (
            "The founding study. It picks the 100 companies that at least QUADRUPLED their market "
            "value in five years (32%+ a year) and asks what they have in common. The answers set "
            "the template for the next thirty years: speed comes from SMALL — 80 of the 100 started "
            "under ₹150 crore, while the giants gave safety but never topped the speed charts. "
            "87% stuck to ONE business instead of diversifying. Moderate sales growth (median just "
            "23%) was enough, because margins and asset productivity did the compounding. 68 of 100 "
            "shrank their debt or barely grew it. High tax-payers earned HIGHER P/Es — honest "
            "earnings get rewarded. And the crown jewel: in 1991, 63 of the 100 traded at a P/E far "
            "BELOW their ROE — by 1996 the P/E had caught up. The rule: a stock's fair P/E sits "
            "near its sustainable ROE, so buying well below that line carries a built-in margin of "
            "safety."
        ),
        "prism": (
            "The P/E-below-ROE margin-of-safety rule is implemented directly (the pe-vs-ROE spread "
            "and ratio signals). The small-cap-speed finding feeds the mid/small-cap compounder "
            "boost. ROCE-and-ROE rising over time — stated here first — became the ROCE-expansion "
            "thesis behind Economic Profit."
        ),
    },
    {
        "study": "2nd Study", "years": "1992–1997", "pub": "February 1998",
        "theme": "Swimming against the tide — wealth creation in a falling market",
        "says": (
            "The bear-market edition: the Sensex FELL 22% over this window, and only 45 companies "
            "qualified (versus ~300 the year before). The survivors shared three things. LEADERSHIP: "
            "30 of the 45 were #1 or #2 in their business — 'winners are passionate leaders'. "
            "IMPROVING ECONOMICS: average ROCE rose from 20.4% to 24.5% — a right business is not "
            "just superior, it 'gets better with time'. SELF-FUNDING: the first DuPont analysis in "
            "the series showed margins rising, asset turnover improving, and leverage FALLING every "
            "year — growth paid for internally. Blue chips jumped from 40% to 69% of the list: in "
            "bad markets, proven quality wins. The valuation lesson cuts the other way from the 1st "
            "study: 1992's P/Es sat far ABOVE ROE (wealth creators at 44× vs 22% ROE) and were "
            "crushed back to parity — 'prices accorded relatively higher in comparison to long-term "
            "ROE are clearly unsustainable'. Its first-ever Wealth DILUTORS table shows the wreckage: "
            "companies at P/E 100+ with single-digit ROE destroyed thousands of crores."
        ),
        "prism": (
            "The Wealth Dilutors table is the empirical ancestor of the valuation-multiple trap "
            "(high P/E on low ROE gets a valuation-score penalty). The DuPont decomposition is "
            "implemented as the ROE-attribution signals. Blue-chip resilience in downturns echoes "
            "in the cyclicality tiers and regime detection."
        ),
    },
    {
        "study": "3rd Study", "years": "1993–1998", "pub": "January 1999",
        "theme": "Creators and Destroyers — the first two-sided study",
        "says": (
            "Two firsts. FIRST: information technology enters the wealth-creator list (Satyam #1 at "
            "87% a year, Wipro #2) — the study that caught the IT wave before the boom. SECOND: it "
            "ranks the top 50 Wealth DESTROYERS alongside the 100 creators, and the contrast is the "
            "lesson. Destroyers are asset-heavy commodity businesses (76% of them) making "
            "undifferentiated intermediate products with no pricing power — and they destroyed not "
            "just existing capital but ₹127 billion of FRESH capital raised on the way down. "
            "Both groups were focused, so focus alone is not enough: the NATURE of the business "
            "decides. Creators' average ROCE rose again (16% to 20% — third study in a row), and "
            "their earnings were remarkably steady (ROCE varied only 6% around its mean). The "
            "valuation chapter states the identity Price/Book = ROE × P/E — 'P/E multiples are a "
            "function of ROE' — and introduces the first ROE/P-E matrix: companies moving from "
            "high-ROE-low-P/E to high-ROE-high-P/E returned 49% a year, 'the safest investment "
            "strategy', while good businesses that stayed high-ROE 'tend to get better over time'."
        ),
        "prism": (
            "Third independent validation of the P/E-below-ROE margin-of-safety signals (the "
            "high-ROE-low-P/E quadrant was the safest 49%-a-year cohort). The destroyer profile — "
            "asset-heavy, undifferentiated, fed by fresh capital — is what the Atoms/Bits design "
            "labels, the external-financing red flag, and the dilution checks screen against. The "
            "earnings-steadiness finding is the earliest ancestor of the Consistents framework."
        ),
    },
]
