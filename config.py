"""
Multibagger Discovery System — Configuration
=============================================
All thresholds, weights, gate conditions, and scoring parameters.
Single source of truth — every magic number lives here.
"""

# ═══════════════════════════════════════════════════════════════
# 1. DATA PATHS
# ═══════════════════════════════════════════════════════════════
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _get_actual_path(base, folder_name, file_name):
    # Case-insensitive resolution for Linux (Streamlit Cloud).
    # Supports multi-level folder_name (e.g. "Other Resources/CSV Data") — splits on
    # forward slash and resolves each level independently so the same code works on
    # both Windows (backslash native) and Linux (Streamlit Cloud).
    try:
        current = base
        for part in folder_name.replace("\\", "/").split("/"):
            dir_map = {
                item.lower(): item
                for item in os.listdir(current)
                if os.path.isdir(os.path.join(current, item))
            }
            current = os.path.join(current, dir_map.get(part.lower(), part))
        file_map = {item.lower(): item for item in os.listdir(current)}
        actual_file = file_map.get(file_name.lower(), file_name)
        return os.path.join(current, actual_file)
    except Exception:
        return os.path.join(base, *folder_name.replace("\\", "/").split("/"), file_name)

DATA_DIR_NAME = "Other Resources/CSV Data"

# ═══════════════════════════════════════════════════════════════
# 2. MACRO ECONOMIC CONSTANTS
# ═══════════════════════════════════════════════════════════════

# India institutional Cost of Equity — Motilal Oswal 23rd WCS baseline.
# Used in Economic Profit: EP = Net Worth × (RoE − COST_OF_EQUITY).
# Referenced in data_engine.py; update here to stress-test against 12% or 15% hurdle rates.
COST_OF_EQUITY = 12.0
INDIA_GSEC_YIELD = 7.0   # India 10-year G-Sec proxy (update when RBI policy shifts)

CSV_FILES = {
    "ratio":         _get_actual_path(BASE_DIR, DATA_DIR_NAME, "Stockscan - Ratio.csv"),
    "income":        _get_actual_path(BASE_DIR, DATA_DIR_NAME, "Stockscan - Income Statement.csv"),
    "balance":       _get_actual_path(BASE_DIR, DATA_DIR_NAME, "Stockscan - Balance Sheet.csv"),
    "cashflow":      _get_actual_path(BASE_DIR, DATA_DIR_NAME, "Stockscan - Cashflow.csv"),
    "shareholding":  _get_actual_path(BASE_DIR, DATA_DIR_NAME, "Stockscan - Shareholdings.csv"),
    "technical":     _get_actual_path(BASE_DIR, DATA_DIR_NAME, "Stockscan - Technicals.csv"),
}

# Google Sheets Configuration
DEFAULT_SHEET_ID = None  # Must be set before using data_source="sheet"; "" causes cryptic errors deep in data_engine

# Exact tab names inside the user's Google Spreadsheet.
# INVARIANT: These names must never be changed — the data pipeline loads by tab name,
# not by GID. GIDs are per-spreadsheet and always wrong for a different user's sheet.
SHEET_TAB_NAMES = {
    "ratio":        "Ratio",
    "income":       "Income Statement",
    "balance":      "Balance Sheet",
    "cashflow":     "Cashflow",
    "shareholding": "Shareholdings",
    "technical":    "Technicals",
}

# ═══════════════════════════════════════════════════════════════
# 3. MARKET CAP TIERS  (₹ Crores)
# ═══════════════════════════════════════════════════════════════
MCAP_TIERS = {
    "Mega Cap":   {"label": "Mega Cap",   "min": 200_000, "emoji": "🏛️"},
    "Large Cap":  {"label": "Large Cap",  "min": 20_000,  "emoji": "🏢"},
    "Mid Cap":    {"label": "Mid Cap",    "min": 5_000,   "emoji": "🏗️"},
    "Small Cap":  {"label": "Small Cap",  "min": 500,     "emoji": "🔬"},
    "Micro Cap":  {"label": "Micro Cap",  "min": 100,     "emoji": "⚗️"},
    "Nano Cap":   {"label": "Nano Cap",   "min": 0,       "emoji": "🔭"},
}

MCAP_MIN_FLOOR = 0  # No floor — all 2107 stocks included (sheet already categorises)

# ═══════════════════════════════════════════════════════════════
# 3. HARD GATES — Binary Pass/Fail (Layer 1)
# ═══════════════════════════════════════════════════════════════
# Every stock must pass ALL gates before scoring begins.
# Source frameworks tagged for audit trail.

HARD_GATES = {
    "debt_safety": {
        "column": "debt_to_equity",
        "operator": "<=",
        "threshold": 1.0,
        "penalty": -0.20,
        "source": "Coffee Can / Forensic / Baid",
        "description": "D/E ≤ 1.0 — balance sheet risk gate (Baid prefers ≤0.5)",
    },
    "current_ratio": {
        "column": "current_ratio",
        "operator": ">=",
        "threshold": 1.0,
        "source": "Forensic Shenanigans",
        "description": "CR ≥ 1.0 — liquidity safety gate",
    },
    "pledge_safety": {
        "column": "pledged_percentage",
        "operator": "<=",
        "threshold": 20.0,
        "source": "Fisher Point 15 / SQGLP",
        "description": "Pledged % ≤ 20% — promoter collateral risk",
    },
    "pledge_direction": {
        "column": "pledge_rising",  # derived: pledged > pledged_1qb
        "operator": "==",
        "threshold": 0,
        "source": "Forensic Shenanigans",
        "description": "Pledge not increasing QoQ",
    },
    "promoter_alignment": {
        "column": "promoter_holdings",
        "operator": ">=",
        "threshold": 30.0,
        "source": "Fisher / SQGLP",
        "description": "Promoter holdings ≥ 30% — skin in the game",
    },
    "cash_quality": {
        "column": "cfo_to_pat",
        "operator": ">=",
        "threshold": 70.0,  # cfo_to_pat is PERCENTAGE in CSV (e.g. 73.04 = 73%). Was 0.7 (ratio) — gate always passed.
        "source": "Coffee Can Clean Accounts",
        "description": "CFO/PAT ≥ 70% — earnings are real cash",
    },
    # NOTE: SMA200 was here as a hard gate. REMOVED.
    # Rationale: Binary elimination conflicts with the fundamental philosophy.
    # HDFC Bank, Asian Paints, Page Industries all broke below 200D SMA in March 2020.
    # The GOD SCREEN would have eliminated them at their best buying opportunity.
    # SMA200 is now a CONTINUOUS TREND_SIGNAL (20% of trend score) — it penalizes,
    # not eliminates. A quality stock in a correction naturally scores lower on momentum.
    # The human investor's conviction decides whether to act on the fundamental signal.
    # The signal lives in TREND_SIGNALS["above_sma200"] and `d45_trend_structure`.

    "no_dilution": {
        "column": "dilution_flag",   # 0=Clean, 1=ESOP-level, 2=Meaningful, 3=Predatory QIP
        "operator": "<",
        "threshold": 3,
        "source": "Fisher Point 13 — Materiality-adjusted",
        "description": "Dilution <10% tolerated (ESOPs pass). Hard reject only predatory QIP >10%.",
    },
    "positive_ocf": {
        "column": "operating_cash_flow",
        "operator": ">",
        "threshold": 0,
        "source": "Forensic Shenanigans",
        "description": "Operating cash flow must be positive",
    },
    # HG-07: No loss-making companies (PAT must be positive)
    "positive_pat": {
        "column": "pat",
        "operator": ">",
        "threshold": 0,
        "source": "Handbook HG-07 / Buffett quality",
        "description": "Annual PAT > 0 — no loss-makers pass the screen",
    },
    # HG-08: Revenue floor — no revenue-negative or collapsing businesses
    "revenue_floor": {
        "column": "rev_gr_yoy",
        "operator": ">=",
        "threshold": -20.0,
        "source": "Handbook HG-08",
        "description": "Revenue growth YoY ≥ -20% — excludes businesses in freefall",
    },
}

# ═══════════════════════════════════════════════════════════════

# Financial sector stocks get a separate gate set
# Exact INDUSTRY names for df["industry"].isin() — verified against 354 CSV industry values.
# Previous list ("Banking", "NBFC", etc.) matched 0/10 — all were sector names, not industry names.
FINANCIAL_SECTORS = frozenset([
    "Finance - Capital Markets",
    "Finance - Capital Markets - Brokers",
    "Finance - Capital Markets - RTA",
    "Finance - Capital Markets - Wealth Management",
    "Finance - Holding Company",
    "Finance - Housing",
    "Finance - Investment/Others",
    "Finance - PSU Lending",
    "Finance & Investments - Others",
    "NBFC - Holding Companies",
    "NBFC - Others",
    "Insurance - Proxy",
    "Exchanges",
    "Credit Rating Agencies",
    "Infra/Real Estate Investment Trust",
    "Infrastructure Investment Trusts",
    "Real Estate Investment Trusts",
])

# Exact SECTOR names for df["sector"].isin() — verified against 81 CSV sector values.
# Covers: Financial Services, Finance, brokers, and credit agencies.
FINANCIAL_SECTOR_NAMES = frozenset([
    "Financial Services",
    "Finance",
    "Stock/ Commodity Brokers",
    "Credit Rating Agencies",
])

# Regulated utilities — Greenblatt's Magic Formula explicitly excludes "utilities and
# financial stocks" because their returns are rate-capped, distorting the ROC ranking.
# ONLY regulated power/gas distribution — NOT electrical-equipment makers or upstream E&P.
UTILITY_SECTOR_NAMES = frozenset([
    "Power Generation & Distribution",
    "Gas Distribution",
    "Power Infrastructure",
])

# ═══════════════════════════════════════════════════════════════
# 4. QUALITY SCORE WEIGHTS (Layer 2) — 0 to 100
# ═══════════════════════════════════════════════════════════════
QUALITY_WEIGHTS = {
    "moat":          0.22,   # ROCE trajectory, ROE — SQGLP Quality
    "growth":        0.22,   # Revenue/PAT/EPS CAGR — SQGLP Growth
    "cash":          0.20,   # CFO/PAT, FCF yield, self-funding — Coffee Can
    "margin":        0.13,   # NPM, OPM, GPM medians + acceleration — Fisher
    "balance_sheet": 0.13,   # Net debt, reserves growth, CWIP — Baid/Marks
    "valuation":     0.10,   # PE discount, PEG, FCF yield — Marks/Baid Entry Price
}

# Moat sub-signals and their weights within the moat bucket
MOAT_SIGNALS = {
    "roce_med_10y":     0.35,
    "roce_trajectory":  0.15,  # roce_med_7y - roce_med_10y
    "roe_med_10y":      0.25,
    "roe_trajectory":   0.10,
    "roce_current_vs_med": 0.15,  # roce - roce_med_10y (inflection)
}

GROWTH_SIGNALS = {
    "pat_gr_5y":           0.17,
    "pat_gr_10y":          0.10,
    "rev_gr_5y":           0.17,
    "rev_gr_10y":          0.10,
    "eps_gr_5y":           0.15,
    "ebitda_gr_5y":        0.10,
    "pat_acceleration":    0.06,   # pat_gr_3y - pat_gr_5y
    "rev_acceleration":    0.05,   # rev_gr_3y - rev_gr_5y
    "ebitda_acceleration": 0.04,   # ebitda_gr_3y - ebitda_gr_5y
    # Quarterly freshness layer (q_pat_yoy × 0.60 + q_rev_yoy × 0.40) × 0.06
    # NOTE: dict intentionally sums to 0.94. The remaining 0.06 is applied separately in
    # scoring_engine.py compute_growth_score() because quarterly freshness requires a
    # blended sub-formula (0.60/0.40 split) that can't be expressed as a flat signal weight.
}
# Weights sum: 0.17+0.10+0.17+0.10+0.15+0.10+0.06+0.05+0.04 = 0.94 + 0.06 quarterly = 1.00 total

CASH_SIGNALS = {
    "cfo_to_pat":      0.20,   # CFO/PAT % — earnings cash-backing
    "cfo_to_ebitda":   0.15,   # CFO/EBITDA % — clean accounts filter
    "fcf_yield":       0.15,   # FCF/MCap — absolute attractiveness
    "fcf_to_cfo_pct":  0.15,   # FCF/CFO — capex discipline (0 when OCF≤0, not neutral 50)
    "capex_coverage":  0.10,   # OCF/capex multiple
    "fcf_consistency": 0.15,   # FCF consistently positive (binary)
    "self_funding":    0.10,   # SSGR ≥ actual growth — no external debt needed (binary)
}
# Weights sum: 0.20+0.15+0.15+0.15+0.10+0.15+0.10 = 1.00

MARGIN_SIGNALS = {
    "npm_med_5y":       0.25,
    "opm_med_5y":       0.25,
    "gpm_med_5y":       0.15,
    "npm_acceleration": 0.15,   # npm_lq - npm_1yb
    "opm_acceleration": 0.10,   # opm_lq - opm_1yb
    "opm_stable":       0.10,   # OPM within ±20% of 5Y median = pricing power (binary)
}
# Weights sum: 0.25+0.25+0.15+0.15+0.10+0.10 = 1.00

BALANCE_SHEET_SIGNALS = {
    "net_debt_negative": 0.25,  # negative net_debt = fortress (binary)
    "debt_slope_3y":     0.20,  # negative = deleveraging (ascending=False)
    "reserves_growth":   0.15,
    "cwip_conversion":   0.15,  # positive = capacity came online
    "cash_change":       0.15,  # positive = building cash
    "nfat":              0.10,  # Net Fixed Asset Turnover — capital-light moat (Malik)
}
# Weights sum: 0.25+0.20+0.15+0.15+0.15+0.10 = 1.00

# ═══════════════════════════════════════════════════════════════
# 4b. VALUATION SCORE SIGNALS (Marks + Baid Entry Price Discipline)
# ═══════════════════════════════════════════════════════════════
VALUATION_SIGNALS = {
    "pe_discount":     0.20,   # PE vs 10Y median — higher discount = better
    "peg_ratio":       0.25,   # PEG < 1.0 = cheap (confirmed in all 30 MOSL studies)
    "payback_ratio":   0.15,   # Payback < 1x = most reliable MOSL supernormal-return signal
    "ev_compression":  0.15,   # EV/EBITDA falling = value creation
    "fcf_yield_val":   0.15,   # FCF Yield > 3% = attractive
    "de_fortress":     0.10,   # D/E < 0.5 (Baid's fortress gate) = bonus
}
# Weights sum: 0.20+0.25+0.15+0.15+0.15+0.10 = 1.00 ✓

# Payback Ratio zones (market_cap / 5Y cumulative estimated PAT)
# < 1.0: market cap recovered within 5Y of profits = supernormal return territory (MOSL)
# < 2.0: attractive; < 3.0: fair; > 3.0: expensive; > 5.0: very expensive
PAYBACK_ZONES = {
    "supernormal":  {"min": 0,   "max": 1.0, "score": 100},
    "attractive":   {"min": 1.0, "max": 2.0, "score": 80},
    "fair":         {"min": 2.0, "max": 3.0, "score": 60},
    "expensive":    {"min": 3.0, "max": 5.0, "score": 35},
    "very_exp":     {"min": 5.0, "max": 999, "score": 10},
}

# PEG zone scoring (Baid + Marks)
PEG_ZONES = {
    "deep_value":  {"min": 0,   "max": 0.8,  "score": 100},
    "undervalued": {"min": 0.8, "max": 1.2,  "score": 85},
    "fair":        {"min": 1.2, "max": 1.5,  "score": 70},
    "full":        {"min": 1.5, "max": 2.0,  "score": 45},
    "expensive":   {"min": 2.0, "max": 2.5,  "score": 20},
    "extreme":     {"min": 2.5, "max": 999,  "score": 5},
}

# ═══════════════════════════════════════════════════════════════
# 4c. MEAN REVERSION RISK (Marks: "Extremes revert")
# ═══════════════════════════════════════════════════════════════
# Flag stocks where current margins >> 5Y median as cyclical peak risk
MEAN_REVERSION = {
    "opm_spike_threshold": 1.3,    # if OPM_LQ / OPM_Med_5Y > 1.3 = cyclical peak risk
    "npm_spike_threshold": 1.3,    # if NPM_LQ / NPM_Med_5Y > 1.3 = cyclical peak risk
    "penalty_factor":      0.85,   # multiply quality score by this if cyclical peak
}

# ═══════════════════════════════════════════════════════════════
# 5. MOMENTUM SCORE WEIGHTS (Layer 3) — 0 to 100
# ═══════════════════════════════════════════════════════════════
MOMENTUM_WEIGHTS = {
    "relative_strength": 0.30,
    "trend_quality":     0.25,
    "breakout_proximity":0.20,
    "volume_confirm":    0.10,
    "sector_leadership": 0.15,
}

RS_SIGNALS = {
    "crs_50d":          0.40,
    "crs_52w":          0.30,
    "crs_26w":          0.30,
}

TREND_SIGNALS = {
    # above_sma200 moved FROM hard gate TO scoring signal.
    # Penalises stocks below 200D SMA continuously (−20 pts on trend score)
    # rather than eliminating them. Fundamental quality carries through corrections.
    "above_sma200":     0.20,   # Price > SMA 200D — trend direction (was hard gate)
    "vstop_green":      0.20,   # VSTOP 14W 2.5 = green (reduced: correlated with sma200)
    "vstop_fresh":      0.15,   # last change ≤ 30 days (reduced)
    "adx_strong":       0.20,   # ADX 14W > 25 — trend strength (independent signal)
    "rsi_zone":         0.15,   # RSI 55-70 sweet spot
    "golden_cross":     0.10,   # recent golden cross — trend recovery signal
}
# Weights sum: 0.20+0.20+0.15+0.20+0.15+0.10 = 1.00 ✅

BREAKOUT_SIGNALS = {
    "52wh_distance":    0.30,   # % below 52WH — #1 backtested alpha signal (+0.55%/wk)
    "52wh_days":        0.20,   # days since 52WH was set — recency of momentum peak (new)
    "13wh_distance":    0.20,   # quarterly high proximity — near-term breakout
    "breakout_window":  0.20,   # binary breakout flag
    "ath_distance":     0.10,   # all-time high proximity (reduced: ATH can be years old)
}
# Weights sum: 0.30+0.20+0.20+0.20+0.10 = 1.00 ✅

SECTOR_SIGNALS = {
    "ret_vs_industry_1y":  0.55,
    "ret_vs_industry_3m":  0.45,
}

# ═══════════════════════════════════════════════════════════════
# 6. COMPOSITE SCORE BLEND (Layer 4)
# ═══════════════════════════════════════════════════════════════
# Only governance weight lives here — quality/momentum weights are per-ANALYSIS_MODES,
# not global constants. (Previously had "quality": 0.55 / "momentum": 0.30 here,
# but they were never read by the scoring engine and created false documentation.)
COMPOSITE_WEIGHTS = {
    "governance": 0.15,
}

# Governance bonus components — positive values sum to 133 (raw max), engine clamps to 100 via _safe_clip.
# Dilution penalties are DEDUCTIONS applied inside compute_governance_bonus.
# Tier 3 (>10%) never reaches scoring — hard gate eliminates it first.
# Tier 2 (3-10%): passes gate, but loses 25 governance pts — visible, proportional.
# Tier 1 (<3% ESOP): passes gate, -5 pts — distinguishes from zero-dilution companies.
# Tier 0 (clean): no penalty.
GOVERNANCE_BONUS = {
    "promoter_buying":         20,   # promoter increased holding this Q
    "fii_accumulating":        15,   # FII buying this Q
    "dii_accumulating":        10,   # DII buying this Q
    "inst_convergence":        15,   # FII + DII both buying same Q — intentional triple-stack (FII+DII+convergence=40pts): convergence of smart money is qualitatively stronger than either alone
    "insider_trading_present": 15,   # directors buying
    "pledge_falling_1y":       10,   # pledge reduced over 1 year
    "undiscovered_alpha":      15,   # low FII + Tier C mcap
    # Promoter holding alignment — Mayer 100-Bagger: present in 10/10 Indian 100-baggers.
    # Rewards the BASELINE alignment level, not just quarterly buying activity.
    # Dynasty mode (≥60%): founder's wealth IS the stock — decades-horizon thinking.
    # Well-aligned (50-60%): meaningful skin in game without full dynasty mode.
    # Selling from low base (<40% + declining): promoter telling you something the price hasn't yet reflected.
    "promoter_high_alignment":  15,  # holdings ≥ 60%: dynasty mode
    "promoter_good_alignment":   8,  # holdings 50-60%: well-aligned owner-operator
    # 3-year promoter trend — most powerful ownership signal; single quarter buys/sells are noise
    "promoter_3y_accumulation": 10,  # 3Y net buying > 3%: sustained conviction = dynasty building
    # Dilution: tier-1 minor ESOP stays a small additive deduction; tier-2 (3-10%) is a
    # HARD RISK SIGNAL handled by GOVERNANCE_RISK_MULTIPLIERS below, not additive points.
    "dilution_tier1_minor":    -5,   # <3% ESOP dilution: minor deduction vs zero-dilution
}

# ── Asymmetric Governance Risk Shield ──
# Negative ownership signals predict DISASTERS far better than positive signals predict
# winners (Yes Bank, DHFL, Zee, Manpasand all showed promoter exit / pledge / dilution
# patterns before collapse, while fundamentals still screened fine). Therefore:
#   Positive signals → additive governance_bonus (engine, GOVERNANCE_BONUS above)
#   Negative signals → composite MULTIPLIER (shield, this table)
# A multiplier scales with conviction: a 90-composite stock loses more absolute points
# than a 20-composite stock — exactly right, because the risk threatens a larger position.
# The four hard risk signals counted in gov_risk_count (scoring_engine.compute_governance_bonus):
#   1. Tier-2 dilution (dilution_flag == 2: 3-10% share dilution)
#   2. Promoter 3Y systematic exit  (change_promoter_3y < -5)
#   3. Promoter 2Y recent exit      (change_promoter_2y < -3 AND 3Y >= -5) — mutually
#      exclusive with #2 by construction; the same exit is never double-counted
#   4. Low + declining promoter     (promoter_holdings < 40 AND change_promoter_1y < 0)
# Deliberately milder than the forensic cascade (x0.50 floor): ownership signals are
# warnings; forensic red flags are evidence.
GOVERNANCE_RISK_MULTIPLIERS = {
    0: 1.00,   # no ownership risk signals
    1: 0.92,   # one signal — caution
    2: 0.82,   # two signals — structural concern
    3: 0.70,   # three or more — the promoter is telling you something
}

# ═══════════════════════════════════════════════════════════════
# 7. CONVICTION TIERS
# ═══════════════════════════════════════════════════════════════
CONVICTION_TIERS = [
    {"min": 85, "tier": 1, "label": "Crown Jewels",       "emoji": "🏆", "color": "#FFD700",
     "description": "Highest conviction compounders — deep-dive and build position"},
    {"min": 70, "tier": 2, "label": "Strong Compounders",  "emoji": "🥇", "color": "#3fb950",
     "description": "Quality with momentum confirmation — watchlist priority"},
    {"min": 55, "tier": 3, "label": "Emerging Quality",    "emoji": "🥈", "color": "#58a6ff",
     "description": "Quality building, momentum developing — monitor for upgrade"},
    {"min": 40, "tier": 4, "label": "On Radar",            "emoji": "🥉", "color": "#d29922",
     "description": "Some quality signals, needs time — early watchlist"},
    {"min": 0,  "tier": 5, "label": "Not Ready",           "emoji": "❌", "color": "#f85149",
     "description": "Insufficient quality or momentum — ignore"},
]

# ═══════════════════════════════════════════════════════════════
# 7b. MARKS CYCLE TEMPERATURE GAUGE
# ═══════════════════════════════════════════════════════════════
# 5-Dimension market temperature (scored 1-5 each, total 5-25)
# This is a MANUAL input updated quarterly — system provides the framework.
MARKS_CYCLE = {
    "posture_aggressive": {"max_score": 10, "label": "🟢 Aggressive",
                           "action": "Deploy capital into quality. Fat pitch territory."},
    "posture_neutral":    {"max_score": 18, "label": "🟡 Neutral",
                           "action": "Maintain portfolio, selective additions only."},
    "posture_defensive":  {"max_score": 25, "label": "🔴 Defensive",
                           "action": "Reduce equity, accumulate dry powder, wait."},
}
# Default temperature (user adjusts via Config tab)
DEFAULT_CYCLE_TEMPERATURE = {
    "valuations": 3,         # 1=cold (PE<17) to 5=hot (PE>25)
    "credit_conditions": 3,  # 1=tight to 5=loose
    "investor_psychology": 3, # 1=fear to 5=greed
    "capital_markets": 3,    # 1=no IPOs to 5=IPO mania
    "market_quality": 3,     # 1=quality leads to 5=junk leads
}

# Baid's 3 Sell Triggers (alert system)
BAID_SELL_TRIGGERS = {
    "thesis_broken": {
        "description": "ROCE declining structurally (3Y trajectory negative)",
        "check": "roce_trajectory < -3",
    },
    "management_deteriorated": {
        "description": "Pledge rising + promoter selling + D/E rising",
        "check": "pledge_rising AND change_promoter_lq < 0 AND de_slope_3y > 0",
    },
    "cash_quality_collapse": {
        "description": "CFO/PAT dropped below 50% (was above 70%) — cfo_to_pat is stored as PERCENTAGE",
        "check": "cfo_to_pat < 50",
    },
}

# ═══════════════════════════════════════════════════════════════
# 7c. ANALYSIS MODES — Controls Fundamental vs Technical balance
#     Each mode specifies which Scoring Profiles are valid for it.
# ═══════════════════════════════════════════════════════════════
ANALYSIS_MODES = {
    "Hybrid": {
        "label": "🔀 Hybrid (Quantamental)",
        "fundamental_w": 0.70,
        "momentum_w": 0.30,
        "description": "Best of both — great business + institutions are buying it now",
        "allowed_profiles": [
            "Balanced", "Value", "Growth", "Quality",
            "Momentum", "GARP", "Turnaround", "Defensive",
        ],
    },
    "Fundamental": {
        "label": "📚 Fundamental Only",
        "fundamental_w": 1.00,
        "momentum_w": 0.00,
        "description": "Pure business quality — for long-term buy-and-hold Coffee Can investors",
        "allowed_profiles": [
            "Balanced", "Value", "Growth", "Quality", "GARP", "Defensive",
        ],
    },
    "Technical": {
        "label": "📈 Technical Only",
        "fundamental_w": 0.10,
        "momentum_w": 0.90,
        "description": "Pure price action — follow institutional money flow with O'Neil rules",
        "allowed_profiles": [
            "Momentum", "Turnaround",
        ],
    },
}

# ═══════════════════════════════════════════════════════════════
# 7d. MASTER PROFILES — The Policy Engine (Config Factory Pattern)
# Each profile carries its own QGLP weights, gate thresholds,
# forensic sensitivity, and UI priority columns.
# ═══════════════════════════════════════════════════════════════
MASTER_PROFILES = {
    # ── FUNDAMENTAL-DOMINANT PROFILES ──
    "Balanced": {
        "label": "Balanced (QGLP)",    "icon": "⚖️",
        "description": "Raamdeo Agrawal's QGLP — balanced Quality, Growth, Longevity, Price",
        "quality_w": 0.35, "growth_w": 0.35, "longevity_w": 0.15, "price_w": 0.15,
        "roce_gate": 15.0, "growth_gate": 15.0, "peg_gate": 1.5,
        "forensic_boost": 1.0,
        "priority_cols": ["quality_score", "growth_score", "roce", "pat_gr_5y", "peg"],
    },
    "Value": {
        "label": "Value (Marks / Vijay Kedia)",    "icon": "💰",
        "description": "Beaten-down great businesses — high margin of safety, mean reversion",
        "quality_w": 0.40, "growth_w": 0.20, "longevity_w": 0.20, "price_w": 0.20,
        "roce_gate": 12.0, "growth_gate": 8.0, "peg_gate": 2.0,
        "forensic_boost": 1.2,
        "priority_cols": ["pe_discount", "ev_ebitda", "dist_52wh", "peg", "valuation_score"],
    },
    "Growth": {
        "label": "Growth (Philip Fisher)",    "icon": "🚀",
        "description": "Earnings acceleration — tolerates higher PE for 20%+ sustained growth",
        "quality_w": 0.20, "growth_w": 0.50, "longevity_w": 0.15, "price_w": 0.15,
        "roce_gate": 15.0, "growth_gate": 20.0, "peg_gate": 2.5,
        "forensic_boost": 0.8,
        "priority_cols": ["pat_gr_5y", "rev_gr_5y", "eps_gr_5y", "pat_gr_yoy", "growth_score"],
    },
    "Quality": {
        "label": "Quality (Coffee Can / Buffett)",    "icon": "🛡️",
        "description": "Pure moat — ROCE 10Y consistency, free cashflow, zero debt. Ignores noise",
        "quality_w": 0.55, "growth_w": 0.20, "longevity_w": 0.20, "price_w": 0.05,
        "roce_gate": 20.0, "growth_gate": 10.0, "peg_gate": 3.0,
        "forensic_boost": 1.5,
        "priority_cols": ["roce_med_10y", "cfo_to_pat", "npm_med_5y", "debt_to_equity", "moat_score"],
    },
    "GARP": {
        "label": "GARP (Peter Lynch)",    "icon": "🎯",
        "description": "PEG < 1.0 mandated — Growth at a Reasonable Price. Lynch's golden rule",
        "quality_w": 0.30, "growth_w": 0.35, "longevity_w": 0.15, "price_w": 0.20,
        "roce_gate": 15.0, "growth_gate": 15.0, "peg_gate": 1.0,
        "forensic_boost": 1.0,
        "priority_cols": ["peg", "pat_gr_5y", "pe", "valuation_score", "growth_score"],
    },
    "Defensive": {
        "label": "Defensive / Cash Cow",    "icon": "🏰",
        "description": "Free cash flow fortress, zero debt, capital protection mode",
        "quality_w": 0.50, "growth_w": 0.10, "longevity_w": 0.35, "price_w": 0.05,
        "roce_gate": 12.0, "growth_gate": 5.0, "peg_gate": 4.0,
        "forensic_boost": 1.8,
        "priority_cols": ["free_cash_flow", "debt_to_equity", "cfo_to_pat", "current_ratio", "moat_score"],
    },
    # ── MOMENTUM-DOMINANT PROFILES ──
    "Momentum": {
        "label": "Momentum (O'Neil CAN-SLIM)",    "icon": "⚡",
        "description": "Price + Earnings momentum — buy what FII/DII are accumulating RIGHT NOW",
        "quality_w": 0.25, "growth_w": 0.35, "longevity_w": 0.15, "price_w": 0.25,  # weights sum: 1.00
        "roce_gate": 12.0, "growth_gate": 15.0, "peg_gate": 3.0,
        "forensic_boost": 0.7,
        "priority_cols": ["crs_50d", "ret_vs_n500_3m", "momentum_score", "rsi_14d", "dist_52wh"],
    },
    "Turnaround": {
        "label": "Turnaround / Special Situation",    "icon": "🔄",
        "description": "QoQ acceleration + promoter buying + volume surge. High risk, high reward",
        "quality_w": 0.20, "growth_w": 0.45, "longevity_w": 0.15, "price_w": 0.20,  # weights sum: 1.00
        "roce_gate": 8.0, "growth_gate": 0.0, "peg_gate": 5.0,
        "forensic_boost": 1.3,
        "priority_cols": ["pat_gr_yoy", "change_promoter_lq", "crs_50d", "volume", "pat_lq"],
    },
}

# ═══════════════════════════════════════════════════════════════
# 7e. REGIME-ADAPTIVE WEIGHT ADJUSTMENTS
# When the market regime is auto-detected, these adjustments are
# applied ON TOP of the selected Scoring Profile's base weights.
# Positive = boost that factor, Negative = suppress that factor.
# Gates tighten in greed, loosen in fear.
# ═══════════════════════════════════════════════════════════════
REGIME_ADJUSTMENTS = {
    "BULL": {
        "label": "🟢 Bull Market — Offence Mode",
        # Shift weights: boost Growth + reduce Price conservatism
        "quality_delta":   -0.05,   # slightly less defensive
        "growth_delta":    +0.10,   # chase earnings acceleration
        "longevity_delta": -0.05,   # longevity less critical in bull
        "price_delta":     +0.00,   # keep price neutral
        # Gates loosen slightly — rising tide lifts quality boats
        "roce_gate_delta":   0.0,
        "growth_gate_delta": +5.0,  # demand even higher growth in bull
        "peg_gate_delta":   +0.5,   # tolerate slightly higher PEG
        # Momentum gets extra weight in composite blend
        "momentum_boost": 1.10,
    },
    "BEAR": {
        "label": "🔴 Bear Market — Defence Mode",
        # Shift weights: boost Quality + Longevity, suppress Growth
        "quality_delta":   +0.15,   # fortress quality demanded
        "growth_delta":    -0.10,   # growth doesn't matter if market is crashing
        "longevity_delta": +0.05,   # survivors with 10Y track record
        "price_delta":     -0.10,   # ignore valuation (everything looks cheap)
        # Gates tighten — only the best survive
        "roce_gate_delta":   +5.0,  # ROCE > 20% demanded
        "growth_gate_delta": -5.0,  # relax growth gate (everyone is suffering)
        "peg_gate_delta":   +1.0,   # relax PEG (denominator is depressed)
        # Momentum is dangerous in bear — suppress
        "momentum_boost": 0.70,
    },
    "SIDEWAYS": {
        "label": "🟡 Sideways Market — Neutral",
        # No adjustments — pure profile weights apply
        "quality_delta":   0.0,
        "growth_delta":    0.0,
        "longevity_delta": 0.0,
        "price_delta":     0.0,
        "roce_gate_delta":   0.0,
        "growth_gate_delta": 0.0,
        "peg_gate_delta":   0.0,
        "momentum_boost": 1.0,
    },
}


def get_adaptive_weights(profile_name: str, regime: str = "SIDEWAYS") -> dict:
    """The Weight Factory — cascades Profile → Regime → Final Weights.
    
    Returns a dict with final QGLP weights, gate thresholds, and momentum boost,
    all adjusted for the current market regime.
    """
    profile = MASTER_PROFILES.get(profile_name, MASTER_PROFILES["Balanced"])
    adj = REGIME_ADJUSTMENTS.get(regime, REGIME_ADJUSTMENTS["SIDEWAYS"])

    # 1. Apply regime deltas to base profile weights
    raw_q = profile["quality_w"]   + adj["quality_delta"]
    raw_g = profile["growth_w"]    + adj["growth_delta"]
    raw_l = profile["longevity_w"] + adj["longevity_delta"]
    raw_p = profile["price_w"]     + adj["price_delta"]

    # 2. Clamp to [0.05, 0.80] — never zero out a factor completely
    raw_q = max(0.05, min(0.80, raw_q))
    raw_g = max(0.05, min(0.80, raw_g))
    raw_l = max(0.05, min(0.80, raw_l))
    raw_p = max(0.05, min(0.80, raw_p))

    # 3. Re-normalize so they sum to exactly 1.0
    total = raw_q + raw_g + raw_l + raw_p
    final_q = round(raw_q / total, 3)
    final_g = round(raw_g / total, 3)
    final_l = round(raw_l / total, 3)
    final_p = round(1.0 - final_q - final_g - final_l, 3)  # absorb rounding error

    # 4. Apply regime deltas to gate thresholds
    final_roce_gate   = max(5.0, profile["roce_gate"]   + adj["roce_gate_delta"])
    final_growth_gate = max(0.0, profile["growth_gate"]  + adj["growth_gate_delta"])
    final_peg_gate    = max(0.5, profile["peg_gate"]     + adj["peg_gate_delta"])

    return {
        "quality_w":     final_q,
        "growth_w":      final_g,
        "longevity_w":   final_l,
        "price_w":       final_p,
        "roce_gate":     final_roce_gate,
        "growth_gate":   final_growth_gate,
        "peg_gate":      final_peg_gate,
        "forensic_boost": profile["forensic_boost"],
        "momentum_boost": adj["momentum_boost"],
        "priority_cols":  profile["priority_cols"],
        "regime":         regime,
        "profile_name":   profile_name,
        "regime_label":   adj["label"],
    }


# ═══════════════════════════════════════════════════════════════
# 7f. WAVE DETECTION ANALYTICS (Institutional Smart Money)
# ═══════════════════════════════════════════════════════════════
WAVE_DETECTION = {
    "vqs_liquidity": 0.50,    # VQS: Volume Strength
    "vqs_smart_money": 0.20,  # VQS: Smart Money Flow
    "vqs_consistency": 0.20,  # VQS: Pattern Consistency
    "vqs_efficiency": 0.10,   # VQS: Price Efficiency
}

# ═══════════════════════════════════════════════════════════════
# EPOCH 2: REINVESTMENT MOAT THRESHOLDS (7th–12th WCS, 2002–2007)
# Three mathematical identities for self-funding compound machines:
#   Identity A: Reinvestment Rate (RR) = 1 − DPR
#   Identity B: Fundamental Growth Capacity = ROE × RR
#   Identity C: Buffett Value Creation Ratio (VCR) ≥ 1.0 to pass
# ═══════════════════════════════════════════════════════════════
EPOCH2_REINVESTMENT = {
    "min_reinvestment_rate":    0.60,   # DPR < 40% — retaining ≥60% of earnings
    "min_capital_efficiency":   20.0,   # ROCE/ROE baseline hurdle rate (%)
    "min_value_creation_ratio": 1.0,    # Buffett 1-to-1 dollar baseline test hurdle
    "elite_vcr_threshold":      2.0,    # VCR ≥ 2.0 = elite capital allocator
    "quality_boost_pts":        10.0,   # Quality score boost for flag_epoch2_compounder
    "misallocation_penalty":    0.90,   # 10% quality score penalty for capital misallocators
}


# ═══════════════════════════════════════════════════════════════
# EPOCH 3: TAXONOMY & MOAT ENDURANCE (13th–18th WCS, 2008–2013)
# ═══════════════════════════════════════════════════════════════
EPOCH3_TAXONOMY = {
    # Great/Good/Gruesome classification boundaries
    "great_roce_spread_floor": 10.0,    # ROCE - CoC ≥ 10% = structural monopoly
    "great_fcf_velocity_min":  0.60,    # FCF/OCF ≥ 60% = organic cash machine
    "good_roce_spread_floor":  5.0,     # ROCE - CoC ≥ 5% = efficient operator
    "good_fcf_velocity_max":   0.60,    # FCF/OCF < 60% = capex-heavy growth model
    "gruesome_roce_ceiling":   12.0,    # 10Y median ROCE < 12% = value destruction zone

    # Moat Endurance Factor (MEF) — 17th WCS
    "mef_expanding_threshold": 1.0,     # MEF ≥ 1.0 = moat intact/expanding
    "mef_eroding_threshold":   0.80,    # MEF < 0.80 = severe moat degradation

    # Payback Ratio enhancement thresholds (15th WCS)
    "payback_dislocation_max": 2.0,     # Payback < 2.0 during crises = asymmetric setup

    # Capital Return Floor (Epoch 3 structural filter)
    "capital_return_floor_10y": 20.0,   # 10Y median ROCE ≥ 20%
    "capital_return_floor_7y":  20.0,   # 7Y median ROCE ≥ 20%

    # Debt Solvency Perimeter
    "min_interest_coverage":   4.0,     # ICR ≥ 4.0x (stricter than Malik's 3x)
    "max_debt_to_equity":      1.0,     # D/E < 1.0 absolute ceiling
    "preferred_de_non_fin":    0.5,     # D/E < 0.5 preferred for industrials

    # Free Cash Verification Gate
    "cfo_pat_structural_min":  80.0,    # CFO ≥ 80% of PAT (percentage)

    # Anti-Pattern: Cyclical Profit Mirage
    "mirage_rev_growth_min":   25.0,    # Revenue YoY > 25%
    "mirage_roce_10y_max":     12.0,    # 10Y median ROCE < 12%

    # Scoring impact
    "gruesome_quality_penalty": 0.50,   # 50% haircut to quality_score
    "great_quality_boost":      1.10,   # 10% boost for Great companies
}


# ═══════════════════════════════════════════════════════════════
# EPOCH 4: SQGLP & ACCOUNTING INTEGRITY (19th–25th WCS, 2014–2020)
# ═══════════════════════════════════════════════════════════════
EPOCH4_SQGLP = {
    "min_cfo_to_pat_ratio": 80.0,       # Saurabh Mukherjea / 24th WCS Cash Quality Floor — PERCENTAGE (cfo_to_pat is stored as 73.04, not 0.73)
    "max_promoter_pledge": 10.0,        # Strict 10% maximum pledging boundary
    "max_peg_ratio": 1.5,               # 23rd WCS anchor — relaxed from 1.0: Indian quality mid-caps rarely trade at PEG < 1
}


# ═══════════════════════════════════════════════════════════════
# EPOCH 5: MODERN DIGITAL & MACRO FRONTIER (26th–30th WCS, 2011–2025)
# Atoms vs Bits paradigm, Consistents vs Volatiles classification,
# Bruised Blue Chip entry matrix, Multi-Trillion Tipping Point velocity.
# ═══════════════════════════════════════════════════════════════
EPOCH5_MODERN = {
    "bbc_roce_floor":    20.0,          # 29th WCS: minimum 10Y median ROCE for BBC flag
    "bbc_pb_ceiling":    2.0,           # 29th WCS: "typically P/B less than 2x" — verbatim from p.285
    "tipping_sectors": [                # 30th WCS: sectors hitting multi-trillion tipping point
        # Names verified against CSV Sector column (81 unique values)
        "Financial Services",
        "Stock/ Commodity Brokers",     # CSV has space after /
        "Consumer Durables",
        "Automobile",                   # CSV has no trailing 's'
        "Quick Service Restaurant",     # CSV has no trailing 's'
    ],
    "max_volatility_ratio": 0.40,       # 27th WCS: Consistency Coefficient ceiling for Consistents
}

# ═══════════════════════════════════════════════════════════════
# EPOCH 3.5: UNUSUAL BILLIONAIRES — Greatness Formula Config
# Source: Mukherjea, "The Unusual Billionaires" (2016)
# Sector-routed growth hurdles: non-financials use revenue CAGR;
# financials use loan-book expansion proxy (same Greatness Formula
# research base as Coffee Can's explicit 15% financial hurdle).
# ═══════════════════════════════════════════════════════════════
EPOCH35_UNUSUAL_BILLIONAIRES = {
    "non_financial_growth_hurdle": 10.0,  # Revenue CAGR ≥ 10% for industrials/consumer/pharma
    "financial_growth_hurdle":     15.0,  # Loan/Revenue CAGR ≥ 15% for banks and NBFCs
    "capital_efficiency_hurdle":   15.0,  # ROCE (non-fin) / ROE (fin) ≥ 15% — Greatness Formula floor
    "min_promoter_stake":          45.0,  # Owner-operator alignment barrier (UB case studies)
}

# 27th WCS: Sectors MOSL classified as Consistent compounders (sustained earnings 2007–2022).
# Verified against 81 CSV Sector column values — previous inline set in data_engine had 7 wrong names.
# Removed: "Banks - Private Sector" (no such sector), "Utilities" (no such sector).
# Fixed: "Cigarettes"→"Tobacco Products", "Diamonds/Gems/Jewellery"→"Diamond, Gems and Jewellery",
#        "IT"→"IT - Software"+"IT - Hardware", "Oil & Gas"→"Crude Oil & Natural Gas", "Pharma"→"Pharmaceuticals".
CONSISTENT_SECTORS = frozenset([
    "Agro Chemicals",
    "Auto Ancillaries",
    "Cement",
    "Chemicals",
    "Credit Rating Agencies",
    "Diamond, Gems and Jewellery",
    "Engineering",
    "Finance",
    "FMCG",
    "IT - Software",
    "IT - Hardware",
    "Logistics",
    "Crude Oil & Natural Gas",
    "Paints/Varnish",
    "Pharmaceuticals",
    "Refineries",
    "Tobacco Products",
])


# ═══════════════════════════════════════════════════════════════
# 8. FORENSIC ENGINE THRESHOLDS
# ═══════════════════════════════════════════════════════════════
# Fixed denominator for forensic_score: (max_flags - red_flag_count) / max_flags × 100.
# Must be updated manually when a new rf_ flag is added to forensic_engine.py.
# Current flags: 28 active rf_ columns (rf_snoa added 2026-06-13, Quantitative Value audit).
# rf_capex_mirage, rf_ccc_worsening, rf_cwip_bloat, rf_debt_ebitda_high, rf_dilution,
# rf_expense_rising, rf_fcf_to_cfo_low, rf_high_accruals, rf_high_cash_debt, rf_high_receivables,
# rf_inventory_bloat, rf_itr_declining, rf_lease_inflation, rf_low_cfo_ebitda, rf_low_cfo_pat,
# rf_low_fcf_ebitda, rf_margin_squeeze, rf_negative_fcf, rf_nfat_very_low, rf_opm_volatile,
# rf_pledge_elevated, rf_psu_value_destruction, rf_receivables_bloat, rf_rising_debt,
# rf_snoa, rf_ssgr_deficit, rf_tax_panic, rf_wc_double_squeeze.
FORENSIC_MAX_FLAGS = 28
FORENSIC = {
    "cfo_pat_alert":              70.0,   # below this = Level 1 red flag (percentage, e.g. 73.04 = 73%)
    # FORENSIC CFO/EBITDA floor — recalibrated 90 → 50 (2026-06-12 Schilit audit).
    # CFO is AFTER tax + interest + working capital; EBITDA is BEFORE all three, so
    # ~75% is mathematical PAR for a clean 25%-tax company (live universe median: 68.8%).
    # The old 90% line was Mukherjea's Coffee Can ELITE-QUALITY gate (it stays in
    # fw_coffee_can, untouched) — as a RED FLAG it fired for 54% of the universe,
    # counting ordinary tax mathematics toward red_flag_count and the cascading
    # multiplier. Schilit's actual cash detections are CFFO vs Net Income
    # (rf_low_cfo_pat < 70%) and FCF/EBITDA < 0.3 (rf_low_fcf_ebitda) — both exact.
    # 50% = conversion clearly below what taxes alone explain → genuine anomaly only.
    "cfo_ebitda_clean_threshold": 50.0,  # percentage threshold; rf_low_cfo_ebitda fires below this
    "receivable_rise_days":  15,    # DSO rising more than this = flag
    "inventory_vs_revenue":  True,  # inv growth > rev growth = flag
    "capex_depr_ratio_max":  3.0,   # capex/depr > 3 without rev jump
    "pledge_watch":          10.0,  # above this = watch
    "pledge_critical":       20.0,  # above this = critical
    "expense_ratio_rising":  True,  # rising expense ratio = flag
}

# Piotroski F-Score thresholds
PIOTROSKI = {
    "strong": 7,    # F-Score ≥ 7 = strong
    "moderate": 5,  # F-Score 5-6 = moderate
    "weak": 0,      # F-Score ≤ 4 = weak
}

# ═══════════════════════════════════════════════════════════════
# 9. RSI ZONE SCORING
# ═══════════════════════════════════════════════════════════════
RSI_ZONES = {
    "overbought":   {"min": 80, "max": 100, "score": 10},
    "strong_trend":  {"min": 70, "max": 80,  "score": 60},
    "sweet_spot":    {"min": 55, "max": 70,  "score": 100},
    "neutral":       {"min": 45, "max": 55,  "score": 50},
    "weak":          {"min": 30, "max": 45,  "score": 20},
    "oversold":      {"min": 0,  "max": 30,  "score": 40},  # mean-reversion potential
}

# ═══════════════════════════════════════════════════════════════
# 9b. 52WH AGE ZONE SCORING
# ═══════════════════════════════════════════════════════════════
# Scores the FRESHNESS of the 52-week high (how many days ago it was set).
# Pairs with 52wh_distance (magnitude): both needed for complete breakout picture.
# A stock 5% from its 52WH set yesterday is completely different from
# one 5% from its 52WH set 280 days ago (stale overhead supply).
HIGH_AGE_ZONES = {
    "breakout_fresh":  {"min": 0,   "max": 31,   "score": 100},  # 0-30d: just made/near new high
    "consolidating":   {"min": 31,  "max": 91,   "score": 75},   # 1-3m: classic base/flag setup
    "building":        {"min": 91,  "max": 181,  "score": 45},   # 3-6m: overhead supply forming
    "stale":           {"min": 181, "max": 253,  "score": 20},   # 6-12m: high becoming resistance
    "very_stale":      {"min": 253, "max": 9999, "score": 5},    # 1y+: failed breakout or downtrend
}
# NaN → default 50 (neutral) — missing data never penalises

# ═══════════════════════════════════════════════════════════════
# 10. UI CONFIGURATION
# ═══════════════════════════════════════════════════════════════
UI = {
    "app_title": "PRISM",
    "app_icon": "🔷",
    "app_subtitle": "Every lens. One verdict.",
    "version": "1.0.0",
    "max_display_default": 100,
    "font_url": "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap",
}

# Color palette
COLORS = {
    "bg_primary":    "#0d1117",
    "bg_secondary":  "#161b22",
    "bg_tertiary":   "#21262d",
    "border":        "#30363d",
    "border_hover":  "#484f58",
    "text_primary":  "#e6edf3",
    "text_secondary":"#8b949e",
    "text_muted":    "#6e7681",
    "gold":          "#e3b341",
    "green":         "#3fb950",
    "blue":          "#58a6ff",
    "red":           "#f85149",
    "purple":        "#8b5cf6",
    "orange":        "#FF6B35",
    "cyan":          "#00CED1",
    "gradient_start":"#1a1a2e",
    "gradient_mid":  "#16213e",
    "gradient_end":  "#0f3460",
}

# Tier-specific colors for the conviction table
TIER_COLORS = {
    1: {"bg": "rgba(255,215,0,0.06)",  "border": "rgba(255,215,0,0.3)",  "text": "#FFD700"},
    2: {"bg": "rgba(63,185,80,0.06)",  "border": "rgba(63,185,80,0.3)",  "text": "#3fb950"},
    3: {"bg": "rgba(88,166,255,0.06)", "border": "rgba(88,166,255,0.3)", "text": "#58a6ff"},
    4: {"bg": "rgba(210,153,34,0.06)", "border": "rgba(210,153,34,0.3)", "text": "#d29922"},
    5: {"bg": "rgba(248,81,73,0.06)",  "border": "rgba(248,81,73,0.3)",  "text": "#f85149"},
}

# Framework taxonomy — the 5 §7 (CLAUDE.md) groups. Used to categorize the 37 frameworks in the UI:
# compact category COUNTS on cards, grouped chips on the tearsheet. Reveals a stock's conviction
# CHARACTER (quality-moat vs momentum vs value play) instead of a flat undifferentiated pill list.
# Names MUST match _FW_META / frameworks_passed exactly (the §7 zero-duplicate contract;
# the 36 live names were verified against frameworks_passed 2026-06-14; "Blue Chip Quality" is the
# 37th — currently DPR-dead so it never appears, but it is mapped for when the source column is fixed).
FRAMEWORK_CATEGORIES = [
    ("🏛️", "MOSL",     COLORS["blue"], [
        "QGLP", "MOSL Wealth Creator", "SQGLP Century Stock", "100x Candidate",
        "Fallen Quality", "CAP-GAP Compounder", "Economic Moat", "Blue Chip Quality",
        "Consistent in Volatile", "EP Hockey Stick", "Bruised Blue Chip 29", "Multi-Trillion Cap",
    ]),
    ("📚", "Moats",    COLORS["green"], [
        "Coffee Can", "Diamond", "Peaceful Investing", "Unusual Billionaires",
        "Long Game Quality", "Baid Compounder", "Basant 30% Club", "Quality Compounder",
    ]),
    ("⚡", "Momentum", COLORS["orange"], [
        "CAN SLIM", "SEPA Momentum", "Quality Momentum", "Lynch Dream", "EP Improver", "SMILE",
    ]),
    ("🛡️", "Value",    COLORS["gold"], [
        "Magic Formula", "Dhandho Asymmetry", "Parikh Contrarian", "Wide Moat",
        "Outsider CEO", "Expectations Matrix", "Financial Shenanigans", "Marks Cycle Shield",
    ]),
    ("🎣", "Fisher",   COLORS["cyan"], [
        "Fisher Quality", "Fisher Scalability", "100-Bagger",
    ]),
]

# Reverse lookup: framework name → (emoji, short label, color).
FRAMEWORK_TO_CATEGORY = {
    fw: (emoji, label, color)
    for emoji, label, color, fws in FRAMEWORK_CATEGORIES
    for fw in fws
}
