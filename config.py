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
    # Case-insensitive resolution for Linux (Streamlit Cloud)
    try:
        # Find folder
        actual_folder = folder_name
        for item in os.listdir(base):
            if item.lower() == folder_name.lower() and os.path.isdir(os.path.join(base, item)):
                actual_folder = item
                break
        
        folder_path = os.path.join(base, actual_folder)
        
        # Find file
        actual_file = file_name
        for item in os.listdir(folder_path):
            if item.lower() == file_name.lower():
                actual_file = item
                break
                
        return os.path.join(folder_path, actual_file)
    except Exception:
        return os.path.join(base, folder_name, file_name)

DATA_DIR_NAME = "CSV Data"

CSV_FILES = {
    "ratio":         _get_actual_path(BASE_DIR, DATA_DIR_NAME, "Stockscan - Ratio.csv"),
    "income":        _get_actual_path(BASE_DIR, DATA_DIR_NAME, "Stockscan - Income Statement.csv"),
    "balance":       _get_actual_path(BASE_DIR, DATA_DIR_NAME, "Stockscan - Balance Sheet.csv"),
    "cashflow":      _get_actual_path(BASE_DIR, DATA_DIR_NAME, "Stockscan - Cashflow.csv"),
    "shareholding":  _get_actual_path(BASE_DIR, DATA_DIR_NAME, "Stockscan - Shareholdings.csv"),
    "technical":     _get_actual_path(BASE_DIR, DATA_DIR_NAME, "Stockscan - Technicals.csv"),
}

# Google Sheets Configuration
DEFAULT_SHEET_ID = ""
DEFAULT_GIDS = {
    "ratio": "1823439984",
    "income": "1179123585",
    "balance": "492995744",
    "cashflow": "458676223",
    "shareholding": "1334428374",
    "technical": "1818626554"
}

# ═══════════════════════════════════════════════════════════════
# 2. MARKET CAP TIERS  (₹ Crores)
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
        "threshold": 0.7,
        "source": "Coffee Can Clean Accounts",
        "description": "CFO/PAT ≥ 0.7 — earnings are real cash",
    },
    "trend_filter": {
        "column": "above_sma200",  # derived: close_price > sma_200d
        "operator": "==",
        "threshold": 1,
        "source": "CAN-SLIM Market Direction",
        "description": "Price > SMA 200D — long-term uptrend only",
    },
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
}

# Financial sector stocks get a separate gate set
FINANCIAL_SECTORS = [
    "Banking", "NBFC", "Insurance", "Financial Services",
    "Banks - Private Sector", "Banks - Public Sector",
    "Finance - NBFC", "Finance - Housing Finance",
    "Life Insurance", "General Insurance",
]

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
    "pat_gr_5y":        0.20,
    "pat_gr_10y":       0.10,
    "rev_gr_5y":        0.20,
    "rev_gr_10y":       0.10,
    "eps_gr_5y":        0.15,
    "ebitda_gr_5y":     0.10,
    "pat_acceleration": 0.08,  # pat_gr_3y - pat_gr_5y
    "rev_acceleration": 0.07,  # rev_gr_3y - rev_gr_5y
}

CASH_SIGNALS = {
    "cfo_to_pat":       0.25,
    "cfo_to_ebitda":    0.15,
    "fcf_yield":        0.20,
    "fcf_consistency":  0.15,
    "capex_coverage":   0.10,
    "self_funding":     0.15,
}

MARGIN_SIGNALS = {
    "npm_med_5y":       0.30,
    "opm_med_5y":       0.25,
    "gpm_med_5y":       0.15,
    "npm_acceleration": 0.15,  # npm_lq - npm_1yb
    "opm_acceleration": 0.15,  # opm_lq - opm_1yb
}

BALANCE_SHEET_SIGNALS = {
    "net_debt_negative":  0.25,  # negative net_debt = fortress
    "debt_slope_3y":      0.20,  # negative = deleveraging
    "reserves_growth":    0.20,
    "cwip_conversion":    0.15,  # positive = capacity coming online
    "cash_change":        0.20,  # positive = building cash
}

# ═══════════════════════════════════════════════════════════════
# 4b. VALUATION SCORE SIGNALS (Marks + Baid Entry Price Discipline)
# ═══════════════════════════════════════════════════════════════
VALUATION_SIGNALS = {
    "pe_discount":     0.25,   # PE vs 10Y median — higher discount = better
    "peg_ratio":       0.30,   # PEG < 1.0 = cheap, > 2.5 = expensive
    "ev_compression":  0.15,   # EV/EBITDA falling = value creation
    "fcf_yield_val":   0.20,   # FCF Yield > 3% = attractive
    "de_fortress":     0.10,   # D/E < 0.5 (Baid's fortress gate) = bonus
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
    "vstop_green":      0.30,   # VSTOP 14W 2.5 = green
    "vstop_fresh":      0.25,   # last change ≤ 30 days
    "adx_strong":       0.20,   # ADX 14W > 25
    "rsi_zone":         0.15,   # RSI 55-70 sweet spot
    "golden_cross":     0.10,   # recent golden cross
}

BREAKOUT_SIGNALS = {
    "52wh_distance":    0.30,
    "13wh_distance":    0.25,
    "breakout_window":  0.25,
    "ath_distance":     0.20,
}

SECTOR_SIGNALS = {
    "ret_vs_industry_1y":  0.55,
    "ret_vs_industry_3m":  0.45,
}

# ═══════════════════════════════════════════════════════════════
# 6. COMPOSITE SCORE BLEND (Layer 4)
# ═══════════════════════════════════════════════════════════════
COMPOSITE_WEIGHTS = {
    "quality":    0.55,
    "momentum":   0.30,
    "governance": 0.15,
}

# Governance bonus components (0–100 total possible)
GOVERNANCE_BONUS = {
    "promoter_buying":         20,  # promoter increased holding this Q
    "fii_accumulating":        15,  # FII buying this Q
    "dii_accumulating":        10,  # DII buying this Q
    "inst_convergence":        15,  # FII + DII both buying same Q
    "insider_trading_present": 15,  # directors buying
    "pledge_falling_1y":       10,  # pledge reduced over 1 year
    "undiscovered_alpha":      15,  # low FII + Tier C mcap
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
        "description": "CFO/PAT dropped below 0.5 (was above 0.7)",
        "check": "cfo_to_pat < 0.5",
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
        "quality_w": 0.20, "growth_w": 0.25, "longevity_w": 0.10, "price_w": 0.15,
        "roce_gate": 12.0, "growth_gate": 15.0, "peg_gate": 3.0,
        "forensic_boost": 0.7,
        "priority_cols": ["crs_50d", "ret_vs_n500_3m", "momentum_score", "rsi_14d", "dist_52wh"],
    },
    "Turnaround": {
        "label": "Turnaround / Special Situation",    "icon": "🔄",
        "description": "QoQ acceleration + promoter buying + volume surge. High risk, high reward",
        "quality_w": 0.20, "growth_w": 0.40, "longevity_w": 0.10, "price_w": 0.10,
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
# 8. FORENSIC ENGINE THRESHOLDS
# ═══════════════════════════════════════════════════════════════
FORENSIC = {
    "cfo_pat_alert":         0.7,   # below this = Level 1 red flag
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
# 10. UI CONFIGURATION
# ═══════════════════════════════════════════════════════════════
UI = {
    "app_title": "Multibagger Discovery System",
    "app_icon": "🏆",
    "app_subtitle": "Quantamental Compounding Engine",
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
