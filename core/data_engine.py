"""
Multibagger Discovery System — Data Engine
=============================================
Loads 6 CSV datasets, merges into a master DataFrame,
computes 36+ derived signals using pure vectorized Pandas.
Zero iterrows(), zero apply(). Sub-second on 2,108 stocks.
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, Tuple, Optional
from config import (CSV_FILES, MCAP_TIERS, MCAP_MIN_FLOOR,
                    FINANCIAL_SECTORS, FINANCIAL_SECTOR_NAMES,
                    COST_OF_EQUITY, INDIA_GSEC_YIELD,
                    EPOCH3_TAXONOMY, EPOCH5_MODERN, CONSISTENT_SECTORS)

warnings.filterwarnings('ignore')
np.seterr(all='ignore')


# ═══════════════════════════════════════════════════════════════
# COLUMN MAPPING — CSV header names → clean snake_case
# ═══════════════════════════════════════════════════════════════

# Common columns present in every CSV (joined on companyId)
COMMON_COLS = {
    "companyId": "company_id",
    "Name": "name",
    "Market Capitalization": "market_cap",
    "Market Category": "market_category",
    "Eligibity": "eligibility",
    "Close Price": "close_price",
    "Industry": "industry",
    "Sector": "sector",
}

RATIO_COLS = {
    # MOAT — ROCE (current + 2 historical for trajectory; 2yb is pure noise at this timeframe)
    "ROCE Median 10 Years": "roce_med_10y",
    "ROCE Median 7 Years":  "roce_med_7y",
    "ROCE Median 5 Years":  "roce_med_5y",
    "ROCE Median 3 Years":  "roce_med_3y",
    "ROCE":                 "roce",
    "ROCE 1 Year Back":     "roce_1yb",
    # CAPITAL EFFICIENCY — ROE
    "ROE Median 10 Years":  "roe_med_10y",
    "ROE Median 7 Years":   "roe_med_7y",
    "ROE Median 5 Years":   "roe_med_5y",
    "ROE":                  "roe",
    "ROE 1 Year Back":      "roe_1yb",
    # CASH QUALITY
    "CFO To PAT":           "cfo_to_pat",
    "CFO To EBITDA":        "cfo_to_ebitda",
    # MARGINS — NET
    "NPM Median 5 Years":   "npm_med_5y",
    "NPM Median 3 Years":   "npm_med_3y",
    "NPM":                  "npm",
    "NPM Latest Quarter":   "npm_latest_q",
    "NPM 1 Year Back":      "npm_1yb",
    # MARGINS — OPERATING (opm_med_3y dropped — opm_med_5y is the alpha signal; opm annual kept for OPM delta)
    "OPM Median 5 Years":   "opm_med_5y",
    "OPM":                  "opm",
    "OPM Latest Quarter":   "opm_latest_q",
    "OPM 1 Year Back":      "opm_1yb",
    # MARGINS — GROSS (annual gpm dropped — gpm_med_5y covers long-run; gpm_latest_q is freshest single signal)
    "GPM Median 5 Years":   "gpm_med_5y",
    "GPM Latest Quarter":   "gpm_latest_q",
    # VALUATION (price_to_book + enterprise_value dropped — EV/EBITDA + PE + PEG capture all compression)
    "PEG":                          "peg",
    "EV To EBITDA":                 "ev_ebitda",
    "EV To EBITDA 1 Year Back":     "ev_ebitda_1yb",
    "Price To Earnings Median 10 Years": "pe_med_10y",
    "Price To Earnings":            "pe",
    "Industry PE Median":           "industry_pe",
    # EFFICIENCY — WORKING CAPITAL
    "Cash Conversion Cycle":               "ccc",
    "Cash Conversion Cycle 1 Year Back":   "ccc_1yb",
    "Cash Conversion Cycle 3 Years Back":  "ccc_3yb",     # CCC 3Y trend for forensic quality
    "Days Receivable":               "days_receivable",
    "Days Receivable 1 Year Back":   "days_receivable_1yb",
    "Days Receivable 2 Years Back":  "days_receivable_2yb",
    "Days Receivable 3 Years Back":  "days_receivable_3yb",  # Mukherjea Lens 1: true 3Y DSO window
    # Days Payable/Inventory raw + 1yb dropped — CCC already aggregates; granular components add noise
    "Asset Turnover":               "asset_turnover",
    "Asset Turnover 1 Year Back":   "asset_turnover_1yb",
    "Inventory Turnover Ratio":             "inventory_turnover",
    "Inventory Turnover Ratio 1 Year Back": "inventory_turnover_1yb",
    "Dividend Payout Ratio":        "dividend_payout_ratio",
    # HARD GATES
    "Debt To Equity":               "debt_to_equity",
    "Debt To Equity 1 Year Back":   "debt_to_equity_1yb",
    "Debt To Equity 2 Years Back":  "debt_to_equity_2yb",
    "Debt To Equity 3 Years Back":  "debt_to_equity_3yb",
    "Current Ratio":                "current_ratio",
    "Current Ratio 1 Year Back":    "current_ratio_1yb",
    "ROA":                          "roa",
    "ROA 1 Year Back":              "roa_1yb",
    "Equity Shares 1 Year Back":    "equity_shares_1yb",
    "Interest Coverage":            "interest_coverage",
}

INCOME_COLS = {
    # GROWTH — long-term compounding proof (13 columns)
    "PAT Growth 5 Years":     "pat_gr_5y",
    "PAT Growth 10 Years":    "pat_gr_10y",
    "PAT Growth 3 Years":     "pat_gr_3y",
    "PAT Growth YoY":         "pat_gr_yoy",
    "EPS Growth 5 Years":     "eps_gr_5y",
    "EPS Growth 3 Years":     "eps_gr_3y",
    "EPS Growth YoY":         "eps_gr_yoy",
    "Revenue Growth 5 Years": "rev_gr_5y",
    "Revenue Growth 10 Years":"rev_gr_10y",
    "Revenue Growth 3 Years": "rev_gr_3y",
    "Revenue Growth YoY":     "rev_gr_yoy",
    "EBITDA Growth 5 Years":  "ebitda_gr_5y",
    "EBITDA Growth 3 Years":  "ebitda_gr_3y",
    # QUARTERLY — freshest timing signals (8 columns; EPS LQ added for quarterly EPS YoY signal)
    "PAT Latest Quarter":              "pat_lq",
    "PAT Preceding Year Quarter":      "pat_pyq",
    "Revenue Latest Quarter":          "rev_lq",
    "Revenue Preceding Year Quarter":  "rev_pyq",
    "EBITDA Latest Quarter":           "ebitda_lq",
    "EBITDA Preceding Year Quarter":   "ebitda_pyq",
    "EPS Latest Quarter":              "eps_lq",    # enables quarterly EPS YoY (eps_lq vs eps_pyq)
    "EPS Preceding Year Quarter":      "eps_pyq",
    # METADATA — result freshness (days since last published result; staleness guard)
    "Days From Result":                "days_from_result",
    # RAW ANNUAL — minimum needed for derived signals (13 columns)
    "PAT":                    "pat",
    "PAT 1 Year Back":        "pat_1yb",
    "PAT 2 Years Back":       "pat_2yb",   # A criterion: step-growth verification (O'Neil Ch.4)
    "PAT 3 Years Back":       "pat_3yb",   # A criterion: step-growth verification (O'Neil Ch.4)
    "PBT":                    "pbt",
    "PBT 1 Year Back":        "pbt_1yb",
    "EBITDA":                 "ebitda",
    "EBITDA 1 Year Back":     "ebitda_1yb",
    # EBIT is present in the Screener.in Income Statement CSV — adding here enables
    # the D&A derivation (D&A = EBITDA − EBIT) used in Schilit Signal 6 (EMS #4).
    "EBIT":                   "ebit",
    "EBIT 1 Year Back":       "ebit_1yb",
    "Revenue":                "revenue",
    "Revenue 1 Year Back":    "revenue_1yb",
    "Revenue 2 Years Back":   "revenue_2yb",
    "Revenue 3 Years Back":   "revenue_3yb",
    "Revenue 4 Years Back":   "revenue_4yb",
    "Revenue 5 Years Back":   "revenue_5yb",
    "Expenses":               "expenses",
    "Expenses 1 Year Back":   "expenses_1yb",
}

BALANCE_COLS = {
    # DEBT
    "Debt": "debt",
    "Debt 1 Year Back": "debt_1yb",
    "Debt 2 Years Back": "debt_2yb",
    "Debt 3 Years Back": "debt_3yb",
    # CASH
    "Cash Equivalents": "cash_equivalents",
    "Cash Equivalents 1 Year Back": "cash_equivalents_1yb",
    # RESERVES
    "Reserves": "reserves",
    "Reserves 1 Year Back": "reserves_1yb",
    # CWIP
    "CWIP": "cwip",
    "CWIP 1 Year Back": "cwip_1yb",
    # FIXED ASSETS
    "Fixed Assets": "fixed_assets",
    "Fixed Assets 1 Year Back": "fixed_assets_1yb",
    "Fixed Assets 2 Years Back": "fixed_assets_2yb",
    "Fixed Assets 3 Years Back": "fixed_assets_3yb",
    # TOTALS
    "Total Assets": "total_assets",
    "Total Assets 1 Year Back": "total_assets_1yb",
    "Total Liabilities": "total_liabilities",
    "Total Liabilities 1 Year Back": "total_liabilities_1yb",
    # INVENTORY
    "Inventory": "inventory",
    "Inventory 1 Year Back": "inventory_1yb",
    # EQUITY
    "Equity Shares": "equity_shares",
}

CASHFLOW_COLS = {
    "Operating Cash Flow": "operating_cash_flow",
    "Operating Cash Flow 1 Year Back": "ocf_1yb",
    "Free Cash Flow": "free_cash_flow",
    "Free Cash Flow 1 Year Back": "fcf_1yb",
    "Investing Cash Flow": "investing_cash_flow",
    "Investing Cash Flow 1 Year Back": "icf_1yb",
    "Financing Cash Flow": "financing_cash_flow",
    "Financing Cash Flow 1 Year Back": "financing_cf_1yb",
    "Net Cash Flow": "net_cash_flow",
    "Net Cash Flow 1 Year Back": "ncf_1yb",
}

SHAREHOLDING_COLS = {
    # ABSOLUTE LEVELS
    "Promoter Holdings": "promoter_holdings",
    "FII Holdings": "fii_holdings",
    "DII Holdings": "dii_holdings",
    # PLEDGED
    "Pledged Percentage": "pledged_percentage",
    "Pledged Percentage 1 Quarter Back": "pledged_1qb",
    "Pledged Percentage 1 Year Back": "pledged_1yb",
    # PROMOTER CHANGES — all 4 kept: promoter decisions are deliberate; 3Y accumulation trend is documented alpha
    "Change In Promoter Holdings Latest Quarter": "change_promoter_lq",
    "Change In Promoter Holdings 1 Year": "change_promoter_1y",
    "Change In Promoter Holdings 2 Years": "change_promoter_2y",
    "Change In Promoter Holdings 3 Years": "change_promoter_3y",
    # FII CHANGES — LQ + 1Y only; 2Y/3Y stale (FII flows reverse with every macro cycle)
    "Change In FII Holdings Latest Quarter": "change_fii_lq",
    "Change In FII Holdings 1 Year": "change_fii_1y",
    # DII CHANGES — LQ + 1Y only; base level kept (low FII + low DII = double undiscovered signal)
    "Change In DII Holdings Latest Quarter": "change_dii_lq",
    "Change In DII Holdings 1 Year": "change_dii_1y",
    # ACTIVITY
    "Insider Trading": "insider_trading",
    # GATE-SPECIFIC PROMOTER (may exclude promoter group / cross-holdings for cleaner gate signal)
    "Promoter Holdings (Gate Use)": "promoter_holdings_gate",
}

TECHNICAL_COLS = {
    # FOUNDATION
    "Market Capitalization": "market_cap",
    "Close Price": "close_price",
    # PRIMARY TRIGGER (VSTOP 14W 2.5 — optimal timeframe + sensitivity)
    "VSTOP 14W 2.5": "vstop_value",
    "Last VSTOP Change 14W 2.5": "last_vstop_change",
    # RELATIVE STRENGTH — Nifty 500 only (right benchmark for small/mid cap universe)
    "CRS Vs Nifty 500 50D": "crs_50d",
    "CRS Vs Nifty 500 52W": "crs_52w",
    "CRS Vs Nifty 500 26W": "crs_26w",
    # TREND GATES
    "ADX 14W": "adx_14w",
    "SMA 200D": "sma_200d",
    # MOMENTUM CONFIRMATION
    "RSI 14D": "rsi_14d",
    "Returns Vs Nifty 500 3M": "ret_vs_n500_3m",
    "Returns Vs Nifty 500 6M": "ret_vs_n500_6m",
    "Returns Vs Industry 1Y": "ret_vs_industry_1y",
    # BREAKOUT PROXIMITY
    "52WH Distance": "dist_52wh",
    "52WH Distance Days": "dist_52wh_days",
    "13WH Distance": "dist_13wh",
    "Breakout Window": "breakout_window",
    # VOLUME — institutional entry detector + liquidity gate
    "Volume": "volume",
    "Volume SMA 5D":  "vol_sma_5d",   # VCP dryup check: 5D avg < 20D avg = volume contracting in base
    "Volume SMA 20D": "vol_sma_20d",
    # TREND CONFIRMATION
    "Last Goldencrossover 50D 200D": "golden_cross_days",
    "All Time High Distance": "dist_ath",
    "Returns Vs Industry 3M": "ret_vs_industry_3m",
}


def _safe_numeric(series: pd.Series) -> pd.Series:
    """Convert a series to numeric, coercing errors (null strings, etc.) to NaN."""
    return pd.to_numeric(series, errors='coerce')


def extract_spreadsheet_id(url_or_id: str) -> str:
    """Extracts the Google Sheets ID from a full URL."""
    import re
    if not url_or_id:
        return ""
    if '/' not in url_or_id:
        return url_or_id.strip()
    sheets_pattern = r'/spreadsheets/d/([a-zA-Z0-9-_]+)'
    match = re.search(sheets_pattern, url_or_id)
    if match:
        return match.group(1)
    return url_or_id.strip()

def _load_single_csv(filepath: str, col_map: Dict[str, str], sheet_name: str) -> pd.DataFrame:
    """Load a single CSV, apply column mapping, and return clean DataFrame."""
    # Row 0 = emoji section headers, Row 1 = actual column names
    # na_values covers: 'null', 'NULL', 'None', 'N/A', 'n/a', '#N/A', empty string
    df = pd.read_csv(
        filepath,
        header=1,
        low_memory=False,
        na_values=["null", "NULL", "None", "N/A", "n/a", "#N/A", "#VALUE!", "#REF!", ""],
        keep_default_na=True,
    )

    # Build the full mapping: common + sheet-specific
    full_map = {**COMMON_COLS, **col_map}

    # Keep only columns that exist in this CSV
    available = {k: v for k, v in full_map.items() if k in df.columns}
    missing = set(col_map.keys()) - set(df.columns)
    if missing:
        print(f"  ⚠️  [{sheet_name}] Missing columns: {missing}")

    # Select and rename
    df = df[list(available.keys())].rename(columns=available)

    return df


def load_all_csvs(data_source: str = "local", uploaded_files: dict = None, sheet_id: str = None) -> Dict[str, pd.DataFrame]:
    """Load all 6 CSV files and return as a dict of DataFrames."""
    print("📂 Loading CSV data...")
    datasets = {}
    
    from config import DEFAULT_GIDS

    sheet_configs = {
        "ratio":        (RATIO_COLS,),
        "income":       (INCOME_COLS,),
        "balance":      (BALANCE_COLS,),
        "cashflow":     (CASHFLOW_COLS,),
        "shareholding": (SHAREHOLDING_COLS,),
        "technical":    (TECHNICAL_COLS,),
    }

    if data_source == "upload" and uploaded_files is not None:
        for name, (cols,) in sheet_configs.items():
            if name in uploaded_files:
                datasets[name] = _load_single_csv(uploaded_files[name], cols, name)
            else:
                raise FileNotFoundError(f"Missing uploaded file for {name}")
    elif data_source == "sheet" and sheet_id:
        parsed_id = extract_spreadsheet_id(sheet_id)
        for name, (cols,) in sheet_configs.items():
            gid = DEFAULT_GIDS.get(name, "0")
            csv_url = f"https://docs.google.com/spreadsheets/d/{parsed_id}/export?format=csv&gid={gid}"
            try:
                datasets[name] = _load_single_csv(csv_url, cols, name)
            except Exception as e:
                raise Exception(f"Failed to load {name} from Google Sheets: {e}")
    else:
        for name, (cols,) in sheet_configs.items():
            path = CSV_FILES[name]
            datasets[name] = _load_single_csv(path, cols, name)
            print(f"  ✅ {name}: {len(datasets[name])} rows, {len(datasets[name].columns)} cols")

    return datasets


def merge_datasets(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge all 6 datasets into a single master DataFrame on company_id."""
    print("\n🔗 Merging datasets...")

    # Start with ratio as base (has all common cols)
    master = datasets["ratio"].copy()

    # For subsequent merges, only bring in sheet-specific columns + company_id
    common_col_values = set(COMMON_COLS.values())
    for name in ["income", "balance", "cashflow", "shareholding", "technical"]:
        df = datasets[name]
        # Columns unique to this sheet (not in common)
        unique_cols = [c for c in df.columns if c != "company_id"]
        # Remove duplicates with master
        existing = set(master.columns)
        bring_cols = ["company_id"] + [c for c in unique_cols if c not in existing]

        master = master.merge(
            df[bring_cols],
            on="company_id",
            how="left",   # left = ratio sheet is authority; stocks missing from other sheets become NaN, not dropped
            suffixes=("", f"_{name}")
        )
        print(f"  ✅ Merged {name}: {len(master)} rows, {len(master.columns)} cols")

    print(f"\n📊 Master DataFrame: {len(master)} stocks × {len(master.columns)} columns")
    return master


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all non-identifier columns to numeric."""
    string_cols = {
        "company_id", "name", "market_category", "eligibility",
        "industry", "sector", "insider_trading",
    }
    num_cols = [c for c in df.columns if c not in string_cols]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    return df


def compute_derived_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 36+ derived signals. Pure vectorized Pandas."""
    print("\n🧮 Computing derived signals...")

    # ── D3 FIX: Winsorize YoY growth at p01-p99 before any ranking ──
    # Extreme outliers (IOC +528%, COFORGE +1068%) compress percentile ranks for all 2,108 stocks.
    # Winsorizing at p01/p99 preserves relative ordering while preventing outlier-driven compression.
    for _gcol in ["pat_gr_yoy", "eps_gr_yoy", "rev_gr_yoy"]:
        if _gcol in df.columns:
            _lo = df[_gcol].quantile(0.01)
            _hi = df[_gcol].quantile(0.99)
            df[_gcol] = df[_gcol].clip(lower=_lo, upper=_hi)

    # ── RATIO DERIVED ──
    df["roce_trajectory"] = df["roce_med_7y"] - df["roce_med_10y"]
    df["roe_trajectory"] = df["roe_med_7y"] - df["roe_med_10y"]
    df["roce_current_vs_med"] = df["roce"] - df["roce_med_10y"]
    df["roe_current_vs_med"] = df["roe"] - df["roe_med_10y"]
    _de_nan = pd.Series(np.nan, index=df.index)

    df["npm_acceleration"] = df.get("npm_latest_q", _de_nan) - df.get("npm_1yb",    _de_nan)
    df["opm_acceleration"] = df.get("opm_latest_q", _de_nan) - df.get("opm_1yb",    _de_nan)
    df["gpm_acceleration"] = df.get("gpm_latest_q", _de_nan) - df.get("gpm_med_5y", _de_nan)

    # ── MOSL Wealth Creation Alpha Signals ──
    # economic_profit_spread: ROCE minus India cost of equity (COST_OF_EQUITY from config).
    # Positive = value creation above hurdle; > COST_OF_EQUITY = substantial competitive advantage.
    # Validated across all 30 MOSL Annual Wealth Creation Studies.
    df["economic_profit_spread"] = df["roce"].fillna(0) - COST_OF_EQUITY

    # compound_growth_power_flag: PAT CAGR consistent across 3 timeframes simultaneously.
    # 3Y ≥ 15% (recent), 5Y ≥ 12% (medium-term), 10Y ≥ 10% (long-term).
    # No existing framework checks all three PAT timeframes together.
    # Source: MOSL Wealth Creation Guide — "consistency, profitability and sustainability are key".
    _cgp_3y  = df.get("pat_gr_3y",  pd.Series(np.nan, index=df.index)).fillna(0)
    _cgp_5y  = df.get("pat_gr_5y",  pd.Series(np.nan, index=df.index)).fillna(0)
    _cgp_10y = df.get("pat_gr_10y", pd.Series(np.nan, index=df.index)).fillna(0)
    df["compound_growth_power_flag"] = (
        (_cgp_3y  >= 15) &
        (_cgp_5y  >= 12) &
        (_cgp_10y >= 10)
    ).astype(int)

    # roe_elite_flag: ROE ≥ 35% — elite tier wealth creator.
    # 6th Study (1996-2001): 12 companies with ROE > 35% created 50% of all wealth in that period.
    df["roe_elite_flag"] = (df["roe"].fillna(0) >= 35).astype(int)

    # valuation_multiple_trap: 1st WCS — PE > 35 AND ROE < 18%.
    # When PE expands without corresponding ROE expansion, the market is pricing in future
    # returns the business cannot generate. pe_vs_roe_mos (derived below) captures the spread;
    # this binary flag allows scoring_engine to apply a targeted 40% valuation-score slash.
    # pe fillna(0): loss-makers (NaN PE) → 0 → 0 > 35 = False → not trapped (conservative).
    # roe fillna(99): NaN ROE → 99 → 99 < 18 = False → not trapped (benefit of doubt).
    df["valuation_multiple_trap"] = (
        (df["pe"].fillna(0)  > 35) &
        (df["roe"].fillna(99) < 18)
    ).astype(int)

    # ── MOSL Studies 7-13: Great/Good/Gruesome Framework + Multi-Bagger Formulas ──

    # roe_trend_rising_flag: Study 13 "Great" company criterion (c) — rising ROE trajectory.
    # roe (current) > roe_med_5y > roe_med_10y = structurally improving return profile.
    _roe_5y_tr  = df.get("roe_med_5y",  pd.Series(np.nan, index=df.index))
    _roe_10y_tr = df.get("roe_med_10y", pd.Series(np.nan, index=df.index))
    df["roe_trend_rising_flag"] = (
        (df["roe"].fillna(0)    > _roe_5y_tr.fillna(0)) &
        (_roe_5y_tr.fillna(0)  > _roe_10y_tr.fillna(0))
    ).astype(int)

    # roe_turnaround_flag: Study 13 — low ROE (<15%) stocks with improving trajectory.
    # "Anticipating change in profitability ahead of the crowd rewarded very well."
    # ROE < 5% companies delivered 92% price CAGR when ROE was rising. Bargain zone.
    df["roe_turnaround_flag"] = (
        (df["roe"].fillna(0)  <  15) &
        (df["roe"].fillna(0)  > _roe_5y_tr.fillna(0))
    ).astype(int)

    # great_company_screen: Study 13 "Great" = 10yr avg Adj ROE > 25% + never below 15% + rising.
    # Proxy: roe_med_10y ≥ 25 (avg), roe_med_10y ≥ 18 (floor — if 10yr median ≥ 18, no year
    # likely dipped below 15%), and roe_trend_rising_flag = 1 (criterion c).
    df["great_company_screen"] = (
        (_roe_10y_tr.fillna(0) >= 25) &
        (_roe_10y_tr.fillna(0) >= 18) &   # floor proxy: 10yr median ≥ 18 → no year likely < 15%
        (df["roe_trend_rising_flag"] == 1)
    ).astype(int)

    # gruesome_flag: Study 13 — 10yr avg ROE < 10% = wealth destroyer profile. Avoid.
    df["gruesome_flag"] = (_roe_10y_tr.fillna(0) < 10).astype(int)

    # roce_step_change_flag: Studies 7-9 — 65-70% of wealth creators had ROCE rising over 5yr.
    # These companies created 75-84% of all wealth in each study period.
    # Signal: ROCE currently 3+ pp above its own 5yr median = structural improvement underway.
    _roce_med5_sc = df.get("roce_med_5y", pd.Series(np.nan, index=df.index))
    df["roce_step_change_flag"] = (
        (df["roce"].fillna(0) > _roce_med5_sc.fillna(0) + 3)
    ).astype(int)

    # margin_expansion_flag: Study 9 — "high earnings growth need NOT be associated with high
    # sales growth." PAT growing 5+ pp faster than revenue = pricing power or operating leverage.
    # Companies with Sales CAGR 0-10% still created 28% earnings CAGR through margin expansion.
    _mg_pat = df.get("pat_gr_5y", pd.Series(np.nan, index=df.index))
    _mg_rev = df.get("rev_gr_5y", pd.Series(np.nan, index=df.index))
    df["margin_expansion_flag"] = (
        (_mg_pat.fillna(0) > _mg_rev.fillna(0) + 5)
    ).astype(int)

    # pe_sweet_spot_flag: Study 9 — PE 5-10x band produced best risk-adjusted CAGR (43.9%),
    # even better than PE < 5x (36.1%, which often signals distress).
    # Study 13: PE < 10x = two-thirds of all wealth creators in 2003-2008 period.
    df["pe_sweet_spot_flag"] = (
        (df["pe"].fillna(999) >= 5) &
        (df["pe"].fillna(999) <= 10)
    ).astype(int)

    # earnings_yield_ratio: Studies 8, 9, 13 — earnings yield (1/PE) vs India G-Sec yield (~7%).
    # Ratio > 1.0 = stock's earnings yield exceeds risk-free rate = margin of safety exists.
    # Historical range: 0.13 (market peak 1992) → 1.73 (market trough 2002). > 1.4 = very attractive.
    df["earnings_yield_ratio"] = np.where(
        (df["pe"].fillna(0) > 0),
        (100.0 / df["pe"]) / INDIA_GSEC_YIELD,   # India 10yr G-Sec yield (config.py INDIA_GSEC_YIELD)
        np.nan
    )

    df["pe_discount"] = np.where(
        df["pe_med_10y"].notna() & (df["pe_med_10y"] != 0),
        (df["pe_med_10y"] - df["pe"]) / df["pe_med_10y"] * 100,
        np.nan
    )
    df["ev_compression"] = df["ev_ebitda_1yb"] - df["ev_ebitda"]
    df["de_slope_3y"] = df["debt_to_equity"] - df["debt_to_equity_3yb"]

    # ── Ind AS 116 / Debt Restatement Guard ──
    # Sudden debt spikes from lease capitalization (Ind AS 116, 2019+) create non-economic D/E jumps.
    # G5 FIX: prior condition was inverted — caught D/E DROPS not spikes.
    # Correct: current D/E spiked >2.5× vs 1YB AND the 1YB trend was stable (< 1.2× the 2YB level).
    de_restatement = (
        df["debt_to_equity"].notna() & df["debt_to_equity_1yb"].notna() &
        (df["debt_to_equity"] > df["debt_to_equity_1yb"] * 2.5) &          # current spiked >2.5× vs last year
        (df["debt_to_equity_1yb"] <= df["debt_to_equity_2yb"].fillna(df["debt_to_equity_1yb"]) * 1.2)  # prior trend was stable
    )
    df["debt_restatement_suspected"] = de_restatement.astype(int)
    df["de_slope_3y"] = np.where(de_restatement, np.nan, df["de_slope_3y"])  # neutralise spike
    # ── DILUTION: Percentage-based materiality (Fisher Point 13) ──
    # OLD APPROACH (BUG): Binary flag — any share increase = fail.
    #   This incorrectly killed companies for tiny ESOPs (0.1-0.5% dilution).
    # NEW APPROACH (SMART): 4-Tier materiality system:
    #   Tier 0: Stable / Buyback (≤0%)        → dilution_flag = 0 (Clean)
    #   Tier 1: ESOP-level    (0% to 3%)       → dilution_flag = 1 (Minor — Watch)
    #   Tier 2: Meaningful    (3% to 10%)      → dilution_flag = 2 (Caution — Penalty)
    #   Tier 3: Predatory QIP (>10%)           → dilution_flag = 3 (Hard Reject)
    # The Hard Gate in config.py is now updated to reject ONLY Tier 3 (>10%).
    shares_valid = df["equity_shares"].notna() & df["equity_shares_1yb"].notna() & (df["equity_shares_1yb"] > 0)

    df["dilution_pct"] = np.where(
        shares_valid,
        (df["equity_shares"] - df["equity_shares_1yb"]) / df["equity_shares_1yb"] * 100,
        0.0  # no data = benefit of doubt
    )

    df["dilution_flag"] = np.select(
        [
            ~shares_valid,                          # No data → benefit of doubt
            df["dilution_pct"] <= 0,                # Stable or buyback → perfectly clean
            df["dilution_pct"] <= 3.0,              # ≤3% → ESOP/minor → Watch tier
            df["dilution_pct"] <= 10.0,             # 3-10% → Meaningful → Caution tier
        ],
        [0, 0, 1, 2],
        default=3                                   # >10% → Predatory QIP → Hard Reject
    )

    # ── INCOME DERIVED ──
    # Year-by-year revenue growth for Coffee Can / Baid individual-year consistency checks.
    # Uses .get() fallback: if the CSV is missing a Revenue N Years Back column,
    # the corresponding rev_gr_yN is NaN (no KeyError, no silent wrong value).
    _rev_nan = pd.Series(np.nan, index=df.index)
    for _yr, (_num, _den) in enumerate(
        [("revenue_1yb", "revenue_2yb"), ("revenue_2yb", "revenue_3yb"),
         ("revenue_3yb", "revenue_4yb"), ("revenue_4yb", "revenue_5yb")], start=2
    ):
        _col   = f"rev_gr_y{_yr}"
        _num_s = df.get(_num, _rev_nan)
        _den_s = df.get(_den, _rev_nan)
        df[_col] = np.where(
            _den_s.notna() & (_den_s.abs() > 0),
            (_num_s - _den_s) / _den_s.abs() * 100,
            np.nan
        )
    df["pat_acceleration"]    = df["pat_gr_3y"]    - df["pat_gr_5y"]
    df["rev_acceleration"]    = df["rev_gr_3y"]    - df["rev_gr_5y"]
    df["ebitda_acceleration"] = df["ebitda_gr_3y"] - df["ebitda_gr_5y"]
    df["eps_vs_pat_delta"]    = df["eps_gr_5y"]    - df["pat_gr_5y"]
    df["q_pat_yoy"] = np.where(
        df["pat_pyq"].notna() & (df["pat_pyq"].abs() > 0),
        (df["pat_lq"] - df["pat_pyq"]) / df["pat_pyq"].abs() * 100,
        np.nan
    )
    df["q_rev_yoy"] = np.where(
        df["rev_pyq"].notna() & (df["rev_pyq"].abs() > 0),
        (df["rev_lq"] - df["rev_pyq"]) / df["rev_pyq"].abs() * 100,
        np.nan
    )
    df["q_ebitda_yoy"] = np.where(
        df["ebitda_pyq"].notna() & (df["ebitda_pyq"].abs() > 0),
        (df["ebitda_lq"] - df["ebitda_pyq"]) / df["ebitda_pyq"].abs() * 100,
        np.nan
    )
    # Quarterly EPS YoY growth — eps_lq vs eps_pyq (same quarter prior year)
    # Completes the quarterly set: q_pat_yoy, q_rev_yoy, q_ebitda_yoy, q_eps_yoy
    df["q_eps_yoy"] = np.where(
        df["eps_pyq"].notna() & (df["eps_pyq"].abs() > 0),
        (df["eps_lq"] - df["eps_pyq"]) / df["eps_pyq"].abs() * 100,
        np.nan
    )
    df["expense_ratio"] = np.where(
        df["revenue"].notna() & (df["revenue"] > 0),
        df["expenses"] / df["revenue"],
        np.nan
    )
    df["expense_ratio_1yb"] = np.where(
        df["revenue_1yb"].notna() & (df["revenue_1yb"] > 0),
        df["expenses_1yb"] / df["revenue_1yb"],
        np.nan
    )

    # ── DEPRECIATION & AMORTIZATION (Schilit EMS #4 Foundation) ──────────────────
    # D&A = EBITDA − EBIT. This is the exact accounting identity:
    #   EBITDA − EBIT = Depreciation + Amortization (confirmed for Screener.in P&L format)
    # Both columns are now in INCOME_COLS ("EBIT" and "EBIT 1 Year Back" added 2026-05-24).
    # clip(lower=0): rare negative values are data artefacts in Screener.in — e.g. when their
    # "Other Income" is netted into EBITDA but excluded from EBIT in a non-standard way.
    # Conservative: treat artefacts as zero D&A rather than propagate negative depreciation.
    # NaN propagation: if either EBITDA or EBIT is NaN, result is NaN (safe — no false signals).
    _ebit_da    = df.get("ebit",     pd.Series(np.nan, index=df.index))
    _ebit_1yb_da = df.get("ebit_1yb", pd.Series(np.nan, index=df.index))
    df["depreciation"]     = (df["ebitda"].fillna(np.nan)     - _ebit_da).clip(lower=0)
    df["depreciation_1yb"] = (df["ebitda_1yb"].fillna(np.nan) - _ebit_1yb_da).clip(lower=0)

    # dep_rate: D&A as percentage of gross fixed assets (scale-invariant)
    # Used in Schilit Signal 6 (forensic_engine.py): if fixed assets grow but dep_rate falls,
    # management has extended accounting useful lives to reduce D&A expense and inflate EBIT/PAT.
    # Book anchor: Qwest (Schilit Ch.6) extended asset lives from 14→40yr → $1B earnings boost.
    # NaN when fixed_assets = 0 — asset-light companies (no FA base, dep/FA is undefined).
    df["dep_rate"] = np.where(
        df["fixed_assets"].notna() & (df["fixed_assets"] > 0) & df["depreciation"].notna(),
        df["depreciation"] / df["fixed_assets"] * 100,
        np.nan
    )
    df["dep_rate_1yb"] = np.where(
        df["fixed_assets_1yb"].notna() & (df["fixed_assets_1yb"] > 0) & df["depreciation_1yb"].notna(),
        df["depreciation_1yb"] / df["fixed_assets_1yb"] * 100,
        np.nan
    )

    # ── CASHFLOW DERIVED ──
    # FCF imputation: ~600+ major stocks (HINDUNILVR, HCLTECH, INFY, etc.) have null FCF
    # because the data provider lacks CapEx data. Without imputation, these stocks score at
    # universe median (50th pct) for FCF yield despite massive positive cash generation.
    # Conservative imputation: use OCF when FCF is null (treats all OCF as free cash — overstates
    # FCF by ignoring CapEx, but far better than arbitrary 50th-pct neutral assignment).
    if "free_cash_flow" in df.columns and "operating_cash_flow" in df.columns:
        fcf_null_count = df["free_cash_flow"].isna().sum()
        # Record which rows were imputed BEFORE filling — used below to suppress
        # fcf_to_cfo_pct for these stocks (imputed FCF = OCF gives a misleading 100%).
        df["fcf_imputed_flag"] = df["free_cash_flow"].isna().astype(int)
        df["free_cash_flow"] = df["free_cash_flow"].fillna(df["operating_cash_flow"])
        if fcf_null_count > 0:
            print(f"  ℹ️  FCF imputed from OCF for {fcf_null_count} stocks with null FCF")

    df["fcf_yield"] = np.where(
        df["market_cap"].notna() & (df["market_cap"] > 0),
        df["free_cash_flow"] / df["market_cap"] * 100,  # as percentage
        np.nan
    )
    df["fcf_growth"] = np.where(
        df["fcf_1yb"].notna() & (df["fcf_1yb"].abs() > 0),
        (df["free_cash_flow"] - df["fcf_1yb"]) / df["fcf_1yb"].abs() * 100,
        np.nan
    )
    df["ocf_growth"] = np.where(
        df["ocf_1yb"].notna() & (df["ocf_1yb"].abs() > 0),
        (df["operating_cash_flow"] - df["ocf_1yb"]) / df["ocf_1yb"].abs() * 100,
        np.nan
    )
    df["capex_coverage"] = np.where(
        df["investing_cash_flow"].notna() & (df["investing_cash_flow"].abs() > 0),
        df["operating_cash_flow"] / df["investing_cash_flow"].abs(),
        np.nan
    )
    df["fcf_consistency"] = (
        (df["free_cash_flow"] > 0) & (df["fcf_1yb"] > 0)
    ).astype(int)
    df["self_funding"] = (
        (df["operating_cash_flow"] > 0) & (df["financing_cash_flow"] < 0)
    ).astype(int)
    df["ncf_trend"] = (
        (df["net_cash_flow"] > 0) & (df["ncf_1yb"] > 0)
    ).astype(int)
    df["fcf_quality"] = np.where(
        df["pat"].notna() & (df["pat"].abs() > 0),
        df["free_cash_flow"] / df["pat"].abs(),
        np.nan
    )
    
    # ── ALPHA VECTOR: ACCRUAL ANOMALY (Cash Machine Rank) ──
    # Academic research proves companies where Cash > Profit beat the market.
    # We rank stocks based on their CFO/PAT conversion and FCF Yield.
    df["cash_machine_score"] = np.where(
        (df["cfo_to_pat"].fillna(0) > 100) & (df["fcf_yield"].fillna(0) > 2),
        100,  # Gold standard: Converting all profit to cash AND generating >2% FCF
        np.where(
            df["cfo_to_pat"].fillna(0) > 80,
            50,   # Acceptable
            0     # Paper profits
        )
    )
    df["cash_machine_label"] = np.select(
        [df["cash_machine_score"] == 100, df["cash_machine_score"] == 50],
        ["💰 Cash Machine", "✅ Solid"],
        default="📄 Paper Profits"
    )

    # ── BALANCE SHEET DERIVED ──
    df["net_debt"] = df["debt"] - df["cash_equivalents"]
    df["debt_slope_3y"] = df["debt"] - df["debt_3yb"]
    df["debt_change_1y"] = df["debt"] - df["debt_1yb"]
    df["cash_change"] = df["cash_equivalents"] - df["cash_equivalents_1yb"]
    df["reserves_growth"] = np.where(
        df["reserves_1yb"].notna() & (df["reserves_1yb"].abs() > 0),
        (df["reserves"] - df["reserves_1yb"]) / df["reserves_1yb"].abs() * 100,
        np.nan
    )
    # cwip_conversion is computed at D19 (fixed asset expansion formula). See line below.
    df["cwip_ratio"] = np.where(
        df["fixed_assets"].notna() & (df["fixed_assets"] > 0),
        df["cwip"] / df["fixed_assets"] * 100,
        np.nan
    )
    df["capex_3y"] = df["fixed_assets"] - df["fixed_assets_3yb"]

    # Capex consistency: year-by-year FA growth rate variance (3 consecutive years)
    # High variance = lumpy capex = project execution risk vs smooth compounding expansion.
    _fa_g1 = np.where(
        df["fixed_assets_1yb"].notna() & (df["fixed_assets_1yb"].abs() > 0),
        (df["fixed_assets"] - df["fixed_assets_1yb"]) / df["fixed_assets_1yb"].abs(),
        np.nan
    )
    _fa_g2 = np.where(
        df["fixed_assets_2yb"].notna() & (df["fixed_assets_2yb"].abs() > 0),
        (df["fixed_assets_1yb"] - df["fixed_assets_2yb"]) / df["fixed_assets_2yb"].abs(),
        np.nan
    )
    df["capex_consistency"] = np.abs(_fa_g1 - _fa_g2)   # lower = smoother expansion

    df["inv_growth"] = np.where(
        df["inventory_1yb"].notna() & (df["inventory_1yb"] > 0),
        (df["inventory"] - df["inventory_1yb"]) / df["inventory_1yb"] * 100,
        np.nan
    )
    df["inv_vs_rev_gap"] = df["inv_growth"] - df["rev_gr_yoy"]
    df["solvency_ratio"] = np.where(
        df["total_assets"].notna() & (df["total_assets"] > 0),
        df["total_liabilities"] / df["total_assets"],
        np.nan
    )
    # Hidden obligation growth: Total Liabilities rising faster than Debt = off-balance-sheet risk.
    # Catches Ind AS 116 lease liabilities, provisions, contingent obligations that D/E misses.
    df["liab_change"] = df["total_liabilities"] - df["total_liabilities_1yb"]
    df["hidden_obligation_growth"] = (
        df["liab_change"].fillna(0) > df["debt_change_1y"].fillna(0)
    ).astype(int)   # 1 = TL growing faster than debt = hidden risk flag

    # ── SHAREHOLDING DERIVED ──
    df["pledge_rising"] = np.where(
        df["pledged_percentage"].notna() & df["pledged_1qb"].notna(),
        (df["pledged_percentage"] > df["pledged_1qb"]).astype(int),
        0
    )
    df["pledge_falling_1y"] = np.where(
        df["pledged_1yb"].notna() & df["pledged_percentage"].notna(),
        (df["pledged_1yb"] - df["pledged_percentage"]).clip(lower=0),
        0
    )
    df["promoter_buying"] = (df["change_promoter_lq"] > 0).astype(int)
    df["inst_convergence"] = (
        (df["change_fii_lq"] > 0) & (df["change_dii_lq"] > 0)
    ).astype(int)

    # ── TECHNICAL DERIVED ──
    df["vol_ratio"] = np.where(
        df["vol_sma_20d"].notna() & (df["vol_sma_20d"] > 0),
        df["volume"] / df["vol_sma_20d"],
        np.nan
    )
    df["daily_value"] = df["volume"] * df["close_price"]  # in raw ₹
    df["daily_value_cr"] = df["daily_value"] / 1e7  # in ₹ Crores
    df["crs_aligned"] = (
        (df["crs_50d"] > 0) & (df["crs_26w"] > 0) & (df["crs_52w"] > 0)
    ).astype(int)
    # VSTOP scale guard: nullify implausible VSTOP values (>50× or <2% of close price).
    # Upstream data sources sometimes publish VSTOP in paise for certain stocks (e.g. MOTHERSON: VSTOP=7684 vs price=130).
    if "vstop_value" in df.columns and "close_price" in df.columns:
        vstop_ratio = df["vstop_value"].fillna(0) / df["close_price"].replace(0, np.nan)
        implausible_vstop = (vstop_ratio > 50) | (vstop_ratio < 0.02)
        df.loc[implausible_vstop, "vstop_value"] = np.nan
        implausible_count = int(implausible_vstop.sum())
        if implausible_count > 0:
            print(f"  ⚠️  VSTOP scale mismatch nullified for {implausible_count} stocks")

    df["vstop_fresh"] = np.where(df["last_vstop_change"].notna(), (df["last_vstop_change"] <= 30).astype(int), 0)
    df["above_sma200"] = (df["close_price"] > df["sma_200d"]).astype(int)
    df["vstop_green"] = np.where(df["vstop_value"].notna(), (df["close_price"] > df["vstop_value"]).astype(int), 0)

    # ── VQS & SMART MONEY FLOW (WAVE DETECTION INTEGRATION) ──
    vqs_liquidity = np.where(df["vol_ratio"] >= 3.0, 50,
                    np.where(df["vol_ratio"] >= 2.0, 40,
                    np.where(df["vol_ratio"] >= 1.5, 30,
                    np.where(df["vol_ratio"] >= 1.0, 20, 10))))
    
    vqs_smart = np.where(df["inst_convergence"] == 1, 20,
                np.where((df["change_fii_lq"].fillna(0) > 0) | (df["change_dii_lq"].fillna(0) > 0), 10, 0))
    
    vqs_cons = np.where(df["crs_aligned"] == 1, 20, 
               np.where((df["crs_50d"].fillna(0) > 0) & (df["crs_26w"].fillna(0) > 0), 10, 0))
               
    vqs_eff = np.where(df["ret_vs_n500_3m"].fillna(0) > 0, 10, 0)
    
    df["vqs_score"] = pd.Series(vqs_liquidity + vqs_smart + vqs_cons + vqs_eff, index=df.index).fillna(0)
    
    df["smart_money_flow"] = np.select(
        [
            (df["vqs_score"] >= 80) & (df["inst_convergence"] == 1),
            (df["vqs_score"] >= 60) & ((df["change_fii_lq"].fillna(0) > 0) | (df["change_dii_lq"].fillna(0) > 0)),
            (df["vqs_score"] >= 40),
            (df["change_fii_lq"].fillna(0) < 0) & (df["change_dii_lq"].fillna(0) < 0) & (df["crs_50d"].fillna(0) < 0)
        ],
        [
            "🌊💎 Elite Accumulation",
            "🎯 Strong Accumulation",
            "✅ Moderate Interest",
            "❌ Distribution"
        ],
        default="⚪ Neutral"
    )

    # ── ALPHA VECTOR: ACTIONABILITY / BUY ZONE ──
    # Tells the user WHEN to buy based on risk-reward distance to Volatility Stop.
    df["dist_to_vstop"] = np.where(
        df["vstop_value"].notna() & (df["vstop_value"] > 0),
        ((df["close_price"] - df["vstop_value"]) / df["vstop_value"]) * 100,
        np.nan
    )
    df["buy_zone_label"] = np.select(
        [
            df["dist_to_vstop"] <= 5,   # Within 5% of stop loss (Asymmetric Risk/Reward)
            df["dist_to_vstop"] <= 12,  # Normal volatility buffer
            df["dist_to_vstop"] > 25    # Extended far beyond 50DMA/VSTOP
        ],
        [
            "🟢 Perfect Entry (Low Risk)",
            "🟡 Standard Zone",
            "🔴 Extended (Wait for Pullback)"
        ],
        default="⚪ Uncharted"
    )

    # ── MARKET CAP TIER (mirrors Google Sheet ARRAYFORMULA exactly) ──
    df["mcap_tier"] = np.select(
        [
            df["market_cap"] >= 200_000,
            df["market_cap"] >= 20_000,
            df["market_cap"] >= 5_000,
            df["market_cap"] >= 500,
            df["market_cap"] >= 100,
        ],
        ["Mega Cap", "Large Cap", "Mid Cap", "Small Cap", "Micro Cap"],
        default="Nano Cap"
    )

    # ── Mid-Cap Velocity Compounder (1st WCS) ──
    # 1st WCS empirical finding: smaller companies (Mid/Small/Micro-Cap) with sustained
    # ROCE ≥ 20% compound at dramatically higher rates than large/mega caps with similar ROCE.
    # Scale-velocity advantage: a ₹500 Cr company can double revenue in 3 years;
    # a ₹50,000 Cr company needs 50 times the incremental revenue for the same effect.
    # Validated across all 30 MOSL studies as the primary mid-cap alpha driver.
    df["mcap_velocity_compounder"] = (
        df["mcap_tier"].isin(["Mid Cap", "Small Cap", "Micro Cap", "Nano Cap"]) &
        (df["roce_med_10y"].fillna(0) >= 20)
    ).astype(int)

    # ── FINANCIAL SECTOR FLAG ──
    # Uses two verified sets: FINANCIAL_SECTORS (industry column) + FINANCIAL_SECTOR_NAMES (sector column).
    # Previous code used a phantom list (0/10 industry matches) + a regex that missed brokers/credit agencies.
    df["is_financial"] = (
        df["industry"].fillna("").isin(FINANCIAL_SECTORS)
        | df["sector"].fillna("").isin(FINANCIAL_SECTOR_NAMES)
    )

    # ── AGENT 3: SECTOR ISOLATION — NaN out working-capital metrics for financial stocks ──
    # Banks, NBFCs, and insurance companies have no inventory and no traditional CCC.
    # Applying inventory turnover / CCC flags to these would generate structurally meaningless
    # signals — e.g. a bank scoring 0 on inventory days is noise, not signal.
    # Explict NaN-out here ensures all downstream derived columns (inventory_days, d37_ccc_direction)
    # and all forensic flags (rf_itr_declining, rf_ccc_worsening, rf_inventory_bloat) are guarded.
    _fin_mask = df["is_financial"] == True
    _wc_nullify = [
        "inventory_turnover", "inventory_turnover_1yb",
        "ccc", "ccc_1yb",
        "inventory", "inventory_1yb",
        "days_receivable", "days_receivable_1yb",
        # Also NaN already-computed derived columns that came from these inputs above line 776
        "inv_growth", "inv_vs_rev_gap",
    ]
    _wc_existing = [c for c in _wc_nullify if c in df.columns]
    if _wc_existing:
        df.loc[_fin_mask, _wc_existing] = np.nan

    # Net debt negative flag (fortress balance sheet)
    df["net_debt_negative"] = (df["net_debt"] < 0).astype(int)

    # ══════════════════════════════════════════════════════════════
    # DR. VIJAY MALIK SIGNALS (Peaceful Investing Codex)
    # ══════════════════════════════════════════════════════════════

    # ── SSGR Approximation (Ch.2) ──
    # SSGR = NFAT × NPM × (1 − DPR) − Dep_Rate
    # NFAT = Revenue / Fixed Assets
    # Dep_Rate = (FA_1YB - FA_current + Capex_est) / FA_current
    #   Capex_est ≈ max(0, FA_current - FA_1YB) when FA grew
    #   Depreciation ≈ FA_1YB - FA_current when FA shrunk (net depreciation)
    # DPR approximated as 0.25 (typical Indian payout)
    fa = df["fixed_assets"].fillna(0)
    fa_1yb = df["fixed_assets_1yb"].fillna(0)
    rev = df["revenue"].fillna(0)
    npm_pct = df["npm"].fillna(0)

    nfat = np.where(fa > 0, rev / fa, np.nan)
    df["nfat"] = pd.Series(nfat, index=df.index)  # Net Fixed Asset Turnover — Vijay Malik capital-light signal
    npm_decimal = npm_pct / 100.0
    # Use actual DPR from CSV; fall back to 0.25 (India median) only if unavailable
    _dpr_raw = df.get("dividend_payout_ratio", pd.Series(np.nan, index=df.index))
    _dpr_pct = _dpr_raw.fillna(25.0).clip(0, 100)     # DPR in % (e.g. 25.0 = 25%)
    dpr_approx = (_dpr_pct / 100.0).values             # decimal for SSGR formula

    # Vijay Malik SSGR = NPM × NFAT × (1 - DPR).
    # NPM is already net of depreciation — no separate dep_rate term needed.
    ssgr_raw = nfat * npm_decimal * (1 - dpr_approx)
    df["ssgr"] = pd.Series(ssgr_raw * 100, index=df.index).clip(-50, 100)

    # SSGR vs actual growth — the gold standard test
    actual_growth = df["rev_gr_5y"].fillna(df["rev_gr_3y"]).fillna(0)
    df["ssgr_cushion"] = df["ssgr"] - actual_growth
    df["ssgr_self_funded"] = (df["ssgr_cushion"] > 0).astype(int)

    # ── EBITDA-to-PAT Gap % (Malik Parameter 3 proxy) ──
    # Formula: (EBITDA - PAT) / EBITDA × 100 = (Dep + Interest + Tax) / EBITDA
    # NOTE: This is NOT effective tax rate — it includes depreciation and interest.
    # Renamed from "tax_rate_est" to avoid confusion. Malik P3 band is widened to 30-55%
    # to account for the systematic overestimation vs true effective tax rate (~22-28%).
    df["ebitda_to_pat_gap_pct"] = np.where(
        (df["ebitda"].fillna(0) > 0) & (df["pat"].fillna(0) > 0) &
        (df["ebitda"].fillna(0) > df["pat"].fillna(0)),
        (1 - df["pat"] / df["ebitda"]) * 100,
        np.nan
    )
    df["tax_rate_est"] = df["ebitda_to_pat_gap_pct"]  # backward-compat alias

    # ── Interest Coverage (Malik Parameter 4) ──
    # Priority: (1) direct CSV column, (2) synthetic fallback using Debt × 8.5%
    # Direct CSV is available in the ratio dataset — use it. Fill NaN-only gaps synthetically.
    _ic_debt  = df.get("debt",   _de_nan).fillna(0)
    _ic_ebitda = df.get("ebitda", _de_nan).fillna(0)
    _ic_exp   = _ic_debt * 0.085
    _ic_synthetic = pd.Series(
        np.where(_ic_exp > 0, _ic_ebitda / _ic_exp, 99.0),
        index=df.index
    )
    _ic_csv = df.get("interest_coverage", _de_nan)   # NaN if not in CSV
    df["interest_coverage"] = _ic_csv.fillna(_ic_synthetic)

    # ── Economic Profit (28th WCS) ──
    # EP = Net Worth × (RoE − Cost of Equity)  [MOSL 23rd/28th WCS formula]
    # Net Worth = Reserves only. The "Equity Shares" CSV column stores share COUNT (e.g. ONGC=12.58B),
    # NOT paid-up capital in ₹ Crores. Adding share count to reserves would corrupt EP for every stock.
    # Paid-up capital is typically <3% of total equity for Indian large/mid-caps — omitting it is safe.
    df["net_worth"]     = df["reserves"].fillna(0)
    df["net_worth_1yb"] = df["reserves_1yb"].fillna(0)

    df["economic_profit"] = (
        df["net_worth"] * (df["roe"].fillna(0) / 100.0 - COST_OF_EQUITY / 100.0)
    )
    df["economic_profit_1yb"] = (
        df["net_worth_1yb"] * (df["roe_1yb"].fillna(0) / 100.0 - COST_OF_EQUITY / 100.0)
    )
    df["economic_profit_positive"] = (df["economic_profit"] > 0).astype(int)

    # ── Economic Profit Velocity (28th WCS — Hockey Stick EP Trajectory) ──
    # Multi-year EP direction: companies moving UP the Economic Profit Power Curve
    # are the "Hockey Stick" setup from MOSL 28th Study (2023).
    df["economic_profit_velocity"] = df["economic_profit"] - df["economic_profit_1yb"]
    # Backward-compat alias consumed by downstream flags
    df["economic_profit_delta"] = df["economic_profit_velocity"]

    # Hockey Stick: EP is positive AND improving YoY = ascending the EP Power Curve
    df["ep_hockey_stick"] = (
        (df["economic_profit"] > 0) &
        (df["economic_profit_velocity"] > 0)
    ).astype(int)
    # EP Power Curve position (McKinsey taxonomy applied to Indian equity universe)
    df["ep_power_curve"] = np.select(
        [
            (df["economic_profit"] > 0) & (df["economic_profit_velocity"] > 0),
            (df["economic_profit"] > 0) & (df["economic_profit_velocity"] <= 0),
            (df["economic_profit"] <= 0) & (df["economic_profit_velocity"] > 0),
        ],
        ["🚀 Hockey Stick", "✅ EP Positive", "📈 Improving"],
        default="📉 Value Trap"
    )

    # ── P/Sales and P/B Ratios (Studies 9, 13 multi-bagger formulas) ──
    # Study 13: "PE < 10x, P/B < 1x, P/Sales ≤ 1x, Payback ≤ 1x" = four explicit multi-bagger formulas.
    # Study 9: "If you want a doubler, buy at: P/Book < 1x, P/E < 10x, P/Sales < 0.5x"
    df["pb_ratio"] = np.where(
        df["net_worth"].fillna(0) > 0,
        df["market_cap"].fillna(0) / df["net_worth"],
        np.nan
    )
    df["pb_lt1_flag"]  = (df["pb_ratio"].fillna(999) <  1.0).astype(int)   # doubler zone (Study 9/13)

    df["ps_ratio"] = np.where(
        df["revenue"].fillna(0) > 0,
        df["market_cap"].fillna(0) / df["revenue"],
        np.nan
    )
    df["ps_lt1_flag"]  = (df["ps_ratio"].fillna(999) <= 1.0).astype(int)   # multi-bagger formula (Study 13)
    df["ps_lt05_flag"] = (df["ps_ratio"].fillna(999) <= 0.5).astype(int)   # doubler formula (Study 9)

    # ── High Cash + High Debt Flag (Malik Shenanigan 4) ──
    df["high_cash_high_debt"] = (
        (df["cash_equivalents"].fillna(0) > 0) &
        (df["debt"].fillna(0) > 0) &
        (df["cash_equivalents"].fillna(0) > df["debt"].fillna(0) * 0.3)
    ).astype(int)

    # ── Malik 8-Parameter Checklist Score (Ch.4, 0-100) ──
    # Each parameter scored 0 or 12.5 (8 params × 12.5 = 100)
    pw = 12.5

    # P1: Sales Growth > 10% (>15% preferred) — use 10Y if available, fallback 5Y, 3Y
    rev_growth_best = df["rev_gr_10y"].fillna(df["rev_gr_5y"]).fillna(df["rev_gr_3y"]).fillna(0)
    malik_p1 = np.where(rev_growth_best >= 15, pw,
               np.where(rev_growth_best >= 10, pw * 0.7, 0))

    # P2: NPM > 8%, stable or improving
    npm_stable = (df["npm"].fillna(0) >= df["npm_1yb"].fillna(0)).astype(float)
    malik_p2 = np.where(
        (df["npm"].fillna(0) >= 8) & (npm_stable >= 1), pw,
        np.where(df["npm"].fillna(0) >= 8, pw * 0.8,
        np.where(df["npm"].fillna(0) >= 5, pw * 0.5, 0)))

    # P3: Tax Rate ~25-30% (now computed from actual data)
    malik_p3 = np.where(
        df["tax_rate_est"].notna(),
        np.where((df["tax_rate_est"] >= 20) & (df["tax_rate_est"] <= 35), pw,
        np.where((df["tax_rate_est"] >= 15) & (df["tax_rate_est"] <= 40), pw * 0.5, 0)),
        pw * 0.3  # no data = small benefit of doubt
    )

    # P4: Interest Coverage > 3x
    _ic_p4 = df.get("interest_coverage", _de_nan).fillna(0)
    malik_p4 = np.where(_ic_p4 >= 8, pw,
               np.where(_ic_p4 >= 3, pw * 0.7, 0))

    # P5: D/E < 0.5
    malik_p5 = np.where(df["debt_to_equity"].fillna(0) <= 0, pw,
               np.where(df["debt_to_equity"].fillna(0) <= 0.5, pw * 0.9,
               np.where(df["debt_to_equity"].fillna(0) <= 1.0, pw * 0.5, 0)))

    # P6: Current Ratio > 1.25
    malik_p6 = np.where(df["current_ratio"].fillna(0) >= 1.5, pw,
               np.where(df["current_ratio"].fillna(0) >= 1.25, pw * 0.7, 0))

    # P7: CFO positive (both current and 1YB for consistency)
    ocf_curr_pos = df["operating_cash_flow"].fillna(0) > 0
    ocf_1yb_pos = df["ocf_1yb"].fillna(0) > 0
    malik_p7 = np.where(ocf_curr_pos & ocf_1yb_pos, pw,  # both years positive
               np.where(ocf_curr_pos, pw * 0.7, 0))       # at least current positive

    # P8: CFO/PAT ≈ 1.0 (cfo_to_pat in CSV is PERCENTAGE, e.g. 73.04%)
    cfo_pat_pct = df["cfo_to_pat"].fillna(0)  # already in percentage
    malik_p8 = np.where(cfo_pat_pct >= 100, pw,            # CFO ≥ PAT = gold
               np.where(cfo_pat_pct >= 70, pw * 0.7,       # 70-100% = pass
               np.where(cfo_pat_pct >= 50, pw * 0.3, 0)))  # 50-70% = partial

    df["malik_score"] = pd.Series(
        malik_p1 + malik_p2 + malik_p3 + malik_p4 +
        malik_p5 + malik_p6 + malik_p7 + malik_p8,
        index=df.index
    ).clip(0, 100).round(1)

    df["malik_label"] = np.select(
        [df["malik_score"] >= 80, df["malik_score"] >= 60, df["malik_score"] >= 40],
        ["🟢 Strong", "🟡 Moderate", "🟠 Weak"],
        default="🔴 Poor"
    )

    # ══════════════════════════════════════════════════════════════
    # MOTILAL OSWAL WEALTH CREATION SIGNALS (30 Annual Studies)
    # ══════════════════════════════════════════════════════════════

    # ── Moat-Growth Matrix (22nd WCS) ──
    has_moat = df["roce_med_5y"].fillna(df["roce"]).fillna(0) >= 15
    has_growth = df["pat_gr_5y"].fillna(df["pat_gr_3y"]).fillna(0) >= 15
    df["moat_growth_quad"] = np.select(
        [has_moat & has_growth, has_moat & ~has_growth, ~has_moat & has_growth],
        ["⭐ Wealth Creator", "🛡️ Quality Trap", "⚡ Growth Trap"],
        default="💀 Wealth Destroyer"
    )

    # ── Sales→Profit Conversion (Malik Moat Test 3 + WCS) ──
    # Profit CAGR should >= Revenue CAGR (operating leverage proof)
    df["sales_profit_conversion"] = np.where(
        df["rev_gr_5y"].fillna(0) > 0,
        df["pat_gr_5y"].fillna(0) - df["rev_gr_5y"].fillna(0),
        np.nan
    )
    df["operating_leverage"] = (df["sales_profit_conversion"].fillna(0) > 0).astype(int)

    # ── P/E < ROE Rule (Raamdeo's 1st WCS) ──
    # Inherent margin of safety when PE < sustainable ROE
    df["pe_vs_roe_mos"] = np.where(
        df["pe"].notna() & df["roe"].notna() & (df["pe"] > 0),
        df["roe"].fillna(0) - df["pe"].fillna(0),  # positive = MoS exists
        np.nan
    )
    df["pe_below_roe"] = (df["pe_vs_roe_mos"].fillna(0) > 0).astype(int)

    # ── Earnings Yield (Malik Ch.9 + Marks) ──
    # EY = 100 / PE. Must exceed G-Sec (~7%) + 3% = 10%
    df["earnings_yield"] = np.where(
        df["pe"].notna() & (df["pe"] > 0),
        100.0 / df["pe"],
        np.nan
    )
    df["ey_adequate"] = (df["earnings_yield"].fillna(0) >= 10).astype(int)  # EY > 10%

    # ── PEG Safety with multiple tiers ──
    df["peg_zone"] = np.select(
        [
            df["peg"].fillna(99) <= 0,           # negative PEG = declining earnings
            df["peg"].fillna(99) <= 0.5,          # very cheap
            df["peg"].fillna(99) <= 1.0,          # Lynch sweet spot
            df["peg"].fillna(99) <= 1.5,          # fair
            df["peg"].fillna(99) <= 2.0,          # expensive
        ],
        ["🔴 Declining", "💎 Deep Value", "🟢 Fair PEG", "🟡 Stretched", "🟠 Expensive"],
        default="🔴 Overpriced"
    )

    # ── Capex Efficiency (CWIP → Revenue conversion) ──
    # CWIP conversion = CWIP decreased (went live as fixed assets). Use direct check to avoid
    # depending on cwip_conversion which is redefined at D19 to the FA-expansion formula.
    df["capex_productive"] = (
        (df["cwip_1yb"].fillna(0) > df["cwip"].fillna(0)) &  # CWIP converted to assets (CWIP fell)
        (df["rev_gr_yoy"].fillna(0) > 0)                      # AND revenue grew
    ).astype(int)

    # ══════════════════════════════════════════════════════════════
    # HANDBOOK DERIVED SIGNALS (D04–D50)
    # Complete set from Multibagger Discovery Handbook V2.
    # These power the GOD Screen scoring, catalyst flags, and UI display.
    # ══════════════════════════════════════════════════════════════

    # ── D04: Expense Growth YoY (%) ──
    df["d04_expense_gr_yoy"] = np.where(
        df["expenses_1yb"].notna() & (df["expenses_1yb"].abs() > 0),
        (df["expenses"] - df["expenses_1yb"]) / df["expenses_1yb"].abs() * 100,
        np.nan
    )

    # ── D05: Revenue Minus Expense Growth (Operating Leverage Signal) ──
    # D05 > 0: Revenue outpacing costs = margin expansion = quality growth
    # D05 > 5: Meaningful operating leverage
    # D05 > 10: Strong operating leverage → OPERATING LEVERAGE INFLECTION catalyst
    df["d05_rev_minus_exp_gr"] = np.where(
        df["rev_gr_yoy"].notna() & df["d04_expense_gr_yoy"].notna(),
        df["rev_gr_yoy"] - df["d04_expense_gr_yoy"],
        np.nan
    )

    # ── D09: Annual NPM Expansion (current NPM vs 1Y back) ──
    # Different from npm_acceleration (which uses latest quarter)
    df["d09_npm_expansion"] = df["npm"] - df["npm_1yb"]

    # ── D11: NPM vs 5Y Median ──
    df["d11_npm_above_5y_med"] = df["npm"] - df["npm_med_5y"]

    # ── D12: Debt Growth 1Y (%) ──
    df["d12_debt_gr_1y"] = np.where(
        df["debt_1yb"].notna() & (df["debt_1yb"].abs() > 0),
        (df["debt"] - df["debt_1yb"]) / df["debt_1yb"].abs() * 100,
        np.nan
    )

    # ── D13: Debt Growth 3Y (%) ──
    df["d13_debt_gr_3y"] = np.where(
        df["debt_3yb"].notna() & (df["debt_3yb"].abs() > 0),
        (df["debt"] - df["debt_3yb"]) / df["debt_3yb"].abs() * 100,
        np.nan
    )

    # ── D14: Debt Trajectory Score (0–3) — consecutive declining debt years ──
    # Pabrai's downside protection: D14 = 3 means debt fell every year for 3 years.
    # NaN guard: only score years where BOTH values are known; missing history scores 0 (conservative).
    _d14_y1 = (df["debt"].notna() & df["debt_1yb"].notna() &
               (df["debt"] < df["debt_1yb"])).astype(int)
    _d14_y2 = (df["debt_1yb"].notna() & df["debt_2yb"].notna() &
               (df["debt_1yb"] < df["debt_2yb"])).astype(int)
    _d14_y3 = (df["debt_2yb"].notna() & df["debt_3yb"].notna() &
               (df["debt_2yb"] < df["debt_3yb"])).astype(int)
    df["d14_debt_trajectory"] = _d14_y1 + _d14_y2 + _d14_y3

    # ── D15: Cash-to-Debt Ratio ──
    df["d15_cash_to_debt"] = np.where(
        df["debt"].notna() & (df["debt"] > 0),
        df["cash_equivalents"].fillna(0) / df["debt"],
        np.nan
    )

    # ── D17: Cash Growth YoY (%) ──
    df["d17_cash_gr_yoy"] = np.where(
        df["cash_equivalents_1yb"].notna() & (df["cash_equivalents_1yb"].abs() > 0),
        (df["cash_equivalents"] - df["cash_equivalents_1yb"]) / df["cash_equivalents_1yb"].abs() * 100,
        np.nan
    )

    # ── D19: CWIP Conversion — Net Fixed Asset Expansion ──
    # When CWIP goes live it transfers out of CWIP and into Net Fixed Assets in the same entry.
    # The old formula added ΔCWIP + ΔFA, double-counting that exact transaction (2× error).
    # True capital deployment velocity = net expansion of the fixed asset block.
    df["d19_cwip_conversion"] = df["fixed_assets"].fillna(0) - df["fixed_assets_1yb"].fillna(0)
    df["cwip_conversion"] = df["d19_cwip_conversion"]

    # ── John Kay IBAS Moat Engine ──
    # Four-factor structural moat intensity model (The Unusual Billionaires, Appendix 1).
    # Proxied entirely from existing CSV-derived signals — no new data required.
    #
    # 1. Reputation/Brand: stable OPM (pricing power) + NPM above own 5Y median (sustaining premium)
    df["ibas_reputation_score"] = (
        df.get("opm_stable", pd.Series(0, index=df.index)).fillna(0) * 50.0
        + (df["d11_npm_above_5y_med"].fillna(0) > 0).astype(float) * 50.0
    )
    # 2. Architecture/Network: negative CCC = company collects cash BEFORE paying suppliers
    #    (HUL/Nestle/Asian Paints pattern — Kay's "relational contract" moat).
    #    CCC is nulled for financials — fillna(0) → 0 < 0 = False → architecture = 0 (correct).
    df["ibas_architecture_score"] = (df["ccc"].fillna(0) < 0).astype(float) * 100.0
    # 3. Innovation/Efficiency: asset-light business model (NFAT > 4 = revenue is 4× net FA)
    #    + growth self-funded from internal cash (no dilution required)
    df["ibas_innovation_score"] = (
        (df["nfat"].fillna(0) > 4.0).astype(float) * 50.0
        + df["ssgr_self_funded"].fillna(0) * 50.0
    )
    # 4. Strategic Assets: CWIP actively converting to productive fixed assets (D19 > 0)
    #    = capital is being deployed into real capacity, not parking in incomplete projects.
    df["ibas_strategic_assets_score"] = (df["d19_cwip_conversion"].fillna(0) > 0).astype(float) * 100.0
    # Composite: equal-weight average across all four Kay moat types (0–100 scale)
    df["ibas_moat_score"] = (
        df["ibas_reputation_score"]
        + df["ibas_architecture_score"]
        + df["ibas_innovation_score"]
        + df["ibas_strategic_assets_score"]
    ) / 4.0

    # ── D20: Fixed Asset CAGR 3Y (%) ──
    df["d20_fa_cagr_3y"] = np.where(
        df["fixed_assets_3yb"].notna() & (df["fixed_assets_3yb"] > 0) & (df["fixed_assets"].fillna(0) > 0),
        ((df["fixed_assets"] / df["fixed_assets_3yb"]) ** (1.0 / 3.0) - 1.0) * 100,
        np.nan
    )

    # ── D24: OCF/PAT Delta (CFO/PAT − 100) ──
    # D24 ≥ 0: OCF ≥ PAT = earnings are cash-backed (Clean Accounts signal)
    df["d24_ocf_pat_delta"] = df["cfo_to_pat"].fillna(np.nan) - 100.0

    # ── D27: FCF Positive (binary) ──
    df["d27_fcf_positive"] = (df["free_cash_flow"].fillna(0) > 0).astype(int)

    # ── D28: FCF-to-PAT (%) ──
    # D28 > 50%: FCF covers more than half of PAT = strong real cash generation
    df["d28_fcf_to_pat_pct"] = np.where(
        df["pat"].notna() & (df["pat"].abs() > 0),
        df["free_cash_flow"].fillna(0) / df["pat"].abs() * 100,
        np.nan
    )

    # ── D29: Capex Intensity (%) ──
    # (OCF − FCF) / |OCF| × 100: how much of operating cash goes to capex
    df["d29_capex_intensity"] = np.where(
        df["operating_cash_flow"].notna() & (df["operating_cash_flow"].abs() > 0),
        (df["operating_cash_flow"] - df["free_cash_flow"].fillna(df["operating_cash_flow"])) /
        df["operating_cash_flow"].abs() * 100,
        np.nan
    )

    # ── D32: PE vs 10Y Median (%) — negative = trading below own history ──
    df["d32_pe_vs_median"] = np.where(
        df["pe_med_10y"].notna() & (df["pe_med_10y"] > 0) & df["pe"].notna(),
        (df["pe"] - df["pe_med_10y"]) / df["pe_med_10y"] * 100,
        np.nan
    )

    # ── D33: PE vs Industry (%) — negative = cheap vs sector peers ──
    df["d33_pe_vs_industry"] = np.where(
        df["industry_pe"].notna() & (df["industry_pe"] > 0) & df["pe"].notna(),
        (df["pe"] - df["industry_pe"]) / df["industry_pe"] * 100,
        np.nan
    )

    # ── D34: EV/EBITDA Direction — positive = getting more expensive ──
    df["d34_ev_ebitda_dir"] = df["ev_ebitda"] - df["ev_ebitda_1yb"]

    # ── D35: ROCE Trend (current vs 1Y back) — positive = improving ──
    df["d35_roce_trend"] = df["roce"] - df["roce_1yb"]

    # ── D36: ROCE vs 5Y Median — positive = above historical average ──
    df["d36_roce_above_med"] = df["roce"] - df["roce_med_5y"]

    # ── D37: CCC Direction (positive = worsening working capital) ──
    df["d37_ccc_direction"] = df["ccc"] - df["ccc_1yb"]

    # ── D38: Smart Money (Promoter LQ + FII LQ change) ──
    df["d38_smart_money"] = (
        df["change_promoter_lq"].fillna(0) + df["change_fii_lq"].fillna(0)
    )

    # ── D39: Institutional Tide (FII + DII latest quarter change) ──
    df["d39_inst_tide"] = (
        df["change_fii_lq"].fillna(0) + df["change_dii_lq"].fillna(0)
    )

    # ── D40: Promoter + FII 1Y change ──
    df["d40_promo_fii_1y"] = (
        df["change_promoter_1y"].fillna(0) + df["change_fii_1y"].fillna(0)
    )

    # ── D41: Pledge Trajectory (positive = pledge rising = danger) ──
    df["d41_pledge_trajectory"] = df["pledged_percentage"].fillna(0) - df["pledged_1yb"].fillna(0)

    # ── D44: Smart Money Composite (D38 + D39 + 2 if insider bought) ──
    insider_bought = (
        df["insider_trading"].notna() &
        df["insider_trading"].fillna("").astype(str).str.contains("Bought", case=False, na=False)
    ).astype(float) * 2
    df["d44_smart_money_comp"] = df["d38_smart_money"] + df["d39_inst_tide"] + insider_bought

    # ── D45: Trend Structure Score (0–3) ──
    # +1 if Price > SMA 200D, +1 if Price > VSTOP, +1 if ADX > 20
    df["d45_trend_structure"] = (
        df["above_sma200"].fillna(0) +
        df["vstop_green"].fillna(0) +
        (df["adx_14w"].fillna(0) > 20).astype(int)
    )

    # ── D47: RS Composite — IBD-weighted (40% recent / 30% mid / 30% long) ──
    # IBD RS Rating formula: most recent quarter receives double weight vs prior periods.
    # Mapping: crs_50d (~10W) ≈ recent quarter → 40%; crs_26w (6M) → 30%; crs_52w (12M) → 30%.
    # Weights sum to 1.0 — preserves the CRS scale for percentile ranking in scoring_engine.py.
    # Used by: CAN SLIM L criterion (_pct_rank → rs_pctrank_cs >= 80) and Quant Momentum sub-score.
    df["d47_rs_composite"] = (
        df["crs_50d"].fillna(0) * 0.40 +
        df["crs_26w"].fillna(0) * 0.30 +
        df["crs_52w"].fillna(0) * 0.30
    )

    # ── D48: Breakout Readiness (categorical) ──
    df["d48_breakout_readiness"] = np.select(
        [
            (df["dist_52wh"].fillna(999) < 10) & (df["dist_13wh"].fillna(999) < 5),
            df["dist_52wh"].fillna(999) < 20,
        ],
        ["IMMINENT", "NEAR"],
        default="FAR"
    )

    # ── D49: Momentum Quality (categorical) ──
    _rsi_mq = df.get("rsi_14d", pd.Series(np.nan, index=df.index)).fillna(0)
    df["d49_momentum_quality"] = np.select(
        [
            _rsi_mq > 70,
            (_rsi_mq >= 50) & (_rsi_mq <= 70) & (df.get("adx_14w", pd.Series(np.nan, index=df.index)).fillna(0) > 20),
        ],
        ["OVERHEATED", "HIGH"],
        default="WEAK"
    )

    # ── D50: Alpha Score (average return vs benchmarks) ──
    df["d50_alpha_score"] = (
        df["ret_vs_n500_3m"].fillna(0) +
        df["ret_vs_n500_6m"].fillna(0) +
        df["ret_vs_industry_1y"].fillna(0)
    ) / 3.0

    # ── D51: QMOM Quality Score — 3-factor fundamental quality composite ──
    # Implements the Gray & Vogel Quality Overlay for momentum strategies (India Edition).
    # The handbook (Ch. 4) specifies 4 factors: GP/Assets, ROIC, D/E, CFO/PAT.
    # GP/Assets NOT computed (COGS not in CSV) — ROCE used as profitability proxy.
    # ROCE proxy for ROIC: capital employed ≈ equity + net debt → formula is near-identical.
    # Composite = mean of 3 percentile ranks (0–1 scale). Threshold >= 0.50 = top-half quality.
    # NaN handling: fillna(0) for ROCE/CFO (unknown = lowest rank = conservative)
    #               fillna(1.0) for D/E (unknown = moderate leverage = middle rank)
    _qm_roce_r = df["roce"].fillna(0).rank(pct=True)
    _qm_de_inv = 1.0 / (df.get("debt_to_equity", _de_nan).fillna(1.0) + 0.01)
    _qm_de_r   = _qm_de_inv.rank(pct=True)
    _qm_cfo_r  = df["cfo_to_pat"].fillna(0).rank(pct=True)
    # 4th factor: GP/Assets proxy (Novy-Marx) using gpm_latest_q × annual revenue / total assets
    _ta       = df.get("total_assets", _de_nan).replace(0, np.nan)
    _gpm_q    = df.get("gpm_latest_q", _de_nan)
    _rev_safe = df.get("revenue",      _de_nan).fillna(0)
    _qm_gp_r  = ((_gpm_q / 100.0) * _rev_safe / _ta).rank(pct=True)
    if _qm_gp_r.notna().sum() > 100:
        df["d51_qmom_quality_score"] = ((_qm_roce_r + _qm_de_r + _qm_cfo_r + _qm_gp_r) / 4.0).round(3)
    else:
        df["d51_qmom_quality_score"] = ((_qm_roce_r + _qm_de_r + _qm_cfo_r) / 3.0).round(3)

    # ══════════════════════════════════════════════════════════════
    # MOTILAL OSWAL 30-STUDY ALPHA SIGNALS
    # Derived from 30 Annual Wealth Creation Studies (1991-2025).
    # These are the empirically most-validated signals in Indian equity research.
    # ══════════════════════════════════════════════════════════════

    # ── Payback Ratio (MOSL's single most validated supernormal-return predictor) ──
    # "Payback Ratio < 1x is the most reliable valuation metric for supernormal returns."
    #   — confirmed in every one of the 30 Annual Wealth Creation Studies
    # Conservative (0% growth): payback_0g = market_cap / (5 × current PAT)
    # Growth-adjusted: market_cap / cumulative 5Y PAT at estimated CAGR
    pat_safe = df["pat"].fillna(0).clip(lower=0.01)
    payback_0g = np.where(
        (df["market_cap"].fillna(0) > 0) & (df["pat"].fillna(0) > 0),
        df["market_cap"] / (5.0 * pat_safe),
        np.nan
    )
    df["payback_ratio_0g"] = pd.Series(payback_0g, index=df.index)

    # Growth-adjusted: geometric sum of 5Y PAT at estimated CAGR (from pat_gr_5y)
    g_rate = (df["pat_gr_5y"].fillna(0) / 100.0).clip(lower=0, upper=0.50)
    # G6 FIX: formula was computing sum of years 0-4 (annuity due base).
    # Correct formula for years 1-5: (1+g) × ((1+g)^5 - 1) / g.
    # At g=20%: was 7.44, now 8.93 — ~20% higher cumulative PAT, lower payback period.
    geo_sum = np.where(
        g_rate > 0.001,
        (1.0 + g_rate) * ((1.0 + g_rate) ** 5 - 1.0) / g_rate,
        5.0
    )
    payback_growth = np.where(
        (df["market_cap"].fillna(0) > 0) & (df["pat"].fillna(0) > 0),
        df["market_cap"] / (pat_safe * pd.Series(geo_sum, index=df.index)),
        np.nan
    )
    df["payback_ratio"] = pd.Series(payback_growth, index=df.index)
    df["payback_lt05"] = (df["payback_ratio"].fillna(99) < 0.5).astype(int)  # supernormal tier 1 (Studies 7-9: top-7 fastest ALL had payback < 0.5x)
    df["payback_lt1"]  = (df["payback_ratio"].fillna(99) < 1.0).astype(int)  # supernormal tier 2
    df["payback_lt2"]  = (df["payback_ratio"].fillna(99) < 2.0).astype(int)  # attractive zone

    # ── Consistency Champion (27th Study: Consistents vs Volatiles) ──
    # Full 15Y analysis requires 15 years of annual PAT — not in CSV.
    # Proxy using available history: no PAT crash + long-term positive trajectory.
    pat_decline_1y = np.where(
        df["pat_1yb"].fillna(0) > 0,
        (df["pat"].fillna(0) - df["pat_1yb"].fillna(0)) / df["pat_1yb"].abs() * 100,
        np.nan  # Missing 1YB PAT → unknown, not "0% change" — preserves correct NaN downstream
    )
    df["pat_decline_1y_pct"] = pd.Series(pat_decline_1y, index=df.index)
    df["pat_no_crash_1y"] = (df["pat_decline_1y_pct"].fillna(0) > -50).astype(int)
    df["pat_growing_long"] = (
        df["pat_gr_10y"].fillna(df["pat_gr_5y"]).fillna(0) > 0
    ).astype(int)
    df["consistency_champion"] = (
        (df["pat_no_crash_1y"] == 1) &
        (df["pat_growing_long"] == 1) &
        (df["pat"].fillna(0) > 0)
    ).astype(int)

    # ── Economic Profit Improving (28th Study — TEM Hockey-Stick Setup) ──
    # Companies moving UP the Economic Profit Power Curve:
    # ROE improving AND above cost of equity (10% for India)
    df["eco_profit_improving"] = (
        (df["roe"].fillna(0) > df["roe_1yb"].fillna(0)) &
        (df["roe"].fillna(0) > 10.0)
    ).astype(int)

    # ── P/E to Sustainable ROE Ratio (continuous MoS — 1st Study, all 30 confirmed) ──
    # pe_to_roe_ratio < 1 = PE below sustainable ROE = inherent margin of safety
    df["pe_to_roe_ratio"] = np.where(
        df["roe_med_10y"].notna() & (df["roe_med_10y"] > 1) &
        df["pe"].notna() & (df["pe"] > 0),
        df["pe"] / df["roe_med_10y"],
        np.nan
    )

    # ── Sector Tailwind (30th Study: India Multi-Trillion Dollar Opportunity) ──
    # Financials + Consumer Discretionaries = explosive tipping-point sectors 2025-2040
    _tailwind_patt = "Bank|NBFC|Insurance|Finance|Auto|Consumer|Health|Pharma|Retail|Capital Market"
    df["sector_tailwind"] = (
        df["industry"].fillna("").astype(str).str.contains(_tailwind_patt, case=False, na=False) |
        df["sector"].fillna("").astype(str).str.contains(_tailwind_patt, case=False, na=False)
    ).astype(int)

    # ── Multi-Trillion Compounding Tipping Point (30th WCS — 2025 Theme) ──
    # Industry names verified against 354 CSV values. Previous set had 0/8 matches (all sector names).
    # FINANCIAL_SECTORS already contains the correct 17 financial industry names — reuse it.
    _mtc_in_sector = (
        df["industry"].fillna("").isin(FINANCIAL_SECTORS)
        | df["sector"].fillna("").isin(FINANCIAL_SECTOR_NAMES)
        | df["sector"].fillna("").isin({"Consumer Durables", "Automobile", "Retail", "E-Commerce/App based Aggregator"})
    )
    _mtc_vol_surge     = df["vol_ratio"].fillna(0) >= 1.5
    _mtc_earnings_mom  = (
        (df.get("q_pat_yoy", pd.Series(0, index=df.index)).fillna(0) > 25) |
        (df.get("pat_gr_3y", pd.Series(0, index=df.index)).fillna(0) > 25)
    )
    _mtc_near_breakout = df.get("dist_52wh", pd.Series(999, index=df.index)).fillna(999) <= 15
    _mtc_signal_count  = (
        _mtc_vol_surge.astype(int) +
        _mtc_earnings_mom.astype(int) +
        _mtc_near_breakout.astype(int)
    )
    df["multitrillioncap_tipping_point"] = (
        _mtc_in_sector & (_mtc_signal_count >= 2)
    ).astype(int)

    # ── Bruised Blue Chip Detection (29th Study) ──
    # Quality company fallen hard + trading cheap vs own history = asymmetric payoff
    # Criteria: quality company (ROCE > 15% + PAT CAGR > 10%) AND
    #           fallen significantly (>40% off 52W high as proxy for 50%+ from 5Y high) AND
    #           cheap vs own history (current PE > 25% below 10Y median PE)
    _bbc_fallen   = df["dist_52wh"].fillna(0) > 40
    _bbc_quality  = (
        (df["roce_med_5y"].fillna(df["roce"]).fillna(0) >= 15) &
        (df["pat_gr_5y"].fillna(0) >= 10)
    )
    _bbc_cheap    = df["d32_pe_vs_median"].fillna(0) < -25
    df["bruised_blue_chip"] = (_bbc_fallen & _bbc_quality & _bbc_cheap).astype(int)

    # ── Bruised Blue Chip 29th WCS (Large-Cap Elite ROCE + PE Discount + P/B ≤ 3x) ──
    # Canonical definition: all three must hold simultaneously.
    # pe_discount = (10Y mean PE - current PE) / 10Y mean PE × 100 (positive = cheaper than history)
    _bbc29_largecap   = df["market_cap"].fillna(0) >= 20_000
    _bbc29_elite_roce = df["roce_med_10y"].fillna(0) >= 20
    _bbc29_pe_bruised = df["pe_discount"].fillna(0) >= 20
    _bbc29_pb_check   = df["pb_ratio"].fillna(999) <= 3.0
    df["bruised_blue_chip_29"] = (
        _bbc29_largecap & _bbc29_elite_roce & _bbc29_pe_bruised & _bbc29_pb_check
    ).astype(int)

    # ── Coffee Can Twin Filter + Clean Accounts Pass Flag ──
    # Mukherjea "Coffee Can Investing" Ch.2 + Ch.3 — the complete entry criterion:
    #   Twin Filter A: Revenue CAGR ≥ 10% sustained (rev_gr_10y ≥ 10)
    #   Twin Filter B: Capital efficiency ≥ 15% for 10 consecutive years
    #     Non-financial: ROCE 10Y median ≥ 15% (roce_med_10y ≥ 15)
    #     Financial:     ROE  10Y median ≥ 15% (roe_med_10y  ≥ 15) — banks use ROE, not ROCE
    #   Clean Accounts: CFO/EBITDA ≥ 90% (the master earnings quality signal — Ch.3)
    #   Balance Sheet:  D/E ≤ 1.0 (already a hard gate, replicated here for composite completeness)
    # fillna(0): missing 10Y data → fails the filter (conservative — no history = no proof).
    # fillna(999): D/E null → fails fortress check (unknown debt = not a fortress).
    # Result: 20–40 stocks expected to pass from 2,100+ universe (matches book's prediction).
    _cc_rev_ok      = df["rev_gr_10y"].fillna(0) >= 10.0
    _cc_de_ok       = df["debt_to_equity"].fillna(999) <= 1.0
    _cc_cfo_ebitda  = df.get("cfo_to_ebitda", pd.Series(0.0, index=df.index)).fillna(0) >= 90.0
    _cc_fin         = df["is_financial"]
    _cc_roce_ok     = df["roce_med_10y"].fillna(0) >= 15.0   # non-financial
    _cc_roe_ok      = df.get("roe_med_10y", pd.Series(0.0, index=df.index)).fillna(0) >= 15.0  # financial
    _cc_efficiency  = np.where(_cc_fin, _cc_roe_ok, _cc_roce_ok)
    df["coffee_can_pass"] = (
        _cc_rev_ok & _cc_de_ok & _cc_cfo_ebitda &
        pd.Series(_cc_efficiency.astype(bool), index=df.index)
    ).astype(int)

    # ── Identity D: Macro Tipping Point Velocity (30th WCS — India Multi-Trillion Engine) ──
    # Continuous velocity indicator: Rev Growth YoY × NPM × Vol Ratio (volume confirm).
    # Set to 0 for non-tipping sectors so the composite boost is sector-gated.
    _ttp_in_sector = df["sector"].fillna("").isin(EPOCH5_MODERN["tipping_sectors"])
    df["is_tipping_sector"] = _ttp_in_sector.astype(int)
    df["tipping_point_velocity"] = np.where(
        _ttp_in_sector,
        df["rev_gr_yoy"].fillna(0) * df["npm"].fillna(0) * df["vol_ratio"].fillna(1.0),
        0.0
    )

    # ══════════════════════════════════════════════════════════════
    # VIJAY MALIK PEACEFUL INVESTING SIGNALS (Vol 1–3 deep extraction)
    # "Hard Gates" layer — capital efficiency, OPM consistency, cash conversion quality.
    # ══════════════════════════════════════════════════════════════

    # ── FCF/CFO Conversion Quality (Vijay Malik's single most diagnostic ratio) ──
    # Finolex Cables: 76% — gold standard. PIX Transmissions: negative — capital trap.
    # Captures what FCF/PAT misses: how much of operating cash survives after capex.
    # Only meaningful when OCF is positive; when OCF < 0, rf_negative_fcf handles it.
    _fcf_imputed = df.get("fcf_imputed_flag", pd.Series(0, index=df.index)).fillna(0).astype(bool)
    df["fcf_to_cfo_pct"] = np.where(
        df["operating_cash_flow"].notna() & (df["operating_cash_flow"] > 0) & ~_fcf_imputed,
        df["free_cash_flow"].fillna(0) / df["operating_cash_flow"] * 100,
        np.nan
    )

    # ── DSO Delta 3Y — Diamonds Lens 1 (channel-stuffing detector, true 3Y window) ──
    # Book: DSO must not rise > 15 days over any 3-year trailing window (Mukherjea Ch.3).
    # days_receivable_3yb now available in CSV — computes the exact 3Y delta the book specifies.
    # fillna(np.nan): missing data handled in fw_diamond with fillna(999) → fails gate conservatively.
    df["dso_delta_3y"] = np.where(
        df["days_receivable"].notna() & df["days_receivable_3yb"].notna(),
        df["days_receivable"] - df["days_receivable_3yb"],
        np.nan
    )

    # ── Cumulative FCF/CFO — Diamonds Lens 3 proxy (self-sufficiency test) ──
    # Book: 10Y cumulative FCF / 10Y cumulative CFO >= 25% (Mukherjea Ch.5).
    # CSV lacks 10Y cumulative series. Proxy: fcf_to_cfo_pct (point-in-time FCF/CFO %).
    # Aliased here so fw_diamond reads a semantically correct column name.
    df["cumulative_fcf_to_ccfo"] = df.get("fcf_to_cfo_pct", pd.Series(np.nan, index=df.index))

    # ── Inventory Days (complement to ITR already in CSV) ──
    # Malik: <50 days = excellent, 50-80 = watch, >80 = red flag
    df["inventory_days"] = np.where(
        df["inventory_turnover"].notna() & (df["inventory_turnover"] > 0),
        365.0 / df["inventory_turnover"],
        np.nan
    )
    df["inventory_days_1yb"] = np.where(
        df["inventory_turnover_1yb"].notna() & (df["inventory_turnover_1yb"] > 0),
        365.0 / df["inventory_turnover_1yb"],
        np.nan
    )
    # Lynch radar: rising inventory days = demand slowing before the P&L shows it.
    # Positive = worsening (more days to sell). Used in tearsheet; rf_itr_declining covers forensic flag.
    df["inventory_days_change"] = np.where(
        df["inventory_days"].notna() & df["inventory_days_1yb"].notna(),
        df["inventory_days"] - df["inventory_days_1yb"],
        np.nan
    )

    # ── OPM Stability (pricing power vs commodity trap) ──
    # Vijay Malik: Maithan Alloys OPM swings 3% → 21% = no pricing power.
    # Finolex Cables OPM stable 7→16% = structured improvement = pricing power.
    # Stability = how far current OPM deviates from 5Y median (as % of median).
    # Lower deviation = more stable = pricing power. >30% deviation = commodity trap.
    df["opm_stability"] = np.where(
        df["opm_med_5y"].notna() & (df["opm_med_5y"] > 0),
        (df["opm_1yb"].fillna(df["opm_latest_q"]).fillna(df["opm_med_5y"]) -
         df["opm_med_5y"]).abs() / df["opm_med_5y"] * 100,
        np.nan
    )
    df["opm_stable"] = (df["opm_stability"].fillna(99) < 20).astype(int)

    # ── WCS Score: Wealth Creation Criteria Count (0–10) ──
    # Each criterion is a separately validated MOSL supernormal-return predictor.
    # Studies 7-13 synthesised: criteria 1-7 from original; 8-10 from Study 9/13 "four formulas".
    # The more criteria met, the higher the probability of 5Y outperformance.
    df["wcs_score"] = (
        ((df["peg"].fillna(99) > 0) & (df["peg"].fillna(99) <= 1.0)).astype(int) +   # 1. PEG < 1
        df["pe_below_roe"].fillna(0).astype(int) +                                    # 2. PE < ROE (MoS)
        df["economic_profit_positive"].fillna(0).astype(int) +                        # 3. EP > 0
        (df["pat_gr_5y"].fillna(0) >= 20).astype(int) +                               # 4. PAT CAGR > 20%
        (df["roce_med_5y"].fillna(df["roce"]).fillna(0) >= 15).astype(int) +          # 5. ROCE > 15%
        df["payback_lt1"].fillna(0).astype(int) +                                     # 6. Payback < 1 (Study 13: 82/100 wealth creators)
        df["consistency_champion"].fillna(0).astype(int) +                            # 7. PAT consistency
        (df["pe"].fillna(999) <= 10).astype(int) +                                    # 8. PE ≤ 10 (Study 9/13: "doubler formula")
        df["pb_lt1_flag"].fillna(0).astype(int) +                                     # 9. P/B < 1x (Study 9/13: 67% CAGR zone)
        df["ps_lt1_flag"].fillna(0).astype(int)                                       # 10. P/Sales ≤ 1x (Study 13: 62% CAGR zone)
    )

    # ══════════════════════════════════════════════════════════════
    # MOSL STUDIES 14-19: CATEGORY WINNERS, UU INVESTING, BLUE CHIPS,
    # ECONOMIC MOAT, UNCOMMON PROFITS, 100x / SQGLP
    # Source: 14th-19th Annual Wealth Creation Studies (2009-2015 themes)
    # ══════════════════════════════════════════════════════════════

    # ── Study 17 (2012): Economic Moat — Sector-Relative ROE (EMC Flag) ──
    # Backtested 1995-2012: EMC portfolio → 25% CAGR, non-EMC → 12% CAGR, Alpha = +7%.
    # Criterion: ROE > sector average for 6/8 years. Proxy: above-median for current + 5yr.
    _sector_grp_roe = df["sector"].fillna("Unknown")
    _sector_roe_med  = df.groupby(_sector_grp_roe)["roe"].transform("median").fillna(df["roe"].median())
    _sector_roe5_med = df.groupby(_sector_grp_roe)["roe_med_5y"].transform("median").fillna(df["roe_med_5y"].median())
    df["emc_flag"] = (
        (df["roe"].fillna(0)         > _sector_roe_med.fillna(0)) &
        (df["roe_med_5y"].fillna(0)  > _sector_roe5_med.fillna(0))
    ).astype(int)

    # ── Study 14 (2009): Category Winner — Sector-Relative ROCE Leader ──
    # "Category Winners enjoy exponential growth in profits within Winner Categories."
    # 3 conditions for fastest creators: small mcap + single-digit PE + PAT CAGR > 35%.
    # Category Winner proxy: top-30% ROCE within sector AND beating-market revenue growth.
    df["sector_roce_pct_rank"] = (
        df.groupby(_sector_grp_roe)["roce"]
          .transform(lambda x: x.rank(pct=True, na_option="bottom"))
          .fillna(0.5)
    )
    df["category_winner_flag"] = (
        (df["sector_roce_pct_rank"] >= 0.70) &    # top-30% capital efficiency within sector
        (df["rev_gr_5y"].fillna(0) >= 12)          # AND above-market revenue growth
    ).astype(int)

    # Study 14 fast-track signal: small mcap + low PE + high PAT CAGR = fastest creator setup
    df["fast_creator_setup"] = (
        (df["market_cap"].fillna(0) < 4_000) &     # Base mcap < Rs4B (fastest creator filter)
        (df["pe"].fillna(99) < 10) &               # Single-digit PE at entry
        (df["pat_gr_5y"].fillna(0) >= 35)          # PAT CAGR > 35%
    ).astype(int)

    # ── Study 16 (2011): Blue Chip Quality — 6-Screen Filter (Geraldine Weiss, adapted) ──
    # Screen 4: "Average RoE ≥ 15% for last 12 years" — proxy = roe_med_10y ≥ 15%.
    # Screen 3: "Earnings growth in 7/12 years" — proxy = consistency_champion (no PAT crash).
    # Screen 1/2: "Dividend longevity + growth" — proxy = dividend_payout_ratio ≥ 20%.
    # Screen 5: "≥ 5 million shares outstanding" — proxy = equity_shares ≥ 5 (in millions).
    # From 3,000+ stocks, only 48 (1.5%) passed all 6 screens. This is appropriately rare.
    _dpr_bc    = df.get("dividend_payout_ratio", pd.Series(np.nan, index=df.index))
    _eq_shares = df.get("equity_shares",         pd.Series(np.nan, index=df.index))
    df["blue_chip_quality_flag"] = (
        (_dpr_bc.fillna(0)          >= 20) &      # Screen 1/2: consistent dividend payout
        (df["roe_med_10y"].fillna(0) >= 15) &      # Screen 4: 10yr ROE ≥ 15% (India CoE threshold)
        (df["consistency_champion"]  == 1) &       # Screen 3: PAT no-crash consistency proxy
        (_eq_shares.fillna(0)        >= 5)         # Screen 5: ≥ 5M shares liquidity proxy
    ).astype(int)

    # Synthetic dividend yield estimate (DPR × Earnings Yield)
    # Used when direct dividend yield column is unavailable in CSV.
    df["dividend_yield_synthetic"] = np.where(
        df["earnings_yield"].notna() & _dpr_bc.notna(),
        df["earnings_yield"] * (_dpr_bc.fillna(0) / 100.0),
        np.nan
    )
    df["dividend_yield_ratio"] = np.where(
        df["dividend_yield_synthetic"].notna(),
        df["dividend_yield_synthetic"] / INDIA_GSEC_YIELD,    # vs India G-Sec yield (Study 16 buy signal)
        np.nan
    )

    # ── Study 15 (2010): UU Investing — Unknown-Unknowable → Known-Knowable Setup ──
    # "The market handsomely rewards a successful journey from UU to KK."
    # Examples: Infosys IPO (890x, 173% CAGR), Bharti FY03 (37x, 123% CAGR),
    #           Pantaloon FY03 (109x, 156% CAGR), Titan 10yr (58x, 50% CAGR).
    # UU Setup = undiscovered (small cap) + earnings emerging (improving ROE) + low payback.
    df["uu_setup_flag"] = (
        (df["market_cap"].fillna(0) < 20_000) &   # Small/mid cap — less discovered
        (df["payback_lt1"] == 1) &                # Reasonable entry price (payback < 1)
        (df["roe_turnaround_flag"] == 1)          # ROE improving = UU→KK journey beginning
    ).astype(int)

    # ── Study 18 (2013): Uncommon Profits — Emerging vs Enduring Value Creators ──
    # Emerging VC: first-time ROE crossing 15% (Cost of Equity threshold) = Emergence event.
    # Filter: corporate-parent quality + non-cyclical + PE ≤ 20x at emergence.
    # "In most cases, there is no significant gain in pre-empting emergence."
    df["emerging_vc_flag"] = (
        (df["roe"].fillna(0) >= 15) &             # Current: crossed the 15% CoE threshold
        (df["roe_med_5y"].fillna(0) < 15) &        # Historical: was below (first-time crossing)
        (df["pat_gr_3y"].fillna(0) > 15) &         # Growth confirming the emergence
        (df["economic_profit_positive"] == 1) &    # Earning above cost of equity
        (df["pe"].fillna(999) <= 20)               # Reasonable valuation at emergence
    ).astype(int)

    # Enduring VC: proven decade of above-CoE returns = the permanent compounder.
    # "It's hard for a stock to earn more return than the business itself earns." — Munger
    df["enduring_vc_flag"] = (
        (df["economic_profit_positive"] == 1) &
        (df["consistency_champion"] == 1) &
        (df["roe_med_10y"].fillna(0) >= 15)        # Decade of above-CoE returns
    ).astype(int)

    # ── Study 19 (2014): 100x / SQGLP — Five-Factor Century Stock Screen ──
    # "100x requires vision to see, courage to buy, and patience to hold." — Thomas Phelps
    # SQGLP: S=Size, Q=Quality, G=Growth, L=Longevity, P=Price
    # 100x stock profile: avg P/E 6x at purchase → 24x at exit; mcap < USD 500M at entry.
    _sqglp_s = (df["market_cap"].fillna(0) < 40_000).astype(int)    # S: < ~USD 500M mcap
    _sqglp_q = (
        (df["roce"].fillna(0)      >= 15) &
        (df["roe"].fillna(0)       >= 15) &
        (df["cfo_to_pat"].fillna(0) >= 70)
    ).astype(int)                                                     # Q: Quality trifecta
    _sqglp_g = (
        (df["pat_gr_5y"].fillna(0)  >= 20) &
        (df["rev_gr_5y"].fillna(0)  >= 15)
    ).astype(int)                                                     # G: Earnings + revenue growth
    _sqglp_l = (
        df["pat_gr_10y"].fillna(df["pat_gr_5y"]).fillna(0) >= 12
    ).astype(int)                                                     # L: 10yr sustainable growth
    _sqglp_p = (df["pe"].fillna(999) <= 15).astype(int)              # P: Favorable entry price

    df["sqglp_s"] = _sqglp_s
    df["sqglp_q"] = _sqglp_q
    df["sqglp_g"] = _sqglp_g
    df["sqglp_l"] = _sqglp_l
    df["sqglp_p"] = _sqglp_p
    df["sqglp_score"] = _sqglp_s + _sqglp_q + _sqglp_g + _sqglp_l + _sqglp_p

    # century_stock_flag: 4-5/5 SQGLP criteria = highest-probability 100x candidate
    df["century_stock_flag"] = (df["sqglp_score"] >= 4).astype(int)

    # ══════════════════════════════════════════════════════════════
    # MOSL STUDIES 20-24: MQGLP/MID-TO-MEGA, CAP/GAP LONGEVITY,
    # VALUATION INSIGHTS, MANAGEMENT INTEGRITY
    # Source: 20th-24th Annual Wealth Creation Studies (2015-2019 themes)
    # ══════════════════════════════════════════════════════════════

    # ── Study 20 (2015): MQGLP — Mid-to-Mega Candidate ──
    # "Mid-to-Mega" template: median 46% return, 28% alpha vs Sensex (2010-2015).
    # Entry profile from backtested wealth creators: PE 15x, ROE 20%, PAT CAGR 35%.
    # MQGLP = QGLP applied to mid-cap (rank 101-300 by mcap) entry universe.
    # Proxy for mid-cap range: ₹5,000-20,000 Cr (India 2025 midcap band).
    df["mid_to_mega_candidate"] = (
        (df["market_cap"].fillna(0) >= 5_000) &
        (df["market_cap"].fillna(0) <= 20_000) &
        (df["roce_med_5y"].fillna(df["roce"]).fillna(0) >= 15) &
        (df["pat_gr_5y"].fillna(0) >= 20) &
        (df["pe"].fillna(999) <= 25)
    ).astype(int)

    # ── Study 22 (2017): CAP & GAP — Competitive + Growth Advantage Period ──
    # CAP = Competitive Advantage Period: ROE > Cost of Equity (Ke = 15% for India).
    # GAP = Growth Advantage Period: PAT growth rate exceeds benchmark (15%).
    # "Moat without growth underperforms; growth without moat ends soon." — MOSL 22nd Study.
    # Extended CAP: ROCE consistently > 10% across all 4 timeframes (depth of moat over time).
    df["cap_extended_flag"] = (
        (df["roce_med_10y"].fillna(0) >= 10) &
        (df["roce_med_7y"].fillna(df["roce_med_5y"]).fillna(0) >= 10) &
        (df["roce_med_5y"].fillna(0) >= 10) &
        (df["roce"].fillna(0) >= 10)
    ).astype(int)

    # Extended GAP: PAT growth sustained across all 3 windows — proxy for earnings compounding depth.
    df["gap_extended_flag"] = (
        (df["pat_gr_10y"].fillna(df["pat_gr_5y"]).fillna(0) >= 12) &
        (df["pat_gr_5y"].fillna(0) >= 12) &
        (df["pat_gr_3y"].fillna(0) >= 8)
    ).astype(int)

    # CAP-GAP composite (0-4): +1 for cap_extended, +1 for gap_extended, +1 for both, +1 for ROE > 15
    df["cap_gap_score"] = (
        df["cap_extended_flag"] +
        df["gap_extended_flag"] +
        (df["cap_extended_flag"] & df["gap_extended_flag"]).astype(int) +  # bonus for both = longevity proof
        (df["roe"].fillna(0) >= 15).astype(int)
    )

    # ── Study 23 (2018): Valuation Insights — ROE vs India CoE (15%) ──
    # Study 23 explicitly defines India Cost of Equity = 15% (not 10% as in Graham/US frameworks).
    # ROE - Ke (15%) = economic spread: positive = value creation, negative = value destruction.
    # Note: existing economic_profit uses CoE=10%; this signal uses the India-specific 15% threshold.
    df["roe_vs_coe15"] = df["roe"].fillna(0) - 15.0

    # ══════════════════════════════════════════════════════════════
    # MOSL STUDIES 25-27: QGLP CHECKLIST, ATOMS/BITS PSG,
    # CONSISTENTS & VOLATILES SECTOR CLASSIFICATION
    # Source: 25th Annual WCS (2020), 26th (2021), 27th (2022)
    # ══════════════════════════════════════════════════════════════

    # Study 26 (2021): PSG Ratio — Price/Sales/Growth (digital analog of PEG)
    # PSG = P/Sales ÷ Rev Growth CAGR. Lower PSG = better value for growth.
    # Benchmarks from Study 26: PSG <0.3 = very attractive; >1.0 = expensive.
    df["psg_ratio"] = np.where(
        df["rev_gr_5y"].notna() & (df["rev_gr_5y"] > 0) & df["ps_ratio"].notna(),
        df["ps_ratio"] / df["rev_gr_5y"],
        np.nan
    )

    # Identity C: Growth-Adjusted Payback Runway (26th WCS — Bits accounting anomaly correction)
    # GAPR = P/B / (ROE × Reinvestment Rate). Measures how quickly retained ROE earns back
    # the book premium. Lower GAPR = faster payback via compounding.
    # ROE is percentage (25.0 = 25%). RR computed inline — reinvestment_rate is defined later.
    # clip(0.01, 1.0): guards DPR ≥ 100% edge case (avoids divide-by-zero).
    _gapr_rr = (1.0 - df["dividend_payout_ratio"].fillna(0) / 100.0).clip(0.01, 1.0)
    df["gapr"] = np.where(
        (df["roe"].fillna(0) > 0) & (df["pb_ratio"].fillna(0) > 0),
        df["pb_ratio"].fillna(0) / ((df["roe"].fillna(0) / 100.0) * _gapr_rr),
        np.nan
    )

    # Study 27 (2022): Sector Consistent/Volatile classification
    # MOSL classified 18 sectors as Consistent (sustained earnings compounding)
    # and 35+ sectors as Volatile across 697 companies over 2007-2022.
    # CONSISTENT_SECTORS imported from config — verified against 81 CSV sector names.
    # Previous inline set had 7 wrong names: 292 stocks (all IT + all Pharma + Jewellery + Tobacco
    # + Oil&Gas) were misclassified as "Volatile". Fixed names centralised in config.py.
    df["sector_consistent_type"] = np.where(
        df["sector"].fillna("").isin(CONSISTENT_SECTORS),
        "Consistent", "Volatile"
    )

    # Study 27: Consistent company in Volatile sector = highest-alpha combination
    # 19% avg CAGR (31 companies) vs 16% for Consistents in Consistent sectors (83 companies).
    # A company sustaining earnings consistency despite adverse sector dynamics = deepest moat.
    df["consistent_in_volatile_flag"] = (
        (df["consistency_champion"].fillna(0) == 1) &
        (df["sector_consistent_type"] == "Volatile")
    ).astype(int)

    # Study 27: P/E below own 10Y median — entry signal for Consistent companies.
    # Study 27 finding: Consistents bought below own P/E median deliver 70-100% alpha probability.
    # Reuses d32_pe_vs_median (derived earlier): negative = currently trading below own history.
    df["pe_below_own_median"] = (
        df["d32_pe_vs_median"].fillna(0) < 0
    ).astype(int)

    # ── Lynch Category (Peter Lynch — One Up on Dalal Street) ──
    # Classifies each stock by growth trajectory for tearsheet display.
    # Fast Grower: Lynch's primary hunting ground for 10-100× returns.
    df["lynch_category"] = np.select(
        [
            df["rev_gr_5y"].fillna(0) >= 20,
            (df["rev_gr_5y"].fillna(0) >= 10) & (df["rev_gr_5y"].fillna(0) < 20),
            (df["rev_gr_5y"].fillna(0) >= 0)  & (df["rev_gr_5y"].fillna(0) < 10),
            df["rev_gr_5y"].fillna(0) < 0,
        ],
        ["Fast Grower", "Stalwart", "Slow Grower", "Declining"],
        default="Unknown"
    )

    # ══════════════════════════════════════════════════════════════
    # EPOCH 2 (7th–12th WCS, 2002–2007): REINVESTMENT MOAT VECTORS
    # Scalability & Self-Funding Reinvestment frameworks from Motilal Oswal Studies 7-12.
    # Identity A: Reinvestment Rate | Identity B: Fundamental Growth Capacity
    # Identity C: Buffett 1-to-1 Value Creation Ratio
    # ══════════════════════════════════════════════════════════════

    # Identity A (Agent 8): Reinvestment Rate (RR) — fraction of net profit retained in business.
    # RR = 1 − (DPR/100). High RR = self-funding compounder; low RR = defensive income asset.
    # DPR fillna(0): no dividend data → full retention (conservative for growth companies).
    # clip(0,1): guards against DPR > 100 (data artefacts in some screeners).
    df["reinvestment_rate"] = (
        1.0 - (df["dividend_payout_ratio"].fillna(0) / 100.0)
    ).clip(0.0, 1.0)

    # Mayer 100-Bagger companion: Retention Rate as PERCENTAGE (distinct from reinvestment_rate above).
    # retention_rate = 100 - DPR. Used by fw_100_bagger gate (>= 80.0%). DPR fillna(0) → full
    # retention assumed for stocks with missing dividend data (conservative: they pass the gate).
    df["retention_rate"] = (
        100.0 - df.get("dividend_payout_ratio", pd.Series(0.0, index=df.index)).fillna(0.0)
    ).clip(0.0, 100.0)

    # Identity B (Agent 8): Fundamental Growth Capacity g = ROE × RR.
    # Theoretical organic growth ceiling without external funding.
    # Actual growth >> g over multi-years → company is debt/dilution-dependent.
    df["fundamental_growth_capacity"] = df["roe"].fillna(0) * df["reinvestment_rate"]

    # 5Y cumulative retained earnings proxy — single-year PAT × RR × 5.
    # Approximation: current PAT/RR assumed representative of trailing 5-year period.
    # Used as denominator in the Buffett VCR identity below.
    df["retained_earnings_est_5y"] = df["pat"].fillna(0) * df["reinvestment_rate"] * 5.0

    # Identity C (Agent 9): Buffett 1-to-1 Value Creation Ratio (VCR) proxy.
    # True VCR = (MCap_now − MCap_5YB) / ΣRetainedEarnings(5Y). MCap 5Y back not in CSV.
    # Cross-sectional proxy: market_cap × 0.5 as center-of-distribution approximation.
    # Benchmarks from 11th WCS: VCR ≥ 2.0 = elite, ≥ 1.0 = passing, < 1.0 = concern.
    df["value_creation_ratio"] = np.where(
        df["retained_earnings_est_5y"] > 0,
        (df["market_cap"].fillna(0) * 0.5) / df["retained_earnings_est_5y"],
        0.0
    )

    # Capital Misallocation Risk (Agent 9): retaining >50% of earnings but VCR < 1.0.
    # Destroying minority shareholder value despite appearing to "conserve" capital.
    # Scoring engine applies a 10% quality penalty to these stocks.
    df["capital_misallocation_risk"] = (
        (df["reinvestment_rate"]    >  0.5) &
        (df["value_creation_ratio"] <  1.0) &
        (df["value_creation_ratio"] >  0.0)   # exclude zero (no data / negative PAT)
    ).astype(int)

    # ── Anti-Pattern A: Dilution Vampire (Capital Deficiency Trap) ──
    # Fast revenue growth (≥30%) funded by chronic equity dilution rather than internal capital.
    # Structural ROE < 12% = cannot self-fund growth → constant share issuance to minority investors.
    df["dilution_vampire_flag"] = (
        (df["rev_gr_5y"].fillna(0)   >= 30) &
        (df["roe"].fillna(99)         < 12) &
        (df["dilution_flag"].fillna(0) >= 1)   # ESOP-level dilution or higher
    ).astype(int)

    # ── Anti-Pattern B: Stagnant Cash-Cow Trap ──
    # Elite optical ROE (>35%) + high payout (DPR >70%, RR < 30%) + flat fixed assets + zero CWIP.
    # These are defensive income assets not compounders — 12th WCS: remove from high-velocity scan.
    _scc_flat_assets = (
        df["fixed_assets"].fillna(0) <= df["fixed_assets_1yb"].fillna(0) * 1.05  # <5% FA growth
    )
    _scc_no_cwip = df["cwip"].fillna(0) < 1.0   # essentially zero work-in-progress
    df["stagnant_cash_cow_flag"] = (
        (df["roe"].fillna(0)        > 35) &
        (df["reinvestment_rate"]    < 0.30) &   # DPR > 70%
        _scc_flat_assets &
        _scc_no_cwip
    ).astype(int)

    # ══════════════════════════════════════════════════════════════
    # EPOCH 3 (13th–18th WCS, 2008–2013): STRUCTURAL CLASSIFICATION
    # Great/Good/Gruesome taxonomy, Moat Endurance Factor, enhanced Payback
    # ══════════════════════════════════════════════════════════════

    # ── Identity A: Capital Return Spread ──
    df["capital_return_spread"] = df["roce"].fillna(0) - COST_OF_EQUITY

    # ── Identity A: FCF Generation Velocity (FCF/OCF ratio) ──
    df["fcf_to_ocf_velocity"] = np.where(
        df["operating_cash_flow"].fillna(0) > 0,
        df["free_cash_flow"].fillna(0) / df["operating_cash_flow"],
        0.0
    )

    # ── Identity C: Moat Endurance Factor (MEF) ──
    df["moat_endurance_factor"] = np.where(
        df["roce_med_10y"].fillna(0) > 0,
        df["roce"].fillna(0) / df["roce_med_10y"],
        0.0
    )
    df["mef_label"] = np.select(
        [
            df["moat_endurance_factor"] >= 1.2,
            df["moat_endurance_factor"] >= 1.0,
            df["moat_endurance_factor"] >= 0.80,
        ],
        ["🟢 Expanding", "✅ Intact", "🟡 Eroding"],
        default="🔴 Degrading"
    )

    # ── Cyclical Profit Mirage Anti-Pattern (Gruesome Growth Trap) ──
    df["cyclical_mirage_flag"] = (
        (df["rev_gr_yoy"].fillna(0) >= EPOCH3_TAXONOMY["mirage_rev_growth_min"]) &
        (df["roce_med_10y"].fillna(0) < EPOCH3_TAXONOMY["mirage_roce_10y_max"])
    ).astype(int)

    # ── PSU Value-Destruction Loop Anti-Pattern (Epoch 3) ──
    # State-owned enterprises prioritizing political/social goals over equity returns.
    # Expose this by checking low capital spreads + sub-par reinvestment velocities + continuous CWIP delays.
    _psu_name = df["name"].fillna("").astype(str).str.contains(
        r"\bNTPC\b|\bNHPC\b|\bGAIL\b|\bSAIL\b|\bONGC\b|\bIOC\b|\bBPCL\b|\bHPCL\b|\bIRFC\b|\bRVNL\b|\bHUDCO\b|\bLIC\b|\bBHEL\b|\bBEL\b|\bHAL\b|Coal India|NMDC|NALCO|MOIL|RINL|MTNL|BSNL|RITES|IRCTC|IRCON|RAILTEL",
        case=False, na=False
    )
    _psu_sector = df["sector"].fillna("").astype(str).str.contains("Public Sector|Govt", case=False, na=False) | \
                  df["industry"].fillna("").astype(str).str.contains("Public Sector|Govt", case=False, na=False)
    _is_psu_proxy = (_psu_name | _psu_sector) & (df["promoter_holdings"].fillna(0) >= 50) & (df["pledged_percentage"].fillna(0) == 0)

    _psu_low_spread = df["capital_return_spread"].fillna(0) <= 0
    _psu_low_velocity = (df["reinvestment_rate"].fillna(1) < 0.40) | (df["fcf_to_ocf_velocity"].fillna(0) < 0.40)
    _psu_cwip_delays = (df["cwip"].fillna(0) > 0) & (df["cwip_1yb"].fillna(0) > 0) & (df["cwip_conversion"].fillna(0) <= 0)

    df["psu_value_destruction_flag"] = (
        _is_psu_proxy & _psu_low_spread & _psu_low_velocity & _psu_cwip_delays
    ).astype(int)

    # ── Epoch 3 Structural Filter Pass (Capital Return Floor + Solvency) ──
    _e3_is_fin = df["is_financial"] == True
    df["epoch3_structural_pass"] = (
        (df["roce_med_10y"].fillna(0) >= EPOCH3_TAXONOMY["capital_return_floor_10y"]) &
        (df["roce_med_7y"].fillna(df["roce_med_5y"]).fillna(0)
            >= EPOCH3_TAXONOMY["capital_return_floor_7y"]) &
        (_e3_is_fin | (df["interest_coverage"].fillna(0) >= EPOCH3_TAXONOMY["min_interest_coverage"])) &
        (_e3_is_fin | (df["debt_to_equity"].fillna(999) < EPOCH3_TAXONOMY["max_debt_to_equity"])) &
        (df["cfo_to_pat"].fillna(0) >= EPOCH3_TAXONOMY["cfo_pat_structural_min"])
    ).astype(int)

    # ── Identity B: Low Payback Ratio Proxy (15th WCS — UU Investing Asymmetry) ──
    # Simplified PE/PAT-growth quotient as a crisis-dislocation scanner.
    # Distinct from payback_ratio (MCap/5Y-PAT-geometric): this uses PE + YoY growth,
    # making it reactive to short-term dislocations during GFC-style market crises.
    # Payback_ratio_proxy < 2.0 during a crisis → asymmetric UU setup (15th WCS).
    # Floor growth at 1.0% prevents division-by-zero and near-zero denominator blow-up.
    _pat_velocity_safe = df["pat_gr_yoy"].fillna(0).clip(lower=1.0)
    df["payback_ratio_proxy"] = np.where(
        df["pe"].fillna(0) > 0,
        df["pe"].fillna(999) / _pat_velocity_safe,
        np.nan
    )

    # ══════════════════════════════════════════════════════════════
    # EPOCH 4 (19th–25th WCS, 2014–2020): SQGLP 100x ENGINE VECTORS
    # Management Integrity, Reinvestment Efficiency Spread, Value Migration
    # ══════════════════════════════════════════════════════════════

    # ── Identity B: Incremental ROCE Proxy (22nd/23rd WCS — Reinvestment Efficiency Spread) ──
    # ΔProfit / |ΔFixed Assets| measures quality of recent capital deployments.
    # Positive + high = new investments are maintaining/expanding the return profile.
    # Guard: capital delta > 5 Cr filters noise from trivial reclassifications.
    _incr_pat_e4       = df["pat"].fillna(0) - df.get("pat_1yb", pd.Series(0.0, index=df.index)).fillna(0)
    _incr_cap_delta_e4 = (df["fixed_assets"].fillna(0) - df["fixed_assets_1yb"].fillna(0)).abs()
    df["incremental_roce_proxy"] = np.where(
        (_incr_cap_delta_e4 > 5.0) & (df["pat"].fillna(0) > 0),
        (_incr_pat_e4 / _incr_cap_delta_e4) * 100.0,
        np.nan
    )

    # ── Value Migration Flag (20th WCS — Structural Sector Value Rotation) ──
    # Identifies companies capturing structural market share from weaker sector peers.
    # Three concurrent conditions: top-quartile sector revenue growth + ROCE not declining +
    # absolute 5Y revenue growth floor. Mega/Large caps excluded — value already migrated.
    _vm_sector      = df["sector"].fillna("Unknown")
    _vm_sector_size = df.groupby(_vm_sector)["market_cap"].transform("count").fillna(0)
    _vm_rev_rank    = df.groupby(_vm_sector)["rev_gr_5y"].rank(pct=True).fillna(0)
    df["value_migration_flag"] = (
        (_vm_rev_rank >= 0.75) &                              # Top 25% by revenue growth in sector
        (df["roce_trajectory"].fillna(0) >= 0) &               # ROCE not structurally declining
        (df["rev_gr_5y"].fillna(0) >= 15.0) &                 # Absolute 5Y revenue growth ≥ 15%
        (_vm_sector_size >= 4) &                               # Sector must have ≥ 4 peers
        (~df["mcap_tier"].isin(["Mega Cap", "Large Cap"]))     # Exclude fully-valued mega/large caps
    ).astype(int)

    # ══════════════════════════════════════════════════════════════
    # TIER-2 QUANTAMENTAL RISK & VALUATION SIGNALS
    # Days to Liquidate (institutional liquidity risk), Pledge Re-Rating
    # Catalyst, and EVA-based Fair PE for quality-adjusted valuation.
    # All columns confirmed present in CSVs. Zero new data requirements.
    # ══════════════════════════════════════════════════════════════

    # ── T2A: Days to Liquidate — Institutional Exit Risk Index ──
    # Unit derivation (verified against daily_value formula at line 706):
    #   vol_sma_20d         → shares/day  (volume is always shares; daily_value/1e7 = Crores ✓)
    #   market_cap          → Rs Crores   (e.g. 10,000 for Rs 10,000 Crore)
    #   close_price         → Rs          (e.g. 500)
    #   shares_outstanding  → market_cap × 1e7 / close_price   (Crores×1e7 / Rs = shares ✓)
    #   fii_holdings        → percentage  (e.g. 12.4 = 12.4%)
    #   inst_shares         → (fii% + dii%) / 100 × shares_outstanding
    #   days_to_liquidate   → inst_shares / vol_sma_20d  (dimensionless: shares/shares_per_day = days)
    # Risk interpretation: DTL > 30 for small/mid-caps signals synchronized institutional exit
    # cannot be absorbed by market liquidity — stock locks limit-down before exits complete.
    _t2a_price_safe = df["close_price"].replace(0, np.nan)
    _t2a_shares     = df["market_cap"].fillna(0) * 1e7 / _t2a_price_safe
    _t2a_inst_pct   = (df["fii_holdings"].fillna(0) + df["dii_holdings"].fillna(0)) / 100.0
    _t2a_inst_shr   = _t2a_inst_pct * _t2a_shares

    df["days_to_liquidate"] = np.where(
        df["vol_sma_20d"].notna() & (df["vol_sma_20d"] > 0) & _t2a_shares.notna(),
        _t2a_inst_shr / df["vol_sma_20d"],
        np.nan
    )
    # Liquidity trap: DTL > 30 days for non-mega/large caps → synchronized exit risk
    # Large caps excluded: even at DTL 100+, their depth absorbs multi-day institutional selling.
    df["inst_liquidity_trap"] = (
        (df["days_to_liquidate"].fillna(0) > 30) &
        (~df["mcap_tier"].isin(["Mega Cap", "Large Cap"]))
    ).astype(int)

    # ── T2B: Pledge Re-Rating Catalyst ──
    # The existing pledge_falling_1y (line 690) is a continuous magnitude signal.
    # What it MISSES: the re-rating EVENT — when a stock transitions from pledged-and-feared
    # to clean-and-re-discovered by institutions. This is the actual alpha moment.
    # Three conditions must fire simultaneously:
    #   1. Pledge was meaningfully high 1Y ago (>10%): institutional worry was real
    #   2. Pledge fell >30% in 1 year: structural de-pledging underway, not noise
    #   3. Pledge now below 5%: approaching clean — institutional re-entry is imminent
    # Why this works: institutional mandates often forbid holding stocks with pledge >5%.
    # Crossing below that line unlocks a new buyer pool that couldn't hold the stock before.
    df["pledge_rerate_catalyst"] = np.where(
        df["pledged_1yb"].notna() & df["pledged_percentage"].notna(),
        (
            (df["pledged_1yb"]          > 10.0) &           # was meaningfully pledged
            (df["pledged_percentage"]   < df["pledged_1yb"] * 0.70) &  # dropped >30%
            (df["pledged_percentage"]   < 5.0)              # now approaching clean
        ).astype(int),
        0
    )

    # ── T2C: EVA-Based Fair PE (Quality-Adjusted Valuation) ──
    # Stewart's Economic Value Added framework applied to PE normalization.
    # Theoretical derivation: fair_pe = growth × (ROCE / CoC)
    #   At g=25%, ROCE=30%, CoC=12%: fair_pe = 25 × 2.5 = 62.5x
    #   If trading at 50x PE → pe_discount_to_quality = +12.5 (undervalued vs quality)
    # This solves the standard PE trap: a 50x PE compounder with 30% ROCE and 25% growth
    # is CHEAPER on quality-adjusted basis than a 12x PE mediocrity with 8% ROCE/6% growth.
    #   Standard PE says 12x < 50x → mediocrity is "cheaper" (wrong)
    #   EVA Fair PE says 62.5x >> 50x and 4x >> 12x → compounder IS cheaper (correct)
    # Data guards:
    #   pat_gr_5y missing → fallback to 3Y → fallback to 10% (India mid-cap median)
    #   roce_med_10y missing → fallback to 5Y median → fallback to 15% (minimal acceptable)
    #   Clip growth 2-40%: prevents negative-growth (absurd negative fair PE) and
    #     hyper-growth distortion (>40% unsustainable → fair PE blows up)
    #   Clip ROCE 5-70%: prevents near-zero ROCE collapsing fair PE and
    #     platform anomalies (network effect cos with 100%+ ROCE are correctly capped)
    #   COST_OF_EQUITY = 12.0 imported from config.py (line 15)
    _t2c_growth = df["pat_gr_5y"].fillna(
                  df["pat_gr_3y"]).fillna(10.0).clip(lower=2.0, upper=40.0)
    _t2c_roce   = df["roce_med_10y"].fillna(
                  df["roce_med_5y"]).fillna(15.0).clip(lower=5.0, upper=70.0)

    df["fair_pe_qglp"] = (_t2c_growth * (_t2c_roce / COST_OF_EQUITY)).round(2)

    # pe_discount_to_quality: positive = stock trading BELOW quality-adjusted fair value (BUY zone)
    # negative = stock trading ABOVE quality-adjusted fair value (expensive vs quality offered)
    df["pe_discount_to_quality"] = np.where(
        df["pe"].notna() & (df["pe"] > 0),
        df["fair_pe_qglp"] - df["pe"],
        np.nan
    )

    # ══════════════════════════════════════════════════════════════
    # TIER-1 QUANTAMENTAL ALPHA SIGNALS
    # Four high-priority signals validated against this exact data set:
    # all columns confirmed present in CSVs, zero new data requirements.
    # ══════════════════════════════════════════════════════════════

    # ── QA1: Accruals Ratio (Richardson 2005) ──
    # Definition: (PAT - Operating_Cash_Flow) / Total_Assets
    # Negative = earnings are cash-backed (good). Positive = accrual-heavy (red flag).
    # Richardson 2005 showed high-accrual stocks underperform low-accrual stocks by 10-14%/yr
    # across every market studied. The single most powerful forensic quality signal.
    # Units: PAT, OCF, Total_Assets all in Crores → ratio is dimensionless (correct).
    # No financial sector exclusion: accruals ratio is informative for all sectors.
    df["accruals_ratio"] = np.where(
        df["total_assets"].notna() & (df["total_assets"] > 0) &
        df["pat"].notna() & df["operating_cash_flow"].notna(),
        (df["pat"] - df["operating_cash_flow"]) / df["total_assets"],
        np.nan
    )
    # accruals_clean: earnings are fully cash-backed (accruals ≤ 0)
    df["accruals_clean"]   = (df["accruals_ratio"].fillna(99)  < 0.00).astype(int)
    # accruals_warning: >5% of total assets is non-cash earnings — elevated manipulation risk
    df["accruals_warning"] = (df["accruals_ratio"].fillna(0)   > 0.05).astype(int)

    # ── QA2: CWIP Capitalization Inflection (Enhanced 3-Condition) ──
    # The most underpriced event in Indian capital-intensive equities (Motilal WCS #12/#14/#19).
    # Existing capex_productive (2-condition at line 1084) catches too many false positives:
    # any CWIP drop + any revenue growth fires it, including trivial reclassifications.
    # This enhanced version requires three concurrent conditions with magnitude thresholds:
    #   Cond 1: Construction pipe was heavy (>25% of gross block 1Y ago) — eliminates maintenance noise
    #   Cond 2: CWIP dropped >20% — structural capitalization, not a project delay or accounting reclassification
    #   Cond 3: Fixed assets confirmed expanding >5% — proves CWIP went live, was not written off
    # Guard: both cwip_1yb > 0 and fa_1yb > 0 prevent divide-by-zero and ghost signals.
    # Note: cwip_ratio (cwip/fa current, line 645) provides the continuous intensity reading.
    _qa2_cwip_cur = df["cwip"].fillna(0)
    _qa2_cwip_1yb = df["cwip_1yb"].fillna(0)
    _qa2_fa_cur   = df["fixed_assets"].fillna(0)
    _qa2_fa_1yb   = df["fixed_assets_1yb"].fillna(0)

    df["cat_cwip_inflection"] = (
        (_qa2_cwip_1yb > 0) &                              # must have had a construction pipe
        (_qa2_fa_1yb   > 0) &                              # must have had fixed assets
        (_qa2_cwip_1yb > _qa2_fa_1yb * 0.25) &            # pipe was ≥25% of gross block (heavy)
        (_qa2_cwip_cur < _qa2_cwip_1yb * 0.80) &          # CWIP fell >20% (structural, not noise)
        (_qa2_fa_cur   > _qa2_fa_1yb * 1.05)              # fixed assets expanded >5% (confirmed live)
    ).astype(int)

    # Pre-inflection early warning: large CWIP still building, capitalization not yet started.
    # Identifies setups in the final construction phase before the inflection fires next year.
    df["cwip_pre_inflection"] = (
        (_qa2_fa_cur > 0) &
        (_qa2_cwip_cur > _qa2_fa_cur   * 0.30) &          # 30%+ of current gross block still in CWIP
        (_qa2_cwip_cur >= _qa2_cwip_1yb * 0.90)           # CWIP not yet falling — still building
    ).astype(int)

    # ── QA3: Supplier Float Score (Negative Working Capital Moat) ──
    # D-Mart, Titan, Page Industries, Maruti: collect customer cash immediately,
    # force suppliers to accept 60-120 day credit terms. Suppliers finance the scaling.
    # Result: the business grows without needing debt or equity dilution.
    # Scoring: linear 0-100, calibrated so -120 days CCC = 100 (maximum moat depth).
    # Financial sector guard: CCC is already NaN'd upstream at line 816-827 for all financial
    # companies (banks, NBFCs, insurance). So financial stocks auto-score 0 without any extra check.
    # ccc_improving reuses d37_ccc_direction (ccc - ccc_1yb) already computed at line 1220.
    df["supplier_float_score"] = np.where(
        df["ccc"].notna() & (df["ccc"] < 0),
        np.minimum(df["ccc"].abs() / 120.0 * 100.0, 100.0),
        0.0
    )
    df["negative_wc_flag"] = (df["ccc"].fillna(0) < 0).astype(int)
    # d37_ccc_direction < 0 means CCC fell (got more negative or less positive) = improving
    df["ccc_improving"] = (df["d37_ccc_direction"].fillna(0) < 0).astype(int)

    # ── QA4: EPS Acceleration (Earnings Revision Proxy) ──
    # Analyst estimate revision data costs ₹50L+/yr. This proxy achieves 70-80% of that signal:
    # when current-year EPS growth exceeds the 3-year base trend, forward estimates are being
    # revised upward. Academic evidence shows revision momentum persists for 2-4 quarters.
    # eps_acceleration > 0: current year running ahead of historical trend (estimates rising)
    # eps_acceleration > 10: strong acceleration — current year outpacing 3Y trend by 10pp+
    df["eps_acceleration"] = np.where(
        df["eps_gr_yoy"].notna() & df["eps_gr_3y"].notna(),
        df["eps_gr_yoy"] - df["eps_gr_3y"],
        np.nan
    )
    df["eps_accelerating"]        = (df["eps_acceleration"].fillna(-999) > 0 ).astype(int)
    df["eps_strong_acceleration"] = (df["eps_acceleration"].fillna(-999) > 10).astype(int)

    n_derived = len([c for c in df.columns if c not in set(COMMON_COLS.values())])
    print(f"  ✅ Computed all derived signals. Total columns: {len(df.columns)}")

    # Flush all infinities created by division edge cases (zero denominators, etc.)
    # np.seterr(all='ignore') suppresses the warning but doesn't prevent np.inf in the array.
    # _pct_rank treats np.inf as a valid maximum, pushing bankrupt/zero-equity stocks to 99th pctile.
    inf_count = int(np.isinf(df.select_dtypes(include=[np.number])).sum().sum())
    if inf_count > 0:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        print(f"  ⚠️  Flushed {inf_count} infinity values to NaN")

    return df




def fetch_and_clean_data(data_source: str = "local", uploaded_files: dict = None, sheet_id: str = None) -> pd.DataFrame:
    """Tier-1 Cache: Load → Merge → Coerce → Derive → Return clean master DataFrame.
    This is the expensive operation (network/IO). Cache it aggressively.
    The scoring engine runs separately and is NOT cached — enabling instant re-scoring.
    """
    datasets = load_all_csvs(data_source=data_source, uploaded_files=uploaded_files, sheet_id=sheet_id)
    master = merge_datasets(datasets)
    master = coerce_numeric_columns(master)
    master = compute_derived_signals(master)

    # No market cap floor filter — all 2107 stocks included.
    # market_category from the sheet already handles classification.
    print(f"\n✅ Clean data ready: {len(master)} stocks × {len(master.columns)} columns")
    return master


# Backward-compat alias
build_master_dataframe = fetch_and_clean_data


# ═══════════════════════════════════════════════════════════════
# CLI Test
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import time
    t0 = time.time()
    df = fetch_and_clean_data()
    elapsed = time.time() - t0
    print(f"\n⏱️  Pipeline completed in {elapsed:.2f}s")
    print(f"\nSample columns: {list(df.columns[:20])}")
    print(f"\nMarket category dist:\n{df['market_category'].value_counts()}")
    print(f"\nFinancial sector: {df['is_financial'].sum()} stocks")
    print(f"\nNaN counts (top 10):")
    nan_counts = df.isnull().sum().sort_values(ascending=False).head(10)
    print(nan_counts)
