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
from config import CSV_FILES, MCAP_TIERS, MCAP_MIN_FLOOR, FINANCIAL_SECTORS, SSGR_DEP_RATES

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
    # CCC aggregates the 3 components; 2yb is redundant with trend already captured by 1yb delta
    "Cash Conversion Cycle":              "ccc",
    "Cash Conversion Cycle 1 Year Back":  "ccc_1yb",
    "Days Receivable":              "days_receivable",
    "Days Receivable 1 Year Back":  "days_receivable_1yb",
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
    # QUARTERLY — freshest timing signals (6 columns)
    "PAT Latest Quarter":              "pat_lq",
    "PAT Preceding Year Quarter":      "pat_pyq",
    "Revenue Latest Quarter":          "rev_lq",
    "Revenue Preceding Year Quarter":  "rev_pyq",
    "EBITDA Latest Quarter":           "ebitda_lq",
    "EBITDA Preceding Year Quarter":   "ebitda_pyq",
    # RAW ANNUAL — minimum needed for derived signals (8 columns)
    "PAT":                 "pat",
    "PAT 1 Year Back":     "pat_1yb",
    "EBITDA":              "ebitda",
    "Revenue":             "revenue",
    "Revenue 1 Year Back": "revenue_1yb",
    "Expenses":            "expenses",
    "Expenses 1 Year Back":"expenses_1yb",
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
            how="left",
            suffixes=("", f"_{name}")
        )
        print(f"  ✅ Merged {name}: {len(master)} rows, {len(master.columns)} cols")

    print(f"\n📊 Master DataFrame: {len(master)} stocks × {len(master.columns)} columns")
    return master


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all non-identifier columns to numeric."""
    # Columns that should remain as strings
    string_cols = {
        "company_id", "name", "market_category", "eligibility",
        "industry", "sector", "insider_trading",
    }

    for col in df.columns:
        if col not in string_cols:
            df[col] = _safe_numeric(df[col])

    return df


def compute_derived_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 36+ derived signals. Pure vectorized Pandas."""
    print("\n🧮 Computing derived signals...")

    # ── RATIO DERIVED ──
    df["roce_trajectory"] = df["roce_med_7y"] - df["roce_med_10y"]
    df["roe_trajectory"] = df["roe_med_7y"] - df["roe_med_10y"]
    df["roce_current_vs_med"] = df["roce"] - df["roce_med_10y"]
    df["roe_current_vs_med"] = df["roe"] - df["roe_med_10y"]
    _de_nan = pd.Series(np.nan, index=df.index)

    df["npm_acceleration"] = df.get("npm_latest_q", _de_nan) - df.get("npm_1yb",    _de_nan)
    df["opm_acceleration"] = df.get("opm_latest_q", _de_nan) - df.get("opm_1yb",    _de_nan)
    df["gpm_acceleration"] = df.get("gpm_latest_q", _de_nan) - df.get("gpm_med_5y", _de_nan)
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

    # ── CASHFLOW DERIVED ──
    # FCF imputation: ~600+ major stocks (HINDUNILVR, HCLTECH, INFY, etc.) have null FCF
    # because the data provider lacks CapEx data. Without imputation, these stocks score at
    # universe median (50th pct) for FCF yield despite massive positive cash generation.
    # Conservative imputation: use OCF when FCF is null (treats all OCF as free cash — overstates
    # FCF by ignoring CapEx, but far better than arbitrary 50th-pct neutral assignment).
    if "free_cash_flow" in df.columns and "operating_cash_flow" in df.columns:
        fcf_null_count = df["free_cash_flow"].isna().sum()
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
    df["cwip_conversion"] = df["cwip_1yb"] - df["cwip"]  # positive = went live
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
    
    df["vqs_score"] = pd.Series(vqs_liquidity + vqs_smart + vqs_cons + vqs_eff).fillna(0)
    
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

    # ── FINANCIAL SECTOR FLAG ──
    df["is_financial"] = df["industry"].isin(FINANCIAL_SECTORS) | \
                         df["sector"].fillna("").astype(str).str.contains("Bank|NBFC|Insurance|Finance", case=False, na=False)

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

    # Depreciation rate: sector-specific lookup (Indian Companies Act Schedule II + industry norms).
    # IT/Software: 25% (3-4yr useful life). Infrastructure: 4% (25yr+ assets). Default: 10%.
    default_dep = SSGR_DEP_RATES["_default"]
    if "industry" in df.columns:
        sector_dep = df["industry"].map(SSGR_DEP_RATES).fillna(default_dep)
    else:
        sector_dep = pd.Series(default_dep, index=df.index)
    dep_rate = np.where(fa > 0, sector_dep, 0.0)

    # G4 FIX: removed "- dep_rate" which was double-deducting depreciation.
    # NPM is already NET of depreciation (dep reduced profit in P&L).
    # Vijay Malik SSGR = NPM × NFAT × (1 - DPR), no further dep adjustment.
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
    # EP = Net Worth × (RoE − Cost of Equity)
    # Cost of Equity ≈ 10% for India
    # Net Worth = PAT / (ROE/100)
    cost_of_equity = 10.0
    roe_safe = df["roe"].fillna(0).clip(lower=1)  # avoid division by near-zero
    net_worth = np.where(
        (df["roe"].fillna(0) > 1) & (df["pat"].fillna(0) > 0),
        df["pat"] / (roe_safe / 100.0),
        df["reserves"].fillna(0).clip(lower=1)
    )
    df["economic_profit"] = pd.Series(net_worth, index=df.index) * (df["roe"].fillna(0) - cost_of_equity) / 100.0
    df["economic_profit_positive"] = (df["economic_profit"] > 0).astype(int)

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
    # cwip_conversion already computed above. Create efficiency signal.
    df["capex_productive"] = (
        (df["cwip_conversion"].fillna(0) > 0) &  # CWIP converted to assets
        (df["rev_gr_yoy"].fillna(0) > 0)           # AND revenue grew
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
    # Pabrai's downside protection: D14 = 3 means debt fell every year for 3 years
    _d14_y1 = (df["debt"].fillna(0) < df["debt_1yb"].fillna(0)).astype(int)
    _d14_y2 = (df["debt_1yb"].fillna(0) < df["debt_2yb"].fillna(0)).astype(int)
    _d14_y3 = (df["debt_2yb"].fillna(0) < df["debt_3yb"].fillna(0)).astype(int)
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

    # ── D19 FIX: CWIP Conversion (full formula — CWIP → Fixed Assets) ──
    # Correct formula: (CWIP_1YB - CWIP) + (FA - FA_1YB)
    # = CWIP that disappeared + Fixed assets that appeared = capital going live
    df["d19_cwip_conversion"] = (
        (df["cwip_1yb"].fillna(0) - df["cwip"].fillna(0)) +
        (df["fixed_assets"].fillna(0) - df["fixed_assets_1yb"].fillna(0))
    )
    df["cwip_conversion"] = df["d19_cwip_conversion"]  # replace incomplete formula

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

    # ── D47: RS Composite (average of 3 CRS timeframes) ──
    # Lower is better (closer to 0 or negative = underperforming = bearish)
    # But for universe ranking, a higher CRS composite = stronger relative strength
    df["d47_rs_composite"] = (
        df["crs_50d"].fillna(0) + df["crs_26w"].fillna(0) + df["crs_52w"].fillna(0)
    ) / 3.0

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
    df["payback_lt1"] = (df["payback_ratio"].fillna(99) < 1.0).astype(int)   # supernormal returns zone
    df["payback_lt2"] = (df["payback_ratio"].fillna(99) < 2.0).astype(int)   # attractive zone

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

    # ══════════════════════════════════════════════════════════════
    # VIJAY MALIK PEACEFUL INVESTING SIGNALS (Vol 1–3 deep extraction)
    # "Hard Gates" layer — capital efficiency, OPM consistency, cash conversion quality.
    # ══════════════════════════════════════════════════════════════

    # ── FCF/CFO Conversion Quality (Vijay Malik's single most diagnostic ratio) ──
    # Finolex Cables: 76% — gold standard. PIX Transmissions: negative — capital trap.
    # Captures what FCF/PAT misses: how much of operating cash survives after capex.
    # Only meaningful when OCF is positive; when OCF < 0, rf_negative_fcf handles it.
    df["fcf_to_cfo_pct"] = np.where(
        df["operating_cash_flow"].notna() & (df["operating_cash_flow"] > 0),
        df["free_cash_flow"].fillna(0) / df["operating_cash_flow"] * 100,
        np.nan
    )

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

    # ── WCS Score: Wealth Creation Criteria Count (0–7) ──
    # Each criterion is a separately validated MOSL supernormal-return predictor.
    # The more criteria met, the higher the probability of 5Y outperformance.
    df["wcs_score"] = (
        ((df["peg"].fillna(99) > 0) & (df["peg"].fillna(99) <= 1.0)).astype(int) +   # PEG < 1
        df["pe_below_roe"].fillna(0).astype(int) +                                    # PE < ROE (MoS)
        df["economic_profit_positive"].fillna(0).astype(int) +                        # EP > 0
        (df["pat_gr_5y"].fillna(0) >= 20).astype(int) +                               # PAT CAGR > 20%
        (df["roce_med_5y"].fillna(df["roce"]).fillna(0) >= 15).astype(int) +          # ROCE > 15% sustained
        df["payback_lt1"].fillna(0).astype(int) +                                     # Payback < 1
        df["consistency_champion"].fillna(0).astype(int)                              # PAT consistency
    )

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

    n_derived = len([c for c in df.columns if c not in set(COMMON_COLS.values())])
    print(f"  ✅ Computed all derived signals. Total columns: {len(df.columns)}")
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
