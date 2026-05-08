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
from config import CSV_FILES, MCAP_TIERS, MCAP_MIN_FLOOR, FINANCIAL_SECTORS

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
    # MOAT — ROCE
    "ROCE Median 10 Years": "roce_med_10y",
    "ROCE Median 7 Years": "roce_med_7y",
    "ROCE Median 5 Years": "roce_med_5y",
    "ROCE": "roce",
    "ROCE 1 Year Back": "roce_1yb",
    # CAPITAL EFFICIENCY — ROE
    "ROE Median 10 Years": "roe_med_10y",
    "ROE Median 7 Years": "roe_med_7y",
    "ROE Median 5 Years": "roe_med_5y",
    "ROE": "roe",
    "ROE 1 Year Back": "roe_1yb",
    # CASH QUALITY
    "CFO To PAT": "cfo_to_pat",
    "CFO To EBITDA": "cfo_to_ebitda",
    # MARGINS
    "NPM Median 5 Years": "npm_med_5y",
    "NPM Median 3 Years": "npm_med_3y",
    "NPM": "npm",
    "NPM Latest Quarter": "npm_latest_q",
    "NPM 1 Year Back": "npm_1yb",
    "OPM Median 5 Years": "opm_med_5y",
    "OPM Latest Quarter": "opm_latest_q",
    "OPM 1 Year Back": "opm_1yb",
    "GPM Median 5 Years": "gpm_med_5y",
    # VALUATION
    "PEG": "peg",
    "EV To EBITDA": "ev_ebitda",
    "EV To EBITDA 1 Year Back": "ev_ebitda_1yb",
    "Price To Earnings Median 10 Years": "pe_med_10y",
    "Price To Earnings": "pe",
    "Industry PE Median": "industry_pe",
    # EFFICIENCY
    "Cash Conversion Cycle": "ccc",
    "Cash Conversion Cycle 1 Year Back": "ccc_1yb",
    "Asset Turnover": "asset_turnover",
    "Days Receivable": "days_receivable",
    "Inventory Turnover Ratio": "inventory_turnover",
    "Inventory Turnover Ratio 1 Year Back": "inventory_turnover_1yb",
    # HARD GATES
    "Debt To Equity": "debt_to_equity",
    "Debt To Equity 1 Year Back": "debt_to_equity_1yb",
    "Debt To Equity 2 Years Back": "debt_to_equity_2yb",
    "Debt To Equity 3 Years Back": "debt_to_equity_3yb",
    "Current Ratio": "current_ratio",
    "Current Ratio 1 Year Back": "current_ratio_1yb",
    "ROA": "roa",
    "Equity Shares 1 Year Back": "equity_shares_1yb",
}

INCOME_COLS = {
    # GROWTH
    "PAT Growth 5 Years": "pat_gr_5y",
    "PAT Growth 10 Years": "pat_gr_10y",
    "PAT Growth 3 Years": "pat_gr_3y",
    "PAT Growth YoY": "pat_gr_yoy",
    "EPS Growth 5 Years": "eps_gr_5y",
    "EPS Growth 3 Years": "eps_gr_3y",
    "EPS Growth YoY": "eps_gr_yoy",
    "Revenue Growth 5 Years": "rev_gr_5y",
    "Revenue Growth 10 Years": "rev_gr_10y",
    "Revenue Growth 3 Years": "rev_gr_3y",
    "Revenue Growth YoY": "rev_gr_yoy",
    "EBITDA Growth 5 Years": "ebitda_gr_5y",
    "EBITDA Growth 3 Years": "ebitda_gr_3y",
    # QUARTERLY
    "PAT Latest Quarter": "pat_lq",
    "PAT Preceding Year Quarter": "pat_pyq",
    "Revenue Latest Quarter": "rev_lq",
    "Revenue Preceding Year Quarter": "rev_pyq",
    "EBITDA Latest Quarter": "ebitda_lq",
    "EBITDA Preceding Year Quarter": "ebitda_pyq",
    # RAW ANNUAL
    "PAT": "pat",
    "PAT 1 Year Back": "pat_1yb",
    "EBITDA": "ebitda",
    "EBITDA 1 Year Back": "ebitda_1yb",
    "Revenue": "revenue",
    "Revenue 1 Year Back": "revenue_1yb",
    "Expenses": "expenses",
    "Expenses 1 Year Back": "expenses_1yb",
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
    "Promoter Holdings": "promoter_holdings",
    "Change In Promoter Holdings Latest Quarter": "change_promoter_lq",
    "Change In Promoter Holdings 1 Year": "change_promoter_1y",
    "Pledged Percentage": "pledged_percentage",
    "Pledged Percentage 1 Quarter Back": "pledged_1qb",
    "Pledged Percentage 1 Year Back": "pledged_1yb",
    "FII Holdings": "fii_holdings",
    "Change In FII Holdings Latest Quarter": "change_fii_lq",
    "Change In FII Holdings 1 Year": "change_fii_1y",
    "Change In DII Holdings Latest Quarter": "change_dii_lq",
    "Change In DII Holdings 1 Year": "change_dii_1y",
    "Insider Trading": "insider_trading",
    "Promoter Holdings (Gate Use)": "promoter_gate",
}

TECHNICAL_COLS = {
    "Last VSTOP Change 14W 2.5": "last_vstop_change",
    "VSTOP 14W 2.5": "vstop_value",
    "CRS Vs Nifty 500 50D": "crs_50d",
    "CRS Vs Nifty 500 52W": "crs_52w",
    "CRS Vs Nifty 500 26W": "crs_26w",
    "ADX 14W": "adx_14w",
    "SMA 200D": "sma_200d",
    "RSI 14D": "rsi_14d",
    "Returns Vs Nifty 500 3M": "ret_vs_n500_3m",
    "Returns Vs Nifty 500 6M": "ret_vs_n500_6m",
    "Returns Vs Industry 1Y": "ret_vs_industry_1y",
    "52WH Distance": "dist_52wh",
    "52WH Distance Days": "dist_52wh_days",
    "13WH Distance": "dist_13wh",
    "Breakout Window": "breakout_window",
    "Volume": "volume",
    "Volume SMA 20D": "vol_sma_20d",
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
    # Row 0 = section headers (emojis), Row 1 = actual column names
    df = pd.read_csv(filepath, header=1, low_memory=False)

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
        "industry", "sector", "insider_trading", "promoter_gate",
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
    df["npm_acceleration"] = df["npm_latest_q"] - df["npm_1yb"]
    df["opm_acceleration"] = df["opm_latest_q"] - df["opm_1yb"]
    df["pe_discount"] = np.where(
        df["pe_med_10y"].notna() & (df["pe_med_10y"] != 0),
        (df["pe_med_10y"] - df["pe"]) / df["pe_med_10y"] * 100,
        np.nan
    )
    df["ev_compression"] = df["ev_ebitda_1yb"] - df["ev_ebitda"]
    df["de_slope_3y"] = df["debt_to_equity"] - df["debt_to_equity_3yb"]
    df["dilution_flag"] = np.where(
        df["equity_shares"].notna() & df["equity_shares_1yb"].notna(),
        (df["equity_shares"] > df["equity_shares_1yb"]).astype(int),
        0
    )

    # ── INCOME DERIVED ──
    df["pat_acceleration"] = df["pat_gr_3y"] - df["pat_gr_5y"]
    df["rev_acceleration"] = df["rev_gr_3y"] - df["rev_gr_5y"]
    df["eps_vs_pat_delta"] = df["eps_gr_5y"] - df["pat_gr_5y"]
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
    df["vstop_fresh"] = (df["last_vstop_change"] <= 30).astype(int)
    df["above_sma200"] = (df["close_price"] > df["sma_200d"]).astype(int)
    df["vstop_green"] = (df["close_price"] > df["vstop_value"]).astype(int)

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

    # ── MARKET CAP TIER ──
    df["mcap_tier"] = np.select(
        [
            df["market_cap"] >= MCAP_TIERS["Tier A"]["min"],
            df["market_cap"] >= MCAP_TIERS["Tier B"]["min"],
            df["market_cap"] >= MCAP_MIN_FLOOR,
        ],
        ["Tier A", "Tier B", "Tier C"],
        default="Below Floor"
    )

    # ── FINANCIAL SECTOR FLAG ──
    df["is_financial"] = df["industry"].isin(FINANCIAL_SECTORS) | \
                         df["sector"].str.contains("Bank|NBFC|Insurance|Finance", case=False, na=False)

    # Net debt negative flag (fortress balance sheet)
    df["net_debt_negative"] = (df["net_debt"] < 0).astype(int)

    n_derived = len([c for c in df.columns if c not in set(COMMON_COLS.values())])
    print(f"  ✅ Computed all derived signals. Total columns: {len(df.columns)}")
    return df


def build_master_dataframe(data_source: str = "local", uploaded_files: dict = None, sheet_id: str = None) -> pd.DataFrame:
    """Complete pipeline: Load → Merge → Coerce → Derive → Return."""
    datasets = load_all_csvs(data_source=data_source, uploaded_files=uploaded_files, sheet_id=sheet_id)
    master = merge_datasets(datasets)
    master = coerce_numeric_columns(master)
    master = compute_derived_signals(master)

    # Filter out stocks below minimum market cap floor
    before = len(master)
    master = master[master["market_cap"] >= MCAP_MIN_FLOOR].reset_index(drop=True)
    filtered = before - len(master)
    if filtered > 0:
        print(f"\n🚫 Filtered {filtered} stocks below ₹{MCAP_MIN_FLOOR} Cr market cap floor")

    print(f"\n✅ Master DataFrame ready: {len(master)} stocks × {len(master.columns)} columns")
    return master


# ═══════════════════════════════════════════════════════════════
# CLI Test
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import time
    t0 = time.time()
    df = build_master_dataframe()
    elapsed = time.time() - t0
    print(f"\n⏱️  Pipeline completed in {elapsed:.2f}s")
    print(f"\nSample columns: {list(df.columns[:20])}")
    print(f"\nTier distribution:\n{df['mcap_tier'].value_counts()}")
    print(f"\nFinancial sector: {df['is_financial'].sum()} stocks")
    print(f"\nNaN counts (top 10):")
    nan_counts = df.isnull().sum().sort_values(ascending=False).head(10)
    print(nan_counts)
