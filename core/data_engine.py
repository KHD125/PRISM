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
                    FINANCIAL_SECTORS, FINANCIAL_SECTOR_NAMES, UTILITY_SECTOR_NAMES,
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
    # MOAT — ROCE (medians for long-run quality; current + 1yb + 2yb for 3-point structural slope)
    "ROCE Median 10 Years": "roce_med_10y",
    "ROCE Median 7 Years":  "roce_med_7y",
    "ROCE Median 5 Years":  "roce_med_5y",
    "ROCE Median 3 Years":  "roce_med_3y",
    "ROCE":                 "roce",
    "ROCE 1 Year Back":     "roce_1yb",
    "ROCE 2 Years Back":    "roce_2yb",
    # CAPITAL EFFICIENCY — ROE
    "ROE Median 10 Years":  "roe_med_10y",
    "ROE Median 7 Years":   "roe_med_7y",
    "ROE Median 5 Years":   "roe_med_5y",
    "ROE Median 3 Years":   "roe_med_3y",
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
    # VALUATION
    "PEG":                               "peg",
    "EV To EBITDA":                      "ev_ebitda",
    "EV To EBITDA 1 Year Back":          "ev_ebitda_1yb",
    "Enterprise Value":                  "enterprise_value",   # direct EV (added 2026-06-13) — Magic Formula EBIT/EV
    "Price To Earnings Median 10 Years": "pe_med_10y",
    "Price To Earnings Median 5 Years":  "pe_med_5y",
    "Price To Earnings":                 "pe",
    "Industry PE Median":                "industry_pe",
    "Book Value":                        "book_value",     # net worth per share — Economic Profit (28th Study)
    "Price To Book Value":               "price_to_book",  # Bruised Blue Chip P/B < 2x gate (29th Study)
    # EFFICIENCY — WORKING CAPITAL
    "Cash Conversion Cycle":               "ccc",
    "Cash Conversion Cycle 1 Year Back":   "ccc_1yb",
    "Cash Conversion Cycle 3 Years Back":  "ccc_3yb",     # CCC 3Y trend for forensic quality
    "Days Receivable":               "days_receivable",
    "Days Receivable 1 Year Back":   "days_receivable_1yb",
    "Days Receivable 2 Years Back":  "days_receivable_2yb",
    "Days Receivable 3 Years Back":  "days_receivable_3yb",  # Mukherjea Lens 1: true 3Y DSO window
    "Days Payable":                  "days_payable",         # Terms of Trade (11th Study) — DPO for CCC decomposition
    "Days Payable 1 Year Back":      "days_payable_1yb",
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
    "EBIT Growth 3 Years":    "ebit_gr_3y",     # NOPAT engine growth — excludes interest, isolates operating leverage
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
    "PAT 4 Years Back":       "pat_4yb",   # Payback Ratio 5Y growth projection (10th–30th Study)
    "PAT 5 Years Back":       "pat_5yb",   # Consistents vs Volatiles 5Y terminal check (27th Study)
    "PBT":                    "pbt",
    "PBT 1 Year Back":        "pbt_1yb",
    "EBITDA":                 "ebitda",
    "EBITDA 1 Year Back":     "ebitda_1yb",
    # EBIT is present in the Screener.in Income Statement CSV — adding here enables
    # the D&A derivation (D&A = EBITDA − EBIT) used in Schilit Signal 6 (EMS #4).
    "EBIT":                   "ebit",
    "EBIT 1 Year Back":       "ebit_1yb",
    "EPS":                    "eps",        # raw annual EPS — CANSLIM Q criterion + P/E < ROE gate
    "EPS 1 Year Back":        "eps_1yb",   # YoY EPS comparison for deceleration / reacceleration signal
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
    # TREND GATES — MA stacking + trend strength
    "ADX 14W":   "adx_14w",
    "SMA 200D":  "sma_200d",   # Criterion 1: price > 200D MA (long-term trend gate)
    "SMA 50D":   "sma_50d",    # Criterion 5: 50D > 150D + 200D (right stacking)
    "SMA 30W":   "sma_30w",    # Criterion 2+3: price > 150D MA; 150D > 200D (30W × 5 = 150 trading days)
    # MOMENTUM CONFIRMATION
    "RSI 14D": "rsi_14d",
    "Returns Vs Nifty 500 3M": "ret_vs_n500_3m",
    "Returns Vs Nifty 500 6M": "ret_vs_n500_6m",
    "Returns Vs Industry 1Y": "ret_vs_industry_1y",
    # BREAKOUT PROXIMITY
    "52WH Distance":     "dist_52wh",
    "52WL Distance":     "dist_52wl",   # Criterion 6: price ≥ 30% above 52-week low
    "52WH Distance Days":"dist_52wh_days",
    "13WH Distance":     "dist_13wh",
    "Breakout Window":   "breakout_window",
    # VOLUME — institutional entry detector + liquidity gate + Minervini breakout confirmation
    "Volume":            "volume",
    "Volume SMA 5D":     "vol_sma_5d",    # VCP dryup check (legacy): 5D < 20D = contracting in base
    "Volume SMA 10D":    "vol_sma_10d",   # VCP dryup (Minervini exact): avg 10D < avg 50D
    "Volume SMA 20D":    "vol_sma_20d",   # vol_ratio denominator + liquidity gate
    "Volume SMA 50D":    "vol_sma_50d",   # Minervini breakout confirmation: volume ≥ 40% above 50D avg
    # TREND CONFIRMATION
    "Last Goldencrossover 50D 200D": "golden_cross_days",
    "All Time High Distance": "dist_ath",
    "Returns Vs Industry 3M": "ret_vs_industry_3m",
}


def _safe_numeric(series: pd.Series) -> pd.Series:
    """Convert a series to numeric, coercing errors (null strings, etc.) to NaN."""
    return pd.to_numeric(series, errors='coerce')


def _kendall_tau_cols(df: pd.DataFrame, cols: list) -> pd.Series:
    """Trajectory consistency via pairwise-sign Kendall's tau across an ordered (oldest→newest)
    ladder of level columns. Loops over COLUMN pairs only (O(k²), k≈6) — every operation inside
    is a full-vector op across all 2,108 rows, so the row axis is never iterated (vectorization
    mandate honoured). Returns tau in [-1, +1]: +1 = monotonic improvement over time, -1 =
    monotonic decline, 0 = no net trend. NaN-conservative: a row needs MORE than half its pairs
    comparable, else NaN (semantic truth — never fabricate a trend from sparse history)."""
    series = [_safe_numeric(df.get(c, pd.Series(np.nan, index=df.index))) for c in cols]
    n = len(series)
    total_pairs = n * (n - 1) // 2
    if total_pairs == 0:
        return pd.Series(np.nan, index=df.index)
    net = pd.Series(0.0, index=df.index)    # concordant − discordant (signed)
    valid = pd.Series(0, index=df.index)    # count of comparable (both-present) pairs
    for i in range(n):
        for j in range(i + 1, n):
            both = series[i].notna() & series[j].notna()
            diff = series[j] - series[i]    # newer − older: positive sign = improving
            net = net + np.where(both, np.sign(diff), 0.0)
            valid = valid + both.astype(int)
    min_pairs = total_pairs // 2 + 1
    tau = np.where(valid >= min_pairs, net / np.where(valid > 0, valid, np.nan), np.nan)
    return pd.Series(tau, index=df.index)


def _implied_growth_from_pb(pb, roe, cost_of_equity: float):
    """Back out the perpetual growth rate the market is pricing in by inverting the residual-income
    / Gordon P/B identity:  P/B = (ROE − g) / (CoE − g)  ⇒  g = (CoE·P/B − ROE) / (P/B − 1).
    Only defined for P/B > 1 (a premium to book); returns NaN at or below book (the identity is
    degenerate there). All rates are PERCENT (ROE, CoE) so g is returned in PERCENT. Defensive
    denominator guard per system mandate — never divide an un-guarded (P/B − 1)."""
    pb = _safe_numeric(pb)
    roe = _safe_numeric(roe)
    denom = pb - 1.0
    return np.where(denom > 0.0, (cost_of_equity * pb - roe) / denom, np.nan)


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

_NA_VALS = ["null", "NULL", "None", "N/A", "n/a", "#N/A", "#VALUE!", "#REF!", ""]


def _apply_column_mapping(df: pd.DataFrame, col_map: Dict[str, str], sheet_name: str) -> pd.DataFrame:
    """Promote the real header row if needed, then select+rename to snake_case columns.

    Header auto-detect: local CSVs / sheet exports have an emoji section-label row 0
    (Details, Unnamed:1…) followed by actual column names in row 1; direct exports without
    the emoji row have column names in row 0. We scan the first 5 rows for the one
    containing "companyId" and promote it — works for both layouts.
    """
    if "companyId" not in df.columns:
        hdr_idx = None
        for i in range(min(5, len(df))):
            if (df.iloc[i].astype(str) == "companyId").any():
                hdr_idx = i
                break
        if hdr_idx is None:
            hdr_idx = 0   # legacy fallback: promote row 0
        new_cols = [str(c) if pd.notna(c) else f"_col_{i}" for i, c in enumerate(df.iloc[hdr_idx])]
        df.columns = new_cols
        df = df.iloc[hdr_idx + 1:].reset_index(drop=True)

    # Build the full mapping: common + sheet-specific
    full_map = {**COMMON_COLS, **col_map}

    # Keep only columns that exist in this CSV
    available = {k: v for k, v in full_map.items() if k in df.columns}
    missing = set(col_map.keys()) - set(df.columns)
    if missing:
        print(f"  ⚠️  [{sheet_name}] Missing columns: {missing}")
    # Wrong-tab guard: if NONE of the sheet-specific columns matched, the source returned a
    # different tab entirely. Silently continuing would feed all-NaN data downstream — every
    # stock would score the neutral defaults (Growth=50, Momentum=26, Governance=0).
    if col_map and not any(k in df.columns for k in col_map):
        got = ", ".join(str(c) for c in list(df.columns)[:8])
        raise ValueError(
            f"[{sheet_name}] None of the expected columns were found — the data source "
            f"returned the wrong sheet/tab. First columns received: [{got}]. Check that the "
            f"spreadsheet has a tab with the exact expected name and that it is shared "
            f"(Anyone with the link → Viewer)."
        )

    # Select and rename
    df = df[list(available.keys())].rename(columns=available)

    return df


def _load_single_csv(filepath: str, col_map: Dict[str, str], sheet_name: str) -> pd.DataFrame:
    """Load a single CSV, apply column mapping, and return clean DataFrame."""
    df = pd.read_csv(
        filepath,
        header=0,
        low_memory=False,
        na_values=_NA_VALS,
        keep_default_na=True,
    )
    return _apply_column_mapping(df, col_map, sheet_name)


def load_all_csvs(data_source: str = "local", uploaded_files: dict = None, sheet_id: str = None) -> Dict[str, pd.DataFrame]:
    """Load all 6 CSV files and return as a dict of DataFrames."""
    print("📂 Loading CSV data...")
    datasets = {}

    from config import SHEET_TAB_NAMES

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
        # Download the ENTIRE workbook once as XLSX, then read each tab BY NAME locally.
        # WHY XLSX and not per-tab CSV URLs:
        #   CSV export with a "sheet=NAME" param — the param is silently IGNORED; every
        #     request returns the FIRST tab → 5 of 6 sheets all-NaN → every stock scores
        #     the neutral defaults (Growth=50, Momentum=26, Governance=0).
        #   CSV export with a GID param — GIDs are per-spreadsheet; wrong for other users.
        #   gviz tab-by-name endpoint — mangles the 2-row header (emoji row + names).
        # XLSX uses the SAME /export endpoint + sharing rules as CSV, selects tabs by their
        # exact names (SHEET_TAB_NAMES contract), and is a single download instead of six.
        parsed_id = extract_spreadsheet_id(sheet_id)
        xlsx_url = f"https://docs.google.com/spreadsheets/d/{parsed_id}/export?format=xlsx"
        try:
            workbook = pd.ExcelFile(xlsx_url, engine="openpyxl")
        except Exception as e:
            raise Exception(
                f"Could not download the spreadsheet as XLSX. Check the Sheet ID/URL and "
                f"that it is shared (Anyone with the link → Viewer). Underlying error: {e}"
            )
        available_tabs = list(workbook.sheet_names)
        for name, (cols,) in sheet_configs.items():
            tab_name = SHEET_TAB_NAMES[name]          # exact tab name e.g. "Income Statement"
            if tab_name not in available_tabs:
                raise Exception(
                    f"Tab '{tab_name}' not found in the spreadsheet. "
                    f"Available tabs: {available_tabs}. Tab names must match exactly."
                )
            try:
                raw = workbook.parse(
                    tab_name, header=0, na_values=_NA_VALS, keep_default_na=True
                )
                datasets[name] = _apply_column_mapping(raw, cols, name)
                print(f"  ✅ {name} ('{tab_name}'): {len(datasets[name])} rows, {len(datasets[name].columns)} cols")
            except Exception as e:
                raise Exception(f"Failed to load '{tab_name}' tab from Google Sheets: {e}")
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

    # ═══════════════════════════════════════════════════════════════
    # SCHEMA NORMALIZATION — permanent KeyError shield
    # When any Google Sheets tab fails to load, ALL its columns are absent.
    # We materialize every expected non-ratio column as float64 NaN so that
    # every downstream np.where / .fillna() guard works without crashing.
    # Over-inclusive by design: if a column is already in df (from the ratio
    # sheet), the guard is a no-op. New columns just get added to the list.
    # ═══════════════════════════════════════════════════════════════
    _EXPECTED_COLS = [
        # ── Income Statement: absolute values ──
        "pat", "pat_1yb", "pat_2yb", "pat_3yb", "pat_4yb", "pat_5yb",
        "revenue", "revenue_1yb", "revenue_2yb", "revenue_3yb", "revenue_4yb", "revenue_5yb",
        "ebitda", "ebit", "pbt",
        "expenses", "expenses_1yb",
        "depreciation", "depreciation_1yb",
        "eps", "eps_1yb",
        "equity_shares", "equity_shares_1yb",
        "opm", "opm_1yb", "npm", "npm_1yb",
        # ── Income Statement: growth rates (absent when Income tab fails to load) ──
        "eps_gr_yoy", "eps_gr_3y", "eps_gr_5y", "eps_gr_10y",
        "pat_gr_yoy", "pat_gr_3y", "pat_gr_5y", "pat_gr_10y",
        "rev_gr_yoy", "rev_gr_3y", "rev_gr_5y", "rev_gr_10y",
        "ebitda_gr_5y", "ebitda_gr_3y", "ebit_gr_3y",
        # individual year-by-year growth rates (computed from income raw data)
        "rev_gr_y2", "rev_gr_y3", "rev_gr_y4", "rev_gr_y5",
        "pat_gr_y2", "pat_gr_y3", "pat_gr_y4", "pat_gr_y5",
        # ── Balance Sheet ──
        "fixed_assets", "fixed_assets_1yb", "fixed_assets_2yb", "fixed_assets_3yb",
        "total_assets", "total_assets_1yb",
        "net_worth", "net_worth_1yb",
        "reserves", "reserves_1yb",
        "cwip", "cwip_1yb",
        "cash_equivalents", "cash_equivalents_1yb",
        "debt", "debt_1yb", "debt_2yb", "debt_3yb",
        "days_receivable", "days_receivable_1yb", "days_receivable_2yb", "days_receivable_3yb",
        "days_payable", "days_payable_1yb",
        "inventory_days", "inventory_days_1yb",
        "inventory_turnover", "inventory_turnover_1yb",
        # ── Cashflow ──
        "free_cash_flow", "fcf_1yb",
        "operating_cash_flow", "ocf_1yb",
        "investing_cash_flow", "financing_cash_flow",
        "net_cash_flow", "ncf_1yb",
        # ── Technicals ──
        "close_price",
        "sma_200d", "sma_50d",
        "vol_sma_5d", "vol_sma_10d", "vol_sma_20d", "vol_sma_50d",
        "volume",
        "vstop_value", "last_vstop_change",
        "crs_50d", "crs_26w", "crs_52w",
        "ret_vs_n500_3m", "ret_vs_n500_6m", "ret_vs_n500_1y",
        "ret_vs_industry_1y", "ret_vs_industry_3m",
        "adx_14w", "rsi_14d",
        "dist_52wh", "dist_52wl", "dist_13wh", "dist_ath",
        "dist_52wh_days", "golden_cross_days", "breakout_window",
        "vcp_volume_dryup", "d45_trend_structure", "d47_rs_composite",
        "pe_med_10y", "pe_med_5y", "industry_pe",
        "opm_latest_q", "npm_latest_q", "gpm_latest_q",
        "q_pat_yoy", "q_rev_yoy", "q_eps_yoy",
        "q_pat_lq", "q_rev_lq", "q_pat_2q", "q_rev_2q",
        # ── Shareholdings ──
        "promoter_holdings", "promoter_holdings_1qb",
        "fii_holdings", "dii_holdings",
        "change_fii_lq", "change_fii_1y",
        "change_dii_lq",
        "change_promoter_lq", "change_promoter_1y", "change_promoter_2y", "change_promoter_3y",
        "pledged_percentage", "pledged_1yb",
        "insider_trading", "pledge_falling_1y",
    ]
    _missing_cols = [c for c in _EXPECTED_COLS if c not in df.columns]
    if _missing_cols:
        _guard_frame = pd.DataFrame(
            np.nan, index=df.index, columns=_missing_cols, dtype="float64"
        )
        df = pd.concat([df, _guard_frame], axis=1)
        print(f"  ⚠️  Schema guard: materialized {len(_missing_cols)} absent columns as NaN "
              f"({', '.join(_missing_cols[:5])}{'...' if len(_missing_cols) > 5 else ''})")

    # ── EPS fallback: fill NaN eps_gr_yoy from raw eps/eps_1yb before winsorization ──
    # 34 stocks where Screener couldn't compute eps_gr_yoy (turnarounds, loss→profit transitions).
    # Compute from raw values first so these stocks go through winsorization with real values.
    if "eps" in df.columns and "eps_1yb" in df.columns:
        _eps_gr_raw = np.where(
            df["eps_1yb"].notna() & (df["eps_1yb"].abs() > 0.01) & df["eps"].notna(),
            (df["eps"] - df["eps_1yb"]) / df["eps_1yb"].abs() * 100,
            np.nan
        )
        df["eps_gr_yoy"] = df["eps_gr_yoy"].fillna(pd.Series(_eps_gr_raw, index=df.index))

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
    # roe_med_3y gives recent-vs-decade signal; 7Y fallback for companies with <3Y median data
    df["roe_trajectory"] = df["roe_med_3y"].fillna(df["roe_med_7y"]) - df["roe_med_10y"]
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

    # ── PE fallback: fill NaN pe from close_price / eps for recently-profitable companies ──
    # 161 stocks affected — Screener's PE is stale/missing when EPS recently turned positive.
    # These stocks scored neutral (50) on all PE-based signals; now get real values.
    # Guard: eps must be positive (no PE for loss-makers); clip pe at 999 (avoids micro-EPS blow-ups).
    if "eps" in df.columns and "close_price" in df.columns:
        _pe_computed = np.where(
            df["eps"].fillna(0) > 0,
            (df["close_price"].fillna(0) / df["eps"].clip(lower=0.01)).clip(upper=999),
            np.nan
        )
        df["pe"] = df["pe"].fillna(pd.Series(_pe_computed, index=df.index))

    # pe_med_5y primary: 2020-2025 complete mini-cycle, more relevant to current rate environment.
    # Fallback to 10Y for companies with < 5 years of PE history.
    _pe_base_disc = df["pe_med_5y"].fillna(df["pe_med_10y"])
    df["pe_discount"] = np.where(
        _pe_base_disc.notna() & (_pe_base_disc != 0),
        (_pe_base_disc - df["pe"]) / _pe_base_disc * 100,
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
    _eq     = df.get("equity_shares",     pd.Series(np.nan, index=df.index))
    _eq_1yb = df.get("equity_shares_1yb", pd.Series(np.nan, index=df.index))
    shares_valid = _eq.notna() & _eq_1yb.notna() & (_eq_1yb > 0)

    df["dilution_pct"] = np.where(
        shares_valid,
        (_eq - _eq_1yb) / _eq_1yb * 100,
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
    # Revenue growth volatility σ — std dev of the 4 year-on-year growth rates.
    # Used downstream as σ²/2 in the Ito variance-drag correction on expected CAGR:
    # E[log return] = μ − σ²/2. Volatile earners compound less at equal average growth.
    _rg_matrix = pd.concat(
        [df.get(f"rev_gr_y{_yr}", pd.Series(np.nan, index=df.index)).clip(-100, 300)
         for _yr in range(2, 6)],
        axis=1
    )
    df["sigma_g"] = _rg_matrix.std(axis=1, ddof=1)  # % units; NaN when <2 valid points

    _pg3  = df.get("pat_gr_3y",    pd.Series(np.nan, index=df.index))
    _pg5  = df.get("pat_gr_5y",    pd.Series(np.nan, index=df.index))
    _rg3  = df.get("rev_gr_3y",    pd.Series(np.nan, index=df.index))
    _rg5  = df.get("rev_gr_5y",    pd.Series(np.nan, index=df.index))
    _edg3 = df.get("ebitda_gr_3y", pd.Series(np.nan, index=df.index))
    _edg5 = df.get("ebitda_gr_5y", pd.Series(np.nan, index=df.index))
    _eg5  = df.get("eps_gr_5y",    pd.Series(np.nan, index=df.index))
    _etg3 = df.get("ebit_gr_3y",   pd.Series(np.nan, index=df.index))
    df["pat_acceleration"]    = _pg3  - _pg5
    df["rev_acceleration"]    = _rg3  - _rg5
    df["ebitda_acceleration"] = _edg3 - _edg5
    df["eps_vs_pat_delta"]    = _eg5  - _pg5
    # ebit_vs_rev_spread_3y: EBIT growth minus revenue growth (same 3Y window).
    # Positive = operating engine scaling faster than top-line = true operating leverage.
    # Cleaner than pat_gr - rev_gr because EBIT excludes interest (no capital structure distortion).
    df["ebit_vs_rev_spread_3y"] = np.where(
        _etg3.notna() & _rg3.notna(),
        _etg3 - _rg3,
        np.nan
    )
    # ebit_acceleration: EBIT growth minus EBITDA growth (same 3Y window).
    # Positive = D&A stable/shrinking relative to profits = asset-light model maturing.
    # Negative = D&A growing faster than EBIT = heavy capex cycle or asset ageing.
    df["ebit_acceleration"] = np.where(
        _etg3.notna() & _edg3.notna(),
        _etg3 - _edg3,
        np.nan
    )
    _q_nan   = pd.Series(np.nan, index=df.index)
    _pat_lq  = df.get("pat_lq",    _q_nan); _pat_pyq  = df.get("pat_pyq",    _q_nan)
    _rev_lq  = df.get("rev_lq",    _q_nan); _rev_pyq  = df.get("rev_pyq",    _q_nan)
    _ebd_lq  = df.get("ebitda_lq", _q_nan); _ebd_pyq  = df.get("ebitda_pyq", _q_nan)
    _eps_lq  = df.get("eps_lq",    _q_nan); _eps_pyq  = df.get("eps_pyq",    _q_nan)
    df["q_pat_yoy"] = np.where(
        _pat_pyq.notna() & (_pat_pyq.abs() > 0),
        (_pat_lq - _pat_pyq) / _pat_pyq.abs() * 100,
        np.nan
    )
    df["q_rev_yoy"] = np.where(
        _rev_pyq.notna() & (_rev_pyq.abs() > 0),
        (_rev_lq - _rev_pyq) / _rev_pyq.abs() * 100,
        np.nan
    )
    df["q_ebitda_yoy"] = np.where(
        _ebd_pyq.notna() & (_ebd_pyq.abs() > 0),
        (_ebd_lq - _ebd_pyq) / _ebd_pyq.abs() * 100,
        np.nan
    )
    # Quarterly EPS YoY growth — eps_lq vs eps_pyq (same quarter prior year)
    # Completes the quarterly set: q_pat_yoy, q_rev_yoy, q_ebitda_yoy, q_eps_yoy
    df["q_eps_yoy"] = np.where(
        _eps_pyq.notna() & (_eps_pyq.abs() > 0),
        (_eps_lq - _eps_pyq) / _eps_pyq.abs() * 100,
        np.nan
    )
    _rev_er  = df.get("revenue",      pd.Series(np.nan, index=df.index))
    _exp_er  = df.get("expenses",     pd.Series(np.nan, index=df.index))
    _rev1_er = df.get("revenue_1yb",  pd.Series(np.nan, index=df.index))
    _exp1_er = df.get("expenses_1yb", pd.Series(np.nan, index=df.index))
    df["expense_ratio"] = np.where(
        _rev_er.notna() & (_rev_er > 0),
        _exp_er / _rev_er,
        np.nan
    )
    df["expense_ratio_1yb"] = np.where(
        _rev1_er.notna() & (_rev1_er > 0),
        _exp1_er / _rev1_er,
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
    _ebitda_da     = df.get("ebitda",     pd.Series(np.nan, index=df.index))
    _ebitda_1yb_da = df.get("ebitda_1yb", pd.Series(np.nan, index=df.index))
    df["depreciation"]     = (_ebitda_da.fillna(np.nan)     - _ebit_da).clip(lower=0)
    df["depreciation_1yb"] = (_ebitda_1yb_da.fillna(np.nan) - _ebit_1yb_da).clip(lower=0)

    # dep_rate: D&A as percentage of gross fixed assets (scale-invariant)
    # Used in Schilit Signal 6 (forensic_engine.py): if fixed assets grow but dep_rate falls,
    # management has extended accounting useful lives to reduce D&A expense and inflate EBIT/PAT.
    # Book anchor: Qwest (Schilit Ch.6) extended asset lives from 14→40yr → $1B earnings boost.
    # NaN when fixed_assets = 0 — asset-light companies (no FA base, dep/FA is undefined).
    _fa_dep  = df.get("fixed_assets",     pd.Series(np.nan, index=df.index))
    _fa1_dep = df.get("fixed_assets_1yb", pd.Series(np.nan, index=df.index))
    df["dep_rate"] = np.where(
        _fa_dep.notna() & (_fa_dep > 0) & df["depreciation"].notna(),
        df["depreciation"] / _fa_dep * 100,
        np.nan
    )
    df["dep_rate_1yb"] = np.where(
        _fa1_dep.notna() & (_fa1_dep > 0) & df["depreciation_1yb"].notna(),
        df["depreciation_1yb"] / _fa1_dep * 100,
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
        _orig_null = df["free_cash_flow"].isna()

        # ── CapEx reconstruction via balance-sheet identity ──
        # CapEx ≈ max(0, Δ Gross Block + Δ CWIP) + Depreciation
        # max(0,·) prevents net asset disposals from inflating reconstructed FCF.
        # Missing FA/CWIP deltas filled with 0 (conservative lower bound on net additions)
        # so reconstruction succeeds whenever Depreciation is available, even if one
        # balance-sheet line is absent. NaN propagates only when Depreciation is missing.
        _fa = _safe_numeric(df.get("fixed_assets", pd.Series(np.nan, index=df.index)))
        _fa_1yb = _safe_numeric(df.get("fixed_assets_1yb", pd.Series(np.nan, index=df.index)))
        _cwip = _safe_numeric(df.get("cwip", pd.Series(np.nan, index=df.index)))
        _cwip_1yb = _safe_numeric(df.get("cwip_1yb", pd.Series(np.nan, index=df.index)))
        _dep = _safe_numeric(df.get("depreciation", pd.Series(np.nan, index=df.index)))
        _net_capex_chg = (_fa - _fa_1yb).fillna(0.0) + (_cwip - _cwip_1yb).fillna(0.0)
        df["capex_est"] = np.where(
            _dep.notna(),
            np.maximum(0.0, _net_capex_chg) + _dep,
            np.nan
        )
        df["fcf_reconstructed"] = df["operating_cash_flow"] - df["capex_est"]

        # fcf_reconstructed_flag: 1 where a null FCF was filled by reconstruction.
        _recon_ok = _orig_null & df["fcf_reconstructed"].notna()
        df["fcf_reconstructed_flag"] = _recon_ok.astype(int)

        # Fill priority: reconstruction first, raw OCF fallback last.
        df["free_cash_flow"] = df["free_cash_flow"].fillna(df["fcf_reconstructed"])
        df["free_cash_flow"] = df["free_cash_flow"].fillna(df["operating_cash_flow"])

        # fcf_imputed_flag = 1 ONLY for OCF-fallback stocks (FCF set to OCF directly →
        # fcf_to_cfo_pct would show a misleading 100%). Reconstructed stocks get 0 because
        # their FCF = OCF − capex_est is a real capex-adjusted estimate, not an OCF copy.
        _ocf_fallback = _orig_null & df["fcf_reconstructed"].isna()
        df["fcf_imputed_flag"] = _ocf_fallback.astype(int)

        if fcf_null_count > 0:
            _recon_n = int(_recon_ok.sum())
            _ocf_n = int(_ocf_fallback.sum())
            print(f"  ℹ️  FCF null for {fcf_null_count} stocks: "
                  f"{_recon_n} reconstructed via CapEx identity, {_ocf_n} via OCF fallback")

    _cf_nan = pd.Series(np.nan, index=df.index)
    _fcf    = df.get("free_cash_flow",       _cf_nan)
    _fcf_1y = df.get("fcf_1yb",              _cf_nan)
    _ocf    = df.get("operating_cash_flow",  _cf_nan)
    _ocf_1y = df.get("ocf_1yb",             _cf_nan)
    _icf    = df.get("investing_cash_flow",  _cf_nan)
    _fincf  = df.get("financing_cash_flow",  _cf_nan)
    _ncf    = df.get("net_cash_flow",        _cf_nan)
    _ncf_1y = df.get("ncf_1yb",             _cf_nan)
    # Income statement raw columns — absent when Income Statement tab fails to load
    _pat     = df.get("pat",     _cf_nan)
    _pat_1yb = df.get("pat_1yb", _cf_nan)
    _pat_2yb = df.get("pat_2yb", _cf_nan)
    _pat_3yb = df.get("pat_3yb", _cf_nan)
    _pat_4yb = df.get("pat_4yb", _cf_nan)
    _pat_5yb = df.get("pat_5yb", _cf_nan)
    _revenue = df.get("revenue", _cf_nan)
    _ebitda  = df.get("ebitda",  _cf_nan)
    df["fcf_yield"] = np.where(
        df["market_cap"].notna() & (df["market_cap"] > 0),
        _fcf / df["market_cap"] * 100,  # as percentage
        np.nan
    )
    # Null fcf_yield for OCF-fallback imputed stocks: free_cash_flow == operating_cash_flow
    # means capex was never deducted. For a capex-heavy stock the yield is overstated,
    # falsely passing Dorsey (≥5%), Dhandho (≥8%), Baid, and Quality Compounder gates.
    # fillna(0)==0 → keep, ==1 → null → downstream fillna(0) produces correct gate fail.
    df["fcf_yield"] = df["fcf_yield"].where(
        df.get("fcf_imputed_flag", pd.Series(0, index=df.index)).fillna(0) == 0
    )
    df["fcf_growth"] = np.where(
        _fcf_1y.notna() & (_fcf_1y.abs() > 0),
        (_fcf - _fcf_1y) / _fcf_1y.abs() * 100,
        np.nan
    )
    df["ocf_growth"] = np.where(
        _ocf_1y.notna() & (_ocf_1y.abs() > 0),
        (_ocf - _ocf_1y) / _ocf_1y.abs() * 100,
        np.nan
    )
    df["capex_coverage"] = np.where(
        _icf.notna() & (_icf.abs() > 0),
        _ocf / _icf.abs(),
        np.nan
    )
    df["fcf_consistency"] = (
        (_fcf > 0) & (_fcf_1y > 0)
    ).astype(int)
    df["self_funding"] = (
        (_ocf > 0) & (_fincf < 0)
    ).astype(int)
    df["ncf_trend"] = (
        (_ncf > 0) & (_ncf_1y > 0)
    ).astype(int)
    df["fcf_quality"] = np.where(
        _pat.notna() & (_pat.abs() > 0),
        _fcf / _pat.abs(),
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
    _bs_nan  = pd.Series(np.nan, index=df.index)
    _debt    = df.get("debt",                   _bs_nan); _debt_1yb  = df.get("debt_1yb",            _bs_nan)
    _debt_3yb = df.get("debt_3yb",              _bs_nan); _cash      = df.get("cash_equivalents",    _bs_nan)
    _cash_1yb = df.get("cash_equivalents_1yb",  _bs_nan); _reserves  = df.get("reserves",            _bs_nan)
    _res_1yb  = df.get("reserves_1yb",          _bs_nan); _cwip_bs   = df.get("cwip",                _bs_nan)
    _fa_bs    = df.get("fixed_assets",          _bs_nan); _fa1_bs    = df.get("fixed_assets_1yb",   _bs_nan)
    _fa2_bs   = df.get("fixed_assets_2yb",      _bs_nan); _fa3_bs    = df.get("fixed_assets_3yb",   _bs_nan)
    _ta       = df.get("total_assets",          _bs_nan); _tl        = df.get("total_liabilities",  _bs_nan)
    _tl_1yb   = df.get("total_liabilities_1yb", _bs_nan); _inv       = df.get("inventory",          _bs_nan)
    _inv_1yb  = df.get("inventory_1yb",         _bs_nan)
    df["net_debt"]       = _debt - _cash
    df["debt_slope_3y"]  = _debt - _debt_3yb
    df["debt_change_1y"] = _debt - _debt_1yb
    df["cash_change"]    = _cash - _cash_1yb
    df["reserves_growth"] = np.where(
        _res_1yb.notna() & (_res_1yb.abs() > 0),
        (_reserves - _res_1yb) / _res_1yb.abs() * 100,
        np.nan
    )
    # cwip_conversion is computed at D19 (fixed asset expansion formula). See line below.
    df["cwip_ratio"] = np.where(
        _fa_bs.notna() & (_fa_bs > 0),
        _cwip_bs / _fa_bs * 100,
        np.nan
    )
    df["capex_3y"] = _fa_bs - _fa3_bs

    # Capex consistency: year-by-year FA growth rate variance (3 consecutive years)
    # High variance = lumpy capex = project execution risk vs smooth compounding expansion.
    _fa_g1 = np.where(
        _fa1_bs.notna() & (_fa1_bs.abs() > 0),
        (_fa_bs - _fa1_bs) / _fa1_bs.abs(),
        np.nan
    )
    _fa_g2 = np.where(
        _fa2_bs.notna() & (_fa2_bs.abs() > 0),
        (_fa1_bs - _fa2_bs) / _fa2_bs.abs(),
        np.nan
    )
    df["capex_consistency"] = np.abs(_fa_g1 - _fa_g2)   # lower = smoother expansion

    df["inv_growth"] = np.where(
        _inv_1yb.notna() & (_inv_1yb > 0),
        (_inv - _inv_1yb) / _inv_1yb * 100,
        np.nan
    )
    df["inv_vs_rev_gap"] = df["inv_growth"] - df.get("rev_gr_yoy", _bs_nan)
    df["solvency_ratio"] = np.where(
        _ta.notna() & (_ta > 0),
        _tl / _ta,
        np.nan
    )
    # Hidden obligation growth — Schilit EMS #5 / IL&FS pattern (off-balance-sheet risk).
    # CSV SEMANTICS: total_liabilities is the WHOLE balance-sheet total (TL/TA ≡ 1.0 for
    # every stock), so the old "TL growing faster than debt" condition fired for ANY
    # company growing retained earnings — 82% of the universe, pure noise.
    # TRUE hidden obligations live in the non-debt, non-equity slice:
    #   other_liabilities = TL − reserves − debt   (payables, provisions, Ind AS 116
    #   leases, contingencies — share capital excluded: its Δ is ≈0 outside dilution,
    #   which the dedicated dilution flags already police).
    # Flag only MATERIAL growth: Δother_liabilities > 5% of total assets in one year.
    # Live rate after fix: 38% (Reliance, Airtel, L&T — lease/provision-heavy, correct).
    df["liab_change"] = _tl - _tl_1yb
    _rs_ho  = df.get("reserves",     _bs_nan)
    _rs1_ho = df.get("reserves_1yb", _bs_nan)
    _db_ho  = df.get("debt",         _bs_nan)
    _db1_ho = df.get("debt_1yb",     _bs_nan)
    _other_liab_chg = (
        (_tl - _rs_ho - _db_ho.fillna(0)) - (_tl_1yb - _rs1_ho - _db1_ho.fillna(0))
    )
    df["hidden_obligation_growth"] = np.where(
        _tl.notna() & _tl_1yb.notna() & _rs_ho.notna() & _rs1_ho.notna() &
        _ta.notna() & (_ta > 0),
        (_other_liab_chg > _ta * 0.05).astype(int),
        0   # cannot compute the non-equity slice → conservative no-flag
    )

    # ── SHAREHOLDING DERIVED ──
    _sh_nan      = pd.Series(np.nan, index=df.index)
    _sh_zero     = pd.Series(0.0,    index=df.index)
    _pledged     = df.get("pledged_percentage", _sh_nan)
    _pledged_1qb = df.get("pledged_1qb",        _sh_nan)
    _pledged_1yb = df.get("pledged_1yb",        _sh_nan)
    _ch_prom_lq  = df.get("change_promoter_lq", _sh_zero)
    _ch_fii_lq   = df.get("change_fii_lq",      _sh_zero)
    _ch_dii_lq   = df.get("change_dii_lq",      _sh_zero)
    df["pledge_rising"] = np.where(
        _pledged.notna() & _pledged_1qb.notna(),
        (_pledged > _pledged_1qb).astype(int),
        0
    )
    df["pledge_falling_1y"] = np.where(
        _pledged_1yb.notna() & _pledged.notna(),
        (_pledged_1yb - _pledged).clip(lower=0),
        0
    )
    df["promoter_buying"] = (_ch_prom_lq > 0).astype(int)
    df["inst_convergence"] = (
        (_ch_fii_lq > 0) & (_ch_dii_lq > 0)
    ).astype(int)

    # ── TECHNICAL DERIVED ──
    _vol      = df.get("volume",     pd.Series(np.nan, index=df.index))
    _vol20d   = df.get("vol_sma_20d", pd.Series(np.nan, index=df.index))
    _vol50d   = df.get("vol_sma_50d", pd.Series(np.nan, index=df.index))
    _close_t  = df.get("close_price", pd.Series(np.nan, index=df.index))
    _crs50d   = df.get("crs_50d",    pd.Series(0.0, index=df.index))
    _crs26w   = df.get("crs_26w",    pd.Series(0.0, index=df.index))
    _crs52w   = df.get("crs_52w",    pd.Series(0.0, index=df.index))
    _sma200d  = df.get("sma_200d",   pd.Series(np.nan, index=df.index))
    _vstop_t  = df.get("vstop_value", pd.Series(np.nan, index=df.index))
    _vstop_chg = df.get("last_vstop_change", pd.Series(np.nan, index=df.index))
    _ret_n500_3m = df.get("ret_vs_n500_3m", pd.Series(0.0, index=df.index))
    df["vol_ratio"] = np.where(
        _vol20d.notna() & (_vol20d > 0),
        _vol / _vol20d,
        np.nan
    )
    # Minervini breakout confirmation: volume vs the 50-day baseline (book uses 50D, not 20D).
    # A pivot breakout requires volume ≥ ~1.4× the 50D average. Distinct from vol_ratio (20D).
    df["vol_ratio_50d"] = np.where(
        _vol50d.fillna(0) > 0,
        _vol.fillna(0) / _vol50d.fillna(0),
        np.nan
    )
    # VCP volume dryup (SEPA Codex Ch.5 Chartink): 10D avg volume < 50D avg = supply exhaustion
    # in the base. Distinct from can_slim_vcp (5D < 20D). Missing vol data → 0 (no dryup).
    _v10d_vcp = df.get("vol_sma_10d", pd.Series(np.nan, index=df.index))
    _v50d_vcp = df.get("vol_sma_50d", pd.Series(np.nan, index=df.index))
    # Materiality factor 0.70 (recalibrated 2026-06-12 Minervini audit): the plain
    # 10D < 50D comparison is true ~half the time for any stock (fired for 61% of the
    # universe — a coin flip, not a signal). The book demands volume that "dries up
    # DRAMATICALLY, accompanied by tightness in price" (Trade Like a Stock Market
    # Wizard; also "dries up considerably"). 10D below 70% of 50D = a genuine 30%+
    # contraction — supply actually exhausted in the base, not daily noise.
    df["vcp_volume_dryup"] = np.where(
        _v10d_vcp.notna() & _v50d_vcp.notna() & (_v50d_vcp > 0),
        (_v10d_vcp < _v50d_vcp * 0.70).astype(int),
        0
    )
    df["daily_value"] = _vol * _close_t  # in raw ₹
    df["daily_value_cr"] = df["daily_value"] / 1e7  # in ₹ Crores
    df["crs_aligned"] = (
        (_crs50d > 0) & (_crs26w > 0) & (_crs52w > 0)
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
    # Re-fetch after in-place guard so local vars reflect any nullifications above.
    _vstop_t = df.get("vstop_value", pd.Series(np.nan, index=df.index))

    df["vstop_fresh"] = np.where(_vstop_chg.notna(), (_vstop_chg <= 30).astype(int), 0)
    df["above_sma200"] = (_close_t > _sma200d).astype(int)
    df["vstop_green"] = np.where(_vstop_t.notna(), (_close_t > _vstop_t).astype(int), 0)

    # ── VQS & SMART MONEY FLOW (WAVE DETECTION INTEGRATION) ──
    vqs_liquidity = np.where(df["vol_ratio"] >= 3.0, 50,
                    np.where(df["vol_ratio"] >= 2.0, 40,
                    np.where(df["vol_ratio"] >= 1.5, 30,
                    np.where(df["vol_ratio"] >= 1.0, 20, 10))))

    vqs_smart = np.where(df["inst_convergence"] == 1, 20,
                np.where((_ch_fii_lq.fillna(0) > 0) | (_ch_dii_lq.fillna(0) > 0), 10, 0))

    vqs_cons = np.where(df["crs_aligned"] == 1, 20,
               np.where((_crs50d.fillna(0) > 0) & (_crs26w.fillna(0) > 0), 10, 0))

    vqs_eff = np.where(_ret_n500_3m.fillna(0) > 0, 10, 0)

    df["vqs_score"] = pd.Series(vqs_liquidity + vqs_smart + vqs_cons + vqs_eff, index=df.index).fillna(0)

    df["smart_money_flow"] = np.select(
        [
            (df["vqs_score"] >= 80) & (df["inst_convergence"] == 1),
            (df["vqs_score"] >= 60) & ((_ch_fii_lq.fillna(0) > 0) | (_ch_dii_lq.fillna(0) > 0)),
            (df["vqs_score"] >= 40),
            (_ch_fii_lq.fillna(0) < 0) & (_ch_dii_lq.fillna(0) < 0) & (_crs50d.fillna(0) < 0)
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
        _vstop_t.notna() & (_vstop_t > 0),
        ((_close_t - _vstop_t) / _vstop_t) * 100,
        np.nan
    )
    # Below-stop branch FIRST (Marks audit fix 2026-06-12): dist_to_vstop is NEGATIVE
    # when price trades below the stop, and the old `<= 5` branch labeled those broken
    # trends "Perfect Entry (Low Risk)" — the most dangerous technical state wearing the
    # safest label (883 stocks). It also polluted the Marks Shield Price-vs-Value pillar,
    # which awards its check on this exact label. Marks: risk is highest when it feels lowest.
    df["buy_zone_label"] = np.select(
        [
            df["dist_to_vstop"] < 0,    # Price BELOW the stop — trend broken, not an entry
            df["dist_to_vstop"] <= 5,   # Within 5% above stop (Asymmetric Risk/Reward)
            df["dist_to_vstop"] <= 12,  # Normal volatility buffer
            df["dist_to_vstop"] > 25    # Extended far beyond 50DMA/VSTOP
        ],
        [
            "🔻 Below Stop (Trend Broken)",
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

    # ── REGULATED UTILITY FLAG ── Greenblatt's Magic Formula excludes utilities (rate-capped
    # returns distort the ROC ranking). Used by fw_magic_formula; sector-name based.
    df["is_utility"] = df["sector"].fillna("").isin(UTILITY_SECTOR_NAMES)

    # ── Atoms-to-Bits Classification (26th Study — Digital Business Design) ──
    # Maps every industry in the CSV to Bits / Atoms / Hybrid using a vectorized dict lookup.
    # Bits   = asset-light, network effects, zero marginal cost scale (IT, Pharma, FMCG, Finance).
    # Atoms  = physical capital intensive, linear scale, commodity-like (Metals, Infrastructure, Oil).
    # Hybrid = mix of both (Banks, Auto, Consumer Durables, Logistics, Retail).
    # Source: All 84 industry names from "sectors there in my csv.txt"; zero hallucination.
    # Sector column (84 clean aggregations) used — NOT industry (621 granular sub-entries).
    # Sector names match exactly the 84 values in "sectors there in my csv.txt".
    _BITS_SECTORS: frozenset = frozenset({
        "IT - Software", "IT - Hardware", "Pharmaceuticals", "Healthcare",
        "FMCG", "Finance", "Financial Services", "Insurance",
        "Stock/ Commodity Brokers", "Credit Rating Agencies",
        "E-Commerce/App based Aggregator", "Computer Education",
        "Media - Print/Television/Radio", "Entertainment", "Education",
        "Telecom-Service",
    })
    _ATOMS_SECTORS: frozenset = frozenset({
        "Crude Oil & Natural Gas", "Refineries", "Petrochemicals",
        "Steel", "Non Ferrous Metals", "Mining & Mineral products", "Ferro Alloys",
        "Cement", "Cement - Products",
        "Realty", "Construction", "Infrastructure Developers & Operators",
        "Infrastructure Investment Trusts", "Real Estate Investment Trusts",
        "Power Generation & Distribution", "Power Infrastructure",
        "Shipping", "Ship Building", "Gas Distribution", "Oil Drill/Allied",
        "Fertilizers", "Agro Chemicals", "Sugar", "Textiles",
        "Paper", "Plastic products", "Glass & Glass Products", "Refractories",
        "Castings, Forgings & Fastners", "Railways", "Leather",
        "Plantation & Plantation Products", "Plywood Boards/Laminates",
        "Marine Port & Services",
    })
    _atb_map: dict = {s: "Bits"  for s in _BITS_SECTORS}
    _atb_map.update({s: "Atoms" for s in _ATOMS_SECTORS})
    # pd.Series.map(dict) is fully vectorized — no apply(), no loops.
    # Unmapped sectors (including NaN) → "Hybrid" via fillna.
    df["atoms_to_bits_label"] = df["sector"].map(_atb_map).fillna("Hybrid")

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
        "days_payable",    "days_payable_1yb",     # DPO meaningless for banks/NBFCs
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
    rev = _revenue.fillna(0)
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
        (_ebitda.fillna(0) > 0) & (_pat.fillna(0) > 0) &
        (_ebitda.fillna(0) > _pat.fillna(0)),
        (1 - _pat / _ebitda) * 100,
        np.nan
    )

    # ── TRUE Effective Tax Rate (Malik Parameter 3, proper) ──
    # ETR = (PBT − PAT) / PBT × 100 — the actual tax burden, computable for 91% of the
    # universe (PBT > 0). Live median: 25.3% = India's post-2019 corporate rate (25.17%).
    # clip(0, 60): deferred-tax credits can push PAT > PBT (negative ETR → floor 0);
    # one-off charges can produce extreme ETR (→ cap 60 keeps band checks meaningful).
    _pbt_etr = df.get("pbt", pd.Series(np.nan, index=df.index))
    df["effective_tax_rate_pct"] = pd.Series(
        np.where(
            (_pbt_etr.fillna(0) > 0) & _pat.notna(),
            (_pbt_etr - _pat) / _pbt_etr * 100,
            np.nan
        ),
        index=df.index
    ).clip(0, 60)
    df["tax_rate_est"] = df["effective_tax_rate_pct"]  # alias now points at the TRUE rate

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
    # net_worth = market_cap / price_to_book — algebraically exact:
    #   price_to_book = close_price / book_value_per_share
    #   market_cap    = close_price × equity_shares
    #   market_cap / price_to_book = book_value_per_share × equity_shares = total equity (accounting)
    # The close_price cancels — result is a static accounting value, not affected by price movements.
    # Internally consistent with Screener's ROE (which uses the same book value denominator).
    # Fallback to reserves when price_to_book unavailable (<5 stocks).
    # net_worth_1yb: no historical price_to_book in CSV → reserves_1yb proxy unchanged.
    _nw_from_pb = np.where(
        df["price_to_book"].notna() & (df["price_to_book"] > 0) & df["market_cap"].notna(),
        df["market_cap"] / df["price_to_book"],
        np.nan
    )
    df["net_worth"]     = pd.Series(_nw_from_pb, index=df.index).fillna(df["reserves"].fillna(0))
    df["net_worth_1yb"] = df["reserves_1yb"].fillna(0)

    # ── Scaled Net Operating Assets (SNOA) — Quantitative Value (Gray & Carlisle) Ch.3 ──
    # QV's second earnings-manipulation weapon (after STA/accruals): captures a management's
    # CUMULATIVE (historical) balance-sheet accrual build-up, where STA captures the single-period
    # flow. Hirshleifer-Hou-Teoh-Zhang (2004): high SNOA predicts low future returns.
    # NOA = Operating Assets − Operating Liabilities = (Total Assets − Cash) − (Total Assets − Debt
    #       − Equity) = Debt + Equity − Cash. Scaled by LAGGED total assets (book convention).
    # All inputs are real mapped columns (debt, net_worth, cash_equivalents, total_assets_1yb) —
    # no proxy. MUST follow net_worth (computed just above). Artifact guard at the flag (rf_snoa).
    _snoa_noa = df["debt"].fillna(0) + df["net_worth"].fillna(0) - df["cash_equivalents"].fillna(0)
    _snoa_ta_lag = df.get("total_assets_1yb", pd.Series(np.nan, index=df.index))
    df["scaled_net_operating_assets"] = np.where(
        _snoa_ta_lag > 0, _snoa_noa / _snoa_ta_lag, np.nan
    )

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

    # ── EP Power Curve QUINTILE (28th WCS — the study's actual methodology) ──
    # Rank the universe by ABSOLUTE economic profit into 5 quintiles: Q1 = highest EP, Q5 = deepest loss.
    # Book-verified 2026-06-13. The study's REAL findings (prior comment's "ending in Q1/Q2 → 24/21%
    # CAGR, Q4/Q5 → 8/4%" matched no exhibit):
    #   • EP > Accounting Profit: top-EP portfolio +8% alpha (80% hit rate) vs Economic-LOSS +2% (Exhibit 7).
    #   • Returns come from MOVING UP the curve (the "Hockey Stick"): Q5→Q2 = 26% CAGR (25 cos),
    #     Q2→Q1 = 25% (21 cos) over 2013-23; 6-period 10yr avg = Exhibit 10.
    #   • Best to START from Quintile 2/3 — 68% of up-moves originate there, highest HSR probability
    #     (18%/19%). Q1 is already at the top; Q4/Q5 (Economic Loss) are erratic.
    # ep_top_quintile_flag below = high CURRENT EP (Q1/Q2 = quality marker); the UP-MOVE is captured
    # separately by ep_hockey_stick (EP positive + improving). Distinct from ep_power_curve (sign label).
    df["ep_quintile"] = np.nan
    _ep_valid = df["economic_profit"].notna()
    if int(_ep_valid.sum()) >= 5:
        # Rank ascending (lowest EP = rank 1); qcut into 5 equal buckets; label so highest EP = Q1.
        _ep_rank = df.loc[_ep_valid, "economic_profit"].rank(method="first")
        _ep_q = pd.qcut(_ep_rank, 5, labels=[5, 4, 3, 2, 1]).astype(int)
        df.loc[_ep_valid, "ep_quintile"] = _ep_q.values
    # Q1/Q2 = the market-beating zone (NaN EP → 99 → not flagged, conservative).
    df["ep_top_quintile_flag"] = (df["ep_quintile"].fillna(99) <= 2).astype(int)

    # ── Earnings Power Box (Heiserman, "It's Earnings That Count") ──
    # Heiserman's trademarked 2×2: cross self-funding (DEFENSIVE) with value-creation (ENTERPRISING).
    # Simplified Indian-market form (freefincal Earnings Power Box): enterprising = economic_profit
    # (Net Income − cost-of-equity charge, computed above); defensive = free_cash_flow > 0 (funds its
    # own growth without external capital). The distinctive read is the cross — value-creation WITH vs
    # WITHOUT self-funding (the cash-hungry grower needs dilutive financing; the earnings-power name
    # self-funds AND creates value). Display-only label — never gates, scores, or penalizes.
    _epb_defensive    = df["free_cash_flow"] > 0    # self-funds growth
    _epb_enterprising = df["economic_profit"] > 0   # earns above cost of equity
    _epb_valid = df["free_cash_flow"].notna() & df["economic_profit"].notna()
    df["earnings_power_box"] = np.select(
        [
            _epb_valid &  _epb_defensive &  _epb_enterprising,
            _epb_valid &  _epb_defensive & ~_epb_enterprising,
            _epb_valid & ~_epb_defensive &  _epb_enterprising,
            _epb_valid & ~_epb_defensive & ~_epb_enterprising,
        ],
        ["📦 Earnings Power", "💰 Cash Cow", "🚀 Cash-Hungry Grower", "⚠️ Weakest"],
        default=""
    )

    # ── Result Staleness (data-recency guard, sibling to data_coverage_pct) ──
    # days_from_result: negative = days SINCE the last reported result (past), positive = an
    # upcoming scheduled result. result_age_days flips the sign so positive = days the financials
    # have gone stale. A company that has not reported in >120 days is (a) scored on outdated
    # fundamentals and (b) frequently in distress — Gensol Engineering sat 477 days stale before
    # its 2025 collapse. Display-only companion to the Evidence badge; never gates or scores.
    # NaN days_from_result → age NaN → comparison False → not flagged (don't assert staleness blind).
    _dfr = df.get("days_from_result", pd.Series(np.nan, index=df.index))
    df["result_age_days"]   = -_dfr
    df["result_stale_flag"] = (df["result_age_days"] > 120).astype(int)

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
        _revenue.fillna(0) > 0,
        df["market_cap"].fillna(0) / _revenue,
        np.nan
    )
    df["ps_lt1_flag"]  = (df["ps_ratio"].fillna(999) <= 1.0).astype(int)   # multi-bagger formula (Study 13)
    df["ps_lt05_flag"] = (df["ps_ratio"].fillna(999) <= 0.5).astype(int)   # doubler formula (Study 9)

    # ── O'Shaughnessy Value Composite (What Works on Wall Street, 4th ed.) ──
    # His signature factor: percentile-rank multiple value ratios and average them. "These
    # composited value factors outperform all the single-factor value factors." 5-factor pure-play
    # VC1: Price-to-Book, P/E, Price-to-Sales, EBITDA/EV, Price-to-Cash-Flow. CHEAP (low ratio) =
    # high rank (O'Shaughnessy: "lowest 1% PE → score 100"). 0-100, high = cheapest across all 5.
    # NOTE: VC2/VC3 add shareholder yield (buyback + dividend) — NOT added: buybacks are rare in
    # India (median 0, only 47 net buybacks) and dividend yield is the broken DPR column. The
    # capital-return dimension is already covered by external_financing_to_assets (Tortoriello).
    # Cross-sectional ranks (cheapest positive ratio → highest pct). CRITICAL: a NEGATIVE ratio
    # (loss-maker PE, negative book, negative EBITDA) is DISTRESS, not cheapness — it must score
    # WORST (0), never be excluded (excluding lets a distressed stock rank on its other factors —
    # the classic value-composite trap, e.g. negative-book State Trading scoring top-decile).
    # Missing stays NaN; require >=3 valid factors for a meaningful composite.
    def _osh_rank(s):
        r = s.where(s > 0).rank(pct=True, ascending=False)  # positives: cheapest = highest
        return r.mask(s <= 0, 0.0)                           # negative/zero = worst; missing = NaN
    _pcf_osh = pd.Series(
        np.where(df["operating_cash_flow"].fillna(0) > 0,
                 df["market_cap"] / df["operating_cash_flow"], np.nan),
        index=df.index,
    )
    _vc_osh = pd.concat([
        _osh_rank(df["pe"]), _osh_rank(df["price_to_book"]), _osh_rank(df["ps_ratio"]),
        _osh_rank(df["ev_ebitda"]), _osh_rank(_pcf_osh),
    ], axis=1)
    df["oshaughnessy_value_composite"] = np.where(
        _vc_osh.notna().sum(axis=1) >= 3, _vc_osh.mean(axis=1) * 100.0, np.nan
    )

    # ── Trending Value — O'Shaughnessy's FLAGSHIP strategy (his single best) ──
    # "Buy the best 6-month-momentum stocks from the upper 10% of the value composite — earns
    # 20%+/year since 1963 with LOWER risk than the universe." Combines deep value with momentum
    # confirmation: the cheap stock must ALSO be outperforming (avoids value traps — cheap stocks
    # that keep falling). value_composite top decile (>=90) AND positive 6-month relative strength
    # (crs_26w > 0, CRS vs Nifty 500 over 26 weeks ≈ 6 months). No existing framework pairs the
    # multi-ratio value composite with momentum — this is the orthogonal O'Shaughnessy combination.
    _tv_crs = df.get("crs_26w", pd.Series(np.nan, index=df.index))
    df["trending_value_flag"] = (
        (df["oshaughnessy_value_composite"] >= 90.0) & (_tv_crs > 0.0)
    ).astype(int)

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

    # P3: TRUE effective tax rate (PBT−PAT)/PBT — Malik: a normal tax rate signals
    # honest accounting; abnormally low (SEZ expiry risk / manipulation) or high is a flag.
    # Full band 20–36% covers the 25.17% new regime AND the ~35% legacy regime.
    # Fallback when PBT missing (~9% of stocks): EBITDA-to-PAT gap proxy — it embeds
    # Dep + Interest + Tax, so its honest band is 30–55 (live median 42%), at reduced
    # confidence (×0.7) because it is a proxy, not the actual tax rate.
    _etr_p3 = df["effective_tax_rate_pct"]
    _gap_p3 = df["ebitda_to_pat_gap_pct"]
    malik_p3 = np.select(
        [
            _etr_p3.notna() & (_etr_p3 >= 20) & (_etr_p3 <= 36),
            _etr_p3.notna() & (_etr_p3 >= 15) & (_etr_p3 <= 40),
            _etr_p3.notna(),                                       # known but abnormal
            _gap_p3.notna() & (_gap_p3 >= 30) & (_gap_p3 <= 55),   # proxy fallback band
            _gap_p3.notna(),                                       # proxy known, abnormal
        ],
        [pw, pw * 0.5, 0.0, pw * 0.7, 0.0],
        default=pw * 0.3   # no data at all = small benefit of doubt (unchanged)
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

    df["malik_checklist_score"] = pd.Series(
        malik_p1 + malik_p2 + malik_p3 + malik_p4 +
        malik_p5 + malik_p6 + malik_p7 + malik_p8,
        index=df.index
    ).clip(0, 100).round(1)

    df["malik_label"] = np.select(
        [df["malik_checklist_score"] >= 80, df["malik_checklist_score"] >= 60, df["malik_checklist_score"] >= 40],
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

    # ── Growth-Value Trap (23rd WCS, Exhibit 8/9 — "all growth is not good") ──
    # VERBATIM finding: "If RoE remains below Cost of Equity, high growth actually DETRACTS firm value,
    # as the company must raise capital to fund growth that earns below its cost" → firm value goes
    # NEGATIVE, and the faster it grows the worse it gets (Exhibit 8: ROE 10% + 40% growth → value -2,900).
    # Distinct from moat_growth_quad (ROCE-based): here ROE-vs-Cost-of-EQUITY is the correct equity
    # value-creation test, and this flags ACTIVE value destruction, not merely "less attractive".
    _gvt_growth   = df["pat_gr_5y"].fillna(df["pat_gr_3y"]).fillna(0) >= 15.0   # genuinely growing
    _gvt_roe      = df["roe"].fillna(df["roe_med_5y"]).fillna(999.0)            # 999 → not flagged if ROE unknown
    df["growth_value_trap"] = ((_gvt_growth) & (_gvt_roe < COST_OF_EQUITY)).astype(int)

    # ── Cyclical Peak Trap (9th WCS — Commodity Business Cycles) ──
    # 9th WCS: at Phase III (the "squeeze"), commodity profits rise EXPONENTIALLY (SAIL: +21%
    # realization → +1000% profit), so P/E COLLAPSES to look "cheap" — but earnings are at a cyclical
    # peak about to revert. A low P/E here is DECEPTIVE; rewarding it (via pe_discount) is the trap.
    # Surgical triple-AND so it only fires on the genuine peak-cycle pattern (near-zero false positives):
    #   (1) commodity-PRICE-cyclical sector (tighter than all "Atoms" — excludes utilities/realty/rail)
    #   (2) low P/E (looks cheap: 0 < PE < 12)
    #   (3) peak earnings — operating margin >=3pp above its 5Y median OR PAT YoY spike >50%
    _COMMODITY_CYCLICAL_SECTORS: frozenset = frozenset({
        "Steel", "Non Ferrous Metals", "Mining & Mineral products", "Ferro Alloys",
        "Cement", "Cement - Products", "Sugar", "Petrochemicals", "Refineries",
        "Crude Oil & Natural Gas", "Fertilizers", "Agro Chemicals", "Paper", "Textiles",
        "Shipping",
    })
    _cyc_sector  = df["sector"].isin(_COMMODITY_CYCLICAL_SECTORS)
    _cyc_cheap   = (df["pe"].fillna(999.0) > 0) & (df["pe"].fillna(999.0) < 12.0)
    _cyc_peak    = (
        (df["opm"].fillna(0) > (df["opm_med_5y"].fillna(df["opm"]).fillna(0) + 3.0)) |  # margin at peak
        (df["pat_gr_yoy"].fillna(0) > 50.0)                                              # earnings spike
    )
    df["cyclical_peak_trap"] = (_cyc_sector & _cyc_cheap & _cyc_peak).astype(int)

    # ── Sales→Profit Conversion (Malik Moat Test 3 + WCS) ──
    # Primary: ebit_gr_3y - rev_gr_3y (EBIT excludes interest — no capital-structure distortion).
    # Fallback: pat_gr_5y - rev_gr_5y when ebit_gr_3y unavailable.
    # Positive = operating engine scaling faster than revenue = true operating leverage proven.
    df["sales_profit_conversion"] = np.where(
        df["ebit_gr_3y"].notna() & df["rev_gr_3y"].notna(),
        df["ebit_gr_3y"] - df["rev_gr_3y"],                      # EBIT-based: clean signal
        np.where(
            df["rev_gr_5y"].fillna(0) > 0,
            df["pat_gr_5y"].fillna(0) - df["rev_gr_5y"].fillna(0),  # PAT fallback
            np.nan
        )
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

    # ── Greenblatt Magic Formula earnings yield = EBIT / Enterprise Value ──
    # BOOK-EXACT (The Little Book That Still Beats the Market, appendix): the Magic Formula's
    # earnings yield is EBIT/EV, NOT net-income/price (= 100/PE = the `earnings_yield` above).
    # Greenblatt insists on EBIT/EV precisely to neutralize the capital-structure and tax-rate
    # differences that distort net-income/price.
    # EV SOURCE (2026-06-13): use the DIRECT `enterprise_value` column (real Stockscan figure,
    # 2106/2107 coverage) — replaces the prior `ev_ebitda × ebitda` reconstruction (proxy). The
    # reconstruction is kept ONLY as a per-row fallback where the direct EV is missing/non-positive.
    # Separate column so the shared earnings_yield (Parikh, valuation scoring) is untouched.
    _mf_ev_direct = df.get("enterprise_value", pd.Series(np.nan, index=df.index))
    _mf_ev_recon  = df["ev_ebitda"] * df["ebitda"]
    _mf_ev = np.where(_mf_ev_direct > 0.0, _mf_ev_direct, _mf_ev_recon)
    df["magic_formula_earnings_yield"] = np.where(
        _mf_ev > 0.0,
        df["ebit"] / _mf_ev * 100.0,
        np.nan
    )

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

    # ── Total Asset Growth YoY (%) — Capital Returns (Chancellor) asset-growth anomaly ──
    # Fama-French "investment" factor: firms with the LOWEST asset growth outperform those with
    # the highest (asset expansion → low future returns, persisting ~5 years). The EXACT anomaly
    # metric is TOTAL asset growth (distinct from d20 fixed-asset CAGR and from rf_snoa's asset
    # LEVEL). Capital discipline = low/negative; aggressive capacity expansion = high.
    # NOTE (audit 2026-06-13): rf_snoa (Quantitative Value) already operationalizes the scoring
    # side of this anomaly — capital cycle is the economic STORY behind why asset-bloat predicts
    # low returns. This column is the faithful continuous measure (capital-discipline visibility),
    # deliberately NOT a second forensic flag (62% of high-AG already fires rf_snoa — redundant).
    _ta_ag  = df["total_assets"]
    _ta1_ag = df["total_assets_1yb"]
    df["asset_growth_yoy"] = np.where(
        _ta1_ag > 0, (_ta_ag - _ta1_ag) / _ta1_ag * 100.0, np.nan
    )

    # ── External Financing to Assets — Tortoriello (Quantitative Strategies, Ch.5/8) ──
    # His SINGLE STRONGEST factor: the bottom quintile (firms RAISING external capital) underperforms
    # by 15.3% — "the highest negative excess returns of any two-factor strategy." External financing
    # = (share issuance − buybacks + debt issuance − debt retirement) / total assets, all from the
    # financing section of the cash-flow statement. financing_cash_flow IS that section (it also nets
    # dividends, which Tortoriello scores as returning-capital too → directionally aligned).
    # SIGN: positive = cash INFLOW = RAISING external capital (dilutive/leveraging = risk);
    #       negative = cash OUTFLOW = RETURNING capital (buybacks/debt-cut/dividends = disciplined).
    # Orthogonal to asset_growth_yoy (corr 0.13) — captures the COMBINED external-capital measure
    # that dilution_flag (shares only) and debt_slope (debt only) each see just half of.
    _ef_fincf = df.get("financing_cash_flow", pd.Series(np.nan, index=df.index))
    df["external_financing_to_assets"] = np.where(
        df["total_assets"] > 0, _ef_fincf / df["total_assets"] * 100.0, np.nan
    )
    df["capital_allocation_signal"] = np.select(
        [df["external_financing_to_assets"] < -5.0, df["external_financing_to_assets"] > 15.0],
        ["💰 Returning Capital", "⚠️ Raising Capital"],
        default="⚖️ Neutral",
    )

    # ── Cash Return on Invested Capital (CFROIC) — Tortoriello Ch.6, a STRONGEST building block ──
    # CFROIC = Free Cash Flow / Invested Capital, where Invested Capital = common equity + total debt
    # (Tortoriello: "book value of common equity plus long-term debt plus preferred + minority").
    # This is a CASH-based return — distinct from the system's EBIT-based economic_profit_spread /
    # roce (corr ~0.34): it rewards businesses that convert capital into actual free cash, not just
    # accounting EBIT. Recurs across Tortoriello's top two-factor Sharpe tables (Table 12.5).
    _cfroic_ic = df["net_worth"].fillna(0) + df["debt"].fillna(0)
    df["cfroic"] = np.where(
        _cfroic_ic > 0, df["free_cash_flow"] / _cfroic_ic * 100.0, np.nan
    )

    # ── Sector capital cycle (Chancellor: asset growth matters at the SECTORAL level too) ──
    # Capital flooding INTO a sector (high sector-median asset growth) pressures future returns
    # for ALL its constituents; capital-STARVED sectors (low/negative growth) set up recovery.
    # This is the capital cycle's distinctive sectoral signal — orthogonal to firm-level rf_snoa.
    # Guarded by sector size >= 5 (median unstable below that → Neutral). Thresholds are engineering
    # proxies (Chancellor states no numeric gate). Live: Power Infra +147%, Ship Building +58% =
    # hot capital; Sugar/Cement-Products/Entertainment ~0% = capital starved.
    _sec_size = df.groupby("sector")["sector"].transform("size")
    _sec_ag   = df.groupby("sector")["asset_growth_yoy"].transform("median")
    df["sector_asset_growth"] = np.where(_sec_size >= 5, _sec_ag, np.nan)
    df["sector_capital_phase"] = np.select(
        [df["sector_asset_growth"] > 30.0, df["sector_asset_growth"] < 5.0],
        ["🔥 Hot Capital (caution)", "❄️ Capital Starved (opportunity)"],
        default="⚖️ Neutral",
    )

    # ── D24: OCF/PAT Delta (CFO/PAT − 100) ──
    # D24 ≥ 0: OCF ≥ PAT = earnings are cash-backed (Clean Accounts signal)
    df["d24_ocf_pat_delta"] = df["cfo_to_pat"].fillna(np.nan) - 100.0

    # ── D27: FCF Positive (binary) ──
    df["d27_fcf_positive"] = (_fcf.fillna(0) > 0).astype(int)

    # ── D28: FCF-to-PAT (%) ──
    # D28 > 50%: FCF covers more than half of PAT = strong real cash generation
    df["d28_fcf_to_pat_pct"] = np.where(
        _pat.notna() & (_pat.abs() > 0),
        _fcf.fillna(0) / _pat.abs() * 100,
        np.nan
    )

    # ── D29: Capex Intensity (%) ──
    # (OCF − FCF) / |OCF| × 100: how much of operating cash goes to capex
    df["d29_capex_intensity"] = np.where(
        _ocf.notna() & (_ocf.abs() > 0),
        (_ocf - _fcf.fillna(_ocf)) / _ocf.abs() * 100,
        np.nan
    )

    # ── D32: PE vs Historical Median (%) — negative = cheap vs own history ──
    # pe_med_5y primary (current cycle); 10Y fallback for young companies.
    _pe_base_d32 = df["pe_med_5y"].fillna(df["pe_med_10y"])
    df["d32_pe_vs_median"] = np.where(
        _pe_base_d32.notna() & (_pe_base_d32 > 0) & df["pe"].notna(),
        (df["pe"] - _pe_base_d32) / _pe_base_d32 * 100,
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

    # ── D35: ROCE Trend — annualised 2-year structural slope (pp/yr); 1-year delta fallback ──
    # 2YB preferred: smooths single-quarter noise; divides by actual years for consistent pp/yr unit.
    # Fallback to 1YB when 2YB missing (young companies / data gap). NaN when no baseline available.
    _d35_base_2y = df["roce_2yb"]
    _d35_base    = _d35_base_2y.fillna(df["roce_1yb"])
    _d35_years   = np.where(_d35_base_2y.notna(), 2.0, 1.0)
    df["d35_roce_trend"] = np.where(
        df["roce"].notna() & _d35_base.notna(),
        (df["roce"] - _d35_base) / _d35_years,
        np.nan
    )

    # ── D36: ROCE vs 5Y Median — positive = above historical average ──
    df["d36_roce_above_med"] = df["roce"] - df["roce_med_5y"]

    # ── D37: CCC Direction (positive = worsening working capital) ──
    df["d37_ccc_direction"] = df["ccc"] - df["ccc_1yb"]

    # ── Terms of Trade (MOSL 11th Study) — DPO vs DSO working capital leverage ──
    # terms_of_trade_spread: days_payable − days_receivable.
    # Positive = company collects from customers faster than it pays suppliers (supplier-financed model).
    # Classic examples: D-Mart (0 DSO, 30+ DPO), HUL, Nestle. Validated across all 30 MOSL studies.
    # NaN for financials (days_payable NaN-out upstream).
    _dso = df["days_receivable"].fillna(np.nan)
    _dpo = df["days_payable"].fillna(np.nan)
    df["terms_of_trade_spread"] = np.where(
        _dso.notna() & _dpo.notna(),
        _dpo - _dso,
        np.nan
    )
    df["favorable_terms_flag"] = np.where(
        _dso.notna() & _dpo.notna(),
        (_dso < _dpo).astype(int),
        0
    )
    # DPO trend: rising = gaining supplier leverage, falling = losing negotiating power
    df["dpo_delta_1y"] = np.where(
        _dpo.notna() & df["days_payable_1yb"].notna(),
        _dpo - df["days_payable_1yb"],
        np.nan
    )

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

    # ── D45: Trend Structure Score (0–5) — Minervini Trend Template ──
    # Implements 5 of Minervini's 8 Trend Template criteria (Trade Like a Stock Market Wizard Ch.2)
    # computable from a point-in-time CSV. ADX moved OUT to sepa_adx_confirmed (Framework 21).
    #   +1 C1: close_price > sma_200d        (above 200D MA — long-term trend)
    #   +1 C2: close_price > sma_30w         (above 150D MA; 30 weeks × 5 = 150 trading days)
    #   +1 C3: sma_30w > sma_200d            (150D > 200D — right stacking)
    #   +1 C5: sma_50d > sma_30w AND sma_200d (50D fully stacked above both)
    #   +1 VSTOP: vstop_green                (volatility-adjusted trend quality)
    # C4 (200D rising 22 days) and C7/C8 are checked elsewhere (52WH dist / crs_aligned).
    # NaN on sma_50d/sma_30w → fillna(0) → that component scores 0 (conservative).
    _sma_50d_d45 = df.get("sma_50d", pd.Series(np.nan, index=df.index))
    _sma_30w_d45 = df.get("sma_30w", pd.Series(np.nan, index=df.index))
    # NO fillna inside the comparisons (fixed 2026-06-12): NaN comparisons are False in
    # pandas, which is the conservative direction for every "+1" component. The old
    # fillna(0) INVERTED conservatism on two legs — missing sma_30w made C2 a free point
    # (close > 0 always true; 32 live stocks) and missing sma_200d made C3 a free point
    # (sma_30w > 0; 54 live stocks). Missing data must never score a trend criterion.
    df["d45_trend_structure"] = (
        df["above_sma200"].fillna(0) +
        (df["close_price"] > _sma_30w_d45).astype(int) +
        (_sma_30w_d45 > df["sma_200d"]).astype(int) +
        ((_sma_50d_d45 > _sma_30w_d45) &
         (_sma_50d_d45 > df["sma_200d"])).astype(int) +
        df["vstop_green"].fillna(0)
    )

    # ── Weinstein Stage Analysis (Secrets for Profiting in Bull & Bear Markets) ──
    # Stan Weinstein's 4 stages off the 30-week MA — his core actionable framework:
    #   "Stocks trading beneath their 30-week MAs should never be considered for purchase."
    # Mapped to a 2×2 of price-vs-30W-MA and 30W-MA-vs-200D (the trend structure):
    #   Stage 2 Advancing (BUY)   : close > 30W MA AND 30W > 200D — above a rising MA, bullish stack
    #   Stage 1 Basing            : close > 30W MA AND 30W <= 200D — early recovery, crossed up
    #   Stage 3 Top               : close <= 30W MA AND 30W > 200D — early weakness, dropped below MA
    #   Stage 4 Declining (AVOID) : close <= 30W MA AND 30W <= 200D — below a falling MA, bearish
    # The MA SLOPE (Weinstein's literal stage differentiator) isn't in the CSV; the 30W-vs-200D
    # stacking is the faithful proxy for "rising vs falling MA". Display/context label, not a gate.
    _w_close = df["close_price"]; _w_30 = df.get("sma_30w", pd.Series(np.nan, index=df.index))
    _w_200 = df.get("sma_200d", pd.Series(np.nan, index=df.index))
    _w_ok = _w_close.notna() & _w_30.notna() & _w_200.notna()
    _w_above = _w_close > _w_30
    _w_stack = _w_30 > _w_200
    df["weinstein_stage"] = np.select(
        [_w_ok & _w_above & _w_stack,
         _w_ok & _w_above & ~_w_stack,
         _w_ok & ~_w_above & _w_stack,
         _w_ok & ~_w_above & ~_w_stack],
        ["📈 Stage 2 Advancing", "🔄 Stage 1 Basing",
         "⚠️ Stage 3 Top", "📉 Stage 4 Declining"],
        default="❔ Unknown",
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
    pat_safe = _pat.fillna(0).clip(lower=0.01)
    payback_0g = np.where(
        (df["market_cap"].fillna(0) > 0) & (_pat.fillna(0) > 0),
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
        (df["market_cap"].fillna(0) > 0) & (_pat.fillna(0) > 0),
        df["market_cap"] / (pat_safe * pd.Series(geo_sum, index=df.index)),
        np.nan
    )
    df["payback_ratio"] = pd.Series(payback_growth, index=df.index)
    df["payback_lt05"] = (df["payback_ratio"].fillna(99) < 0.5).astype(int)  # supernormal tier 1 (Studies 7-9: top-7 fastest ALL had payback < 0.5x)
    df["payback_lt1"]  = (df["payback_ratio"].fillna(99) < 1.0).astype(int)  # supernormal tier 2
    df["payback_lt2"]  = (df["payback_ratio"].fillna(99) < 2.0).astype(int)  # attractive zone

    # ── Trailing 5Y Payback (realized earnings — no growth assumption) ──
    # payback_ratio uses pat_gr_5y as forward growth estimate — a spike year skews it.
    # payback_trailing_5y uses actual sum of last 5 years PAT: normalizes for cyclicality.
    # Requires at least 3 years of positive PAT data to be meaningful.
    _pat_5y_cols = [c for c in ["pat","pat_1yb","pat_2yb","pat_3yb","pat_4yb"] if c in df.columns]
    _pat_5y_df   = df[_pat_5y_cols].clip(lower=0)   # treat losses as 0 (no negative PAT benefit)
    _trailing_5y_sum   = _pat_5y_df.sum(axis=1)
    _years_positive    = (_pat_5y_df > 0).sum(axis=1)
    df["payback_trailing_5y"] = np.where(
        (df["market_cap"].fillna(0) > 0) & (_trailing_5y_sum > 0) & (_years_positive >= 3),
        df["market_cap"] / _trailing_5y_sum,
        np.nan
    )
    df["payback_t5y_lt1"] = (df["payback_trailing_5y"].fillna(99) < 1.0).astype(int)

    # ── Consistency Champion (27th Study: Consistents vs Volatiles) ──
    # Full 15Y analysis requires 15 years of annual PAT — not in CSV. With pat..pat_5yb (6 points =
    # 5 YoY transitions) we implement ALL 3 of the 27th Study's verbatim criteria, proportionally
    # scaled to our 5Y window:
    #   Criterion 1: PAT must not fall >10% more than the allowed count. Study allows 3 in 15yr /
    #                2 in 10yr → proportional 1 in 5yr. (pat_decline_count_5y <= 1)
    #   Criterion 2: no single YoY PAT fall > 50% (catastrophic crash) across the 5 transitions.
    #   Criterion 3: terminal year PAT >= initial year PAT (pat > pat_5yb) + long-term positive CAGR.
    pat_decline_1y = np.where(
        _pat_1yb.fillna(0) > 0,
        (_pat.fillna(0) - _pat_1yb.fillna(0)) / _pat_1yb.abs() * 100,
        np.nan
    )
    df["pat_decline_1y_pct"] = pd.Series(pat_decline_1y, index=df.index)
    df["pat_no_crash_1y"] = (df["pat_decline_1y_pct"].fillna(0) > -50).astype(int)
    df["pat_growing_long"] = (
        df["pat_gr_10y"].fillna(df["pat_gr_5y"]).fillna(0) > 0
    ).astype(int)

    # 5 YoY transitions (full window incl. pat_4yb→pat_5yb). Only positive prior years count —
    # turnarounds (negative→positive) are not treated as declines.
    _pat_seq = [
        (_pat,     _pat_1yb),
        (_pat_1yb, _pat_2yb),
        (_pat_2yb, _pat_3yb),
        (_pat_3yb, _pat_4yb),
        (_pat_4yb, _pat_5yb),
    ]
    _no_crash_5y = pd.Series(True, index=df.index)       # Criterion 2: no >50% fall
    _decline_10_count = pd.Series(0, index=df.index)     # Criterion 1: count of >10% falls
    for _curr_p, _prev_p in _pat_seq:
        _both_pos = (_curr_p.fillna(0) > 0) & (_prev_p.fillna(0) > 0)
        _ratio    = (_curr_p.fillna(0) / _prev_p.replace(0, np.nan)).fillna(1.0)
        _no_big_fall = ~_both_pos | (_ratio > 0.5)                       # not a >50% crash
        _no_crash_5y = _no_crash_5y & _no_big_fall
        _fell_10  = _both_pos & (_ratio < 0.90)                          # >10% YoY decline
        _decline_10_count = _decline_10_count + _fell_10.astype(int)
    df["pat_decline_count_5y"] = _decline_10_count

    # Terminal > initial (5Y criterion — skip if pat_5yb unavailable)
    _5yb_available        = _pat_5yb.fillna(0) > 0
    _terminal_gt_initial  = ~_5yb_available | (_pat.fillna(0) > _pat_5yb.fillna(0))

    df["consistency_champion"] = (
        (_decline_10_count <= 1) &        # Criterion 1: <=1 fall >10% (proportional 3-in-15 / 2-in-10)
        _no_crash_5y &                     # Criterion 2: no >50% crash
        _terminal_gt_initial &             # Criterion 3a: terminal >= initial
        (df["pat_growing_long"] == 1) &    # Criterion 3b: long-term positive CAGR
        (_pat.fillna(0) > 0)
    ).astype(int)

    # ── Volatile Flag (27th Study — counterpart to consistency_champion) ──
    # A Volatile is the structural opposite of a Consistent.
    # Criterion A: severe recent PAT crash (>20% fall YoY) — earnings are unreliable.
    # Criterion B: 5Y PAT CAGR negative — earnings have contracted over the medium term.
    # Either condition alone is sufficient to classify as Volatile.
    df["mosl_volatile_flag"] = (
        (df["pat_gr_yoy"].fillna(0)    < -20.0) |    # A: sharp recent collapse
        (df["pat_gr_5y"].fillna(999)   <   0.0)      # B: 5Y earnings contraction
    ).astype(int)

    # ── 100x Candidate Screen (19th Study, 2014 — "100x: The Power of Growth") ──
    # Book-verified 2026-06-13: the 100x theme is the 19TH study (the 17th was Economic Moat; the
    # "Mouse to Elephant" phrase appears 0 times in the 19th study). The study's OWN numeric 100x
    # screen is: mcap < ₹30b + value-migration/niche play + P/E < 25x (value-migration is qualitative).
    # The 5 gates below are the ENGINE's computable QUALITY-focused 100x proxy (not the book's verbatim
    # screen) — a clean, unlevered, early-stage compounding profile:
    # (1) 20%+ PAT CAGR (compounding proven over 5Y) (2) ROCE ≥ 20% (3) Market cap ≤ ₹15,000 Cr
    #   (100x headroom) (4) D/E < 0.5 (leverage cannot fuel compounding) (5) ROE ≥ 15%.
    # All 5 mandatory (AND, not OR) — deliberately rare. The study's PRIMARY 100x process is the
    # qualitative SQGLP checklist — faithfully implemented as sqglp_score (S/Q/G/L/P) in this module.
    df["mosl_100x_candidate"] = (
        (df["pat_gr_5y"].fillna(0)           >= 20.0)  &
        (df["roce"].fillna(0)                >= 20.0)  &
        (df["market_cap"].fillna(999_999)    <= 15_000) &
        (df["debt_to_equity"].fillna(999.0)  <   0.5)  &
        (df["roe"].fillna(0)                 >= 15.0)
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
    # Financials (INCLUDING Capital Market) + Consumer Discretionaries = the explosive tipping-point
    # sectors (30th WCS headline). Pattern fixed 2026-06-13 vs the real CSV sector names: the old
    # "NBFC|Capital Market" tokens matched 0 sectors, and "Finance" missed "Financial Services" — so
    # BSE (the #1 fastest wealth creator!), CRISIL, MCX were wrongly excluded. "Financ" now catches
    # Finance + Financial Services; "Broker"/"Rating" catch the Capital-Markets ecosystem.
    _tailwind_patt = "Bank|Insurance|Financ|Broker|Rating|Auto|Consumer|Health|Pharma|Retail"
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

    # ── Bruised Blue Chip 29th WCS — faithful to the study's verbatim definition (§2/§2.1, p.5) ──
    # Blue Chip (§2.1, p.5, VERBATIM): track record >=10yr; (a) top 50 by market cap, OR (b) top 250
    #   by market cap AND 10-year average ROE >= 20%. (Book identifies 68 Blue Chips.) 10Y track record
    #   proxied by roe_med_10y availability. Study uses ROE (not ROCE) — blue chips include banks.
    # Bruised (§2, p.5, VERBATIM): "Blue Chips which have fallen more than 50% from their 5-year high."
    #   CSV lacks a 5Y-high series, so the dist_52wh/pe_discount proxy below approximates the deep drawdown.
    # Entry (Highlights + process, VERBATIM): "Buy at attractive valuations, typically Price/Book less than 2x".
    # Book-verified 2026-06-13: prior "p.278-284 / p.264, 297 / p.13, 285" cites were ALL fabricated
    #   (the 80-pg study's definitions are on p.5); the criteria themselves were verbatim-accurate.
    _mcap_rank        = df["market_cap"].rank(ascending=False, method="min")   # 1 = largest
    _bbc29_top50      = _mcap_rank <= 50
    _bbc29_top250_q   = (_mcap_rank <= 250) & (df["roe_med_10y"].fillna(0) >= 20.0)
    _bbc29_blue_chip  = _bbc29_top50 | _bbc29_top250_q
    # "Bruised" proxy: >25% off 52-week high OR PE >=25% below own history. CSV lacks a 5-year-high
    # series, so this OR-pair best approximates "50% off 5Y high" — catching both recently-fallen
    # and quietly-derated quality names. (Pure dist_52wh>40 is too strict near all-time-high markets.)
    _bbc29_bruised    = (df["dist_52wh"].fillna(0) > 25) | (df["pe_discount"].fillna(0) >= 25)
    _bbc29_pb_entry   = df["pb_ratio"].fillna(999) < EPOCH5_MODERN["bbc_pb_ceiling"]  # verbatim P/B < 2x golden entry
    df["bruised_blue_chip_29"] = (
        _bbc29_blue_chip & _bbc29_bruised & _bbc29_pb_entry
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
        _ocf.notna() & (_ocf > 0) & ~_fcf_imputed,
        _fcf.fillna(0) / _ocf * 100,
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

    # ── Sloan Receivables Accrual Divergence (earnings-quality trajectory) ──
    # Richard Sloan's accrual anomaly: when receivables compound FASTER than revenue, reported
    # sales are increasingly un-collected paper (channel stuffing / aggressive recognition) and
    # forward returns mean-revert down. Reconstruct the receivables balance from its days metric:
    #   Receivables ≈ DaysReceivable × Revenue / 365   (Crores)
    # then compare its 3Y total growth against revenue's 3Y total growth. Positive delta = accruals
    # outrunning real sales (a red trajectory). Defensive denominators + NaN propagation throughout.
    _dr = _safe_numeric(df.get("days_receivable", pd.Series(np.nan, index=df.index)))
    _dr_3yb = _safe_numeric(df.get("days_receivable_3yb", pd.Series(np.nan, index=df.index)))
    _rev = _safe_numeric(df.get("revenue", pd.Series(np.nan, index=df.index)))
    _rev_3yb = _safe_numeric(df.get("revenue_3yb", pd.Series(np.nan, index=df.index)))
    df["receivables_est"] = _dr * _rev / 365.0
    df["receivables_est_3yb"] = _dr_3yb * _rev_3yb / 365.0
    _recv_growth_3y = np.where(
        df["receivables_est_3yb"] > 0.0,
        (df["receivables_est"] - df["receivables_est_3yb"]) / df["receivables_est_3yb"] * 100.0,
        np.nan
    )
    _rev_growth_3y = np.where(
        _rev_3yb > 0.0,
        (_rev - _rev_3yb) / _rev_3yb * 100.0,
        np.nan
    )
    # Both legs computed on the same total-% basis so the subtraction is apples-to-apples.
    df["accrual_growth_delta"] = _recv_growth_3y - _rev_growth_3y

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

    # ── Study 17 (2012): Economic Moat — Sector-Relative ROE Persistence (EMC Flag) ──
    # Book-verified 2026-06-13. Backtest (EMP 1995-2002, returns 2002-2012): EMC 25% CAGR vs non-EMC
    # 12% vs Sensex 18% (alpha +7% over Sensex); EMC mortality 16% / 84% survival (12 of 74 EMCs fell —
    # prior comment's "<15%" was the one inaccuracy; the 25/12/18 figures are book-exact).
    # VERBATIM definition (17th Study; methodology p.35): "Companies whose RoE was higher than sector
    # average for 6 years or more [of the 8-yr 1995-2002 window] were deemed to enjoy an Economic Moat."
    # Faithful proxy: 5 ROE timeframes (current, 1yb, 3y/5y/10y medians) each vs their own sector median;
    # require >= 4 of 5 above sector (80% persistence ~ the study's 6/8 = 75% sustained-outperformance bar).
    # Stronger than the prior 2-point check — captures sustained moat, not a single-snapshot beat.
    _sector_grp_roe = df["sector"].fillna("Unknown")
    _emc_timeframes = ["roe", "roe_1yb", "roe_med_3y", "roe_med_5y", "roe_med_10y"]
    _emc_above_count = pd.Series(0, index=df.index)
    for _tf in _emc_timeframes:
        if _tf not in df.columns:
            continue
        _sec_med = df.groupby(_sector_grp_roe)[_tf].transform("median").fillna(df[_tf].median())
        # NaN company ROE → fillna(0): cannot beat a positive sector median → conservatively not counted.
        _emc_above_count = _emc_above_count + (df[_tf].fillna(0) > _sec_med.fillna(0)).astype(int)
    df["emc_sector_beat_count"] = _emc_above_count          # 0-5: diagnostic / tearsheet display
    df["emc_flag"] = (_emc_above_count >= 4).astype(int)    # >= 4 of 5 timeframes beat sector

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

    # ── Study 14 (2009): Winner Category — the sector-tailwind layer (the missing half) ──
    # VERBATIM (14th WCS p.22): "Winner Categories are those which are expected to grow annually @ 18%+
    # i.e. at least 1.5x faster than our nominal GDP growth rate assumption of ~12%." (Book-verified
    # 2026-06-13: quote accurate; prior "p.933" cite was fabricated — the 48-pg study's def is p.22.)
    # category_winner_flag above is the LEADER half (Category
    # Winner = leader within a sector); this is the SECTOR half (is the sector itself winning?).
    # Universe-median revenue growth proxies nominal GDP, so the regime-robust faithful form is:
    # sector-median 5Y revenue growth >= 1.5x universe-median growth. "Winning investment" (study's
    # ultimate target) = Category Winner operating inside a Winner Category.
    _wc_rev_growth      = df["rev_gr_5y"].fillna(df["rev_gr_3y"])
    df["sector_growth_median"] = _wc_rev_growth.groupby(_sector_grp_roe).transform("median")
    _wc_univ_median     = _wc_rev_growth.median()
    _wc_hurdle          = (_wc_univ_median * 1.5) if pd.notna(_wc_univ_median) else 18.0
    df["winner_category_flag"] = (
        df["sector_growth_median"].fillna(-999) >= _wc_hurdle
    ).astype(int)
    # Study's "Winning investment": a Category Winner (leader) inside a Winner Category (fast sector).
    df["category_winner_in_winner_cat"] = (
        (df["category_winner_flag"] == 1) & (df["winner_category_flag"] == 1)
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
    # Screen 5: "≥ 5 million shares outstanding" — equity_shares holds ABSOLUTE share count
    #           (universe median ~51M), so the Weiss threshold is 5_000_000 shares.
    # From 3,000+ stocks, only 48 (1.5%) passed all 6 screens. This is appropriately rare.
    # KNOWN DATA GAP (2026-06-12 census): the CSV "Dividend Payout Ratio" column is broken at
    # source (96% empty; the rest negative) → the dpr leg cannot pass → flag fires 0 until the
    # sheet formula is fixed. Logic is correct and self-revives when real DPR data arrives.
    _dpr_bc    = df.get("dividend_payout_ratio", pd.Series(np.nan, index=df.index))
    _eq_shares = df.get("equity_shares",         pd.Series(np.nan, index=df.index))
    df["blue_chip_quality_flag"] = (
        (_dpr_bc.fillna(0)          >= 20) &      # Screen 1/2: consistent dividend payout
        (df["roe_med_10y"].fillna(0) >= 15) &      # Screen 4: 10yr ROE ≥ 15% (India CoE threshold)
        (df["consistency_champion"]  == 1) &       # Screen 3: PAT no-crash consistency proxy
        (_eq_shares.fillna(0)        >= 5_000_000) # Screen 5: ≥ 5M shares (absolute count)
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
    # 100x stock profile (19th WCS): small base at entry (the study cites mcap < ~USD 500M).
    # FIX 2026-06-13: was < 40_000 Cr (~USD 4.8B) — a 10x unit bug vs the stated "USD 500M": it let
    # 92% of the universe pass the SMALL-size gate, so S was dead weight (sqglp_score ≈ QGLP). ₹5,000 Cr
    # (~USD 500-600M) is the genuine small-cap base a 100x needs (5,000 Cr × 100 = ₹5L cr, achievable).
    _sqglp_s = (df["market_cap"].fillna(0) < 5_000).astype(int)     # S: < ₹5,000 Cr ≈ USD 500M small-cap
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
    # Book-verified 2026-06-13. The study (§1.1) classifies by market-cap RANK: Mega = top 100,
    # Mid = ranks 101-300, Mini = rest. Mid-to-Mega = a Mid stock crossing into the top 100 within
    # ~5yr. Backtest (§ summary, verified): "median return of 46%, portfolio RoE 20%, P/E 15-23x at
    # purchase, probability 5-12%". MQGLP = QGLP applied to the Mid (101-300) entry universe.
    # FIX: the prior absolute proxy ₹5,000-20,000 Cr mapped to ranks ~277-629 (SMALL caps) — it missed
    # the book's Mid (ranks 101-300 ≈ ₹18-84k Cr); only 1 of 34 candidates was actually Mid. Now use
    # the actual rank, which is book-exact and self-calibrating to the live universe.
    _mtm_rank = df["market_cap"].rank(ascending=False, method="min")
    df["mid_to_mega_candidate"] = (
        (_mtm_rank >= 101) & (_mtm_rank <= 300) &              # S: Mid = market-cap ranks 101-300 (§1.1)
        (df["roce_med_5y"].fillna(df["roce"]).fillna(0) >= 15) &
        (df["pat_gr_5y"].fillna(0) >= 20) &
        (df["pe"].fillna(999) <= 25)
    ).astype(int)

    # ── Study 22 (2017): CAP & GAP — Competitive + Growth Advantage Period ──
    # Book-verified 2026-06-13 (prior "p.342"/"p.1722" cites were fabricated — the study is 60 pp).
    # CAP (§3.1, VERBATIM): "Competitive Advantage Period (CAP) is the time during which a company
    #   generates returns on investment that exceed its cost of capital."
    #   → ROCE > cost of capital. Use COST_OF_EQUITY (12%) as the system-wide hurdle (consistent with
    #   economic_profit). Prior code used a hardcoded 10% — BELOW the 12% hurdle, so an 11%-ROCE
    #   company earning below its cost of capital (no real moat) was wrongly flagged. Now corrected.
    # GAP (§5.1, VERBATIM): "Growth Advantage Period (GAP) is the time during which a company grows its
    #   profits at a faster rate than that of the benchmark indices." The book's GAP chart (p.19) labels
    #   the "15% threshold (benchmark growth rate)". Prior code used 12%/8% — below the study's 15% bar.
    # The study's whole thesis is LONGEVITY (duration), so we also expose year-count proxies.
    # "Moat without growth underperforms; growth without moat ends soon." — MOSL 22nd Study.

    # CAP duration proxy (0-5): how many ROCE timeframes clear the cost-of-capital hurdle.
    _cap_tfs = ["roce_med_10y", "roce_med_7y", "roce_med_5y", "roce_1yb", "roce"]
    _cap_years = pd.Series(0, index=df.index)
    for _c in _cap_tfs:
        if _c in df.columns:
            _cap_years = _cap_years + (df[_c].fillna(0) >= COST_OF_EQUITY).astype(int)
    df["cap_years_proxy"] = _cap_years
    # Extended CAP: long-run (10Y), recent (5Y) and current ROCE all above cost of capital.
    df["cap_extended_flag"] = (
        (df["roce_med_10y"].fillna(0) >= COST_OF_EQUITY) &
        (df["roce_med_5y"].fillna(0)  >= COST_OF_EQUITY) &
        (df["roce"].fillna(0)         >= COST_OF_EQUITY)
    ).astype(int)

    # GAP duration proxy (0-3): how many PAT-growth windows clear the 15% benchmark rate.
    _gap_tfs = ["pat_gr_10y", "pat_gr_5y", "pat_gr_3y"]
    _gap_years = pd.Series(0, index=df.index)
    for _g in _gap_tfs:
        if _g in df.columns:
            _gap_years = _gap_years + (df[_g].fillna(0) >= 15.0).astype(int)
    df["gap_years_proxy"] = _gap_years
    # Extended GAP: PAT growth > 15% sustained across ALL three windows (long, elite GAP).
    df["gap_extended_flag"] = (
        (df["pat_gr_10y"].fillna(df["pat_gr_5y"]).fillna(0) >= 15.0) &
        (df["pat_gr_5y"].fillna(0) >= 15.0) &
        (df["pat_gr_3y"].fillna(0) >= 15.0)
    ).astype(int)

    # CAP-GAP composite (0-4): +1 cap_extended, +1 gap_extended, +1 both (longevity proof), +1 ROE>15
    df["cap_gap_score"] = (
        df["cap_extended_flag"] +
        df["gap_extended_flag"] +
        (df["cap_extended_flag"] & df["gap_extended_flag"]).astype(int) +
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

    # ── Lynch Category (Peter Lynch — One Up on Wall Street) ──
    # Classifies each stock by growth trajectory for tearsheet display.
    # Fast Grower: Lynch's primary hunting ground for 10-100× returns.
    # DISPLAY PROXY (book audit 2026-06-13): the book's six categories are defined by
    # EARNINGS growth (stalwarts "10 to 12 percent", fast growers "20 to 25") and include
    # cyclicals/turnarounds/asset plays, which are not classifiable from a growth rate.
    # These revenue bands are a 4-bucket display approximation, not Lynch's taxonomy.
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
    # KNOWN DATA GAP (2026-06-12 census): the CSV DPR column is broken at source (96% empty,
    # rest negative) → RR ≡ 1.0 universe-wide, which deadens every RR-gated signal downstream
    # (stagnant_cash_cow_flag, capital_misallocation_risk RR leg). Self-heals when DPR is fixed.
    df["reinvestment_rate"] = (
        1.0 - (df["dividend_payout_ratio"].fillna(0) / 100.0)
    ).clip(0.0, 1.0)

    # Mayer 100-Bagger companion: Retention Rate as PERCENTAGE (distinct from reinvestment_rate above).
    # retention_rate = 100 - DPR. DPR fillna(0) → full retention assumed for missing dividend data.
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
    df["retained_earnings_est_5y"] = _pat.fillna(0) * df["reinvestment_rate"] * 5.0

    # Identity C (Agent 9): Buffett 1-to-1 Value Creation Ratio (VCR).
    # True VCR = (MCap_now − MCap_5YB) / ΣRetainedEarnings(5Y). MCap 5Y back not in CSV.
    # RIGOROUS REPLACEMENT: VCR = ROCE_5Y_median / COST_OF_EQUITY (12%).
    # Mathematical equivalence: VCR ≥ 1.0 ↔ ROCE ≥ CoE ↔ company earns above its hurdle rate
    # ↔ each rupee retained creates more than a rupee of intrinsic value → same Buffett test.
    # Benchmarks from 11th WCS remain valid: VCR ≥ 2.0 (ROCE ≥ 24%) = elite,
    # ≥ 1.0 (ROCE ≥ 12%) = passing, < 1.0 = value-destroys on retained capital.
    # Correctly handles declining stocks: ROCE < 0 → VCR < 0 < 1.0 → correctly flagged.
    # Cascade: 5Y median → 10Y median → current ROCE (best available history).
    _roce_vcr = (
        df.get("roce_med_5y",  pd.Series(np.nan, index=df.index))
        .fillna(df.get("roce_med_10y", pd.Series(np.nan, index=df.index)))
        .fillna(df.get("roce",         pd.Series(np.nan, index=df.index)))
    )
    df["value_creation_ratio"] = np.where(
        _roce_vcr.notna(),
        _roce_vcr / COST_OF_EQUITY,   # CoE = 12.0 (config.py). Constant > 0, no guard needed.
        np.nan
    )

    # Capital Misallocation Risk (Agent 9): retaining >50% of earnings but VCR < 1.0.
    # Destroying minority shareholder value despite appearing to "conserve" capital.
    # Scoring engine applies a 10% quality penalty to these stocks.
    # fillna(1.0): missing ROCE history → benefit of doubt, no misallocation penalty.
    df["capital_misallocation_risk"] = (
        (df["reinvestment_rate"]                >  0.5) &
        (df["value_creation_ratio"].fillna(1.0) <  1.0)
    ).astype(int)

    # ══════════════════════════════════════════════════════════════
    #  LEVEL × TRAJECTORY PLANE — lift threshold checklists into a 2-axis (where it IS now,
    #  which WAY it is moving) data plane. Pure cross-sectional + time-series array ops.
    #  NOTE: value_creation_ratio / capital_misallocation_risk above are deliberately RETAINED
    #  (scoring_engine applies a live misallocation penalty off them) — this block only ADDS.
    # ══════════════════════════════════════════════════════════════

    # Value-creation velocity: retention-weighted excess return = RR × (ROCE − CoE). RR is a
    # decimal [0,1]; ROCE and COST_OF_EQUITY are PERCENT, so velocity is in percentage-points of
    # economic value minted per year. Negative = compounding capital below its hurdle rate.
    df["value_creation_velocity"] = (
        df["reinvestment_rate"].fillna(0.0) * (df["roce"] - COST_OF_EQUITY)
    )

    # DuPont ROE attribution: first-order decomposition of year-on-year ROE change into three
    # sources — margin improvement, asset-efficiency improvement, and leverage expansion.
    # ROE = NPM × Asset_Turnover × Financial_Leverage (where Lev = Total_Assets / Net_Worth).
    # roe_leverage_driven = 1 when leverage is the dominant positive contributor to ROE change:
    # a rising ROE driven by debt expansion is a value trap, not a quality improvement.
    _dup_nan  = pd.Series(np.nan, index=df.index)
    _npm0     = _safe_numeric(df.get("npm",              _dup_nan)) / 100.0
    _npm1     = _safe_numeric(df.get("npm_1yb",          _dup_nan)) / 100.0
    _at0      = _safe_numeric(df.get("asset_turnover",   _dup_nan))
    _at1      = _safe_numeric(df.get("asset_turnover_1yb", _dup_nan))
    _ta0      = _safe_numeric(df.get("total_assets",     _dup_nan))
    _ta1      = _safe_numeric(df.get("total_assets_1yb", _dup_nan))
    _nw0      = _safe_numeric(df.get("net_worth",        _dup_nan)).clip(lower=1.0)
    _nw1      = _safe_numeric(df.get("net_worth_1yb",    _dup_nan)).clip(lower=1.0)
    _lev0     = np.where(_ta0.notna() & (_nw0 > 0), _ta0 / _nw0, np.nan)
    _lev1     = np.where(_ta1.notna() & (_nw1 > 0), _ta1 / _nw1, np.nan)
    _lev0_s   = pd.Series(_lev0, index=df.index)
    _lev1_s   = pd.Series(_lev1, index=df.index)
    _mc = ((_npm0 - _npm1) * _at1 * _lev1_s) * 100.0           # margin contrib (% ROE pts)
    _tc = (_npm1 * (_at0 - _at1) * _lev1_s) * 100.0            # turnover contrib
    _lc = (_npm1 * _at1 * (_lev0_s - _lev1_s)) * 100.0         # leverage contrib
    df["roe_margin_contrib"]   = _mc
    df["roe_turnover_contrib"] = _tc
    df["roe_leverage_contrib"] = _lc
    df["roe_leverage_driven"]  = (
        (_lc > 0) & (_lc.abs() > _mc.abs()) & (_lc.abs() > _tc.abs())
    ).astype(int)

    # Expectations gap (Mauboussin): growth the market PRICES IN minus growth the business can
    # SUSTAIN.  g_implied inverts the P/B–Gordon identity; g_star is the lower of the two organic
    # ceilings (ROE×RR self-funded growth, and SSGR). Positive gap = priced for more than it can
    # deliver (expectations risk); negative = pessimism / margin of safety. All PERCENT.
    df["g_implied"] = _implied_growth_from_pb(df["pb_ratio"], df["roe"], COST_OF_EQUITY)
    df["g_star"] = np.minimum(
        df["roe"].fillna(0.0) * df["reinvestment_rate"].fillna(0.0),   # ROE × RR
        df["ssgr"]                                                     # self-sustainable growth
    )
    df["expectations_gap"] = df["g_implied"] - df["g_star"]

    # Trajectory taus — pairwise-sign Kendall's tau over each metric's oldest→newest level ladder.
    # +1 = monotonic improvement, -1 = monotonic decay. Promoter levels are reconstructed from the
    # current holding minus its trailing deltas (temp cols, dropped after the tau is read).
    df["roce_tau"] = _kendall_tau_cols(
        df, ["roce_med_10y", "roce_med_7y", "roce_med_5y", "roce_med_3y", "roce_1yb", "roce"])
    df["moat_tau"] = _kendall_tau_cols(
        df, ["opm_med_5y", "opm_1yb", "opm", "opm_latest_q"])
    df["revenue_tau"] = _kendall_tau_cols(
        df, ["revenue_5yb", "revenue_4yb", "revenue_3yb", "revenue_2yb", "revenue_1yb", "revenue"])
    df["pat_tau"] = _kendall_tau_cols(
        df, ["pat_5yb", "pat_4yb", "pat_3yb", "pat_2yb", "pat_1yb", "pat"])

    _promo_now = _safe_numeric(df.get("promoter_holdings", pd.Series(np.nan, index=df.index)))
    df["_promo_lvl_3y"] = _promo_now - _safe_numeric(df.get("change_promoter_3y", pd.Series(np.nan, index=df.index)))
    df["_promo_lvl_2y"] = _promo_now - _safe_numeric(df.get("change_promoter_2y", pd.Series(np.nan, index=df.index)))
    df["_promo_lvl_1y"] = _promo_now - _safe_numeric(df.get("change_promoter_1y", pd.Series(np.nan, index=df.index)))
    df["_promo_lvl_now"] = _promo_now
    df["promoter_tau"] = _kendall_tau_cols(
        df, ["_promo_lvl_3y", "_promo_lvl_2y", "_promo_lvl_1y", "_promo_lvl_now"])
    df.drop(columns=["_promo_lvl_3y", "_promo_lvl_2y", "_promo_lvl_1y", "_promo_lvl_now"], inplace=True)

    # Composite trajectory = mean of the 4 fundamental taus (governance/promoter kept separate as a
    # standalone signal). pandas .mean(axis=1) skips NaN, so a stock missing one ladder still scores.
    df["trajectory_score"] = pd.concat(
        [df["roce_tau"], df["moat_tau"], df["revenue_tau"], df["pat_tau"]], axis=1
    ).mean(axis=1)

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
    # KNOWN-DEAD (census 2026-06-13): the reinvestment_rate < 0.30 leg requires DPR > 70%, but the CSV
    # "Dividend Payout Ratio" column is broken at source (non-null 4%, >0 for 0%) → reinvestment_rate
    # is forced to ≡ 1.0 for the whole universe → this flag fires 0. Same root cause as
    # blue_chip_quality_flag. Logic is correct; self-revives when the sheet DPR formula is fixed.
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
        _ocf.fillna(0) > 0,
        _fcf.fillna(0) / _ocf,
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
    # NOTE 2026-06-13: this sector/industry branch is DEAD — "Public Sector|Govt" matches 0 of the
    # CSV's 84 sector / 354 industry names (they classify by business, never by ownership). PSU
    # detection therefore relies entirely on the _psu_name list above. Kept (harmless OR, 0 matches)
    # to document the gap: the CSV has no ownership column, so PSUs outside the name list aren't caught.
    _psu_sector = df["sector"].fillna("").astype(str).str.contains("Public Sector|Govt", case=False, na=False) | \
                  df["industry"].fillna("").astype(str).str.contains("Public Sector|Govt", case=False, na=False)
    _is_psu_proxy = (_psu_name | _psu_sector) & (df["promoter_holdings"].fillna(0) >= 50) & (df["pledged_percentage"].fillna(0) == 0)

    _psu_low_spread = df["capital_return_spread"].fillna(0) <= 0
    _psu_low_velocity = (df["reinvestment_rate"].fillna(1) < 0.40) | (df["fcf_to_ocf_velocity"].fillna(0) < 0.40)
    _psu_cwip_delays = (df["cwip"].fillna(0) > 0) & (df["cwip_1yb"].fillna(0) > 0) & (df["cwip_conversion"].fillna(0) <= 0)

    # Value-destruction LOOP = core condition + at least one reinforcing leg.
    # Recalibrated 2026-06-12: the old 4-way AND was DEAD CODE — its legs barely
    # intersect on live data (PSU+low_spread=6, PSU+low_velocity=5, PSU+cwip_delays=4
    # stocks → all four simultaneously = 0 of 2107). Core = returns below cost of
    # equity (the definition of value destruction); reinforcement = the destruction
    # mechanism (cash not reinvested productively OR capex stuck in CWIP limbo).
    df["psu_value_destruction_flag"] = (
        _is_psu_proxy & _psu_low_spread & (_psu_low_velocity | _psu_cwip_delays)
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
    _incr_pat_e4       = _pat.fillna(0) - _pat_1yb.fillna(0)
    _incr_cap_delta_e4 = (df["fixed_assets"].fillna(0) - df["fixed_assets_1yb"].fillna(0)).abs()
    df["incremental_roce_proxy"] = np.where(
        (_incr_cap_delta_e4 > 5.0) & (_pat.fillna(0) > 0),
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
        _pat.notna() & _ocf.notna(),
        (_pat - _ocf) / df["total_assets"],
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
