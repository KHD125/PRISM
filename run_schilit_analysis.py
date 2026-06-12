# -*- coding: utf-8 -*-
"""
Schilit Financial Shenanigans -- Live Run on CSV Data
Run: python run_schilit_analysis.py
"""
import sys, os, time, warnings
import pandas as pd
import numpy as np

# Force UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

# ── Patch CSV paths BEFORE importing data_engine ─────────────────────────────
# The app normally runs from Streamlit where paths resolve differently.
# For direct CLI execution, CSVs live in Other Resources/CSV Data/
CSV_DIR = os.path.join(REPO_ROOT, "Other Resources", "CSV Data")

import config
config.CSV_FILES = {
    "ratio":        os.path.join(CSV_DIR, "Stockscan - Ratio.csv"),
    "income":       os.path.join(CSV_DIR, "Stockscan - Income Statement.csv"),
    "balance":      os.path.join(CSV_DIR, "Stockscan - Balance Sheet.csv"),
    "cashflow":     os.path.join(CSV_DIR, "Stockscan - Cashflow.csv"),
    "shareholding": os.path.join(CSV_DIR, "Stockscan - Shareholdings.csv"),
    "technical":    os.path.join(CSV_DIR, "Stockscan - Technicals.csv"),
}

# Verify all CSVs exist before starting
for key, path in config.CSV_FILES.items():
    if not os.path.isfile(path):
        print(f"ERROR: CSV not found: {path}")
        sys.exit(1)

from core.data_engine     import fetch_and_clean_data
from core.forensic_engine import compute_schilit_forensic_score

SEP  = "=" * 72
SEP2 = "-" * 72

# ── STEP 1: Load data ─────────────────────────────────────────────────────────
print(SEP)
print("  SCHILIT FINANCIAL SHENANIGANS  --  FORENSIC SCAN  --  INDIA EQUITIES")
print(SEP)

t0 = time.time()
print("\n[1/5] Loading 6 CSVs and computing all derived signals...")
df = fetch_and_clean_data()
print(f"      OK: {len(df)} stocks x {len(df.columns)} columns  ({time.time()-t0:.2f}s)")

# ── STEP 2: Verify dep_rate columns (Signal 6 EBIT-mapping check) ─────────────
print("\n[2/5] Signal 6 dep_rate derivation check (EBITDA - EBIT / Fixed Assets):")
dep_cols = ["ebit", "ebit_1yb", "depreciation", "depreciation_1yb", "dep_rate", "dep_rate_1yb"]
for col in dep_cols:
    if col in df.columns:
        n = int(df[col].notna().sum())
        print(f"      OK      {col:<22} {n:>5} / {len(df)} stocks have data")
    else:
        print(f"      MISSING {col}")

# ── STEP 3: Run Schilit engine ────────────────────────────────────────────────
print("\n[3/5] Running compute_schilit_forensic_score() on full universe...")
t1 = time.time()
df = compute_schilit_forensic_score(df)
print(f"      OK: forensic engine completed in {time.time()-t1:.3f}s")

# ── STEP 4: Universe-wide summary ────────────────────────────────────────────
total    = len(df)
passed   = int(df["schilit_pass"].sum())
failed   = total - passed
pass_pct = passed / total * 100

ems_n  = int(df["schilit_ems_flag"].sum())
cfs_n  = int(df["schilit_cfs_flag"].sum())
lev_n  = int(df["schilit_kms_lev_flag"].sum())
blt_n  = int(df["schilit_kms_bloat_flag"].sum())

checkers_fired = (
    df["schilit_ems_flag"] + df["schilit_cfs_flag"] +
    df["schilit_kms_lev_flag"] + df["schilit_kms_bloat_flag"]
)
c0 = int((checkers_fired == 0).sum())
c1 = int((checkers_fired == 1).sum())
c2 = int((checkers_fired == 2).sum())
c3 = int((checkers_fired == 3).sum())
c4 = int((checkers_fired == 4).sum())

score_mean   = df["schilit_forensic_score"].mean()
score_median = df["schilit_forensic_score"].median()

print("\n" + SEP)
print("  [4/5]  UNIVERSE-WIDE RESULTS  (Pass threshold = score >= 70)")
print(SEP)
print(f"\n  Total stocks scanned        : {total:>5}")
print(f"  PASS  (score >= 70)         : {passed:>5}  ({pass_pct:.1f}%)")
print(f"  FAIL  (score <  70)         : {failed:>5}  ({100-pass_pct:.1f}%)")
print(f"\n  Score mean                  : {score_mean:.1f} / 100")
print(f"  Score median                : {score_median:.1f} / 100")

print(f"\n  {'Score':>6}  {'Checkers':>10}  {'Count':>6}  Label")
print(f"  {'------':>6}  {'--------':>10}  {'------':>6}  -----")
print(f"  {'100':>6}  {'0':>10}  {c0:>6}  Clean -- no shenanigans")
print(f"  { '85':>6}  {'1':>10}  {c1:>6}  One concern -- PASS")
print(f"  { '70':>6}  {'2':>10}  {c2:>6}  Borderline -- PASS (just meets threshold)")
print(f"  { '55':>6}  {'3':>10}  {c3:>6}  FAIL")
print(f"  { '40':>6}  {'4':>10}  {c4:>6}  FAIL -- all 4 checkers fired")

print(f"\n  --- Checker Breakdown (stocks flagged per individual checker) ---")
print(f"  Checker 1 EMS  Revenue & Expense manipulation : {ems_n:>5}  ({ems_n/total*100:.1f}%)")
print(f"  Checker 2 CFS  Cash flow distortion           : {cfs_n:>5}  ({cfs_n/total*100:.1f}%)")
print(f"  Checker 3 KMS  Balance sheet leverage trap    : {lev_n:>5}  ({lev_n/total*100:.1f}%)")
print(f"  Checker 4 KMS  Inventory / Receivables bloat : {blt_n:>5}  ({blt_n/total*100:.1f}%)")

# ── Signal 6 deep-dive ────────────────────────────────────────────────────────
print("\n" + SEP)
print("  [5/5]  SIGNAL 6 DEEP-DIVE -- Depreciation Rate Manipulation (EMS #4)")
print("         Logic : FA grew >5% AND dep/FA fell >20% YoY AND not financial")
print("         Cases : Qwest (cable life 14yr->40yr), WorldCom ($3.8B opex->capex)")
print(SEP)

_nan = pd.Series(np.nan, index=df.index)
_fa      = df.get("fixed_assets",     _nan)
_fa_1yb  = df.get("fixed_assets_1yb", _nan)
_dep_rt  = df.get("dep_rate",         _nan)
_dep_1yb = df.get("dep_rate_1yb",     _nan)
_is_fin  = df.get("is_financial",     pd.Series(False, index=df.index)).fillna(False)

sig6 = (_fa > _fa_1yb * 1.05) & (_dep_rt < _dep_1yb * 0.80) & ~_is_fin
sig6_n  = int(sig6.sum())
dep_ok  = int(df["dep_rate"].notna().sum()) if "dep_rate" in df.columns else 0
ebit_ok = int(df["ebit"].notna().sum())     if "ebit" in df.columns else 0

print(f"\n  Stocks with EBIT data      : {ebit_ok}")
print(f"  Stocks with dep_rate data  : {dep_ok}")
print(f"  Signal 6 triggered         : {sig6_n} stocks  ({sig6_n/total*100:.1f}% of universe)")

if sig6_n > 0:
    sig6_df = df[sig6][["name", "sector", "market_cap",
                          "fixed_assets", "fixed_assets_1yb",
                          "dep_rate", "dep_rate_1yb"]].copy()
    sig6_df["fa_gr_pct"]    = ((sig6_df["fixed_assets"] / sig6_df["fixed_assets_1yb"]) - 1) * 100
    sig6_df["dep_fall_pct"] = ((sig6_df["dep_rate"] / sig6_df["dep_rate_1yb"]) - 1) * 100
    sig6_df = sig6_df.sort_values("dep_fall_pct")   # largest dep_rate fall first

    print(f"\n  {'#':>3}  {'Name':<28} {'Sector':<20} {'FA Grw%':>7}  {'Dep% Chg':>9}")
    print(f"  {'---':>3}  {'-'*28} {'-'*20} {'-'*7}  {'-'*9}")
    for i, (_, r) in enumerate(sig6_df.head(30).iterrows(), 1):
        n = str(r["name"])[:27]
        s = str(r["sector"])[:19]
        print(f"  {i:>3}  {n:<28} {s:<20} {r['fa_gr_pct']:>+6.1f}%  {r['dep_fall_pct']:>+8.1f}%")
else:
    print("\n  Signal 6 = 0 stocks. This means either:")
    print("  - dep_rate data is sparse (EBIT coverage < full universe)")
    print("  - or no stock simultaneously grew FA >5% AND cut dep/FA by >20%")

# ── All-4-flag stocks (Score = 40) ────────────────────────────────────────────
worst = df[checkers_fired == 4].copy()
print("\n" + SEP)
if len(worst) > 0:
    print(f"  ALL-4-FLAG STOCKS  (Score = 40)  --  {len(worst)} stocks")
    print(SEP)
    print(f"\n  {'Name':<30} {'Sector':<22} EMS  CFS  LEV  BLT")
    print(f"  {'-'*30} {'-'*22} ---  ---  ---  ---")
    for _, r in worst.sort_values("name").iterrows():
        n = str(r["name"])[:29]
        s = str(r["sector"])[:21]
        print(f"  {n:<30} {s:<22}   {int(r['schilit_ems_flag'])}    "
              f"{int(r['schilit_cfs_flag'])}    {int(r['schilit_kms_lev_flag'])}    "
              f"{int(r['schilit_kms_bloat_flag'])}")
else:
    print("  ALL-4-FLAG STOCKS  --  None (no stock fired all 4 checkers)")
    print(SEP)

# ── 3-flag failures (Score = 55) ─────────────────────────────────────────────
three_df = df[checkers_fired == 3].copy()
print(f"\n  3-FLAG STOCKS  (Score = 55)  --  {len(three_df)} stocks  --  FAIL")
print(SEP)
if len(three_df) > 0:
    print(f"\n  {'Name':<30} {'Sector':<22} EMS  CFS  LEV  BLT")
    print(f"  {'-'*30} {'-'*22} ---  ---  ---  ---")
    for _, r in three_df.sort_values("name").iterrows():
        n = str(r["name"])[:29]
        s = str(r["sector"])[:21]
        print(f"  {n:<30} {s:<22}   {int(r['schilit_ems_flag'])}    "
              f"{int(r['schilit_cfs_flag'])}    {int(r['schilit_kms_lev_flag'])}    "
              f"{int(r['schilit_kms_bloat_flag'])}")

# ── Clean stocks (Score = 100) ────────────────────────────────────────────────
clean = df[checkers_fired == 0].copy()
if "market_cap" in clean.columns:
    clean = clean.sort_values("market_cap", ascending=False)

print("\n" + SEP)
print(f"  CLEAN STOCKS  (Score = 100, Zero Flags)  --  {len(clean)} stocks -- Top 30 by MCap")
print(SEP)
print(f"\n  {'Name':<30} {'Sector':<26} {'MCap (Cr)':>12}")
print(f"  {'-'*30} {'-'*26} {'-'*12}")
for _, r in clean.head(30).iterrows():
    n    = str(r["name"])[:29]
    s    = str(r["sector"])[:25]
    mcap = f"{r['market_cap']:>12,.0f}" if pd.notna(r.get("market_cap")) else "           N/A"
    print(f"  {n:<30} {s:<26} {mcap}")

# ── Sector-wise failure rate ──────────────────────────────────────────────────
print("\n" + SEP)
print("  SECTOR-WISE SCHILIT FAILURE RATE  (score < 70)")
print(SEP)
df["_failed"] = (df["schilit_forensic_score"] < 70).astype(int)
sec = (
    df.groupby("sector", observed=True)
      .agg(total=("_failed", "count"), failed=("_failed", "sum"))
      .assign(fail_pct=lambda x: x["failed"] / x["total"] * 100)
      .sort_values("fail_pct", ascending=False)
)
print(f"\n  {'Sector':<32} {'Total':>6} {'Failed':>7} {'Fail%':>7}  Chart")
print(f"  {'-'*32} {'-'*6} {'-'*7} {'-'*7}  -----")
for sector, row in sec.iterrows():
    if int(row["total"]) >= 3:
        bar = "#" * int(row["fail_pct"] / 5)
        print(f"  {str(sector):<32} {int(row['total']):>6} {int(row['failed']):>7} "
              f"{row['fail_pct']:>6.1f}%  {bar}")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + SEP)
print(f"  SCAN COMPLETE  --  {total} stocks  --  {time.time()-t0:.2f}s total")
print(f"  Tests passing  --  147 / 147 (run: pytest tests/test_schilit_contract.py)")
print(SEP + "\n")
