"""
ui_group_views.py — the measures behind the Market Pulse 📈 Sectors and 🏭 Industry column views.

ONE DEFINITION FOR BOTH TABS (2026-10-05): what each column measures, how a GROUP of stocks is
summarised, and the header and tooltip it carries. Pure pandas, no Streamlit, so it is tested
directly (tests/test_market_pulse_views.py).

HOW A GROUP IS SUMMARISED
  mean    PRISM's own 0-100 scores — an average of percentile scores is well behaved.
  median  raw multiples and distances (P/E, EV/EBITDA, FCF yield, distance from the 52-week high):
          one P/E of 900 drags an average anywhere; the middle stock is the honest summary.
  share   yes/no conditions (above the 200-day average, few red flags): the % of the group's stocks
          for which it holds, counted ONLY among stocks that have the data — an unknown is not a "no".
          The engine records some unknowns as 0 — a stock with no 200-day average yet is "not above"
          it, one with no cash-flow figure "not flagged" for accruals — so those measures name the
          inputs they NEED, and a stock missing one is left out. Measured 2026-10-05: counting the 180
          stocks with no 200-day average as "below" read Engineering Heavy General 45% above its
          200-DMA when 83% of the stocks that have one are (23 industries moved 10 points or more).
          Composite verdicts — the red-flag count, Schilit pass, the F-Score — are the engine's own.

WHY THESE COLUMNS. Measured on 2026-10-05 across 67 sectors (>= 5 stocks) and 252 industries (>= 3):
no two columns in one view rank-correlate at 0.85 or more, and none is flat. Dropped on that test:
median ROCE (0.90 with Quality), earnings yield (-0.99 with P/E — it is 1/P/E), % at the forensic
penalty floor (0.90 with the average red-flag count), RS score (0.96 with Momentum), breakout score
(0.90 with the distance from the 52-week high), the median 52-week relative strength against the
Nifty 500 (0.85 with % above the 200-day average, which overlaps less with the rest of the view),
and dividend yield / pledge (median 0 almost everywhere). Momentum lives in Core, so 📈 Technical carries breadth (% above the 200-day average,
0.89 with Momentum) instead of repeating it.
"""
import numpy as np
import pandas as pd

from config import FORENSIC_MAX_FLAGS, FORENSIC_PENALTY_TIERS

VIEWS = ("🏆 Core", "📊 Quality", "💰 Valuation", "🔬 Forensic", "📈 Technical")

# "Few red flags" is the forensic penalty's own line: the most flags that still cost only ×0.90.
_FEW_FLAGS = next(t["max_flags"] for t in FORENSIC_PENALTY_TIERS if t["multiplier"] == 0.90)


def _share(cond, known):
    """1.0 / 0.0 where the condition can be judged, NaN where it cannot (so a mean is a share of
    the stocks WITH data)."""
    return cond.astype(float).where(known)


# key -> source column, statistic, header, number format, bar (0-100 progress bar), tooltip, value;
# `needs` (optional) names the further inputs `value` reads — for a share, the inputs without which
# the engine could not judge the stock;
# NUMBER columns carry their own width (sized to the header) — bars share the app's one bar width,
# so every view of both tables fits a 1536px laptop screen (measured in the browser, 2026-10-05).
MEASURES = {
    # 📊 Quality
    "grp_roce_rising": dict(
        source="roce_expansion", stat="share", header="ROCE rising", fmt="%.0f%%", bar=True,
        help="Share of the group's stocks whose return on capital is rising: the 3-year median ROCE "
             "is above the 10-year median. Counted among stocks with the history to measure it.",
        value=lambda f: _share(f["roce_expansion"] > 0, f["roce_expansion"].notna())),
    "grp_piotroski_strong": dict(
        source="piotroski_fscore", stat="share", header="Piotroski 7+", fmt="%.0f%%", bar=True,
        help="Share of the group's stocks with a Piotroski F-Score of 7 or more out of 9 — healthy, "
             "improving books.",
        value=lambda f: _share(f["piotroski_fscore"] >= 7, f["piotroski_fscore"].notna())),
    "grp_opm": dict(
        source="opm", stat="median", header="Median OPM", fmt="%.1f%%", bar=False, width=105,
        help="Operating margin of the group's middle stock, latest full year.",
        value=lambda f: f["opm"]),
    # 💰 Valuation
    "grp_pe": dict(
        source="pe", stat="median", header="Median P/E", fmt="%.1f", bar=False, width=105,
        help="Price to earnings of the group's middle stock, among stocks with positive earnings — a "
             "loss-maker has no meaningful P/E. A median, so one extreme multiple cannot move it.",
        value=lambda f: f["pe"].where(f["pe"] > 0)),
    "grp_ev_ebitda": dict(
        source="ev_ebitda", stat="median", header="Median EV/EBITDA", fmt="%.1f", bar=False, width=130,
        help="Enterprise value to EBITDA of the group's middle stock, among stocks with positive "
             "EBITDA. Unlike P/E it counts debt, so it compares leveraged and debt-free groups fairly.",
        value=lambda f: f["ev_ebitda"].where(f["ev_ebitda"] > 0)),
    "grp_fcf_yield": dict(
        source="fcf_yield", stat="median", header="Median FCF yield", fmt="%.1f%%", bar=False, width=130,
        help="Free cash flow as a % of market value for the group's middle stock. Negative means the "
             "typical stock in the group is burning cash.",
        value=lambda f: f["fcf_yield"]),
    "grp_pe_vs_history": dict(
        source="d32_pe_vs_median", stat="median", header="P/E vs history", fmt="%+.0f%%", bar=False, width=120,
        help="How far each stock's P/E sits from its own 5-year median P/E, for the group's middle "
             "stock. Negative = the group trades cheaper than it usually does.",
        value=lambda f: f["d32_pe_vs_median"]),
    # 🔬 Forensic
    "grp_red_flags": dict(
        source="red_flag_count", stat="mean", header="Avg red flags", fmt="%.1f", bar=False, width=115,
        help=f"Average number of forensic red flags per stock, out of {FORENSIC_MAX_FLAGS} checks.",
        value=lambda f: f["red_flag_count"]),
    "grp_few_flags": dict(
        source="red_flag_count", stat="share", header=f"≤{_FEW_FLAGS} red flags", fmt="%.0f%%", bar=True,
        help=f"Share of the group's stocks with at most {_FEW_FLAGS} red flags — the ones the forensic "
             f"penalty barely touches (×0.90, or none at all).",
        value=lambda f: _share(f["red_flag_count"] <= _FEW_FLAGS, f["red_flag_count"].notna())),
    "grp_schilit_pass": dict(
        source="schilit_pass", stat="share", header="Schilit pass", fmt="%.0f%%", bar=True,
        help="Share passing the Schilit forensic screen: at most 2 of its 4 accounting-shenanigan "
             "checks fired.",
        value=lambda f: _share(f["schilit_pass"] == 1, f["schilit_pass"].notna())),
    "grp_high_accruals": dict(
        source="rf_high_accruals", stat="share", header="High accruals", fmt="%.0f%%", bar=True,
        needs=("pat_cf_year", "operating_cash_flow", "total_assets", "total_assets_1yb"),
        help="Share whose profit runs well ahead of its cash: accruals above 5% of assets, with profit "
             "and cash taken from the same year. Counted among stocks with profit, cash flow and assets "
             "on record.",
        # The engine's own guard (forensic_engine, rf_high_accruals): profit and cash flow of one year,
        # and assets this year OR last (it falls back to last year's) — without them it records 0.
        value=lambda f: _share(f["rf_high_accruals"] == 1,
                               f["pat_cf_year"].notna() & f["operating_cash_flow"].notna()
                               & ((f["total_assets"] > 0) | (f["total_assets_1yb"] > 0)))),
    # 📈 Technical
    "grp_above_200dma": dict(
        source="above_sma200", stat="share", header="Above 200-DMA", fmt="%.0f%%", bar=True,
        needs=("close_price", "sma_200d"),
        help="Share of the group's stocks trading above their 200-day moving average — the classic "
             "breadth reading of how much of a group is in an uptrend. Counted among stocks with 200 "
             "days of history: a recent listing is not 'below' an average it does not have yet.",
        # The engine sets above_sma200 = close > sma_200d, so a missing average reads 0, never blank.
        value=lambda f: _share(f["above_sma200"] == 1, f["close_price"].notna() & f["sma_200d"].notna())),
    "grp_stage2": dict(
        source="weinstein_stage", stat="share", header="Stage 2", fmt="%.0f%%", bar=True,
        help="Share in Weinstein Stage 2 — the advancing phase of a stock's cycle — among stocks whose "
             "stage is known.",
        value=lambda f: _share(f["weinstein_stage"].astype(str).str.contains("Stage 2", regex=False),
                               f["weinstein_stage"].notna()
                               & ~f["weinstein_stage"].astype(str).str.contains("Unknown", regex=False))),
    "grp_from_52w_high": dict(
        source="dist_52wh", stat="median", header="Below 52w high", fmt="%.0f%%", bar=False, width=120,
        help="How far below its 52-week high the group's middle stock trades. Small = the group is "
             "close to its highs.",
        value=lambda f: f["dist_52wh"]),
}

# Each view's columns after the tab's Count and ranking column. The avg_* entries are each table's
# own averages; everything else comes from MEASURES. 🏆 Core is each tab's own column list.
VIEW_MEASURES = {
    "📊 Quality":   ("avg_quality", "avg_improvement", "grp_roce_rising", "grp_piotroski_strong", "grp_opm"),
    "💰 Valuation": ("avg_valuation", "grp_pe", "grp_ev_ebitda", "grp_fcf_yield", "grp_pe_vs_history"),
    "🔬 Forensic":  ("grp_red_flags", "grp_few_flags", "grp_schilit_pass", "grp_high_accruals"),
    "📈 Technical": ("grp_above_200dma", "grp_stage2", "grp_from_52w_high"),
}

def _inputs(m):
    """Every column a measure reads: its source, then anything it needs to judge a stock."""
    return (m["source"], *m.get("needs", ()))


# Every input the measures read — 🏭 Industry projects its stocks to a fixed column list first.
SOURCE_COLUMNS = tuple(sorted({c for m in MEASURES.values() for c in _inputs(m)}))


def _per_stock(frame: pd.DataFrame, m: dict) -> pd.Series:
    """One measure's value per stock: 1 / 0 / NaN for a share, the number for a mean or median."""
    src = frame[list(_inputs(m))].copy()
    for c in src.columns:
        if c != "weinstein_stage":
            src[c] = pd.to_numeric(src[c], errors="coerce")
    return m["value"](src).astype(float)


def group_measures(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    """One row per group (index = the `key` values), one column per MEASURES entry.

    Vectorised: each measure's per-stock value is built once, then ONE groupby takes the means
    (scores and shares) and one the medians. Shares come back as percentages (0-100). A measure
    whose inputs are absent is left out rather than invented. The input is never modified.
    """
    present = {k: m for k, m in MEASURES.items() if all(c in frame.columns for c in _inputs(m))}
    vals = pd.DataFrame(index=frame.index)
    for k, m in present.items():
        vals[k] = _per_stock(frame, m)
    vals[key] = frame[key].values
    means = [k for k, m in present.items() if m["stat"] in ("mean", "share")]
    medians = [k for k, m in present.items() if m["stat"] == "median"]
    out = pd.concat([vals.groupby(key)[means].mean(), vals.groupby(key)[medians].median()], axis=1)
    shares = [k for k, m in present.items() if m["stat"] == "share"]
    out[shares] = out[shares] * 100.0
    return out[[k for k in MEASURES if k in out.columns]]
