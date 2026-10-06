"""
🎯 Scans — measured stock lists, one definition each (2026-10-06).

A screener usually shows a list with no evidence. Each scan here is shown beside ITS OWN record: how
its stocks did against the market on PRISM's saved snapshots. Only recipes that beat the market in
every test window get a card (docs/december-2026-plan.md section 8); the rest wait as "being
measured" until December's review.

THREE RULES, each pinned in tests/test_scans.py:
1. ONE DEFINITION. Each recipe is one function below. The tab calls it on the whole market and the
   local record tool (tools/scan_records.py) imports the same function, so a record always describes
   exactly the stocks on screen.
2. A RECORD NEVER OUTLIVES ITS RECIPE. scan_records.json stores a fingerprint of the logic it
   measured (the recipe plus every engine function it reads). Change the logic and the card shows no
   numbers until it is re-measured.
3. NOTHING HIDDEN, NOTHING PROMISED. A flagged stock stays in its list with the flag named; a missing
   value never passes a condition; the wording is "matched / record / provisional".

Display only, like ui_movers: no widgets here. app.py owns the scan choice, the liquidity floor and
the row click.
"""
import ast
import hashlib
import html
import inspect
import json
import textwrap
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from config import COLORS
from core.scoring_engine import strong_quarter
from ui.ui_group_views import group_measures

RECORDS_PATH = Path(__file__).with_name("scan_records.json")
RECORD_VERSION = 1


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """The column as numbers; all-NaN when absent, so a missing input passes nothing."""
    return (pd.to_numeric(df[col], errors="coerce") if col in df.columns
            else pd.Series(np.nan, index=df.index))


def _pct(df: pd.DataFrame, col: str) -> pd.Series:
    """Percentile rank (0-1] across the frame given — the whole market, never a filtered slice."""
    return _num(df, col).rank(pct=True)


# ── The recipes ────────────────────────────────────────────────────────────────────────────────────
def playbook_shortlist(df: pd.DataFrame) -> pd.Series:
    """Momentum and Breakout in the market's top third, a 🔥 Strong Quarter (the engine's own
    definition; a quarter that cannot be judged is not strong), Improvement in the top half, and the
    price at or above its stop. Measured 2026-10-06: +7.1 / +10.7 / +3.3 / +2.9pp against the market
    median, 61-69% of matches beating it."""
    return ((_pct(df, "momentum_score") >= 2 / 3)
            & (_pct(df, "breakout_score") >= 2 / 3)
            & (strong_quarter(df) == 1)
            & (_pct(df, "improvement_score") >= 0.5)
            & (_num(df, "dist_to_vstop") >= 0)).fillna(False).astype(bool)


def leading_sectors(df: pd.DataFrame, n: int = 5, min_stocks: int = 5) -> list:
    """The n sectors (of at least min_stocks stocks) ranked best on the 📈 Sectors tab's own two
    breadth shares: stocks within 5% of their 52-week high, and stocks reporting a strong quarter.
    Ties break on the near-high share, then the sector name, so the answer never depends on row
    order."""
    if "sector" not in df.columns:
        return []
    g = group_measures(df, "sector")
    if not {"grp_near_high", "grp_strong_quarter"} <= set(g.columns):
        return []
    g = g.join(df["sector"].value_counts().rename("_n"))
    g = g[g["_n"] >= min_stocks].dropna(subset=["grp_near_high", "grp_strong_quarter"])
    order = pd.DataFrame({
        "score": (g["grp_near_high"].rank(ascending=False) + g["grp_strong_quarter"].rank(ascending=False)).to_numpy(),
        "near": g["grp_near_high"].to_numpy(), "sector": g.index.astype(str)})
    order = order.sort_values(["score", "near", "sector"], ascending=[True, False, True])
    return order["sector"].head(n).tolist()


def leading_sector_stocks(df: pd.DataFrame) -> pd.Series:
    """Every stock in the leading sectors — the whole sector, not only its leaders. Measured
    2026-10-06: +2.9 / +8.6 / +2.4 / +3.4pp against the market median."""
    if "sector" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["sector"].isin(leading_sectors(df)).astype(bool)


SCANS = (
    dict(key="playbook", emoji="🚀", name="Playbook Shortlist", mask=playbook_shortlist,
         uses=(strong_quarter, _pct, _num),
         recipe="Momentum and Breakout in the top third of the market, a 🔥 Strong Quarter, Improvement "
                "in the top half, and the price above its stop."),
    dict(key="leaders", emoji="🌬️", name="Leading-Sector Stocks", mask=leading_sector_stocks,
         uses=(leading_sectors, group_measures, strong_quarter),
         recipe="Every stock in the 5 sectors (of 5+ stocks) with the most stocks near their 52-week "
                "high and reporting strong quarters, as ranked on 📈 Sectors."),
)

# Recorded in tools/validate.py; December decides (docs/december-2026-plan.md, sections 2d and 8).
CANDIDATES = (
    ("Supply-chain velocity", "cash conversion cycle down 15+ days and revenue up more than 15%"),
    ("WC Bullies", "3-year revenue growth above 20%, cash cycle not worse, cash flow at least profit"),
    ("J-Curve inflection", "operating margin and profit accelerating while capital work converts"),
)


# ── A record never outlives its recipe ─────────────────────────────────────────────────────────────
def _recipe_functions(scan: dict) -> list:
    return [scan["mask"], *scan.get("uses", ())]


def _fingerprint_of(funcs) -> str:
    """Hash of the functions' LOGIC: the parsed code without docstrings or the function's own name, so
    a reworded comment or a rename keeps a record valid and a changed threshold does not."""
    parts = []
    for fn in funcs:
        node = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
        if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:]
        node.name = "_"
        parts.append(ast.dump(node))
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def recipe_fingerprint(scan: dict) -> str:
    return _fingerprint_of(_recipe_functions(scan))


def load_records(path: Path = RECORDS_PATH) -> dict:
    """The committed records; {} when missing, unreadable or another version (cards then say "Not yet
    measured" rather than guess)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("version") != RECORD_VERSION or "scans" not in data:
        return {}
    return data


def record_state(records: dict, scan: dict):
    """("measured", rec) · ("stale", rec) when the recipe changed since it was measured ·
    ("unmeasured", None)."""
    rec = (records.get("scans") or {}).get(scan["key"])
    if not rec:
        return "unmeasured", None
    if rec.get("fingerprint") != recipe_fingerprint(scan):
        return "stale", rec
    return "measured", rec


# ── The list ───────────────────────────────────────────────────────────────────────────────────────
MEASURED_FLAGS = {"rf_wc_double_squeeze": "Squeeze", "rf_receivables_bloat": "Receivables",
                  "rf_negative_fcf": "Neg FCF"}
LIQUIDITY_STEPS = (None, 0.5, 1.0, 2.0, 5.0)    # ₹ crore a day; None = any

TABLE_COLUMNS = ("rank", "_new", "name", "sector", "composite_score", "momentum_score",
                 "improvement_score", "_strong_q", "_measured_flags", "weinstein_stage", "_liq_cr")
# Set widths (an auto-sized text column grows to its longest cell): 1,025px, inside the 1,030px a
# 1536px laptop leaves after the 44px row-selection column (the Deep Scanner's measured budget).
TABLE_WIDTHS = {"rank": 55, "_new": 40, "name": 170, "sector": 110, "composite_score": 105,
                "momentum_score": 85, "improvement_score": 100, "_strong_q": 40,
                "_measured_flags": 140, "weinstein_stage": 100, "_liq_cr": 80}


def scan_table(df: pd.DataFrame, scan: dict, members_at_latest=None, min_cr=None) -> pd.DataFrame:
    """The scan's stocks, sorted by PRISM Rank. The recipe runs on the frame given (the whole market);
    the liquidity floor only narrows the finished list, and a stock with no traded value cannot clear
    a floor. Flagged stocks stay, with their measured flags named. 🆕 marks a stock that was not in the
    scan on the last measured snapshot (no record = no baseline = nothing marked)."""
    out = df.loc[scan["mask"](df)].copy()
    out["_liq_cr"] = _num(out, "vol_sma_20d") * _num(out, "close_price") / 1e7
    if min_cr is not None:
        out = out[out["_liq_cr"] >= min_cr]
    named = pd.Series("", index=out.index, dtype=object)
    for col, lab in MEASURED_FLAGS.items():                 # three columns, never a loop over rows
        named = named.where(~(_num(out, col) == 1), np.where(named == "", lab, named + " · " + lab))
    out["_measured_flags"] = named
    if members_at_latest is None:
        out["_new"] = ""
    else:
        out["_new"] = np.where(out["company_id"].astype(str).isin(set(members_at_latest)), "", "🆕")
    out["_strong_q"] = np.where(strong_quarter(out) == 1, "✓", "")
    out = out.sort_values("rank", kind="mergesort")
    for c in TABLE_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    return out[[*TABLE_COLUMNS, "company_id"]].reset_index(drop=True)


def table_config(latest_label: str) -> dict:
    """Column headers — the same header each column carries in the Deep Scanner — widths and tips."""
    w = TABLE_WIDTHS
    cc = st.column_config
    since = f"on {latest_label}" if latest_label else "on the last measured snapshot"
    return {
        "rank": cc.NumberColumn("Rank", format="%.0f", width=w["rank"],
                                help="PRISM's overall rank. The list is sorted by it for reading only: inside "
                                     "this scan the top of the Rank did not beat the rest in the 2026-10-06 "
                                     "test, so every match is the scan."),
        "_new": cc.TextColumn("🆕", width=w["_new"], help=f"Not in this scan {since}."),
        "name": cc.TextColumn("Stock", width=w["name"]),
        "sector": cc.TextColumn("Sector", width=w["sector"]),
        "composite_score": cc.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f",
                                             width=w["composite_score"]),
        "momentum_score": cc.NumberColumn("Momentum", format="%.0f", width=w["momentum_score"]),
        "improvement_score": cc.NumberColumn("Improvement", format="%.0f", width=w["improvement_score"]),
        "_strong_q": cc.TextColumn("🔥", width=w["_strong_q"],
                                   help="🔥 Strong Quarter: latest quarter vs the same quarter last year — "
                                        "revenue up more than 15%, profit up more than 20%, margin wider."),
        "_measured_flags": cc.TextColumn(
            "⚠ Measured flags", width=w["_measured_flags"],
            help="The red flags that went with weaker returns in the 2026-10-06 measurement (provisional; "
                 "December re-tests them): working-capital double squeeze (customers pay 10+ days slower "
                 "AND suppliers are paid 10+ days faster) — weaker even for improving businesses; "
                 "receivables bloat and negative free cash flow — weaker mainly for businesses that are "
                 "not improving, and negative free cash flow alongside capex mirage was not weaker. "
                 "Shown, never removed: open the Tear-Sheet's Forensics tab before deciding."),
        "weinstein_stage": cc.TextColumn("Trend", width=w["weinstein_stage"]),
        "_liq_cr": cc.NumberColumn("₹ Cr/day", format="%.1f", width=w["_liq_cr"],
                                   help="Average daily traded value over 20 days (average volume × price), "
                                        "in ₹ crore."),
    }


# ── The card ───────────────────────────────────────────────────────────────────────────────────────
def _month_span(w: dict) -> str:
    a, b = date.fromisoformat(w["start"]), date.fromisoformat(w["end"])
    return f"{a:%b}→{b:%b}"


def _range(vals, fmt) -> str:
    lo, hi = min(vals), max(vals)
    return fmt.format(lo) if lo == hi else f"{fmt.format(lo)}–{fmt.format(hi)}"


def card_html(scan: dict, n_now: int, state: str, rec, selected: bool, latest: str) -> str:
    """One scan card: name, today's count, the recipe in words, and its record — one bar per test
    window, the share of matches that beat the market, the sample size, and "provisional". A stale or
    missing record shows no numbers."""
    c = COLORS
    esc = html.escape
    border = c["gold"] if selected else c["border"]
    head = (f'<div style="display:flex;justify-content:space-between;align-items:baseline;gap:8px;">'
            f'<span style="font-size:1.0rem;font-weight:800;color:{c["text_primary"]};">'
            f'{scan["emoji"]} {esc(scan["name"])}</span>'
            f'<span style="font-size:1.35rem;font-weight:900;color:{c["gold"]};">{n_now:,}'
            f'<span style="font-size:0.7rem;font-weight:600;color:{c["text_muted"]};"> now</span></span></div>'
            f'<div style="font-size:0.76rem;color:{c["text_secondary"]};margin:4px 0 10px 0;line-height:1.35;">'
            f'{esc(scan["recipe"])}</div>')
    if state == "measured" and rec and rec.get("windows"):
        ws = rec["windows"]
        bars = ""
        for w in ws:
            v = float(w["median_excess_pp"])
            h = 6 + round(28 * min(abs(v) / 12.0, 1.0))
            col = c["green"] if v > 0 else c["red"]
            bars += (f'<div style="display:flex;flex-direction:column;align-items:center;justify-content:flex-end;'
                     f'min-width:46px;" title="{esc(w["start"])} → {esc(w["end"])}, {int(w["days"])} days, '
                     f'{int(w["n"])} stocks">'
                     f'<div style="width:14px;height:{h}px;background:{col};border-radius:3px;"></div>'
                     f'<div style="font-size:0.78rem;font-weight:800;color:{col};margin-top:3px;">{v:+.1f}</div>'
                     f'<div style="font-size:0.6rem;color:{c["text_muted"]};">{_month_span(w)}</div></div>')
        beat = _range([int(w["beat_pct"]) for w in ws], "{}") + "%"
        n = _range([int(w["n"]) for w in ws], "{}")
        body = (f'<div style="font-size:0.66rem;color:{c["text_muted"]};margin-bottom:4px;">Record — points '
                f'above the market median, per test window</div>'
                f'<div style="display:flex;gap:6px;align-items:flex-end;min-height:56px;">{bars}</div>'
                f'<div style="font-size:0.74rem;color:{c["text_secondary"]};margin-top:8px;">'
                f'{beat} of matches beat the market · {n} stocks per window</div>'
                f'<div style="font-size:0.66rem;color:{c["text_muted"]};margin-top:3px;">provisional · '
                f'{len(ws)} short windows{f" · measured to {esc(latest)}" if latest else ""}</div>')
    elif state == "stale":
        body = (f'<div style="font-size:0.74rem;color:{c["gold"]};">The recipe changed since its record was '
                f'measured — no record is shown until it is re-measured.</div>')
    else:
        body = f'<div style="font-size:0.74rem;color:{c["text_muted"]};">Not yet measured.</div>'
    return (f'<div style="background:{c["bg_secondary"]};border:1px solid {border};border-radius:10px;'
            f'padding:12px 14px;min-height:205px;">{head}{body}</div>')


def latest_label(records: dict) -> str:
    """'03 Oct' for the last snapshot a record was measured on; '' with no records."""
    d = records.get("latest_snapshot") if records else None
    try:
        return f"{date.fromisoformat(d):%d %b}"
    except (TypeError, ValueError):
        return ""
