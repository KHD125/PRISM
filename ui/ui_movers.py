"""PRISM — 🔁 Movers: what changed between two data vintages.

The engine's whole thesis is that DIRECTION beats level (Vel%, the 28th WCS), yet every other
surface shows one instant. This module diffs two vintages of the scored frame — the live sheet
against an archived Drive copy — and reports the moves: wealth-tier upgrades, soundness flips,
rank jumps, new gate passers, red-flag rises, fresh results.

THREE RULES THIS MODULE IS BUILT ON
  1. Same engine on both sides, by construction. The caller re-scores the archived RAW copy with
     the running engine (app.py `_load_vintage`), so a difference can only be "the company
     changed" — never "PRISM changed". A scored file from the past would confound the two.
  2. Never a fabricated delta. A stock present on one side only is NEW or DROPPED; a NaN on
     either side of a numeric column drops that row from that section. No sentinels, no fillna
     into a delta (CLAUDE.md semantic-truth mandate). `.eq(1)` treats NaN as "not a move".
  3. Zero scoring logic and zero mutation. Reads columns compute_verdict / run_full_scoring
     already produce; works on column subsets; both inputs come back byte-identical.

STATELESS, like ui_tearsheet: no st.button/slider/number_input, no st.columns/st.metric — the
header strip is inline flex. Widgets (the vintage picker, the compare button) live in app.py.
"""
import numpy as np
import pandas as pd
import streamlit as st

from config import COLORS

JOIN_KEY = "company_id"
# The wealth ladder, best first. N/A means "an input is missing — never condemned on absent
# evidence"; it sits OUTSIDE the ladder so a move to or from it is reported as unverifiable,
# never as an upgrade or a downgrade.
WEALTH_LADDER = ["BUY★", "BUY", "WATCH★", "WATCH", "AVOID"]
UNVERIFIABLE = "N/A"
SOUND_LADDER = ["SOUND", "MIXED", "FLAWED"]

# Every column the diff reads. Only those present on BOTH sides are diffed; a column missing from
# one vintage simply yields an empty section rather than a crash — an older engine may not have
# emitted it, and honesty beats coverage.
_COLS = ["name", "sector", "market_category", "wealth_tier", "verdict_direction",
         "conviction_tier", "rank", "composite_score", "gate_pass", "tsunami_signal",
         "red_flag_count", "result_age_days"]


# ── Vintage bookkeeping (pure) ────────────────────────────────────────────────
def fy_quarter(iso: str) -> str:
    """Indian-FY quarter label for a vintage date, by RESULT-FILING window — a byte-for-byte
    mirror of Prism.gs::_fyQuarter (the archiver), pinned by the same five cases so the label
    PRISM computes for the live side can never disagree with the label the index carries for
    the archived side. A vintage belongs to a quarter once that quarter's deadline has passed
    (30 May Q4 · 14 Aug Q1 · 14 Nov Q2 · 14 Feb Q3); captured in the deadline's month or the one
    after = on-cycle, later = "(off-cycle)"."""
    y, mo, _ = (int(p) for p in iso.split("-"))
    md = iso[5:]
    gates = [("02-14", 2, "Q3", y), ("05-30", 5, "Q4", y), ("08-14", 8, "Q1", y + 1),
             ("11-14", 11, "Q2", y + 1)]
    passed = None
    for g in gates:
        if md >= g[0]:
            passed = g
    if passed is None:
        passed = ("", -1, "Q2", y)
    label = f"FY{str(passed[3])[2:]}{passed[2]}"
    months_since = 99 if passed[1] < 0 else mo - passed[1]
    return label if months_since <= 1 else f"{label} (off-cycle)"


def usable_vintages(index_df: pd.DataFrame) -> pd.DataFrame:
    """The archive index → the rows Movers may load: status 'ok' only, newest first, dates as
    'YYYY-MM-DD' text. Raises if the sheet is not a PRISM Archive Index — a wrong id must fail
    loud, not render an empty picker."""
    need = {"vintage_date", "fy_quarter", "spreadsheet_id", "status"}
    missing = sorted(need - set(map(str, index_df.columns)))
    if missing:
        raise ValueError(f"not a PRISM Archive Index — missing columns {missing}")
    ok = index_df[index_df["status"].astype(str).str.strip() == "ok"].copy()
    ok["vintage_date"] = ok["vintage_date"].astype(str).str[:10]
    ok["fy_quarter"] = ok["fy_quarter"].astype(str).str.strip()
    ok["spreadsheet_id"] = ok["spreadsheet_id"].astype(str).str.strip()
    ok = ok.sort_values("vintage_date", ascending=False, kind="mergesort").reset_index(drop=True)
    return ok[["vintage_date", "fy_quarter", "spreadsheet_id"]]


def default_vintage(ok: pd.DataFrame):
    """The row to compare against by default: the most recent ON-CYCLE quarter (a clean
    quarter boundary), else the most recent row of any kind, else None."""
    if ok.empty:
        return None
    on_cycle = ok[~ok["fy_quarter"].str.contains("off-cycle", regex=False)]
    return (on_cycle if not on_cycle.empty else ok).iloc[0]


def load_index(sheet_id: str) -> pd.DataFrame:
    """Fetch the archive index by id (the same XLSX export path the data loader uses — never
    per-tab CSV, never gviz) and return usable_vintages of its first tab."""
    from core.data_engine import extract_spreadsheet_id
    xid = extract_spreadsheet_id(sheet_id)
    wb = pd.ExcelFile(f"https://docs.google.com/spreadsheets/d/{xid}/export?format=xlsx",
                      engine="openpyxl")
    return usable_vintages(wb.parse(wb.sheet_names[0]))


# ── The diff (pure) ───────────────────────────────────────────────────────────
def _pos(series: pd.Series, ladder) -> pd.Series:
    """Ladder position (0 = best); NaN for anything not on the ladder (N/A, blank, unknown)."""
    return series.map({v: i for i, v in enumerate(ladder)}).astype(float)


def _ladder_moves(both: pd.DataFrame, col: str, ladder, keep):
    """(improved, worsened, unverifiable) for a labelled ladder column. `steps` = rungs climbed
    (positive = improved). A change touching a label off the ladder is unverifiable."""
    now, was = both[col], both[f"{col}_prev"]
    changed = now.notna() & was.notna() & (now.astype(str) != was.astype(str))
    p_now, p_was = _pos(now, ladder), _pos(was, ladder)
    on_ladder = p_now.notna() & p_was.notna()
    steps = (p_was - p_now).where(changed & on_ladder)
    cols = keep + [f"{col}_prev", col]
    up = both.loc[changed & on_ladder & (steps > 0), cols].assign(steps=steps[changed & on_ladder & (steps > 0)].astype(int))
    dn = both.loc[changed & on_ladder & (steps < 0), cols].assign(steps=steps[changed & on_ladder & (steps < 0)].astype(int))
    un = both.loc[changed & ~on_ladder, cols]
    up = up.sort_values(["steps", "rank", "name"], ascending=[False, True, True], kind="mergesort")
    dn = dn.sort_values(["steps", "rank", "name"], ascending=[True, True, True], kind="mergesort")
    un = un.sort_values(["rank", "name"], kind="mergesort")
    return up, dn, un


def compute_movers(prev: pd.DataFrame, cur: pd.DataFrame, days_between=None) -> dict:
    """Diff two scored vintages joined on company_id. Pure; vectorized; never mutates inputs.

    `days_between` — the calendar gap between the two vintage dates, when the caller knows it.
    It makes `fresh` exact ("the result date is after the previous vintage": 0 ≤ age < gap);
    without it the diff falls back to the age-reset rule, which cannot see a result that landed
    between two vintages whose ages happen to be similar — the quarterly case.

    Returns a dict of frames, every one sorted deterministically (tie-broken by name):
      new, dropped                       stocks on one side only (no deltas — there are none)
      wealth_up, wealth_down             ladder moves with `steps` (rungs)
      wealth_unverifiable                a move touching N/A / an unknown label
      sound_up, sound_down               SOUND/MIXED/FLAWED moves
      rank                               every stock with rank on both sides: rank_delta
                                         (positive = climbed), composite_delta, tier_delta
      gate_new, gate_lost, tsunami_new   0→1 / 1→0 flag transitions (NaN = not a move)
      flags                              red_flag_count changes, rises first
      fresh                              a result landed between the vintages
    plus `counts` (ints) and `n_both`.
    """
    for side, f in (("previous", prev), ("current", cur)):
        if JOIN_KEY not in f.columns:
            raise ValueError(f"the {side} vintage has no {JOIN_KEY} column — cannot join")
        d = int(f[JOIN_KEY].duplicated().sum())
        if d:
            raise ValueError(f"the {side} vintage has {d} duplicate {JOIN_KEY} rows — the join would fan out")

    cols = [c for c in _COLS if c in cur.columns and c in prev.columns]
    a = cur[[JOIN_KEY] + cols]
    b = prev[[JOIN_KEY] + cols]
    m = a.merge(b, on=JOIN_KEY, how="outer", suffixes=("", "_prev"), indicator=True, sort=True)

    ident = [c for c in ["name", "sector", "market_category"] if c in cols]
    # `rank` and `name` are ALWAYS present in the working frame — every section sorts on them —
    # materialized as NaN / the key when a vintage lacks them, so a thin frame degrades to an
    # unsorted-by-rank view instead of a KeyError.
    if "name" not in cols:
        m["name"] = m[JOIN_KEY].astype(str)
        ident.append("name")
    if "rank" not in cols:
        m["rank"] = np.nan
    keep = [JOIN_KEY] + ident + ["rank"] + (["composite_score"] if "composite_score" in cols else [])

    both = m[m["_merge"] == "both"].drop(columns="_merge")
    new = m.loc[m["_merge"] == "left_only", [JOIN_KEY] + cols].sort_values(["rank", "name"] if "rank" in cols else ["name"], kind="mergesort")
    dropped = (m.loc[m["_merge"] == "right_only", [JOIN_KEY] + [f"{c}_prev" for c in cols]]
                .rename(columns={f"{c}_prev": c for c in cols}))
    dropped = dropped.sort_values(["rank", "name"] if "rank" in cols else ["name"], kind="mergesort")

    out = {"new": new.reset_index(drop=True), "dropped": dropped.reset_index(drop=True),
           "n_both": int(len(both))}
    empty = both.iloc[0:0]

    if "wealth_tier" in cols:
        up, dn, un = _ladder_moves(both, "wealth_tier", WEALTH_LADDER, keep)
    else:
        up = dn = un = empty
    out["wealth_up"], out["wealth_down"], out["wealth_unverifiable"] = up, dn, un

    if "verdict_direction" in cols:
        su, sd, _ = _ladder_moves(both, "verdict_direction", SOUND_LADDER, keep)
    else:
        su = sd = empty
    out["sound_up"], out["sound_down"] = su, sd

    if "rank" in cols:
        r = both[keep + ["rank_prev"] + [c for c in ["composite_score_prev", "conviction_tier",
                                                     "conviction_tier_prev"] if c in both.columns]].copy()
        r["rank_delta"] = r["rank_prev"] - r["rank"]                 # positive = climbed
        if "composite_score" in cols:
            r["composite_delta"] = r["composite_score"] - r["composite_score_prev"]
        if "conviction_tier" in cols:
            r["tier_delta"] = r["conviction_tier_prev"] - r["conviction_tier"]   # positive = better tier
        r = r[r["rank_delta"].notna()]
        # Climbers first, biggest climb on top; the renderer reverses the negative tail so the
        # biggest FALL leads its own section. One sort, one order, tie-broken by name.
        r = r.sort_values(["rank_delta", "name"], ascending=[False, True], kind="mergesort")
        out["rank"] = r.reset_index(drop=True)
    else:
        out["rank"] = empty

    def _flag_move(col, frm, to):
        if col not in cols:
            return empty
        mask = both[col].eq(to) & both[f"{col}_prev"].eq(frm)
        return both.loc[mask, keep].sort_values(["rank", "name"], kind="mergesort").reset_index(drop=True)

    out["gate_new"] = _flag_move("gate_pass", 0, 1)
    out["gate_lost"] = _flag_move("gate_pass", 1, 0)
    out["tsunami_new"] = _flag_move("tsunami_signal", 0, 1)

    if "red_flag_count" in cols:
        fl = both[keep + ["red_flag_count_prev", "red_flag_count"]].copy()
        fl["flag_delta"] = fl["red_flag_count"] - fl["red_flag_count_prev"]
        fl = fl[fl["flag_delta"].notna() & (fl["flag_delta"] != 0)]
        fl = fl.sort_values(["flag_delta", "rank", "name"], ascending=[False, True, True], kind="mergesort")
        out["flags"] = fl.reset_index(drop=True)
    else:
        out["flags"] = empty

    if "result_age_days" in cols:
        age, age_prev = both["result_age_days"], both["result_age_days_prev"]
        # A result landed iff the age RESET — smaller now than it was — OR the previous side was a
        # SCHEDULED result (negative age) and now it is declared (non-negative). The second clause
        # was missing at first: the June-29 vintage carried a scheduled Q1 date for most of the
        # universe, so `cur < prev` alone counted 193 of ~2,000 fresh results. A negative age NOW
        # is still never fresh — nothing has landed yet.
        if days_between is not None:
            # EXACT, given the gap: the result date (today − age) is after the previous vintage
            # iff age < gap. Quarterly vintages both sit ~20-40 days after a deadline, so the
            # age-reset rule below cannot see most of them — live it counted 877 of ~2,000.
            fresh_mask = age.notna() & (age >= 0) & (age < float(days_between))
        else:
            fresh_mask = (age.notna() & age_prev.notna() & (age >= 0)
                          & ((age_prev < 0) | (age < age_prev)))
        fr = both.loc[fresh_mask, keep + ["result_age_days"]]
        out["fresh"] = fr.sort_values(["result_age_days", "rank", "name"], kind="mergesort").reset_index(drop=True)
    else:
        out["fresh"] = empty

    # CHURN — the headline. Share of COMPARABLE stocks (column present on both sides) whose label
    # or count changed. Measured live on the first quarter-over-quarter run: 47% of wealth tiers,
    # 80% of flag counts. A fact about the engine's label stability that nothing else can see.
    churn = {}
    for col in ("wealth_tier", "verdict_direction", "red_flag_count", "gate_pass"):
        if col not in cols:
            continue
        now, was = both[col], both[f"{col}_prev"]
        comparable = now.notna() & was.notna()
        n = int(comparable.sum())
        changed = comparable & ((now.astype(str) != was.astype(str))
                                if col in ("wealth_tier", "verdict_direction") else (now != was))
        churn[col] = float(changed.sum() / n) if n else float("nan")
    out["churn"] = churn

    # IDENTITY for every both-side stock — what `material` joins its reasons onto.
    ident = both[[JOIN_KEY] + [c for c in ("name", "sector", "wealth_tier", "rank") if c in both.columns]].copy()
    ident["rank_delta"] = (both["rank_prev"] - both["rank"]) if "rank_prev" in both.columns else np.nan
    out["ident"] = ident.reset_index(drop=True)

    out["counts"] = {k: int(len(v)) for k, v in out.items() if isinstance(v, pd.DataFrame)}
    return out


_REASON_ORDER = {"→": 0, "BUY★": 1, "gate": 2, "🌊": 3, "rank": 4, "flags": 5}   # by first token


def _signed(s: pd.Series) -> pd.Series:
    """+220 / −680 as text, vectorized (a true minus sign, so it cannot be misread as a hyphen)."""
    v = s.astype(int)
    return pd.Series(np.where(v > 0, "+" + v.astype(str), "−" + (-v).astype(str)), index=s.index)


def material(res: dict, top_rank: int = 25, min_flags: int = 3, cap: int = 40) -> pd.DataFrame:
    """'What matters' — one row per stock with a MATERIAL move, reasons joined in a fixed order:
    into / out of BUY★ (ladder moves only — an N/A transition is unverifiable, not a verdict),
    crossed the gate either way, a new Tsunami setup, the `top_rank` biggest |Δ rank|, and
    |Δ flags| ≥ `min_flags`. Sorted by number of reasons (desc), then rank; capped at `cap`.
    The six full sections underneath are the evidence; this is the page's first table. Pure."""
    def tag(frame, why):
        return frame[[JOIN_KEY]].assign(why=why)

    up, dn, r, fl = res["wealth_up"], res["wealth_down"], res["rank"], res["flags"]
    parts = [tag(res["gate_new"], "gate ✓"), tag(res["gate_lost"], "gate ✗"), tag(res["tsunami_new"], "🌊 new")]
    if not up.empty:
        parts.append(tag(up[up["wealth_tier"] == WEALTH_LADDER[0]], "→ BUY★"))
    if not dn.empty:
        parts.append(tag(dn[dn["wealth_tier_prev"] == WEALTH_LADDER[0]], "BUY★ →"))
    if not r.empty and top_rank > 0:
        top = r.reindex(r["rank_delta"].abs().sort_values(ascending=False, kind="mergesort").index).head(top_rank)
        parts.append(top[[JOIN_KEY]].assign(why="rank " + _signed(top["rank_delta"])))
    if not fl.empty:
        big = fl[fl["flag_delta"].abs() >= min_flags]
        parts.append(big[[JOIN_KEY]].assign(why="flags " + _signed(big["flag_delta"])))
    cols_out = [JOIN_KEY, "name", "sector", "why", "wealth_tier", "rank", "rank_delta"]
    reasons = pd.concat(parts, ignore_index=True)
    if reasons.empty:
        return pd.DataFrame(columns=cols_out)
    reasons["_o"] = reasons["why"].str.split(" ").str[0].map(_REASON_ORDER)
    reasons = reasons.sort_values([JOIN_KEY, "_o"], kind="mergesort")
    agg = reasons.groupby(JOIN_KEY, sort=True).agg(why=("why", " · ".join), n=("why", "size")).reset_index()
    ident = res["ident"]
    for c in ("sector", "wealth_tier"):
        if c not in ident.columns:
            ident = ident.assign(**{c: np.nan})
    m = agg.merge(ident, on=JOIN_KEY, how="left")
    m = m.sort_values(["n", "rank", "name"], ascending=[False, True, True], kind="mergesort")
    return m[cols_out].head(cap).reset_index(drop=True)


def restrict(res: dict, ids) -> dict:
    """Apply the lens row to a computed result: keep only stocks whose CURRENT-side company_id is
    in `ids`. Computed AFTER the diff, never before — filtering the current frame first would
    turn every filtered-out stock into a fake 'dropped' row. `dropped` has no current side to
    filter on, so it is left whole (its note says so). Counts are recomputed. Pure."""
    keep = set(map(str, ids))
    out = {}
    for k, v in res.items():
        if isinstance(v, pd.DataFrame) and k != "dropped":
            out[k] = v[v[JOIN_KEY].astype(str).isin(keep)].reset_index(drop=True)
        else:
            out[k] = v
    out["counts"] = {k: int(len(v)) for k, v in out.items() if isinstance(v, pd.DataFrame)}
    return out


# ── Rendering (stateless) ─────────────────────────────────────────────────────
_HDR = {
    JOIN_KEY: None, "name": "Stock", "sector": "Sector", "market_category": "Market Cap",
    "rank": "Rank", "rank_prev": "Rank was", "rank_delta": "Δ Rank",
    "composite_score": "Score", "composite_score_prev": "Score was", "composite_delta": "Δ Score",
    "conviction_tier": "Tier", "conviction_tier_prev": "Tier was", "tier_delta": "Δ Tier",
    "wealth_tier": "Now", "wealth_tier_prev": "Was", "steps": "Rungs",
    "verdict_direction": "Now", "verdict_direction_prev": "Was",
    "red_flag_count": "🚩 Flags", "red_flag_count_prev": "🚩 Flags was", "flag_delta": "Δ Flags",
    "result_age_days": "Days since result", "gate_pass": "Gate", "tsunami_signal": "Tsunami",
    "why": "Why",
}


def _table(df: pd.DataFrame, limit: int = 40) -> None:
    """One section table. Column names are never shown raw: every column has a header in _HDR,
    the join key is hidden, and the rest follow the app-wide vocabulary (Stock · Sector · Score
    · 🚩 Flags). Capped at `limit` rows — the count in the section title says how many exist."""
    if df.empty:
        st.markdown(f"<div style='font-size:0.74rem;color:{COLORS['text_muted']};"
                    f"padding:2px 0 10px 2px;'>none</div>", unsafe_allow_html=True)
        return
    show = [c for c in df.columns if _HDR.get(c, c) is not None]
    cfg = {}
    for c in show:
        label = _HDR.get(c, c.replace("_", " ").title())
        if c in ("composite_score", "composite_score_prev", "composite_delta"):
            cfg[c] = st.column_config.NumberColumn(label, format="%+.1f" if "delta" in c else "%.1f", width="small")
        elif c in ("rank", "rank_prev", "rank_delta", "steps", "tier_delta", "flag_delta",
                   "red_flag_count", "red_flag_count_prev", "result_age_days", "conviction_tier",
                   "conviction_tier_prev"):
            cfg[c] = st.column_config.NumberColumn(label, format="%+d" if ("delta" in c or c == "steps") else "%d", width="small")
        else:
            cfg[c] = st.column_config.TextColumn(label, width="medium" if c == "name" else "small")
    st.dataframe(df[show].head(limit).reset_index(drop=True), column_config=cfg,
                 use_container_width=True, hide_index=True,
                 height=min(420, 60 + min(len(df), limit) * 35))


def _section(title: str, n: int, note: str = "") -> None:
    st.markdown(
        f"<div style='display:flex;align-items:baseline;gap:8px;margin:14px 0 4px 0;'>"
        f"<span style='font-size:0.9rem;font-weight:800;color:{COLORS['text_primary']};'>{title}</span>"
        f"<span style='font-size:0.72rem;font-weight:700;color:{COLORS['purple']};'>{n}</span>"
        + (f"<span style='font-size:0.68rem;color:{COLORS['text_muted']};'>{note}</span>" if note else "")
        + "</div>", unsafe_allow_html=True)


def _churn_line(churn: dict) -> str:
    """The header's first line: what share of comparable stocks changed, per label. High churn is
    a fact about the ENGINE's label stability — 47% of wealth tiers flipping in one quarter says
    the thresholds sit where small data changes cross them — and it belongs at the top."""
    if not churn:
        return ""
    names = {"wealth_tier": "wealth tier", "verdict_direction": "soundness",
             "red_flag_count": "red-flag count", "gate_pass": "gate"}
    bits = [f"<b>{names[k]}</b> {v:.0%}" for k, v in churn.items() if k in names and v == v]
    return (f"<div style='color:{COLORS['text_primary']};font-weight:700;margin-top:8px;'>"
            f"Churn — share of stocks whose label changed: " + " · ".join(bits) + "</div>")


def render_movers(res: dict, meta: dict) -> None:
    """The Movers page below the picker. `meta` carries what only the caller knows:
    prev_vintage, cur_vintage (ISO), prev_label, cur_label (FY quarter), engine, prev_regime,
    cur_regime, mode, profile."""
    c = res["counts"]
    same_engine = meta.get("prev_engine", meta.get("engine")) == meta.get("engine")
    regime_note = ("" if meta.get("prev_regime") == meta.get("cur_regime") else
                   f" — <b>regime changed</b> ({meta.get('prev_regime')} → {meta.get('cur_regime')}): with the "
                   f"adaptive profile every composite shifts with it, so read rank and label moves first")
    st.markdown(
        f"<div style='background:{COLORS['bg_secondary']};border:1px solid {COLORS['border']};"
        f"border-radius:10px;padding:10px 14px;margin:6px 0 10px 0;font-size:0.76rem;line-height:1.6;'>"
        f"<div style='display:flex;flex-wrap:wrap;gap:6px 18px;align-items:baseline;'>"
        f"<span style='font-size:1rem;font-weight:800;color:{COLORS['text_primary']};'>"
        f"{meta['prev_label']} → {meta['cur_label']}</span>"
        f"<span style='color:{COLORS['text_muted']};'>{meta['prev_vintage']} → {meta['cur_vintage']}</span>"
        f"<span style='color:{COLORS['text_muted']};'>{res['n_both']:,} stocks on both sides · "
        f"{c['new']} new · {c['dropped']} dropped</span>"
        f"<span style='color:{COLORS['text_muted']};'>engine {meta.get('engine', 'unknown')}"
        f"{'' if same_engine else ' ⚠ differs'} · {meta.get('mode', '')}/{meta.get('profile', '')}</span>"
        f"<span style='color:{COLORS['text_muted']};'>regime {meta.get('prev_regime', '?')} → {meta.get('cur_regime', '?')}</span>"
        f"</div>"
        + _churn_line(res.get("churn", {}))
        + f"<div style='color:{COLORS['text_secondary']};margin-top:6px;'>Both sides scored by the "
        f"<b>same engine, moments apart</b>, so every move below is the company changing — never PRISM "
        f"changing{regime_note}.</div>"
        + (f"<div style='color:{COLORS['gold']};margin-top:4px;'>⚠️ <b>Cash-flow caveat.</b> Cash-flow "
           f"statements are filed half-yearly, so cash-driven signals (Cash Machine, CFO/EBITDA, the accrual "
           f"flags) genuinely refresh only in the June and December vintages — a change in them here is "
           f"rare and worth a second look.</div>"
           if any(q in meta.get("cur_label", "") for q in ("Q1", "Q3")) else "")
        + "</div>", unsafe_allow_html=True)

    # THE FIRST TABLE: material moves only, capped. Measured on the first live run, the six full
    # sections held 400 upgrades, 450 downgrades and ~1,000 rank moves — a firehose nobody reads.
    # This is the thirty rows that matter; everything below is the evidence behind them.
    mat = material(res)
    _section("⭐ What matters", len(mat),
             "into / out of BUY★ · crossed the gate · new Tsunami · top-25 rank jumps · |Δ flags| ≥ 3 — one row per stock")
    _table(mat, limit=40)

    _section("💹 Wealth tier — upgrades", c["wealth_up"], "rungs climbed on BUY★ › BUY › WATCH★ › WATCH › AVOID")
    _table(res["wealth_up"])
    _section("💹 Wealth tier — downgrades", c["wealth_down"])
    _table(res["wealth_down"])
    if c["wealth_unverifiable"]:
        _section("💹 Wealth tier — unverifiable", c["wealth_unverifiable"], "moved to or from N/A: an input went missing or came back — not a verdict")
        _table(res["wealth_unverifiable"])

    _section("🧭 Soundness — improved", c["sound_up"], "FLAWED › MIXED › SOUND")
    _table(res["sound_up"])
    _section("🧭 Soundness — worsened", c["sound_down"])
    _table(res["sound_down"])

    r = res["rank"]
    _section("📈 Rank climbers", int((r["rank_delta"] > 0).sum()) if not r.empty else 0, "Δ Rank positive = climbed")
    _table(r[r["rank_delta"] > 0] if not r.empty else r)
    _section("📉 Rank fallers", int((r["rank_delta"] < 0).sum()) if not r.empty else 0)
    _table(r[r["rank_delta"] < 0].iloc[::-1] if not r.empty else r)

    _section("✅ New gate passers", c["gate_new"])
    _table(res["gate_new"])
    _section("❌ Lost the gate", c["gate_lost"])
    _table(res["gate_lost"])
    _section("🌊 New Tsunami setups", c["tsunami_new"])
    _table(res["tsunami_new"])

    fl = res["flags"]
    _section("🚩 Red flags — rises", int((fl["flag_delta"] > 0).sum()) if not fl.empty else 0, "forensics deteriorating")
    _table(fl[fl["flag_delta"] > 0] if not fl.empty else fl)
    _section("🚩 Red flags — falls", int((fl["flag_delta"] < 0).sum()) if not fl.empty else 0)
    _table(fl[fl["flag_delta"] < 0].iloc[::-1] if not fl.empty else fl)

    _section("🆕 Fresh results", c["fresh"], "a result landed between the two vintages")
    _table(res["fresh"])

    _section("➕ New to the universe", c["new"], "no deltas — there is no previous side")
    _table(res["new"])
    _section("➖ Dropped from the universe", c["dropped"])
    _table(res["dropped"])
