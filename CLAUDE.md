# CLAUDE.md — Multibagger Discovery System Codex

Behavioral guidelines to reduce common LLM coding mistakes integrated with strict quantamental system invariants.

**Tradeoff:** These guidelines bias toward execution precision and defensive structure over raw speed.

## 0. Google Sheets Tab Name Invariant — NEVER CHANGE

The six Google Sheets tab names are fixed contract values. The data pipeline loads by tab name
(not by GID). Renaming any tab breaks the pipeline for every user. These names are locked:

| Internal key   | Exact tab name      |
|----------------|---------------------|
| `ratio`        | `Ratio`             |
| `income`       | `Income Statement`  |
| `balance`      | `Balance Sheet`     |
| `cashflow`     | `Cashflow`          |
| `shareholding` | `Shareholdings`     |
| `technical`    | `Technicals`        |

These are defined in `SHEET_TAB_NAMES` in `config.py`. Do not rename tabs, do not change these strings,
do not revert to GID-based loading. GIDs are per-spreadsheet and always wrong for a different user's sheet.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

- Before implementing: State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.
- Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified via Pytest.**

- Fix the bug / refactor → write a test that reproduces it, then make it pass.
- For multi-step tasks, state a brief plan and run verifications at each step.
- Full system verification command: `python -m pytest tests/ -q`

## 5. System Architecture & Hard Guardrails

**Pipeline sequence in `get_scored_data()` — absolute, never reorder:**
```
1. compute_forensic_signals(df)   — Piotroski + 27 red flags + Schilit + labels
2. run_full_scoring(df)           — quality/momentum/composite + all framework flags
3. apply_forensic_penalty(df)     — cascading multiplier on composite_score
```
WHY 1 before 2: Diamond, Dhandho, SQGLP Engine, Schilit, and Fisher framework gates read `forensic_score`, `forensic_label`, `red_flag_count`, and `schilit_pass` — these columns must exist before scoring runs.
WHY 3 after 2: `apply_forensic_penalty` multiplies `composite_score` which only exists after step 2.
Both constraints are non-negotiable.

**Single source of truth:** No database layer. The 6 CSV sheets inside `Other Resources/CSV Data/` are the absolute authority on all financial data.

**Absolute vectorization mandate:** Row-wise iterations, loops over rows, lambda row mappings, and `.apply(axis=1)` are strictly banned. All transformations across the 2,108-stock matrix must use pure Pandas/NumPy array operations.

**Defensive denominator protection:** All division operations must use:
```python
np.where(denom > 0.0, num / denom, np.nan)
```
Never divide without this guard. Propagate `np.nan` on zero/missing denominators.

**Semantic truth principle:** Never inject sentinel values (1.0, 0, -1) into intermediate ratio arrays when data is missing. Propagate true `np.nan` through intermediate steps. Only handle safe fallbacks via `.fillna()` at the final flag/score assignment coordinate.

**Stateless UI/presentation layer:** Display scripts inside `ui/ui_tearsheet.py` must remain 100% stateless. Do not introduce interactive widgets (`st.number_input`, `st.button`, `st.columns` for metric layouts, or `st.slider`) inside that module — they mutate global session and stock-selection states managed in `app.py`. Use compact inline HTML/CSS flex/grid containers to eliminate Streamlit padding bloat.

**Forensic flag constant lock:** `FORENSIC_MAX_FLAGS` in `config.py` is currently **27** (27 active `rf_` columns as of 2026-06-01 audit). Increment it manually if and only if a new `rf_` forensic check column is added to `forensic_engine.py`. Verify count with: `python -c "import re; src=open('core/forensic_engine.py').read(); cols=set(re.findall(r'df\[\"(rf_[^\"]+)\"\]\s*=', src)); print(len(cols))"`

**Howard Marks gauge:** The 5-slider cycle gauge in the Config tab is display-only. Those sliders must never modify scoring dataframes or alter composite scores. Macro adaptation is handled objectively by `detect_market_regime()`.

## 6. Test Suite Contract

- Current count: **1419 tests** across `tests/`
- Never reduce test count without explicit user approval
- Test files are contracts — edit a test only when engine logic changes, never to make a failure green
- Stale tests (testing removed code) must be updated to match the current architecture, not deleted

## 7. Zero-Duplicate Framework Emoji Registry

The `_FW_META` dictionary in `ui/ui_tearsheet.py` enforces absolute emoji uniqueness across all 37 frameworks. Enforce this mapping exactly — no two frameworks may share an emoji.

### 🏛️ Motilal Oswal Wealth Creation Frameworks
- "QGLP" ➔ 🥇
- "MOSL Wealth Creator" ➔ 🌟
- "SQGLP Century Stock" ➔ 👑
- "100x Candidate" ➔ 🐘
- "Fallen Quality" ➔ 🩹
- "CAP-GAP Compounder" ➔ 📐
- "Economic Moat" ➔ 🏰
- "Blue Chip Quality" ➔ 💙
- "Consistent in Volatile" ➔ 🌪️
- "EP Hockey Stick" ➔ 🏒
- "Bruised Blue Chip 29" ➔ 🏛️
- "Multi-Trillion Cap" ➔ 🌐

### 📚 Fundamental & Cash Quality Moats
- "Coffee Can" ➔ ☕
- "Diamond" ➔ 💎
- "Peaceful Investing" ➔ 🕊️
- "Unusual Billionaires" ➔ 💰
- "Long Game Quality" ➔ ⏳
- "Baid Compounder" ➔ 📚
- "Basant 30% Club" ➔ 🏅
- "Quality Compounder" ➔ ⭐

### ⚡ Technical Momentum & Growth Sieves
- "CAN SLIM" ➔ 📡
- "SEPA Momentum" ➔ ⚡
- "Quality Momentum" ➔ 🚀
- "Lynch Dream" ➔ 👓
- "EP Improver" ➔ 📈
- "SMILE" ➔ 😊

### 🛡️ Valuation, Capital Allocation & System Defense Shields
- "Magic Formula" ➔ 🧮
- "Dhandho Asymmetry" ➔ 🎲
- "Parikh Contrarian" ➔ 🔄
- "Wide Moat" ➔ 🌊
- "Outsider CEO" ➔ 🎯
- "Expectations Matrix" ➔ 🔮
- "Financial Shenanigans" ➔ 🕵️
- "Marks Cycle Shield" ➔ 🛡️

### 🎣 Fisher & Mayer (not in the 34-row matrix above; kept unique)
- "Fisher Quality" ➔ 🎣
- "Fisher Scalability" ➔ 📶
- "100-Bagger" ➔ 💯
