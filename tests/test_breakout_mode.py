"""
test_breakout_mode.py
=====================
Contract: the "Breakout" Analysis Mode — quality, momentum and breakout in EQUAL THIRDS — is a
selectable forward CANDIDATE, and the default composite is byte-identical to before it existed.

WHY A MODE AND NOT THE DEFAULT (2026-09-19). The user proposed blending the composite from
quality + momentum + breakout. Measured on both forward windows in EXACTLY the form the engine
computes (governance 15%, every framework boost, the governance shield and the forensic multiplier
all kept — NOT the bare rank blend, which flatters it):

    form                                   W1 06-17→08-22   W2 08-22→09-19
    Hybrid composite (current default)          +0.134           +0.034
    Q+M+B, engine form                          +0.148           +0.094      <- this mode
    Q+B,   engine form                          +0.148           +0.066
    Q+M,   engine form                          +0.139           +0.065
    quality_score alone                         +0.116           -0.025

It beats the default in both windows and beats Q+M and Q+B; Q+B is 0.975-correlated with it. Two
facts kept it OUT of the default. (1) breakout_score is already 40% of momentum_score
(breakout_proximity + breakout_window), so this mode is really "price strength at ~2/3 of the
composite instead of ~1/4". (2) The windows are 28 and 66 days, and price strength autocorrelates
at that horizon by construction — a one-month test will always drift toward momentum, while the
engine's objective is multi-year compounding, which nothing we hold can measure before December.
So: selectable now, reported by tools/validate.py on every window as composite_qmb_candidate, and
PROMOTED ONLY if it still wins on the December window AND the first 6-month window.

Run with: pytest tests/test_breakout_mode.py -v
"""

import contextlib
import io
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

import numpy as np
import pandas as pd
import pytest

import config as C
from scoring_engine import compute_composite_score

_ROOT = Path(__file__).resolve().parent.parent
_APP = _ROOT / "app.py"


# ── 1. the mode exists and is well-formed ───────────────────────────────────────────────
def test_breakout_mode_is_equal_thirds_and_well_formed():
    m = C.ANALYSIS_MODES["Breakout"]
    for k in ("fundamental_w", "momentum_w", "breakout_w"):
        assert abs(m[k] - 1 / 3) < 1e-12, f"{k} is {m[k]}, expected exactly one third"
    assert m["label"].strip() and m["description"].strip()
    assert set(m["allowed_profiles"]) <= set(C.MASTER_PROFILES), "unknown profile in allowed_profiles"
    assert set(m["allowed_profiles"]) == set(C.ANALYSIS_MODES["Hybrid"]["allowed_profiles"]), (
        "the candidate should offer the same profiles as the mode it is a candidate to replace"
    )


def test_only_the_breakout_mode_carries_a_third_leg():
    """Every other mode is unchanged: no breakout_w key at all (not 0.0 — the leg must not exist
    there, so nothing can quietly grow one)."""
    for name, m in C.ANALYSIS_MODES.items():
        if name == "Breakout":
            continue
        assert "breakout_w" not in m, f"{name} grew a breakout_w — the third leg is Breakout-only"


def test_the_default_mode_is_still_hybrid():
    src = _APP.read_text(encoding="utf-8")
    assert 'st.session_state.setdefault("cfg_mode", "Hybrid")' in src, (
        "the default Analysis Mode changed. The Breakout mode is a CANDIDATE gated on the December "
        "window and the first 6-month window — see its comment in config.ANALYSIS_MODES"
    )


# ── 2. the engine: inert leg by default, exact blend when asked ────────────────────────
def _frame(n=6, with_breakout=True):
    rng = np.random.default_rng(7)
    f = pd.DataFrame({
        "quality_score":    rng.uniform(10, 90, n),
        "momentum_score":   rng.uniform(10, 90, n),
        "governance_bonus": rng.uniform(0, 60, n),
        "pb_ratio":         rng.uniform(0.5, 8, n),
        "roe":              rng.uniform(5, 30, n),
        "rev_gr_5y":        rng.uniform(0, 30, n),
        "roce_med_5y":      rng.uniform(5, 30, n),
        "market_cap":       rng.uniform(500, 50000, n),
    })
    if with_breakout:
        f["breakout_score"] = rng.uniform(0, 100, n)
    return f


def test_default_modes_are_byte_identical_and_never_touch_breakout_score():
    """breakout_w = 0 must SKIP the leg, not multiply by zero: a frame with NO breakout_score
    column (every pre-existing synthetic test) must still score, and a frame WITH one must give
    the identical composite whether the column is present or absent."""
    gov = C.COMPOSITE_WEIGHTS["governance"]
    with_b = compute_composite_score(_frame(with_breakout=True), 0.70, 0.30)
    without = compute_composite_score(_frame(with_breakout=False), 0.70, 0.30)
    assert np.allclose(with_b["composite_score"], without["composite_score"])
    f = _frame(with_breakout=True)
    out = compute_composite_score(f, 0.70, 0.30)
    expected = (f["quality_score"] * 0.70 + f["momentum_score"] * 0.30) * (1 - gov) + f["governance_bonus"] * gov
    # no framework flag fires on this frame, so the only post-blend operation is the clip
    assert np.allclose(out["composite_score"], expected.clip(0, 100))


def test_breakout_mode_blends_the_three_legs_in_equal_thirds():
    gov = C.COMPOSITE_WEIGHTS["governance"]
    f = _frame(with_breakout=True)
    out = compute_composite_score(f, 1 / 3, 1 / 3, breakout_w=1 / 3)
    expected = ((f["quality_score"] + f["momentum_score"] + f["breakout_score"]) / 3) * (1 - gov) \
               + f["governance_bonus"] * gov
    assert np.allclose(out["composite_score"], expected.clip(0, 100))
    # and it is genuinely different from Hybrid on the same frame
    hyb = compute_composite_score(_frame(with_breakout=True), 0.70, 0.30)["composite_score"]
    assert not np.allclose(out["composite_score"], hyb)


def test_a_positive_breakout_weight_requires_the_column():
    """No silent fallback: asking for the leg without the column is a programming error, not a
    NaN composite or a quiet 50 — the engine always has breakout_score by the time it blends."""
    with pytest.raises(KeyError):
        compute_composite_score(_frame(with_breakout=False), 1 / 3, 1 / 3, breakout_w=1 / 3)


# ── 3. live: the mode re-ranks, stays NaN-free, and the default is untouched ────────────
@pytest.fixture(scope="module")
def live_pair():
    from core import fetch_and_clean_data, run_scoring_pipeline
    with contextlib.redirect_stdout(io.StringIO()):
        clean = fetch_and_clean_data("local")
        hyb = run_scoring_pipeline(clean.copy(), analysis_mode="Hybrid")
        brk = run_scoring_pipeline(clean.copy(), analysis_mode="Breakout")
    h = hyb.set_index("company_id")                       # set_index returns a NEW frame —
    b = brk.set_index("company_id").reindex(h.index)      # align on the indexed copy, not `hyb`
    assert b["quality_score"].notna().sum() == h["quality_score"].notna().sum(), "alignment lost rows"
    return h, b


def test_breakout_mode_scores_every_stock_and_genuinely_reranks(live_pair):
    hyb, brk = live_pair
    c_h = pd.to_numeric(hyb["composite_score"], errors="coerce")
    c_b = pd.to_numeric(brk["composite_score"], errors="coerce")
    assert c_b.isna().sum() == 0, "the Breakout composite has NaN rows"
    assert c_b.between(0, 100).all()
    rho = c_h.rank().corr(c_b.rank())
    assert 0.50 < rho < 0.97, f"rank-corr Hybrid vs Breakout {rho:+.3f}: either not a real re-rank or a different engine"
    top_h, top_b = set(c_h.nlargest(50).index), set(c_b.nlargest(50).index)
    assert 15 <= len(top_h & top_b) <= 48, f"top-50 overlap {len(top_h & top_b)}/50 (35/50 at the switch)"


def test_breakout_mode_leaves_everything_but_the_composite_chain_untouched(live_pair):
    """The mode changes the BLEND, nothing upstream of it: quality, momentum, breakout, governance
    and the forensic multiplier must be identical between the two runs, or the mode is doing
    something other than what its name says."""
    hyb, brk = live_pair
    for c in ("quality_score", "momentum_score", "breakout_score", "governance_bonus",
              "forensic_multiplier", "gate_pass", "red_flag_count"):
        a, b = pd.to_numeric(hyb[c], errors="coerce"), pd.to_numeric(brk[c], errors="coerce")
        assert np.allclose(a, b, equal_nan=True), f"{c} differs between modes — the mode leaked upstream"


# ── 4. surfaces: the third leg is visible where the other two are ──────────────────────
def test_config_tab_shows_the_breakout_weight_only_for_modes_that_carry_it():
    src = _APP.read_text(encoding="utf-8")
    i = src.index("_mode_rows = ")
    block = src[i:i + 1400]
    assert "breakout_w" in block and "Breakout" in block, "the weights row does not print the third leg"
    assert 'if _v.get("breakout_w", 0.0) > 0.0 else ""' in block, (
        "the Breakout cell must be conditional — printing 0% on Hybrid claims a leg it does not have"
    )
    assert '"Breakout": "⚡"' in src, "no icon for the new mode in _mode_icon"
    assert "Breakout 33/33/33" in src, "the selector help does not mention the new mode"
    # caught by the browser, not the tests: the FORMULA line above the rows still read
    # "Quality × F + Momentum × M + Governance" — incomplete for the mode that has a third leg
    assert "Breakout × B" in src, "the composite formula line does not name the third leg"
    assert "F, M and B (B in the Breakout mode only)" in src


def test_validate_registers_the_candidate_and_reads_weights_from_config():
    v = (_ROOT / "tools" / "validate.py")
    if not v.exists():
        pytest.skip("tools/ is gitignored; the harness is local-only")
    src = v.read_text(encoding="utf-8")
    assert '"composite_qmb_candidate"' in src
    assert "ANALYSIS_MODES" in src and 'ANALYSIS_MODES.get("Breakout")' in src
    assert not re.search(r"0\.595|0\.255", src), "Hybrid weights retyped in the harness — read them from config"
