"""The QGLP screen is FIXED to Raamdeo Agrawal's own weights — it is not a user setting.

WHY THE SELECTOR WAS REMOVED (2026-09-20). It offered eight "profiles" and was measured across
all eight on the live 2,717-stock universe:

  * composite_score, rank, conviction_tier and the top-50 were IDENTICAL for every profile.
    Nothing scored reads the profile — that is a behavioural proof, stronger than a grep.
  * It moved exactly three columns: qglp_score, qglp_pass (307 under Quality -> 1,274 under
    Turnaround, a 4.1x swing) and frameworks_passed (up to 861 rows).
  * compute_qglp_score was the ONLY scorer of 38 frameworks taking a `profile` argument, so
    "QGLP passed" was the only framework verdict that was a preference rather than a fact.

THE DEFECT THAT DECIDED IT — the profile changed WITHOUT BEING TOUCHED. Each ANALYSIS_MODES
entry carried an `allowed_profiles` list, and app.py snapped cfg_profile into it on every rerun:

    _allowed_profiles = ANALYSIS_MODES[cfg_mode]["allowed_profiles"]
    if cfg_profile not in _allowed_profiles:
        cfg_profile = _allowed_profiles[0]          # one-way; never restored

"Technical Only" allowed only ["Momentum", "Turnaround"], so the round trip
Breakout -> Technical Only -> Breakout left the QGLP screen on **Momentum** and it stuck, because
Momentum is legal under Breakout too. Live cost of that trip, with the user never having opened
the profile dropdown: QGLP passers 413 -> 631, the MOSL convergence table 709 -> 789, 218 stocks'
mosl_n changed — while composite_score, rank and every header tile stayed IDENTICAL, so nothing
on screen moved and nothing warned. Path-dependent, silent, and invisible to every check a
reader would make.

TWO CONSEQUENCES BEYOND THE CONTROL ITSELF:
 1. The 🔭 MOSL tab counts QGLP as one of ten Wealth-Creation lenses and argues in its own
    caption that "these come from one research programme, so agreement between them means
    something". A dropdown that moves one of the ten manufactures that agreement.
 2. ui_export.stamp_snapshot records five provenance columns (snapshot_vintage, snapshot_source,
    engine_version, scored_at, scored_mode) and NO profile. The Cloud download scores under the
    live session profile, so a snapshot could carry qglp_pass from a screen nothing recorded —
    the same hole scored_mode closed for the analysis mode on 2026-09-19, never closed for the
    profile. Fixing the profile closes it without a sixth column.

WHAT IS KEPT: QGLP the FRAMEWORK is untouched — one of 38, with its own Market Pulse tab,
tearsheet radar, All-Data cell, Reference entry and `fw_qglp` frame tag. So is the regime
cascade: get_adaptive_weights("Balanced", regime) still moves the gates, which is why
MASTER_PROFILES keeps its (single) entry rather than being inlined.

These pins are ABSENCE assertions on purpose. The "existence is not reachability" trap this repo
has paid for four times bites PRESENCE scans — `if False and ...` satisfies "the source contains
X". An absence assertion fails on dead code too, so it is safe in the direction it is used here.
"""

import ast
import io
import re

import pytest

import config as C
from core.scoring_engine import compute_qglp_score


_APP = io.open("app.py", encoding="utf-8").read()
_APP_AST = ast.parse(_APP)

# Agrawal's QGLP as the 25th Wealth Creation Study states it. Retyping these anywhere else is
# the "re-derived threshold" class that produced the Dilution 48.6% contradiction (2026-09-19).
_BOOK_GATES = {"roce_gate": 15.0, "growth_gate": 15.0, "peg_gate": 1.5}
_BOOK_WEIGHTS = {"quality_w": 0.35, "growth_w": 0.35, "longevity_w": 0.15, "price_w": 0.15}


# ── 1. The profile is a constant, not a menu ──────────────────────────────────────────────

def test_exactly_one_qglp_profile_exists():
    """Eight profiles meant eight meanings for one framework's verdict. There is now one."""
    assert list(C.MASTER_PROFILES) == ["Balanced"], (
        f"MASTER_PROFILES must hold exactly the one book profile, got {sorted(C.MASTER_PROFILES)}"
    )


def test_the_one_profile_carries_the_books_gates_and_weights():
    """If a refresh or a refactor moves these, the QGLP screen stops being Agrawal's."""
    p = C.MASTER_PROFILES["Balanced"]
    for k, v in {**_BOOK_GATES, **_BOOK_WEIGHTS}.items():
        assert p[k] == pytest.approx(v), f"Balanced.{k} is {p[k]}, book says {v}"
    four = [p["quality_w"], p["growth_w"], p["longevity_w"], p["price_w"]]
    assert sum(four) == pytest.approx(1.0), f"QGLP base weights sum to {sum(four)}"


# ── 2. The silent-rewrite path is gone, structurally ──────────────────────────────────────

def test_no_analysis_mode_carries_an_allowed_profiles_list():
    """The list was the mechanism: it is what app.py snapped cfg_profile into on a mode change.

    With no list there is nothing to snap into, so the Technical-Only round trip cannot move the
    QGLP screen. This fails the moment anyone reintroduces the key.
    """
    offenders = sorted(m for m, cfg in C.ANALYSIS_MODES.items() if "allowed_profiles" in cfg)
    assert not offenders, (
        f"{offenders} carry allowed_profiles — reintroducing it reopens the silent-rewrite path "
        f"(Breakout -> Technical Only -> Breakout left the screen on Momentum)"
    )


def test_app_never_writes_the_profile_into_session_state():
    """An ABSENCE pin over the AST: any assignment targeting st.session_state["cfg_profile"].

    A source-text scan would be satisfied by a commented-out line; the AST sees only real
    assignments, and dead-but-parsed code still counts, which is the safe direction here.
    """
    writes = []
    for node in ast.walk(_APP_AST):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for t in targets:
            if (isinstance(t, ast.Subscript)
                    and isinstance(t.slice, ast.Constant)
                    and t.slice.value == "cfg_profile"):
                writes.append(getattr(node, "lineno", "?"))
    assert not writes, (
        f"app.py writes st.session_state['cfg_profile'] at line(s) {writes} — that assignment IS "
        f"the silent-rewrite defect; the profile must be a constant, not session state"
    )


def test_exactly_one_scoring_selectbox_and_it_is_the_analysis_mode():
    """Walk the AST for st.selectbox calls carrying a cfg_* key. There must be one: cfg_mode.

    Counting real Call nodes, not text — an `if False:` block around a second selectbox would
    still be found, which is what this pin wants.
    """
    keys = []
    for node in ast.walk(_APP_AST):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == "selectbox"):
            continue
        for kw in node.keywords:
            if kw.arg == "key" and isinstance(kw.value, ast.Constant):
                if str(kw.value.value).startswith("cfg_"):
                    keys.append(kw.value.value)
    assert sorted(keys) == ["cfg_mode"], (
        f"the Config tab must offer exactly one scoring control (cfg_mode); found {sorted(keys)}"
    )


# ── 3. The engine's default path IS the book path ─────────────────────────────────────────

def _qglp_frame():
    """Minimal frame exercising both sides of every QGLP gate (roce 15, growth 15, peg 1.5)."""
    import pandas as pd
    return pd.DataFrame({
        # passes all three / fails roce / fails growth / fails peg / negative peg
        "roce":        [25.0, 10.0, 25.0, 25.0, 25.0],
        "pat_gr_5y":   [22.0, 22.0,  5.0, 22.0, 22.0],
        "eps_gr_5y":   [20.0, 20.0,  4.0, 20.0, 20.0],
        "peg":         [ 0.9,  0.9,  0.9,  3.0, -1.0],
        "roe_med_10y": [18.0, 12.0, 18.0, 18.0, 18.0],
    })


def test_calling_compute_qglp_score_with_no_profile_uses_the_book():
    """`profile=None` falls back to MASTER_PROFILES["Balanced"] — the ONE remaining path.

    Behavioural, not structural: the frame is built so each row sits on a different side of a
    different gate, so a changed gate changes the output rather than merely the source text.
    """
    bare = compute_qglp_score(_qglp_frame())
    explicit = compute_qglp_score(_qglp_frame(), profile=C.MASTER_PROFILES["Balanced"])
    assert list(bare["qglp_pass"]) == list(explicit["qglp_pass"])
    assert bare["qglp_score"].round(9).tolist() == explicit["qglp_score"].round(9).tolist()
    # row 0 clears all three; rows 1-4 each fail exactly one (negative PEG is a fail, not a pass)
    assert list(bare["qglp_pass"]) == [1, 0, 0, 0, 0], (
        f"the book gates must admit only the all-clear row, got {list(bare['qglp_pass'])}"
    )


def test_the_radars_gate_fallbacks_agree_with_the_book():
    """ui_tearsheet's radar resolves gates from the profile with a literal fallback:

        roce_gate = prof.get("roce_gate", 15.0)

    The fallback is the same safe shape as MASTER_PROFILES.get(name, ...["Balanced"]) and is
    unreachable while the key exists, so it is NOT a defect. The drift hazard is real though:
    change config's gate to 18 and leave the literal at 15 and two numbers describe one gate,
    the Fisher/Dilution class. This pin makes the literal track the book instead of banning it.
    """
    ts = io.open("ui/ui_tearsheet.py", encoding="utf-8").read()
    # the radar block only — a stray 15.0 elsewhere in a 200KB file is not this test's business
    start = ts.index("def render_qglp_radar")
    block = ts[start:start + 6000]
    assert "MASTER_PROFILES" in block, "the radar must resolve its thresholds from the profile"
    found = dict(re.findall(r'\.get\(\s*"(roce_gate|growth_gate|peg_gate)"\s*,\s*([\d.]+)\s*\)',
                            block))
    assert set(found) == set(_BOOK_GATES), (
        f"expected all three gate reads in render_qglp_radar, found {sorted(found)}"
    )
    for gate, literal in found.items():
        assert float(literal) == pytest.approx(_BOOK_GATES[gate]), (
            f"render_qglp_radar falls back to {gate}={literal} but the book profile says "
            f"{_BOOK_GATES[gate]} — one gate, two numbers"
        )


# ── 4. The regime cascade survives — it was never the problem ─────────────────────────────

def test_the_regime_still_moves_the_qglp_gates():
    """Fixing the PROFILE must not freeze the REGIME. BEAR must tighten against SIDEWAYS.

    This is the reason MASTER_PROFILES keeps an entry instead of the gates being inlined:
    get_adaptive_weights layers regime deltas on top of the book numbers.
    """
    sw = C.get_adaptive_weights("Balanced", "SIDEWAYS")
    br = C.get_adaptive_weights("Balanced", "BEAR")
    bl = C.get_adaptive_weights("Balanced", "BULL")
    # SIDEWAYS is the identity: every delta is 0.0, so it must reproduce the book exactly.
    for k, v in _BOOK_GATES.items():
        assert sw[k] == pytest.approx(v), f"SIDEWAYS must be the book unchanged; {k}={sw[k]}"
    # BEAR tightens the QUALITY bar (+5.0). This is the leg the 2026-08-30 audit measured
    # (qglp_pass 325 SIDEWAYS -> 309 BULL -> 249 BEAR) and the one that must never go inert.
    assert br["roce_gate"] > sw["roce_gate"], "BEAR must tighten the ROCE gate"
    # It deliberately RELAXES growth and PEG — config's own comments give the reason
    # ("everyone is suffering", "denominator is depressed"). Asserting a uniform tightening
    # would pin an intuition over the engine's documented design, so the direction is pinned
    # per gate, from config, rather than assumed.
    assert br["growth_gate"] < sw["growth_gate"], "BEAR relaxes the growth gate by design"
    assert br["peg_gate"] > sw["peg_gate"], "BEAR relaxes the PEG gate by design"
    assert (br["roce_gate"], br["peg_gate"]) != (bl["roce_gate"], bl["peg_gate"]), (
        "BEAR and BULL must differ, or the regime cascade is inert"
    )
    for r in ("BULL", "BEAR", "SIDEWAYS"):
        w = C.get_adaptive_weights("Balanced", r)
        four = [w["quality_w"], w["growth_w"], w["longevity_w"], w["price_w"]]
        assert sum(four) == pytest.approx(1.0), f"{r} weights sum to {sum(four)}"


def test_an_unknown_profile_name_still_degrades_to_the_book():
    """The `.get(name, MASTER_PROFILES["Balanced"])` shape is the safety net that makes this
    removal safe: a stale caller passing "Turnaround" gets the book, never a KeyError."""
    stale = C.get_adaptive_weights("Turnaround", "SIDEWAYS")
    book = C.get_adaptive_weights("Balanced", "SIDEWAYS")
    assert stale == book, "an unknown profile must fall back to Balanced, not raise or invent"


# ── 5. QGLP the FRAMEWORK is untouched ────────────────────────────────────────────────────

def test_no_surface_prints_the_constant_profile_at_the_reader():
    """`scoring_profile` is now the literal "Balanced". Interpolating it into anything the reader
    sees prints a constant that names a control they cannot change.

    Four surfaces did exactly that before 2026-09-20 and were cleaned: the Deep Scanner header
    chip, the scanner export button + filename, the scoring spinner, and the Movers status line
    and header cell. Passing the variable as an ARGUMENT stays legal — that is how the radar and
    get_adaptive_weights receive it — so this scans only for interpolation into a display string.
    """
    bad = []
    for line_no, line in enumerate(_APP.splitlines(), 1):
        code = line.split("#", 1)[0]                      # never match a comment (prose-match trap)
        if "{scoring_profile" not in code:
            continue
        # A CACHE KEY IS NOT A SURFACE. `_score_key = f"{sig}::{mode}::{profile}"` interpolates it
        # and must keep doing so — the key identifies what was scored, it is never rendered.
        if re.match(r"\s*_\w*key\s*=", code):
            continue
        bad.append(f"app.py:{line_no}: {code.strip()[:90]}")
    assert not bad, (
        "these interpolate the constant profile into a rendered string:\n  " + "\n  ".join(bad)
    )


def test_removing_the_control_did_not_remove_the_framework():
    """The selector went; the MOSL framework stays. Its frame tag must still be emitted."""
    src = io.open("core/scoring_engine.py", encoding="utf-8").read()
    assert re.search(r'np\.where\(\s*fw_qglp\s*,\s*"QGLP\|"', src), (
        "the QGLP frame tag must still be written into frameworks_passed — it is one of the "
        "ten MOSL convergence lenses the 🔭 tab counts"
    )
    assert 'df["qglp_pass"]' in src and 'df["qglp_score"]' in src
