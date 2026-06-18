"""Contract: glossary integrity — every rendered "?" tooltip resolves, and no glossary entry is
dead. Static (no data dir, no Streamlit runtime): AST-walks the UI for help_chip()/_cell() call
sites and the scanner's _GLOSSARY[...] refs, then cross-checks the _RAW_GLOSSARY single source of
truth. Pins the 2026-06-17 tooltip-accuracy audit so coverage can't silently rot.

NO threshold check by design: the glossary DELIBERATELY states definitional thresholds for binary
flags (e.g. "Elite ROE fires at >=35%") — stripping them would gut the explanations, and a regex
can't tell a flag's defining threshold from an editorial value-judgment.
"""
import ast
import re
from pathlib import Path

from ui.ui_components import _RAW_GLOSSARY
from ui.ui_scanner import _SCANNER_HEADER_TIPS

_ROOT = Path(__file__).resolve().parent.parent
_PY_FILES = sorted([_ROOT / "app.py", *(_ROOT / "ui").glob("*.py")])


def _src(path):
    return path.read_text(encoding="utf-8")


# ── label-extraction helpers (pure; unit-tested for teeth below) ──
def _static_help_chip_labels(src):
    """Labels passed to help_chip(...) as a STRING LITERAL with NO explicit tip — these MUST
    resolve in _RAW_GLOSSARY. Dynamic/variable labels and explicit-tip calls are out of scope."""
    out = []
    for node in ast.walk(ast.parse(src)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "help_chip"):
            continue
        has_tip = len(node.args) >= 2 or any(kw.arg == "tip" for kw in node.keywords)
        if has_tip:
            continue
        label_node = node.args[0] if node.args else next(
            (kw.value for kw in node.keywords if kw.arg == "label"), None)
        if isinstance(label_node, ast.Constant) and isinstance(label_node.value, str):
            out.append(label_node.value)
    return out


def _call_first_str_args(src, func_name):
    """First positional string-literal arg of every func_name(...) call (e.g. _cell labels)."""
    out = []
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == func_name and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            out.append(node.args[0].value)
    return out


def _scanner_glossary_refs(src):
    """Glossary keys the scanner pulls via _GLOSSARY["Key"] / _GLOSSARY['Key']."""
    return [m or n for m, n in re.findall(r'_GLOSSARY\[(?:"([^"]+)"|\'([^\']+)\')\]', src)]


# Dynamic (variable / f-string) help_chip labels the AST cannot see — each cited to its render site.
_DYNAMIC_LABELS = {
    # render_verdict_scorecard 6-axis grid — help_chip(axis_key), ui_tearsheet.py ~L1361/1378
    "Moat Axis", "Growth Axis", "Valuation Axis", "Balance Axis", "Governance Axis", "Forensics Axis",
    # Deep Signals strip — _ds(label) -> help_chip(label), ~L1402-1408
    "WCS", "Econ-Profit", "VCR", "Terms-of-Trade", "Cash-Machine",
    # Entry Timing strip — _ds(label), ~L1430-1436
    "RS", "Traj", "EPS-Accel", "Vol",
    # Score strip — help_chip(f"{label} Score"), ~L1632 + app.py:500 (_SS_LABELS + " Score")
    "Moat Score", "Growth Score", "Cash Score", "Momentum Score", "Governance Score",
}

# Glossary entries intentionally retained though no current render site references them.
# Grandfathered here with a reason; NEVER deleted without approval. Any NEW unused entry must fail
# test_no_dead_glossary_entries and force a conscious decision.
_INTENTIONAL_UNUSED = set()


def _used_keys():
    used = set(_DYNAMIC_LABELS)
    for f in _PY_FILES:
        s = _src(f)
        used.update(_static_help_chip_labels(s))
        used.update(_call_first_str_args(s, "_cell"))
        used.update(_scanner_glossary_refs(s))
    return used


# ── 1. quality net ──
def test_every_glossary_value_is_plain_language():
    for k, v in _RAW_GLOSSARY.items():
        # 40-char floor (raised from 20, 2026-06-18): catches SHALLOW, not just empty — and matches
        # the concept-reference quality floor. All 173 entries already clear it (audited; min = 40).
        assert isinstance(v, str) and len(v.strip()) >= 40, f"glossary {k!r} too short/shallow"


def test_every_scanner_tip_is_plain_language():
    for col, tip in _SCANNER_HEADER_TIPS.items():
        assert isinstance(tip, str) and len(tip.strip()) >= 20, f"scanner tip {col!r} too short/empty"


# ── 2. coverage: static help_chip labels resolve ──
def test_static_help_chip_labels_resolve():
    missing = []
    for f in _PY_FILES:
        for lbl in _static_help_chip_labels(_src(f)):
            if lbl not in _RAW_GLOSSARY:
                missing.append((f.name, lbl))
    assert not missing, f"help_chip string-literal labels with no glossary entry: {missing}"


def test_coverage_helper_has_teeth():
    found = _static_help_chip_labels('help_chip("__missing_label__")')
    assert "__missing_label__" in found and "__missing_label__" not in _RAW_GLOSSARY
    # explicit tip / dynamic label must NOT be collected as a must-resolve static label
    assert _static_help_chip_labels("help_chip(axis_key)") == []
    assert _static_help_chip_labels('help_chip("X", "bespoke tip text here")') == []
    assert _static_help_chip_labels('help_chip(tip="bespoke")') == []


# ── 3. no dead entries ──
def test_no_dead_glossary_entries():
    orphans = sorted(set(_RAW_GLOSSARY) - _used_keys() - _INTENTIONAL_UNUSED)
    assert not orphans, (
        "glossary entries no render site uses (wire up, or grandfather in _INTENTIONAL_UNUSED "
        f"with a reason): {orphans}"
    )


def test_dead_check_has_teeth():
    fake = "__definitely_unused_entry__"
    assert fake not in _used_keys() and fake not in _INTENTIONAL_UNUSED
