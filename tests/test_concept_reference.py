"""Contract: the concept reference — EVERY categorical value-label PRISM produces is explained, and
the explanations are not shallow. The completeness guarantee is now CODE-DRIVEN, not hand-curated:
the universe of labels is extracted from the code that produces them (np.select / np.where string
choices + a few importable label constants + the catalyst/alert dicts), so a category can never be
silently forgotten — the test RED-fails listing the exact missing labels.

Guards:
  1. COVERAGE — every code-produced categorical label has a CONCEPT_REFERENCE explanation (or is
     consciously grandfathered in _INTENTIONAL_INTERNAL with a reason).
  2. QUALITY — every explanation clears a 40-char floor (catches "there but not properly explained").
  3. PURITY  — ui_reference_data.py is pure data (no imports / no Streamlit).
"""
import ast
import re
from pathlib import Path

from ui.ui_reference_data import CONCEPT_REFERENCE

_ROOT = Path(__file__).resolve().parent.parent

# Files that PRODUCE categorical display labels (np.select choice-lists, etc.).
_ENGINE_FILES = ["core/data_engine.py", "core/forensic_engine.py",
                 "core/scoring_engine.py", "core/verdict_engine.py"]


def _np_call_str_labels(src, attr):
    """Every STRING-literal choice/branch of every `np.<attr>(...)` call — the label VALUES.
    np.select: args[1] (choicelist List) + default= / args[2].  np.where: the branch args.
    Numeric-choice calls (scores) yield no strings and are naturally excluded."""
    out = set()
    for n in ast.walk(ast.parse(src)):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == attr
                and isinstance(n.func.value, ast.Name) and n.func.value.id == "np"):
            continue
        cands = []
        if attr == "select":
            if len(n.args) >= 2 and isinstance(n.args[1], ast.List):
                cands += n.args[1].elts
            cands += [kw.value for kw in n.keywords if kw.arg == "default"]
            if len(n.args) >= 3:
                cands.append(n.args[2])
        else:  # where — the true/false branch values
            cands += list(n.args[1:])
        out |= {c.value for c in cands
                if isinstance(c, ast.Constant) and isinstance(c.value, str)}
    return out


def _dict_keys(src, var):
    """String keys of the `var = {…}` dict literal (e.g. _CATALYSTS / _SELL_ALERTS)."""
    for n in ast.walk(ast.parse(src)):
        if (isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == var for t in n.targets)
                and isinstance(n.value, ast.Dict)):
            return {k.value for k in n.value.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    return set()


def _fw_meta_keys():
    """The 37 framework names — AST-parsed from the function-local _FW_META in ui_tearsheet.py (no
    import / no refactor of that 3k-line, parallel-owned module; same AST approach as the emoji test)."""
    src = (_ROOT / "ui" / "ui_tearsheet.py").read_text(encoding="utf-8")
    for n in ast.walk(ast.parse(src)):
        if (isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "_FW_META" for t in n.targets)
                and isinstance(n.value, ast.Dict)):
            return {k.value for k in n.value.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    return set()


def _universe():
    """The full categorical-label universe, extracted FROM THE CODE that produces it."""
    u = set()
    for f in _ENGINE_FILES:
        u |= _np_call_str_labels((_ROOT / f).read_text(encoding="utf-8"), "select")
    disc = (_ROOT / "ui" / "ui_discovery.py").read_text(encoding="utf-8")
    u |= _np_call_str_labels(disc, "where")                  # piotroski strength tiers
    u |= _dict_keys(disc, "_CATALYSTS") | _dict_keys(disc, "_SELL_ALERTS")
    u |= _fw_meta_keys()                                      # the 37 frameworks
    from config import CONVICTION_TIERS
    u |= {t["label"] for t in CONVICTION_TIERS}
    from core.cyclicality_map import TIER_LABELS
    u |= set(TIER_LABELS.values())
    # Genuine display LABELS only — drop: format/template strings; and NARRATIVE sentences (the verdict
    # synthesis produces full self-explanatory sentences via np.select — they end with '.' / run long /
    # contain mid-sentence '. '; those are explanations already, not labels to define).
    def _is_label(x):
        x = x.strip()
        return (re.search(r"[A-Za-z]", x) and "{" not in x and "\n" not in x and "%" not in x
                and not x.endswith(".") and ". " not in x and len(x) <= 48)
    return {x for x in u if _is_label(x)}


def _clean(s):
    """Emoji/punct-insensitive key — sources differ (emoji labels vs plain 'label' fields)."""
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()


def _clean_key(s):
    return " ".join(_clean(s))


# Labels the enumerator pulls that are genuinely INTERNAL / never shown to a user — excused with a
# REASON only after review. Anything NOT here MUST be explained. (Mirrors the glossary grandfather.)
_INTENTIONAL_INTERNAL = {}


def test_every_categorical_label_is_explained():
    explained = {_clean_key(l) for cat in CONCEPT_REFERENCE.values() for l, _ in cat}
    missing = sorted(u for u in _universe()
                     if _clean_key(u) not in explained and u not in _INTENTIONAL_INTERNAL)
    assert not missing, (
        f"{len(missing)} categorical labels the code PRODUCES but CONCEPT_REFERENCE never explains "
        f"(explain each from its source, or grandfather with a reason):\n  " + "\n  ".join(missing))


def test_no_shallow_explanation():
    thin = [(lbl, len(exp.strip())) for entries in CONCEPT_REFERENCE.values()
            for lbl, exp in entries if len(exp.strip()) < 40]
    assert not thin, f"explanations under the 40-char quality floor: {thin}"


def test_reference_data_is_pure():
    src = (_ROOT / "ui" / "ui_reference_data.py").read_text(encoding="utf-8")
    assert "import streamlit" not in src and "\nimport " not in src, \
        "ui_reference_data.py must be pure data — no imports / no Streamlit"


def test_every_forensic_flag_has_a_display_description():
    """Every forensic rf_ flag the engine computes MUST have a _FLAG_DISPLAY description — else it
    fires on a Tear-Sheet with no explanation AND is absent from the Reference tab (which renders
    _FLAG_DISPLAY directly). This is the forensic-flag completeness guarantee, drift-proof."""
    from ui.ui_tearsheet import _FLAG_DISPLAY
    from config import FORENSIC_MAX_FLAGS
    src = (_ROOT / "core" / "forensic_engine.py").read_text(encoding="utf-8")
    rf_cols = set(re.findall(r'df\["(rf_[^"]+)"\]\s*=', src))
    missing = sorted(rf_cols - set(_FLAG_DISPLAY))
    assert not missing, f"forensic rf_ flags with NO _FLAG_DISPLAY description: {missing}"
    assert len(_FLAG_DISPLAY) == FORENSIC_MAX_FLAGS, \
        f"_FLAG_DISPLAY has {len(_FLAG_DISPLAY)} entries; FORENSIC_MAX_FLAGS = {FORENSIC_MAX_FLAGS}"
