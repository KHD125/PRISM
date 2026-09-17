"""
test_dependency_contract.py
===========================
Contract for requirements.txt — the file that decides which engine Streamlit Cloud actually runs.

WHY THIS EXISTS, in one outage. On 2026-09-17 the file carried `pandas>=2.0.0` while the dev venv
held pandas 3.0.0, so all 2,839 tests were verified on a combination pip REFUSES TO BUILD
(`streamlit 1.54.0 has requirement pandas<3,>=1.4.0, but you have pandas 3.0.0`) while Cloud, under
the same floor, resolved to pandas 2.3.3. Local and deploy had silently diverged across 724 derived
columns, straddling the pandas 2 -> 3 copy-on-write boundary, and NOTHING said so: a floor never
errors, it just quietly ships a different engine than the one the suite signed off.

Two more defects fell out of the same audit, both of the same class — something real that no file
declared:

  * `requests` is imported at core/sheet_meta.py:84 and was NOT in requirements.txt. It worked only
    because streamlit happens to pull it transitively.
  * `scipy` is needed by three tests (`Series.corr(method="spearman")` imports it) and is in no
    requirements file at all. They passed on the dev venv, which happened to have it, and died the
    instant the suite ran against the deployed set. Fixed at the source instead of by declaring it:
    tests/test_roce_inflection.py::_spearman is now Pearson-on-ranks, verified identical to 1.4e-17.

WHAT IS DELIBERATELY NOT ASSERTED. Pinning the eight direct dependencies does NOT make the
environments identical, and this file does not pretend otherwise: 33 transitive packages still
float, and Cloud actively rewrites at least one of them ("Detected pyarrow 25.0.1 (known segfault,
apache/arrow#50471). Replacing with pyarrow<25."). Local python is 3.13, Cloud runs 3.14.7. What
pinning buys is the elimination of MAJOR-version surprise in the numeric stack, which is the class
that silently changes results. Full parity would need a frozen lockfile and is not worth the
maintenance until something forces it.

Run with: pytest tests/test_dependency_contract.py -v
"""
import ast
import os
import pathlib
import re
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_REQ = _ROOT / "requirements.txt"

# Packages whose import name differs from their distribution name.
_IMPORT_TO_DIST = {
    "PIL": "pillow",
    "python_calamine": "python-calamine",
}

# Declared but never imported, ON PURPOSE. Keep this list tiny and always say why — it is the
# escape hatch that stops test_every_declared_requirement_is_used from deleting a live dependency.
_RUNTIME_ONLY = {
    "openpyxl": "the _xlsx_engine() fallback names it as a STRING (core/data_engine.py:356), so no "
                "import scan can see it; removing it turns a missing calamine wheel into a boot failure",
}

# Everything the app itself ships. tools/ is gitignored and tests/ never reaches Cloud, so neither
# belongs in requirements.txt — their dependencies are a separate question by design.
_SHIPPED = ([_ROOT / "app.py", _ROOT / "config.py"]
            + sorted((_ROOT / "core").glob("*.py"))
            + sorted((_ROOT / "ui").glob("*.py")))

_FIRST_PARTY = {"core", "ui", "config", "app", "alpha", "tools", "tests", "stockscans_sync"}


def _requirements():
    """-> {dist_name: pinned_version_or_None}, inline comments stripped."""
    out = {}
    for raw in _REQ.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*(==|>=|<=|~=|>|<)?\s*([0-9][\w.\-]*)?", line)
        assert m, f"requirements.txt line is not parseable: {raw!r}"
        out[m.group(1).lower()] = (m.group(2), m.group(3))
    return out


def _shipped_third_party():
    """Top-level third-party modules imported ANYWHERE in shipped code, including inside functions.

    ast.walk (not just module-level nodes) is load-bearing: `requests` is imported inside a function
    body and a top-level-only scan — or a grep for '^import' — misses it entirely. That is exactly
    how it stayed undeclared.
    """
    mods = set()
    for path in _SHIPPED:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                mods.add(n.module.split(".")[0])
    std = set(sys.stdlib_module_names)
    return sorted(m for m in mods
                  if m not in std and m not in _FIRST_PARTY and not m.startswith("_"))


def test_every_shipped_import_is_declared():
    """A module the app imports but requirements.txt omits runs only by luck of a transitive pull."""
    req = _requirements()
    missing = []
    for mod in _shipped_third_party():
        dist = _IMPORT_TO_DIST.get(mod, mod).lower()
        if dist not in req:
            missing.append(f"{mod} (distribution {dist!r})")
    assert not missing, (
        "shipped code imports these, but requirements.txt does not declare them — they reach Cloud "
        "only as somebody else's transitive dependency, and vanish the day that changes:\n  "
        + "\n  ".join(missing))


def test_every_requirement_is_pinned_exactly():
    """A floor (>=) means the DEPLOYED engine is not the one the suite verified."""
    loose = [f"{name}{op or ''}{ver or ''}" for name, (op, ver) in _requirements().items()
             if op != "==" or not ver]
    assert not loose, (
        "these are not pinned to an exact version, so a Cloud rebuild can silently change the "
        "engine under 724 derived columns:\n  " + "\n  ".join(loose))


def test_every_declared_requirement_is_used():
    """A dependency nothing imports still gets installed — and can constrain the ones that matter.

    `streamlit-aggrid` was the live case: retired from the code (see ui/ui_scanner.py's docstring),
    absent from the dev venv, so 2,839 tests passed without it — while Cloud installed it on every
    build, where it constrains streamlit's own version range.
    """
    imported = {_IMPORT_TO_DIST.get(m, m).lower() for m in _shipped_third_party()}
    unused = [name for name in _requirements()
              if name not in imported and name not in _RUNTIME_ONLY]
    assert not unused, (
        "declared but never imported — remove them, or add to _RUNTIME_ONLY WITH THE REASON:\n  "
        + "\n  ".join(unused))


def test_the_runtime_only_allowlist_is_still_true():
    """The allowlist is the one place this file can lie. Check its claims rather than trust them."""
    for name in _RUNTIME_ONLY:
        assert name in _requirements(), f"{name} is allowlisted but no longer declared"
    src = (_ROOT / "core" / "data_engine.py").read_text(encoding="utf-8")
    assert '"openpyxl"' in src or "'openpyxl'" in src, (
        "openpyxl is allowlisted as a string-named engine, but data_engine.py no longer names it — "
        "either the fallback is gone (then drop the requirement) or it moved (then update the reason)")


def test_installed_versions_match_the_pins():
    """The guard that would have caught the 2026-09-17 outage before it shipped.

    Fails loudly on a drifted dev venv rather than letting the suite certify an engine that Cloud
    will never build. If this is red, the fix is `pip install -r requirements.txt`.
    """
    import importlib.metadata as md
    drift = []
    for name, (op, ver) in _requirements().items():
        if op != "==" or not ver:
            continue                      # already reported by the pinning test
        try:
            have = md.version(name)
        except md.PackageNotFoundError:
            drift.append(f"{name}: pinned {ver}, NOT INSTALLED")
            continue
        if have != ver:
            drift.append(f"{name}: pinned {ver}, installed {have}")
    assert not drift, (
        "this environment does not match what Cloud deploys, so the suite is certifying a "
        "different engine than production runs:\n  " + "\n  ".join(drift)
        + "\n\nfix: pip install -r requirements.txt")
