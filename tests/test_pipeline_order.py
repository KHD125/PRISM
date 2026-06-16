"""
test_pipeline_order.py
======================
Guards the ONE invariant CLAUDE.md §5 calls "absolute, never reorder": the 4-step scoring
sequence —
    compute_forensic_signals → run_full_scoring → apply_forensic_penalty → compute_verdict.

WHY this test exists (Phase-1 audit finding C2): the sequence used to live ONLY inside
app.get_scored_data, which cannot be imported under pytest (app.py runs st.set_page_config /
inject_css at module top level). So a reorder would have shipped green, and the two
integration tests that build a full frame each re-implemented the sequence differently — one
omitted compute_verdict, another omitted both the penalty and verdict steps. The fix pins the
canonical sequence as a single importable unit (core.run_scoring_pipeline) and this test
locks its order + data-threading so neither can silently regress.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

import core


def test_pipeline_runs_all_four_stages_in_locked_order(monkeypatch):
    calls = []

    def _stage(name):
        def _fn(df, *args, **kwargs):
            calls.append(name)
            return df
        return _fn

    monkeypatch.setattr(core, "compute_forensic_signals", _stage("compute_forensic_signals"))
    monkeypatch.setattr(core, "run_full_scoring", _stage("run_full_scoring"))
    monkeypatch.setattr(core, "apply_forensic_penalty", _stage("apply_forensic_penalty"))
    monkeypatch.setattr(core, "compute_verdict", _stage("compute_verdict"))

    core.run_scoring_pipeline(pd.DataFrame({"name": ["A", "B"]}), "Hybrid", "Balanced")

    assert calls == [
        "compute_forensic_signals",
        "run_full_scoring",
        "apply_forensic_penalty",
        "compute_verdict",
    ]


def test_pipeline_threads_frame_through_every_stage(monkeypatch):
    """Each stage must receive the PRIOR stage's output (catches a stage whose result is
    computed but not reassigned — e.g. `compute_verdict(df)` instead of `df = ...`)."""
    def _tag(name):
        def _fn(df, *args, **kwargs):
            df = df.copy()
            df[name] = 1
            return df
        return _fn

    monkeypatch.setattr(core, "compute_forensic_signals", _tag("s1"))
    monkeypatch.setattr(core, "run_full_scoring", _tag("s2"))
    monkeypatch.setattr(core, "apply_forensic_penalty", _tag("s3"))
    monkeypatch.setattr(core, "compute_verdict", _tag("s4"))

    out = core.run_scoring_pipeline(pd.DataFrame({"name": ["A"]}))

    assert {"s1", "s2", "s3", "s4"}.issubset(out.columns)


def test_pipeline_defaults_match_run_full_scoring(monkeypatch):
    """Calling with no mode/profile must forward run_full_scoring's documented defaults
    (Hybrid / Balanced) so the standalone snapshot path scores identically to the app."""
    seen = {}

    monkeypatch.setattr(core, "compute_forensic_signals", lambda df, *a, **k: df)
    monkeypatch.setattr(core, "apply_forensic_penalty", lambda df, *a, **k: df)
    monkeypatch.setattr(core, "compute_verdict", lambda df, *a, **k: df)

    def _capture(df, analysis_mode="Hybrid", scoring_profile="Balanced"):
        seen["mode"] = analysis_mode
        seen["profile"] = scoring_profile
        return df

    monkeypatch.setattr(core, "run_full_scoring", _capture)

    core.run_scoring_pipeline(pd.DataFrame({"name": ["A"]}))

    assert seen == {"mode": "Hybrid", "profile": "Balanced"}
