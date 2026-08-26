"""
test_moat_growth_matrix.py
==========================
Contracts for the Moat-Growth Matrix (22nd WCS, Exhibit 5) — the scatter on the Matrix & WCS tab.

WHAT WAS WRONG (measured on the live universe, 2026-08-26):
  * The x-axis clipped at a flat 300% growth while the universe's p99 is ~178% — so the canvas
    stretched to fit under 1% of stocks and 1,704 of 2,038 points (83.6%) were crushed into a box
    worth 8.8% of the plot area, exactly where the median stock lives (growth 11.9%, moat 12.9%).
  * Markers were fully opaque, so that dense core was a flat silhouette: one stock and fifty
    looked identical in the only region that mattered.
  * `px.scatter` titles the legend with the column it colours by, so the chart printed the literal
    string "moat_growth_quad" — a dataframe column name — to users.
  * The highlight label was "🎯 <name>", and 🎯 renders as a filled circular glyph indistinguish-
    able from a plotted marker: one stock appeared as two dots.
  * 79 stocks (3.7%) were dropped for a missing axis with nothing on the chart saying so.

THE INVARIANT THAT MUST NOT BREAK: clipping is a VIEWPORT operation. No row is ever removed for
being an outlier (the original G9 fix), and the count beyond the edge is stated — the axis exists
to keep the crowded core readable, never to hide a stock.

WHY THE PERCENTILE STRIP EXISTS: 51.6% of the universe sits within ±5pp of a 15/15 dividing line
and the median stock sits essentially ON the crossing, so for half the universe the categorical
quadrant verdict turns on noise. A percentile does not.

Run with: pytest tests/test_moat_growth_matrix.py -v
"""

import contextlib
import io as _io
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import pytest

import ui.ui_tearsheet as T
from data_engine import (coerce_numeric_columns, compute_derived_signals, load_all_csvs,
                         merge_datasets)


@pytest.fixture(scope="module")
def live():
    with contextlib.redirect_stdout(_io.StringIO()):
        return compute_derived_signals(
            coerce_numeric_columns(merge_datasets(load_all_csvs("local"))))


def _render(frame, highlight=None):
    """Render the matrix and return (figure, emitted markdown)."""
    figs, out = [], []

    class _Rec:
        def markdown(self, *a, **k):
            if a:
                out.append(str(a[0]))

        def plotly_chart(self, fig, *a, **k):
            figs.append(fig)

        def __getattr__(self, _n):
            return lambda *a, **k: None

    real = T.st
    try:
        T.st = _Rec()
        T.render_moat_growth_matrix(frame, highlight_stock=highlight)
    finally:
        T.st = real
    assert figs, "no chart was rendered"
    return figs[0], " ".join(out)


# ── 1. The viewport fits the data, and never drops a row ────────────────────────────────
def test_axis_fits_the_data_not_the_outliers(live):
    fig, _ = _render(live)
    lo, hi = fig.layout.xaxis.range
    plotted = T._moat_growth_plot_frame(live)
    p98, mx = plotted["Growth_X"].quantile(0.98), plotted["Growth_X"].max()
    assert lo == -50
    assert hi < mx, "the axis stretches to the extreme outlier again"
    assert hi == pytest.approx(p98 * 1.02, rel=0.01), f"axis {hi} is not the p98 viewport"
    assert hi < 300, "back to the flat 300% clip that crushed 83.6% of the points"


def test_clipping_is_a_viewport_operation_and_removes_no_stock(live):
    """The whole point of the G9 fix: the canvas clips, the dataset does not."""
    fig, _ = _render(live)
    plotted = T._moat_growth_plot_frame(live)
    drawn = sum(len(tr.x) for tr in fig.data if tr.name != "Selected Stock")
    assert drawn == len(plotted), (
        f"{len(plotted) - drawn} rows vanished from the traces — clipping must not drop points"
    )


def test_stocks_beyond_the_edge_are_counted_on_the_chart(live):
    _, md = _render(live)
    plotted = T._moat_growth_plot_frame(live)
    beyond = int((plotted["Growth_X"] > plotted["Growth_X"].quantile(0.98) * 1.02).sum())
    if beyond:
        m = re.search(r"([\d,]+) sit beyond", md)
        assert m, "stocks are off the right edge and the chart does not say so"
        assert int(m.group(1).replace(",", "")) == beyond


def test_axis_adapts_to_a_filtered_universe(live):
    """A low-growth filter must not keep a wide empty canvas."""
    calm = live[live["pat_gr_5y"].fillna(0).between(-10, 30)]
    if len(calm) < 30:
        pytest.skip("not enough rows in the calm slice")
    fig, _ = _render(calm)
    assert fig.layout.xaxis.range[1] < 100, "viewport did not tighten for a low-growth universe"


# ── 2. Readability of the dense core ────────────────────────────────────────────────────
def test_markers_are_translucent_so_density_is_visible(live):
    fig, _ = _render(live)
    for tr in fig.data:
        if tr.name != "Selected Stock" and tr.mode == "markers":
            assert tr.marker.opacity is not None and tr.marker.opacity < 1.0, (
                "opaque markers turn the dense core back into a flat silhouette"
            )


# ── 3. Nothing internal leaks to the reader ─────────────────────────────────────────────
def test_legend_title_is_not_a_dataframe_column_name(live):
    fig, _ = _render(live)
    title = str(fig.layout.legend.title.text or "")
    assert "_" not in title, f"a raw column name is printed as the legend title: {title!r}"
    assert title == "Quadrant"


def test_highlight_label_carries_no_glyph_that_looks_like_a_point(live):
    fig, _ = _render(live, highlight="Shilchar Technologies Ltd")
    hl = [tr for tr in fig.data if tr.name == "Selected Stock"]
    if not hl:
        pytest.skip("highlight stock not in this snapshot")
    for txt in hl[0].text:
        assert "🎯" not in txt, "the 🎯 glyph renders as a phantom second data point"


def test_omitted_stocks_are_named_not_silently_dropped(live):
    _, md = _render(live)
    omitted = len(live) - len(T._moat_growth_plot_frame(live))
    if omitted:
        m = re.search(r"([\d,]+) of ([\d,]+) stocks are not plotted", md)
        assert m, f"{omitted} stocks are missing from the chart with no note"
        assert int(m.group(1).replace(",", "")) == omitted


# ── 4. The percentile strip — and it must match what is drawn ───────────────────────────
def test_percentile_strip_matches_the_plotted_distribution(live):
    name = "Shilchar Technologies Ltd"
    plotted = T._moat_growth_plot_frame(live)
    if name not in set(plotted["name"]):
        pytest.skip("highlight stock not plotted in this snapshot")
    _, md = _render(live, highlight=name)
    row = plotted[plotted["name"] == name].iloc[0]
    exp_m = (plotted["Moat_Y"] < row["Moat_Y"]).mean() * 100
    exp_g = (plotted["Growth_X"] < row["Growth_X"]).mean() * 100
    found = re.findall(r"([\d.]+)%</span><span[^>]*>(\d+)th percentile", md)
    assert len(found) == 2, f"expected two percentile cells, got {found}"
    (m_val, m_pct), (g_val, g_pct) = found
    assert float(m_val) == pytest.approx(float(row["Moat_Y"]), abs=0.06)
    assert float(g_val) == pytest.approx(float(row["Growth_X"]), abs=0.06)
    assert int(m_pct) == round(exp_m), f"moat percentile {m_pct} != {exp_m:.0f}"
    assert int(g_pct) == round(exp_g), f"growth percentile {g_pct} != {exp_g:.0f}"


def test_percentile_ordinals_are_english(live):
    """Shipped as "93th percentile" first time round — the suffix was a hardcoded 'th'.
    Sweeps real stocks so every ordinal the strip can emit is checked, teens included
    (11/12/13 take 'th', not st/nd/rd)."""
    plotted = T._moat_growth_plot_frame(live)
    seen = {}
    for name in plotted["name"]:
        _, md = _render(live, highlight=name)
        for val, suf in re.findall(r">(\d+)(st|nd|rd|th) percentile", md):
            seen.setdefault(int(val), suf)
        if len(seen) > 40:
            break
    assert len(seen) > 10, "not enough distinct percentiles to be a real sweep"
    wrong = {v: s for v, s in seen.items()
             if s != ("th" if 11 <= v % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(v % 10, "th"))}
    assert not wrong, f"bad ordinal suffixes: {wrong}"


def test_no_percentile_strip_without_a_selected_stock(live):
    _, md = _render(live)
    assert "percentile" not in md, "the strip rendered with no stock selected"


def test_unplotted_highlight_does_not_fabricate_a_percentile(live):
    """A stock excluded for a missing axis has no position — it must not be given one."""
    plotted = set(T._moat_growth_plot_frame(live)["name"])
    missing = [n for n in live["name"] if n not in plotted]
    if not missing:
        pytest.skip("every stock is plottable in this snapshot")
    _, md = _render(live, highlight=missing[0])
    assert "percentile" not in md, f"{missing[0]} is not on the chart but was given a percentile"
