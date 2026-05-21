"""
UI Rendering Components
=======================
Exposes all frontend visualization and layout rendering modules.
"""

from .ui_scanner import render_scanner_grid
try:
    from .ui_tearsheet import (
        render_moat_growth_matrix,
        render_fisher_module,
        render_ep_power_curve_module,
        render_bruised_blue_chip_badge,
        render_multitrillioncap_card,
        render_forensic_perimeter,
        render_guru_frameworks,
    )
except ImportError as e:
    import traceback
    traceback.print_exc()
    # Soft-fail: create stub functions so app.py doesn't crash
    def _stub(*args, **kwargs):
        import streamlit as st
        st.warning(f"⚠️ Tearsheet module unavailable: {e}")
    render_moat_growth_matrix = _stub
    render_fisher_module = _stub
    render_ep_power_curve_module = _stub
    render_bruised_blue_chip_badge = _stub
    render_multitrillioncap_card = _stub
    render_forensic_perimeter = _stub
    render_guru_frameworks = _stub
from .ui_components import (
    inject_css,
    render_hero_banner,
    render_metric_strip,
    render_stock_card,
    render_radar_chart,
    render_tier_summary,
    render_score_bar,
    render_sidebar_brand,
    render_bruised_blue_chips,
    render_multi_trillion_tipping_points,
)

__all__ = [
    "render_scanner_grid",
    "render_moat_growth_matrix",
    "render_fisher_module",
    "render_ep_power_curve_module",
    "render_bruised_blue_chip_badge",
    "render_multitrillioncap_card",
    "render_forensic_perimeter",
    "render_guru_frameworks",
    "inject_css",
    "render_hero_banner",
    "render_metric_strip",
    "render_stock_card",
    "render_radar_chart",
    "render_tier_summary",
    "render_score_bar",
    "render_sidebar_brand",
    "render_bruised_blue_chips",
    "render_multi_trillion_tipping_points",
]
