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
        render_financial_insights,
        render_stock_hero,
        render_score_strip,
        render_sell_alerts_panel,
        render_raw_signals,
        render_canslim_radar,
        render_schilit_shield,
        render_dorsey_radar,
        render_outsider_radar,
        render_marks_radar,
        render_malik_radar,
        render_lynch_radar,
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
    render_financial_insights = _stub
    render_stock_hero = _stub
    render_score_strip = _stub
    render_sell_alerts_panel = _stub
    render_raw_signals = _stub
    render_canslim_radar = _stub
    render_schilit_shield = _stub
    render_dorsey_radar = _stub
    render_outsider_radar = _stub
    render_marks_radar    = _stub
    render_malik_radar    = _stub
    render_lynch_radar    = _stub

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
    "render_financial_insights",
    "render_stock_hero",
    "render_score_strip",
    "render_sell_alerts_panel",
    "render_raw_signals",
    "render_canslim_radar",
    "render_schilit_shield",
    "render_dorsey_radar",
    "render_outsider_radar",
    "render_marks_radar",
    "render_malik_radar",
    "render_lynch_radar",
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
