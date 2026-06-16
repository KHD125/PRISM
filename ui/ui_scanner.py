import streamlit as st
import pandas as pd
from config import COLORS

# Graceful fallback for AgGrid
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, ColumnsAutoSizeMode, GridUpdateMode
    AGGRID_AVAILABLE = True
except ImportError:
    AGGRID_AVAILABLE = False

# Plain-language header tooltips for the scanner's machine-named columns. Reuse the SAME glossary
# the tearsheet "?" chip uses (single source of truth — a term can never drift between the grid
# header and the tearsheet); bespoke sentences only for scanner-only columns with no tearsheet cell.
from ui.ui_tearsheet import _RAW_GLOSSARY as _GLOSSARY

_SCANNER_HEADER_TIPS = {
    "rank":             "The stock's overall rank in the current screen (1 = highest conviction).",
    "verdict_direction":"The engine's one-word call (e.g. BUY / WATCH / AVOID) synthesised from all 6 axes after the forensic penalty.",
    "corporate_class":  _GLOSSARY["Corporate Class"],
    "composite_score":  _GLOSSARY["Composite Score"],
    "conviction_tier":  _GLOSSARY["Conviction Tier"],
    "moat_growth_quad": "Where the stock sits on the moat-versus-growth map (e.g. Wealth Creator, Quality Trap, Growth Trap).",
    "fisher_lifecycle_quadrant": "Phil Fisher's growth-versus-quality lifecycle quadrant for the business (e.g. Catalyst Play, Laggard).",
    "cash_score":       "A 0-100 score for how strongly the business converts reported profit into real operating cash.",
    "buy_zone_label":   _GLOSSARY["Buy Zone"],
    "forensic_score":   _GLOSSARY["Forensic Scr"],
    "red_flag_count":   _GLOSSARY["Red Flags"],
    "momentum_score":   _GLOSSARY["Momentum Scr"],
}

def render_scanner_grid(df: pd.DataFrame, priority_cols: list = None):
    """
    Renders the data grid. Uses AgGrid if available for institutional filtering,
    otherwise falls back to standard st.dataframe.
    """
    if len(df) == 0:
        st.warning("No stocks match the current filters.")
        return
        
    # Reorder columns — exact name matching prevents "cash_machine" from pinning
    # both cash_machine_score and cash_machine_label unintentionally
    all_cols = list(df.columns)
    _PINNED = [
        "rank", "name", "verdict_direction", "corporate_class", "sector", "composite_score",
        "moat_growth_quad", "fisher_lifecycle_quadrant", "cash_score", "buy_zone_label",
    ]
    seen: set = set()
    first_cols: list = []
    for col in _PINNED:
        if col in all_cols and col not in seen:
            first_cols.append(col)
            seen.add(col)
    
    if priority_cols:
        for c in priority_cols:
            if c in all_cols and c not in first_cols:
                first_cols.append(c)
                
    remaining = [c for c in all_cols if c not in first_cols]
    final_cols = first_cols + remaining
    
    display_df = df[final_cols].copy()

    if not AGGRID_AVAILABLE:
        st.info("💡 Pro Tip: Install `streamlit-aggrid` for advanced Excel-like filtering.")
        st.dataframe(display_df, use_container_width=True, height=600)
        return
        
    # AgGrid Configuration
    gb = GridOptionsBuilder.from_dataframe(display_df)
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=50)
    gb.configure_side_bar() # Shows the filtering sidebar
    gb.configure_default_column(
        filterable=True,
        sortable=True,
        resizable=True,
        minWidth=100,
        # Display rounding without mutating underlying sort data
        valueFormatter=(
            "params.value === null || params.value === undefined ? '' : "
            "typeof params.value === 'number' ? "
            "(params.value % 1 === 0 ? params.value.toFixed(0) : params.value.toFixed(2)) : "
            "params.value"
        ),
    )
    
    # Pin important columns
    gb.configure_column("rank", pinned="left", width=80, headerTooltip=_SCANNER_HEADER_TIPS["rank"])
    gb.configure_column("name", pinned="left", width=250)

    # Plain-language header tooltips (hover any header) — same glossary as the tearsheet "?" chip.
    for _col, _tip in _SCANNER_HEADER_TIPS.items():
        if _col in ("rank", "name"):
            continue
        if _col in display_df.columns:
            gb.configure_column(_col, headerTooltip=_tip)

    grid_options = gb.build()
    
    st.markdown(f"""
        <style>
        /* Only one AgGrid instance in this system — global scope is safe */
        .ag-theme-streamlit {{
            --ag-header-background-color: {COLORS['bg_secondary']};
            --ag-header-foreground-color: {COLORS['text_primary']};
            --ag-row-hover-color: {COLORS['bg_tertiary']};
        }}
        </style>
    """, unsafe_allow_html=True)

    grid_response = AgGrid(
        display_df,
        gridOptions=grid_options,
        enable_enterprise_modules=False,
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        theme='streamlit',
        height=650,
        width='100%'
    )
    
    return grid_response
