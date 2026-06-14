import streamlit as st
import pandas as pd
from config import COLORS

# Graceful fallback for AgGrid
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, ColumnsAutoSizeMode, GridUpdateMode
    AGGRID_AVAILABLE = True
except ImportError:
    AGGRID_AVAILABLE = False

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
    gb.configure_column("rank", pinned="left", width=80)
    gb.configure_column("name", pinned="left", width=250)
    
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
