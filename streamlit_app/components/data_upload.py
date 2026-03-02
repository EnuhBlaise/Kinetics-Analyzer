"""
Data upload component for experimental data.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from ..config import DATA_DIR


def render_data_upload():
    """
    Render the data upload section.
    
    Returns:
        Uploaded DataFrame or None
    """
    st.header("📁 Experimental Data")
    
    # Upload options
    upload_method = st.radio(
        "Data source:",
        ["Upload CSV file", "Use example data"],
        horizontal=True
    )
    
    df = None
    
    if upload_method == "Upload CSV file":
        uploaded_file = st.file_uploader(
            "Upload your experimental data (CSV)",
            type=['csv'],
            help="CSV file with time, substrate, and biomass columns"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state['uploaded_filename'] = uploaded_file.name
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    else:
        # Show available example files
        example_files = list(DATA_DIR.glob("*.csv")) if DATA_DIR.exists() else []
        
        if example_files:
            selected_file = st.selectbox(
                "Select example data:",
                [f.name for f in example_files]
            )
            
            if selected_file:
                file_path = DATA_DIR / selected_file
                try:
                    df = pd.read_csv(file_path)
                    st.session_state['uploaded_filename'] = selected_file
                except Exception as e:
                    st.error(f"Error loading example: {e}")
        else:
            st.warning("No example data files found in data/example/")
    
    # Display data preview and validation
    if df is not None:
        _display_data_preview(df)
        _validate_data(df)
        st.session_state['experimental_data'] = df
    
    return df


def _display_data_preview(df: pd.DataFrame):
    """Display a preview of the uploaded data."""
    
    with st.expander("📋 Data Preview", expanded=True):
        # Basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            # Count conditions
            conditions = _detect_conditions(df)
            st.metric("Conditions", len(conditions))
        
        # Show first few rows
        st.dataframe(df.head(10), use_container_width=True)
        
        # Column info
        st.caption(f"**Columns:** {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")


def _validate_data(df: pd.DataFrame):
    """Validate the uploaded data format."""
    
    issues = []
    warnings = []
    
    # Check for time column
    time_cols = [c for c in df.columns if 'time' in c.lower()]
    if not time_cols:
        issues.append("❌ No time column found (should contain 'time' in name)")
    else:
        # Check for days column
        day_cols = [c for c in time_cols if 'day' in c.lower()]
        if not day_cols:
            warnings.append("⚠️ No 'days' time column found - will use first time column")
    
    # Check for substrate columns - look for (mg/L) or (mM) columns that are NOT biomass
    # Data format can be: {conc}_{SubstrateName} (mg/L) e.g., 5mM_Glucose (mg/L), 5mM_VanillicAcid (mg/L)
    substrate_cols = [c for c in df.columns if ('(mg/L)' in c or '(mM)' in c) and 'biomass' not in c.lower()]
    if not substrate_cols:
        issues.append("❌ No substrate columns found (expected format: '{conc}_{SubstrateName} (mg/L)' e.g., '5mM_Glucose (mg/L)')")
    else:
        # Show detected substrate columns
        warnings.append(f"✅ Found {len(substrate_cols)} substrate columns")
    
    # Check for biomass columns
    biomass_cols = [c for c in df.columns if 'biomass' in c.lower()]
    if not biomass_cols:
        issues.append("❌ No biomass columns found (format: '{conc}_Biomass (OD)' or similar)")
    else:
        warnings.append(f"✅ Found {len(biomass_cols)} biomass columns")
    
    # Check for NaN values
    nan_counts = df.isna().sum()
    if nan_counts.any():
        nan_cols = nan_counts[nan_counts > 0]
        warnings.append(f"⚠️ Missing values in {len(nan_cols)} columns (will be handled automatically)")
    
    # Display validation results
    if issues:
        for issue in issues:
            st.error(issue)
    
    if warnings:
        for warning in warnings:
            st.warning(warning)
    
    if not issues:
        st.success("✅ Data format looks valid!")
        
        # Show detected conditions
        conditions = _detect_conditions(df)
        if conditions:
            st.info(f"**Detected conditions:** {', '.join(conditions)}")
            st.session_state['conditions'] = conditions


def _detect_conditions(df: pd.DataFrame) -> list:
    """Detect experimental conditions from column names."""
    conditions = set()
    
    for col in df.columns:
        # Look for patterns like "5mM_", "10mM_"
        parts = col.split('_')
        if len(parts) >= 2 and ('mM' in parts[0] or 'mg' in parts[0].lower()):
            conditions.add(parts[0])
    
    return sorted(list(conditions), key=lambda x: float(''.join(c for c in x if c.isdigit()) or 0))
