"""
Plotting components for visualizing results.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, List


def render_plots(predictions: pd.DataFrame, experimental_data: pd.DataFrame, conditions: List[str]):
    """
    Render interactive plots of fitting results.
    
    Args:
        predictions: Model predictions DataFrame
        experimental_data: Original experimental data
        conditions: List of experimental conditions
    """
    st.header("📈 Visualization")
    
    if predictions is None or predictions.empty:
        st.info("Run the parameter fitting to see plots here.")
        return
    
    # Plot type selection
    plot_type = st.radio(
        "Plot style:",
        ["Combined", "Individual Conditions", "Residuals"],
        horizontal=True
    )
    
    if plot_type == "Combined":
        _render_combined_plot(predictions, experimental_data, conditions)
    elif plot_type == "Individual Conditions":
        _render_individual_plots(predictions, experimental_data, conditions)
    else:
        _render_residual_plots(predictions, experimental_data, conditions)


def _render_combined_plot(predictions: pd.DataFrame, experimental_data: pd.DataFrame, conditions: List[str]):
    """Render combined substrate and biomass plots."""
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Substrate Consumption', 'Biomass Growth'),
        horizontal_spacing=0.1
    )
    
    colors = px.colors.qualitative.Set2
    
    for i, cond in enumerate(conditions):
        color = colors[i % len(colors)]
        
        # Get predictions for this condition
        pred_cond = predictions[predictions['Condition'] == cond] if 'Condition' in predictions.columns else predictions
        
        # Substrate plot
        if 'Substrate' in pred_cond.columns:
            fig.add_trace(
                go.Scatter(
                    x=pred_cond['Time'],
                    y=pred_cond['Substrate'],
                    mode='lines',
                    name=f'{cond} (Model)',
                    line=dict(color=color),
                    legendgroup=cond,
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # Biomass plot
        if 'Biomass' in pred_cond.columns:
            fig.add_trace(
                go.Scatter(
                    x=pred_cond['Time'],
                    y=pred_cond['Biomass'],
                    mode='lines',
                    name=f'{cond} (Model)',
                    line=dict(color=color),
                    legendgroup=cond,
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Add experimental points if available
        _add_experimental_points(fig, experimental_data, cond, color, i)
    
    fig.update_layout(
        height=500,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    fig.update_xaxes(title_text="Time (days)", row=1, col=1)
    fig.update_xaxes(title_text="Time (days)", row=1, col=2)
    fig.update_yaxes(title_text="Substrate (mg/L)", row=1, col=1)
    fig.update_yaxes(title_text="Biomass (mg cells/L)", row=1, col=2)
    
    st.plotly_chart(fig, use_container_width=True)


def _render_individual_plots(predictions: pd.DataFrame, experimental_data: pd.DataFrame, conditions: List[str]):
    """Render individual plots for each condition."""
    
    n_conds = len(conditions)
    cols = st.columns(min(n_conds, 2))
    
    colors = px.colors.qualitative.Set2
    
    for i, cond in enumerate(conditions):
        col_idx = i % 2
        color = colors[i % len(colors)]
        
        with cols[col_idx]:
            st.subheader(f"Condition: {cond}")
            
            # Create subplot for this condition
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Substrate', 'Biomass'),
                shared_xaxes=True,
                vertical_spacing=0.15
            )
            
            # Get predictions
            pred_cond = predictions[predictions['Condition'] == cond] if 'Condition' in predictions.columns else predictions
            
            # Substrate
            if 'Substrate' in pred_cond.columns:
                fig.add_trace(
                    go.Scatter(
                        x=pred_cond['Time'],
                        y=pred_cond['Substrate'],
                        mode='lines',
                        name='Model',
                        line=dict(color=color, width=2)
                    ),
                    row=1, col=1
                )
            
            # Biomass
            if 'Biomass' in pred_cond.columns:
                fig.add_trace(
                    go.Scatter(
                        x=pred_cond['Time'],
                        y=pred_cond['Biomass'],
                        mode='lines',
                        name='Model',
                        line=dict(color=color, width=2),
                        showlegend=False
                    ),
                    row=2, col=1
                )
            
            fig.update_layout(
                height=400,
                showlegend=True,
                template="plotly_white"
            )
            
            fig.update_xaxes(title_text="Time (days)", row=2, col=1)
            fig.update_yaxes(title_text="mg/L", row=1, col=1)
            fig.update_yaxes(title_text="mg cells/L", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)


def _render_residual_plots(predictions: pd.DataFrame, experimental_data: pd.DataFrame, conditions: List[str]):
    """Render residual analysis plots using actual residuals."""

    st.subheader("Residual Analysis")
    st.caption("Residuals show the difference between model predictions and experimental data")

    if experimental_data is None or experimental_data.empty or predictions is None or predictions.empty:
        st.info("Both predictions and experimental data are required for residual analysis.")
        return

    # Find time column
    time_col = None
    for col in experimental_data.columns:
        if 'time' in col.lower():
            time_col = col
            break

    if time_col is None:
        st.warning("No time column found in experimental data.")
        return

    # Compute actual residuals by matching experimental and predicted data
    all_substrate_residuals = []
    all_biomass_residuals = []
    all_substrate_pred = []
    all_biomass_pred = []
    all_times = []

    for cond in conditions:
        # Find experimental columns for this condition
        substrate_col = None
        biomass_col = None
        for col in experimental_data.columns:
            if col.startswith(f"{cond}_") or col.startswith(f"{cond} "):
                col_lower = col.lower()
                if 'biomass' in col_lower:
                    biomass_col = col
                elif ('(mg/l)' in col_lower or '(mm)' in col_lower) and 'biomass' not in col_lower:
                    substrate_col = col

        if substrate_col is None and biomass_col is None:
            continue

        # Get predictions for this condition
        if 'Condition' in predictions.columns:
            pred_cond = predictions[predictions['Condition'] == cond]
        else:
            pred_cond = predictions

        if pred_cond.empty:
            continue

        exp_time = experimental_data[time_col].values

        # Interpolate predictions to experimental time points
        if 'Time' in pred_cond.columns and 'Substrate' in pred_cond.columns:
            pred_time = pred_cond['Time'].values
            pred_substrate = pred_cond['Substrate'].values
            pred_biomass = pred_cond['Biomass'].values if 'Biomass' in pred_cond.columns else None

            interp_substrate = np.interp(exp_time, pred_time, pred_substrate)

            if substrate_col:
                obs_substrate = experimental_data[substrate_col].values
                valid = ~np.isnan(obs_substrate)
                residuals_s = obs_substrate[valid] - interp_substrate[valid]
                all_substrate_residuals.extend(residuals_s)
                all_substrate_pred.extend(interp_substrate[valid])
                all_times.extend(exp_time[valid])

            if pred_biomass is not None and biomass_col:
                interp_biomass = np.interp(exp_time, pred_time, pred_biomass)
                obs_biomass = experimental_data[biomass_col].values
                if 'od' in biomass_col.lower():
                    obs_biomass = obs_biomass * 83.0
                valid = ~np.isnan(obs_biomass)
                residuals_b = obs_biomass[valid] - interp_biomass[valid]
                all_biomass_residuals.extend(residuals_b)
                all_biomass_pred.extend(interp_biomass[valid])

    # Plot actual residuals
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Substrate Residual Distribution', 'Substrate Residuals vs Time',
            'Biomass Residual Distribution', 'Biomass Residuals vs Time'
        ),
        vertical_spacing=0.15
    )

    if all_substrate_residuals:
        s_resid = np.array(all_substrate_residuals)
        s_times = np.array(all_times[:len(s_resid)])

        fig.add_trace(
            go.Histogram(x=s_resid, name='Substrate', nbinsx=15,
                         marker_color='steelblue', opacity=0.7),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=s_times, y=s_resid, mode='markers', name='Substrate',
                       marker=dict(size=7, color='steelblue', opacity=0.7)),
            row=1, col=2
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)

    if all_biomass_residuals:
        b_resid = np.array(all_biomass_residuals)
        b_times = np.array(all_times[:len(b_resid)])

        fig.add_trace(
            go.Histogram(x=b_resid, name='Biomass', nbinsx=15,
                         marker_color='coral', opacity=0.7),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=b_times, y=b_resid, mode='markers', name='Biomass',
                       marker=dict(size=7, color='coral', opacity=0.7)),
            row=2, col=2
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)

    if not all_substrate_residuals and not all_biomass_residuals:
        st.info("Could not compute residuals. Ensure predictions and experimental data have matching conditions.")
        return

    fig.update_layout(
        height=600,
        template="plotly_white",
        showlegend=False
    )
    fig.update_xaxes(title_text="Residual (mg/L)", row=1, col=1)
    fig.update_xaxes(title_text="Time (days)", row=1, col=2)
    fig.update_xaxes(title_text="Residual (mg cells/L)", row=2, col=1)
    fig.update_xaxes(title_text="Time (days)", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Residual", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_yaxes(title_text="Residual", row=2, col=2)

    st.plotly_chart(fig, use_container_width=True)

    # Summary statistics
    col1, col2 = st.columns(2)
    if all_substrate_residuals:
        s_arr = np.array(all_substrate_residuals)
        with col1:
            st.markdown(f"**Substrate residuals**: mean = {np.mean(s_arr):.2f}, std = {np.std(s_arr):.2f}")
    if all_biomass_residuals:
        b_arr = np.array(all_biomass_residuals)
        with col2:
            st.markdown(f"**Biomass residuals**: mean = {np.mean(b_arr):.2f}, std = {np.std(b_arr):.2f}")

    st.info("Good fits show residuals randomly scattered around zero with no systematic patterns.")


def _add_experimental_points(fig, experimental_data: pd.DataFrame, condition: str, color: str, idx: int):
    """Add experimental data points to the figure."""
    
    if experimental_data is None or experimental_data.empty:
        return
    
    # Try to find time column
    time_col = None
    for col in experimental_data.columns:
        if 'time' in col.lower() and 'day' in col.lower():
            time_col = col
            break
    
    if time_col is None:
        for col in experimental_data.columns:
            if 'time' in col.lower():
                time_col = col
                break
    
    if time_col is None:
        return
    
    # Find substrate column for this condition
    # Columns can be: "5mM_Glucose (mg/L)", "5mM_Substrate (mg/L)", etc.
    substrate_col = None
    biomass_col = None
    
    for col in experimental_data.columns:
        # Check if this column belongs to this condition
        # Must match exactly at start: "5mM_" should NOT match "15mM_"
        # Use prefix matching with underscore or space separator
        if col.startswith(f"{condition}_") or col.startswith(f"{condition} "):
            col_lower = col.lower()
            if 'biomass' in col_lower:
                biomass_col = col
            # Substrate columns have (mg/L) or (mM) but NOT biomass
            elif ('(mg/l)' in col_lower or '(mm)' in col_lower) and 'biomass' not in col_lower:
                substrate_col = col
    
    time = experimental_data[time_col].values
    
    # Add substrate points to FIRST subplot (col=1)
    if substrate_col and substrate_col in experimental_data.columns:
        substrate = experimental_data[substrate_col].values
        valid = ~np.isnan(substrate)
        
        if np.any(valid):
            fig.add_trace(
                go.Scatter(
                    x=time[valid],
                    y=substrate[valid],
                    mode='markers',
                    name=f'{condition} (Data)',
                    marker=dict(color=color, size=8, symbol='circle'),
                    legendgroup=condition,
                    showlegend=False
                ),
                row=1, col=1
            )
    
    # Add biomass points to SECOND subplot (col=2)
    if biomass_col and biomass_col in experimental_data.columns:
        biomass = experimental_data[biomass_col].values
        valid = ~np.isnan(biomass)
        
        if np.any(valid):
            # Convert OD to cells if needed (rough conversion)
            if 'od' in biomass_col.lower():
                biomass = biomass * 83.0  # OD to mg cells/L
            
            fig.add_trace(
                go.Scatter(
                    x=time[valid],
                    y=biomass[valid],
                    mode='markers',
                    name=f'{condition} (Data)',
                    marker=dict(color=color, size=8, symbol='circle'),
                    legendgroup=condition,
                    showlegend=False
                ),
                row=1, col=2
            )
