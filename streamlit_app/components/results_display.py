"""
Results display component for showing fitting results.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any


def render_results(results: Dict[str, Any]):
    """
    Render the fitting results.

    Args:
        results: Dictionary containing fitting results
    """
    st.header("📊 Fitting Results")

    if not results:
        st.info("Run the parameter fitting to see results here.")
        return

    # Status indicator
    success = results.get('success', False)
    if success:
        st.success("✅ Optimization completed successfully!")
    else:
        st.error("❌ Optimization failed or did not converge.")

    # Individual condition fitting has its own display
    if results.get('fit_method') == 'individual':
        _render_individual_results(results)
        return

    # Main metrics
    _render_summary_metrics(results)

    # Detailed tabs
    tab1, tab2, tab3 = st.tabs(["📈 Parameters", "📉 Statistics", "📋 Full Report"])

    with tab1:
        _render_parameters_table(results)

    with tab2:
        _render_statistics(results)

    with tab3:
        _render_full_report(results)


def _render_summary_metrics(results: Dict[str, Any]):
    """Render summary metrics at the top."""
    
    statistics = results.get('statistics', {})
    parameters = results.get('parameters', {})
    
    # Check if we have separate statistics (new format)
    has_separate = 'R_squared_substrate' in statistics
    
    if has_separate:
        # Show separate substrate and biomass statistics
        st.subheader("Fit Quality (Separate Metrics)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Substrate Fit**")
            r2_sub = statistics.get('R_squared_substrate', 0)
            rmse_sub = statistics.get('RMSE_substrate', 0)
            nrmse_sub = statistics.get('NRMSE_substrate', 0)
            
            color = "🟢" if r2_sub >= 0.9 else ("🟡" if r2_sub >= 0.7 else "🔴")
            st.metric("R² (Substrate)", f"{r2_sub:.4f} {color}")
            st.metric("RMSE (Substrate)", f"{rmse_sub:.2f}")
            st.metric("NRMSE (Substrate)", f"{nrmse_sub:.4f}")
        
        with col2:
            st.markdown("**Biomass Fit**")
            r2_bio = statistics.get('R_squared_biomass', 0)
            rmse_bio = statistics.get('RMSE_biomass', 0)
            nrmse_bio = statistics.get('NRMSE_biomass', 0)
            
            color = "🟢" if r2_bio >= 0.9 else ("🟡" if r2_bio >= 0.7 else "🔴")
            st.metric("R² (Biomass)", f"{r2_bio:.4f} {color}")
            st.metric("RMSE (Biomass)", f"{rmse_bio:.2f}")
            st.metric("NRMSE (Biomass)", f"{nrmse_bio:.4f}")
        
        with col3:
            st.markdown("**Combined (Weighted)**")
            r2 = statistics.get('R_squared', 0)
            aic = statistics.get('AIC', 0)
            bic = statistics.get('BIC', 0)
            
            color = "🟢" if r2 >= 0.9 else ("🟡" if r2 >= 0.7 else "🔴")
            st.metric("R² (Combined)", f"{r2:.4f} {color}")
            st.metric("AIC", f"{aic:.1f}")
            st.metric("Parameters", len(parameters))
        
        # Add note about separate statistics
        st.caption("💡 Substrate and biomass are fitted with normalized errors so both contribute equally. NRMSE values allow direct comparison across variables with different scales.")
    else:
        # Legacy single combined metric display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            r2 = statistics.get('R_squared', 0)
            delta_color = "normal" if r2 >= 0.9 else ("off" if r2 >= 0.7 else "inverse")
            st.metric(
                "R² Score",
                f"{r2:.4f}",
                delta="Good" if r2 >= 0.9 else ("Fair" if r2 >= 0.7 else "Poor"),
                delta_color=delta_color
            )
        
        with col2:
            rmse = statistics.get('RMSE', 0)
            st.metric("RMSE", f"{rmse:.2f}")
        
        with col3:
            aic = statistics.get('AIC', 0)
            st.metric("AIC", f"{aic:.1f}")
        
        with col4:
            n_params = len(parameters)
            st.metric("Parameters", n_params)


def _render_parameters_table(results: Dict[str, Any]):
    """Render parameters with confidence intervals."""
    
    parameters = results.get('parameters', {})
    confidence_intervals = results.get('confidence_intervals', {})
    units = results.get('units', {})
    
    # Build dataframe
    data = []
    for param_name, value in parameters.items():
        ci = confidence_intervals.get(param_name, {})
        unit = units.get(param_name, '')
        
        std_err = ci.get('std_error', np.nan)
        ci_lower = ci.get('ci_lower', np.nan)
        ci_upper = ci.get('ci_upper', np.nan)
        rel_err = ci.get('relative_error_pct', np.nan)
        
        data.append({
            'Parameter': param_name,
            'Value': value,
            'Std Error': std_err if not np.isnan(std_err) else '-',
            '95% CI Lower': ci_lower if not np.isnan(ci_lower) else '-',
            '95% CI Upper': ci_upper if not np.isnan(ci_upper) else '-',
            'Rel. Error (%)': f"{rel_err:.1f}" if not np.isnan(rel_err) else '-',
            'Unit': unit
        })
    
    df = pd.DataFrame(data)
    
    # Style the dataframe
    st.dataframe(
        df,
        use_container_width=True,
        column_config={
            'Value': st.column_config.NumberColumn(format="%.6f"),
            'Std Error': st.column_config.TextColumn(),
            '95% CI Lower': st.column_config.TextColumn(),
            '95% CI Upper': st.column_config.TextColumn(),
        }
    )
    
    # Parameter interpretation
    st.markdown("---")
    st.subheader("Parameter Interpretation")
    
    for param_name, value in parameters.items():
        ci = confidence_intervals.get(param_name, {})
        std_err = ci.get('std_error', np.nan)
        
        if np.isnan(std_err) or std_err == 0:
            st.markdown(f"**{param_name}** = {value:.4f} ⚠️ _At boundary constraint_")
        elif std_err / abs(value) > 0.3 if value != 0 else False:
            st.markdown(f"**{param_name}** = {value:.4f} ± {std_err:.4f} ⚠️ _High uncertainty_")
        else:
            st.markdown(f"**{param_name}** = {value:.4f} ± {std_err:.4f} ✓")


def _render_statistics(results: Dict[str, Any]):
    """Render detailed statistics."""
    
    statistics = results.get('statistics', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Goodness of Fit")
        
        r2 = statistics.get('R_squared', 0)
        rmse = statistics.get('RMSE', 0)
        mae = statistics.get('MAE', 0)
        
        st.markdown(f"""
        | Metric | Value | Interpretation |
        |--------|-------|----------------|
        | **R²** | {r2:.4f} | {'Excellent' if r2 >= 0.95 else 'Good' if r2 >= 0.85 else 'Fair' if r2 >= 0.70 else 'Poor'} |
        | **RMSE** | {rmse:.4f} | Root mean square error |
        | **MAE** | {mae:.4f} | Mean absolute error |
        """)
    
    with col2:
        st.subheader("Model Selection")
        
        aic = statistics.get('AIC', 0)
        bic = statistics.get('BIC', 0)
        n_obs = statistics.get('n_observations', 0)
        n_params = statistics.get('n_parameters', 0)
        
        st.markdown(f"""
        | Criterion | Value | Notes |
        |-----------|-------|-------|
        | **AIC** | {aic:.2f} | Lower is better |
        | **BIC** | {bic:.2f} | Penalizes complexity more |
        | **N observations** | {n_obs} | Data points |
        | **N parameters** | {n_params} | Free parameters |
        """)
    
    # SSE breakdown
    sse = statistics.get('SSE', 0)
    st.markdown(f"**Sum of Squared Errors (SSE):** {sse:.4f}")


def _render_full_report(results: Dict[str, Any]):
    """Render a full JSON report."""
    
    import json
    
    # Prepare report
    report = {
        'model_type': results.get('model_type', 'unknown'),
        'conditions': results.get('conditions', []),
        'success': results.get('success', False),
        'parameters': results.get('parameters', {}),
        'confidence_intervals': results.get('confidence_intervals', {}),
        'statistics': results.get('statistics', {}),
        'units': results.get('units', {})
    }
    
    # Convert numpy types for JSON
    def convert_numpy(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj
    
    report = convert_numpy(report)

    # Display
    st.json(report)

    # Download button
    st.download_button(
        label="📥 Download Full Report (JSON)",
        data=json.dumps(report, indent=2),
        file_name="fitting_results.json",
        mime="application/json"
    )


def _render_individual_results(results: Dict[str, Any]):
    """Render results from individual condition fitting."""

    per_cond = results.get('per_condition_results', {})
    individual_losses = results.get('individual_losses', {})
    param_summary = results.get('parameter_summary', {})
    global_params = results.get('parameters', {})
    global_cis = results.get('confidence_intervals', {})
    units = results.get('units', {})

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Per-Condition Parameters",
        "🌐 Global Parameters",
        "📉 Fit Quality",
        "📋 Full Report"
    ])

    with tab1:
        _render_per_condition_parameters(per_cond, units)

    with tab2:
        _render_global_parameters(
            global_params, global_cis, param_summary, units, results
        )

    with tab3:
        _render_per_condition_statistics(per_cond, individual_losses)

    with tab4:
        _render_full_report(results)


def _render_per_condition_parameters(
    per_cond: Dict[str, Any], units: Dict[str, str]
):
    """Render parameter tables for each condition with CIs."""

    st.subheader("Individual Condition Parameters")
    st.caption(
        "Each condition was fitted independently. Parameters with high "
        "relative error or CIs spanning zero may be poorly identified."
    )

    for cond, cond_data in per_cond.items():
        success = cond_data.get('success', False)
        status = "✅" if success else "❌"
        with st.expander(f"{status} {cond}", expanded=True):
            params = cond_data.get('parameters', {})
            cis = cond_data.get('confidence_intervals', {})

            data = []
            for param, value in params.items():
                ci = cis.get(param, {})
                se = ci.get('std_error', np.nan)
                ci_lo = ci.get('ci_lower', np.nan)
                ci_hi = ci.get('ci_upper', np.nan)
                rel_err = ci.get('relative_error_pct', np.nan)

                data.append({
                    'Parameter': param,
                    'Value': value,
                    'Std Error': f"{se:.4f}" if not np.isnan(se) else '-',
                    '95% CI Lower': f"{ci_lo:.4f}" if not np.isnan(ci_lo) else '-',
                    '95% CI Upper': f"{ci_hi:.4f}" if not np.isnan(ci_hi) else '-',
                    'Rel. Error (%)': f"{rel_err:.1f}" if not np.isnan(rel_err) else '-',
                    'Unit': units.get(param, '')
                })

            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)

            # Flag problematic parameters
            for param, value in params.items():
                ci = cis.get(param, {})
                se = ci.get('std_error', np.nan)
                if not np.isnan(se) and value != 0 and abs(se / value) > 0.5:
                    st.warning(
                        f"**{param}**: Relative error "
                        f"({abs(se/value)*100:.0f}%) is high — "
                        f"parameter may be poorly identified for this condition."
                    )


def _render_global_parameters(
    global_params: Dict[str, float],
    global_cis: Dict[str, Any],
    param_summary: Dict[str, Any],
    units: Dict[str, str],
    results: Dict[str, Any]
):
    """Render global parameter estimates with CIs and summary statistics."""

    st.subheader("Recommended Global Parameters")
    st.caption(
        "Global parameters obtained by minimizing J(θ) = Σ Lᵢ(θ) across "
        "all conditions. CIs computed from Hessian of J at the global optimum."
    )

    # Global parameters table
    data = []
    for param, value in global_params.items():
        ci = global_cis.get(param, {})
        se = ci.get('std_error', np.nan)
        ci_lo = ci.get('ci_lower', np.nan)
        ci_hi = ci.get('ci_upper', np.nan)
        rel_err = ci.get('relative_error_pct', np.nan)
        summary = param_summary.get(param, {})
        cv = summary.get('cv', np.nan)

        data.append({
            'Parameter': param,
            'Global Value': value,
            'Std Error': f"{se:.4f}" if not np.isnan(se) else '-',
            '95% CI Lower': f"{ci_lo:.4f}" if not np.isnan(ci_lo) else '-',
            '95% CI Upper': f"{ci_hi:.4f}" if not np.isnan(ci_hi) else '-',
            'CV across conditions (%)': f"{cv:.1f}" if not np.isnan(cv) else '-',
            'Unit': units.get(param, '')
        })

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

    # Global optimization info
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        gl = results.get('global_loss')
        st.metric("Global Cost J(θ*)", f"{gl:.6f}" if gl is not None else "-")
    with col2:
        conv = results.get('global_optimization_converged')
        st.metric("Converged", "Yes" if conv else "No")
    with col3:
        nfev = results.get('global_n_function_evals')
        st.metric("Function Evals", nfev if nfev is not None else "-")

    # Parameter variation across conditions
    st.markdown("---")
    st.subheader("Parameter Variation Across Conditions")
    st.caption("Summary statistics of individual condition estimates.")
    summary_data = []
    for param, stats in param_summary.items():
        summary_data.append({
            'Parameter': param,
            'Mean': f"{stats['mean']:.6f}",
            'Std': f"{stats['std']:.6f}",
            'CV (%)': f"{stats['cv']:.1f}",
            'Min': f"{stats['min']:.6f}",
            'Max': f"{stats['max']:.6f}",
            'Median': f"{stats['median']:.6f}",
        })
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary, use_container_width=True)

    # Strategy explanation
    with st.expander("How were global parameters obtained?"):
        st.markdown("""
**Two-Stage Global Estimation Strategy:**

1. **Stage 1 (Individual fits):** Each condition was fitted independently,
   minimizing its own local loss function Lᵢ(θᵢ).

2. **Initial guess:** The median of individual estimates was used as the
   starting point (robust to outliers from poorly identified conditions).

3. **Stage 2 (Global optimization):** A global cost function
   J(θ) = Σᵢ Lᵢ(θ) was minimized via L-BFGS-B, finding the single
   parameter set that best explains all conditions simultaneously.

4. **Confidence intervals:** Computed from the Hessian of J at the
   global optimum using the t-distribution with (n_total - p)
   degrees of freedom.
        """)


def _render_per_condition_statistics(
    per_cond: Dict[str, Any],
    individual_losses: Dict[str, float]
):
    """Render fit quality statistics for each condition."""

    st.subheader("Fit Quality by Condition")

    # Summary metrics
    conditions = list(per_cond.keys())
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**R² Values**")
        r2_data = []
        for cond in conditions:
            stats = per_cond[cond].get('statistics', {})
            r2_sub = stats.get('substrate', {}).get('R_squared', 0)
            r2_bio = stats.get('biomass', {}).get('R_squared', 0)
            loss = individual_losses.get(cond, np.nan)
            r2_data.append({
                'Condition': cond,
                'R² (Substrate)': f"{r2_sub:.4f}",
                'R² (Biomass)': f"{r2_bio:.4f}",
                'Loss Lᵢ': f"{loss:.4f}" if not np.isnan(loss) else '-'
            })
        st.dataframe(pd.DataFrame(r2_data), use_container_width=True)

    with col2:
        st.markdown("**NRMSE Values**")
        nrmse_data = []
        for cond in conditions:
            stats = per_cond[cond].get('statistics', {})
            nrmse_sub = stats.get('substrate', {}).get('NRMSE', 0)
            nrmse_bio = stats.get('biomass', {}).get('NRMSE', 0)
            nrmse_data.append({
                'Condition': cond,
                'NRMSE (Substrate)': f"{nrmse_sub:.4f}",
                'NRMSE (Biomass)': f"{nrmse_bio:.4f}",
            })
        st.dataframe(pd.DataFrame(nrmse_data), use_container_width=True)

    # Detailed per-condition statistics
    for cond in conditions:
        stats = per_cond[cond].get('statistics', {})
        combined = stats.get('combined', {})
        with st.expander(f"Detailed: {cond}"):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown("**Substrate**")
                sub_stats = stats.get('substrate', {})
                for key in ['R_squared', 'RMSE', 'NRMSE', 'MAE', 'SSE']:
                    val = sub_stats.get(key, 0)
                    st.text(f"{key}: {val:.4f}")
            with col_b:
                st.markdown("**Biomass**")
                bio_stats = stats.get('biomass', {})
                for key in ['R_squared', 'RMSE', 'NRMSE', 'MAE', 'SSE']:
                    val = bio_stats.get(key, 0)
                    st.text(f"{key}: {val:.4f}")
            with col_c:
                st.markdown("**Combined**")
                for key in ['R_squared', 'NRMSE', 'AIC', 'BIC']:
                    val = combined.get(key, 0)
                    st.text(f"{key}: {val:.4f}")
