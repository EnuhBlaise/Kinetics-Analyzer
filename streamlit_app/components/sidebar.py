"""
Sidebar component for navigation and model selection.
"""

import streamlit as st
from ..config import MODEL_OPTIONS, APP_TITLE


def render_sidebar():
    """Render the sidebar with navigation and settings."""
    
    with st.sidebar:
        st.title(f"🧬 {APP_TITLE}")
        st.markdown("---")
        
        # Model selection
        st.subheader("Model Selection")
        model_choice = st.selectbox(
            "Choose kinetic model:",
            options=list(MODEL_OPTIONS.keys()),
            index=4,  # Default to Dual Monod + Lag
            help="Select the kinetic model complexity"
        )

        # Store in session state
        st.session_state['model_type'] = MODEL_OPTIONS[model_choice]
        st.session_state['model_name'] = model_choice

        # Model description
        model_descriptions = {
            "Single Monod": "📝 **4 parameters**: qmax, Ks, Y, b_decay\n\nBasic Monod without substrate inhibition",
            "Single Monod (Haldane)": "📝 **5 parameters**: qmax, Ks, Ki, Y, b_decay\n\nHaldane model with substrate inhibition",
            "Single Monod + Lag": "📝 **5 parameters**: qmax, Ks, Y, b_decay, lag_time\n\nBasic Monod with lag phase, no oxygen",
            "Single Monod + Lag (Haldane)": "📝 **6 parameters**: + Ki\n\nHaldane with lag phase, no oxygen",
            "Dual Monod": "📝 **6 parameters**: + K_O2, Y_O2\n\nAerobic systems with oxygen dynamics",
            "Dual Monod (Haldane)": "📝 **7 parameters**: + Ki\n\nOxygen dynamics with substrate inhibition",
            "Dual Monod + Lag": "📝 **7 parameters**: + lag_time\n\nOxygen dynamics with lag phase",
            "Dual Monod + Lag (Haldane)": "📝 **8 parameters**: + Ki\n\nFull model with lag phase and substrate inhibition",
        }
        st.info(model_descriptions.get(model_choice, ""))
        
        st.markdown("---")
        
        # Optimization settings
        st.subheader("Optimization Settings")

        optimizer = st.selectbox(
            "Optimizer:",
            ["L-BFGS-B", "Differential Evolution"],
            help="L-BFGS-B is faster; DE is more robust for difficult problems"
        )
        st.session_state['optimizer'] = optimizer

        fit_method = st.radio(
            "Fitting method:",
            ["Global (all conditions)", "Individual (per condition)", "Robust (weighted + bootstrap)"],
            help="Global fits one parameter set; Individual fits each condition; Robust adds weighting, two-stage init, and bootstrap CIs"
        )
        if "Robust" in fit_method:
            st.session_state['fit_method'] = "robust"
        elif "Global" in fit_method:
            st.session_state['fit_method'] = "global"
        else:
            st.session_state['fit_method'] = "individual"

        # Robust fitting options (shown when robust is selected)
        if st.session_state.get('fit_method') == 'robust':
            st.markdown("**Robust Fitting Options**")

            weighting = st.selectbox(
                "Weighting strategy:",
                ["max_value", "uniform", "variance", "range"],
                help="How to weight conditions with different data scales. 'max_value' is recommended."
            )
            st.session_state['weighting_strategy'] = weighting

            use_two_stage = st.checkbox(
                "Two-stage initialization",
                value=True,
                help="Use algebraic Monod fitting to generate data-informed initial guesses"
            )
            st.session_state['use_two_stage'] = use_two_stage

            bootstrap_iters = st.number_input(
                "Bootstrap iterations:",
                min_value=0,
                max_value=2000,
                value=500,
                step=100,
                help="Number of bootstrap iterations for confidence intervals (0 to disable)"
            )
            st.session_state['bootstrap_iterations'] = bootstrap_iters

            if bootstrap_iters > 0:
                bootstrap_workers = st.number_input(
                    "Bootstrap workers:",
                    min_value=1,
                    max_value=16,
                    value=4,
                    help="Number of parallel workers for bootstrap"
                )
                st.session_state['bootstrap_workers'] = bootstrap_workers

        st.markdown("---")
        
        # AI Settings
        st.subheader("🤖 AI Analysis")
        
        enable_ai = st.checkbox(
            "Enable AI interpretation",
            value=True,
            help="Use LLM to explain results and provide recommendations"
        )
        st.session_state['enable_ai'] = enable_ai
        
        if enable_ai:
            api_token = st.text_input(
                "Hugging Face API Token:",
                type="password",
                help="Enter your HF token for AI features (optional - fallback analysis available)"
            )
            st.session_state['hf_token'] = api_token
            
            if not api_token:
                st.caption("💡 Without API token, rule-based analysis will be used")
        
        st.markdown("---")
        
        # Help section
        with st.expander("ℹ️ Help"):
            st.markdown("""
            **Quick Start:**
            1. Configure substrate parameters
            2. Upload experimental data (CSV)
            3. Click 'Run Fitting'
            4. Review results and AI analysis
            
            **Data Format:**
            - CSV with time column
            - Columns: `{conc}_Substrate (mg/L)`
            - Columns: `{conc}_Biomass (OD or mg/L)`
            """)
        
        # Version info
        st.markdown("---")
        st.caption("v2.0.0 | Kinetic Parameter Estimation")
    
    return st.session_state.get('model_type', 'dual_monod_lag')
