"""
Main Streamlit application for Kinetic Parameter Estimation.

Run with: streamlit run streamlit_app/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from io import StringIO
import tempfile

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import components
from streamlit_app.config import APP_TITLE, APP_ICON, APP_LAYOUT
from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.components.config_panel import render_config_panel
from streamlit_app.components.data_upload import render_data_upload
from streamlit_app.components.results_display import render_results
from streamlit_app.components.plots import render_plots
from streamlit_app.llm_integration import get_llm_analyzer

# Import workflow runner (handles core module imports)
try:
    from streamlit_app.workflow_runner import run_fitting_workflow, validate_data_format, TimeoutError
    CORE_AVAILABLE = True
except ImportError as e:
    CORE_AVAILABLE = False
    IMPORT_ERROR = str(e)


def main():
    """Main application entry point."""
    
    # Page configuration
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout=APP_LAYOUT,
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .stAlert {
        padding: 0.5rem 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
    }
    .ai-box {
        background-color: #f0f7ff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Check if core modules are available
    if not CORE_AVAILABLE:
        st.error(f"Core modules not available: {IMPORT_ERROR}")
        st.info("Please install the main project: `pip install -e .`")
        return
    
    # Render sidebar and get model type
    model_type = render_sidebar()
    
    # Main content
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.markdown("Fit Monod kinetic parameters to microbial growth data with AI-powered analysis.")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "⚙️ Configuration",
        "📁 Data",
        "📊 Results",
        "🤖 AI Analysis"
    ])
    
    # Tab 1: Configuration
    with tab1:
        config = render_config_panel(model_type)
        st.session_state['config'] = config
    
    # Tab 2: Data Upload
    with tab2:
        df = render_data_upload()
    
    # Run fitting section
    st.markdown("---")
    
    # Timeout setting
    col1, col2, col3 = st.columns([1, 1, 1])
    with col3:
        timeout = st.number_input(
            "Timeout (sec):",
            min_value=10,
            max_value=300,
            value=60,
            help="Maximum fitting time"
        )
        st.session_state['fitting_timeout'] = timeout
    
    with col2:
        run_button = st.button(
            "🚀 Run Parameter Fitting",
            type="primary",
            use_container_width=True,
            disabled=df is None
        )
    
    with col1:
        st.caption("💡 Reduce timeout or use simpler model if too slow")
    
    # Execute fitting
    if run_button and df is not None:
        progress_bar = st.progress(0, text="Initializing...")
        status = st.empty()
        
        try:
            progress_bar.progress(10, text="Loading data...")
            status.info("🔄 Running parameter fitting... Please wait.")
            
            progress_bar.progress(30, text="Optimizing parameters...")
            results = run_fitting(df, config, model_type)
            
            progress_bar.progress(100, text="Complete!")
            
            if results and results.get('success'):
                st.session_state['results'] = results
                st.session_state['ai_analysis'] = None
                status.success("✅ Fitting complete! Check the Results tab.")
            elif results and results.get('parameters'):
                st.session_state['results'] = results
                st.session_state['ai_analysis'] = None
                status.warning("⚠️ Fitting completed. Check results quality.")
            else:
                status.error("❌ Fitting failed. Check error messages above.")
                
        except Exception as e:
            progress_bar.empty()
            status.error(f"Error: {e}")
    
    # Tab 3: Results
    with tab3:
        results = st.session_state.get('results', {})
        render_results(results)
        
        # Plots
        if results and results.get('predictions') is not None:
            predictions = results.get('predictions')
            conditions = results.get('conditions', [])
            render_plots(predictions, df, conditions)
    
    # Tab 4: AI Analysis
    with tab4:
        render_ai_analysis()

def run_fitting(df: pd.DataFrame, config: dict, model_type: str) -> dict:
    """
    Run the parameter fitting workflow.
    
    Args:
        df: Experimental data DataFrame
        config: Configuration dictionary
        model_type: Type of model to use
        
    Returns:
        Results dictionary
    """
    import streamlit as st
    
    timeout = st.session_state.get('fitting_timeout', 120)
    
    try:
        results = run_fitting_workflow(
            experimental_df=df,
            config_dict=config,
            model_type=model_type,
            fit_method=st.session_state.get('fit_method', 'global'),
            optimizer=st.session_state.get('optimizer', 'L-BFGS-B'),
            verbose=False,
            timeout=timeout,
            weighting_strategy=st.session_state.get('weighting_strategy', 'max_value'),
            use_two_stage=st.session_state.get('use_two_stage', True),
            bootstrap_iterations=st.session_state.get('bootstrap_iterations', 500),
            bootstrap_workers=st.session_state.get('bootstrap_workers', 4)
        )
        return results
        
    except TimeoutError:
        st.error(f"⏰ Fitting timed out after {timeout} seconds. Try:")
        st.markdown("""
        - Reducing parameter bounds
        - Using a simpler model (Single Monod)
        - Increasing timeout in sidebar
        """)
        return {'success': False, 'message': 'Timeout'}
        
    except Exception as e:
        st.error(f"Fitting error: {e}")
        import traceback
        with st.expander("Show error details"):
            st.code(traceback.format_exc())
        return {'success': False, 'message': str(e)}


def render_ai_analysis():
    """Render the AI analysis section."""
    
    st.header("🤖 AI-Powered Analysis")
    
    results = st.session_state.get('results', {})
    
    if not results:
        st.info("Run parameter fitting first to get AI analysis of results.")
        return
    
    # Check if AI is enabled
    enable_ai = st.session_state.get('enable_ai', True)
    
    if not enable_ai:
        st.warning("AI analysis is disabled. Enable it in the sidebar settings.")
        return
    
    # Ask user if they want analysis
    st.markdown("""
    Get an AI-powered interpretation of your fitting results, including:
    - 📊 Assessment of fit quality (R², RMSE analysis)
    - 🔬 Parameter interpretation (biological meaning)
    - 📈 Analysis of model predictions vs experimental data
    - 🔍 Residual analysis for each experimental condition
    - 📉 Uncertainty assessment and confidence intervals
    - 💡 Recommendations for improvement
    """)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        analyze_button = st.button(
            "🔍 Generate AI Analysis",
            type="primary",
            use_container_width=True
        )
    
    # Run analysis
    if analyze_button:
        with st.spinner("Generating AI analysis (analyzing all results data)..."):
            analyzer = get_llm_analyzer(
                api_token=st.session_state.get('hf_token', '')
            )
            
            # Get experimental data from session state
            exp_df = st.session_state.get('experimental_data', None)
            
            # Get predictions and conditions from results
            predictions_df = results.get('predictions', None)
            conditions = results.get('conditions', [])
            
            analysis = analyzer.analyze_results(
                parameters=results.get('parameters', {}),
                statistics=results.get('statistics', {}),
                confidence_intervals=results.get('confidence_intervals', {}),
                model_type=results.get('model_type', 'unknown'),
                substrate_name=st.session_state.get('substrate_name', 'Unknown'),
                predictions_df=predictions_df,
                experimental_df=exp_df,
                conditions=conditions
            )
            
            st.session_state['ai_analysis'] = analysis
    
    # Display analysis
    if st.session_state.get('ai_analysis'):
        st.markdown("---")
        st.subheader("Analysis Results")
        
        # Analysis box with nice styling
        st.markdown(
            f"""
            <div class="ai-box">
            {st.session_state['ai_analysis'].replace(chr(10), '<br>')}
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Also show as markdown for proper formatting
        with st.expander("📝 View as Markdown", expanded=False):
            st.markdown(st.session_state['ai_analysis'])
        
        # Feedback and download
        st.markdown("---")
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            if st.button("👍 Helpful", use_container_width=True):
                st.success("Thanks for your feedback!")
        with col2:
            if st.button("👎 Not helpful", use_container_width=True):
                st.info("We'll try to improve!")
        with col3:
            st.download_button(
                label="📥 Download",
                data=st.session_state['ai_analysis'],
                file_name="ai_analysis.md",
                mime="text/markdown",
                use_container_width=True
            )
        with col4:
            if st.button("🔄 Regenerate", use_container_width=True):
                st.session_state['ai_analysis'] = None
                st.rerun()


if __name__ == "__main__":
    main()
