"""
Configuration panel component for parameter input.
"""

import streamlit as st
import json
from pathlib import Path
from ..config import DEFAULT_BOUNDS, PARAMETER_INFO, CONFIG_DIR


def render_config_panel(model_type: str):
    """
    Render the configuration panel for substrate parameters.
    
    Args:
        model_type: Selected model type
    """
    st.header("⚙️ Configuration")
    
    # Substrate info
    col1, col2 = st.columns(2)
    
    with col1:
        substrate_name = st.text_input(
            "Substrate Name:",
            value="Glucose",
            help="Name of the substrate being studied"
        )
        st.session_state['substrate_name'] = substrate_name
    
    with col2:
        molecular_weight = st.number_input(
            "Molecular Weight (g/mol):",
            min_value=10.0,
            max_value=1000.0,
            value=180.16,
            help="Molecular weight for unit conversions"
        )
        st.session_state['molecular_weight'] = molecular_weight
    
    # Load existing config option
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Parameter Settings")
    with col2:
        existing_configs = list(CONFIG_DIR.glob("*.json")) if CONFIG_DIR.exists() else []
        if existing_configs:
            load_config = st.selectbox(
                "Load existing config:",
                ["-- New Config --"] + [f.stem for f in existing_configs],
                help="Load parameters from existing configuration"
            )
            if load_config != "-- New Config --":
                _load_config_file(CONFIG_DIR / f"{load_config}.json")
    
    # Determine which parameters to show based on model
    if model_type == "single_monod":
        param_names = ["qmax", "Ks", "Ki", "Y", "b_decay"]
    elif model_type == "single_monod_lag":
        param_names = ["qmax", "Ks", "Ki", "Y", "b_decay", "lag_time"]
    elif model_type == "dual_monod":
        param_names = ["qmax", "Ks", "Ki", "Y", "b_decay", "K_o2", "Y_o2"]
    else:  # dual_monod_lag
        param_names = ["qmax", "Ks", "Ki", "Y", "b_decay", "K_o2", "Y_o2", "lag_time"]
    
    # Parameter input tabs
    tab1, tab2 = st.tabs(["📊 Initial Guesses", "📏 Parameter Bounds"])
    
    with tab1:
        st.caption("Enter starting values for optimization")
        _render_initial_guesses(param_names)
    
    with tab2:
        st.caption("Define min/max bounds for each parameter")
        _render_bounds(param_names)
    
    # Oxygen settings (for dual models)
    if model_type in ["dual_monod", "dual_monod_lag"]:
        st.markdown("---")
        st.subheader("🫧 Oxygen Settings")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            o2_max = st.number_input("O₂ Saturation (mg/L):", value=8.0, min_value=0.1, max_value=20.0)
        with col2:
            o2_min = st.number_input("O₂ Minimum (mg/L):", value=0.1, min_value=0.0, max_value=2.0)
        with col3:
            reaeration = st.number_input("Reaeration Rate (day⁻¹):", value=15.0, min_value=0.0, max_value=100.0)
        
        st.session_state['oxygen_settings'] = {
            'o2_max': o2_max,
            'o2_min': o2_min,
            'reaeration_rate': reaeration
        }
    
    # Simulation settings
    st.markdown("---")
    st.subheader("⏱️ Simulation Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        t_final = st.number_input(
            "Simulation End Time (days):",
            value=0.5,
            min_value=0.1,
            max_value=30.0,
            step=0.1,
            help="Should match your experimental data time range"
        )
    with col2:
        num_points = st.number_input(
            "Number of Points:",
            value=1000,
            min_value=100,
            max_value=50000,
            step=500,
            help="Lower = faster fitting. 1000-5000 is usually enough."
        )
    
    st.session_state['simulation_settings'] = {
        't_final': t_final,
        'num_points': int(num_points)
    }
    
    return _build_config_dict()


def _render_initial_guesses(param_names: list):
    """Render initial guess inputs."""
    
    # Default values
    defaults = {
        "qmax": 5.0, "Ks": 100.0, "Ki": 10000.0, "Y": 0.35,
        "b_decay": 0.01, "K_o2": 0.15, "Y_o2": 1.0, "lag_time": 0.1
    }
    
    cols = st.columns(min(len(param_names), 4))
    
    initial_guesses = {}
    for i, param in enumerate(param_names):
        col_idx = i % 4
        info = PARAMETER_INFO.get(param, {})
        
        with cols[col_idx]:
            # Get stored value or default
            stored = st.session_state.get('initial_guesses', {}).get(param, defaults.get(param, 1.0))
            
            value = st.number_input(
                f"{param}",
                value=float(stored),
                format="%.4f",
                help=f"{info.get('description', '')} | Typical: {info.get('typical_range', 'varies')}",
                key=f"init_{param}"
            )
            initial_guesses[param] = value
    
    st.session_state['initial_guesses'] = initial_guesses


def _render_bounds(param_names: list):
    """Render parameter bounds inputs."""
    
    bounds = {}
    
    for param in param_names:
        info = PARAMETER_INFO.get(param, {})
        default_bound = DEFAULT_BOUNDS.get(param, [0.001, 100.0])
        
        stored_bounds = st.session_state.get('bounds', {}).get(param, default_bound)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"**{param}** ({info.get('unit', '')})")
        with col2:
            min_val = st.number_input(
                "Min",
                value=float(stored_bounds[0]),
                format="%.4f",
                key=f"bound_min_{param}",
                label_visibility="collapsed"
            )
        with col3:
            max_val = st.number_input(
                "Max",
                value=float(stored_bounds[1]),
                format="%.4f",
                key=f"bound_max_{param}",
                label_visibility="collapsed"
            )
        
        bounds[param] = [min_val, max_val]
    
    st.session_state['bounds'] = bounds


def _load_config_file(config_path: Path):
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Update session state
        st.session_state['substrate_name'] = config.get('substrate', {}).get('name', 'Unknown')
        st.session_state['molecular_weight'] = config.get('substrate', {}).get('molecular_weight', 180.0)
        st.session_state['initial_guesses'] = config.get('initial_guesses', {})
        st.session_state['bounds'] = config.get('bounds', {})
        st.session_state['oxygen_settings'] = config.get('oxygen', {})
        st.session_state['simulation_settings'] = config.get('simulation', {})
        
        st.success(f"Loaded configuration: {config_path.stem}")
    except Exception as e:
        st.error(f"Error loading config: {e}")


def _build_config_dict() -> dict:
    """Build configuration dictionary from session state."""
    
    model_type = st.session_state.get('model_type', 'dual_monod_lag')
    
    # Filter parameters based on model type
    if model_type == "single_monod":
        param_names = ["qmax", "Ks", "Ki", "Y", "b_decay"]
    elif model_type == "single_monod_lag":
        param_names = ["qmax", "Ks", "Ki", "Y", "b_decay", "lag_time"]
    elif model_type == "dual_monod":
        param_names = ["qmax", "Ks", "Ki", "Y", "b_decay", "K_o2", "Y_o2"]
    else:
        param_names = ["qmax", "Ks", "Ki", "Y", "b_decay", "K_o2", "Y_o2", "lag_time"]
    
    initial_guesses = {k: v for k, v in st.session_state.get('initial_guesses', {}).items() if k in param_names}
    bounds = {k: v for k, v in st.session_state.get('bounds', {}).items() if k in param_names}
    
    config = {
        "substrate": {
            "name": st.session_state.get('substrate_name', 'Unknown'),
            "molecular_weight": st.session_state.get('molecular_weight', 180.0),
            "unit": "mg/L"
        },
        "initial_guesses": initial_guesses,
        "bounds": bounds,
        "oxygen": st.session_state.get('oxygen_settings', {
            "o2_max": 8.0,
            "o2_min": 0.1,
            "reaeration_rate": 15.0,
            "o2_range": 8.0
        }),
        "simulation": st.session_state.get('simulation_settings', {
            "t_final": 2.0,
            "num_points": 10000,
            "time_unit": "days"
        })
    }
    
    return config
