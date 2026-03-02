"""
Configuration settings for the Streamlit application.
"""

import os
from pathlib import Path

# App settings
APP_TITLE = "Kinetic Parameter Estimation"
APP_ICON = "🧬"
APP_LAYOUT = "wide"

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config" / "substrates"
DATA_DIR = PROJECT_ROOT / "data" / "example"
RESULTS_DIR = PROJECT_ROOT / "results"

# Default parameter bounds
DEFAULT_BOUNDS = {
    "qmax": [0.1, 50.0],
    "Ks": [1.0, 2000.0],
    "Ki": [100.0, 50000.0],
    "Y": [0.05, 1.0],
    "b_decay": [0.001, 0.5],
    "K_o2": [0.01, 2.0],
    "Y_o2": [0.1, 3.0],
    "lag_time": [0.0, 5.0]
}

# App version
APP_VERSION = "2.0.0"

# Model options
MODEL_OPTIONS = {
    "Single Monod": "single_monod",
    "Single Monod (Haldane)": "single_haldane",
    "Single Monod + Lag": "single_monod_lag",
    "Single Monod + Lag (Haldane)": "single_haldane_lag",
    "Dual Monod": "dual_monod",
    "Dual Monod (Haldane)": "dual_haldane",
    "Dual Monod + Lag": "dual_monod_lag",
    "Dual Monod + Lag (Haldane)": "dual_haldane_lag",
}

# Weighting strategy options
WEIGHTING_OPTIONS = {
    "Max Value (recommended)": "max_value",
    "Uniform": "uniform",
    "Variance": "variance",
    "Range": "range"
}

# Hugging Face settings
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")

# Parameter descriptions for UI
PARAMETER_INFO = {
    "qmax": {
        "name": "Maximum Uptake Rate (qmax)",
        "unit": "mg substrate/(mg cells·day)",
        "description": "Maximum specific substrate uptake rate",
        "typical_range": "1-50"
    },
    "Ks": {
        "name": "Half-Saturation Constant (Ks)",
        "unit": "mg/L",
        "description": "Substrate concentration at half-maximum uptake rate",
        "typical_range": "10-500"
    },
    "Ki": {
        "name": "Inhibition Constant (Ki)",
        "unit": "mg/L",
        "description": "Substrate concentration causing 50% inhibition",
        "typical_range": "1000-50000"
    },
    "Y": {
        "name": "Yield Coefficient (Y)",
        "unit": "mg cells/mg substrate",
        "description": "Biomass produced per substrate consumed",
        "typical_range": "0.1-0.8"
    },
    "b_decay": {
        "name": "Decay Coefficient (b)",
        "unit": "day⁻¹",
        "description": "Rate of biomass decay/maintenance",
        "typical_range": "0.001-0.1"
    },
    "K_o2": {
        "name": "Oxygen Half-Saturation (K_O2)",
        "unit": "mg O₂/L",
        "description": "Oxygen concentration at half-maximum uptake",
        "typical_range": "0.1-1.0"
    },
    "Y_o2": {
        "name": "Oxygen Yield (Y_O2)",
        "unit": "mg O₂/mg substrate",
        "description": "Oxygen consumed per substrate consumed",
        "typical_range": "0.5-2.0"
    },
    "lag_time": {
        "name": "Lag Phase Duration",
        "unit": "days",
        "description": "Time before exponential growth begins",
        "typical_range": "0-2"
    }
}
