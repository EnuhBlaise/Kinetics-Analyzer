# Streamlit GUI for Kinetic Parameter Estimation

A web-based graphical interface for fitting Monod kinetic parameters to microbial growth data.

## Features

- **Interactive Configuration**: Input substrate parameters, initial guesses, and bounds through a user-friendly interface
- **Data Upload**: Upload CSV files or use example datasets
- **Multiple Fitting Modes**: Global, individual condition, and robust fitting (with weighting, two-stage init, bootstrap CIs)
- **Robust Fitting Options**: Condition weighting strategies, two-stage initialization, bootstrap confidence intervals
- **Interactive Plots**: Visualize fits with Plotly charts including actual residual analysis
- **AI Analysis**: Get LLM-powered interpretation of results and recommendations

## Quick Start

### 1. Install Dependencies

```bash
# From the project root
pip install -r streamlit_app/requirements.txt
```

### 2. Run the Application

```bash
# From the project root
streamlit run streamlit_app/app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage Guide

### Configuration Tab
1. Enter substrate name and molecular weight
2. Set initial parameter guesses
3. Define parameter bounds
4. Configure oxygen settings (for dual models)

### Data Tab
1. Upload your CSV file or select example data
2. Review the data preview
3. Check validation messages

### Fitting Options (Sidebar)
1. **Model selection**: Single Monod, Dual Monod, or Dual Monod + Lag
2. **Fitting method**:
   - **Global**: Fits one parameter set to all conditions
   - **Individual**: Fits each condition separately
   - **Robust**: Weighted fitting + two-stage initialization + bootstrap CIs
3. **Robust options** (when Robust is selected):
   - Weighting strategy (max_value recommended)
   - Two-stage initialization toggle
   - Bootstrap iteration count and worker count

### Results Tab
1. Click "Run Parameter Fitting"
2. View fitted parameters with confidence intervals (Hessian-based or bootstrap)
3. Examine fit statistics (separate substrate/biomass R², RMSE, NRMSE)
4. View actual residual analysis plots
5. Download results as JSON

### AI Analysis Tab
1. Enable AI analysis in sidebar
2. (Optional) Enter Hugging Face API token for enhanced analysis
3. Click "Analyze Results"
4. Review recommendations for improving fits

## Data Format

Your CSV file should have columns in this format:

```csv
Time (days),5mM_Glucose (mg/L),5mM_Biomass (OD),10mM_Glucose (mg/L),10mM_Biomass (OD)
0,900,0.01,1800,0.01
0.1,850,0.02,1750,0.02
...
```

## AI Integration

The app uses Hugging Face's Inference API for AI analysis. Features:

- **With API token**: Full LLM-powered analysis using Mistral-7B
- **Without token**: Rule-based fallback analysis (still useful!)

To get a Hugging Face token:
1. Create account at [huggingface.co](https://huggingface.co)
2. Go to Settings → Access Tokens
3. Create a new token with "read" permissions

## Environment Variables

Optional environment variables:

```bash
export HF_API_TOKEN="your_huggingface_token"
export HF_MODEL_ID="mistralai/Mistral-7B-Instruct-v0.2"
```

## Project Structure

```
streamlit_app/
├── app.py                 # Main application
├── config.py              # App configuration and constants
├── workflow_runner.py     # Fitting workflow integration (standard + robust)
├── llm_integration.py     # Hugging Face LLM integration
├── requirements.txt       # Dependencies
├── README.md             # This file
└── components/
    ├── sidebar.py         # Sidebar with model, optimizer, and robust fitting options
    ├── config_panel.py    # Parameter configuration
    ├── data_upload.py     # Data upload handling
    ├── results_display.py # Results visualization (separate substrate/biomass metrics)
    └── plots.py           # Interactive plots with actual residual analysis
```

## Troubleshooting

### "Core modules not available"
Make sure you've installed the main project:
```bash
pip install -e .
```

### Slow fitting
- Try reducing `num_points` in simulation settings
- Use L-BFGS-B optimizer (faster than Differential Evolution)

### Poor AI analysis
- Enter your Hugging Face API token for better results
- The fallback analysis is rule-based but still helpful

## License

Part of the Kinetic Parameter Estimation Software project.
