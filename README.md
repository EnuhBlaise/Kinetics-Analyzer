# User Guide

## Overview
<img width="1601" height="888" alt="Screenshot 2026-03-19 at 1 46 18 AM" src="https://github.com/user-attachments/assets/10b28e5a-0266-447f-b90a-40ddb79a860d" />

**Fig**: UI interface with Streamlit.

The Kinetic Parameter Estimation Software is a tool for fitting Monod kinetic parameters to microbial growth data. It supports eight model variants organized into four families, each available with or without Haldane substrate inhibition (Ki):

| Model | Parameters | Description |
|-------|-----------|-------------|
| **Single Monod** | 4: qmax, Ks, Y, b_decay | Basic Monod without substrate inhibition |
| **Single Monod (Haldane)** | 5: + Ki | Haldane model with substrate inhibition |
| **Single Monod + Lag** | 5: + lag_time | Lag phase, no oxygen dynamics |
| **Single Monod + Lag (Haldane)** | 6: + Ki | Lag phase with substrate inhibition |
| **Dual Monod** | 6: + K_o2, Y_o2 | Oxygen dynamics, no inhibition |
| **Dual Monod (Haldane)** | 7: + Ki | Oxygen dynamics with substrate inhibition |
| **Dual Monod + Lag** | 7: + lag_time | Oxygen dynamics with lag phase |
| **Dual Monod + Lag (Haldane)** | 8: + Ki | Full model with lag phase and inhibition |

## Installation

### Requirements
- Python 3.8 or higher
- pip package manager

### Installation Steps

```bash
# Clone or download the repository
cd Kinetics-Analyzer

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

### 1. Prepare Your Data

Create a CSV file with your experimental data. Required format:

```csv
Time (days),5mM_Glucose (mg/L),5mM_Biomass (mgCells/L),10mM_Glucose (mg/L),10mM_Biomass (mgCells/L)
0,900,1,1800,1
1,700,15,1500,20
2,300,45,800,60
...
```

Column naming convention:
- Time column: `Time (days)` or `Time (hours)`
- Substrate: `{concentration}_{Substrate} (mg/L)` e.g., `5mM_Glucose (mg/L)`
- Biomass: `{concentration}_Biomass (mgCells/L)` or `{concentration}_Biomass (OD)`

### 2. Create or Modify Configuration

Copy an existing config file and modify for your substrate:

```bash
cp config/substrates/glucose.json config/substrates/my_substrate.json
```

Edit the JSON file to update:
- Substrate name and molecular weight
- Initial parameter guesses
- Parameter bounds

### 3. Run Parameter Fitting


```bash
# Fit parameters using the dual Monod + lag model (with Ki / Haldane)
python scripts/fit_parameters.py \
    --config config/substrates/my_substrate.json \
    --data data/my_experiment.csv \
    --workflow dual_monod_lag

# Fit with the simple single Monod model (with Ki / Haldane)
python scripts/fit_parameters.py \
    --config config/substrates/my_substrate.json \
    --data data/my_experiment.csv \
    --workflow single_monod

# Haldane aliases are also accepted
python scripts/fit_parameters.py \
    --config config/substrates/my_substrate.json \
    --data data/my_experiment.csv \
    --workflow dual_haldane_lag
```

### 4. Fit Individual Conditions

For detailed analysis of each concentration condition with confidence intervals:

```bash
# Basic Monod (4 params, no Ki) — displays as "Single Monod"
python scripts/fit_individual.py \
    --config config/substrates/glucose.json \
    --data data/example/Experimental_data_glucose.csv \
    --model single_monod \
    --verbose

# With Haldane substrate inhibition (5 params, with Ki) — displays as "Single Monod (Haldane)"
python scripts/fit_individual.py \
    --config config/substrates/glucose.json \
    --data data/example/Experimental_data_glucose.csv \
    --model single_haldane \
    --verbose

# Full model with lag and inhibition (8 params) — displays as "Dual Monod + Lag (Haldane)"
python scripts/fit_individual.py \
    --config config/substrates/glucose.json \
    --data data/example/Experimental_data_glucose.csv \
    --model dual_haldane_lag \
    --verbose
```

All eight model variants are available: `single_monod`, `single_haldane`, `single_monod_lag`, `single_haldane_lag`, `dual_monod`, `dual_haldane`, `dual_monod_lag`, `dual_haldane_lag`.

To speed up fitting with MCMC confidence intervals, use `--workers` for parallel per-condition fitting:

```bash
python scripts/fit_individual.py \
    --config config/substrates/glucose.json \
    --data data/example/Experimental_data_glucose.csv \
    --model single_monod --ci-method mcmc --workers 4
```

This generates:
- **substrate_summary.png**: Substrate fit curves with all conditions overlaid
- **biomass_summary.png**: Biomass fit curves with all conditions overlaid
- **parameter_comparison.png**: Parameter values with error bars across conditions
- **residual_diagnostics.png**: Residual analysis for model validation
- **confidence_intervals.png**: 95% CIs for each parameter by condition
- **goodness_of_fit.png**: R² and NRMSE summary heatmap
- **individual_condition_results.csv**: Tabular results
- **individual_condition_results.json**: Full results with CIs
- **results_report.pdf**: PDF report with all results

### 5. Run All Models Across All Substrates

To systematically compare all 8 model variants across all 6 substrates (48 fits total):

```bash
# Run all 48 fits in parallel and build the master comparison CSV
python scripts/run_all_models.py

# Run a subset of substrates/models
python scripts/run_all_models.py --substrates glucose xylose --models single_monod single_haldane

# Limit parallelism or run sequentially
python scripts/run_all_models.py --workers 4
python scripts/run_all_models.py --workers 1

# Use MCMC confidence intervals (slower but more robust)
python scripts/run_all_models.py --ci-method mcmc

# Skip plot generation for faster batch runs
python scripts/run_all_models.py --no-plots
```

This produces `output/master_results.csv` with one row per (substrate, model) combination containing global parameters, 95% CIs, Total Error, R², AIC, and AIC weights.

### 6. Generate Publication Analysis Figures

After running batch fits, generate 6 publication-quality comparison figures:

```bash
# Generate all figures from the master CSV and per-run result JSONs
python scripts/analyze_results.py

# Specify custom paths
python scripts/analyze_results.py \
    --master-csv output/master_results.csv \
    --results-dir results \
    --output-dir output/figures

# Control output formats and resolution
python scripts/analyze_results.py --formats png pdf --dpi 300
```

This produces 14 files (7 figures x PNG + PDF) in `output/figures/`:

| Figure | File | Description |
|--------|------|-------------|
| A | `figure_A_r2_heatmap` | R² heatmap (substrate R² and biomass R² side by side) |
| B | `figure_B_delta_aic` | ΔAIC grouped bar chart (model comparison per substrate) |
| C | `figure_C_cv_dotplot` | Parameter identifiability (CV%) dot plot for best models |
| D | `figure_D_residuals` | Residual diagnostics panels for best model per substrate |
| E | `figure_E_total_error_bars` | Total Error grouped bar chart (log scale) |
| F | `figure_F_total_error_heatmap` | Total Error heatmap (log₁₀ scale) |
| G | `figure_G_parameter_dotplots` | Parameter values across models per substrate (★ = best AIC) |

### 7. Compare Models

```bash
# Run all three models and compare performance
python scripts/compare_models.py \
    --config config/substrates/my_substrate.json \
    --data data/my_experiment.csv \
    --output results/comparison
```

### 8. Robust Parameter Fitting (with Bootstrap CIs)

For publication-quality results with bootstrap confidence intervals, condition weighting, and two-stage initialization:

```bash
# Full robust fit with bootstrap uncertainty quantification
python scripts/robust_fit.py \
    --config config/substrates/xylose.json \
    --data data/example/data.csv \
    --workflow dual_monod_lag \
    --weighting max_value \
    --two-stage \
    --bootstrap 500 \
    --workers 4 \
    --output results/xylose/

# Quick robust fit without bootstrap
python scripts/robust_fit.py \
    --config config/substrates/glucose.json \
    --data data/example/Experimental_data_glucose.csv \
    --workflow single_monod \
    --bootstrap 0

# Reproducible fit with seed
python scripts/robust_fit.py \
    --config config/substrates/xylose.json \
    --data data/example/data.csv \
    --seed 42 \
    --bootstrap 1000
```

This generates:
- **{substrate}_robust_params.json**: Parameters with 95% bootstrap confidence intervals
- **{substrate}_bootstrap_distribution.csv**: Full bootstrap parameter distribution
- **{substrate}_robust_summary.txt**: Publication-ready summary text

## Command-Line Interface

### fit_parameters.py

Fits kinetic parameters globally across all conditions. These workflows always include Ki (Haldane inhibition).

```
Usage: python scripts/fit_parameters.py [OPTIONS]

Options:
  -c, --config PATH    Substrate configuration JSON file (required)
  -d, --data PATH      Experimental data CSV file (required)
  -w, --workflow TYPE  Model type (default: dual_monod_lag)
                       single_monod / single_haldane   — 5 params (with Ki)
                       single_monod_lag / single_haldane_lag — 6 params (with Ki)
                       dual_monod / dual_haldane       — 7 params (with Ki)
                       dual_monod_lag / dual_haldane_lag — 8 params (with Ki)
  -o, --output DIR     Output directory (default: results)
  -m, --method TYPE    Fitting method: global or individual
  --optimizer TYPE     Optimization algorithm: L-BFGS-B or differential_evolution
  -v, --verbose        Enable verbose output
  --no-plots           Skip plot generation
```

### fit_individual.py

Fits parameters separately for each experimental condition with detailed statistics. Supports parallel execution of per-condition fits.

```
Usage: python scripts/fit_individual.py [OPTIONS]

Core options:
  -c, --config PATH    Substrate configuration JSON file (required)
  -d, --data PATH      Experimental data CSV file (required)
  -m, --model TYPE     Model type (default: single_monod):
                        single_monod       — Single Monod (4 params: qmax, Ks, Y, b_decay)
                        single_haldane     — Single Monod + Haldane (5 params: + Ki)
                        single_monod_lag   — Single Monod + Lag (5 params: + lag_time)
                        single_haldane_lag — Single Monod + Lag + Haldane (6 params: + Ki)
                        dual_monod         — Dual Monod (6 params: + K_o2, Y_o2)
                        dual_haldane       — Dual Monod + Haldane (7 params: + Ki)
                        dual_monod_lag     — Dual Monod + Lag (7 params: + lag_time)
                        dual_haldane_lag   — Dual Monod + Lag + Haldane (8 params: + Ki)
  -o, --output DIR     Output directory (default: results)
  --optimizer TYPE     Optimization algorithm: L-BFGS-B or differential_evolution
  --workers N          Parallel workers for per-condition fitting (default: 1 = sequential).
                       Use 0 for auto-detect (CPU count - 1).
  -v, --verbose        Enable verbose output
  --no-plots           Skip plot generation

Confidence intervals:
  --ci-method TYPE           CI method: hessian (fast), hessian_log, or mcmc (default: hessian)
  --ci-level FLOAT           Confidence level (default: 0.95)
  --mcmc-samples N           Number of MCMC posterior samples (default: 4000)
  --mcmc-burn-in N           MCMC burn-in iterations (default: 1000)
  --mcmc-step-scale FLOAT    Proposal step scale as fraction of parameter range (default: 0.05)
  --mcmc-seed N              Random seed for MCMC reproducibility

Global optimisation:
  --global-guess TYPE  Strategy for the global optimisation initial guess (default: median):
                        median  — median of per-condition fits (robust to outliers)
                        best_r2 — parameters from the condition with the highest mean R²(S,X)

Scaling options (all default OFF — enable selectively):
  --normalize-objective    Weight S and X residuals by 1/range so both contribute equally
  --mcmc-adaptive          Use Hessian-informed adaptive MCMC proposals
                           (only affects --ci-method mcmc)
  --scale-diagnostics      Run diagnostic re-optimisations in [0,1]-normalised parameter space

Output includes:
  - Parameter estimates with 95% confidence intervals
  - Separate R², RMSE, NRMSE for substrate and biomass
  - Residual diagnostics (mean, std, autocorrelation)
  - Coefficient of variation (CV) across conditions
  - Global parameter recommendations via two-stage optimization
```

**Parallel fitting:** When using MCMC confidence intervals (slow), parallelism across conditions provides significant speedup:

```bash
# Sequential (default) — conditions fitted one at a time
python scripts/fit_individual.py -c config/substrates/glucose.json \
    -d data/example/Experimental_data_glucose.csv -m single_monod --ci-method mcmc

# Parallel — fit all 4 conditions simultaneously
python scripts/fit_individual.py -c config/substrates/glucose.json \
    -d data/example/Experimental_data_glucose.csv -m single_monod --ci-method mcmc --workers 4

# Auto-detect worker count
python scripts/fit_individual.py -c config/substrates/glucose.json \
    -d data/example/Experimental_data_glucose.csv -m single_monod --ci-method mcmc --workers 0
```

**Guess-free fitting with `differential_evolution`:** The L-BFGS-B optimizer (default) is a local search — it starts from the initial guesses in your config JSON and can get stuck in local minima. If you want results that are **independent of initial parameter guesses**, use `differential_evolution`, which samples the entire bounded parameter space via Latin hypercube:

```bash
# Initial guesses are ignored — only bounds matter
python scripts/fit_individual.py \
    -c config/substrates/glucose.json \
    -d data/example/Experimental_data_glucose.csv \
    -m dual_haldane_lag --optimizer differential_evolution -v
```

This is especially useful when running many model families on the same substrate, since the same config file works for all models without adjusting guesses.

**Global optimisation initial guess strategy:** After fitting each condition individually, the workflow runs a global optimisation across all conditions simultaneously. The `--global-guess` flag controls how the starting point is chosen:

```bash
# Default: median of individual fits (robust to outlier conditions)
python scripts/fit_individual.py -c ... -d ... -m dual_monod_lag --global-guess median

# Alternative: use parameters from the best-fitting condition (highest mean R²)
python scripts/fit_individual.py -c ... -d ... -m dual_monod_lag --global-guess best_r2
```

Use `median` when most conditions fit well but one or two are noisy. Use `best_r2` when conditions have very different fit quality and you want to start from the most informative one.

**Scaling options:** When substrate and biomass data have very different magnitudes (e.g., S ∈ [0, 2000] mg/L vs X ∈ [0, 100] mgCells/L), the objective function can be dominated by substrate residuals. Three optional flags address parameter and objective scaling:

```bash
# Normalise objective so S and X contribute equally
python scripts/fit_individual.py -c ... -d ... --normalize-objective

# Adaptive MCMC proposals informed by Hessian covariance (better mixing)
python scripts/fit_individual.py -c ... -d ... --ci-method mcmc --mcmc-adaptive

# All scaling options together
python scripts/fit_individual.py -c ... -d ... \
    --normalize-objective --ci-method mcmc --mcmc-adaptive --scale-diagnostics
```

### run_simulation.py

Run simulations with fitted parameters.

```
Usage: python scripts/run_simulation.py [OPTIONS]

Options:
  -p, --params PATH     Fitted parameters JSON file (required)
  -c, --config PATH     Substrate configuration file (required)
  -o, --output DIR      Output directory
  --conditions LIST     Comma-separated concentrations in mM (e.g., "5,10,15,20")
  --t-final FLOAT       Simulation end time in days
  --initial-biomass FLOAT  Initial biomass concentration
```

### compare_models.py

Compare all three model types on the same dataset.

```
Usage: python scripts/compare_models.py [OPTIONS]

Options:
  -c, --config PATH    Substrate configuration file (required)
  -d, --data PATH      Experimental data file (required)
  -o, --output DIR     Output directory for comparison results
  -v, --verbose        Enable verbose output
```

### run_diagnostics.py

Investigates the objective function landscape around the optimum to reveal identifiability issues, parameter correlations, multi-modality, and conditioning problems.

```
Usage: python scripts/run_diagnostics.py [OPTIONS]

Options:
  -c, --config PATH    Substrate configuration JSON file (required)
  -d, --data PATH      Experimental data CSV file (required)
  -m, --model TYPE     Model type (default: single_haldane)
  -o, --output DIR     Output directory (default: results/<Substrate>/diagnostics)
  --condition LABEL    Specific condition to analyse (default: first condition)
  --n-starts N         Number of multi-start trials (default: 50)
  --profile-points N   Grid points per parameter profile (default: 30)
  --contour-grid N     Grid points per contour axis (default: 25)
  --skip-multi-start   Skip multi-start analysis
  --skip-profiles      Skip 1-D parameter profiles
  --skip-contours      Skip 2-D contour analysis
  --skip-hessian       Skip Hessian analysis
  --skip-convergence   Skip convergence trace
  --profile-params P [P ...]  Only profile specific parameters
  --contour-pairs P [P ...]   Parameter pairs (e.g. 'qmax,Ks qmax,b_decay')
  --scale-params       Run re-optimisations in [0,1]-normalised parameter space
  -v, --verbose        Verbose output

Output:
  diagnostics_summary.json  — Complete results in JSON
  diag_multi_start.png      — Multi-start objective histogram
  diag_profiles.png         — 1-D profile likelihoods
  diag_contours.png         — 2-D objective contours
  diag_hessian.png          — Eigenvalue spectrum & correlation matrix
  diag_convergence.png      — Optimization trajectory
```

**Diagnostics performed:**

1. **Multi-start optimization** — Runs L-BFGS-B from many random starting points to test sensitivity to initial guesses and detect alternative minima.
2. **1-D parameter profiles** — Fixes each parameter across its bounds and re-optimises the others to reveal identifiability. Flat profiles indicate poorly constrained parameters.
3. **2-D contour surfaces** — Objective function contours for parameter pairs showing correlations and ridges.
4. **Hessian eigenvalue analysis** — Condition number and eigenvalue spectrum. High condition numbers (>10⁸) indicate ill-conditioning.
5. **Convergence trace** — Records the full optimisation trajectory showing how parameters evolve.

```bash
# Quick run with just profiles and Hessian
python scripts/run_diagnostics.py \
    -c config/substrates/glucose.json \
    -d data/example/Experimental_data_glucose.csv \
    -m single_haldane --skip-contours --skip-convergence -v

# Run with scaled re-optimisations for better conditioning
python scripts/run_diagnostics.py \
    -c config/substrates/glucose.json \
    -d data/example/Experimental_data_glucose.csv \
    -m dual_haldane_lag --scale-params -v
```

### estimate_bounds.py

Estimates theoretical parameter ceilings from substrate stoichiometry (molecular formula and molecular weight). Useful for setting or validating parameter bounds before fitting.

```
Usage: python scripts/estimate_bounds.py [OPTIONS]

Input (one of):
  --config PATH        Single substrate config JSON file
  --all                Process all config files in config/substrates/
  --formula FORMULA    Molecular formula (e.g. C6H12O6). Requires --name and --mw.

Options:
  --name NAME          Substrate name (required with --formula)
  --mw MW              Molecular weight in g/mol (required with --formula)
  --compare            Compare theoretical bounds against current config bounds
  -o, --output PATH    Save results to a JSON file
  --config-dir DIR     Config directory for --all mode
```

The tool computes:
- **Degree of reduction (γ)** — electron equivalents per C-mol of substrate
- **Theoretical maximum yield (Y_max)** — from electron balance (γ_S / γ_B)
- **Theoretical oxygen demand (ThOD)** — mg O₂ per mg substrate
- **Y_o2 ceiling** — maximum oxygen yield from stoichiometry
- **qmax heuristic range** — based on substrate class (sugar, aromatic, etc.)

```bash
# Report for a single substrate
python scripts/estimate_bounds.py --config config/substrates/glucose.json

# Compare theoretical vs current bounds for all substrates
python scripts/estimate_bounds.py --all --compare

# Ad-hoc formula (no config file needed)
python scripts/estimate_bounds.py --name "Phenol" --formula C6H6O --mw 94.11
```

### robust_fit.py

Robust parameter fitting combining weighting, two-stage initialization, and bootstrap uncertainty quantification.

```
Usage: python scripts/robust_fit.py [OPTIONS]

Options:
  -c, --config PATH       Substrate configuration JSON file (required)
  -d, --data PATH         Experimental data CSV file (required)
  -w, --workflow TYPE     Model type: single_monod, single_monod_lag, dual_monod, dual_monod_lag
                          (Haldane aliases also accepted: single_haldane, single_haldane_lag, dual_haldane, dual_haldane_lag)
  --weighting TYPE        Weighting strategy: uniform, max_value, variance, range
                          (default: max_value)
  --two-stage             Enable two-stage initialization (default: enabled)
  --no-two-stage          Disable two-stage initialization
  -b, --bootstrap N       Number of bootstrap iterations (0 to disable, default: 500)
  --workers N             Number of parallel workers for bootstrap (default: auto)
  --seed N                Random seed for reproducibility
  -o, --output DIR        Output directory (default: results/)
  -q, --quiet             Suppress progress output

Output includes:
  - Parameter estimates with bootstrap 95% confidence intervals
  - Per-condition fit statistics (R², RMSE for substrate and biomass)
  - Two-stage initialization quality (R² of algebraic fit)
  - Bootstrap success rate and parameter distributions
  - Condition weights used for heteroscedasticity correction
```

### run_all_models.py

Runs all model variants across all substrates in batch, building a master comparison CSV.

```
Usage: python scripts/run_all_models.py [OPTIONS]

Substrate / model selection:
  --substrates NAME [NAME ...]   Substrates to fit (default: all 6)
                                 Choices: glucose, xylose, vanillic_acid, p_coumaric_acid,
                                          p_hydroxybenzoic_acid, syringic_acid
  --models MODEL [MODEL ...]     Models to fit (default: all 8)
                                 Choices: single_monod, single_haldane, single_monod_lag,
                                          single_haldane_lag, dual_monod, dual_haldane,
                                          dual_monod_lag, dual_haldane_lag

Output paths:
  --master PATH                  Path to master CSV (default: output/master_results.csv)
  --output DIR                   Base output directory for per-run results (default: results)

Fitting options:
  --optimizer TYPE               Optimization algorithm: L-BFGS-B or differential_evolution
                                 (default: L-BFGS-B)
  --global-guess TYPE            Strategy for global optimisation initial guess (default: median):
                                  median  — median of per-condition fits
                                  best_r2 — params from best-fitting condition
  --ci-method TYPE               Confidence interval method: hessian, hessian_log, mcmc
                                 (default: mcmc)
  --normalize-objective          Weight S and X residuals by 1/range
  --mcmc-adaptive                Use Hessian-informed adaptive MCMC proposals
  --scale-diagnostics            Run re-optimisations in [0,1]-normalised parameter space

Execution:
  --workers N                    Parallel workers (default: CPU count - 1). Use 1 for sequential.
  --no-plots                     Skip plot generation (faster)
  -v, --verbose                  Enable verbose output for each fit

Output:
  - Per-run results: results/{model}/{Substrate}/{timestamp}/
  - Master CSV: output/master_results.csv
    Columns: Substrate, Model, parameters + 95% CIs, Total_Error, R², AIC, AIC_weight
```

The master CSV is the primary input for cross-model analysis. Each row represents one (substrate, model) combination with re-evaluated global metrics.

**Recommended flags for publication-quality runs:**

```bash
# Guess-free global search + best-R² global guess + adaptive MCMC CIs
python scripts/run_all_models.py \
    --optimizer differential_evolution \
    --global-guess best_r2 \
    --ci-method mcmc --mcmc-adaptive \
    --workers 10 --verbose
```

### analyze_results.py

Generates 6 publication-quality analysis figures from the batch fitting results.

```
Usage: python scripts/analyze_results.py [OPTIONS]

Options:
  --master-csv PATH    Path to master results CSV (default: output/master_results.csv)
  --results-dir DIR    Base directory with per-run results (default: results)
  --output-dir DIR     Directory for saved figures (default: output/figures)
  --formats FMT [...]  Output formats (default: png pdf)
  --dpi N              Resolution for raster formats (default: 300)

Output (14 files = 7 figures × 2 formats):
  figure_A_r2_heatmap.{png,pdf}       — R² heatmap (substrate + biomass panels)
  figure_B_delta_aic.{png,pdf}         — ΔAIC grouped bar chart
  figure_C_cv_dotplot.{png,pdf}        — Parameter identifiability (CV%) dot plot
  figure_D_residuals.{png,pdf}         — Residual diagnostics (best model per substrate)
  figure_E_total_error_bars.{png,pdf}  — Total Error bar chart (log scale)
  figure_F_total_error_heatmap.{png,pdf} — Total Error heatmap (log₁₀ scale)
  figure_G_parameter_dotplots.{png,pdf} — Parameter values across models per substrate
```

Figures A, C, and D read per-run JSON files from `results/`. Figures B, E, F, and G use only the master CSV.

### convert_units.py

Convert kinetic parameter units between time bases.

```
Usage: python scripts/convert_units.py [OPTIONS]

Examples:
  # Convert parameter file from days to hours
  python scripts/convert_units.py --params fitted.json --from days --to hours

  # Convert single value
  python scripts/convert_units.py --value 2.5 --param qmax --from days --to minutes

  # Convert concentration
  python scripts/convert_units.py --conc 500 --mw 150.13 --from mg/L --to mM
```

## Configuration File Format

```json
{
  "substrate": {
    "name": "Xylose",
    "molecular_formula": "C5H10O5",
    "molecular_weight": 150.13,
    "unit": "mg/L"
  },
  "initial_guesses": {
    "qmax": 2.5,
    "Ks": 400.0,
    "Ki": 25000.0,
    "Y": 0.35,
    "b_decay": 0.01,
    "K_o2": 0.15,
    "Y_o2": 0.8,
    "lag_time": 3.2
  },
  "bounds": {
    "qmax": [0.1, 10.0],
    "Ks": [10.0, 2000.0],
    "Ki": [50.0, 50000.0],
    "Y": [0.1, 1.0],
    "b_decay": [0.001, 0.2],
    "K_o2": [0.05, 1.0],
    "Y_o2": [0.1, 2.0],
    "lag_time": [0.0, 10.0]
  },
  "oxygen": {
    "o2_max": 8.0,
    "o2_min": 0.01,
    "reaeration_rate": 5.0
  },
  "simulation": {
    "t_final": 5.0,
    "num_points": 10000,
    "time_unit": "days"
  }
}
```

## Parameter Definitions

| Parameter | Description | Typical Units |
|-----------|-------------|---------------|
| qmax | Maximum specific uptake rate | mg substrate/(mg cells·day) |
| Ks | Half-saturation constant | mg substrate/L |
| Ki | Substrate inhibition constant | mg substrate/L |
| Y | Yield coefficient | mg cells/mg substrate |
| b_decay | Decay/maintenance coefficient | day⁻¹ |
| K_o2 | Oxygen half-saturation | mg O2/L |
| Y_o2 | Oxygen yield coefficient | mg O2/mg substrate |
| lag_time | Lag phase duration | days |

## Output Files

### Per-Run Results (fit_individual.py)

After fitting, results are saved to:

```
results/{substrate_name}/{timestamp}/
├── individual_condition_results.json  # Full results with CIs, per-condition stats
├── individual_condition_results.csv   # Tabular per-condition results
├── individual_condition_summary.txt   # Human-readable summary
├── results_report.pdf                 # PDF report
├── substrate_summary.png              # Substrate fit curves (all conditions overlaid)
├── biomass_summary.png                # Biomass fit curves (all conditions overlaid)
├── parameter_comparison.png           # Parameter values across conditions
├── residual_diagnostics.png           # Residual analysis plots
├── confidence_intervals.png           # 95% CIs by condition
└── goodness_of_fit.png                # R² and NRMSE summary heatmap
```

### Batch Results (run_all_models.py)

After batch fitting, additional outputs:

```
output/
├── master_results.csv                 # One row per (substrate, model) combination
│                                      # Columns: Substrate, Model, parameters, CIs,
│                                      #          Total_Error, R², AIC, AIC_weight
└── figures/                           # Generated by analyze_results.py
    ├── figure_A_r2_heatmap.{png,pdf}
    ├── figure_B_delta_aic.{png,pdf}
    ├── figure_C_cv_dotplot.{png,pdf}
    ├── figure_D_residuals.{png,pdf}
    ├── figure_E_total_error_bars.{png,pdf}
    ├── figure_F_total_error_heatmap.{png,pdf}
    └── figure_G_parameter_dotplots.{png,pdf}
```

### Legacy Per-Run Results (fit_parameters.py)

```
results/{substrate_name}/{timestamp}/
├── fitted_parameters.json    # Fitted parameter values with confidence intervals
├── model_predictions.csv     # Model predictions
├── statistics.json           # Fit statistics (R², RMSE, AIC, BIC)
├── run_info.json            # Run metadata
└── figures/
    ├── fit_results_*.png    # Fit plots
    └── fit_results_*.pdf    # PDF versions
```

## Confidence Intervals

The software automatically computes 95% confidence intervals for all fitted parameters using a Hessian-based approximation. These intervals provide uncertainty estimates for publication-quality reporting.

### How Confidence Intervals Are Calculated

1. **Hessian Matrix**: The second-derivative matrix of the objective function is approximated numerically at the optimal parameter values
2. **Covariance Matrix**: Inverted Hessian scaled by residual variance gives the parameter covariance
3. **Standard Errors**: Square root of diagonal elements of covariance matrix
4. **Confidence Bounds**: Parameter ± t-critical × standard error

### Interpreting Confidence Intervals

The output displays parameters in the format: `value ± half-width [lower, upper]`

Example output:
```
Fitted Parameters with 95% Confidence Intervals:
  qmax: 12.45 ± 0.82 [11.63, 13.27] mgGlucose/(mgCells·day)
  Ks: 245.3 ± 18.4 [226.9, 263.7] mgGlucose/L
  Y: 0.152 ± 0.008 [0.144, 0.160] mgCells/mgGlucose
```

### Guidelines for Confidence Intervals

- **Narrow intervals** (< 10% of parameter value): High confidence in parameter estimate
- **Wide intervals** (> 50% of parameter value): Parameter poorly constrained by data
- **Overlapping intervals**: Parameters may be correlated or data insufficient to distinguish
- **Asymmetric intervals**: May indicate parameter near boundary constraint

### MCMC Confidence Intervals

For Bayesian-style uncertainty estimates without distributional assumptions on the Hessian, use MCMC:

```bash
python scripts/fit_individual.py \
    -c config/substrates/glucose.json \
    -d data/example/Experimental_data_glucose.csv \
    -m dual_haldane_lag \
    --ci-method mcmc --mcmc-samples 5000 --mcmc-burn-in 1500 --mcmc-seed 42
```

MCMC runs a multi-chain random-walk Metropolis sampler and reports:
- Percentile-based CIs from the posterior distribution
- R̂ (Gelman–Rubin convergence diagnostic; target < 1.05)
- Effective sample size (ESS; higher is better)
- Acceptance rate (target 20–50%)

**Adaptive MCMC proposals:** By default, proposals are scaled by the parameter range. The `--mcmc-adaptive` flag uses the Hessian covariance to shape proposals (Haario et al., 2001), which improves mixing when parameters have very different scales or are correlated:

```bash
python scripts/fit_individual.py -c ... -d ... --ci-method mcmc --mcmc-adaptive
```

### When Confidence Intervals Cannot Be Computed

The Hessian-based method may fail if:
- The optimization did not converge properly
- The Hessian matrix is singular (parameters highly correlated)
- Parameters are at their boundary constraints

In these cases, a warning message is displayed and empty confidence intervals are returned.

### Bootstrap Confidence Intervals

For more robust uncertainty estimates, use `robust_fit.py` with bootstrap enabled. Bootstrap confidence intervals use residual resampling:

1. Fit model to original data and compute residuals
2. Resample residuals with replacement (preserving time structure)
3. Add resampled residuals to predictions to create synthetic data
4. Re-fit model to synthetic data
5. Repeat 500+ times to build parameter distributions
6. Extract percentile-based confidence intervals

Bootstrap CIs are more robust than Hessian-based CIs because they:
- Make no distributional assumptions
- Handle correlated parameters naturally
- Work even when the Hessian is ill-conditioned

```bash
# Run robust fit with 1000 bootstrap iterations
python scripts/robust_fit.py \
    --config config/substrates/glucose.json \
    --data data/example/Experimental_data_glucose.csv \
    --bootstrap 1000 \
    --workers 4
```

### Weighting Strategies

When fitting across multiple substrate concentrations, data heteroscedasticity can bias parameter estimates. Four weighting strategies are available:

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| `uniform` | All conditions equal weight | Conditions have similar data ranges |
| `max_value` | Weight by 1/max(biomass)² | **Default**: compensates for concentration-dependent scale |
| `variance` | Weight by 1/variance | When measurement noise varies across conditions |
| `range` | Weight by 1/range² | When initial values differ across conditions |

```bash
# Use variance-based weighting
python scripts/robust_fit.py \
    --config config/substrates/xylose.json \
    --data data/example/data.csv \
    --weighting variance
```

## Interpreting Results

### R² (Coefficient of Determination)
- Values close to 1.0 indicate good fit
- Values > 0.9 generally acceptable for biological data

### AIC/BIC (Information Criteria)
- Lower values indicate better model
- Use for comparing models with different numbers of parameters
- AIC penalizes complexity less than BIC

### Model Selection Guidelines

**Choosing a base model family:**
1. Start with the simplest model that includes the relevant physics (Single Monod for anaerobic/O2-excess, Dual Monod for aerobic systems, Dual Monod + Lag if lag is visible)
2. If R² values are similar between families, prefer the simpler model

**Choosing Monod vs Haldane (with/without Ki):**
1. Start with the basic Monod variant (e.g., `single_monod`, `dual_monod`) — fewer parameters, easier to identify
2. Add Haldane inhibition (e.g., `single_haldane`, `dual_haldane`) only if high substrate concentrations show reduced growth rates
3. Compare AIC/BIC values between Monod and Haldane variants to justify the extra parameter

## Batch Workflow: Full Cross-Model Analysis

For a complete publication-ready analysis comparing all model families across all substrates:

```bash
# Step 1: Run all 48 fits (6 substrates × 8 models)
#   - differential_evolution ignores initial guesses (only bounds matter)
#   - best_r2 uses the best-fitting condition as the global starting point
#   - mcmc + mcmc-adaptive provides publication-quality uncertainty estimates
python scripts/run_all_models.py \
    --optimizer differential_evolution \
    --global-guess best_r2 \
    --ci-method mcmc --mcmc-adaptive \
    --workers 10 --verbose

# Step 2: Generate publication figures
python scripts/analyze_results.py

# Step 3: Inspect the master CSV
# output/master_results.csv — open in Excel, pandas, or R

# Step 4: Inspect figures
# output/figures/figure_*.png
```

**Quick (draft) batch run** — for fast iteration while developing configs:

```bash
# Hessian CIs are ~100× faster than MCMC; --no-plots saves disk I/O
python scripts/run_all_models.py --ci-method hessian --no-plots
```

**Recommended workflow for a new substrate:**

```bash
# 1. Create config file
cp config/substrates/glucose.json config/substrates/my_substrate.json
# Edit substrate name, molecular_formula, molecular_weight, and bounds

# 2. Validate bounds against stoichiometry
python scripts/estimate_bounds.py --config config/substrates/my_substrate.json --compare

# 3. Explore with simplest model first (guess-free)
python scripts/fit_individual.py \
    -c config/substrates/my_substrate.json \
    -d data/my_experiment.csv \
    -m single_monod --optimizer differential_evolution --workers 0 -v

# 4. If fit is reasonable, run all 8 models
python scripts/run_all_models.py \
    --substrates my_substrate \
    --optimizer differential_evolution --global-guess best_r2 \
    --ci-method mcmc --mcmc-adaptive --workers 10

# 5. Generate comparison figures
python scripts/analyze_results.py
```

**Master CSV columns explained:**

| Column | Description |
|--------|-------------|
| `Substrate` | Substrate name (e.g., Glucose) |
| `Model` | Display name (e.g., Single Monod, Dual Monod (Haldane)) |
| `qmax`, `Ks`, ... | Global parameter point estimates |
| `qmax_95CI`, `Ks_95CI`, ... | 95% CI as `[lower, upper]` string |
| `Total_Error` | Raw SSE (substrate + biomass) across all conditions |
| `R2` | Combined R² using global parameters |
| `AIC` | Akaike Information Criterion |
| `AIC_weight` | Akaike weight within substrate group (sums to 1.0) |

**Interpreting the figures:**

- **Figure A (R² Heatmap):** Green cells = good fit. Compare substrate R² vs biomass R² to identify which component is harder to fit.
- **Figure B (ΔAIC):** Stars mark the best model per substrate. Bars close to 0 are statistically comparable.
- **Figure C (CV% Dot Plot):** Small green dots = well-identified parameters. Large red dots = poorly constrained. Only shows the best model per substrate.
- **Figure D (Residuals):** Points randomly scattered around zero = good model. Patterns indicate systematic misfit.
- **Figures E-F (Total Error):** Compare absolute fit quality across model families. Lower is better. Log scale handles multi-order-of-magnitude range.
- **Figure G (Parameter Dot Plots):** One panel per substrate. Each dot is a parameter value from one model; horizontal error bars show 95% CIs. Stars (★) mark the best-AIC model. Use this to see how parameter estimates shift across model families and whether CIs overlap (indicating the models agree on that parameter).

## Tutorial: Choosing the Best Model for Your Substrate

This tutorial walks you through the complete workflow — from raw experimental data to a defensible model selection — using **p-Hydroxybenzoic Acid** as a worked example. By the end you will have:

- Fitted all 8 model variants to your data
- Generated publication-quality comparison figures
- Applied a structured decision process to select the best model

### Prerequisites

1. The software is installed (see [Installation](#installation))
2. You have a CSV file with your experimental data in the [required format](#1-prepare-your-data)
3. You have a substrate config JSON in `config/substrates/` (see [Configuration File Format](#configuration-file-format))

---

### Step 1 — Validate your configuration

Before fitting anything, check that your parameter bounds are physically reasonable by comparing them against stoichiometric ceilings:

```bash
python scripts/estimate_bounds.py \
    --config config/substrates/p_hydroxybenzoic_acid.json --compare
```

Look for any parameters where the upper bound exceeds the theoretical maximum — if so, tighten the bounds in your config JSON. This step prevents the optimizer from exploring physically impossible regions.

### Step 2 — Quick exploratory fit with the simplest model

Start with the simplest model (`single_monod`, 4 parameters) to verify that your data loads correctly and the optimizer can find a reasonable solution:

```bash
python scripts/fit_individual.py \
    -c config/substrates/p_hydroxybenzoic_acid.json \
    -d data/example/pHydroxybenzoicAcid_experimental_data.csv \
    -m single_monod \
    --optimizer differential_evolution \
    --verbose
```

**What to look for:**
- Does each condition converge? Check that R² > 0.8 for most conditions.
- Do substrate and biomass curves track the data visually? Open `substrate_summary.png` and `biomass_summary.png`.
- Are parameter values physically plausible? (e.g., Y < 1, qmax > 0)

If the simplest model fails badly, the issue is likely in data formatting or bounds — fix those before proceeding.

### Step 3 — Run all 8 models

Once the exploratory fit looks reasonable, run all 8 model variants systematically:

```bash
python scripts/run_all_models.py \
    --substrates p_hydroxybenzoic_acid \
    --optimizer differential_evolution \
    --global-guess best_r2 \
    --ci-method mcmc --mcmc-adaptive \
    --workers 10 --verbose
```

**What this does:**
- Fits each of the 8 models (single_monod, single_haldane, single_monod_lag, single_haldane_lag, dual_monod, dual_haldane, dual_monod_lag, dual_haldane_lag) to your data
- Uses `differential_evolution` so results are independent of initial guesses
- Uses `best_r2` to start the global optimisation from the best-fitting condition
- Computes MCMC confidence intervals with adaptive proposals
- Writes one row per model to `output/master_results.csv`

> **⏱ Runtime note:** With MCMC CIs this can take 10–30 minutes per model depending on your data and hardware. For a faster draft run, use `--ci-method hessian --no-plots`.

### Step 4 — Generate comparison figures

```bash
python scripts/analyze_results.py
```

This reads `output/master_results.csv` and per-run JSON files to produce 6 publication figures in `output/figures/`.

### Step 5 — Select the best model

Open the figures and apply the following decision tree:

#### 5a. Check overall fit quality (Figure A — R² Heatmap)

| What you see | Interpretation |
|---|---|
| All cells green (R² > 0.9) | All models fit well — model selection will hinge on parsimony |
| Some rows red (R² < 0.7) | Those model families are inadequate — eliminate them |
| Biomass R² much lower than substrate R² | Biomass dynamics are harder to capture — prefer models with more biomass-related parameters (Dual Monod family) |

#### 5b. Compare parsimony (Figure B — ΔAIC)

AIC balances fit quality against model complexity (number of parameters). Lower AIC = better.

| ΔAIC from best model | Interpretation |
|---|---|
| 0 (star marker) | Best model for this substrate |
| < 2 | Statistically indistinguishable from the best |
| 2–7 | Some evidence against this model |
| > 10 | Strong evidence against — discard |

**Rule:** If a simpler model is within ΔAIC < 2 of a more complex model, prefer the simpler one.

#### 5c. Check parameter identifiability (Figure C — CV% Dot Plot)

This figure shows the coefficient of variation (CV = std/mean × 100%) for each parameter of the best model.

| CV% | Colour | Interpretation |
|---|---|---|
| < 20% | Green | Well-identified — data constrains this parameter |
| 20–50% | Yellow | Moderately constrained — report with caution |
| > 50% | Red | Poorly constrained — the data may not support this parameter |

**Red flag:** If the Haldane variant wins on AIC but Ki has CV > 50%, the inhibition effect may not be statistically supported. Consider reverting to the non-Haldane variant.

#### 5d. Inspect residuals (Figure D — Residual Diagnostics)

| Pattern | Interpretation |
|---|---|
| Random scatter around zero | Good — model captures the data structure |
| Systematic curvature (U-shape) | Model is missing a dynamic — consider a more complex family |
| Fan shape (increasing spread) | Heteroscedasticity — consider `--normalize-objective` |
| Clusters of positive/negative | Temporal autocorrelation — model may be over-smoothing |

#### 5e. Cross-check total error (Figures E & F)

Figures E and F show raw SSE. These complement AIC (which penalises complexity).

- If the Haldane variant reduces Total Error by < 5% compared to the base Monod, the extra parameter is not earning its keep.
- If Dual Monod + Lag reduces error dramatically compared to Dual Monod, the lag phase is important.

### Step 6 — Final model selection summary

Combine the evidence into a decision table:

| Criterion | Preferred model |
|---|---|
| Lowest AIC (or within ΔAIC < 2 of lowest) | ✓ |
| R² > 0.9 for both substrate and biomass | ✓ |
| All parameters CV < 50% | ✓ |
| No systematic residual patterns | ✓ |
| Fewest parameters (parsimony) | Tiebreaker |

**Example decision for p-Hydroxybenzoic Acid:**

> Dual Monod + Lag (Haldane) had the lowest AIC but Ki showed CV = 85%. Dual Monod + Lag (no Haldane, 7 params) was within ΔAIC = 1.3 and all parameters had CV < 30%. → **Select Dual Monod + Lag** as the most parsimonious, well-identified model.

### Step 7 — Publication-quality refit of the chosen model

Once you've selected the best model, run a final high-quality fit with more MCMC samples for tighter CIs:

```bash
python scripts/fit_individual.py \
    -c config/substrates/p_hydroxybenzoic_acid.json \
    -d data/example/pHydroxybenzoicAcid_experimental_data.csv \
    -m dual_monod_lag \
    --optimizer differential_evolution \
    --global-guess best_r2 \
    --ci-method mcmc --mcmc-samples 10000 --mcmc-burn-in 3000 --mcmc-adaptive \
    --workers 10 --verbose
```

This produces the final plots, parameter table, and PDF report in `results/pHydroxybenzoicAcid/{timestamp}/`.

### Step 8 (Optional) — Run diagnostics on the chosen model

To investigate parameter identifiability and the objective function landscape in detail:

```bash
python scripts/run_diagnostics.py \
    -c config/substrates/p_hydroxybenzoic_acid.json \
    -d data/example/pHydroxybenzoicAcid_experimental_data.csv \
    -m dual_monod_lag --scale-params -v
```

The diagnostics output includes profile likelihoods, contour plots, multi-start analysis, and Hessian eigenvalue decomposition. Use these to:
- Confirm there is a single well-defined minimum (multi-start histogram should be unimodal)
- Verify all parameters have clear V-shaped profiles (flat profiles = unidentifiable)
- Check for strong parameter correlations (elongated contours)

### Quick-reference: complete pipeline in 4 commands

```bash
# 1. Validate bounds
python scripts/estimate_bounds.py --config config/substrates/my_substrate.json --compare

# 2. Fit all 8 models
python scripts/run_all_models.py \
    --substrates my_substrate \
    --optimizer differential_evolution --global-guess best_r2 \
    --ci-method mcmc --mcmc-adaptive --workers 10 --verbose

# 3. Generate figures
python scripts/analyze_results.py

# 4. Refit chosen model with high-quality CIs
python scripts/fit_individual.py \
    -c config/substrates/my_substrate.json \
    -d data/my_experiment.csv \
    -m <chosen_model> \
    --optimizer differential_evolution --global-guess best_r2 \
    --ci-method mcmc --mcmc-samples 10000 --mcmc-burn-in 3000 --mcmc-adaptive \
    --workers 10 --verbose
```

---

## Troubleshooting

### "Optimization did not converge"
- Try adjusting initial guesses closer to expected values
- Widen parameter bounds
- Use differential_evolution optimizer for global search

### Poor fit quality
- Check data format and units
- Verify initial conditions match data
- Consider using a different model type
- Try `--optimizer differential_evolution` for global search (eliminates dependence on initial guesses)
- Try `--normalize-objective` if substrate and biomass data have very different scales
- Use `--global-guess best_r2` if the median of individual fits seems like a poor starting point

### MCMC acceptance rate too low or too high
- Target acceptance rate: 20–50%
- Too low (< 10%): decrease `--mcmc-step-scale` (e.g., 0.01)
- Too high (> 80%): increase `--mcmc-step-scale` (e.g., 0.1)
- Try `--mcmc-adaptive` which uses Hessian covariance to shape proposals automatically

### Parameter identifiability issues
- Run `scripts/run_diagnostics.py` to inspect the objective landscape
- Check 1-D profiles: flat profiles mean the parameter is poorly constrained
- Check condition number in Hessian analysis: > 10⁸ indicates ill-conditioning
- Run `scripts/estimate_bounds.py --compare` to verify bounds are physically reasonable

### Memory issues
- Reduce num_points in configuration
- Use fewer conditions at once
