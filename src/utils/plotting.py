"""
Plotting utilities for kinetic parameter estimation results.

This module provides publication-quality plotting functions for:
- Experimental vs fitted data
- Model comparison
- Parameter sensitivity
- Residual analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Set default style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')

# Colorblind-friendly palette (Paul Tol's bright)
COLORS = {
    'blue': '#4477AA',
    'red': '#EE6677',
    'green': '#228833',
    'yellow': '#CCBB44',
    'cyan': '#66CCEE',
    'purple': '#AA3377',
    'grey': '#BBBBBB',
    'orange': '#EE7733',
}

COLOR_LIST = list(COLORS.values())


def setup_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: Tuple[float, float] = None,
    dpi: int = 300
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a figure with standard formatting.

    Args:
        nrows: Number of subplot rows
        ncols: Number of subplot columns
        figsize: Figure size in inches. If None, auto-calculated.
        dpi: Resolution in dots per inch

    Returns:
        Tuple of (figure, axes array)
    """
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)

    # Ensure axes is always 2D array for consistent indexing
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    return fig, axes


def style_axis(
    ax: plt.Axes,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    legend: bool = True,
    grid: bool = True
) -> None:
    """
    Apply standard styling to an axis.

    Args:
        ax: Matplotlib axis object
        title: Axis title
        xlabel: X-axis label
        ylabel: Y-axis label
        legend: Whether to show legend
        grid: Whether to show grid
    """
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)

    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=True, fancybox=True, framealpha=0.9, fontsize=9)

    if grid:
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    ax.tick_params(labelsize=10, width=1.2)


def plot_fit_results(
    experimental_data: pd.DataFrame,
    model_results: Dict[str, pd.DataFrame],
    time_column: str = "Time (days)",
    substrate_name: str = "Substrate",
    output_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot experimental data with fitted model curves.

    Args:
        experimental_data: DataFrame with experimental measurements
        model_results: Dictionary mapping condition labels to model DataFrames
        time_column: Name of time column
        substrate_name: Name of substrate for labels
        output_path: If provided, save figure to this path
        show: Whether to display the figure

    Returns:
        Matplotlib figure object
    """
    n_conditions = len(model_results)
    fig, axes = setup_figure(nrows=2, ncols=n_conditions)

    for i, (label, model_df) in enumerate(model_results.items()):
        # Get experimental columns for this condition
        substrate_col = f"{label}_{substrate_name} (mg/L)"
        biomass_col = f"{label}_Biomass (mgCells/L)"

        # Plot substrate
        ax_sub = axes[0, i]
        if substrate_col in experimental_data.columns:
            ax_sub.scatter(
                experimental_data[time_column],
                experimental_data[substrate_col],
                color=COLORS['red'],
                s=60,
                label='Experimental',
                zorder=5,
                edgecolors='white',
                linewidths=1.5
            )
        ax_sub.plot(
            model_df['Time'],
            model_df['Substrate'],
            color=COLORS['blue'],
            linewidth=2,
            label='Model'
        )
        style_axis(
            ax_sub,
            title=f'{substrate_name} - {label}',
            xlabel='Time (days)',
            ylabel=f'{substrate_name} (mg/L)'
        )

        # Plot biomass
        ax_bio = axes[1, i]
        if biomass_col in experimental_data.columns:
            ax_bio.scatter(
                experimental_data[time_column],
                experimental_data[biomass_col],
                color=COLORS['red'],
                s=60,
                label='Experimental',
                zorder=5,
                edgecolors='white',
                linewidths=1.5
            )
        ax_bio.plot(
            model_df['Time'],
            model_df['Biomass'],
            color=COLORS['green'],
            linewidth=2,
            label='Model'
        )
        style_axis(
            ax_bio,
            title=f'Biomass - {label}',
            xlabel='Time (days)',
            ylabel='Biomass (mg cells/L)'
        )

    plt.tight_layout(pad=2.0)

    if output_path:
        save_figure(fig, output_path)

    if show:
        plt.show()

    return fig


def plot_model_comparison(
    comparison_data: Dict[str, Dict[str, Any]],
    metrics: List[str] = None,
    output_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot comparison of different model types.

    Args:
        comparison_data: Dictionary with model names as keys, containing
                        'metrics' dict and 'predictions' DataFrame
        metrics: List of metrics to plot (default: R2, RMSE)
        output_path: Path to save figure
        show: Whether to display

    Returns:
        Matplotlib figure
    """
    if metrics is None:
        metrics = ['R_squared', 'RMSE']

    model_names = list(comparison_data.keys())
    n_metrics = len(metrics)

    fig, axes = setup_figure(nrows=1, ncols=n_metrics, figsize=(5*n_metrics, 4))

    for i, metric in enumerate(metrics):
        ax = axes[0, i]
        values = [comparison_data[m]['metrics'].get(metric, 0) for m in model_names]

        bars = ax.bar(
            model_names,
            values,
            color=[COLOR_LIST[j % len(COLOR_LIST)] for j in range(len(model_names))],
            edgecolor='white',
            linewidth=2
        )

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(
                f'{val:.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontsize=10
            )

        style_axis(ax, title=metric, ylabel=metric, legend=False)
        ax.set_xticklabels(model_names, rotation=45, ha='right')

    plt.tight_layout(pad=2.0)

    if output_path:
        save_figure(fig, output_path)

    if show:
        plt.show()

    return fig


def plot_residuals(
    experimental: np.ndarray,
    predicted: np.ndarray,
    time: np.ndarray = None,
    variable_name: str = "Value",
    output_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot residual analysis for model fit.

    Creates a 2x2 plot with:
    - Residuals vs time
    - Residuals histogram
    - Predicted vs actual
    - Q-Q plot

    Args:
        experimental: Experimental values
        predicted: Model predicted values
        time: Time points (optional)
        variable_name: Name of variable for labels
        output_path: Path to save figure
        show: Whether to display

    Returns:
        Matplotlib figure
    """
    residuals = experimental - predicted

    fig, axes = setup_figure(nrows=2, ncols=2)

    # Residuals vs time (or index)
    ax = axes[0, 0]
    x = time if time is not None else np.arange(len(residuals))
    ax.scatter(x, residuals, color=COLORS['blue'], alpha=0.7)
    ax.axhline(y=0, color=COLORS['red'], linestyle='--', linewidth=1.5)
    style_axis(ax, title='Residuals vs Time', xlabel='Time', ylabel='Residual')

    # Histogram of residuals
    ax = axes[0, 1]
    ax.hist(residuals, bins=20, color=COLORS['green'], edgecolor='white', alpha=0.8)
    style_axis(ax, title='Residual Distribution', xlabel='Residual', ylabel='Frequency', legend=False)

    # Predicted vs actual
    ax = axes[1, 0]
    ax.scatter(experimental, predicted, color=COLORS['purple'], alpha=0.7)
    lims = [min(experimental.min(), predicted.min()), max(experimental.max(), predicted.max())]
    ax.plot(lims, lims, color=COLORS['red'], linestyle='--', linewidth=1.5, label='Perfect fit')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    style_axis(ax, title='Predicted vs Actual', xlabel=f'Actual {variable_name}',
               ylabel=f'Predicted {variable_name}')

    # Residuals vs predicted
    ax = axes[1, 1]
    ax.scatter(predicted, residuals, color=COLORS['orange'], alpha=0.7)
    ax.axhline(y=0, color=COLORS['red'], linestyle='--', linewidth=1.5)
    style_axis(ax, title='Residuals vs Predicted', xlabel=f'Predicted {variable_name}',
               ylabel='Residual', legend=False)

    plt.tight_layout(pad=2.0)

    if output_path:
        save_figure(fig, output_path)

    if show:
        plt.show()

    return fig


def plot_lag_phase(
    time: np.ndarray,
    lag_factor: np.ndarray,
    lag_time: float,
    output_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot the lag phase factor over time.

    Args:
        time: Time array
        lag_factor: Lag phase factor values
        lag_time: Estimated lag time
        output_path: Path to save figure
        show: Whether to display

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    ax.plot(time, lag_factor, color=COLORS['blue'], linewidth=2.5)
    ax.axvline(x=lag_time, color=COLORS['red'], linestyle='--', linewidth=1.5,
               label=f'Lag time = {lag_time:.2f} days')
    ax.axhline(y=0.5, color=COLORS['grey'], linestyle=':', linewidth=1,
               label='50% activity')

    ax.fill_between(time, 0, lag_factor, alpha=0.2, color=COLORS['blue'])

    style_axis(ax, title='Lag Phase Factor', xlabel='Time (days)',
               ylabel='Growth Activity Factor')

    ax.set_ylim(-0.05, 1.1)

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)

    if show:
        plt.show()

    return fig


def save_figure(
    fig: plt.Figure,
    path: Path,
    formats: List[str] = None,
    dpi: int = 300
) -> List[Path]:
    """
    Save a figure in multiple formats.

    Args:
        fig: Matplotlib figure
        path: Base path (without extension)
        formats: List of formats (default: ['png', 'pdf'])
        dpi: Resolution for raster formats

    Returns:
        List of saved file paths
    """
    if formats is None:
        formats = ['png', 'pdf']

    path = Path(path)
    saved_paths = []

    for fmt in formats:
        save_path = path.with_suffix(f'.{fmt}')
        fig.savefig(
            save_path,
            format=fmt,
            dpi=dpi,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none'
        )
        saved_paths.append(save_path)

    return saved_paths
