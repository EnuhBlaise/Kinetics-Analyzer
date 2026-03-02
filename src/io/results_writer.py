"""
Results writing and output management.

This module handles saving optimization results, fitted parameters,
simulation data, and generated figures to organized directories.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class FittedParameters:
    """
    Container for fitted parameter results.

    Attributes:
        parameters: Dictionary of parameter names and fitted values
        units: Dictionary of parameter units
        statistics: Fit statistics (R², RMSE, etc.)
        conditions: Experimental conditions used
        model_type: Type of model used (e.g., "DualMonodLag")
        confidence_intervals: Parameter confidence intervals (optional)
        timestamp: When the fitting was performed
    """
    parameters: Dict[str, float]
    units: Dict[str, str]
    statistics: Dict[str, float]
    conditions: List[str]
    model_type: str
    confidence_intervals: Dict[str, Dict[str, float]] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if self.confidence_intervals is None:
            self.confidence_intervals = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FittedParameters":
        """Create instance from dictionary."""
        return cls(**data)


class ResultsWriter:
    """
    Manages writing results to organized directory structure.

    Directory structure:
        results/
        └── {substrate_name}/
            └── {timestamp}/
                ├── fitted_parameters.json
                ├── model_predictions.csv
                ├── statistics.json
                └── figures/
                    ├── fit_results.png
                    ├── fit_results.pdf
                    └── ...
    """

    def __init__(
        self,
        base_dir: str = "results",
        substrate_name: str = "default",
        create_timestamp_dir: bool = True
    ):
        """
        Initialize the results writer.

        Args:
            base_dir: Base directory for all results
            substrate_name: Name of substrate (creates subdirectory)
            create_timestamp_dir: Whether to create timestamp subdirectory
        """
        self.base_dir = Path(base_dir)
        self.substrate_name = substrate_name

        if create_timestamp_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = self.base_dir / substrate_name / timestamp
        else:
            self.output_dir = self.base_dir / substrate_name

        self.figures_dir = self.output_dir / "figures"

        # Create directories
        self._create_directories()

    def _create_directories(self) -> None:
        """Create output directories if they don't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def save_fitted_parameters(
        self,
        fitted_params: FittedParameters,
        filename: str = "fitted_parameters.json"
    ) -> Path:
        """
        Save fitted parameters to JSON file.

        Args:
            fitted_params: FittedParameters object
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename

        with open(output_path, 'w') as f:
            json.dump(fitted_params.to_dict(), f, indent=2)

        return output_path

    def save_predictions(
        self,
        predictions: pd.DataFrame,
        filename: str = "model_predictions.csv"
    ) -> Path:
        """
        Save model predictions to CSV file.

        Args:
            predictions: DataFrame with model predictions
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        predictions.to_csv(output_path, index=False)
        return output_path

    def save_statistics(
        self,
        statistics: Dict[str, Any],
        filename: str = "statistics.json"
    ) -> Path:
        """
        Save fit statistics to JSON file.

        Args:
            statistics: Dictionary of statistics
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename

        # Convert numpy types for JSON serialization
        serializable = _make_serializable(statistics)

        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)

        return output_path

    def save_figure(
        self,
        figure,
        name: str,
        formats: List[str] = None
    ) -> List[Path]:
        """
        Save a matplotlib figure in multiple formats.

        Args:
            figure: Matplotlib figure object
            name: Base filename (without extension)
            formats: List of formats (default: ['png', 'pdf'])

        Returns:
            List of paths to saved files
        """
        if formats is None:
            formats = ['png', 'pdf']

        saved_paths = []
        for fmt in formats:
            output_path = self.figures_dir / f"{name}.{fmt}"
            figure.savefig(
                output_path,
                format=fmt,
                dpi=300,
                bbox_inches='tight',
                facecolor='white'
            )
            saved_paths.append(output_path)

        return saved_paths

    def save_comparison(
        self,
        comparison_results: Dict[str, Dict[str, Any]],
        filename: str = "model_comparison.json"
    ) -> Path:
        """
        Save model comparison results.

        Args:
            comparison_results: Dictionary with model comparison data
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename

        serializable = _make_serializable(comparison_results)

        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)

        return output_path

    def save_run_info(
        self,
        config: Dict[str, Any],
        data_path: str,
        additional_info: Dict[str, Any] = None
    ) -> Path:
        """
        Save information about the analysis run.

        Args:
            config: Configuration used
            data_path: Path to input data
            additional_info: Additional metadata

        Returns:
            Path to saved file
        """
        run_info = {
            "timestamp": datetime.now().isoformat(),
            "substrate": self.substrate_name,
            "data_file": str(data_path),
            "output_directory": str(self.output_dir),
            "config": _make_serializable(config)
        }

        if additional_info:
            run_info.update(_make_serializable(additional_info))

        output_path = self.output_dir / "run_info.json"

        with open(output_path, 'w') as f:
            json.dump(run_info, f, indent=2)

        return output_path

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of saved files.

        Returns:
            Dictionary with output paths and file counts
        """
        files = list(self.output_dir.glob("*"))
        figures = list(self.figures_dir.glob("*"))

        return {
            "output_directory": str(self.output_dir),
            "total_files": len(files),
            "total_figures": len(figures),
            "files": [str(f.name) for f in files if f.is_file()],
            "figures": [str(f.name) for f in figures if f.is_file()]
        }


def _make_serializable(obj: Any) -> Any:
    """
    Convert numpy types and other non-serializable types for JSON.

    Args:
        obj: Object to convert

    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj


def load_fitted_parameters(file_path: str) -> FittedParameters:
    """
    Load fitted parameters from a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        FittedParameters object
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    return FittedParameters.from_dict(data)


def create_results_summary(
    results_dir: str,
    output_file: str = None
) -> pd.DataFrame:
    """
    Create a summary of all results in a directory.

    Args:
        results_dir: Directory containing results
        output_file: Optional path to save summary CSV

    Returns:
        DataFrame with summary of all runs
    """
    results_dir = Path(results_dir)
    summaries = []

    # Find all run_info.json files
    for run_info_path in results_dir.rglob("run_info.json"):
        with open(run_info_path, 'r') as f:
            run_info = json.load(f)

        # Load fitted parameters if available
        params_path = run_info_path.parent / "fitted_parameters.json"
        if params_path.exists():
            with open(params_path, 'r') as f:
                params = json.load(f)

            summary = {
                "timestamp": run_info.get("timestamp"),
                "substrate": run_info.get("substrate"),
                "model_type": params.get("model_type"),
                "R_squared": params.get("statistics", {}).get("R_squared"),
                "RMSE": params.get("statistics", {}).get("RMSE"),
                "directory": str(run_info_path.parent)
            }
            summaries.append(summary)

    df = pd.DataFrame(summaries)

    if output_file and not df.empty:
        df.to_csv(output_file, index=False)

    return df
