"""
Data loading utilities for experimental data.

This module handles loading and preprocessing of experimental data
from CSV files for kinetic parameter estimation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any


@dataclass
class ExperimentalData:
    """
    Container for experimental data with metadata.

    Attributes:
        data: The main DataFrame containing all measurements
        time_column: Name of the time column
        substrate_columns: List of substrate concentration columns
        biomass_columns: List of biomass concentration columns
        conditions: List of experimental conditions (e.g., ['5mM', '10mM'])
        substrate_name: Name of the substrate
        time_unit: Unit of time (e.g., 'days', 'hours')
    """
    data: pd.DataFrame
    time_column: str
    substrate_columns: List[str]
    biomass_columns: List[str]
    conditions: List[str]
    substrate_name: str = "Substrate"
    time_unit: str = "days"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_condition_data(
        self,
        condition: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get time, substrate, and biomass data for a specific condition.
        
        Automatically removes rows with missing (NaN) values for either
        substrate or biomass measurements.

        Args:
            condition: Condition label (e.g., '5mM')

        Returns:
            Tuple of (time, substrate, biomass) arrays with NaN values removed
        """
        time = self.data[self.time_column].values

        # Find matching columns
        substrate_col = None
        biomass_col = None

        for col in self.substrate_columns:
            # Match condition at the START of the column name (e.g., "5mM_" not "15mM_")
            if col.startswith(f"{condition}_") or col.startswith(f"{condition} "):
                substrate_col = col
                break

        for col in self.biomass_columns:
            # Match condition at the START of the column name
            if col.startswith(f"{condition}_") or col.startswith(f"{condition} "):
                biomass_col = col
                break

        if substrate_col is None:
            raise ValueError(f"No substrate column found for condition: {condition}")
        if biomass_col is None:
            raise ValueError(f"No biomass column found for condition: {condition}")

        substrate = self.data[substrate_col].values
        biomass = self.data[biomass_col].values
        
        # Remove rows with NaN values in either substrate or biomass
        valid_mask = ~(np.isnan(substrate) | np.isnan(biomass))
        time = time[valid_mask]
        substrate = substrate[valid_mask]
        biomass = biomass[valid_mask]
        
        if len(time) == 0:
            raise ValueError(f"No valid data points for condition: {condition} (all values are NaN)")

        return time, substrate, biomass

    def get_initial_conditions(self, condition: str) -> Tuple[float, float]:
        """
        Get initial substrate and biomass for a condition.

        Args:
            condition: Condition label

        Returns:
            Tuple of (initial_substrate, initial_biomass)
        """
        time, substrate, biomass = self.get_condition_data(condition)
        return substrate[0], biomass[0]

    @property
    def n_conditions(self) -> int:
        """Number of experimental conditions."""
        return len(self.conditions)

    @property
    def n_timepoints(self) -> int:
        """Number of time points."""
        return len(self.data)


def load_experimental_data(
    file_path: str,
    substrate_name: str = None,
    time_column: str = None,
    biomass_conversion: float = None,
    od_to_cells_factor: float = 83.0
) -> ExperimentalData:
    """
    Load experimental data from a CSV file.

    This function automatically detects column types and converts units
    as needed. It handles common formats from laboratory experiments.

    Args:
        file_path: Path to the CSV file
        substrate_name: Name of substrate (auto-detected if None)
        time_column: Name of time column (auto-detected if None)
        biomass_conversion: Custom OD to mg cells/L factor
        od_to_cells_factor: Default factor for OD to mgCells/L (83.0)

    Returns:
        ExperimentalData object with processed data

    Example:
        >>> data = load_experimental_data("experiment.csv", substrate_name="Glucose")
        >>> time, sub, bio = data.get_condition_data("5mM")
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # Load raw data
    data = pd.read_csv(file_path)

    # Detect time column
    if time_column is None:
        time_column = _detect_time_column(data)

    # Standardize time column name
    if time_column != "Time (days)":
        if "Time (days)" not in data.columns:
            data = data.rename(columns={time_column: "Time (days)"})
            time_column = "Time (days)"

    # Detect substrate name
    if substrate_name is None:
        substrate_name = _detect_substrate_name(data)

    # Find substrate and biomass columns
    substrate_columns = _find_substrate_columns(data, substrate_name)

    if not substrate_columns:
        raise ValueError(
            f"No substrate columns found for '{substrate_name}' in data file. "
            f"Available columns: {list(data.columns)}\n"
            f"Ensure your data columns contain the substrate name '{substrate_name}' "
            f"or update the config/substrate name."
        )
    biomass_columns = _find_biomass_columns(data)

    # Convert OD to mgCells/L if needed
    conversion_factor = biomass_conversion or od_to_cells_factor
    data, biomass_columns = _convert_od_columns(data, biomass_columns, conversion_factor)

    # Extract condition labels
    conditions = _extract_conditions(substrate_columns)

    return ExperimentalData(
        data=data,
        time_column=time_column,
        substrate_columns=substrate_columns,
        biomass_columns=biomass_columns,
        conditions=conditions,
        substrate_name=substrate_name,
        time_unit="days",
        metadata={
            "source_file": str(file_path),
            "od_conversion_factor": conversion_factor
        }
    )


def _detect_time_column(data: pd.DataFrame) -> str:
    """Detect the time column in the data."""
    candidates = [
        "Time (days)", "Time (Days)", "time (days)",
        "Time", "time", "t",
        "Time (hours)", "Time (Hours)",
        "Hours", "hours",
        "Days", "days"
    ]

    for col in candidates:
        if col in data.columns:
            return col

    # Look for columns containing 'time'
    time_cols = [c for c in data.columns if 'time' in c.lower()]
    if time_cols:
        return time_cols[0]

    raise ValueError(
        f"Could not detect time column. Columns: {list(data.columns)}"
    )


def _detect_substrate_name(data: pd.DataFrame) -> str:
    """Detect the substrate name from column names."""
    common_substrates = ["Glucose", "Xylose", "Fructose", "Sucrose", "Lactose"]

    for substrate in common_substrates:
        if any(substrate in col for col in data.columns):
            return substrate

    # Default to generic name
    return "Substrate"


def _find_substrate_columns(data: pd.DataFrame, substrate_name: str) -> List[str]:
    """Find all substrate concentration columns."""
    columns = []

    for col in data.columns:
        # Look for columns with substrate name and concentration units
        if substrate_name in col and ("mg/L" in col or "mM" in col):
            columns.append(col)

    if not columns:
        # Try broader search
        columns = [c for c in data.columns if substrate_name in c and 'Biomass' not in c]

    return sorted(columns)


def _find_biomass_columns(data: pd.DataFrame) -> List[str]:
    """Find all biomass columns."""
    columns = []

    for col in data.columns:
        if "Biomass" in col:
            columns.append(col)

    return sorted(columns)


def _convert_od_columns(
    data: pd.DataFrame,
    biomass_columns: List[str],
    conversion_factor: float
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convert OD columns to mgCells/L and return updated column list.

    Args:
        data: DataFrame with biomass data
        biomass_columns: List of biomass column names
        conversion_factor: OD to mgCells/L conversion factor

    Returns:
        Tuple of (modified DataFrame, updated biomass column list)
    """
    new_columns = []

    for col in biomass_columns:
        if "(OD)" in col:
            # Convert OD to mgCells/L
            new_col = col.replace("(OD)", "(mgCells/L)")
            data[new_col] = data[col] * conversion_factor
            new_columns.append(new_col)
        elif "(mgCells/L)" in col or "mgCells" in col:
            # Already in correct units
            new_columns.append(col)
        else:
            # Assume mg cells/L
            new_columns.append(col)

    return data, new_columns


def _extract_conditions(substrate_columns: List[str]) -> List[str]:
    """Extract condition labels from substrate column names.

    Returns conditions sorted numerically so that e.g. 5mM < 10mM < 15mM < 20mM
    instead of the default lexicographic order.
    """
    conditions = []

    for col in substrate_columns:
        # Extract condition prefix (e.g., "5mM" from "5mM_Glucose (mg/L)")
        parts = col.split("_")
        if parts:
            condition = parts[0]
            if condition not in conditions:
                conditions.append(condition)

    # Sort numerically by the leading number in each condition label
    conditions.sort(key=lambda x: float(''.join(c for c in x if c.isdigit() or c == '.') or '0'))

    return conditions


def validate_data_format(
    data: pd.DataFrame,
    substrate_name: str,
    conditions: List[str]
) -> bool:
    """
    Validate that data has expected columns for all conditions.

    Args:
        data: DataFrame to validate
        substrate_name: Expected substrate name
        conditions: List of expected conditions

    Returns:
        True if valid, raises ValueError otherwise
    """
    for condition in conditions:
        substrate_col = f"{condition}_{substrate_name} (mg/L)"
        biomass_col = f"{condition}_Biomass (mgCells/L)"

        if substrate_col not in data.columns:
            raise ValueError(f"Missing column: {substrate_col}")
        if biomass_col not in data.columns:
            raise ValueError(f"Missing column: {biomass_col}")

    return True


def create_condition_dataframe(
    time: np.ndarray,
    substrate: np.ndarray,
    biomass: np.ndarray,
    oxygen: np.ndarray = None,
    condition_label: str = ""
) -> pd.DataFrame:
    """
    Create a standard format DataFrame for a single condition.

    Args:
        time: Time array
        substrate: Substrate concentration array
        biomass: Biomass concentration array
        oxygen: Optional oxygen concentration array
        condition_label: Label for this condition

    Returns:
        DataFrame with standardized columns
    """
    data = {
        "Time (days)": time,
        "Substrate (mg/L)": substrate,
        "Biomass (mgCells/L)": biomass,
    }

    if oxygen is not None:
        data["Oxygen (mg/L)"] = oxygen

    df = pd.DataFrame(data)

    if condition_label:
        df["Condition"] = condition_label

    return df
