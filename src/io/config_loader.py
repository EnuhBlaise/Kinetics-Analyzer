"""
Configuration file loading and validation.

This module handles loading JSON configuration files that specify
kinetic parameters, bounds, and simulation settings.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional


@dataclass
class SubstrateConfig:
    """
    Configuration for a specific substrate.

    Attributes:
        name: Substrate name (e.g., "xylose")
        molecular_weight: Molecular weight in g/mol
        molecular_formula: Molecular formula (e.g., "C6H12O6")
        unit: Concentration unit (default: "mg/L")
        initial_guesses: Initial parameter values for optimization
        bounds: Parameter bounds as (lower, upper) tuples
        oxygen: Oxygen model settings
        simulation: Simulation settings
    """
    name: str
    molecular_weight: float
    molecular_formula: Optional[str] = None
    unit: str = "mg/L"
    initial_guesses: Dict[str, float] = field(default_factory=dict)
    bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    oxygen: Dict[str, float] = field(default_factory=dict)
    simulation: Dict[str, Any] = field(default_factory=dict)

    def get_parameter_bounds(self, parameter: str) -> Tuple[float, float]:
        """Get bounds for a specific parameter."""
        if parameter not in self.bounds:
            raise KeyError(f"No bounds defined for parameter: {parameter}")
        return tuple(self.bounds[parameter])

    def get_initial_guess(self, parameter: str) -> float:
        """Get initial guess for a specific parameter."""
        if parameter not in self.initial_guesses:
            raise KeyError(f"No initial guess for parameter: {parameter}")
        return self.initial_guesses[parameter]

    def get_all_bounds_as_list(
        self,
        parameter_names: List[str]
    ) -> List[Tuple[float, float]]:
        """
        Get bounds for multiple parameters as a list (for scipy.optimize).

        Args:
            parameter_names: List of parameter names in desired order

        Returns:
            List of (lower, upper) bound tuples
        """
        return [self.get_parameter_bounds(p) for p in parameter_names]

    def get_all_initial_guesses(
        self,
        parameter_names: List[str]
    ) -> List[float]:
        """
        Get initial guesses as a list (for scipy.optimize).

        Args:
            parameter_names: List of parameter names in desired order

        Returns:
            List of initial values
        """
        return [self.get_initial_guess(p) for p in parameter_names]


def load_config(file_path: str) -> SubstrateConfig:
    """
    Load configuration from a JSON file.

    Args:
        file_path: Path to the JSON configuration file

    Returns:
        SubstrateConfig object with loaded settings

    Example:
        >>> config = load_config("config/substrates/xylose.json")
        >>> print(config.molecular_weight)
        150.13
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")

    with open(file_path, 'r') as f:
        raw_config = json.load(f)

    # Validate structure
    validate_config(raw_config)

    # Extract sections
    substrate = raw_config["substrate"]
    initial_guesses = raw_config["initial_guesses"]
    bounds = raw_config["bounds"]
    oxygen = raw_config.get("oxygen", {})
    simulation = raw_config.get("simulation", {})

    # Convert bounds from lists to tuples
    bounds_tuples = {k: tuple(v) for k, v in bounds.items()}

    return SubstrateConfig(
        name=substrate["name"],
        molecular_weight=substrate["molecular_weight"],
        molecular_formula=substrate.get("molecular_formula"),
        unit=substrate.get("unit", "mg/L"),
        initial_guesses=initial_guesses,
        bounds=bounds_tuples,
        oxygen=oxygen,
        simulation=simulation
    )


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration dictionary structure.

    Args:
        config: Raw configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    required_sections = ["substrate", "initial_guesses", "bounds"]

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Config missing required section: '{section}'")

    # Validate substrate section
    substrate = config["substrate"]
    if "name" not in substrate:
        raise ValueError("Config 'substrate' must have 'name'")
    if "molecular_weight" not in substrate:
        raise ValueError("Config 'substrate' must have 'molecular_weight'")

    mw = substrate["molecular_weight"]
    if not isinstance(mw, (int, float)) or mw <= 0:
        raise ValueError(f"Invalid molecular_weight: {mw}")

    # Validate bounds format
    for param, bounds in config["bounds"].items():
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"Bounds for '{param}' must be [lower, upper]")
        if bounds[0] >= bounds[1]:
            raise ValueError(f"Bounds for '{param}': lower must be < upper")

    # Check initial guesses are within bounds
    for param, guess in config["initial_guesses"].items():
        if param in config["bounds"]:
            lower, upper = config["bounds"][param]
            if not (lower <= guess <= upper):
                raise ValueError(
                    f"Initial guess for '{param}' ({guess}) "
                    f"is outside bounds [{lower}, {upper}]"
                )


def save_config(config: SubstrateConfig, file_path: str) -> None:
    """
    Save a configuration to a JSON file.

    Args:
        config: SubstrateConfig object to save
        file_path: Path for the output file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dictionary format
    config_dict = {
        "substrate": {
            "name": config.name,
            "molecular_formula": config.molecular_formula,
            "molecular_weight": config.molecular_weight,
            "unit": config.unit
        },
        "initial_guesses": config.initial_guesses,
        "bounds": {k: list(v) for k, v in config.bounds.items()},
        "oxygen": config.oxygen,
        "simulation": config.simulation
    }

    with open(file_path, 'w') as f:
        json.dump(config_dict, f, indent=2)


def create_default_config(substrate_name: str, molecular_weight: float) -> SubstrateConfig:
    """
    Create a configuration with default parameter values.

    Args:
        substrate_name: Name of the substrate
        molecular_weight: Molecular weight in g/mol

    Returns:
        SubstrateConfig with default kinetic parameters
    """
    return SubstrateConfig(
        name=substrate_name,
        molecular_weight=molecular_weight,
        unit="mg/L",
        initial_guesses={
            "qmax": 2.5,
            "Ks": 400.0,
            "Ki": 25000.0,
            "Y": 0.35,
            "b_decay": 0.01,
            "K_o2": 0.15,
            "Y_o2": 0.8,
            "lag_time": 3.0
        },
        bounds={
            "qmax": (0.1, 10.0),
            "Ks": (10.0, 2000.0),
            "Ki": (50.0, 50000.0),
            "Y": (0.1, 1.0),
            "b_decay": (0.001, 0.2),
            "K_o2": (0.05, 1.0),
            "Y_o2": (0.1, 2.0),
            "lag_time": (0.0, 10.0)
        },
        oxygen={
            "o2_max": 8.0,
            "o2_min": 0.1,
            "reaeration_rate": 15.0,
            "o2_range": 8.0
        },
        simulation={
            "t_final": 5.0,
            "num_points": 10000,
            "time_unit": "days"
        }
    )


def merge_configs(
    base_config: SubstrateConfig,
    override_config: Dict[str, Any]
) -> SubstrateConfig:
    """
    Merge override values into a base configuration.

    Args:
        base_config: Base configuration
        override_config: Dictionary of values to override

    Returns:
        New SubstrateConfig with merged values
    """
    # Create copies of mutable attributes
    initial_guesses = dict(base_config.initial_guesses)
    bounds = dict(base_config.bounds)
    oxygen = dict(base_config.oxygen)
    simulation = dict(base_config.simulation)

    # Apply overrides
    if "initial_guesses" in override_config:
        initial_guesses.update(override_config["initial_guesses"])

    if "bounds" in override_config:
        for k, v in override_config["bounds"].items():
            bounds[k] = tuple(v) if isinstance(v, list) else v

    if "oxygen" in override_config:
        oxygen.update(override_config["oxygen"])

    if "simulation" in override_config:
        simulation.update(override_config["simulation"])

    return SubstrateConfig(
        name=override_config.get("substrate", {}).get("name", base_config.name),
        molecular_weight=override_config.get("substrate", {}).get(
            "molecular_weight", base_config.molecular_weight
        ),
        unit=override_config.get("substrate", {}).get("unit", base_config.unit),
        initial_guesses=initial_guesses,
        bounds=bounds,
        oxygen=oxygen,
        simulation=simulation
    )
