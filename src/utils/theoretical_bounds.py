"""
Theoretical parameter bounds estimation from substrate stoichiometry.

Computes thermodynamic/stoichiometric ceilings for kinetic parameters
based on the molecular formula and molecular weight of the substrate.
These ceilings help constrain the optimiser search space and flag
physically implausible fitted values.

Key calculations
----------------
1. **Degree of reduction (γ_s)** — electron equivalents per C-mol of
   substrate, measuring how "reduced" the substrate is.

2. **Theoretical maximum yield (Y_max)** — the stoichiometric ceiling
   for biomass yield (mg cells / mg substrate) assuming all available
   electrons are directed to anabolism (no catabolism).  A practical
   ceiling (ε ≈ 0.6) accounts for maintenance energy.

3. **Theoretical oxygen demand (ThOD)** — mg O₂ required to fully
   mineralise 1 mg substrate to CO₂ and H₂O.

4. **Y_o2 ceiling** — maximum biomass produced per mg O₂ consumed,
   derived from the Y and ThOD relationship.

5. **Carbon fraction** — mass fraction of carbon in the substrate,
   which limits the maximum carbon that can be assimilated.

References
----------
- Heijnen, J.J. & Kleerebezem, R. (2010). *Bioenergetics of Microbial
  Growth.* Encyclopedia of Industrial Biotechnology.
- Rittmann, B.E. & McCarty, P.L. (2001). *Environmental Biotechnology:
  Principles and Applications.* McGraw-Hill.
- VanBriesen, J.M. (2002). *Evaluation of methods to predict bacterial
  yield using thermodynamics.* Biodegradation 13, 171–190.
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


# ── Constants ──────────────────────────────────────────────────────────
# Typical biomass formula: CH₁.₈O₀.₅N₀.₂  (Rittmann & McCarty 2001)
BIOMASS_C = 1
BIOMASS_H = 1.8
BIOMASS_O = 0.5
BIOMASS_N = 0.2
# Molecular weight per C-mol of biomass (g / C-mol)
MW_BIOMASS_CMOL = (
    12.011 * BIOMASS_C
    + 1.008 * BIOMASS_H
    + 15.999 * BIOMASS_O
    + 14.007 * BIOMASS_N
)  # ≈ 24.6 g/C-mol
# Degree of reduction of biomass (per C-mol)
GAMMA_BIOMASS = (4 * BIOMASS_C + BIOMASS_H - 2 * BIOMASS_O - 3 * BIOMASS_N)
# ≈ 4.2

# Atomic weights
ATOMIC_WEIGHTS = {
    "C": 12.011,
    "H": 1.008,
    "O": 15.999,
    "N": 14.007,
    "S": 32.065,
    "P": 30.974,
}

# Practical anabolic efficiency ceiling (fraction of electrons to biomass)
# Typical range for aerobic heterotrophs: 0.40–0.72  (Heijnen 2010)
EPSILON_PRACTICAL = 0.60


# ── Formula parser ─────────────────────────────────────────────────────

def parse_formula(formula: str) -> Dict[str, int]:
    """
    Parse a molecular formula string into element counts.

    Handles simple formulas with implicit "1" subscripts.
    Does not handle nested parentheses or hydrates.

    Parameters
    ----------
    formula : str
        e.g. ``"C6H12O6"``, ``"C9H8O3"``, ``"CH4"``

    Returns
    -------
    dict
        ``{"C": 6, "H": 12, "O": 6}``

    Raises
    ------
    ValueError
        If the formula cannot be parsed.
    """
    pattern = re.compile(r"([A-Z][a-z]?)(\d*)")
    matches = pattern.findall(formula)
    if not matches or all(m[0] == "" for m in matches):
        raise ValueError(f"Cannot parse molecular formula: '{formula}'")

    elements: Dict[str, int] = {}
    for element, count_str in matches:
        if element == "":
            continue
        count = int(count_str) if count_str else 1
        elements[element] = elements.get(element, 0) + count
    return elements


def verify_molecular_weight(
    elements: Dict[str, int],
    stated_mw: float,
    tolerance: float = 1.0,
) -> Tuple[float, bool]:
    """
    Compute MW from the elemental composition and compare to the stated
    molecular weight.

    Parameters
    ----------
    elements : dict
        Element counts from :func:`parse_formula`.
    stated_mw : float
        Molecular weight from the config file (g/mol).
    tolerance : float
        Acceptable absolute difference (g/mol).

    Returns
    -------
    computed_mw : float
        Molecular weight calculated from the formula.
    consistent : bool
        True if ``|computed - stated| <= tolerance``.
    """
    computed = sum(
        ATOMIC_WEIGHTS.get(el, 0) * count for el, count in elements.items()
    )
    return computed, abs(computed - stated_mw) <= tolerance


# ── Core calculations ──────────────────────────────────────────────────

def degree_of_reduction(elements: Dict[str, int]) -> float:
    """
    Degree of reduction per C-mol of substrate.

    γ_s = (4·C + H − 2·O − 3·N) / C

    A higher γ_s means the substrate is more reduced and carries more
    electron equivalents per carbon.

    Parameters
    ----------
    elements : dict
        Element counts (must contain ``"C"``).

    Returns
    -------
    float
        γ_s (dimensionless, per C-mol).
    """
    C = elements.get("C", 0)
    H = elements.get("H", 0)
    O = elements.get("O", 0)
    N = elements.get("N", 0)

    if C == 0:
        raise ValueError("Formula must contain carbon to compute γ_s.")

    return (4 * C + H - 2 * O - 3 * N) / C


def theoretical_yield_max(
    elements: Dict[str, int],
    molecular_weight: float,
) -> Dict[str, float]:
    """
    Compute thermodynamic and practical yield ceilings.

    Parameters
    ----------
    elements : dict
        Element counts.
    molecular_weight : float
        Substrate molecular weight (g/mol).

    Returns
    -------
    dict with keys
        ``gamma_s``          – degree of reduction of substrate
        ``gamma_bio``        – degree of reduction of biomass (4.2)
        ``Y_max_Cmol``       – thermodynamic ceiling (C-mol bio / C-mol sub)
        ``Y_practical_Cmol`` – practical ceiling at ε = 0.60
        ``Y_max_mg``         – thermodynamic ceiling (mg cells / mg substrate)
        ``Y_practical_mg``   – practical ceiling (mg cells / mg substrate)
        ``carbon_fraction``  – mass fraction of C in substrate
    """
    n_C = elements.get("C", 0)
    if n_C == 0:
        raise ValueError("Formula must contain carbon.")

    gamma_s = degree_of_reduction(elements)

    # ── C-mol basis ────────────────────────────────────────────────
    # Thermodynamic ceiling: all available electrons → biomass
    Y_max_Cmol = gamma_s / GAMMA_BIOMASS  # C-mol bio / C-mol sub
    # Practical ceiling (ε fraction of electrons to growth)
    Y_practical_Cmol = EPSILON_PRACTICAL * Y_max_Cmol

    # ── Mass basis conversion ─────────────────────────────────────
    # MW per C-mol of substrate
    mw_sub_per_Cmol = molecular_weight / n_C
    conversion = MW_BIOMASS_CMOL / mw_sub_per_Cmol
    Y_max_mg = Y_max_Cmol * conversion
    Y_practical_mg = Y_practical_Cmol * conversion

    # ── Carbon fraction ───────────────────────────────────────────
    carbon_fraction = (n_C * ATOMIC_WEIGHTS["C"]) / molecular_weight

    return {
        "gamma_s": round(gamma_s, 4),
        "gamma_bio": round(GAMMA_BIOMASS, 4),
        "Y_max_Cmol": round(Y_max_Cmol, 4),
        "Y_practical_Cmol": round(Y_practical_Cmol, 4),
        "Y_max_mg": round(Y_max_mg, 4),
        "Y_practical_mg": round(Y_practical_mg, 4),
        "carbon_fraction": round(carbon_fraction, 4),
    }


def theoretical_oxygen_demand(
    elements: Dict[str, int],
    molecular_weight: float,
) -> Dict[str, float]:
    """
    Theoretical oxygen demand (ThOD) for complete mineralisation.

    C_a H_b O_c  +  (a + b/4 − c/2) O₂  →  a CO₂  +  (b/2) H₂O

    For substrates containing nitrogen:
    C_a H_b O_c N_d  +  (a + b/4 − c/2 + 5d/4) O₂
        →  a CO₂  +  (b/2 − 3d/2) H₂O  +  d HNO₃

    Parameters
    ----------
    elements : dict
        Element counts.
    molecular_weight : float
        Substrate molecular weight (g/mol).

    Returns
    -------
    dict with keys
        ``O2_moles``   – moles O₂ per mole substrate
        ``ThOD_mg_mg`` – mg O₂ per mg substrate
        ``ThOD_mg_mmol`` – mg O₂ per mmol substrate
    """
    a = elements.get("C", 0)
    b = elements.get("H", 0)
    c = elements.get("O", 0)
    d = elements.get("N", 0)

    # Stoichiometric O₂ requirement (moles O₂ / mole substrate)
    O2_moles = a + b / 4.0 - c / 2.0 + 5.0 * d / 4.0

    # Mass-based ThOD
    ThOD_mg_mg = O2_moles * 32.0 / molecular_weight     # mg O₂ / mg substrate
    ThOD_mg_mmol = O2_moles * 32.0                        # mg O₂ / mmol substrate

    return {
        "O2_moles": round(O2_moles, 4),
        "ThOD_mg_mg": round(ThOD_mg_mg, 4),
        "ThOD_mg_mmol": round(ThOD_mg_mmol, 4),
    }


def yield_oxygen_ceiling(
    Y_practical_mg: float,
    ThOD_mg_mg: float,
    molecular_weight: float,
    elements: Dict[str, int],
) -> Dict[str, float]:
    """
    Estimate the ceiling for Y_o2 (mg cells / mg O₂).

    When a fraction *f* of substrate carbon is assimilated into biomass,
    the remaining fraction *(1 − f)* is catabolised (oxidised).  The O₂
    consumed equals ``ThOD × (1 − f_carbon)``.

    Y_o2 = Y / [ThOD × (1 − f_carbon)]

    where ``f_carbon = Y × (MW_sub / (n_C × MW_bio_Cmol))``.

    Parameters
    ----------
    Y_practical_mg : float
        Practical yield ceiling (mg cells / mg substrate).
    ThOD_mg_mg : float
        Theoretical oxygen demand (mg O₂ / mg substrate).
    molecular_weight : float
        Substrate MW (g/mol).
    elements : dict
        Element counts.

    Returns
    -------
    dict with keys
        ``f_carbon``   – fraction of substrate C going to biomass
        ``Y_o2_max``   – ceiling for Y_o2 (mg cells / mg O₂)
    """
    n_C = elements.get("C", 0)
    if n_C == 0 or ThOD_mg_mg <= 0:
        return {"f_carbon": 0.0, "Y_o2_max": float("inf")}

    # Fraction of substrate carbon assimilated
    f_carbon = Y_practical_mg * molecular_weight / (n_C * MW_BIOMASS_CMOL)
    f_carbon = min(f_carbon, 0.99)  # guard against division by zero

    # O₂ consumed only for the catabolised fraction
    O2_per_mg_sub = ThOD_mg_mg * (1 - f_carbon)
    if O2_per_mg_sub <= 0:
        return {"f_carbon": round(f_carbon, 4), "Y_o2_max": float("inf")}

    Y_o2_max = Y_practical_mg / O2_per_mg_sub

    return {
        "f_carbon": round(f_carbon, 4),
        "Y_o2_max": round(Y_o2_max, 4),
    }


def qmax_heuristic(
    elements: Dict[str, int],
    molecular_weight: float,
) -> Dict[str, float]:
    """
    Heuristic ceiling for qmax (1/day) based on substrate properties.

    Rationale:
    - Simple sugars (glucose, xylose): qmax typically 5–25 d⁻¹
    - Phenolic acids / aromatics: qmax typically 1–10 d⁻¹
    - General heuristic: lighter, more reduced substrates are metabolised
      faster.  We use the degree of reduction and an empirical scaling.

    The number is an *order-of-magnitude guide*, NOT a hard physical limit.

    Parameters
    ----------
    elements : dict
        Element counts.
    molecular_weight : float
        Substrate MW (g/mol).

    Returns
    -------
    dict
        ``qmax_ceiling`` – suggested upper bound (1/day)
        ``substrate_class`` – "simple_sugar" | "aromatic" | "other"
    """
    n_C = elements.get("C", 0)
    gamma_s = degree_of_reduction(elements)

    # Classify substrate
    # Simple sugars: γ ≈ 4.0, MW < 200, high O:C ratio
    # Aromatics: γ > 4, lower H:C ratio (degree of unsaturation > 2)
    n_H = elements.get("H", 0)
    n_O = elements.get("O", 0)
    degree_unsat = (2 * n_C + 2 - n_H) / 2  # index of hydrogen deficiency

    if degree_unsat >= 4 and n_C >= 6:
        substrate_class = "aromatic"
        # Aromatic compounds: slower uptake, more inhibition-prone
        qmax_ceiling = 10.0
    elif gamma_s <= 4.1 and molecular_weight <= 200:
        substrate_class = "simple_sugar"
        qmax_ceiling = 30.0
    else:
        substrate_class = "other"
        qmax_ceiling = 15.0

    return {
        "qmax_ceiling": qmax_ceiling,
        "substrate_class": substrate_class,
        "degree_of_unsaturation": round(degree_unsat, 1),
    }


# ── High-level report ─────────────────────────────────────────────────

@dataclass
class TheoreticalBoundsReport:
    """Full stoichiometric analysis for one substrate."""

    substrate_name: str
    molecular_formula: str
    molecular_weight: float
    elements: Dict[str, int]
    mw_computed: float
    mw_consistent: bool

    # Core quantities
    gamma_s: float
    carbon_fraction: float

    # Yield
    Y_max_Cmol: float
    Y_practical_Cmol: float
    Y_max_mg: float
    Y_practical_mg: float

    # Oxygen
    ThOD_mg_mg: float
    ThOD_mg_mmol: float
    O2_moles_per_mol_sub: float

    # Y_o2
    f_carbon: float
    Y_o2_max: float

    # qmax heuristic
    qmax_ceiling: float
    substrate_class: str
    degree_of_unsaturation: float

    # Suggested bounds
    suggested_bounds: Dict[str, Tuple[float, float]] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "substrate_name": self.substrate_name,
            "molecular_formula": self.molecular_formula,
            "molecular_weight": self.molecular_weight,
            "molecular_weight_computed": round(self.mw_computed, 2),
            "mw_consistent": self.mw_consistent,
            "elements": self.elements,
            "gamma_s": self.gamma_s,
            "gamma_biomass": round(GAMMA_BIOMASS, 4),
            "carbon_fraction": self.carbon_fraction,
            "yield": {
                "Y_max_Cmol": self.Y_max_Cmol,
                "Y_practical_Cmol": self.Y_practical_Cmol,
                "Y_max_mg": self.Y_max_mg,
                "Y_practical_mg": self.Y_practical_mg,
                "unit": "mg_cells / mg_substrate",
                "note": (
                    f"Practical ceiling assumes ε = {EPSILON_PRACTICAL:.0%} "
                    f"of electrons directed to anabolism."
                ),
            },
            "oxygen_demand": {
                "O2_moles_per_mol_substrate": self.O2_moles_per_mol_sub,
                "ThOD_mg_per_mg_substrate": self.ThOD_mg_mg,
                "ThOD_mg_per_mmol_substrate": self.ThOD_mg_mmol,
            },
            "Y_o2": {
                "f_carbon_to_biomass": self.f_carbon,
                "Y_o2_ceiling": self.Y_o2_max,
                "unit": "mg_cells / mg_O2",
            },
            "qmax": {
                "qmax_ceiling": self.qmax_ceiling,
                "substrate_class": self.substrate_class,
                "degree_of_unsaturation": self.degree_of_unsaturation,
                "unit": "1/day",
            },
            "suggested_bounds": {
                k: list(v) for k, v in self.suggested_bounds.items()
            },
        }

    def summary_text(self) -> str:
        """Human-readable summary string."""
        lines = [
            f"{'═' * 60}",
            f"  Theoretical Parameter Bounds — {self.substrate_name}",
            f"{'═' * 60}",
            f"",
            f"  Formula        : {self.molecular_formula}",
            f"  MW (stated)    : {self.molecular_weight:.2f} g/mol",
            f"  MW (computed)  : {self.mw_computed:.2f} g/mol"
            f"  {'✓' if self.mw_consistent else '⚠ MISMATCH'}",
            f"  Carbon fraction: {self.carbon_fraction:.1%}",
            f"",
            f"  ── Degree of Reduction ──────────────────────────",
            f"  γ_substrate    : {self.gamma_s:.4f}  (per C-mol)",
            f"  γ_biomass      : {GAMMA_BIOMASS:.4f}  (CH₁.₈O₀.₅N₀.₂)",
            f"",
            f"  ── Yield Ceiling (Y) ────────────────────────────",
            f"  Thermodynamic  : {self.Y_max_Cmol:.4f} C-mol/C-mol"
            f"  →  {self.Y_max_mg:.4f} mg/mg",
            f"  Practical (ε={EPSILON_PRACTICAL:.0%}): "
            f"{self.Y_practical_Cmol:.4f} C-mol/C-mol"
            f"  →  {self.Y_practical_mg:.4f} mg/mg",
            f"",
            f"  ── Theoretical O₂ Demand (ThOD) ─────────────────",
            f"  {self.O2_moles_per_mol_sub:.2f} mol O₂ / mol substrate",
            f"  {self.ThOD_mg_mg:.4f} mg O₂ / mg substrate",
            f"",
            f"  ── Y_o2 Ceiling ────────────────────────────────",
            f"  Carbon to biomass (f): {self.f_carbon:.4f}",
            f"  Y_o2 ceiling   : {self.Y_o2_max:.4f} mg cells / mg O₂",
            f"",
            f"  ── qmax Heuristic ──────────────────────────────",
            f"  Substrate class: {self.substrate_class}",
            f"  Degree of unsaturation: {self.degree_of_unsaturation}",
            f"  qmax ceiling   : {self.qmax_ceiling:.1f} 1/day",
            f"",
            f"  ── Suggested Parameter Bounds ───────────────────",
        ]
        for param, (lo, hi) in self.suggested_bounds.items():
            lines.append(f"  {param:12s} : [{lo:.6g}, {hi:.6g}]")
        lines.append(f"{'═' * 60}")
        return "\n".join(lines)


def compute_bounds_report(
    substrate_name: str,
    molecular_formula: str,
    molecular_weight: float,
) -> TheoreticalBoundsReport:
    """
    Run all stoichiometric calculations and return a full report.

    Parameters
    ----------
    substrate_name : str
    molecular_formula : str
        e.g. ``"C6H12O6"``
    molecular_weight : float
        g/mol

    Returns
    -------
    TheoreticalBoundsReport
    """
    elements = parse_formula(molecular_formula)
    mw_computed, mw_ok = verify_molecular_weight(elements, molecular_weight)

    yield_info = theoretical_yield_max(elements, molecular_weight)
    thod_info = theoretical_oxygen_demand(elements, molecular_weight)
    yo2_info = yield_oxygen_ceiling(
        yield_info["Y_practical_mg"],
        thod_info["ThOD_mg_mg"],
        molecular_weight,
        elements,
    )
    qmax_info = qmax_heuristic(elements, molecular_weight)

    # ── Assemble suggested bounds ──────────────────────────────────
    # These are stoichiometrically-informed defaults; users should
    # tighten them with experimental knowledge.
    Y_upper = min(yield_info["Y_practical_mg"] * 1.2, yield_info["Y_max_mg"])
    Y_o2_upper = yo2_info["Y_o2_max"]
    if np.isinf(Y_o2_upper):
        Y_o2_upper = 5.0  # fallback

    suggested_bounds = {
        "Y": (0.01, round(Y_upper, 4)),
        "Y_o2": (0.05, round(Y_o2_upper * 1.1, 4)),
        "qmax": (0.05, round(qmax_info["qmax_ceiling"], 2)),
        "b_decay": (0.0001, 0.20),       # literature range
        "Ks": (0.01, 2000.0),             # very substrate-dependent
        "Ki": (1.0, 50000.0),             # very substrate-dependent
        "K_o2": (0.01, 2.0),              # mg/L O₂
        "lag_time": (0.0, 10.0),          # days
    }

    return TheoreticalBoundsReport(
        substrate_name=substrate_name,
        molecular_formula=molecular_formula,
        molecular_weight=molecular_weight,
        elements=elements,
        mw_computed=mw_computed,
        mw_consistent=mw_ok,
        gamma_s=yield_info["gamma_s"],
        carbon_fraction=yield_info["carbon_fraction"],
        Y_max_Cmol=yield_info["Y_max_Cmol"],
        Y_practical_Cmol=yield_info["Y_practical_Cmol"],
        Y_max_mg=yield_info["Y_max_mg"],
        Y_practical_mg=yield_info["Y_practical_mg"],
        ThOD_mg_mg=thod_info["ThOD_mg_mg"],
        ThOD_mg_mmol=thod_info["ThOD_mg_mmol"],
        O2_moles_per_mol_sub=thod_info["O2_moles"],
        f_carbon=yo2_info["f_carbon"],
        Y_o2_max=yo2_info["Y_o2_max"],
        qmax_ceiling=qmax_info["qmax_ceiling"],
        substrate_class=qmax_info["substrate_class"],
        degree_of_unsaturation=qmax_info["degree_of_unsaturation"],
        suggested_bounds=suggested_bounds,
    )


def compute_from_config(config_path: str) -> TheoreticalBoundsReport:
    """
    Load a substrate config file and compute theoretical bounds.

    Parameters
    ----------
    config_path : str
        Path to a substrate JSON configuration file.

    Returns
    -------
    TheoreticalBoundsReport

    Raises
    ------
    ValueError
        If the config file does not contain a molecular_formula.
    """
    path = Path(config_path)
    with open(path, "r") as f:
        raw = json.load(f)

    substrate = raw.get("substrate", {})
    name = substrate.get("name", path.stem)
    formula = substrate.get("molecular_formula")
    mw = substrate.get("molecular_weight")

    if formula is None:
        raise ValueError(
            f"Config file '{config_path}' does not contain "
            f"'substrate.molecular_formula'.  "
            f"Add it (e.g. \"molecular_formula\": \"C6H12O6\") and retry."
        )
    if mw is None:
        raise ValueError(
            f"Config file '{config_path}' does not contain "
            f"'substrate.molecular_weight'."
        )

    return compute_bounds_report(name, formula, mw)


def compare_with_current_bounds(
    report: TheoreticalBoundsReport,
    config_path: str,
) -> str:
    """
    Compare current config bounds against theoretical ceilings.

    Returns a human-readable comparison highlighting any bounds that
    exceed the stoichiometric ceiling.

    Parameters
    ----------
    report : TheoreticalBoundsReport
    config_path : str
        Path to the config JSON.

    Returns
    -------
    str
    """
    path = Path(config_path)
    with open(path, "r") as f:
        raw = json.load(f)
    current_bounds = raw.get("bounds", {})
    current_guesses = raw.get("initial_guesses", {})

    lines = [
        "",
        f"  ── Comparison: Current vs Theoretical ({report.substrate_name}) ──",
        f"  {'Parameter':12s}  {'Current Bounds':>22s}  {'Theoretical':>22s}  {'Status':>8s}",
        f"  {'─' * 70}",
    ]

    for param, (theo_lo, theo_hi) in report.suggested_bounds.items():
        if param in current_bounds:
            cur_lo, cur_hi = current_bounds[param]
            # Check if current upper exceeds theoretical ceiling
            if cur_hi > theo_hi * 1.5:
                status = "⚠ HIGH"
            elif cur_hi < theo_hi * 0.3:
                status = "⚠ LOW"
            else:
                status = "✓ OK"
            lines.append(
                f"  {param:12s}  [{cur_lo:>9.4g}, {cur_hi:>9.4g}]  "
                f"[{theo_lo:>9.4g}, {theo_hi:>9.4g}]  {status}"
            )

            # Check initial guess
            if param in current_guesses:
                guess = current_guesses[param]
                if guess > theo_hi:
                    lines.append(
                        f"  {'':12s}  ↳ initial guess {guess:.4g} "
                        f"exceeds theoretical ceiling {theo_hi:.4g}"
                    )
        else:
            lines.append(
                f"  {param:12s}  {'(not set)':>22s}  "
                f"[{theo_lo:>9.4g}, {theo_hi:>9.4g}]"
            )

    return "\n".join(lines)
