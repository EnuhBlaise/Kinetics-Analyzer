"""
Tests for theoretical parameter bounds estimation.

Tests cover:
- Formula parsing and MW verification
- Degree of reduction calculation
- Theoretical yield (Y) ceilings
- Theoretical oxygen demand (ThOD)
- Y_o2 ceiling derivation
- qmax heuristic classification
- Full report generation
- Config file integration
"""

import json
import math
import tempfile
from pathlib import Path

import pytest

from src.utils.theoretical_bounds import (
    parse_formula,
    verify_molecular_weight,
    degree_of_reduction,
    theoretical_yield_max,
    theoretical_oxygen_demand,
    yield_oxygen_ceiling,
    qmax_heuristic,
    compute_bounds_report,
    compute_from_config,
    compare_with_current_bounds,
    TheoreticalBoundsReport,
    GAMMA_BIOMASS,
    MW_BIOMASS_CMOL,
)


# ══════════════════════════════════════════════════════════════════════
#  Formula Parsing
# ══════════════════════════════════════════════════════════════════════

class TestParseFormula:
    """Tests for molecular formula parsing."""

    def test_glucose(self):
        result = parse_formula("C6H12O6")
        assert result == {"C": 6, "H": 12, "O": 6}

    def test_xylose(self):
        result = parse_formula("C5H10O5")
        assert result == {"C": 5, "H": 10, "O": 5}

    def test_methane(self):
        """CH4 — implicit '1' for carbon."""
        result = parse_formula("CH4")
        assert result == {"C": 1, "H": 4}

    def test_coumaric_acid(self):
        result = parse_formula("C9H8O3")
        assert result == {"C": 9, "H": 8, "O": 3}

    def test_with_nitrogen(self):
        result = parse_formula("C2H5NO2")  # glycine
        assert result == {"C": 2, "H": 5, "N": 1, "O": 2}

    def test_invalid_formula_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_formula("")

    def test_single_element(self):
        result = parse_formula("C60")  # fullerene
        assert result == {"C": 60}


class TestVerifyMolecularWeight:
    """Tests for MW verification."""

    def test_glucose_consistent(self):
        elements = {"C": 6, "H": 12, "O": 6}
        computed, ok = verify_molecular_weight(elements, 180.16)
        assert ok
        assert abs(computed - 180.156) < 0.1

    def test_mismatch_detected(self):
        elements = {"C": 6, "H": 12, "O": 6}
        _, ok = verify_molecular_weight(elements, 200.0)
        assert not ok

    def test_custom_tolerance(self):
        elements = {"C": 6, "H": 12, "O": 6}
        _, ok = verify_molecular_weight(elements, 180.16, tolerance=0.01)
        assert ok  # computed ≈ 180.156, diff < 0.01 is border but ok within rounding


# ══════════════════════════════════════════════════════════════════════
#  Degree of Reduction
# ══════════════════════════════════════════════════════════════════════

class TestDegreeOfReduction:
    """Tests for γ_s calculation."""

    def test_glucose_gamma_4(self):
        """Glucose C6H12O6: γ = (24+12−12)/6 = 4.0"""
        elements = {"C": 6, "H": 12, "O": 6}
        assert degree_of_reduction(elements) == pytest.approx(4.0)

    def test_methane_gamma_8(self):
        """Methane CH4: γ = (4+4)/1 = 8.0 — most reduced single-C."""
        elements = {"C": 1, "H": 4}
        assert degree_of_reduction(elements) == pytest.approx(8.0)

    def test_co2_gamma_0(self):
        """CO2: γ = (4−4)/1 = 0 — fully oxidised."""
        elements = {"C": 1, "O": 2}
        assert degree_of_reduction(elements) == pytest.approx(0.0)

    def test_coumaric_acid(self):
        """p-Coumaric acid C9H8O3: γ = (36+8−6)/9 = 38/9 ≈ 4.222"""
        elements = {"C": 9, "H": 8, "O": 3}
        assert degree_of_reduction(elements) == pytest.approx(38.0 / 9.0)

    def test_no_carbon_raises(self):
        with pytest.raises(ValueError, match="must contain carbon"):
            degree_of_reduction({"H": 2, "O": 1})

    def test_biomass_gamma(self):
        """Verify the biomass constant: CH1.8O0.5N0.2 → γ = 4.2"""
        assert GAMMA_BIOMASS == pytest.approx(4.2)


# ══════════════════════════════════════════════════════════════════════
#  Theoretical Yield
# ══════════════════════════════════════════════════════════════════════

class TestTheoreticalYield:
    """Tests for Y_max calculations."""

    def test_glucose_yield(self):
        elements = {"C": 6, "H": 12, "O": 6}
        result = theoretical_yield_max(elements, 180.16)

        # γ_s = 4.0, γ_bio = 4.2
        assert result["gamma_s"] == pytest.approx(4.0)
        # Y_max C-mol = 4.0/4.2 ≈ 0.9524
        assert result["Y_max_Cmol"] == pytest.approx(4.0 / 4.2, rel=1e-3)
        # Practical ≈ 0.6 × 0.9524 ≈ 0.5714
        assert result["Y_practical_Cmol"] == pytest.approx(0.6 * 4.0 / 4.2, rel=1e-3)
        # Mass: Y_max_mg = Y_cmol × (MW_bio_Cmol / MW_sub_per_Cmol)
        # MW_sub_per_Cmol = 180.16/6 = 30.027
        # Y_max_mg = 0.9524 × (24.6/30.027) ≈ 0.780
        assert 0.7 < result["Y_max_mg"] < 0.9

    def test_yield_keys_present(self):
        elements = {"C": 5, "H": 10, "O": 5}
        result = theoretical_yield_max(elements, 150.13)
        expected_keys = {
            "gamma_s", "gamma_bio", "Y_max_Cmol", "Y_practical_Cmol",
            "Y_max_mg", "Y_practical_mg", "carbon_fraction",
        }
        assert expected_keys == set(result.keys())

    def test_more_reduced_gives_higher_yield(self):
        """A more reduced substrate should have higher Y_max_Cmol."""
        # Glucose (γ=4) vs p-coumaric acid (γ≈4.22)
        gluc = theoretical_yield_max({"C": 6, "H": 12, "O": 6}, 180.16)
        coum = theoretical_yield_max({"C": 9, "H": 8, "O": 3}, 164.16)
        assert coum["Y_max_Cmol"] > gluc["Y_max_Cmol"]


# ══════════════════════════════════════════════════════════════════════
#  Theoretical Oxygen Demand
# ══════════════════════════════════════════════════════════════════════

class TestThOD:
    """Tests for ThOD calculation."""

    def test_glucose_thod(self):
        """C6H12O6 + 6 O₂ → 6 CO₂ + 6 H₂O"""
        elements = {"C": 6, "H": 12, "O": 6}
        result = theoretical_oxygen_demand(elements, 180.16)
        assert result["O2_moles"] == pytest.approx(6.0)
        # ThOD = 6×32/180.16 ≈ 1.066
        assert result["ThOD_mg_mg"] == pytest.approx(6 * 32 / 180.16, rel=1e-3)

    def test_methane_thod(self):
        """CH4 + 2 O₂ → CO₂ + 2 H₂O"""
        elements = {"C": 1, "H": 4}
        result = theoretical_oxygen_demand(elements, 16.04)
        assert result["O2_moles"] == pytest.approx(2.0)

    def test_aromatic_higher_thod_per_mol(self):
        """Aromatics need more total O₂ per mole (more carbons)."""
        sugar = theoretical_oxygen_demand({"C": 5, "H": 10, "O": 5}, 150.13)
        aromatic = theoretical_oxygen_demand({"C": 9, "H": 8, "O": 3}, 164.16)
        assert aromatic["O2_moles"] > sugar["O2_moles"]


# ══════════════════════════════════════════════════════════════════════
#  Y_o2 Ceiling
# ══════════════════════════════════════════════════════════════════════

class TestYo2Ceiling:
    """Tests for Y_o2 ceiling."""

    def test_positive_yo2(self):
        result = yield_oxygen_ceiling(
            Y_practical_mg=0.5,
            ThOD_mg_mg=1.066,
            molecular_weight=180.16,
            elements={"C": 6, "H": 12, "O": 6},
        )
        assert result["Y_o2_max"] > 0
        assert result["f_carbon"] > 0 and result["f_carbon"] < 1

    def test_zero_thod_returns_inf(self):
        result = yield_oxygen_ceiling(
            Y_practical_mg=0.5,
            ThOD_mg_mg=0.0,
            molecular_weight=100.0,
            elements={"C": 1},
        )
        assert result["Y_o2_max"] == float("inf")


# ══════════════════════════════════════════════════════════════════════
#  qmax Heuristic
# ══════════════════════════════════════════════════════════════════════

class TestQmaxHeuristic:
    """Tests for substrate classification and qmax ceiling."""

    def test_glucose_is_sugar(self):
        result = qmax_heuristic({"C": 6, "H": 12, "O": 6}, 180.16)
        assert result["substrate_class"] == "simple_sugar"
        assert result["qmax_ceiling"] == 30.0

    def test_coumaric_acid_is_aromatic(self):
        result = qmax_heuristic({"C": 9, "H": 8, "O": 3}, 164.16)
        assert result["substrate_class"] == "aromatic"
        assert result["qmax_ceiling"] == 10.0

    def test_vanillic_acid_is_aromatic(self):
        result = qmax_heuristic({"C": 8, "H": 8, "O": 4}, 168.15)
        assert result["substrate_class"] == "aromatic"

    def test_degree_of_unsaturation(self):
        """p-HBA C7H6O3: DoU = (14+2−6)/2 = 5"""
        result = qmax_heuristic({"C": 7, "H": 6, "O": 3}, 138.12)
        assert result["degree_of_unsaturation"] == 5.0


# ══════════════════════════════════════════════════════════════════════
#  Full Report
# ══════════════════════════════════════════════════════════════════════

class TestFullReport:
    """Tests for compute_bounds_report."""

    def test_glucose_report(self):
        report = compute_bounds_report("Glucose", "C6H12O6", 180.16)
        assert report.substrate_name == "Glucose"
        assert report.mw_consistent
        assert report.gamma_s == pytest.approx(4.0)
        assert "Y" in report.suggested_bounds
        assert "qmax" in report.suggested_bounds

    def test_report_to_dict(self):
        report = compute_bounds_report("Glucose", "C6H12O6", 180.16)
        d = report.to_dict()
        assert "yield" in d
        assert "oxygen_demand" in d
        assert "Y_o2" in d
        assert "qmax" in d
        assert "suggested_bounds" in d

    def test_report_summary_text(self):
        report = compute_bounds_report("Xylose", "C5H10O5", 150.13)
        text = report.summary_text()
        assert "Xylose" in text
        assert "C5H10O5" in text
        assert "γ_substrate" in text

    def test_all_six_substrates(self):
        """Verify we can compute reports for all project substrates."""
        substrates = [
            ("Glucose", "C6H12O6", 180.16),
            ("Xylose", "C5H10O5", 150.13),
            ("pCoumaricAcid", "C9H8O3", 164.16),
            ("pHydroxybenzoicAcid", "C7H6O3", 138.12),
            ("SyringicAcid", "C9H10O5", 198.17),
            ("VanillicAcid", "C8H8O4", 168.15),
        ]
        for name, formula, mw in substrates:
            report = compute_bounds_report(name, formula, mw)
            assert report.mw_consistent, f"MW mismatch for {name}"
            assert report.gamma_s > 0
            assert report.Y_max_mg > 0
            assert report.ThOD_mg_mg > 0
            assert report.Y_o2_max > 0


# ══════════════════════════════════════════════════════════════════════
#  Config Integration
# ══════════════════════════════════════════════════════════════════════

class TestConfigIntegration:
    """Tests for loading from config files."""

    def _make_config(self, tmp_path, formula=None):
        """Create a minimal config JSON in tmp_path."""
        config = {
            "substrate": {
                "name": "TestSubstrate",
                "molecular_weight": 180.16,
                "unit": "mg/L",
            },
            "initial_guesses": {
                "qmax": 5.0, "Ks": 100.0, "Y": 0.4,
                "b_decay": 0.01, "K_o2": 0.1, "Y_o2": 0.8,
            },
            "bounds": {
                "qmax": [0.1, 50.0], "Ks": [1.0, 1000.0], "Y": [0.01, 2.0],
                "b_decay": [0.001, 0.5], "K_o2": [0.01, 1.0],
                "Y_o2": [0.05, 5.0],
            },
        }
        if formula:
            config["substrate"]["molecular_formula"] = formula
        cfg_path = tmp_path / "test.json"
        with open(cfg_path, "w") as f:
            json.dump(config, f)
        return str(cfg_path)

    def test_load_from_config_with_formula(self, tmp_path):
        cfg = self._make_config(tmp_path, formula="C6H12O6")
        report = compute_from_config(cfg)
        assert report.substrate_name == "TestSubstrate"
        assert report.gamma_s == pytest.approx(4.0)

    def test_missing_formula_raises(self, tmp_path):
        cfg = self._make_config(tmp_path, formula=None)
        with pytest.raises(ValueError, match="molecular_formula"):
            compute_from_config(cfg)

    def test_compare_with_current_bounds(self, tmp_path):
        cfg = self._make_config(tmp_path, formula="C6H12O6")
        report = compute_from_config(cfg)
        comparison = compare_with_current_bounds(report, cfg)
        assert "Y" in comparison
        assert "qmax" in comparison

    def test_compare_flags_high_bounds(self, tmp_path):
        """A config with Y upper bound >> theoretical ceiling → ⚠ HIGH."""
        config = {
            "substrate": {
                "name": "Test",
                "molecular_formula": "C6H12O6",
                "molecular_weight": 180.16,
                "unit": "mg/L",
            },
            "initial_guesses": {"Y": 5.0},
            "bounds": {"Y": [0.01, 50.0]},
        }
        cfg_path = tmp_path / "high.json"
        with open(cfg_path, "w") as f:
            json.dump(config, f)

        report = compute_from_config(str(cfg_path))
        comparison = compare_with_current_bounds(report, str(cfg_path))
        assert "HIGH" in comparison


# ══════════════════════════════════════════════════════════════════════
#  Serialisation Roundtrip
# ══════════════════════════════════════════════════════════════════════

class TestSerialisation:
    """Test JSON serialisation of the report."""

    def test_to_dict_json_serialisable(self):
        report = compute_bounds_report("Glucose", "C6H12O6", 180.16)
        d = report.to_dict()
        # Must not raise
        serialised = json.dumps(d)
        loaded = json.loads(serialised)
        assert loaded["substrate_name"] == "Glucose"

    def test_suggested_bounds_are_lists(self):
        report = compute_bounds_report("Glucose", "C6H12O6", 180.16)
        d = report.to_dict()
        for param, bounds in d["suggested_bounds"].items():
            assert isinstance(bounds, list)
            assert len(bounds) == 2
