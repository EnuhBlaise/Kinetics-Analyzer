"""
Optimizer diagnostics for investigating parameter landscape behaviour.

This module provides tools to probe the objective function landscape
around an optimum, revealing:
- Multi-start sensitivity (local minima)
- 1-D parameter profiles (identifiability)
- 2-D contour surfaces (parameter correlations)
- Hessian eigenvalue analysis (conditioning)
- Parameter correlation matrix (redundancy)

Usage:
    >>> from src.fitting.diagnostics import OptimizerDiagnostics
    >>> diag = OptimizerDiagnostics(objective_fn, parameter_names, bounds)
    >>> report = diag.run_all(best_params)
    >>> report.save(output_dir)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Any
from pathlib import Path
import json
import warnings

from scipy.optimize import minimize, differential_evolution


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class MultiStartResult:
    """Results from multi-start optimization."""
    starting_points: List[Dict[str, float]]
    converged_points: List[Dict[str, float]]
    objectives: List[float]
    success_flags: List[bool]
    best_params: Dict[str, float]
    best_objective: float
    n_unique_minima: int
    cluster_labels: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_objective": self.best_objective,
            "n_starts": len(self.objectives),
            "n_converged": sum(self.success_flags),
            "n_unique_minima": self.n_unique_minima,
            "objective_range": [float(min(self.objectives)), float(max(self.objectives))],
            "all_objectives": [float(v) for v in self.objectives],
        }


@dataclass
class ProfileResult:
    """Results from 1-D parameter profile analysis."""
    parameter_name: str
    fixed_values: np.ndarray
    profile_objectives: np.ndarray
    conditional_params: List[Dict[str, float]]
    optimal_value: float
    is_identifiable: bool
    curvature: float  # second derivative at optimum

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter": self.parameter_name,
            "optimal_value": float(self.optimal_value),
            "is_identifiable": self.is_identifiable,
            "curvature": float(self.curvature),
            "min_objective": float(np.min(self.profile_objectives)),
            "n_points": len(self.fixed_values),
        }


@dataclass
class ContourResult:
    """Results from 2-D contour analysis."""
    param_x: str
    param_y: str
    x_values: np.ndarray
    y_values: np.ndarray
    objective_grid: np.ndarray
    correlation_direction: str  # 'positive', 'negative', 'none'

    def to_dict(self) -> Dict[str, Any]:
        return {
            "param_x": self.param_x,
            "param_y": self.param_y,
            "correlation_direction": self.correlation_direction,
            "grid_shape": list(self.objective_grid.shape),
        }


@dataclass
class HessianDiagnosticResult:
    """Results from Hessian eigenvalue analysis."""
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    condition_number: float
    parameter_names: List[str]
    correlation_matrix: np.ndarray
    standard_errors: np.ndarray
    sloppy_directions: List[Dict[str, float]]  # eigenvectors with small eigenvalues

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition_number": float(self.condition_number),
            "eigenvalues": [float(v) for v in self.eigenvalues],
            "standard_errors": {
                name: float(se)
                for name, se in zip(self.parameter_names, self.standard_errors)
            },
            "correlation_matrix": {
                self.parameter_names[i]: {
                    self.parameter_names[j]: float(self.correlation_matrix[i, j])
                    for j in range(len(self.parameter_names))
                }
                for i in range(len(self.parameter_names))
            },
            "sloppy_directions": self.sloppy_directions,
        }


@dataclass
class ConvergenceTrace:
    """Parameter trajectory during optimization."""
    iterations: List[int]
    parameter_history: Dict[str, List[float]]
    objective_history: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_iterations": len(self.iterations),
            "final_objective": float(self.objective_history[-1]) if self.objective_history else None,
            "objective_reduction": (
                float(self.objective_history[0] - self.objective_history[-1])
                if len(self.objective_history) > 1 else 0.0
            ),
        }


@dataclass
class DiagnosticsReport:
    """Complete diagnostics report."""
    multi_start: Optional[MultiStartResult] = None
    profiles: Optional[Dict[str, ProfileResult]] = None
    contours: Optional[Dict[str, ContourResult]] = None
    hessian: Optional[HessianDiagnosticResult] = None
    convergence: Optional[ConvergenceTrace] = None

    def save(self, output_dir: str) -> List[Path]:
        """Save all results and plots to output directory."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved = []

        # Save JSON summary
        summary = {}
        if self.multi_start is not None:
            summary["multi_start"] = self.multi_start.to_dict()
        if self.profiles is not None:
            summary["profiles"] = {k: v.to_dict() for k, v in self.profiles.items()}
        if self.contours is not None:
            summary["contours"] = {k: v.to_dict() for k, v in self.contours.items()}
        if self.hessian is not None:
            summary["hessian"] = self.hessian.to_dict()
        if self.convergence is not None:
            summary["convergence"] = self.convergence.to_dict()

        json_path = out / "diagnostics_summary.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2, cls=_NumpyEncoder)
        saved.append(json_path)

        return saved


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Main diagnostics class
# ---------------------------------------------------------------------------

class OptimizerDiagnostics:
    """
    Comprehensive optimizer diagnostics for kinetic parameter estimation.

    Probes the objective function landscape to reveal identifiability issues,
    parameter correlations, and multi-modality.

    Args:
        objective_fn: Callable that maps parameter array → scalar error.
        parameter_names: Ordered list of parameter names.
        bounds: Dict mapping parameter name → (lower, upper).
        verbose: Print progress.
    """

    def __init__(
        self,
        objective_fn: Callable[[np.ndarray], float],
        parameter_names: List[str],
        bounds: Dict[str, Tuple[float, float]],
        verbose: bool = True,
        use_scaled_optimize: bool = False,
    ):
        self.objective_fn = objective_fn
        self.parameter_names = parameter_names
        self.bounds = bounds
        self.n_params = len(parameter_names)
        self.verbose = verbose
        self.use_scaled_optimize = use_scaled_optimize

        # Ordered bounds as arrays
        self._lb = np.array([bounds[p][0] for p in parameter_names])
        self._ub = np.array([bounds[p][1] for p in parameter_names])
        self._bounds_list = [(bounds[p][0], bounds[p][1]) for p in parameter_names]

    # ------------------------------------------------------------------
    # Internal: optionally-scaled re-optimization helper
    # ------------------------------------------------------------------
    def _reoptimize(self, fn, x0, method="L-BFGS-B", bounds=None, options=None):
        """Wrapper around scipy.optimize.minimize, optionally with scaling."""
        from src.fitting.scaling import scaled_minimize
        return scaled_minimize(
            fn, x0=np.asarray(x0, dtype=float),
            bounds=bounds, method=method, options=options,
            use_scaling=self.use_scaled_optimize,
        )

    # ------------------------------------------------------------------
    # 1. Multi-start optimization
    # ------------------------------------------------------------------
    def multi_start(
        self,
        best_params: Dict[str, float],
        n_starts: int = 50,
        method: str = "L-BFGS-B",
        seed: int = 42,
        cluster_tol: float = 0.05,
    ) -> MultiStartResult:
        """
        Run optimizer from many random starting points.

        Latin-hypercube samples *n_starts* initial guesses across the
        parameter bounds, runs L-BFGS-B from each, and clusters the
        converged solutions to count distinct minima.

        Args:
            best_params: Reference best-fit parameters.
            n_starts: Number of random starting points.
            method: Local optimizer (default L-BFGS-B).
            seed: RNG seed for reproducibility.
            cluster_tol: Relative tolerance for clustering converged
                         solutions (fraction of parameter range).

        Returns:
            MultiStartResult with all converged points and cluster info.
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  MULTI-START OPTIMIZATION  ({n_starts} starts)")
            print(f"{'='*60}")

        rng = np.random.default_rng(seed)

        # Latin hypercube sampling
        starts = self._latin_hypercube(n_starts, rng)

        starting_points = []
        converged_points = []
        objectives = []
        success_flags = []

        for i, x0 in enumerate(starts):
            sp = dict(zip(self.parameter_names, x0))
            starting_points.append(sp)

            try:
                res = self._reoptimize(
                    self.objective_fn, x0,
                    method=method,
                    bounds=self._bounds_list,
                    options={"maxiter": 5000, "ftol": 1e-10},
                )
                cp = dict(zip(self.parameter_names, res.x))
                converged_points.append(cp)
                objectives.append(float(res.fun))
                success_flags.append(bool(res.success))
            except Exception:
                converged_points.append(sp)
                objectives.append(1e10)
                success_flags.append(False)

            if self.verbose and (i + 1) % 10 == 0:
                print(f"  completed {i+1}/{n_starts} starts")

        objectives = np.array(objectives)

        # Cluster converged solutions
        converged_array = np.array([
            [cp[p] for p in self.parameter_names] for cp in converged_points
        ])
        cluster_labels = self._cluster_solutions(converged_array, cluster_tol)
        n_unique = len(set(cluster_labels))

        best_idx = int(np.argmin(objectives))

        result = MultiStartResult(
            starting_points=starting_points,
            converged_points=converged_points,
            objectives=objectives.tolist(),
            success_flags=success_flags,
            best_params=converged_points[best_idx],
            best_objective=float(objectives[best_idx]),
            n_unique_minima=n_unique,
            cluster_labels=cluster_labels,
        )

        if self.verbose:
            print(f"\n  Results:")
            print(f"    Converged:       {sum(success_flags)}/{n_starts}")
            print(f"    Unique minima:   {n_unique}")
            print(f"    Best objective:  {objectives[best_idx]:.6e}")
            print(f"    Worst objective: {objectives[~np.isinf(objectives)].max():.6e}"
                  if np.any(~np.isinf(objectives)) else "")
            obj_std = np.std(objectives[objectives < 1e9])
            if len(objectives[objectives < 1e9]) > 1:
                print(f"    Std of converged obj: {obj_std:.6e}")

        return result

    # ------------------------------------------------------------------
    # 2. 1-D parameter profiles
    # ------------------------------------------------------------------
    def parameter_profiles(
        self,
        best_params: Dict[str, float],
        n_points: int = 30,
        parameters: Optional[List[str]] = None,
        method: str = "L-BFGS-B",
    ) -> Dict[str, ProfileResult]:
        """
        Compute 1-D profile likelihoods for each parameter.

        For each parameter, fix it at a grid of values across its bounds
        and re-optimise all other parameters. The resulting profile reveals
        identifiability.

        Args:
            best_params: Best-fit parameter values.
            n_points: Number of grid points per parameter.
            parameters: Subset of parameters to profile (default: all).
            method: Optimizer for the conditional fits.

        Returns:
            Dict mapping parameter name → ProfileResult.
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  1-D PARAMETER PROFILES  ({n_points} points each)")
            print(f"{'='*60}")

        params_to_profile = parameters or self.parameter_names
        profiles: Dict[str, ProfileResult] = {}

        for pname in params_to_profile:
            if self.verbose:
                print(f"\n  Profiling: {pname}")

            pidx = self.parameter_names.index(pname)
            lo, hi = self.bounds[pname]
            grid = np.linspace(lo, hi, n_points)

            # Indices of the *free* parameters (everything except pidx)
            free_idx = [i for i in range(self.n_params) if i != pidx]
            free_names = [self.parameter_names[i] for i in free_idx]
            free_bounds = [self._bounds_list[i] for i in free_idx]
            free_x0 = np.array([best_params[self.parameter_names[i]] for i in free_idx])

            profile_obj = np.full(n_points, np.nan)
            cond_params = []

            for j, fixed_val in enumerate(grid):
                def _obj_reduced(x_free, _fixed=fixed_val, _pidx=pidx, _free_idx=free_idx):
                    x_full = np.empty(self.n_params)
                    x_full[_pidx] = _fixed
                    for k, fi in enumerate(_free_idx):
                        x_full[fi] = x_free[k]
                    return self.objective_fn(x_full)

                try:
                    res = self._reoptimize(
                        _obj_reduced, free_x0,
                        method=method,
                        bounds=free_bounds,
                        options={"maxiter": 3000, "ftol": 1e-10},
                    )
                    profile_obj[j] = res.fun
                    full = dict(zip(self.parameter_names,
                                    np.empty(self.n_params)))
                    full[pname] = fixed_val
                    for k, fi in enumerate(free_idx):
                        full[self.parameter_names[fi]] = res.x[k]
                    cond_params.append(full)

                    # Warm-start next point from this solution
                    free_x0 = res.x.copy()
                except Exception:
                    profile_obj[j] = np.nan
                    cond_params.append({})

            # Assess identifiability
            valid = ~np.isnan(profile_obj)
            obj_min = np.nanmin(profile_obj)
            obj_range = np.nanmax(profile_obj) - obj_min
            # Relative rise: if the objective rises < 5% across the bounds,
            # the parameter is poorly identifiable
            relative_rise = obj_range / (abs(obj_min) + 1e-30)
            is_identifiable = relative_rise > 0.1

            # Estimate curvature at optimum via finite difference
            best_idx = int(np.nanargmin(profile_obj))
            if 0 < best_idx < n_points - 1 and valid[best_idx - 1] and valid[best_idx + 1]:
                h = grid[1] - grid[0]
                curvature = (
                    profile_obj[best_idx - 1]
                    - 2 * profile_obj[best_idx]
                    + profile_obj[best_idx + 1]
                ) / h**2
            else:
                curvature = 0.0

            profiles[pname] = ProfileResult(
                parameter_name=pname,
                fixed_values=grid,
                profile_objectives=profile_obj,
                conditional_params=cond_params,
                optimal_value=float(grid[best_idx]),
                is_identifiable=is_identifiable,
                curvature=float(curvature),
            )

            if self.verbose:
                tag = "✓ identifiable" if is_identifiable else "✗ POORLY identifiable"
                print(f"    {tag}  (relative rise = {relative_rise:.4f}, "
                      f"curvature = {curvature:.4e})")

        return profiles

    # ------------------------------------------------------------------
    # 3. 2-D contour surfaces
    # ------------------------------------------------------------------
    def contour_analysis(
        self,
        best_params: Dict[str, float],
        param_pairs: Optional[List[Tuple[str, str]]] = None,
        n_grid: int = 25,
    ) -> Dict[str, ContourResult]:
        """
        Compute 2-D objective contours for parameter pairs.

        For each pair, create a grid, evaluate the objective at each point
        (fixing the two parameters, keeping all others at their best-fit
        values), and classify the contour shape.

        Args:
            best_params: Best-fit parameter values.
            param_pairs: List of (param_x, param_y) tuples. If None,
                         uses the classic problematic Monod pairs.
            n_grid: Grid points per axis.

        Returns:
            Dict mapping "param_x__param_y" → ContourResult.
        """
        if param_pairs is None:
            # Default problematic pairs in Monod kinetics
            all_pairs = [
                ("qmax", "Ks"), ("qmax", "b_decay"), ("Ks", "Ki"),
                ("qmax", "Ki"), ("Y", "b_decay"),
            ]
            # Filter to only those that exist in this model
            param_pairs = [
                (a, b) for a, b in all_pairs
                if a in self.parameter_names and b in self.parameter_names
            ]
            # Add oxygen pairs if present
            if "K_o2" in self.parameter_names:
                param_pairs.append(("qmax", "K_o2"))
            if "lag_time" in self.parameter_names:
                param_pairs.append(("qmax", "lag_time"))
                param_pairs.append(("b_decay", "lag_time"))

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  2-D CONTOUR ANALYSIS  ({len(param_pairs)} pairs, {n_grid}×{n_grid} grid)")
            print(f"{'='*60}")

        contours: Dict[str, ContourResult] = {}

        for px, py in param_pairs:
            if self.verbose:
                print(f"\n  Contour: {px} vs {py}")

            ix = self.parameter_names.index(px)
            iy = self.parameter_names.index(py)

            x_vals = np.linspace(self.bounds[px][0], self.bounds[px][1], n_grid)
            y_vals = np.linspace(self.bounds[py][0], self.bounds[py][1], n_grid)

            Z = np.full((n_grid, n_grid), np.nan)
            base = np.array([best_params[p] for p in self.parameter_names])

            for i, xv in enumerate(x_vals):
                for j, yv in enumerate(y_vals):
                    x_eval = base.copy()
                    x_eval[ix] = xv
                    x_eval[iy] = yv
                    try:
                        Z[j, i] = self.objective_fn(x_eval)
                    except Exception:
                        Z[j, i] = np.nan

            # Classify correlation from contour shape
            corr_dir = self._classify_contour_correlation(Z, x_vals, y_vals)

            key = f"{px}__{py}"
            contours[key] = ContourResult(
                param_x=px,
                param_y=py,
                x_values=x_vals,
                y_values=y_vals,
                objective_grid=Z,
                correlation_direction=corr_dir,
            )

            if self.verbose:
                print(f"    Correlation direction: {corr_dir}")

        return contours

    # ------------------------------------------------------------------
    # 4. Hessian eigenvalue analysis + correlation matrix
    # ------------------------------------------------------------------
    def hessian_analysis(
        self,
        best_params: Dict[str, float],
    ) -> HessianDiagnosticResult:
        """
        Analyse the Hessian at the optimum.

        Computes eigenvalues (conditioning), standard errors, and the
        parameter correlation matrix from the inverse Hessian.

        Args:
            best_params: Best-fit parameter values.

        Returns:
            HessianDiagnosticResult with eigenvalues, correlations, etc.
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  HESSIAN EIGENVALUE ANALYSIS")
            print(f"{'='*60}")

        try:
            import numdifftools as nd
        except ImportError:
            raise ImportError("numdifftools is required: pip install numdifftools")

        x_opt = np.array([best_params[p] for p in self.parameter_names])

        # Compute Hessian via numdifftools
        hess_func = nd.Hessian(self.objective_fn, step=1e-4)
        H = hess_func(x_opt)

        # Eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(H)

        # Sort by ascending eigenvalue
        sort_idx = np.argsort(eigvals)
        eigvals = eigvals[sort_idx]
        eigvecs = eigvecs[:, sort_idx]

        # Condition number
        pos_eigvals = eigvals[eigvals > 0]
        if len(pos_eigvals) >= 2:
            cond = float(pos_eigvals[-1] / pos_eigvals[0])
        else:
            cond = np.inf

        # Covariance matrix (inverse Hessian, scaled by 2 for SSE-based objective)
        # H ≈ 2 * J^T J, so Cov ≈ sigma^2 * (J^T J)^{-1} ≈ sigma^2 * 2 * H^{-1}
        # For profile-based diagnostics we use H^{-1} directly
        try:
            H_inv = np.linalg.inv(H)
            # Standard errors from diagonal of H_inv
            diag = np.diag(H_inv)
            se = np.where(diag > 0, np.sqrt(diag), np.nan)

            # Correlation matrix
            D = np.sqrt(np.abs(np.diag(H_inv)))
            D_safe = np.where(D > 0, D, 1.0)
            corr = H_inv / np.outer(D_safe, D_safe)
            # Clip to [-1, 1]
            corr = np.clip(corr, -1.0, 1.0)
        except np.linalg.LinAlgError:
            se = np.full(self.n_params, np.nan)
            corr = np.full((self.n_params, self.n_params), np.nan)

        # Identify sloppy directions (small eigenvalues)
        sloppy = []
        threshold = 0.01 * np.max(np.abs(eigvals)) if len(eigvals) > 0 else 0
        for k in range(len(eigvals)):
            if abs(eigvals[k]) < threshold:
                direction = {}
                for i, pname in enumerate(self.parameter_names):
                    if abs(eigvecs[i, k]) > 0.1:
                        direction[pname] = float(eigvecs[i, k])
                sloppy.append({
                    "eigenvalue": float(eigvals[k]),
                    "direction": direction,
                })

        result = HessianDiagnosticResult(
            eigenvalues=eigvals,
            eigenvectors=eigvecs,
            condition_number=cond,
            parameter_names=self.parameter_names,
            correlation_matrix=corr,
            standard_errors=se,
            sloppy_directions=sloppy,
        )

        if self.verbose:
            print(f"\n  Eigenvalues:")
            for k, ev in enumerate(eigvals):
                marker = "  ← SLOPPY" if abs(ev) < threshold else ""
                print(f"    λ_{k+1} = {ev:.4e}{marker}")
            print(f"\n  Condition number: {cond:.2e}")
            print(f"\n  Standard errors:")
            for pname, s in zip(self.parameter_names, se):
                print(f"    {pname:12s}: {s:.4e}")
            print(f"\n  Correlation matrix:")
            header = "              " + "".join(f"{p:>12s}" for p in self.parameter_names)
            print(header)
            for i, pi in enumerate(self.parameter_names):
                row = f"    {pi:10s}"
                for j in range(self.n_params):
                    val = corr[i, j]
                    flag = " *" if abs(val) > 0.9 and i != j else "  "
                    row += f"  {val:8.3f}{flag}"
                print(row)
            if sloppy:
                print(f"\n  ⚠ {len(sloppy)} sloppy direction(s) detected")
                for s in sloppy:
                    print(f"    eigenvalue={s['eigenvalue']:.4e}, "
                          f"involves: {s['direction']}")

        return result

    # ------------------------------------------------------------------
    # 5. Convergence trace
    # ------------------------------------------------------------------
    def trace_convergence(
        self,
        initial_params: Dict[str, float],
        method: str = "L-BFGS-B",
    ) -> ConvergenceTrace:
        """
        Run optimization while recording the parameter trajectory.

        Args:
            initial_params: Starting parameter values.
            method: Optimization method.

        Returns:
            ConvergenceTrace with full parameter history.
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  CONVERGENCE TRACE")
            print(f"{'='*60}")

        history_x: List[np.ndarray] = []
        history_f: List[float] = []

        def _callback(xk):
            history_x.append(xk.copy())
            history_f.append(float(self.objective_fn(xk)))

        x0 = np.array([initial_params[p] for p in self.parameter_names])

        # Record initial point
        history_x.append(x0.copy())
        history_f.append(float(self.objective_fn(x0)))

        minimize(
            self.objective_fn, x0,
            method=method,
            bounds=self._bounds_list,
            options={"maxiter": 10000, "ftol": 1e-12},
            callback=_callback,
        )

        # Build trace
        param_hist = {p: [] for p in self.parameter_names}
        for xk in history_x:
            for i, p in enumerate(self.parameter_names):
                param_hist[p].append(float(xk[i]))

        trace = ConvergenceTrace(
            iterations=list(range(len(history_x))),
            parameter_history=param_hist,
            objective_history=history_f,
        )

        if self.verbose:
            print(f"  Iterations recorded: {len(history_x)}")
            print(f"  Objective: {history_f[0]:.6e} → {history_f[-1]:.6e}")
            print(f"  Reduction: {history_f[0] - history_f[-1]:.6e}")

        return trace

    # ------------------------------------------------------------------
    # Run all diagnostics
    # ------------------------------------------------------------------
    def run_all(
        self,
        best_params: Dict[str, float],
        initial_params: Optional[Dict[str, float]] = None,
        n_starts: int = 50,
        n_profile_points: int = 30,
        n_contour_grid: int = 25,
        param_pairs: Optional[List[Tuple[str, str]]] = None,
        profile_params: Optional[List[str]] = None,
    ) -> DiagnosticsReport:
        """
        Run the complete diagnostics suite.

        Args:
            best_params: Best-fit parameters from optimization.
            initial_params: Starting parameters (for convergence trace).
                            Defaults to best_params if not given.
            n_starts: Number of multi-start trials.
            n_profile_points: Points per 1-D profile.
            n_contour_grid: Grid size for 2-D contours.
            param_pairs: Custom parameter pairs for contours.
            profile_params: Subset of parameters to profile.

        Returns:
            DiagnosticsReport with all results.
        """
        if initial_params is None:
            initial_params = best_params

        report = DiagnosticsReport()

        report.multi_start = self.multi_start(best_params, n_starts=n_starts)
        report.profiles = self.parameter_profiles(
            best_params, n_points=n_profile_points, parameters=profile_params
        )
        report.contours = self.contour_analysis(
            best_params, param_pairs=param_pairs, n_grid=n_contour_grid
        )
        report.hessian = self.hessian_analysis(best_params)
        report.convergence = self.trace_convergence(initial_params)

        return report

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot_multi_start(
        self,
        result: MultiStartResult,
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot multi-start results: objective histogram and parallel coordinates."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Histogram of converged objectives
        valid_obj = [o for o in result.objectives if o < 1e9]
        ax = axes[0]
        ax.hist(valid_obj, bins=min(30, len(valid_obj)), color='#4477AA',
                edgecolor='white', alpha=0.8)
        ax.axvline(result.best_objective, color='#EE6677', linewidth=2,
                   linestyle='--', label=f'Best = {result.best_objective:.4e}')
        ax.set_xlabel('Objective Value')
        ax.set_ylabel('Count')
        ax.set_title(f'Multi-Start Results ({result.n_unique_minima} unique minima)')
        ax.legend()

        # Right: Parallel coordinates of converged solutions
        ax = axes[1]
        # Normalise parameters to [0, 1] for visual comparison
        for i, cp in enumerate(result.converged_points):
            if result.objectives[i] >= 1e9:
                continue
            vals = []
            for p in self.parameter_names:
                lo, hi = self.bounds[p]
                span = hi - lo if hi > lo else 1.0
                vals.append((cp[p] - lo) / span)
            alpha = 0.3 if result.objectives[i] > result.best_objective * 1.01 else 1.0
            lw = 2 if result.objectives[i] <= result.best_objective * 1.01 else 0.5
            color = '#EE6677' if result.objectives[i] <= result.best_objective * 1.01 else '#BBBBBB'
            ax.plot(range(self.n_params), vals, alpha=alpha, linewidth=lw, color=color)

        ax.set_xticks(range(self.n_params))
        ax.set_xticklabels(self.parameter_names, rotation=45, ha='right')
        ax.set_ylabel('Normalised Value (within bounds)')
        ax.set_title('Converged Solutions (red = best cluster)')
        ax.set_ylim(-0.05, 1.05)

        fig.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            fig.savefig(str(Path(output_path).with_suffix('.pdf')), bbox_inches='tight')
        return fig

    def plot_profiles(
        self,
        profiles: Dict[str, ProfileResult],
        best_params: Dict[str, float],
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot 1-D parameter profiles."""
        n = len(profiles)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        if n == 1:
            axes = np.array([axes])
        axes = np.atleast_2d(axes)

        for idx, (pname, prof) in enumerate(profiles.items()):
            row, col = divmod(idx, ncols)
            ax = axes[row, col]

            valid = ~np.isnan(prof.profile_objectives)
            ax.plot(prof.fixed_values[valid], prof.profile_objectives[valid],
                    'o-', color='#4477AA', markersize=3, linewidth=1.5)

            # Mark optimum
            ax.axvline(best_params[pname], color='#EE6677', linestyle='--',
                       linewidth=1.5, label=f'Best = {best_params[pname]:.3g}')

            ax.set_xlabel(pname)
            ax.set_ylabel('Objective')
            tag = "✓" if prof.is_identifiable else "✗ poorly identifiable"
            ax.set_title(f'{pname}  ({tag})')
            ax.legend(fontsize=8)

        # Hide unused axes
        for idx in range(n, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row, col].set_visible(False)

        fig.suptitle('1-D Parameter Profiles', fontsize=14, fontweight='bold')
        fig.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            fig.savefig(str(Path(output_path).with_suffix('.pdf')), bbox_inches='tight')
        return fig

    def plot_contours(
        self,
        contours: Dict[str, ContourResult],
        best_params: Dict[str, float],
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot 2-D objective contour surfaces."""
        n = len(contours)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
        if n == 1:
            axes = np.array([axes])
        axes = np.atleast_2d(axes)

        for idx, (key, cr) in enumerate(contours.items()):
            row, col = divmod(idx, ncols)
            ax = axes[row, col]

            X, Y = np.meshgrid(cr.x_values, cr.y_values)
            Z = cr.objective_grid

            # Use log scale for better visualisation of valleys
            Z_plot = Z.copy()
            Z_min = np.nanmin(Z_plot)
            Z_plot = np.where(np.isnan(Z_plot), np.nanmax(Z_plot), Z_plot)
            Z_log = np.log10(Z_plot - Z_min + 1e-30 * (abs(Z_min) + 1))

            levels = np.linspace(np.nanmin(Z_log), np.nanmax(Z_log), 20)
            if len(np.unique(levels)) < 3:
                # Fall back to linear if log doesn't vary
                cs = ax.contourf(X, Y, Z_plot, levels=20, cmap='viridis')
            else:
                cs = ax.contourf(X, Y, Z_log, levels=levels, cmap='viridis')
            plt.colorbar(cs, ax=ax, label='log₁₀(Δobjective)')

            # Mark best point
            ax.plot(best_params[cr.param_x], best_params[cr.param_y],
                    'r*', markersize=12, zorder=5)

            ax.set_xlabel(cr.param_x)
            ax.set_ylabel(cr.param_y)
            ax.set_title(f'{cr.param_x} vs {cr.param_y}\n({cr.correlation_direction})')

        for idx in range(n, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row, col].set_visible(False)

        fig.suptitle('2-D Objective Contours', fontsize=14, fontweight='bold')
        fig.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            fig.savefig(str(Path(output_path).with_suffix('.pdf')), bbox_inches='tight')
        return fig

    def plot_hessian(
        self,
        result: HessianDiagnosticResult,
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot Hessian diagnostics: eigenvalue spectrum + correlation matrix."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Left: eigenvalue bar chart
        ax = axes[0]
        colors = ['#EE6677' if abs(e) < 0.01 * max(abs(result.eigenvalues))
                  else '#4477AA' for e in result.eigenvalues]
        ax.barh(range(len(result.eigenvalues)), np.log10(np.abs(result.eigenvalues) + 1e-30),
                color=colors)
        ax.set_yticks(range(len(result.eigenvalues)))
        ax.set_yticklabels([f'λ_{i+1}' for i in range(len(result.eigenvalues))])
        ax.set_xlabel('log₁₀|eigenvalue|')
        ax.set_title(f'Eigenvalue Spectrum\n(cond = {result.condition_number:.1e})')

        # Middle: correlation matrix heatmap
        ax = axes[1]
        im = ax.imshow(result.correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(self.n_params))
        ax.set_xticklabels(result.parameter_names, rotation=45, ha='right')
        ax.set_yticks(range(self.n_params))
        ax.set_yticklabels(result.parameter_names)
        # Annotate values
        for i in range(self.n_params):
            for j in range(self.n_params):
                val = result.correlation_matrix[i, j]
                color = 'white' if abs(val) > 0.6 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=7, color=color)
        ax.set_title('Parameter Correlation Matrix')

        # Right: standard errors bar chart
        ax = axes[2]
        se_vals = result.standard_errors
        valid_se = ~np.isnan(se_vals)
        bars = ax.barh(
            [result.parameter_names[i] for i in range(self.n_params) if valid_se[i]],
            [se_vals[i] for i in range(self.n_params) if valid_se[i]],
            color='#4477AA'
        )
        ax.set_xlabel('Standard Error')
        ax.set_title('Parameter Standard Errors')

        fig.suptitle('Hessian Diagnostics', fontsize=14, fontweight='bold')
        fig.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            fig.savefig(str(Path(output_path).with_suffix('.pdf')), bbox_inches='tight')
        return fig

    def plot_convergence(
        self,
        trace: ConvergenceTrace,
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot convergence trace: objective + parameter trajectories."""
        n_params = len(self.parameter_names)
        fig, axes = plt.subplots(n_params + 1, 1,
                                 figsize=(10, 2.5 * (n_params + 1)),
                                 sharex=True)

        iters = trace.iterations

        # Top: objective
        obj_vals = trace.objective_history
        if all(v > 0 for v in obj_vals):
            axes[0].semilogy(iters, obj_vals, color='#4477AA', linewidth=1.5)
        else:
            axes[0].plot(iters, obj_vals, color='#4477AA', linewidth=1.5)
        axes[0].set_ylabel('Objective')
        axes[0].set_title('Convergence Trace')

        # Parameter traces
        colors = plt.cm.tab10(np.linspace(0, 1, n_params))
        for i, pname in enumerate(self.parameter_names):
            ax = axes[i + 1]
            vals = trace.parameter_history[pname]
            ax.plot(iters[:len(vals)], vals, color=colors[i], linewidth=1.5)
            ax.set_ylabel(pname)
            # Show bounds as horizontal bands
            lo, hi = self.bounds[pname]
            ax.axhline(lo, color='grey', linewidth=0.5, linestyle=':')
            ax.axhline(hi, color='grey', linewidth=0.5, linestyle=':')

        axes[-1].set_xlabel('Iteration')
        fig.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            fig.savefig(str(Path(output_path).with_suffix('.pdf')), bbox_inches='tight')
        return fig

    def save_all_plots(
        self,
        report: DiagnosticsReport,
        best_params: Dict[str, float],
        output_dir: str,
    ) -> List[Path]:
        """Generate and save all diagnostic plots."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved = []

        if report.multi_start is not None:
            path = out / "diag_multi_start.png"
            self.plot_multi_start(report.multi_start, str(path))
            plt.close()
            saved.append(path)

        if report.profiles is not None:
            path = out / "diag_profiles.png"
            self.plot_profiles(report.profiles, best_params, str(path))
            plt.close()
            saved.append(path)

        if report.contours is not None:
            path = out / "diag_contours.png"
            self.plot_contours(report.contours, best_params, str(path))
            plt.close()
            saved.append(path)

        if report.hessian is not None:
            path = out / "diag_hessian.png"
            self.plot_hessian(report.hessian, str(path))
            plt.close()
            saved.append(path)

        if report.convergence is not None:
            path = out / "diag_convergence.png"
            self.plot_convergence(report.convergence, str(path))
            plt.close()
            saved.append(path)

        return saved

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _latin_hypercube(self, n: int, rng) -> np.ndarray:
        """Generate Latin Hypercube samples within bounds."""
        samples = np.zeros((n, self.n_params))
        for d in range(self.n_params):
            perm = rng.permutation(n)
            for i in range(n):
                lo = perm[i] / n
                hi = (perm[i] + 1) / n
                u = rng.uniform(lo, hi)
                samples[i, d] = self._lb[d] + u * (self._ub[d] - self._lb[d])
        return samples

    def _cluster_solutions(
        self,
        points: np.ndarray,
        tol: float,
    ) -> List[int]:
        """Simple distance-based clustering of converged solutions."""
        n = len(points)
        # Normalise by range
        ranges = self._ub - self._lb
        ranges = np.where(ranges > 0, ranges, 1.0)
        normed = (points - self._lb) / ranges

        labels = [-1] * n
        cluster_id = 0

        for i in range(n):
            if labels[i] >= 0:
                continue
            labels[i] = cluster_id
            for j in range(i + 1, n):
                if labels[j] >= 0:
                    continue
                dist = np.max(np.abs(normed[i] - normed[j]))
                if dist < tol:
                    labels[j] = cluster_id
            cluster_id += 1

        return labels

    def _classify_contour_correlation(
        self,
        Z: np.ndarray,
        x_vals: np.ndarray,
        y_vals: np.ndarray,
    ) -> str:
        """Classify whether a contour shows positive, negative, or no correlation."""
        # Find the valley: points within 10% of minimum objective
        Z_valid = Z[~np.isnan(Z)]
        if len(Z_valid) == 0:
            return "unknown"

        z_min = np.min(Z_valid)
        z_thresh = z_min + 0.1 * (np.max(Z_valid) - z_min)

        # Get (x, y) coordinates of points in the valley
        valley_mask = Z <= z_thresh
        y_idx, x_idx = np.where(valley_mask)

        if len(x_idx) < 3:
            return "none"

        x_coords = x_vals[x_idx]
        y_coords = y_vals[y_idx]

        # Normalise
        x_norm = (x_coords - np.mean(x_coords)) / (np.std(x_coords) + 1e-30)
        y_norm = (y_coords - np.mean(y_coords)) / (np.std(y_coords) + 1e-30)

        corr = np.corrcoef(x_norm, y_norm)[0, 1]

        if corr > 0.5:
            return "positive"
        elif corr < -0.5:
            return "negative"
        else:
            return "none"
