from pathlib import Path
import numpy as np

from src.analytical import exact_solution
from src.errors import rms_l2_error, observed_order
from src.solvers.solver import solve_advection
from src.visualization import plot_error_comparison


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Physical problem
LENGTH = 1.0
VELOCITY = 2.0
FINAL_TIME = 1.0
SPATIAL_SCHEME = "2nd_upwind"

# Experiment A: temporal refinement on a fixed grid.
TEMPORAL_CELL_COUNT = 40
TIME_STEP_COUNTS = [200, 400, 800, 1600, 3200]

# Experiment B: grid refinement at a low, constant CFL.
GRID_CELL_COUNTS = [20, 40, 80, 160]
GRID_CFL = 0.05

TIME_SCHEMES = {
    "RK2": "rk2",
    "RK4": "rk4",
}


def run_case(n_cells, temporal_scheme, n_steps=None, cfl=0.25):
    """Run one case and calculate its error and actual CFL."""

    x, T_num, dt, actual_steps = solve_advection(
        n_cells=n_cells,
        final_time=FINAL_TIME,
        cfl=cfl,
        length=LENGTH,
        u=VELOCITY,
        n_steps=n_steps,
        spatial_scheme=SPATIAL_SCHEME,
        temporal_scheme=temporal_scheme,
    )

    T_exact = exact_solution(x, FINAL_TIME, VELOCITY)
    error = rms_l2_error(T_num, T_exact)

    dx = LENGTH / n_cells
    actual_cfl = VELOCITY * dt / dx

    return {
        "x": x,
        "numerical_solution": T_num,
        "exact_solution": T_exact,
        "dt": dt,
        "n_steps": actual_steps,
        "cfl": actual_cfl,
        "error": error,
    }


def temporal_refinement_study():
    """
    Fix the spatial grid and decrease dt.

    The error should eventually approach a spatial-discretization
    floor as temporal error becomes less significant.
    """

    errors = {"RK2": [], "RK4": []}

    print("\nExperiment A: Temporal refinement")
    print(f"Fixed grid: {TEMPORAL_CELL_COUNT} cells")
    print("-" * 78)
    print(
        f"{'Steps':>8} {'dt':>12} {'CFL':>9} "
        f"{'RK2 RMS L2':>17} {'RK4 RMS L2':>17}"
    )

    for n_steps in TIME_STEP_COUNTS:
        rk2 = run_case(
            n_cells=TEMPORAL_CELL_COUNT,
            temporal_scheme="rk2",
            n_steps=n_steps,
        )

        rk4 = run_case(
            n_cells=TEMPORAL_CELL_COUNT,
            temporal_scheme="rk4",
            n_steps=n_steps,
        )

        errors["RK2"].append(rk2["error"])
        errors["RK4"].append(rk4["error"])

        print(
            f"{n_steps:8d} "
            f"{rk2['dt']:12.6e} "
            f"{rk2['cfl']:9.4f} "
            f"{rk2['error']:17.6e} "
            f"{rk4['error']:17.6e}"
        )

    print("-" * 78)

    return errors


def grid_refinement_study():
    """
    Refine the grid while keeping CFL low and constant.

    Compare the observed convergence orders of RK2 and RK4.
    """

    errors_by_scheme = {"RK2": [], "RK4": []}
    time_steps = []
    actual_cfls = []

    print("\nExperiment B: Grid refinement")
    print(f"Target CFL: {GRID_CFL}")
    print("-" * 86)
    print(
        f"{'Cells':>8} {'Steps':>8} {'dt':>12} {'CFL':>9} "
        f"{'RK2 RMS L2':>17} {'RK4 RMS L2':>17}"
    )

    for n_cells in GRID_CELL_COUNTS:
        # The same grid and time step are used for RK2 and RK4.
        rk2 = run_case(
            n_cells=n_cells,
            temporal_scheme="rk2",
            cfl=GRID_CFL,
        )

        rk4 = run_case(
            n_cells=n_cells,
            temporal_scheme="rk4",
            cfl=GRID_CFL,
        )

        errors_by_scheme["RK2"].append(rk2["error"])
        errors_by_scheme["RK4"].append(rk4["error"])

        time_steps.append(rk2["n_steps"])
        actual_cfls.append(rk2["cfl"])

        print(
            f"{n_cells:8d} "
            f"{rk2['n_steps']:8d} "
            f"{rk2['dt']:12.6e} "
            f"{rk2['cfl']:9.4f} "
            f"{rk2['error']:17.6e} "
            f"{rk4['error']:17.6e}"
        )

    print("-" * 86)

    # Fit log(error) = slope * log(number of cells) + intercept.
    orders_by_scheme = {}
    fitted_errors_by_scheme = {}

    for scheme, errors in errors_by_scheme.items():
        order = observed_order(GRID_CELL_COUNTS, errors)
        orders_by_scheme[scheme] = order

        slope, intercept = np.polyfit(
            np.log(GRID_CELL_COUNTS),
            np.log(errors),
            1,
        )

        fitted_errors_by_scheme[scheme] = np.exp(
            intercept + slope * np.log(GRID_CELL_COUNTS)
        )

        print(f"{scheme} observed convergence order: {order:.4f}")

    print()

    return {
        "cell_counts": GRID_CELL_COUNTS,
        "time_steps": time_steps,
        "actual_cfls": actual_cfls,
        "errors_by_scheme": errors_by_scheme,
        "orders_by_scheme": orders_by_scheme,
        "fitted_errors_by_scheme": fitted_errors_by_scheme,
    }


def run_study_07_rk2_rk4_accuracy_comparison():
    """Compare RK2 and classical RK4 temporal and overall accuracy."""

    # Experiment A: time-step refinement at fixed spatial resolution.
    temporal_errors = temporal_refinement_study()

    # Experiment B: spatial grid refinement.
    grid_results = grid_refinement_study()

    # Plot spatial convergence for both time integrators.
    figure_path = (
        PROJECT_ROOT
        / "results"
        / "figures"
        / "rk2_rk4_spatial_convergence.png"
    )

    plot_error_comparison(
        cell_counts=grid_results["cell_counts"],
        errors_by_scheme=grid_results["errors_by_scheme"],
        fitted_errors_by_scheme=grid_results["fitted_errors_by_scheme"],
        orders_by_scheme=grid_results["orders_by_scheme"],
        title="Second-Order Upwind: RK2 vs Classical RK4",
        save_path=figure_path,
    )

    return {
        "temporal_refinement_errors": temporal_errors,
        **grid_results,
    }


if __name__ == "__main__":
    run_study_07_rk2_rk4_accuracy_comparison()