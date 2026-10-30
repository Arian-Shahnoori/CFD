from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.config import (
    DOMAIN_LENGTH_X,
    DOMAIN_LENGTH_Y,
    TIME_STEP,
    PECLET_NUMBER,
    BRINKMAN_NUMBER,
    MEAN_VELOCITY,
    RESIDUAL_TOLERANCE,
    MAX_ITERATIONS,
)
from src.diagnostics import (
    l2_error,
    steady_state_residual_rms,
)
from src.initialization import (
    initialize_temperature,
    initialize_velocity_fields,
)
from src.solvers.apf import solve_with_apf
from src.visualization import plot_grid_convergence


# =========================================================
# Study configuration
# =========================================================

GRID_LEVELS = [
    (25, 10),
    (50, 20),
    (100, 40),
    #(200, 80),
]

CONVECTION_COEFFICIENT = 1.0

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIRECTORY = PROJECT_ROOT / "results"


# =========================================================
# Inlet boundary conditions
# =========================================================

def linear_inlet_temperature(y):
    """Linear inlet profile: T_in(y) = y."""
    return y


# =========================================================
# Solve on a specified grid
# =========================================================

def solve_on_grid(
    mesh,
    inlet_temperature,
    solver=solve_with_apf,
    time_step=None,
):
    """
    Solve the energy equation on a specified grid.

    Parameters
    ----------
    mesh : tuple[int, int]
        Number of cells in x and y.
    inlet_temperature : callable
        Prescribed inlet temperature profile.
    solver : callable
        Solver with the same interface and seven-element return
        structure as solve_with_afi().
    time_step : float or None
        Time step. If None, use TIME_STEP from src.config.
    """

    if time_step is None:
        time_step = TIME_STEP

    initial_temperature = initialize_temperature(
        mesh=mesh,
        domain_length_y=DOMAIN_LENGTH_Y,
    )

    u_velocity, v_velocity = initialize_velocity_fields(
        mesh=mesh,
        domain_length_y=DOMAIN_LENGTH_Y,
        mean_velocity=MEAN_VELOCITY,
    )

    result = solver(
        initial_temperature=initial_temperature,
        time_step=time_step,
        mesh=mesh,
        inlet_temperature=inlet_temperature,
        peclet_number=PECLET_NUMBER,
        brinkman_number=BRINKMAN_NUMBER,
        u_velocity=u_velocity,
        v_velocity=v_velocity,
        domain_length_x=DOMAIN_LENGTH_X,
        domain_length_y=DOMAIN_LENGTH_Y,
        mean_velocity=MEAN_VELOCITY,
        residual_tolerance=RESIDUAL_TOLERANCE,
        max_iterations=MAX_ITERATIONS,
        snapshot_interval=MAX_ITERATIONS + 1,
        convection_coefficient=CONVECTION_COEFFICIENT,
    )

    (
        temperature,
        iteration_history,
        residual_history,
        computation_time,
        temperature_history,
        snapshot_iterations,
    ) = result

    # Evaluate the final steady-state residual using the same
    # convection coefficient as the solver.
    final_residual = steady_state_residual_rms(
        mesh=mesh,
        temperature=temperature,
        peclet_number=PECLET_NUMBER,
        brinkman_number=BRINKMAN_NUMBER,
        u_velocity=u_velocity,
        v_velocity=v_velocity,
        domain_length_x=DOMAIN_LENGTH_X,
        domain_length_y=DOMAIN_LENGTH_Y,
        convection_coefficient=CONVECTION_COEFFICIENT,
    )

    return {
        "temperature": np.asarray(temperature),
        "residual": float(final_residual),
        "computation_time": float(computation_time),
        "iterations": len(iteration_history),
        "converged": bool(final_residual <= RESIDUAL_TOLERANCE),
    }


# =========================================================
# Restrict fine-grid solution to the next coarser grid
# =========================================================

def restrict_fine_to_coarse(fine_field, coarse_mesh):
    """
    Restrict a fine cell-centred field using 2x2 cell averaging.

    Both grid dimensions must double at each refinement.
    """

    nx, ny = coarse_mesh
    expected_shape = (2 * nx, 2 * ny)

    if fine_field.shape != expected_shape:
        raise ValueError(
            f"Expected fine-grid shape {expected_shape}, "
            f"received {fine_field.shape}."
        )

    return fine_field.reshape(nx, 2, ny, 2).mean(axis=(1, 3))


# =========================================================
# Calculate successive-grid difference norm
# =========================================================

def calculate_difference_norm(
    coarse_temperature,
    fine_temperature,
    coarse_mesh,
):
    """Calculate the RMS difference between successive-grid solutions."""

    coarse_interior = coarse_temperature[1:-1, 1:-1]
    fine_interior = fine_temperature[1:-1, 1:-1]

    fine_restricted = restrict_fine_to_coarse(
        fine_field=fine_interior,
        coarse_mesh=coarse_mesh,
    )

    difference = coarse_interior - fine_restricted

    return float(np.sqrt(np.mean(difference**2)))


# =========================================================
# Analysis 2: self-convergence
# Linear inlet condition T_in(y) = y
# RK2 / RK4 (also usable for AFI)
# =========================================================

def analyze_linear_inlet(
    solver=solve_with_apf,
    solver_name="AFI",
    time_step=None,
):
    """
    Estimate spatial self-convergence without an exact solution.

    Each solution must be sufficiently converged to steady state.
    """

    if time_step is None:
        time_step = TIME_STEP

    solutions = {}
    residuals = []
    iterations = []
    computation_times = []

    plot_path = (
        RESULTS_DIRECTORY
        / f"grid_self_convergence_{solver_name}.png"
    )

    data_path = (
        RESULTS_DIRECTORY
        / f"grid_self_convergence_{solver_name}.npz"
    )

    print("\n" + "=" * 80)
    print("GRID SELF-CONVERGENCE: LINEAR INLET")
    print(f"Solver: {solver_name}")
    print("=" * 80)

    print(
        f"Pe = {PECLET_NUMBER:g} | "
        f"Br = {BRINKMAN_NUMBER:g} | "
        f"dt = {time_step:g}"
    )

    print("Inlet condition: T_in(y) = y")
    print("-" * 80)

    print(
        f"{'Grid':>12} "
        f"{'Residual':>15} "
        f"{'Iterations':>12} "
        f"{'Time (s)':>10}"
    )

    for mesh in GRID_LEVELS:
        nx, ny = mesh

        print(f"Solving grid {nx} x {ny} ...")

        result = solve_on_grid(
            mesh=mesh,
            inlet_temperature=linear_inlet_temperature,
            solver=solver,
            time_step=time_step,
        )

        if not result["converged"]:
            raise RuntimeError(
                f"{solver_name} did not converge on grid "
                f"{nx} x {ny}. Final residual: "
                f"{result['residual']:.6e}. "
                "Do not interpret the grid-convergence order yet."
            )

        solutions[mesh] = result["temperature"]
        residuals.append(result["residual"])
        iterations.append(result["iterations"])
        computation_times.append(result["computation_time"])

        print(
            f"{nx:4d} x {ny:<5d} "
            f"{result['residual']:15.6e} "
            f"{result['iterations']:12d} "
            f"{result['computation_time']:10.2f}"
        )

    # -----------------------------------------------------
    # Successive-grid difference norms
    # -----------------------------------------------------

    difference_norms = []

    print("\nSuccessive-grid difference norms")
    print("-" * 80)
    print(f"{'Grid pair':<28}{'Difference norm':>20}")

    for index in range(len(GRID_LEVELS) - 1):
        coarse_mesh = GRID_LEVELS[index]
        fine_mesh = GRID_LEVELS[index + 1]

        difference_norm = calculate_difference_norm(
            coarse_temperature=solutions[coarse_mesh],
            fine_temperature=solutions[fine_mesh],
            coarse_mesh=coarse_mesh,
        )

        difference_norms.append(difference_norm)

        nx1, ny1 = coarse_mesh
        nx2, ny2 = fine_mesh

        print(
            f"{nx1}x{ny1} vs {nx2}x{ny2:<12}"
            f"{difference_norm:20.6e}"
        )

    # -----------------------------------------------------
    # Pairwise observed orders
    # -----------------------------------------------------

    observed_orders = []

    print("\nObserved order estimates")
    print("-" * 65)

    for index in range(len(difference_norms) - 1):
        d_coarse = difference_norms[index]
        d_fine = difference_norms[index + 1]

        if (
            not np.isfinite(d_coarse)
            or not np.isfinite(d_fine)
            or d_coarse <= 0.0
            or d_fine <= 0.0
        ):
            order = float("nan")
        else:
            order = float(np.log(d_coarse / d_fine) / np.log(2.0))

        observed_orders.append(order)

        grid_1 = GRID_LEVELS[index]
        grid_2 = GRID_LEVELS[index + 1]
        grid_3 = GRID_LEVELS[index + 2]

        print(
            f"Grids {grid_1}, {grid_2}, {grid_3}: "
            f"p = {order:.4f}"
        )

    # -----------------------------------------------------
    # Fitted order and self-convergence plot
    # -----------------------------------------------------

    # Characteristic spacing h = sqrt(dx * dy).
    # It halves at every refinement for the configured grids.
    characteristic_spacings = np.array([
        np.sqrt(
            (DOMAIN_LENGTH_X / nx)
            * (DOMAIN_LENGTH_Y / ny)
        )
        for nx, ny in GRID_LEVELS[:-1]
    ])

    difference_array = np.asarray(difference_norms, dtype=float)

    valid = (
        np.isfinite(characteristic_spacings)
        & np.isfinite(difference_array)
        & (characteristic_spacings > 0.0)
        & (difference_array > 0.0)
    )

    if np.count_nonzero(valid) >= 2:
        fitted_slope, fitted_intercept = np.polyfit(
            np.log(characteristic_spacings[valid]),
            np.log(difference_array[valid]),
            1,
        )
        fitted_order = float(fitted_slope)
    else:
        fitted_slope = float("nan")
        fitted_intercept = float("nan")
        fitted_order = float("nan")

    plt.figure(figsize=(7, 5))

    plt.loglog(
        characteristic_spacings,
        difference_array,
        "o-",
        label="Successive-grid RMS difference",
    )

    if np.isfinite(fitted_order):
        fitted_values = np.exp(
            fitted_intercept
            + fitted_slope * np.log(characteristic_spacings[valid])
        )

        plt.loglog(
            characteristic_spacings[valid],
            fitted_values,
            "--",
            label=f"Log-log fit: p = {fitted_order:.3f}",
        )

    plt.xlabel(r"Characteristic grid spacing $h$")
    plt.ylabel(r"Successive-grid difference $D$")
    plt.title(f"Spatial self-convergence ({solver_name})")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    np.savez_compressed(
        data_path,
        grids=np.asarray(GRID_LEVELS, dtype=int),
        difference_norms=difference_array,
        characteristic_spacings=characteristic_spacings,
        observed_orders=np.asarray(observed_orders),
        fitted_order=fitted_order,
        residuals=np.asarray(residuals),
        iterations=np.asarray(iterations, dtype=int),
        computation_times=np.asarray(computation_times),
        time_step=time_step,
        peclet_number=PECLET_NUMBER,
        brinkman_number=BRINKMAN_NUMBER,
    )

    print("\nLinear Inlet Results")
    print("-" * 45)
    print(f"Fitted self-convergence order: {fitted_order:.4f}")
    print(
        "Pairwise observed orders: "
        + ", ".join(f"{order:.4f}" for order in observed_orders)
    )
    print(f"Plot saved to: {plot_path}")
    print(f"Data saved to: {data_path}")

    return {
        "difference_norms": difference_norms,
        "observed_orders": observed_orders,
        "fitted_order": fitted_order,
        "residuals": residuals,
        "iterations": iterations,
        "computation_times": computation_times,
    }


# =========================================================
# Main entry point
# =========================================================

def run_grid_convergence_study(
    solver=solve_with_apf,
    solver_name="APF",
    time_step=None,
):
    """
    Run the analyses appropriate for the selected solver.

    AFI:
        1. Fully developed inlet: analytical error.
        2. Linear inlet: spatial self-convergence.

    RK2 / RK4:
        Linear inlet: spatial self-convergence only.

    The solver must use the common seven-element return interface.
    """

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    if time_step is None:
        time_step = TIME_STEP

    print("\n" + "=" * 80)
    print(f"GRID-CONVERGENCE STUDY: {solver_name}")
    print("=" * 80)


    linear_results = analyze_linear_inlet(
        solver=solver,
        solver_name=solver_name,
        time_step=time_step,
    )

    print("\n" + "=" * 80)
    print(f"GRID-CONVERGENCE STUDY COMPLETED: {solver_name}")
    print("=" * 80)


    print(
        "Linear inlet fitted self-convergence order: "
        f"{linear_results['fitted_order']:.4f}"
    )

    return linear_results


