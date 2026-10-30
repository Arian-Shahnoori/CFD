
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.config import (
    DOMAIN_LENGTH_X,
    DOMAIN_LENGTH_Y,
    PECLET_NUMBER,
    BRINKMAN_NUMBER,
    MEAN_VELOCITY,
    MAX_ITERATIONS,
)

from src.initialization import (
    initialize_temperature,
    initialize_velocity_fields,
)

from src.solvers.apf import solve_with_apf


# =========================================================
# Study configuration
# =========================================================

# Keep the spatial grid fixed throughout the study.
MESH = (25, 10)

# Temporal study settings: independent of src.config.TIME_STEP.
INITIAL_TIME_STEP = 0.05
NUMBER_OF_TIME_STEPS = 4
FINAL_TIME = 0.5

CONVECTION_COEFFICIENT = 1.0

# Set to zero to prevent normal positive-residual
# steady-state convergence from stopping the transient early.
TEMPORAL_RESIDUAL_TOLERANCE = 0.0

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIRECTORY = PROJECT_ROOT / "results"


# =========================================================
# Inlet boundary condition
# =========================================================

def linear_inlet_temperature(y):
    """Linear inlet profile: T_in(y) = y."""
    return y


# =========================================================
# RMS difference between two solutions
# =========================================================

def calculate_difference_norm(
    temperature_coarse_dt,
    temperature_fine_dt,
):
    """
    Calculate the RMS difference between two solutions
    computed at the same final physical time.

    Ghost cells are excluded.
    """

    coarse = temperature_coarse_dt[1:-1, 1:-1]
    fine = temperature_fine_dt[1:-1, 1:-1]

    if coarse.shape != fine.shape:
        raise ValueError(
            "Solutions must have the same physical-grid shape."
        )

    difference = coarse - fine

    if not np.all(np.isfinite(difference)):
        raise FloatingPointError(
            "Non-finite values encountered when comparing solutions."
        )

    return float(np.sqrt(np.mean(difference**2)))


# =========================================================
# Run a solver to a fixed physical time
# =========================================================

def solve_to_fixed_time(
    mesh,
    solver,
    time_step,
    final_time,
):
    """
    Advance the selected solver for a fixed number of steps.

    The requested step count is chosen such that:
        number_of_steps * time_step = final_time

    The solver must have the same calling interface and
    seven-element return structure as solve_with_afi().
    """

    ratio = final_time / time_step
    number_of_steps = int(round(ratio))

    if not np.isclose(
        number_of_steps * time_step,
        final_time,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError(
            f"Final time {final_time} is not an integer multiple "
            f"of dt={time_step}. Choose compatible values."
        )

    if number_of_steps < 1:
        raise ValueError("The requested number of steps must be positive.")

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
        inlet_temperature=linear_inlet_temperature,
        peclet_number=PECLET_NUMBER,
        brinkman_number=BRINKMAN_NUMBER,
        u_velocity=u_velocity,
        v_velocity=v_velocity,
        domain_length_x=DOMAIN_LENGTH_X,
        domain_length_y=DOMAIN_LENGTH_Y,
        mean_velocity=MEAN_VELOCITY,
        residual_tolerance=TEMPORAL_RESIDUAL_TOLERANCE,
        max_iterations=number_of_steps,
        snapshot_interval=number_of_steps + 1,
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

    actual_iterations = len(iteration_history)

    # Do not compare solutions reached after different physical times.
    if actual_iterations != number_of_steps:
        raise RuntimeError(
            f"Solver executed {actual_iterations} iterations, "
            f"but {number_of_steps} were required for dt={time_step}. "
            "Check whether the solver stopped early due to its "
            "steady-state convergence criterion."
        )

    temperature = np.asarray(temperature).copy()

    if not np.all(np.isfinite(temperature)):
        raise FloatingPointError(
            f"Non-finite temperature encountered for dt={time_step}."
        )

    final_residual = (
        float(residual_history[-1])
        if len(residual_history) > 0
        else float("nan")
    )

    return {
        "temperature": temperature,
        "iterations": actual_iterations,
        "computation_time": float(computation_time),
        "final_residual": final_residual,
        "time_step": time_step,
        "final_time": number_of_steps * time_step,
    }


# =========================================================
# Temporal accuracy / self-convergence study
# =========================================================

def run_temporal_accuracy_study(
    solver=solve_with_apf,
    solver_name="AFI",
    mesh=MESH,
    initial_time_step=INITIAL_TIME_STEP,
    number_of_time_steps=NUMBER_OF_TIME_STEPS,
    final_time=FINAL_TIME,
):
    """
    Estimate temporal order using successive solution differences.

    For dt, dt/2, dt/4, and dt/8, calculate

        D1 = ||T_dt     - T_dt/2||_RMS
        D2 = ||T_dt/2   - T_dt/4||_RMS
        D3 = ||T_dt/4   - T_dt/8||_RMS

    All solutions use the same mesh and final physical time.

    The observed order is estimated using

        p = log(D_i / D_(i+1)) / log(2).
    """

    if initial_time_step <= 0:
        raise ValueError("Initial time step must be positive.")

    if number_of_time_steps < 3:
        raise ValueError(
            "At least three time-step levels are required "
            "to estimate temporal order."
        )

    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    nx, ny = mesh

    time_steps = np.array([
        initial_time_step / (2**level)
        for level in range(number_of_time_steps)
    ])

    solutions = []
    iteration_counts = []
    computation_times = []
    final_residuals = []

    print("\n" + "=" * 80)
    print(f"TEMPORAL ACCURACY STUDY: {solver_name}")
    print("=" * 80)

    print(f"Mesh: {nx} x {ny}")
    print(f"Final physical time: {final_time:g}")
    print(f"Pe = {PECLET_NUMBER:g}")
    print(f"Br = {BRINKMAN_NUMBER:g}")
    print(f"Mean velocity = {MEAN_VELOCITY:g}")
    print("Inlet: T_in(y) = y")
    print("-" * 80)

    print(
        f"{'dt':>12} "
        f"{'Steps':>10} "
        f"{'Final time':>12} "
        f"{'Residual':>15} "
        f"{'Time (s)':>12}"
    )

    # -----------------------------------------------------
    # 1. Compute every solution on the same mesh and at t_f.
    # -----------------------------------------------------

    for time_step in time_steps:

        result = solve_to_fixed_time(
            mesh=mesh,
            solver=solver,
            time_step=float(time_step),
            final_time=final_time,
        )

        solutions.append(result["temperature"])
        iteration_counts.append(result["iterations"])
        computation_times.append(result["computation_time"])
        final_residuals.append(result["final_residual"])

        print(
            f"{time_step:12.6g} "
            f"{result['iterations']:10d} "
            f"{result['final_time']:12.6g} "
            f"{result['final_residual']:15.6e} "
            f"{result['computation_time']:12.3f}"
        )

    # -----------------------------------------------------
    # 2. Successive-time-step RMS difference norms.
    # -----------------------------------------------------

    difference_norms = []

    print("\nSuccessive-time-step difference norms")
    print("-" * 65)
    print(
        f"{'Time-step pair':>26} "
        f"{'RMS difference':>20}"
    )

    for i in range(len(time_steps) - 1):

        difference = calculate_difference_norm(
            temperature_coarse_dt=solutions[i],
            temperature_fine_dt=solutions[i + 1],
        )

        difference_norms.append(difference)

        print(
            f"{time_steps[i]:.6g} vs {time_steps[i + 1]:.6g}"
            f"{difference:20.6e}"
        )

    difference_norms = np.asarray(difference_norms, dtype=float)

    # -----------------------------------------------------
    # 3. Pairwise observed temporal orders.
    # -----------------------------------------------------

    observed_orders = []

    print("\nObserved temporal orders")
    print("-" * 65)

    for i in range(len(difference_norms) - 1):

        d_coarse = difference_norms[i]
        d_fine = difference_norms[i + 1]

        if (
            not np.isfinite(d_coarse)
            or not np.isfinite(d_fine)
            or d_coarse <= 0.0
            or d_fine <= 0.0
        ):
            order = float("nan")
        else:
            order = float(
                np.log(d_coarse / d_fine)
                / np.log(time_steps[i] / time_steps[i + 1])
            )

        observed_orders.append(order)

        print(
            f"dt levels {i + 1} and {i + 2}: "
            f"p = {order:.5f}"
        )

    observed_orders = np.asarray(observed_orders, dtype=float)

    # -----------------------------------------------------
    # 4. Fit log(D) against log(dt).
    #
    # D_i = ||T(dt_i) - T(dt_i/2)||.
    # In the asymptotic regime, D_i is proportional to dt_i^p.
    # -----------------------------------------------------

    difference_time_steps = time_steps[:-1]

    valid = (
        np.isfinite(difference_norms)
        & np.isfinite(difference_time_steps)
        & (difference_norms > 0.0)
        & (difference_time_steps > 0.0)
    )

    if np.count_nonzero(valid) >= 2:

        fitted_slope, fitted_intercept = np.polyfit(
            np.log(difference_time_steps[valid]),
            np.log(difference_norms[valid]),
            1,
        )

        fitted_order = float(fitted_slope)

    else:
        fitted_slope = float("nan")
        fitted_intercept = float("nan")
        fitted_order = float("nan")

    # -----------------------------------------------------
    # 5. Plot temporal self-convergence.
    # -----------------------------------------------------

    plot_path = (
        RESULTS_DIRECTORY
        / f"temporal_self_convergence_{solver_name}.png"
    )

    data_path = (
        RESULTS_DIRECTORY
        / f"temporal_self_convergence_{solver_name}.npz"
    )

    plt.figure(figsize=(7, 5))

    plt.loglog(
        difference_time_steps,
        difference_norms,
        "o-",
        label="Successive-time-step RMS difference",
    )

    if np.isfinite(fitted_order):

        fitted_values = np.exp(
            fitted_intercept
            + fitted_slope * np.log(difference_time_steps[valid])
        )

        plt.loglog(
            difference_time_steps[valid],
            fitted_values,
            "--",
            label=f"Log-log fit: p = {fitted_order:.3f}",
        )

    plt.xlabel(r"Time step $\Delta t$")
    plt.ylabel(r"Successive-solution RMS difference $D$")
    plt.title(f"Temporal self-convergence ({solver_name})")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # -----------------------------------------------------
    # 6. Save numerical results.
    # -----------------------------------------------------

    np.savez_compressed(
        data_path,
        solver_name=solver_name,
        mesh=np.asarray(mesh, dtype=int),
        final_time=final_time,
        time_steps=time_steps,
        iteration_counts=np.asarray(iteration_counts, dtype=int),
        computation_times=np.asarray(computation_times),
        final_residuals=np.asarray(final_residuals),
        difference_time_steps=difference_time_steps,
        difference_norms=difference_norms,
        observed_orders=observed_orders,
        fitted_order=fitted_order,
        peclet_number=PECLET_NUMBER,
        brinkman_number=BRINKMAN_NUMBER,
        mean_velocity=MEAN_VELOCITY,
    )

    # -----------------------------------------------------
    # 7. Summary.
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("TEMPORAL ACCURACY RESULTS")
    print("=" * 65)

    print(f"Solver: {solver_name}")
    print(f"Grid: {nx} x {ny}")
    print(f"Final time: {final_time:g}")
    print(f"Fitted temporal order: {fitted_order:.5f}")
    print(
        "Pairwise observed orders: "
        + ", ".join(f"{p:.5f}" for p in observed_orders)
    )

    print(f"Plot saved to: {plot_path}")
    print(f"Data saved to: {data_path}")

    return {
        "time_steps": time_steps,
        "difference_norms": difference_norms,
        "observed_orders": observed_orders,
        "fitted_order": fitted_order,
        "iteration_counts": iteration_counts,
        "computation_times": computation_times,
        "final_residuals": final_residuals,
    }


# =========================================================
# Main entry point
# =========================================================

if __name__ == "__main__":

    # Default: AFI.
    run_temporal_accuracy_study(
        solver=solve_with_apf,
        solver_name="AFI",
    )

    # To test RK2 or RK4, import your corresponding solver and call:
    #
    # run_temporal_accuracy_study(
    #     solver=your_rk2_solver,
    #     solver_name="RK2",
    # )
    #
    # run_temporal_accuracy_study(
    #     solver=your_rk4_solver,
    #     solver_name="RK4",
    # )
