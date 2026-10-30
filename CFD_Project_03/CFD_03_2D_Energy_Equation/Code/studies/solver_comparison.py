from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import config
from src.initialization import (
    initialize_temperature,
    initialize_velocity_fields,
)
from src.diagnostics import steady_state_residual_rms
from src.solvers.apf import solve_with_apf
from src.solvers.direct import solve_with_direct
from src.solvers.runge_kutta import solve_with_rk2, solve_with_rk4


# ============================================================
# Study configuration
# ============================================================

GRID = (config.CELL_COUNT_X, config.CELL_COUNT_Y)
REPEATS = 5

SELECTED_TIME_STEPS = {
    "AFI": 0.265,
    "Direct": 1.0,
    "RK2": 0.017,
    "RK4": 0.024,
}

SOLVERS = {
    "AFI": solve_with_apf,
    "Direct": solve_with_direct,
    "RK2": solve_with_rk2,
    "RK4": solve_with_rk4,
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "results" / "solver_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Inlet boundary condition
# ============================================================

def linear_inlet_temperature(y):
    """Linear inlet profile: T_in(y) = y."""
    return y


# ============================================================
# Fresh initial condition for every run
# ============================================================

def build_case():

    mesh = GRID

    initial_temperature = initialize_temperature(
        mesh=mesh,
        domain_length_y=config.DOMAIN_LENGTH_Y,
    )

    u_velocity, v_velocity = initialize_velocity_fields(
        mesh=mesh,
        domain_length_y=config.DOMAIN_LENGTH_Y,
        mean_velocity=config.MEAN_VELOCITY,
    )

    return {
        "mesh": mesh,
        "initial_temperature": initial_temperature,
        "inlet_temperature": linear_inlet_temperature,
        "u_velocity": u_velocity,
        "v_velocity": v_velocity,
        "domain_length_x": config.DOMAIN_LENGTH_X,
        "domain_length_y": config.DOMAIN_LENGTH_Y,
        "peclet_number": config.PECLET_NUMBER,
        "brinkman_number": config.BRINKMAN_NUMBER,
        "mean_velocity": config.MEAN_VELOCITY,
    }


# ============================================================
# Validate convergence histories
# ============================================================

def validate_histories(
    iteration_history,
    residual_history,
    time_history,
):
    """Check that the three convergence histories are consistent."""

    iterations = np.asarray(iteration_history, dtype=float)
    residuals = np.asarray(residual_history, dtype=float)
    times = np.asarray(time_history, dtype=float)

    if len(iterations) == 0:
        raise ValueError("Empty convergence history.")

    if not (
        len(iterations)
        == len(residuals)
        == len(times)
    ):
        raise ValueError(
            "Iteration, residual, and time histories "
            "must have identical lengths."
        )

    if not (
        np.all(np.isfinite(iterations))
        and np.all(np.isfinite(residuals))
        and np.all(np.isfinite(times))
    ):
        raise ValueError("Non-finite values in convergence histories.")

    if np.any(residuals < 0):
        raise ValueError("Residual history contains negative values.")

    if np.any(np.diff(iterations) <= 0):
        raise ValueError("Iteration history must be strictly increasing.")

    if np.any(np.diff(times) < 0):
        raise ValueError("Time history must be non-decreasing.")

    return iterations, residuals, times


# ============================================================
# Plot convergence histories
# ============================================================

def plot_convergence_histories(representative_runs):

    # --------------------------------------------------------
    # Residual versus iteration
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 5))

    for solver_name, run in representative_runs.items():

        residuals = run["residual_history"]

        # Zero residuals cannot be displayed on a logarithmic axis.
        plot_residuals = np.where(
            residuals > 0,
            residuals,
            np.nan,
        )

        dt = SELECTED_TIME_STEPS[solver_name]

        ax.semilogy(
            run["iteration_history"],
            plot_residuals,
            label=f"{solver_name} ($\\Delta t={dt:g}$)",
            linewidth=1.8,
        )

    ax.axhline(
        y=config.RESIDUAL_TOLERANCE,
        linestyle="--",
        linewidth=1.2,
        label="Convergence tolerance",
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Steady-state residual (RMS)")
    ax.set_title("Residual Convergence versus Iteration")

    ax.grid(True, which="both", linestyle=":")
    ax.legend()
    fig.tight_layout()

    iteration_plot_path = (
        OUTPUT_DIR / "residual_vs_iteration.png"
    )

    fig.savefig(
        iteration_plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Residual versus elapsed computation time
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 5))

    for solver_name, run in representative_runs.items():

        residuals = run["residual_history"]

        plot_residuals = np.where(
            residuals > 0,
            residuals,
            np.nan,
        )

        dt = SELECTED_TIME_STEPS[solver_name]

        ax.semilogy(
            run["time_history"],
            plot_residuals,
            label=f"{solver_name} ($\\Delta t={dt:g}$)",
            linewidth=1.8,
        )

    ax.axhline(
        y=config.RESIDUAL_TOLERANCE,
        linestyle="--",
        linewidth=1.2,
        label="Convergence tolerance",
    )

    ax.set_xlabel("Elapsed computation time (s)")
    ax.set_ylabel("Steady-state residual (RMS)")
    ax.set_title("Residual Convergence versus Computation Time")

    ax.grid(True, which="both", linestyle=":")
    ax.legend()
    fig.tight_layout()

    time_plot_path = (
        OUTPUT_DIR / "residual_vs_time.png"
    )

    fig.savefig(
        time_plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\nConvergence plots saved:")
    print(f"  {iteration_plot_path}")
    print(f"  {time_plot_path}")


# ============================================================
# Run the comparison
# ============================================================

def run_solver_comparison():

    records = []
    successful_runs_by_solver = {}

    nx, ny = GRID

    print("\n" + "=" * 90)
    print("STEADY-STATE SOLVER COMPARISON")
    print("=" * 90)

    print(f"Grid: {nx} x {ny}")
    print(f"Pe = {config.PECLET_NUMBER:g}")
    print(f"Br = {config.BRINKMAN_NUMBER:g}")
    print(
        f"Convection coefficient = "
        f"{config.convection_coefficient:g}"
    )
    print(
        f"Residual tolerance = "
        f"{config.RESIDUAL_TOLERANCE:.1e}"
    )
    print(f"Repeats per solver = {REPEATS}")

    # ========================================================
    # Execute each solver
    # ========================================================

    for solver_name, solver in SOLVERS.items():

        dt = SELECTED_TIME_STEPS[solver_name]

        successful_runs = []
        run_statuses = []
        failure_messages = []

        print(f"\nRunning {solver_name} with dt = {dt:g}")

        for repeat in range(REPEATS):

            case = build_case()

            try:

                # Measure the complete solver call, including
                # solver-specific setup and factorization.
                start = perf_counter()

                result = solver(
                    initial_temperature=case["initial_temperature"],
                    time_step=dt,
                    mesh=case["mesh"],
                    inlet_temperature=case["inlet_temperature"],
                    peclet_number=case["peclet_number"],
                    brinkman_number=case["brinkman_number"],
                    u_velocity=case["u_velocity"],
                    v_velocity=case["v_velocity"],
                    domain_length_x=case["domain_length_x"],
                    domain_length_y=case["domain_length_y"],
                    mean_velocity=case["mean_velocity"],
                    residual_tolerance=config.RESIDUAL_TOLERANCE,
                    max_iterations=config.MAX_ITERATIONS,
                    snapshot_interval=config.MAX_ITERATIONS + 1,
                    convection_coefficient=config.convection_coefficient,
                )

                runtime = perf_counter() - start

                (
                    temperature,
                    iteration_history,
                    residual_history,
                    time_history,
                    solver_computation_time,
                    temperature_history,
                    snapshot_iterations,
                ) = result

                temperature = np.asarray(temperature)

                # Validate the histories before storing them.
                iterations, residuals, elapsed_times = (
                    validate_histories(
                        iteration_history,
                        residual_history,
                        time_history,
                    )
                )

                # Independently evaluate the final steady-state residual.
                if np.all(np.isfinite(temperature)):

                    final_residual = float(
                        steady_state_residual_rms(
                            mesh=case["mesh"],
                            temperature=temperature,
                            peclet_number=case["peclet_number"],
                            brinkman_number=case["brinkman_number"],
                            u_velocity=case["u_velocity"],
                            v_velocity=case["v_velocity"],
                            domain_length_x=case["domain_length_x"],
                            domain_length_y=case["domain_length_y"],
                            convection_coefficient=(
                                config.convection_coefficient
                            ),
                        )
                    )

                else:
                    final_residual = np.inf

                converged = (
                    np.isfinite(final_residual)
                    and final_residual <= config.RESIDUAL_TOLERANCE
                )

                run_statuses.append(bool(converged))

                print(
                    f"  Run {repeat + 1}/{REPEATS}: "
                    f"{runtime:.4f} s, "
                    f"{len(iterations)} iterations, "
                    f"residual = {final_residual:.3e}, "
                    f"converged = {converged}"
                )

                if converged:

                    successful_runs.append({
                        "runtime": runtime,
                        "iterations": iterations,
                        "residual": residuals,
                        "time": elapsed_times,
                        "temperature": temperature.copy(),
                    })

                else:

                    failure_messages.append(
                        f"Run {repeat + 1}: "
                        f"did not reach the residual tolerance."
                    )

            except Exception as exc:

                run_statuses.append(False)

                failure_message = (
                    f"{type(exc).__name__}: {exc}"
                )

                failure_messages.append(
                    f"Run {repeat + 1}: {failure_message}"
                )

                print(
                    f"  Run {repeat + 1}/{REPEATS} failed: "
                    f"{failure_message}"
                )

        # Store successful runs for the performance statistics.
        successful_runs_by_solver[solver_name] = successful_runs

        all_runs_converged = (
            len(run_statuses) == REPEATS
            and all(run_statuses)
        )

        # ----------------------------------------------------
        # Calculate statistics using converged runs only
        # ----------------------------------------------------

        runtimes = [
            run["runtime"] for run in successful_runs
        ]

        iteration_counts = [
            len(run["iterations"]) for run in successful_runs
        ]

        final_residuals = [
            run["residual"][-1] for run in successful_runs
        ]

        records.append({
            "solver": solver_name,
            "dt": dt,

            "median_runtime_s": (
                float(np.median(runtimes))
                if runtimes else np.nan
            ),

            "runtime_std_s": (
                float(np.std(runtimes, ddof=1))
                if len(runtimes) > 1 else np.nan
            ),

            "median_iterations": (
                float(np.median(iteration_counts))
                if iteration_counts else np.nan
            ),

            "median_final_residual": (
                float(np.median(final_residuals))
                if final_residuals else np.nan
            ),

            "successful_runs": len(successful_runs),
            "all_runs_converged": all_runs_converged,

            "failure_messages": " | ".join(failure_messages),
        })

    # ========================================================
    # Select representative runs for convergence plots
    # ========================================================

    representative_runs = {}

    for solver_name, runs in successful_runs_by_solver.items():

        if not runs:
            print(
                f"\nNo converged runs available for "
                f"{solver_name}; its curves will be omitted."
            )
            continue

        runtimes = np.array([
            run["runtime"] for run in runs
        ])

        median_runtime = np.median(runtimes)

        # Select the successful run whose runtime is closest
        # to the median runtime of the repeated simulations.
        representative_index = int(
            np.argmin(np.abs(runtimes - median_runtime))
        )

        selected_run = runs[representative_index]

        representative_runs[solver_name] = {
            "iteration_history": selected_run["iterations"],
            "residual_history": selected_run["residual"],
            "time_history": selected_run["time"],
        }

    # ========================================================
    # Generate convergence plots
    # ========================================================

    if representative_runs:
        plot_convergence_histories(representative_runs)

    # ========================================================
    # Results table
    # ========================================================

    results = pd.DataFrame(records)

    csv_path = OUTPUT_DIR / "solver_comparison.csv"
    results.to_csv(csv_path, index=False)

    display = results[
        [
            "solver",
            "dt",
            "median_runtime_s",
            "runtime_std_s",
            "median_iterations",
            "median_final_residual",
            "successful_runs",
            "all_runs_converged",
        ]
    ].copy()

    display["dt"] = display["dt"].map(
        lambda value: f"{value:g}"
    )

    display["median_runtime_s"] = display[
        "median_runtime_s"
    ].map(
        lambda value: (
            f"{value:.4f}" if pd.notna(value) else "N/A"
        )
    )

    display["runtime_std_s"] = display[
        "runtime_std_s"
    ].map(
        lambda value: (
            f"{value:.4f}" if pd.notna(value) else "N/A"
        )
    )

    display["median_iterations"] = display[
        "median_iterations"
    ].map(
        lambda value: (
            f"{value:.0f}" if pd.notna(value) else "N/A"
        )
    )

    display["median_final_residual"] = display[
        "median_final_residual"
    ].map(
        lambda value: (
            f"{value:.3e}" if pd.notna(value) else "N/A"
        )
    )

    print("\n" + "=" * 110)
    print("COMPARISON TABLE")
    print("=" * 110)
    print(display.to_string(index=False))

    print("\nResults saved:")
    print(f"  {csv_path}")

    return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    results = run_solver_comparison()