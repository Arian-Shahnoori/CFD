from pathlib import Path

import numpy as np

from src.analytical import exact_solution
from src.errors import rms_l2_error, observed_order
from src.solvers.solver import solve_advection
from src.visualization import plot_convergence


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Study settings
CELL_COUNTS = [20, 40, 80, 160]
FINAL_TIME = 1.0
CFL = 0.25
LENGTH = 1.0
VELOCITY = 2.0


def run_study_01_rk2_second_order_upwind_convergence() -> dict:
    """Run, report, and visualize the grid-refinement study."""

    errors = []
    time_steps = []
    step_counts = []
    actual_cfls = []

    # Run the numerical solver at each grid resolution.
    for n_cells in CELL_COUNTS:
        x, T_num, dt, n_steps = solve_advection(
            n_cells=n_cells,
            final_time=FINAL_TIME,
            cfl=CFL,
            length=LENGTH,
            u=VELOCITY,
            spatial_scheme="2nd_upwind",
            temporal_scheme="rk2",
        )

        T_exact = exact_solution(x, FINAL_TIME, VELOCITY)
        error = rms_l2_error(T_num, T_exact)

        dx = LENGTH / n_cells

        errors.append(error)
        time_steps.append(dt)
        step_counts.append(n_steps)
        actual_cfls.append(VELOCITY * dt / dx)

    # Fit log(error) = slope * log(number of cells) + intercept.
    slope, intercept = np.polyfit(
        np.log(CELL_COUNTS),
        np.log(errors),
        1,
    )

    order = observed_order(CELL_COUNTS, errors)

    fitted_errors = np.exp(
        intercept + slope * np.log(CELL_COUNTS)
    )

    results = {
        "study_name": "RK2 + Second-Order Upwind Convergence",
        "cell_counts": CELL_COUNTS,
        "errors": errors,
        "time_steps": time_steps,
        "step_counts": step_counts,
        "actual_cfls": actual_cfls,
        "order": order,
        "fitted_errors": fitted_errors,
    }

    # Report the numerical results.
    print(f"\n{results['study_name']}")
    print("-" * 72)
    print(
        f"{'Cells':>8} {'dt':>12} {'Steps':>8} "
        f"{'CFL':>10} {'RMS L2 error':>16}"
    )

    for n, dt, steps, cfl, error in zip(
        results["cell_counts"],
        results["time_steps"],
        results["step_counts"],
        results["actual_cfls"],
        results["errors"],
    ):
        print(
            f"{n:8d} {dt:12.6e} {steps:8d} "
            f"{cfl:10.4f} {error:16.6e}"
        )

    print("-" * 72)
    print(f"Observed convergence order: {order:.4f}")

    # Generate and save the convergence figure.
    figure_path = (
        PROJECT_ROOT
        / "results"
        / "figures"
        / "rk2_second_order_upwind_convergence.png"
    )

    plot_convergence(
        cell_counts=results["cell_counts"],
        errors=results["errors"],
        order=results["order"],
        fitted_errors=results["fitted_errors"],
        title="RK2 + Second-Order Upwind: Grid Convergence",
        save_path=figure_path,
    )

    return results


if __name__ == "__main__":
    run_study_01_rk2_second_order_upwind_convergence()