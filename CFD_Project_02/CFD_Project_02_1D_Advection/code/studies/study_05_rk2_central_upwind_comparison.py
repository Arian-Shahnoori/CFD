from pathlib import Path
import numpy as np

from src.analytical import exact_solution
from src.errors import rms_l2_error
from src.solvers.solver import solve_advection
from src.visualization import plot_solution_comparison


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Study settings
CELL_COUNT = 40
FINAL_TIME = 1.0
LENGTH = 1.0
VELOCITY = 2.0

CFL_VALUES = [0.7, 0.4, 0.1]

SPATIAL_SCHEMES = {
    "2nd_upwind": "Second-order upwind",
    "2nd_central": "Second-order central",
}


def run_study_05_rk2_central_upwind_comparison():
    """Compare central and upwind schemes at different CFL values."""

    dx = LENGTH / CELL_COUNT
    all_results = {}

    print("\nRK2: Second-Order Central vs Upwind")
    print("=" * 100)
    print(
        f"{'Requested CFL':>13} "
        f"{'Actual CFL':>11} "
        f"{'Steps':>7} "
        f"{'Scheme':>24} "
        f"{'RMS L2 error':>15} "
        f"{'Max error':>13} "
        f"{'Max |T|':>11}"
    )
    print("-" * 100)

    for requested_cfl in CFL_VALUES:

        # Choose an integer number of steps to reach FINAL_TIME exactly.
        target_dt = requested_cfl * dx / VELOCITY
        n_steps = max(1, round(FINAL_TIME / target_dt))

        numerical_solutions = {}
        x = None

        # Run both spatial schemes with the same mesh and time step.
        for scheme, scheme_label in SPATIAL_SCHEMES.items():
            CFL = VELOCITY * (FINAL_TIME / n_steps) / (LENGTH / CELL_COUNT)
            x, T_num, dt, actual_steps = solve_advection(
                n_cells=CELL_COUNT,
                final_time=FINAL_TIME,
                cfl=CFL,
                length=LENGTH,
                u=VELOCITY,
                spatial_scheme="2nd_upwind",
                temporal_scheme="rk2",
            )

            actual_cfl = VELOCITY * dt / dx
            numerical_solutions[scheme_label] = T_num

        # Analytical solution at the same cell centres and final time.
        T_exact = exact_solution(x, FINAL_TIME, VELOCITY)

        # Calculate diagnostics for each numerical solution.
        diagnostics = {}

        for scheme_label, T_num in numerical_solutions.items():
            l2_error = rms_l2_error(T_num, T_exact)
            max_error = float(np.max(np.abs(T_num - T_exact)))
            max_abs_T = float(np.max(np.abs(T_num)))

            diagnostics[scheme_label] = {
                "rms_l2_error": l2_error,
                "max_error": max_error,
                "max_abs_T": max_abs_T,
            }

            print(
                f"{requested_cfl:13.3f} "
                f"{actual_cfl:11.4f} "
                f"{actual_steps:7d} "
                f"{scheme_label:>24} "
                f"{l2_error:15.6e} "
                f"{max_error:13.6e} "
                f"{max_abs_T:11.6f}"
            )

        all_results[requested_cfl] = {
            "x": x,
            "numerical_solutions": numerical_solutions,
            "exact_solution": T_exact,
            "diagnostics": diagnostics,
            "actual_cfl": actual_cfl,
            "n_steps": n_steps,
        }

        # Plot and save this CFL comparison.
        figure_path = (
            PROJECT_ROOT
            / "results"
            / "figures"
            / f"rk2_central_upwind_cfl_{requested_cfl:.1f}.png"
        )
        if requested_cfl == 0.7:
            ylimits = (-1.2,0.2)
        elif requested_cfl == 0.4:
            ylimits = (-1.2, -0.6)
        else:
            ylimits = (-1.2, -0.8)
        plot_solution_comparison(
            x=x,
            numerical_solutions=numerical_solutions,
            exact_solution=T_exact,
            title=(
                "RK2: Central vs Upwind | "
                f"Requested CFL = {requested_cfl:.1f}"
            ),
            x_label="x",
            y_label="T",
            save_path=figure_path,
            y_limits=ylimits,
        )

    print("=" * 100)

    return all_results


if __name__ == "__main__":
    run_study_05_rk2_central_upwind_comparison()

