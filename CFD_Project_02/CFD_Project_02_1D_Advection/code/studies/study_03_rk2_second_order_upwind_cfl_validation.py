from pathlib import Path

from src.analytical import exact_solution
from src.solvers.solver import solve_advection
from src.visualization import plot_solution_comparison


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Study settings
CELL_COUNT = 40
FINAL_TIME = 0.25
LENGTH = 1.0
VELOCITY = 2.0
STEP_COUNTS = [50, 40, 35]



def run_study_03_rk2_second_order_upwind_cfl_validation():
    """Compare RK2 solutions at different time-step sizes."""

    numerical_solutions = {}
    x = None

    print("\nRK2 + Second-Order Upwind: CFL Stability Check")
    print("-" * 62)
    print(
        f"{'Steps':>8} {'dt':>14} "
        f"{'Actual CFL':>14}"
    )

    for n_steps in STEP_COUNTS:
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

        dx = LENGTH / CELL_COUNT
        actual_cfl = VELOCITY * dt / dx

        label = f"N = {n_steps} (CFL = {actual_cfl:.3f})"
        numerical_solutions[label] = T_num

        print(
            f"{actual_steps:8d} {dt:14.6e} "
            f"{actual_cfl:14.4f}"
        )

    print("-" * 62)

    # Analytical solution at the same cell centres and final time.
    T_exact = exact_solution(x, FINAL_TIME, VELOCITY)

    # Plot and save the comparison using the shared visualization module.
    figure_path = (
        PROJECT_ROOT
        / "results"
        / "figures"
        / "rk2_second_order_upwind_cfl_validation.png"
    )

    plot_solution_comparison(
        x=x,
        numerical_solutions=numerical_solutions,
        exact_solution=T_exact,
        title="CFL Stability: RK2 + Second-Order Upwind",
        x_label="x",
        y_label="T",
        save_path=figure_path,
        y_limits=None,  
    )

    return {
        "cell_count": CELL_COUNT,
        "final_time": FINAL_TIME,
        "numerical_solutions": numerical_solutions,
        "exact_solution": T_exact,
    }


if __name__ == "__main__":
    run_study_03_rk2_second_order_upwind_cfl_validation()