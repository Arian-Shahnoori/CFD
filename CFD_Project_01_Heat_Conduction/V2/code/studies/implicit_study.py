import numpy as np

from src.initialization import initialize_temperature
from src.grid import create_grid
from src.exact_solution import exact_solution
from src.residual import residual_norm
from src.visualization import (
    plot_contours,
    plot_error,
    plot_solver_cpu_times,
)

from src.solvers.sor import sor
from src.solvers.line_sor import line_sor
from src.solvers.implicit import implicit_solver


# =============================================================================
# Implicit Solver and CPU-Time Study
# =============================================================================

def run_implicit_study():

    tolerance = 1e-12
    mesh_dimensions = [10, 20, 30, 40]
    omega_values = [1.3, 1.5]

    results = []
    implicit_solutions = {}

    # ============================================================
    # Run the solver comparisons
    # ============================================================

    for N in mesh_dimensions:

        # --------------------------------------------------------
        # Point SOR and Line-SOR
        # --------------------------------------------------------

        for solver_name, solver_function in [
            ("SOR", sor),
            ("Line-SOR", line_sor),
        ]:

            for omega in omega_values:

                T = initialize_temperature(N)

                (
                    T,
                    residual_history,
                    time_history,
                    iteration,
                ) = solver_function(
                    T,
                    N,
                    tolerance,
                    omega
                )

                results.append({
                    "N": N,
                    "solver": solver_name,
                    "omega": omega,
                    "iterations": iteration,
                    "cpu_time": time_history[-1],
                    "final_residual": residual_history[-1],
                    "residual_history": residual_history,
                    "time_history": time_history,
                })

        # --------------------------------------------------------
        # Fully implicit direct solver
        # --------------------------------------------------------

        T, cpu_time = implicit_solver(N)

        final_residual = residual_norm(T, N)

        results.append({
            "N": N,
            "solver": "Implicit",
            "omega": None,
            "iterations": None,
            "cpu_time": cpu_time,
            "final_residual": final_residual,
            "residual_history": None,
            "time_history": None,
        })

        implicit_solutions[N] = T

    # ============================================================
    # Print the comparison table
    # ============================================================

    print("\nSolver CPU-Time Comparison")
    print("=" * 90)

    print(
        f"{'N':>5}"
        f"{'Solver':<14}"
        f"{'Omega':>9}"
        f"{'Iterations':>14}"
        f"{'CPU Time [s]':>17}"
        f"{'Final Residual':>21}"
    )

    print("-" * 90)

    for result in results:

        omega = (
            "-"
            if result["omega"] is None
            else f'{result["omega"]:.1f}'
        )

        iterations = (
            "Direct"
            if result["iterations"] is None
            else str(result["iterations"])
        )

        print(
            f'{result["N"]:>5}'
            f'{result["solver"]:<14}'
            f'{omega:>9}'
            f'{iterations:>14}'
            f'{result["cpu_time"]:>17.4f}'
            f'{result["final_residual"]:>21.2e}'
        )

    print("=" * 90)

    # ============================================================
    # Verify the implicit solution against the exact solution
    # ============================================================

    N = mesh_dimensions[-1]

    _, X, Y = create_grid(N)

    T_numerical = implicit_solutions[N][1:N + 1, 1:N + 1]
    T_exact = exact_solution(X, Y)

    error = - T_numerical + T_exact

    L2_error = np.sqrt(np.mean(error**2))
    maximum_error = np.max(np.abs(error))

    print(f"\nImplicit Solver Verification, N = {N}")
    print("-" * 50)
    print(f"{'L2 Error':<25}{L2_error:>15.4e}")
    print(f"{'Maximum Absolute Error':<25}{maximum_error:>15.4e}")
    print("-" * 50)

    plot_contours(X, Y, T_numerical, T_exact)
    plot_error(X, Y, error)
    plot_solver_cpu_times(results)

    return results