from src.initialization import initialize_temperature

from src.solvers.gauss_seidel import gauss_seidel
from src.solvers.sor import sor

from src.visualization import plot_gs_sor_comparison


def run_gs_sor_comparison():

    tolerance = 1e-12
    N = 40
    omega_values = [1.1, 1.3, 1.5]

    results = []

    print("\nSolver Comparison")
    print("-" * 75)

    print(
        f"{'Solver':>12}"
        f"{'Omega':>10}"
        f"{'Iterations':>15}"
        f"{'CPU Time [s]':>18}"
        f"{'Final Residual':>20}"
    )

    print("-" * 75)

    # ============================================================
    # Point Gauss-Seidel
    # ============================================================

    T = initialize_temperature(N)

    T, residual_history, time_history, iteration = gauss_seidel(
        T,
        N,
        tolerance
    )

    cpu_time = time_history[-1]
    final_residual = residual_history[-1]

    results.append({
        "solver": "Gauss-Seidel",
        "omega": None,
        "iterations": iteration,
        "cpu_time": cpu_time,
        "residual_history": residual_history,
        "time_history": time_history,
        "final_residual": final_residual
    })

    print(
        f"{'Gauss-Seidel':>12}"
        f"{'-':>10}"
        f"{iteration:>15}"
        f"{cpu_time:>18.4f}"
        f"{final_residual:>20.2e}"
    )

    # ============================================================
    # Successive Over-Relaxation
    # ============================================================

    for omega in omega_values:

        T = initialize_temperature(N)

        T, residual_history, time_history, iteration = sor(
            T,
            N,
            tolerance,
            omega
        )

        cpu_time = time_history[-1]
        final_residual = residual_history[-1]

        results.append({
            "solver": "SOR",
            "omega": omega,
            "iterations": iteration,
            "cpu_time": cpu_time,
            "residual_history": residual_history,
            "time_history": time_history,
            "final_residual": final_residual
        })

        print(
            f"{'SOR':>12}"
            f"{omega:>10.1f}"
            f"{iteration:>15}"
            f"{cpu_time:>18.4f}"
            f"{final_residual:>20.2e}"
        )

    print("-" * 75)

    plot_gs_sor_comparison(results)

    return results