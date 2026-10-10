from src.initialization import initialize_temperature

from src.solvers.sor import sor
from src.solvers.line_sor import line_sor

from src.visualization import plot_sor_line_sor_comparison


# =============================================================================
# Point SOR vs Line SOR Comparison
# =============================================================================

def run_line_sor_comparison():

    tolerance = 1e-12
    N = 40
    omega = 1.1

    results = []

    print("\nPoint SOR vs Line SOR Comparison")

    print("-" * 75)

    print(
        f"{'Solver':>15}"
        f"{'Omega':>10}"
        f"{'Iterations':>15}"
        f"{'CPU Time [s]':>18}"
        f"{'Final Residual':>20}"
    )

    print("-" * 75)

    # ============================================================
    # Point SOR
    # ============================================================

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
        "solver": "Point SOR",
        "omega": omega,
        "iterations": iteration,
        "cpu_time": cpu_time,
        "residual_history": residual_history,
        "time_history": time_history,
        "final_residual": final_residual
    })

    print(
        f"{'Point SOR':>15}"
        f"{omega:>10.1f}"
        f"{iteration:>15}"
        f"{cpu_time:>18.4f}"
        f"{final_residual:>20.2e}"
    )

    # ============================================================
    # Line SOR
    # ============================================================

    T = initialize_temperature(N)

    T, residual_history, time_history, iteration = line_sor(
        T,
        N,
        tolerance,
        omega
    )

    cpu_time = time_history[-1]
    final_residual = residual_history[-1]

    results.append({
        "solver": "Line SOR",
        "omega": omega,
        "iterations": iteration,
        "cpu_time": cpu_time,
        "residual_history": residual_history,
        "time_history": time_history,
        "final_residual": final_residual
    })

    print(
        f"{'Line SOR':>15}"
        f"{omega:>10.1f}"
        f"{iteration:>15}"
        f"{cpu_time:>18.4f}"
        f"{final_residual:>20.2e}"
    )

    print("-" * 75)

    plot_sor_line_sor_comparison(results)

    return results