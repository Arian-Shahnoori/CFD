from src.initialization import initialize_temperature
from src.solvers.gauss_seidel import gauss_seidel
from src.visualization import plot_mesh_size_convergence

def run_mesh_size_study():

    tolerance = 1e-12
    mesh_dimensions = [10, 20, 30, 40]

    results = []

    print("\nMesh-Size Study")
    print("-" * 65)
    print(
        f"{'N':>8}"
        f"{'Iterations':>15}"
        f"{'CPU Time [s]':>18}"
        f"{'Final Residual':>20}"
    )
    print("-" * 65)

    for N in mesh_dimensions:

        T = initialize_temperature(N)

        T, residual_history, time_history, iteration = gauss_seidel(
            T,
            N,
            tolerance
        )

        cpu_time = time_history[-1]
        final_residual = residual_history[-1]

        results.append({
            "N": N,
            "iterations": iteration,
            "cpu_time": cpu_time,
            "residual_history": residual_history,
            "time_history": time_history,
            "final_residual": final_residual
        })

        print(
            f"{N:>8}"
            f"{iteration:>15}"
            f"{cpu_time:>18.4f}"
            f"{final_residual:>20.2e}"
        )

    print("-" * 65)

    plot_mesh_size_convergence(results)

    return