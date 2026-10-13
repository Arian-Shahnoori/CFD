from pathlib import Path

from src.stability_analysis import rk2_upwind_stability_map
from src.visualization import plot_rk2_upwind_stability


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_study_02_rk2_second_order_upwind_stability():
    """Run and visualize the RK2 stability analysis."""

    results = rk2_upwind_stability_map(
        cfl_max=1.5,
        n_cfl=451,
        n_phi=601,
    )

    print("\nRK2 + Second-Order Upwind Stability")
    print("-" * 45)
    print(
        f"Theoretical CFL limit: "
        f"{results['theoretical_cfl_limit']:.4f}"
    )
    print(
        f"Sampled all-mode CFL limit: "
        f"{results['sampled_cfl_limit']:.4f}"
    )

    figure_path = (
        PROJECT_ROOT
        / "results"
        / "figures"
        / "rk2_second_order_upwind_stability.png"
    )

    plot_rk2_upwind_stability(
        results,
        save_path=figure_path,
    )

    return results


if __name__ == "__main__":
    run_study_02_rk2_second_order_upwind_stability()