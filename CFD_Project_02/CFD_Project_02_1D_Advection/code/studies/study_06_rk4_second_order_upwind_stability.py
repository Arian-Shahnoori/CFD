from pathlib import Path

from src.stability_analysis import rk4_upwind_stability_map
from src.visualization import plot_rk4_upwind_stability


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_study_06_rk4_second_order_upwind_stability():
    """
    Analyze the Fourier stability of classical RK4
    combined with second-order upwind spatial discretization.
    """

    # Stability-analysis settings
    CFL_MAX = 2
    N_CFL = 601
    N_PHI = 601

    # Calculate the amplification factor over the (CFL, phi) grid.
    results = rk4_upwind_stability_map(
        cfl_max=CFL_MAX,
        n_cfl=N_CFL,
        n_phi=N_PHI,
    )

    # Report the stability limit.
    print("\nStudy 6: RK4 + Second-Order Upwind Stability")
    print("=" * 58)
    print(f"Fourier phase points:       {N_PHI}")
    print(f"Courant-number points:      {N_CFL}")
    print(f"Maximum CFL plotted:        {CFL_MAX:.3f}")
    print(
        f"Sampled all-mode CFL limit: "
        f"{results['sampled_cfl_limit']:.4f}"
    )
    print(
        f"Theoretical CFL limit:     "
        f"{results['theoretical_cfl_limit']:.4f}"
    )
    print("=" * 58)

    # Generate and save the stability diagram.
    figure_path = (
        PROJECT_ROOT
        / "results"
        / "figures"
        / "rk4_second_order_upwind_stability.png"
    )

    plot_rk4_upwind_stability(
        results,
        save_path=figure_path,
    )

    print(f"Figure saved to: {figure_path}")

    return results


if __name__ == "__main__":
    run_study_06_rk4_second_order_upwind_stability()