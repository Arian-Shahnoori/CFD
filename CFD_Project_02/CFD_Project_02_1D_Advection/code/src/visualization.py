from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
    "image.cmap": "plasma"
})

def _finish_figure(fig, save_path=None):
    """Optionally save a figure, then display it."""
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_convergence(
    cell_counts,
    errors,
    order,
    fitted_errors,
    title="Grid Refinement",
    x_label="Number of control volumes",
    y_label="RMS L2 error",
    save_path=None,
):
    """Plot error versus grid resolution on logarithmic axes."""

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.loglog(
        cell_counts,
        errors,
        "o-",
        label="Numerical error",
    )
    ax.loglog(
        cell_counts,
        fitted_errors,
        "--",
        label=f"Log-log fit (order = {order:.2f})",
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":")
    ax.legend()
    fig.tight_layout()

    _finish_figure(fig, save_path)


def plot_solution_comparison(
    x,
    numerical_solutions: dict,
    exact_solution,
    title="Numerical Solution Comparison",
    x_label="x",
    y_label="T",
    save_path=None,
    y_limits=None,  
):
    """
    Plot multiple numerical solutions against the analytical solution.

    numerical_solutions maps each legend label to a solution array.
    For example: {"CFL = 0.25": T1, "CFL = 0.4": T2}.
    """

    fig, ax = plt.subplots(figsize=(8, 5))

    for label, solution in numerical_solutions.items():
        ax.plot(x, solution, marker="o", markersize=3, label=label)

    ax.plot(x, exact_solution, "k--", label="Analytical solution")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, linestyle=":")
    ax.legend()
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    fig.tight_layout()

    _finish_figure(fig, save_path)



def plot_rk2_upwind_stability(
    results,
    save_path=None,
):
    """Plot stable and unstable Fourier modes for RK2."""

    Phi = results["Phi"]
    C = results["C"]
    abs_G = results["abs_G"]

    max_level = max(float(np.max(abs_G)) + 1e-6, 1.000001)

    fig, ax = plt.subplots(figsize=(9, 6.5))

    # Green: stable modes. Red: unstable modes.
    ax.contourf(
        Phi,
        C,
        abs_G,
        levels=[0.0, 1.0, max_level],
        colors=["#dcefd8", "#f8dada"],
        alpha=0.9,
    )

    # Boundary of the absolute-stability region.
    boundary = ax.contour(
        Phi,
        C,
        abs_G,
        levels=[1.0],
        colors="black",
        linewidths=1.2,
    )

    ax.clabel(
        boundary,
        inline=True,
        fmt={1.0: r"$|G|=1$"},
        fontsize=10,
    )

    # Theoretical limiting Courant number.
    ax.axhline(
        results["theoretical_cfl_limit"],
        color="navy",
        linestyle="--",
        linewidth=1.6,
    )

    ax.set_title(
        "RK2 Stability: Second-Order Upwind"
    )
    ax.set_xlabel(r"Fourier phase $\phi$")
    ax.set_ylabel(r"Courant number $C=u\Delta t/\Delta x$")

    ax.set_xlim(0.0, 2.0 * np.pi)
    ax.set_ylim(0.0, float(np.max(C)))

    ax.set_xticks(
        [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
        [r"$0$", r"$\pi/2$", r"$\pi$",
         r"$3\pi/2$", r"$2\pi$"],
    )

    ax.grid(True, alpha=0.3)

    legend_handles = [
        Patch(
            facecolor="#dcefd8",
            label=r"Stable mode: $|G|\leq1$",
        ),
        Patch(
            facecolor="#f8dada",
            label=r"Unstable mode: $|G|>1$",
        ),
        Line2D(
            [0],
            [0],
            color="navy",
            linestyle="--",
            linewidth=1.6,
            label=r"Limiting CFL: $C=0.5$",
        ),
    ]

    ax.legend(handles=legend_handles, loc="upper left")
    fig.tight_layout()

    if save_path is not None:
        from pathlib import Path

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()



def plot_error_comparison(
    cell_counts,
    errors_by_scheme,
    fitted_errors_by_scheme,
    orders_by_scheme,
    title="Spatial Grid Convergence",
    save_path=None,
):
    """Compare convergence errors for multiple time integrators."""

    import matplotlib.pyplot as plt
    from pathlib import Path

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for scheme, errors in errors_by_scheme.items():
        # Actual numerical errors.
        line, = ax.loglog(
            cell_counts,
            errors,
            marker="o",
            linestyle="-",
            label=f"{scheme} error",
        )

        # Fitted convergence curve, using the same color.
        ax.loglog(
            cell_counts,
            fitted_errors_by_scheme[scheme],
            linestyle="--",
            color=line.get_color(),
            label=(
                f"{scheme} fit "
                f"(order = {orders_by_scheme[scheme]:.2f})"
            ),
        )

    ax.set_xlabel("Number of control volumes")
    ax.set_ylabel("RMS L2 error")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":")
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()



def plot_rk4_upwind_stability(results, save_path=None):
    """Plot the RK4 stability region for second-order upwind."""

    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    Phi = results["Phi"]
    C = results["C"]
    abs_G = results["abs_G"]

    max_level = max(float(np.max(abs_G)) + 1e-6, 1.000001)

    fig, ax = plt.subplots(figsize=(9, 6.5))

    # Classify individual Fourier modes.
    ax.contourf(
        Phi,
        C,
        abs_G,
        levels=[0.0, 1.0, max_level],
        colors=["#dcefd8", "#f8dada"],
        alpha=0.9,
    )

    # Amplification-factor contours.
    contours = ax.contour(
        Phi,
        C,
        abs_G,
        levels=[ 1.0],
        colors=[ "black"],
        linewidths=[ 1.4],
    )

    ax.clabel(
        contours,
        inline=True,
        fmt={1.0: r"$|G|=1$"},
        fontsize=10,
    )

    # Theoretical stability limit.
    ax.axhline(
        results["theoretical_cfl_limit"],
        color="navy",
        linestyle="--",
        linewidth=1.6,
    )

    ax.set_title("RK4 Stability: Second-Order Upwind")
    ax.set_xlabel(r"Fourier phase $\phi$")
    ax.set_ylabel(r"Courant number $C=u\Delta t/\Delta x$")

    ax.set_xlim(0.0, 2.0 * np.pi)
    ax.set_ylim(0.0, float(np.max(C)))

    ax.set_xticks(
        [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
        [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"],
    )

    ax.grid(True, alpha=0.3)

    legend_handles = [
        Patch(
            facecolor="#dcefd8",
            label=r"Stable mode: $|G|\leq1$",
        ),
        Patch(
            facecolor="#f8dada",
            label=r"Unstable mode: $|G|>1$",
        ),
        Line2D(
            [0],
            [0],
            color="navy",
            linestyle="--",
            linewidth=1.6,
            label=r"Theoretical CFL limit $\approx 0.6963$",
        ),
    ]

    ax.legend(handles=legend_handles, loc="upper left")
    fig.tight_layout()


    plt.show()