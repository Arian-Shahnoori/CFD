from matplotlib import pyplot as plt
from matplotlib import gridspec
import numpy as np


# =============================================================================
# Plot Style
# =============================================================================

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


# =============================================================================
# Contour Comparison
# =============================================================================

def plot_contours(X, Y, T_num, T_exact):

    vmin = min(T_num.min(), T_exact.min())
    vmax = max(T_num.max(), T_exact.max())

    levels = np.linspace(vmin, vmax, 50)

    fig = plt.figure(figsize=(12, 5))

    gs = gridspec.GridSpec(
        1, 3,
        width_ratios=[1, 1, 0.05]
    )

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    cax = fig.add_subplot(gs[2])

    cf = ax1.contourf(
        X, Y, T_num,
        levels=levels,
        vmin=vmin,
        vmax=vmax
    )

    ax2.contourf(
        X, Y, T_exact,
        levels=levels,
        vmin=vmin,
        vmax=vmax
    )

    ax1.set_title("Numerical")
    ax2.set_title("Exact")

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    ax2.set_xlabel("x")

    cbar = fig.colorbar(cf, cax=cax)
    cbar.set_label("Temperature", fontweight='bold')

    plt.tight_layout()


# =============================================================================
# Error Field
# =============================================================================

def plot_error(X, Y, error):

    levels = np.linspace(
        error.min(),
        error.max(),
        50
    )

    fig = plt.figure(figsize=(7, 5))

    gs = gridspec.GridSpec(
        1, 2,
        width_ratios=[1, 0.05]
    )

    ax = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])

    cf = ax.contourf(
        X, Y, error,
        levels=levels
    )

    ax.set_title("Error Field")

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    cbar = fig.colorbar(cf, cax=cax)
    cbar.set_label("Error", fontweight='bold')

    plt.tight_layout()


# =============================================================================
# Line Cuts
# =============================================================================

from matplotlib.lines import Line2D

def plot_line_cuts(x, T_num, T_exact):

    y_cuts = [0.0, 0.25, 0.5, 0.75, 1.0]
    x_cuts = [0.0, 0.25, 0.5]

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(14, 5),
        constrained_layout=True
    )

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # =====================================================
    # Horizontal cuts
    # =====================================================

    color_handles = []

    for n, y0 in enumerate(y_cuts):

        j = np.argmin(np.abs(x - y0))
        color = colors[n % len(colors)]

        ax1.plot(
            x,
            T_exact[:, j],
            color=color,
            linewidth=2
        )

        ax1.plot(
            x,
            T_num[:, j],
            'o',
            color=color,
            linewidth=1.5
        )

        color_handles.append(
            Line2D(
                [0], [0],
                color=color,
                lw=2,
                label=f'y = {x[j]:.2f}'
            )
        )

    ax1.set_title("Horizontal Line Cuts")
    ax1.set_xlabel("x")
    ax1.set_ylabel("Temperature")

    ax1.grid(True, which='major', linestyle='--', alpha=0.7)
    ax1.grid(True, which='minor', linestyle=':', alpha=0.3)

    legend1 = ax1.legend(
        handles=color_handles,
        title="Cut Location",
        loc='upper left'
    )

    ax1.add_artist(legend1)

    style_handles = [
        Line2D([0], [0], color='black', lw=2, linestyle='-', label='Exact'),
        Line2D([0], [0],
       color='black',
       marker='o',
       linestyle='None',
       markersize=6,
       label='Numerical')
    ]

    ax1.legend(
        handles=style_handles,
        loc='lower right'
    )

    # =====================================================
    # Vertical cuts
    # =====================================================

    color_handles = []

    for n, x0 in enumerate(x_cuts):

        i = np.argmin(np.abs(x - x0))
        color = colors[n % len(colors)]

        ax2.plot(
            x,
            T_exact[i, :],
            color=color,
            linewidth=2
        )

        ax2.plot(
            x,
            T_num[i, :],
            'o',
            color=color,
            linewidth=1.5
        )

        color_handles.append(
            Line2D(
                [0], [0],
                color=color,
                lw=2,
                label=f'x = {x[i]:.2f}'
            )
        )

    ax2.set_title("Vertical Line Cuts")
    ax2.set_xlabel("y")
    ax2.set_ylabel("Temperature")

    ax2.grid(True, which='major', linestyle='--', alpha=0.7)
    ax2.grid(True, which='minor', linestyle=':', alpha=0.3)

    legend1 = ax2.legend(
        handles=color_handles,
        title="Cut Location",
        loc='upper left'
    )

    ax2.add_artist(legend1)

    ax2.legend(
        handles=style_handles,
        loc='lower right'
    )


# =============================================================================
# Residual plot
# =============================================================================

def plot_residuals(residual_history, time_history):

    fig = plt.figure(figsize=(12, 5))

    fig.suptitle(
        "Gauss-Seidel Convergence History",
        fontweight='bold'
    )

    gs = gridspec.GridSpec(
        1, 2,
        width_ratios=[1, 1]
    )

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # ============================================================
    # Residual vs Iteration
    # ============================================================

    ax1.semilogy(
        residual_history,
        linewidth=2
    )

    ax1.set_title("Residual vs Iteration")

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Residual Norm")

    ax1.minorticks_on()

    ax1.grid(
        True,
        which='major',
        linestyle='--',
        alpha=0.7
    )

    ax1.grid(
        True,
        which='minor',
        linestyle=':',
        alpha=0.3
    )

    # ============================================================
    # Residual vs CPU Time
    # ============================================================

    ax2.semilogy(
        time_history,
        residual_history,
        linewidth=2
    )

    ax2.set_title("Residual vs CPU Time")

    ax2.set_xlabel("CPU Time [s]")
    ax2.set_ylabel("Residual Norm")

    ax2.minorticks_on()

    ax2.grid(
        True,
        which='major',
        linestyle='--',
        alpha=0.7
    )

    ax2.grid(
        True,
        which='minor',
        linestyle=':',
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


# =============================================================================
# Mesh-Size Convergence
# =============================================================================

def plot_mesh_size_convergence(results):

    fig = plt.figure(figsize=(12, 5))

    fig.suptitle(
        "Point Gauss-Seidel Convergence for Different Mesh Sizes",
        fontweight='bold'
    )

    gs = gridspec.GridSpec(
        1, 2,
        width_ratios=[1, 1]
    )

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # ============================================================
    # Residual vs Iteration
    # ============================================================

    for result in results:

        N = result["N"]
        residual_history = result["residual_history"]

        ax1.semilogy(
            residual_history,
            linewidth=2,
            label=f"N = {N}"
        )

    ax1.set_title("Residual vs Iteration")

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Residual Norm")

    ax1.minorticks_on()

    ax1.grid(
        True,
        which='major',
        linestyle='--',
        alpha=0.7
    )

    ax1.grid(
        True,
        which='minor',
        linestyle=':',
        alpha=0.3
    )

    ax1.legend()

    # ============================================================
    # Residual vs CPU Time
    # ============================================================

    for result in results:

        N = result["N"]
        residual_history = result["residual_history"]
        time_history = result["time_history"]

        ax2.semilogy(
            time_history,
            residual_history,
            linewidth=2,
            label=f"N = {N}"
        )

    ax2.set_title("Residual vs CPU Time")

    ax2.set_xlabel("CPU Time [s]")
    ax2.set_ylabel("Residual Norm")

    ax2.minorticks_on()

    ax2.grid(
        True,
        which='major',
        linestyle='--',
        alpha=0.7
    )

    ax2.grid(
        True,
        which='minor',
        linestyle=':',
        alpha=0.3
    )

    ax2.legend()

    plt.tight_layout()

    plt.show()


# =============================================================================
# GS and SOR Comparison
# =============================================================================
def plot_gs_sor_comparison(results):

    fig = plt.figure(figsize=(12, 5))

    fig.suptitle(
        "Solver Convergence Comparison",
        fontweight='bold'
    )

    gs = gridspec.GridSpec(
        1, 2,
        width_ratios=[1, 1]
    )

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # ============================================================
    # Residual vs Iteration
    # ============================================================

    for result in results:

        solver = result["solver"]
        omega = result["omega"]
        residual_history = result["residual_history"]

        if solver == "SOR":
            label = f"SOR, ω = {omega:.1f}"
        else:
            label = "Gauss-Seidel"

        ax1.semilogy(
            residual_history,
            linewidth=2,
            label=label
        )

    ax1.set_title("Residual vs Iteration")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Residual Norm")

    ax1.minorticks_on()

    ax1.grid(
        True,
        which='major',
        linestyle='--',
        alpha=0.7
    )

    ax1.grid(
        True,
        which='minor',
        linestyle=':',
        alpha=0.3
    )

    ax1.legend()

    # ============================================================
    # Residual vs CPU Time
    # ============================================================

    for result in results:

        solver = result["solver"]
        omega = result["omega"]
        residual_history = result["residual_history"]
        time_history = result["time_history"]

        if solver == "SOR":
            label = f"SOR, ω = {omega:.1f}"
        else:
            label = "Gauss-Seidel"

        ax2.semilogy(
            time_history,
            residual_history,
            linewidth=2,
            label=label
        )

    ax2.set_title("Residual vs CPU Time")
    ax2.set_xlabel("CPU Time [s]")
    ax2.set_ylabel("Residual Norm")

    ax2.minorticks_on()

    ax2.grid(
        True,
        which='major',
        linestyle='--',
        alpha=0.7
    )

    ax2.grid(
        True,
        which='minor',
        linestyle=':',
        alpha=0.3
    )

    ax2.legend()

    plt.tight_layout()
    plt.show()


# =============================================================================
# Point SOR vs Line SOR Convergence
# =============================================================================

def plot_sor_line_sor_comparison(results):

    fig = plt.figure(figsize=(12, 5))

    fig.suptitle(
        "Point SOR vs Line SOR Convergence",
        fontweight='bold'
    )

    gs = gridspec.GridSpec(
        1, 2,
        width_ratios=[1, 1]
    )

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # ============================================================
    # Residual vs Iteration
    # ============================================================

    for result in results:

        ax1.semilogy(
            result["residual_history"],
            linewidth=2,
            label=result["solver"]
        )

    ax1.set_title("Residual vs Iteration")

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Residual Norm")

    ax1.minorticks_on()

    ax1.grid(
        True,
        which='major',
        linestyle='--',
        alpha=0.7
    )

    ax1.grid(
        True,
        which='minor',
        linestyle=':',
        alpha=0.3
    )

    ax1.legend()

    # ============================================================
    # Residual vs CPU Time
    # ============================================================

    for result in results:

        ax2.semilogy(
            result["time_history"],
            result["residual_history"],
            linewidth=2,
            label=result["solver"]
        )

    ax2.set_title("Residual vs CPU Time")

    ax2.set_xlabel("CPU Time [s]")
    ax2.set_ylabel("Residual Norm")

    ax2.minorticks_on()

    ax2.grid(
        True,
        which='major',
        linestyle='--',
        alpha=0.7
    )

    ax2.grid(
        True,
        which='minor',
        linestyle=':',
        alpha=0.3
    )

    ax2.legend()

    plt.tight_layout()
    plt.show()


# =============================================================================
# Solver CPU-Time Comparison
# =============================================================================

def plot_solver_cpu_times(results):

    solver_data = {}

    for result in results:

        if result["omega"] is None:
            label = result["solver"]
        else:
            label = (
                f'{result["solver"]}, '
                f'\u03c9 = {result["omega"]:.1f}'
            )

        if label not in solver_data:
            solver_data[label] = []

        solver_data[label].append(
            (result["N"], result["cpu_time"])
        )

    fig, ax = plt.subplots(figsize=(8, 5))

    for label, data in solver_data.items():

        data.sort(key=lambda point: point[0])

        mesh_sizes = [point[0] for point in data]
        cpu_times = [point[1] for point in data]

        ax.loglog(
            mesh_sizes,
            cpu_times,
            marker="o",
            label=label
        )

    ax.set_xlabel("Number of control volumes per direction, N")
    ax.set_ylabel("CPU Time [s]")
    ax.set_title("Solver CPU-Time Comparison")

    ax.grid(True, which="both")
    ax.legend()

    plt.tight_layout()
    plt.show()