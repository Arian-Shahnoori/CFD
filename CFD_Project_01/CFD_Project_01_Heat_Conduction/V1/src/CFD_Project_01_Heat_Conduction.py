import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import platform
import psutil
import cpuinfo
import json
import time

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
# System Setup
# =============================================================================


def get_system_info():
    cpu = cpuinfo.get_cpu_info()

    info = {
        "processor": cpu.get("brand_raw", "Unknown"),
        "architecture": platform.machine(),
        "system": platform.system(),
        "os_version": platform.version(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2)
    }

    return info


def save_system_info(filename="system_info.json"):
    info = get_system_info()

    with open(filename, "w") as f:
        json.dump(info, f, indent=4)

    print("System info saved to", filename)
    return info


# =============================================================================
# Initial Condition
# =============================================================================

def initialize_temperature(N):

    T = np.zeros((N + 2, N + 2))

    x = np.linspace(1/(2*N), 1 - 1/(2*N), N)

    T[1:N+1, N+1] = 2 * np.sin(np.pi * x)

    return T


# =============================================================================
# Residual Norm
# =============================================================================

def residual_norm(T, N):

    dx = 1.0 / N

    residual = (
        T[2:N+2, 1:N+1]
        + T[0:N, 1:N+1]
        + T[1:N+1, 2:N+2]
        + T[1:N+1, 0:N]
        - 4 * T[1:N+1, 1:N+1]
    ) / dx**2

    return np.sqrt(np.mean(residual**2))


# =============================================================================
# Boundary Conditions
# =============================================================================

def apply_boundary_conditions(T, N):

    x = np.linspace(1/(2*N), 1 - 1/(2*N), N)

    # Left / Right
    T[0, 1:N+1] = -T[1, 1:N+1]
    T[N+1, 1:N+1] = -T[N, 1:N+1]

    # Bottom
    T[1:N+1, 0] = -T[1:N+1, 1]

    # Top
    T[1:N+1, N+1] = (
        2 * np.sin(np.pi * x)
        - T[1:N+1, N]
    )


# =============================================================================
# Gauss-Seidel Solver
# =============================================================================

def gauss_seidel(T, N, tolerance):

    residual_history = []
    time_history = []

    residual = 1.0
    iteration = 0

    start_time = time.perf_counter()

    while residual > tolerance:

        iteration += 1

        for j in range(1, N+1):
            for i in range(1, N+1):

                T[i, j] = 0.25 * (
                    T[i+1, j]
                    + T[i-1, j]
                    + T[i, j+1]
                    + T[i, j-1]
                )

        apply_boundary_conditions(T, N)

        residual = residual_norm(T, N)

        residual_history.append(residual)

        elapsed_time = time.perf_counter() - start_time
        time_history.append(elapsed_time)

    return T, residual_history, time_history, iteration


# =============================================================================
# Exact Solution
# =============================================================================

def exact_solution(X, Y):

    return (
        np.sin(np.pi * X)
        * np.sinh(np.pi * Y)
        / np.sinh(np.pi)
    )


# =============================================================================
# Grid Generation
# =============================================================================

def create_grid(N):

    x = np.linspace(1/(2*N), 1 - 1/(2*N), N)

    X, Y = np.meshgrid(x, x, indexing='ij')

    return x, X, Y


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
# Main
# =============================================================================
save_system_info()

N = 30
tolerance = 1e-10

T = initialize_temperature(N)

T, residual_history, time_history, iteration = gauss_seidel(
    T,
    N,
    tolerance
)

T_num = T[1:N+1, 1:N+1]

x, X, Y = create_grid(N)

T_exact = exact_solution(X, Y)

error = T_exact - T_num

plot_contours(X, Y, T_num, T_exact)

plot_error(X, Y, error)

plot_line_cuts(x, T_num, T_exact)

plot_residuals(residual_history, time_history)