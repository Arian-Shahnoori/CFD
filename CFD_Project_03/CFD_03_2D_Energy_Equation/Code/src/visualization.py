import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

from .grid import create_grid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIRECTORY = PROJECT_ROOT / "results"

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

def plot_temperature(
    temperature,
    mesh,
    domain_length_x,
    domain_length_y,
    reynolds_number,
    prandtl_number,
    eckert_number,
    peclet_number,
    brinkman_number,
    output_directory=RESULTS_DIRECTORY,
):
    """Plot temperature contours with flow parameters in the title."""

    os.makedirs(output_directory, exist_ok=True)

    x, y, _, _ = create_grid(
        mesh,
        domain_length_x,
        domain_length_y,
    )

    temperature_interior = temperature[1:-1, 1:-1].T

    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(10, 3.5))

    contour = ax.contourf(
        X,
        Y,
        temperature_interior,
        levels=40,
        cmap="inferno",
    )

    fig.colorbar(contour, ax=ax, label="Temperature, T")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_title(
        f"Thermal Development in a Channel\n"
        f"Re = {reynolds_number:g} | "
        f"Pr = {prandtl_number:g} | "
        f"Ec = {eckert_number:g} | "
        f"Pe = {peclet_number:g} | "
        f"Br = {brinkman_number:g}"
    )

    ax.set_aspect("auto")

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_directory, "temperature_contours.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

def plot_convergence(
    iteration_history,
    residual_history,
    output_directory="results",
):
    """Plot the steady-equation residual during time marching."""

    os.makedirs(output_directory, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.semilogy(
        iteration_history,
        residual_history,
        label="Steady-equation residual",
    )

    ax.set_xlabel("Time step")
    ax.set_ylabel("RMS residual")
    ax.set_title("Steady-State Convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_directory, "residual_history.png"),
        dpi=300,
    )
    plt.show()


def plot_self_convergence(
    grid_spacings,
    difference_norms,
    output_path="results/grid_self_convergence.png",
):
    """Plot solution-difference norms and estimate the observed order."""

    from pathlib import Path

    grid_spacings = np.asarray(grid_spacings, dtype=float)
    difference_norms = np.asarray(difference_norms, dtype=float)

    if (
        grid_spacings.ndim != 1
        or difference_norms.ndim != 1
        or len(grid_spacings) != len(difference_norms)
        or len(grid_spacings) < 2
    ):
        raise ValueError(
            "Provide at least two matching grid-spacing and "
            "difference-norm values."
        )

    if (
        not np.all(np.isfinite(grid_spacings))
        or not np.all(np.isfinite(difference_norms))
        or np.any(grid_spacings <= 0.0)
        or np.any(difference_norms <= 0.0)
    ):
        raise ValueError(
            "Grid spacings and difference norms must be finite "
            "and strictly positive."
        )

    # Fit log(D) = p*log(h) + intercept.
    # The slope estimates the observed order of self-convergence.
    slope, intercept = np.polyfit(
        np.log(grid_spacings),
        np.log(difference_norms),
        1,
    )

    observed_order = float(slope)

    # Sort points by grid spacing for plotting.
    order = np.argsort(grid_spacings)
    h_data = grid_spacings[order]
    d_data = difference_norms[order]

    h_fit = np.geomspace(h_data.min(), h_data.max(), 200)
    d_fit = np.exp(intercept) * h_fit**slope

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.loglog(
        h_data,
        d_data,
        "o-",
        markersize=7,
        label="Successive-grid difference norm",
    )

    ax.loglog(
        h_fit,
        d_fit,
        "--",
        label=(
            f"Power-law fit: p = {observed_order:.3f}"
        ),
    )

    ax.set_xlabel(
        r"Characteristic grid spacing, $h=\sqrt{\Delta x\,\Delta y}$"
    )
    ax.set_ylabel(
        r"RMS difference norm, $D_h$"
    )
    ax.set_title("Grid Self-Convergence Study")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()

    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return float(slope), observed_order


def plot_outlet_error(
    iteration_history,
    outlet_error_history,
    output_directory="results",
):
    """Plot the outlet-profile difference from the analytical reference."""

    os.makedirs(output_directory, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.semilogy(
        iteration_history,
        np.maximum(outlet_error_history, np.finfo(float).tiny),
    )

    ax.set_xlabel("Time step")
    ax.set_ylabel("Outlet-profile RMS error")
    ax.set_title("Outlet Profile Error")
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_directory, "outlet_error_history.png"),
        dpi=300,
    )
    plt.show()



def animate_temperature_history(
    temperature_history,
    snapshot_iterations,
    time_step,
    domain_length_x,
    domain_length_y,
    reynolds_number,
    prandtl_number,
    eckert_number,
    peclet_number,
    brinkman_number,
    gif_path,
    fps=20,
):
    """Create a temperature-evolution GIF from saved snapshots."""

    temperature_history = np.asarray(temperature_history)
    snapshot_iterations = np.asarray(snapshot_iterations)

    if len(temperature_history) == 0:
        raise ValueError("No temperature snapshots were provided.")

    if len(temperature_history) != len(snapshot_iterations):
        raise ValueError(
            "Temperature snapshots and iteration numbers must have "
            "the same length."
        )

    # Remove ghost cells, transpose for plotting, and flip the x-direction.
    frames = temperature_history[:, 1:-1, 1:-1].transpose(0, 2, 1)
    #frames = frames[:, :, ::-1]

    # Fixed color scale across all frames.
    vmin = float(np.min(frames))
    vmax = float(np.max(frames))

    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-12

    fig, ax = plt.subplots(figsize=(10, 5))

    # Leave space above the plot for the title and flow parameters.
    fig.subplots_adjust(top=0.78)

    image = ax.imshow(
        frames[0],
        origin="lower",
        extent=[0, domain_length_x, 0, domain_length_y],
        aspect="auto",
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    fig.colorbar(image, ax=ax, label="Temperature")

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Fixed title containing the flow parameters.
    fig.suptitle(
        f"Thermal Development in a Channel\n"
        f"Re = {reynolds_number:g} | "
        f"Pr = {prandtl_number:g} | "
        f"Ec = {eckert_number:g} | "
        f"Pe = {peclet_number:g} | "
        f"Br = {brinkman_number:g}",
        fontsize=11,
    )

    # Dynamic title showing the iteration and physical time.
    title = ax.set_title("")

    def update(frame_index):
        image.set_data(frames[frame_index])

        iteration = snapshot_iterations[frame_index]
        physical_time = iteration * time_step

        title.set_text(
            f"Iteration: {iteration} | t = {physical_time:.4f}"
        )

        return image, title

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=100,
        blit=False,
    )

    gif_path = Path(gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)

    animation.save(
        gif_path,
        writer=PillowWriter(fps=fps),
    )

    plt.close(fig)

    print(f"GIF saved to: {gif_path}")

    return gif_path



def plot_grid_convergence(
    cell_counts,
    error_norms,
    output_path="results/grid_convergence.png",
):
    """Plot L2 error against the number of control volumes."""

    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    cell_counts = np.asarray(cell_counts, dtype=float)
    error_norms = np.asarray(error_norms, dtype=float)

    if (
        cell_counts.ndim != 1
        or error_norms.ndim != 1
        or len(cell_counts) != len(error_norms)
        or len(cell_counts) < 2
    ):
        raise ValueError(
            "Provide at least two matching cell-count/error pairs."
        )

    if (
        not np.all(np.isfinite(cell_counts))
        or not np.all(np.isfinite(error_norms))
        or np.any(cell_counts <= 0)
        or np.any(error_norms <= 0)
    ):
        raise ValueError("Cell counts and errors must be finite and positive.")

    # Fit log(error) = slope * log(number of control volumes) + intercept.
    slope, intercept = np.polyfit(
        np.log(cell_counts),
        np.log(error_norms),
        1,
    )

    # Estimate the spatial convergence order.
    observed_order = -2.0 * slope

    # Sort points for plotting the fitted line.
    order = np.argsort(cell_counts)
    x_data = cell_counts[order]
    y_data = error_norms[order]

    x_fit = np.geomspace(x_data.min(), x_data.max(), 200)
    y_fit = np.exp(intercept) * x_fit**slope

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.loglog(
        x_data,
        y_data,
        "o",
        markersize=7,
        label="Numerical error",
    )

    ax.loglog(
        x_fit,
        y_fit,
        "--",
        label=(
            f"Linear fit: m = {slope:.3f}\n"
            f"Observed order: p = {observed_order:.3f}"
        ),
    )

    ax.set_xlabel(r"Number of control volumes, $N_xN_y$")
    ax.set_ylabel(r"RMS $L_2$ error")
    ax.set_title("Grid-Convergence Study")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()

    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    return float(slope), float(observed_order)



# parameter study
def plot_parameter_group(
    results,
    cases,
    case_ids,
    group_title,
    output_path,
    mesh,
    domain_length_x,
    domain_length_y,
    temperature_limits,
):
    """
    Plot a group of parameter-study cases as vertically stacked
    temperature contours with a common color scale.

    For example, a dissipation study contains three contour plots
    in one figure, one per row.
    """
    from pathlib import Path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not case_ids:
        raise ValueError("case_ids cannot be empty.")

    x, y, _, _ = create_grid(
        mesh,
        domain_length_x,
        domain_length_y,
    )

    X, Y = np.meshgrid(x, y)

    vmin, vmax = temperature_limits
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-12

    levels = np.linspace(vmin, vmax, 41)

    fig, axes = plt.subplots(
        nrows=len(case_ids),
        ncols=1,
        figsize=(10, 3.5 * len(case_ids)),
        squeeze=False,
    )
    axes = axes[:, 0]

    contour_sets = []

    for ax, case_id in zip(axes, case_ids):
        case = cases[case_id]

        # Temperature shape: (nx + 2, ny + 2), including ghost cells.
        temperature = np.asarray(
            results[case_id]["temperature"]
        )

        # Remove ghost cells; transpose for contour plotting.
        T = temperature[1:-1, 1:-1].T

        contour = ax.contourf(
            X,
            Y,
            T,
            levels=levels,
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
            extend="both",
        )

        contour_sets.append(contour)

        ax.set_xlim(0, domain_length_x)
        ax.set_ylim(0, domain_length_y)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("auto")

        ax.set_title(
            f"{case['label']} | "
            f"Pe = {case['Pe']:g}, "
            f"Br = {case['Br']:g}, "
            f"convection coefficient = {case['gamma_c']:g}"
        )

    # One common colorbar for the full figure.
    fig.colorbar(
        contour_sets[-1],
        ax=axes.tolist(),
        label="Temperature, T",
        pad=0.02,
        fraction=0.035,
    )

    fig.suptitle(group_title, fontsize=16, fontweight="bold")
    fig.subplots_adjust(
        top=0.94,
        bottom=0.06,
        hspace=0.5,
        right=0.82,
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved parameter-group figure: {output_path}")

    return output_path


def plot_case_cross_sections(
    temperature,
    mesh,
    domain_length_x,
    domain_length_y,
    case,
    output_path,
    temperature_limits,
    section_fractions=(0.1, 0.9),
    mean_velocity=3.0,
):
    """
    Generate one composite figure for a selected parameter case.

    Layout
    ------
    Top row:
        Temperature contour, spanning both columns.

    Bottom row:
        Left: temperature profile at x/Lx = section_fractions[0].
        Right: temperature profile at x/Lx = section_fractions[1].

    Each profile plots:
        - Computed temperature T against y.
        - Baseline T = y against y.
        - Velocity gradient du/dy against y, using a secondary
          horizontal axis.

    For u(y) = 6 * mean_velocity * y * (1-y):
        du/dy = 6 * mean_velocity * (1-2y).
    """
    from pathlib import Path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(section_fractions) != 2:
        raise ValueError(
            "Exactly two section fractions are required."
        )

    if any(
        not 0.0 <= fraction <= 1.0
        for fraction in section_fractions
    ):
        raise ValueError(
            "Section fractions must be between 0 and 1."
        )

    nx, ny = mesh

    x, y, _, _ = create_grid(
        mesh,
        domain_length_x,
        domain_length_y,
    )

    temperature = np.asarray(temperature)

    # Remove ghost cells: T has shape (nx, ny).
    T = temperature[1:-1, 1:-1]

    if T.shape != (nx, ny):
        raise ValueError(
            f"Expected temperature shape {(nx, ny)}, "
            f"got {T.shape} after removing ghost cells."
        )

    vmin, vmax = temperature_limits
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-12

    levels = np.linspace(vmin, vmax, 41)
    X, Y = np.meshgrid(x, y)

    # The prescribed velocity profile and its gradient.
    u_y = 6.0 * mean_velocity * (1.0 - 2.0 * y)

    fig = plt.figure(figsize=(13, 9))

    grid_spec = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=(1.25, 1.0),
        hspace=0.48,
        wspace=0.30,
    )

    # ---------------------------------------------------------
    # Top row: contour spanning both columns.
    # ---------------------------------------------------------
    contour_ax = fig.add_subplot(grid_spec[0, :])

    contour = contour_ax.contourf(
        X,
        Y,
        T.T,
        levels=levels,
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
        extend="both",
    )

    fig.colorbar(
        contour,
        ax=contour_ax,
        label="Temperature, T",
        pad=0.015,
    )

    contour_ax.set_xlabel("x")
    contour_ax.set_ylabel("y")
    contour_ax.set_xlim(0, domain_length_x)
    contour_ax.set_ylim(0, domain_length_y)
    contour_ax.set_aspect("auto")

    contour_ax.set_title(
        f"{case['label']}\n"
        f"Pe = {case['Pe']:g} | "
        f"Br = {case['Br']:g} | "
        f"convection coefficient = {case['gamma_c']:g}"
    )

    # ---------------------------------------------------------
    # Bottom row: one independent profile at each x/Lx.
    # ---------------------------------------------------------
    profile_axes = [
        fig.add_subplot(grid_spec[1, 0]),
        fig.add_subplot(grid_spec[1, 1]),
    ]

    for profile_ax, fraction in zip(
        profile_axes,
        section_fractions,
    ):
        target_x = fraction * domain_length_x

        # Choose the nearest x cell centre.
        ix = int(np.argmin(np.abs(x - target_x)))
        actual_x = x[ix]

        # Mark the section on the contour.
        contour_ax.axvline(
            actual_x,
            color="white",
            linestyle="--",
            linewidth=1.5,
        )

        # Primary horizontal axis: temperature.
        line_temperature, = profile_ax.plot(
            T[ix, :],
            y,
            linewidth=2.2,
            label="Computed temperature",
        )

        line_baseline, = profile_ax.plot(
            y,
            y,
            linestyle="--",
            linewidth=2.0,
            label=r"Baseline: $T=y$",
        )

        profile_ax.set_xlabel("Temperature, T")
        profile_ax.set_ylabel("Vertical coordinate, y")
        profile_ax.set_ylim(0, domain_length_y)
        profile_ax.grid(True, alpha=0.3)

        # Include both the baseline range [0, 1] and computed T
        # in the temperature axis limits.
        local_min = min(
            0.0,
            float(np.min(T[ix, :])),
        )
        local_max = max(
            domain_length_y,
            float(np.max(T[ix, :])),
        )

        span = max(local_max - local_min, 1e-12)
        profile_ax.set_xlim(
            local_min - 0.05 * span,
            local_max + 0.05 * span,
        )

        profile_ax.set_title(
            rf"Cross-section at $x/L_x={fraction:.2f}$"
            f"\nNearest cell centre: x = {actual_x:.4g}"
        )

        # Secondary horizontal axis: velocity gradient.
        # This keeps du/dy from sharing temperature's scale.
        gradient_ax = profile_ax.twiny()

        # Viscous dissipation function: Phi = (du/dy)^2
        phi = u_y**2

        # Normalize Phi to the interval [0, 1].
        phi_max = float(np.max(phi))

        if phi_max > 0.0:
            phi_normalized = phi / phi_max
        else:
            phi_normalized = np.zeros_like(phi)

        line_gradient, = profile_ax.plot(
            phi_normalized,
            y,
            color="tab:green",
            linewidth=1.8,
            linestyle="-.",
            label=r"Normalized dissipation, $\Phi/\Phi_{\max}$",
        )

        profile_ax.set_xlabel(
            r"Normalized viscous dissipation, $\Phi/\Phi_{\max}$"
        )

        # Combined legend for both horizontal axes.
        profile_ax.legend(
            handles=[
                line_temperature,
                line_baseline,
                line_gradient,
            ],
            loc="best",
            fontsize=9,
        )

    fig.suptitle(
        "Temperature contours and cross-sectional profiles",
        fontsize=16,
        fontweight="bold",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.96))

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved cross-section figure: {output_path}")

    return output_path