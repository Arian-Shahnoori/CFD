from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.config import (
    DOMAIN_LENGTH_X,
    DOMAIN_LENGTH_Y,
    CELL_COUNT_X,
    CELL_COUNT_Y,
    TIME_STEP,
    REYNOLDS_NUMBER,
    PRANDTL_NUMBER,
    ECKERT_NUMBER,
    PECLET_NUMBER,
    BRINKMAN_NUMBER,
    MEAN_VELOCITY,
    RESIDUAL_TOLERANCE,
    MAX_ITERATIONS,
    convection_coefficient,
)

from src.initialization import (
    inlet_temperature,
    initialize_temperature,
    initialize_velocity_fields,
)

from src.visualization import animate_temperature_history


# ============================================================
# Output paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIRECTORY = PROJECT_ROOT / "results"

SNAPSHOT_FILE = RESULTS_DIRECTORY / "temperature_history.npz"
FINAL_TEMPERATURE_FILE = RESULTS_DIRECTORY / "final_temperature.png"
GIF_FILE = RESULTS_DIRECTORY / "temperature_evolution.gif"


# ============================================================
# Save final temperature field as PNG
# ============================================================

def save_final_temperature(temperature):
    """Save the final physical-cell temperature field as a PNG."""

    # Exclude ghost cells.
    physical_temperature = np.asarray(temperature)[1:-1, 1:-1]

    if not np.all(np.isfinite(physical_temperature)):
        raise ValueError("Final temperature field contains non-finite values.")

    fig, ax = plt.subplots(figsize=(12, 3.5))

    image = ax.imshow(
        physical_temperature.T,
        origin="lower",
        extent=[
            0.0,
            DOMAIN_LENGTH_X,
            0.0,
            DOMAIN_LENGTH_Y,
        ],
        aspect="equal",
        cmap="inferno",
        interpolation="nearest",
    )

    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Temperature, $T$")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(
    "Temperature-Field Evolution (Pseudo-Time)\n"
    rf"$\mathrm{{Re}}={REYNOLDS_NUMBER:g},\quad \mathrm{{Pr}}={PRANDTL_NUMBER:g},\quad "
    rf"\mathrm{{Ec}}={ECKERT_NUMBER:g},\quad \mathrm{{Pe}}={PECLET_NUMBER:g},\quad "
    rf"\mathrm{{Br}}={BRINKMAN_NUMBER:g}$"
)

    fig.tight_layout()

    fig.savefig(
        FINAL_TEMPERATURE_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Final temperature PNG saved to:\n{FINAL_TEMPERATURE_FILE}")


# ============================================================
# Run simulation and save results
# ============================================================

def run_snapshots():
    """Run the direct solver, save snapshots, and generate a GIF."""

    from src.solvers.direct import solve_with_direct

    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Grid and initial conditions
    # --------------------------------------------------------

    mesh = (CELL_COUNT_X, CELL_COUNT_Y)

    initial_temperature = initialize_temperature(
        mesh=mesh,
        domain_length_y=DOMAIN_LENGTH_Y,
    )

    u_velocity, v_velocity = initialize_velocity_fields(
        mesh=mesh,
        domain_length_y=DOMAIN_LENGTH_Y,
        mean_velocity=MEAN_VELOCITY,
    )

    # --------------------------------------------------------
    # Solve the energy equation
    # --------------------------------------------------------

    (
        temperature,
        iteration_history,
        residual_history,
        time_history,
        computation_time,
        temperature_history,
        snapshot_iterations,
    ) = solve_with_direct(
        initial_temperature=initial_temperature,
        time_step=TIME_STEP,
        mesh=mesh,
        inlet_temperature=inlet_temperature,
        peclet_number=PECLET_NUMBER,
        brinkman_number=BRINKMAN_NUMBER,
        u_velocity=u_velocity,
        v_velocity=v_velocity,
        domain_length_x=DOMAIN_LENGTH_X,
        domain_length_y=DOMAIN_LENGTH_Y,
        mean_velocity=MEAN_VELOCITY,
        residual_tolerance=RESIDUAL_TOLERANCE,
        max_iterations=MAX_ITERATIONS,
        snapshot_interval=1,
        convection_coefficient=convection_coefficient,
    )

    temperature = np.asarray(temperature)

    print(f"\nSolver completed in {computation_time:.2f} seconds.")
    print(f"Total iterations: {len(iteration_history)}")
    print(f"Final residual: {residual_history[-1]:.3e}")

    # --------------------------------------------------------
    # Save final temperature distribution as PNG
    # --------------------------------------------------------

    save_final_temperature(temperature)

    # --------------------------------------------------------
    # Save temperature snapshots
    # --------------------------------------------------------

    np.savez_compressed(
        SNAPSHOT_FILE,
        temperatures=np.asarray(temperature_history),
        iterations=np.asarray(snapshot_iterations),
        final_temperature=temperature,
        time_step=TIME_STEP,
        domain_length_x=DOMAIN_LENGTH_X,
        domain_length_y=DOMAIN_LENGTH_Y,
        residual_history=np.asarray(residual_history),
        time_history=np.asarray(time_history),
    )

    print(f"\nTemperature snapshots saved to:\n{SNAPSHOT_FILE}")

    # --------------------------------------------------------
    # Generate temperature evolution GIF
    # --------------------------------------------------------

    animate_temperature_history(
        temperature_history=temperature_history[:200],
        snapshot_iterations=snapshot_iterations[:200],
        time_step=TIME_STEP,
        domain_length_x=DOMAIN_LENGTH_X,
        domain_length_y=DOMAIN_LENGTH_Y,
        reynolds_number=REYNOLDS_NUMBER,
        prandtl_number=PRANDTL_NUMBER,
        eckert_number=ECKERT_NUMBER,
        peclet_number=PECLET_NUMBER,
        brinkman_number=BRINKMAN_NUMBER,
        gif_path=GIF_FILE,
        fps=30,
    )

    print(f"\nTemperature evolution GIF saved to:\n{GIF_FILE}")

    return temperature


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    run_snapshots()