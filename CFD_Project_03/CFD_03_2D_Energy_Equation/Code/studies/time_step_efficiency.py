
# studies/time_step_efficiency.py

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, FuncFormatter

from src import config
from src.initialization import (
    initialize_temperature,
    initialize_velocity_fields,
)
from src.diagnostics import steady_state_residual_rms
from src.solvers.apf import solve_with_apf
from src.solvers.direct import solve_with_direct


# ============================================================
# Study configuration
# ============================================================

GRID = (config.CELL_COUNT_X, config.CELL_COUNT_Y)

REPEATS = 3

# Refined around the observed AFI minimum near dt = 0.271.
# Direct showed only modest changes in runtime above approximately
# dt = 15000, so this range examines that region more closely.
TIME_STEPS = {
    "AFI": [
        0.265,
        0.266,
        0.267,
        0.2675,
        0.26775,
        0.268,
        0.26825,
        0.2685,
        0.26875,
        0.269,
        0.26925,
        0.2695,
        0.26975,
        0.270,
        0.271,
    ],
    "Direct": [
        5000,
        8000,
        10000,
        12000,
        14000,
        15000,
        16000,
        18000,
        20000,
        25000,
        30000,
        60000,
    ],
}

SOLVERS = {
    "AFI": solve_with_apf,
    "Direct": solve_with_direct,
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "results" / "time_step_efficiency"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Inlet boundary condition
# ============================================================

def linear_inlet_temperature(y):
    """Linear inlet profile: T_in(y) = y."""
    return y


# ============================================================
# Initialize a fresh case
# ============================================================

def build_case():
    """Initialize a fresh problem using the settings in src.config."""

    mesh = GRID

    initial_temperature = initialize_temperature(
        mesh=mesh,
        domain_length_y=config.DOMAIN_LENGTH_Y,
    )

    u_velocity, v_velocity = initialize_velocity_fields(
        mesh=mesh,
        domain_length_y=config.DOMAIN_LENGTH_Y,
        mean_velocity=config.MEAN_VELOCITY,
    )

    return {
        "mesh": mesh,
        "initial_temperature": initial_temperature,
        "inlet_temperature": linear_inlet_temperature,
        "u_velocity": u_velocity,
        "v_velocity": v_velocity,
        "domain_length_x": config.DOMAIN_LENGTH_X,
        "domain_length_y": config.DOMAIN_LENGTH_Y,
        "peclet_number": config.PECLET_NUMBER,
        "brinkman_number": config.BRINKMAN_NUMBER,
        "mean_velocity": config.MEAN_VELOCITY,
    }


# ============================================================
# Axis formatting
# ============================================================

def configure_time_step_axis(ax, solver_name):
    """Format the horizontal axis appropriately for each solver."""

    ax.set_xlabel("Time step, Δt")
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.12)

    if solver_name == "AFI":
        # AFI time steps are tightly clustered, so retain enough
        # decimal places while avoiding a label at every data point.
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.5f"))
        ax.tick_params(axis="x", labelrotation=25)

    else:
        # A linear axis makes the runtime plateau around dt=15000
        # easier to interpret than a logarithmic horizontal axis.
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:,.0f}")
        )


# ============================================================
# Plotting
# ============================================================

def plot_efficiency_results(results):
    """Create separate runtime, overhead, iteration, and residual plots."""

    for solver_name in SOLVERS:
        solver_results = results[results["solver"] == solver_name].copy()

        reliable = solver_results[
            solver_results["all_runs_converged"]
            & solver_results["median_computation_time_s"].notna()
        ].sort_values("dt")

        if reliable.empty:
            print(f"\nNo fully converged cases available to plot for {solver_name}.")
            continue

        best_index = reliable["median_computation_time_s"].idxmin()
        best = reliable.loc[best_index]

        dt = reliable["dt"].to_numpy(dtype=float)
        median_time = reliable["median_computation_time_s"].to_numpy(dtype=float)
        q25_time = reliable["q25_computation_time_s"].to_numpy(dtype=float)
        q75_time = reliable["q75_computation_time_s"].to_numpy(dtype=float)

        # Asymmetric error bars span the interquartile range (25th-75th
        # percentile) of repeated runs.
        runtime_errors = np.vstack([
            np.maximum(median_time - q25_time, 0.0),
            np.maximum(q75_time - median_time, 0.0),
        ])

        # --------------------------------------------------------
        # Plot 1: median computation time versus timestep
        # --------------------------------------------------------

        fig, ax = plt.subplots(figsize=(9, 5.5))

        ax.errorbar(
            dt,
            median_time,
            yerr=runtime_errors,
            marker="o",
            capsize=3,
            linewidth=1.5,
            label="Median runtime (IQR)",
        )

        ax.scatter(
            [best["dt"]],
            [best["median_computation_time_s"]],
            marker="*",
            s=180,
            zorder=3,
            label=f"Fastest tested: dt = {best['dt']:g}",
        )

        ax.set_ylabel("Computation time reported by solver (s)")
        ax.set_title(f"{solver_name}: computation time versus time step")
        configure_time_step_axis(ax, solver_name)
        ax.margins(y=0.15)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / f"runtime_vs_dt_{solver_name}.png",
            dpi=300,
        )
        plt.close(fig)

        # --------------------------------------------------------
        # Plot 2: runtime overhead relative to the fastest tested dt
        # --------------------------------------------------------

        best_time = float(best["median_computation_time_s"])

        overhead = (median_time / best_time - 1.0) * 100.0
        q25_overhead = (q25_time / best_time - 1.0) * 100.0
        q75_overhead = (q75_time / best_time - 1.0) * 100.0

        overhead_errors = np.vstack([
            np.maximum(overhead - q25_overhead, 0.0),
            np.maximum(q75_overhead - overhead, 0.0),
        ])

        fig, ax = plt.subplots(figsize=(9, 5.5))

        ax.errorbar(
            dt,
            overhead,
            yerr=overhead_errors,
            marker="o",
            capsize=3,
            linewidth=1.5,
        )

        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.scatter(
            [best["dt"]],
            [0.0],
            marker="*",
            s=180,
            zorder=3,
            label="Fastest tested",
        )

        ax.set_ylabel("Runtime above minimum (%)")
        ax.set_title(f"{solver_name}: relative runtime overhead")
        configure_time_step_axis(ax, solver_name)
        ax.margins(y=0.15)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / f"runtime_overhead_{solver_name}.png",
            dpi=300,
        )
        plt.close(fig)

        # --------------------------------------------------------
        # Plot 3: median iteration count versus timestep
        # --------------------------------------------------------

        iterations = reliable["median_iterations"].to_numpy(dtype=float)
        q25_iterations = reliable["q25_iterations"].to_numpy(dtype=float)
        q75_iterations = reliable["q75_iterations"].to_numpy(dtype=float)

        iteration_errors = np.vstack([
            np.maximum(iterations - q25_iterations, 0.0),
            np.maximum(q75_iterations - iterations, 0.0),
        ])

        fig, ax = plt.subplots(figsize=(9, 5.5))

        ax.errorbar(
            dt,
            iterations,
            yerr=iteration_errors,
            marker="o",
            capsize=3,
            linewidth=1.5,
        )

        ax.set_ylabel("Iteration count")
        ax.set_title(f"{solver_name}: iteration count versus time step")
        configure_time_step_axis(ax, solver_name)
        ax.margins(y=0.15)
        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / f"iterations_vs_dt_{solver_name}.png",
            dpi=300,
        )
        plt.close(fig)

        print(
            f"\n{solver_name}: fastest fully converged tested case"
            f"\n  dt                    = {best['dt']:g}"
            f"\n  median computation time = "
            f"{best['median_computation_time_s']:.6f} s"
            f"\n  median iterations      = {best['median_iterations']:g}"
        )

        # --------------------------------------------------------
        # Plot 4: residual versus timestep
        # Include non-converged cases if a finite residual was obtained.
        # --------------------------------------------------------

        residual_data = solver_results[
            np.isfinite(solver_results["median_final_residual"])
            & (solver_results["median_final_residual"] > 0.0)
        ].sort_values("dt")

        if not residual_data.empty:
            fig, ax = plt.subplots(figsize=(9, 5.5))

            ax.plot(
                residual_data["dt"],
                residual_data["median_final_residual"],
                marker="o",
                linewidth=1.5,
            )

            ax.axhline(
                config.RESIDUAL_TOLERANCE,
                linestyle="--",
                linewidth=1,
                label="Configured residual tolerance",
            )

            ax.set_yscale("log")
            ax.set_ylabel("Final steady-state residual")
            ax.set_title(f"{solver_name}: final residual versus time step")
            configure_time_step_axis(ax, solver_name)
            ax.legend()
            fig.tight_layout()
            fig.savefig(
                OUTPUT_DIR / f"residual_vs_dt_{solver_name}.png",
                dpi=300,
            )
            plt.close(fig)


# ============================================================
# Main study
# ============================================================

def run_time_step_efficiency_study():
    """Run the AFI and Direct time-step efficiency study."""

    records = []
    converged_solutions = {}

    nx, ny = GRID

    print("\n" + "=" * 80)
    print("TIME-STEP EFFICIENCY STUDY")
    print("=" * 80)
    print(f"Grid: {nx} x {ny}")
    print(f"Domain: {config.DOMAIN_LENGTH_X:g} x {config.DOMAIN_LENGTH_Y:g}")
    print(f"Re = {config.REYNOLDS_NUMBER:g}")
    print(f"Pr = {config.PRANDTL_NUMBER:g}")
    print(f"Pe = {config.PECLET_NUMBER:g}")
    print(f"Br = {config.BRINKMAN_NUMBER:g}")
    print(f"Mean velocity = {config.MEAN_VELOCITY:g}")
    print(f"Convection coefficient = {config.convection_coefficient:g}")
    print(f"Residual tolerance = {config.RESIDUAL_TOLERANCE:.1e}")
    print(f"Maximum iterations = {config.MAX_ITERATIONS}")
    print(f"Repeats per timestep = {REPEATS}")
    print(f"Output directory = {OUTPUT_DIR}")

    for solver_name, solver in SOLVERS.items():

        print("\n" + "-" * 80)
        print(f"Solver: {solver_name}")
        print("-" * 80)

        for dt in TIME_STEPS[solver_name]:

            computation_times = []
            wall_times = []
            iteration_counts = []
            final_residuals = []
            run_converged = []
            run_temperatures = []
            failures = []

            print(f"\nΔt = {dt:g}")

            for repeat in range(REPEATS):
                start = perf_counter()

                try:
                    # Each repeat starts from the same fresh initial state.
                    case = build_case()

                    result = solver(
                        initial_temperature=case["initial_temperature"],
                        time_step=dt,
                        mesh=case["mesh"],
                        inlet_temperature=case["inlet_temperature"],
                        peclet_number=case["peclet_number"],
                        brinkman_number=case["brinkman_number"],
                        u_velocity=case["u_velocity"],
                        v_velocity=case["v_velocity"],
                        domain_length_x=case["domain_length_x"],
                        domain_length_y=case["domain_length_y"],
                        mean_velocity=case["mean_velocity"],
                        residual_tolerance=config.RESIDUAL_TOLERANCE,
                        max_iterations=config.MAX_ITERATIONS,
                        snapshot_interval=config.MAX_ITERATIONS + 1,
                        convection_coefficient=config.convection_coefficient,
                    )

                    wall_time = perf_counter() - start

                    (
                        temperature,
                        iteration_history,
                        residual_history,
                        computation_time,
                        temperature_history,
                        snapshot_iterations,
                    ) = result

                    temperature = np.asarray(temperature)
                    computation_time = float(computation_time)

                    if not np.all(np.isfinite(temperature)):
                        final_residual = np.inf
                    else:
                        final_residual = steady_state_residual_rms(
                            mesh=case["mesh"],
                            temperature=temperature,
                            peclet_number=case["peclet_number"],
                            brinkman_number=case["brinkman_number"],
                            u_velocity=case["u_velocity"],
                            v_velocity=case["v_velocity"],
                            domain_length_x=case["domain_length_x"],
                            domain_length_y=case["domain_length_y"],
                            convection_coefficient=config.convection_coefficient,
                        )
                        final_residual = float(final_residual)

                    converged = (
                        np.all(np.isfinite(temperature))
                        and np.isfinite(final_residual)
                        and final_residual <= config.RESIDUAL_TOLERANCE
                    )

                    computation_times.append(computation_time)
                    wall_times.append(float(wall_time))
                    iteration_counts.append(len(iteration_history))
                    final_residuals.append(final_residual)
                    run_converged.append(bool(converged))

                    if np.all(np.isfinite(temperature)):
                        run_temperatures.append(temperature.copy())

                    print(
                        f"  Run {repeat + 1}/{REPEATS}: "
                        f"solver time = {computation_time:.4f} s, "
                        f"wall time = {wall_time:.4f} s, "
                        f"iterations = {len(iteration_history)}, "
                        f"residual = {final_residual:.3e}, "
                        f"converged = {converged}"
                    )

                except Exception as exc:
                    # Preserve the rest of the sweep if one case fails.
                    run_converged.append(False)
                    failures.append(f"{type(exc).__name__}: {exc}")

                    print(
                        f"  Run {repeat + 1}/{REPEATS} failed: "
                        f"{type(exc).__name__}: {exc}"
                    )

            all_runs_converged = (
                len(run_converged) == REPEATS
                and all(run_converged)
                and len(computation_times) == REPEATS
            )

            # Calculate a repeat-to-repeat solution difference as an
            # additional check on runs that all converged.
            repeat_solution_difference = np.nan

            if all_runs_converged and len(run_temperatures) > 1:
                reference = run_temperatures[0][1:-1, 1:-1]
                reference_norm = max(np.linalg.norm(reference.ravel()), 1e-30)

                repeat_solution_difference = max(
                    np.linalg.norm(
                        (temperature[1:-1, 1:-1] - reference).ravel()
                    ) / reference_norm
                    for temperature in run_temperatures[1:]
                )

                converged_solutions[(solver_name, dt)] = run_temperatures[0]

            record = {
                "solver": solver_name,
                "nx": nx,
                "ny": ny,
                "dt": dt,
                "repeats_requested": REPEATS,
                "successful_runs": len(computation_times),
                "all_runs_converged": all_runs_converged,
                "median_computation_time_s": (
                    float(np.median(computation_times))
                    if computation_times else np.nan
                ),
                "q25_computation_time_s": (
                    float(np.percentile(computation_times, 25))
                    if computation_times else np.nan
                ),
                "q75_computation_time_s": (
                    float(np.percentile(computation_times, 75))
                    if computation_times else np.nan
                ),
                "median_wall_time_s": (
                    float(np.median(wall_times))
                    if wall_times else np.nan
                ),
                "median_iterations": (
                    float(np.median(iteration_counts))
                    if iteration_counts else np.nan
                ),
                "q25_iterations": (
                    float(np.percentile(iteration_counts, 25))
                    if iteration_counts else np.nan
                ),
                "q75_iterations": (
                    float(np.percentile(iteration_counts, 75))
                    if iteration_counts else np.nan
                ),
                "median_final_residual": (
                    float(np.median(final_residuals))
                    if final_residuals else np.nan
                ),
                "repeat_solution_relative_difference": repeat_solution_difference,
                "failure_messages": " | ".join(failures),
            }

            records.append(record)

    # ========================================================
    # Save results
    # ========================================================

    results = pd.DataFrame(records)

    csv_path = OUTPUT_DIR / "time_step_efficiency.csv"
    results.to_csv(csv_path, index=False)

    summary_columns = [
        "solver",
        "dt",
        "median_computation_time_s",
        "median_iterations",
        "median_final_residual",
        "all_runs_converged",
    ]

    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)
    print(results[summary_columns].to_string(index=False))

    # ========================================================
    # Generate separate solver plots
    # ========================================================

    plot_efficiency_results(results)

    # ========================================================
    # Compare the fastest fully converged solution from each solver
    # ========================================================

    reliable = results[
        results["all_runs_converged"]
        & results["median_computation_time_s"].notna()
    ]

    best_by_solver = {}

    if not reliable.empty:
        for solver_name in SOLVERS:
            solver_data = reliable[reliable["solver"] == solver_name]

            if solver_data.empty:
                print(f"\nNo fully converged tested timestep for {solver_name}.")
                continue

            best_row = solver_data.loc[
                solver_data["median_computation_time_s"].idxmin()
            ]

            best_by_solver[solver_name] = best_row

    if "AFI" in best_by_solver and "Direct" in best_by_solver:
        afi_row = best_by_solver["AFI"]
        direct_row = best_by_solver["Direct"]

        afi_solution = converged_solutions.get(("AFI", afi_row["dt"]))
        direct_solution = converged_solutions.get(("Direct", direct_row["dt"]))

        if afi_solution is not None and direct_solution is not None:
            afi_interior = afi_solution[1:-1, 1:-1]
            direct_interior = direct_solution[1:-1, 1:-1]

            relative_difference = (
                np.linalg.norm((afi_interior - direct_interior).ravel())
                / max(np.linalg.norm(direct_interior.ravel()), 1e-30)
            )

            print("\nFastest tested converged cases:")
            print(
                f"  AFI:    dt = {afi_row['dt']:g}, "
                f"median time = "
                f"{afi_row['median_computation_time_s']:.6f} s"
            )
            print(
                f"  Direct: dt = {direct_row['dt']:g}, "
                f"median time = "
                f"{direct_row['median_computation_time_s']:.6f} s"
            )
            print(
                "  Relative difference between their converged "
                f"temperature fields = {relative_difference:.6e}"
            )

    print(f"\nCSV saved to: {csv_path.resolve()}")
    print(f"Plots saved to: {OUTPUT_DIR.resolve()}")

    return results
