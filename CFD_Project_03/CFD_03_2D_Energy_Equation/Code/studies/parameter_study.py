from pathlib import Path

import numpy as np

from src.initialization import (
    inlet_temperature as inlet_temperature_profile,
    initialize_temperature,
    initialize_velocity_fields,
)
from src.solvers.direct import solve_with_direct

from src.visualization import (
    plot_parameter_group,
    plot_case_cross_sections,
)


# =============================================================
# Project configuration
# =============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIRECTORY = (
    PROJECT_ROOT / "results" / "parameter_study"
)

# Use your established mesh settings if they differ.
CELL_COUNT_X = 100
CELL_COUNT_Y = 40

MESH = (CELL_COUNT_X, CELL_COUNT_Y)

DOMAIN_LENGTH_X = 5.0
DOMAIN_LENGTH_Y = 1.0

TIME_STEP = 1
MAX_ITERATIONS = 100_000
TOLERANCE = 1e-10

MEAN_VELOCITY = 3.0


# =============================================================
# Parameter-study cases
# =============================================================

# Keep Br/Pe fixed for the Peclet-number comparison.
SOURCE_RATIO = 0.07 / 17.5

CASES = {
    # ---------------------------------------------------------
    # Dissipation-strength study
    # ---------------------------------------------------------
    "br_0": {
        "label": "No viscous dissipation",
        "Pe": 17.5,
        "Br": 0.0,
        "gamma_c": 1.0,
    },
    "br_007": {
        "label": "Baseline dissipation",
        "Pe": 17.5,
        "Br": 0.07,
        "gamma_c": 1.0,
    },
    "br_02": {
        "label": "Stronger dissipation",
        "Pe": 17.5,
        "Br": 0.2,
        "gamma_c": 1.0,
    },

    # ---------------------------------------------------------
    # Convection-strength study
    # ---------------------------------------------------------
    "gamma_04": {
        "label": "Weak convection",
        "Pe": 17.5,
        "Br": 0.2,
        "gamma_c": 0.4,
    },
    "gamma_10": {
        "label": "Strong convection",
        "Pe": 17.5,
        "Br": 0.2,
        "gamma_c": 10.0,
    },

    # ---------------------------------------------------------
    # Peclet-number study
    # Br/Pe is fixed across these cases.
    # ---------------------------------------------------------
    "pe_1": {
        "label": "Low Peclet number",
        "Pe": 1.0,
        "Br": SOURCE_RATIO * 1.0,
        "gamma_c": 1.0,
    },
    "pe_50": {
        "label": "High Peclet number",
        "Pe": 50.0,
        "Br": SOURCE_RATIO * 50.0,
        "gamma_c": 1.0,
    },

}


# =============================================================
# Comparison groups
# Each group becomes one figure with vertically stacked contours.
# =============================================================

CASE_GROUPS = {
    "brinkman_number": [
        "br_0",
        "br_007",
        "br_02",
    ],
    "convection_coefficient": [
        "gamma_04",
        "br_02",
        "gamma_10",
    ],
    "peclet_number": [
        "pe_1",
        "br_007",
        "pe_50",
    ],
    
}

GROUP_TITLES = {
    "brinkman_number": "Effect of viscous dissipation strength",
    "convection_coefficient": "Effect of convection strength",
    "peclet_number": "Effect of isolated BC diffusion strength",
}


# These three cases each receive one contour-plus-cross-sections figure.
CROSS_SECTION_CASES = [
    "br_007",    # Pe=17.5, Br=0.07, gamma_c=1
    "br_02",     # Pe=17.5, Br=0.2,  gamma_c=1
    "gamma_04",  # Pe=17.5, Br=0.2,  gamma_c=0.1
]

CROSS_SECTION_FRACTIONS = (0.1, 0.9)


# =============================================================
# Solve one case
# =============================================================

def solve_case(case_id, case):
    """Solve one parameter-study case using the AFI solver."""

    Pe = case["Pe"]
    Br = case["Br"]
    gamma_c = case["gamma_c"]

    print(
        f"\nCase: {case_id}\n"
        f"  Pe = {Pe:g}\n"
        f"  Br = {Br:g}\n"
        f"  convection_coefficient = {gamma_c:g}"
    )

    # ---------------------------------------------------------
    # Initialize temperature and velocity.
    # ---------------------------------------------------------
    initial_temperature = initialize_temperature(
        mesh=MESH,
        domain_length_y=DOMAIN_LENGTH_Y,
    )

    u_velocity, v_velocity = initialize_velocity_fields(
        mesh=MESH,
        domain_length_y=DOMAIN_LENGTH_Y,
        mean_velocity=MEAN_VELOCITY,
    )

    # ---------------------------------------------------------
    # Solve the energy equation.
    # ---------------------------------------------------------
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
        mesh=MESH,
        inlet_temperature=inlet_temperature_profile,
        peclet_number=Pe,
        brinkman_number=Br,
        u_velocity=u_velocity,
        v_velocity=v_velocity,
        domain_length_x=DOMAIN_LENGTH_X,
        domain_length_y=DOMAIN_LENGTH_Y,
        mean_velocity=MEAN_VELOCITY,
        residual_tolerance=TOLERANCE,
        max_iterations=MAX_ITERATIONS,
        convection_coefficient=gamma_c,
    )

    # ---------------------------------------------------------
    # Extract final diagnostics.
    # ---------------------------------------------------------
    iteration_history = np.asarray(iteration_history)
    residual_history = np.asarray(residual_history)
    snapshot_iterations = np.asarray(snapshot_iterations)

    iterations = (
        int(iteration_history[-1])
        if iteration_history.size > 0
        else 0
    )

    residual = (
        float(residual_history[-1])
        if residual_history.size > 0
        else float("nan")
    )

    print(f"  Iterations = {iterations}")
    print(f"  Final residual = {residual:.6e}")
    print(f"  Computation time = {computation_time:.2f} s")

    return {
        "temperature": np.asarray(temperature),
        "residual": residual,
        "elapsed_time": float(computation_time),
        "iterations": iterations,
        "iteration_history": iteration_history,
        "residual_history": residual_history,
        "temperature_history": temperature_history,
        "snapshot_iterations": snapshot_iterations,
    }


# =============================================================
# Run the parameter study
# =============================================================

def run_parameter_study(cases=CASES):
    """Solve all cases, create comparison figures, and save fields."""

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_directory = RESULTS_DIRECTORY / "figures"

    group_directory = figures_directory / "parameter_groups"
    cross_section_directory = figures_directory / "cross_sections"

    group_directory.mkdir(parents=True, exist_ok=True)
    cross_section_directory.mkdir(parents=True, exist_ok=True)

    case_ids = list(cases.keys())

    if not case_ids:
        raise ValueError("No parameter-study cases were defined.")

    # Validate all group and cross-section references.
    for group_name, group_case_ids in CASE_GROUPS.items():
        for case_id in group_case_ids:
            if case_id not in cases:
                raise KeyError(
                    f"Case '{case_id}' in group '{group_name}' "
                    "is not defined in CASES."
                )

    for case_id in CROSS_SECTION_CASES:
        if case_id not in cases:
            raise KeyError(
                f"Cross-section case '{case_id}' is not defined."
            )

    print("\n" + "=" * 72)
    print("PARAMETER STUDY")
    print("=" * 72)
    print(f"Mesh: {MESH[0]} x {MESH[1]}")
    print(
        f"Domain: {DOMAIN_LENGTH_X:g} x {DOMAIN_LENGTH_Y:g}"
    )
    print(f"Time step: {TIME_STEP:g}")
    print(f"Maximum iterations: {MAX_ITERATIONS}")
    print(f"Number of unique cases: {len(case_ids)}")

    # ---------------------------------------------------------
    # A. Solve each unique case once.
    # ---------------------------------------------------------
    results = {}

    for i, case_id in enumerate(case_ids, start=1):
        print(f"\n[{i}/{len(case_ids)}] {case_id}")

        results[case_id] = solve_case(
            case_id=case_id,
            case=cases[case_id],
        )

    # ---------------------------------------------------------
    # B. Calculate common temperature limits.
    # All contour figures use the same color scale.
    # ---------------------------------------------------------
    minima = []
    maxima = []

    for case_id in case_ids:
        interior = results[case_id]["temperature"][1:-1, 1:-1]

        if not np.all(np.isfinite(interior)):
            raise ValueError(
                f"Non-finite temperatures detected in case '{case_id}'."
            )

        minima.append(float(np.min(interior)))
        maxima.append(float(np.max(interior)))

    temperature_limits = (
        min(minima),
        max(maxima),
    )

    print("\nCommon contour limits:")
    print(f"  Tmin = {temperature_limits[0]:.6g}")
    print(f"  Tmax = {temperature_limits[1]:.6g}")

    # ---------------------------------------------------------
    # C. Create one vertically stacked contour figure per group.
    # ---------------------------------------------------------
    print("\nCreating parameter-group contour figures...")

    for group_name, group_case_ids in CASE_GROUPS.items():
        output_path = (
            group_directory / f"{group_name}.png"
        )

        plot_parameter_group(
            results=results,
            cases=cases,
            case_ids=group_case_ids,
            group_title=GROUP_TITLES[group_name],
            output_path=output_path,
            mesh=MESH,
            domain_length_x=DOMAIN_LENGTH_X,
            domain_length_y=DOMAIN_LENGTH_Y,
            temperature_limits=temperature_limits,
        )

    # ---------------------------------------------------------
    # D. Create one composite figure per selected case.
    #
    # Each figure contains:
    #   Top: contour.
    #   Bottom-left: x/Lx = 0.15.
    #   Bottom-right: x/Lx = 0.85.
    # ---------------------------------------------------------
    print("\nCreating contour and cross-section figures...")

    for case_id in CROSS_SECTION_CASES:
        output_path = (
            cross_section_directory
            / f"{case_id}_cross_sections.png"
        )

        plot_case_cross_sections(
            temperature=results[case_id]["temperature"],
            mesh=MESH,
            domain_length_x=DOMAIN_LENGTH_X,
            domain_length_y=DOMAIN_LENGTH_Y,
            case=cases[case_id],
            output_path=output_path,
            temperature_limits=temperature_limits,
            section_fractions=CROSS_SECTION_FRACTIONS,
            mean_velocity=MEAN_VELOCITY,
        )

    # ---------------------------------------------------------
    # E. Save fields and numerical diagnostics.
    # ---------------------------------------------------------
    data_path = (
        RESULTS_DIRECTORY / "parameter_study_fields.npz"
    )

    np.savez_compressed(
        data_path,
        case_ids=np.asarray(case_ids),
        labels=np.asarray([
            cases[case_id]["label"]
            for case_id in case_ids
        ]),
        temperatures=np.stack([
            results[case_id]["temperature"]
            for case_id in case_ids
        ]),
        case_parameters=np.asarray([
            [
                cases[case_id]["Pe"],
                cases[case_id]["Br"],
                cases[case_id]["gamma_c"],
            ]
            for case_id in case_ids
        ], dtype=float),
        parameter_columns=np.asarray([
            "Pe",
            "Br",
            "gamma_c",
        ]),
        residuals=np.asarray([
            results[case_id]["residual"]
            for case_id in case_ids
        ]),
        computation_times=np.asarray([
            results[case_id]["elapsed_time"]
            for case_id in case_ids
        ]),
        iterations=np.asarray([
            results[case_id]["iterations"]
            for case_id in case_ids
        ]),
        mesh=np.asarray(MESH),
        domain_length_x=DOMAIN_LENGTH_X,
        domain_length_y=DOMAIN_LENGTH_Y,
        time_step=TIME_STEP,
        max_iterations=MAX_ITERATIONS,
        tolerance=TOLERANCE,
        temperature_limits=np.asarray(
            temperature_limits,
            dtype=float,
        ),
    )

    print("\n" + "=" * 72)
    print("PARAMETER STUDY COMPLETED")
    print("=" * 72)
    print(f"Group figures: {group_directory}")
    print(f"Cross-section figures: {cross_section_directory}")
    print(f"Numerical data: {data_path}")
    print(f"Unique cases solved: {len(case_ids)}")
    print(f"Parameter-group figures: {len(CASE_GROUPS)}")
    print(f"Cross-section figures: {len(CROSS_SECTION_CASES)}")

    return results


if __name__ == "__main__":
    run_parameter_study()