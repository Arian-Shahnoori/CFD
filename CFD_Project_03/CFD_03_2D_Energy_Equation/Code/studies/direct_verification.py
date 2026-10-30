import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import splu

from src.config import (
    DOMAIN_LENGTH_X,
    DOMAIN_LENGTH_Y,
    PECLET_NUMBER,
    TIME_STEP,
    BRINKMAN_NUMBER,
    MEAN_VELOCITY,
    RESIDUAL_TOLERANCE,
    MAX_ITERATIONS,
)
from src.initialization import initialize_velocity_fields, initialize_temperature
from src.boundary_conditions import apply_temperature_boundary_conditions
from src.solvers.apf import build_rhs
from src.solvers.direct import build_spatial_matrix, build_forcing_field
from src.solvers.apf import solve_with_apf
from src.solvers.direct import solve_with_direct


def inlet_temperature(y):
    """Linear inlet temperature profile."""
    return y


def run_direct_matrix_consistency():
    # Small grid for a quick consistency test
    nx, ny = 24, 12
    mesh = (nx, ny)
    convection_coefficient = 1.0

    # Construct velocity fields
    u_velocity, v_velocity = initialize_velocity_fields(
        mesh, DOMAIN_LENGTH_Y, MEAN_VELOCITY
    )

    # Assemble the spatial matrix A
    A = build_spatial_matrix(
        nx=nx,
        ny=ny,
        peclet_number=PECLET_NUMBER,
        length_x=DOMAIN_LENGTH_X,
        length_y=DOMAIN_LENGTH_Y,
        u_velocity=u_velocity,
        v_velocity=v_velocity,
        mean_velocity=MEAN_VELOCITY,
        convection_coefficient=convection_coefficient,
    )

    # Calculate b, the constant contribution from the boundary conditions
    b = build_forcing_field(
        mesh=mesh,
        inlet_temperature=inlet_temperature,
        peclet_number=PECLET_NUMBER,
        brinkman_number=BRINKMAN_NUMBER,
        u_velocity=u_velocity,
        v_velocity=v_velocity,
        domain_length_x=DOMAIN_LENGTH_X,
        domain_length_y=DOMAIN_LENGTH_Y,
        convection_coefficient=convection_coefficient,
    ).ravel(order="C")

    # Generate an arbitrary physical temperature field
    rng = np.random.default_rng(42)
    temperature = np.zeros((nx + 2, ny + 2))
    temperature[1:-1, 1:-1] = rng.random((nx, ny))

    apply_temperature_boundary_conditions(
        temperature,
        mesh,
        inlet_temperature,
        DOMAIN_LENGTH_Y,
    )

    # Calculate R(T) directly using the existing RHS function
    rhs_field = np.zeros((nx, ny))

    for j in range(1, ny + 1):
        rhs_line = build_rhs(
            cell_counts=mesh,
            y_index=j,
            temperature=temperature,
            peclet_number=PECLET_NUMBER,
            brinkman_number=BRINKMAN_NUMBER,
            u_velocity=u_velocity,
            v_velocity=v_velocity,
            domain_length_x=DOMAIN_LENGTH_X,
            domain_length_y=DOMAIN_LENGTH_Y,
            convection_coefficient=convection_coefficient,
        )
        rhs_field[:, j - 1] = rhs_line[1:nx + 1]

    R_direct = rhs_field.ravel(order="C")

    # Calculate R(T) using the matrix representation
    T_vector = temperature[1:-1, 1:-1].ravel(order="C")
    R_matrix = A @ T_vector + b

    # Compare both calculations
    absolute_error = np.linalg.norm(R_direct - R_matrix)
    relative_error = absolute_error / max(
        np.linalg.norm(R_direct), np.linalg.norm(R_matrix), 1e-14
    )

    print(f"Absolute error: {absolute_error:.6e}")
    print(f"Relative error: {relative_error:.6e}")

    np.testing.assert_allclose(
        R_direct,
        R_matrix,
        rtol=1e-10,
        atol=1e-12,
        err_msg="Matrix assembly does not reproduce the direct spatial RHS.",
    )

    print("PASS: R(T) = A @ T + b")


def run_direct_implicit_step_verification():
    # Grid and numerical settings
    nx, ny = 24, 12
    mesh = (nx, ny)
    convection_coefficient = 1.0
    number_of_cells = nx * ny

    # Velocity fields
    u_velocity, v_velocity = initialize_velocity_fields(
        mesh,
        DOMAIN_LENGTH_Y,
        MEAN_VELOCITY,
    )

    # Assemble spatial operator A
    A = build_spatial_matrix(
        nx=nx,
        ny=ny,
        peclet_number=PECLET_NUMBER,
        length_x=DOMAIN_LENGTH_X,
        length_y=DOMAIN_LENGTH_Y,
        u_velocity=u_velocity,
        v_velocity=v_velocity,
        convection_coefficient=convection_coefficient,
    )

    # Assemble constant contribution b
    b = build_forcing_field(
        mesh=mesh,
        inlet_temperature=inlet_temperature,
        peclet_number=PECLET_NUMBER,
        brinkman_number=BRINKMAN_NUMBER,
        u_velocity=u_velocity,
        v_velocity=v_velocity,
        domain_length_x=DOMAIN_LENGTH_X,
        domain_length_y=DOMAIN_LENGTH_Y,
        convection_coefficient=convection_coefficient,
    ).ravel(order="C")

    # Choose a nonzero old temperature field
    rng = np.random.default_rng(42)
    T_old = rng.random(number_of_cells)

    # Backward Euler system:
    # (I - dt*A) T_new = T_old + dt*b
    M = eye(number_of_cells, format="csc") - TIME_STEP * A.tocsc()
    rhs = T_old + TIME_STEP * b

    # Solve the linear system
    factorized_matrix = splu(M)
    T_new = factorized_matrix.solve(rhs)

    # Evaluate the linear-system residual
    residual = M @ T_new - rhs

    absolute_error = np.linalg.norm(residual)
    relative_error = absolute_error / max(np.linalg.norm(rhs), 1e-14)

    print(f"Absolute residual: {absolute_error:.6e}")
    print(f"Relative residual: {relative_error:.6e}")

    np.testing.assert_allclose(
        M @ T_new,
        rhs,
        rtol=1e-10,
        atol=1e-12,
        err_msg="The backward Euler linear system was not solved accurately.",
    )

    print("PASS: The backward Euler linear system is solved accurately.")


def extract_physical_temperature(result, mesh):
    """Extract the physical temperature field from the solver result."""
    nx, ny = mesh

    # The solver returns a 7-element tuple, with temperature first.
    temperature = np.asarray(result[0], dtype=float)

    if temperature.shape == (nx + 2, ny + 2):
        return temperature[1:-1, 1:-1]

    elif temperature.shape == (nx, ny):
        return temperature

    else:
        raise ValueError(
        f"Unexpected temperature shape: {temperature.shape}"
    )


def run_verify_direct_vs_afi():
    # Grid and common numerical settings
    mesh = (24, 12)
    convection_coefficient = 1.0
    snapshot_interval = 10

    # Generate identical initial and velocity fields
    initial_temperature = initialize_temperature(
        mesh, DOMAIN_LENGTH_Y
    )

    u_velocity, v_velocity = initialize_velocity_fields(
        mesh, DOMAIN_LENGTH_Y, MEAN_VELOCITY
    )

    # Common arguments for both solvers
    common_arguments = {
        "time_step": TIME_STEP,
        "mesh": mesh,
        "inlet_temperature": inlet_temperature,
        "peclet_number": PECLET_NUMBER,
        "brinkman_number": BRINKMAN_NUMBER,
        "u_velocity": u_velocity,
        "v_velocity": v_velocity,
        "domain_length_x": DOMAIN_LENGTH_X,
        "domain_length_y": DOMAIN_LENGTH_Y,
        "mean_velocity": MEAN_VELOCITY,
        "residual_tolerance": RESIDUAL_TOLERANCE,
        "max_iterations": MAX_ITERATIONS,
        "snapshot_interval": snapshot_interval,
        "convection_coefficient": convection_coefficient,
    }

    # Run AFI
    print("Running AFI solver...")
    afi_result = solve_with_apf(
        initial_temperature=initial_temperature.copy(),
        **common_arguments,
    )

    # Run direct solver
    print("Running direct solver...")
    direct_result = solve_with_direct(
        initial_temperature=initial_temperature.copy(),
        **common_arguments,
    )

    # Extract results
    T_afi = np.asarray(afi_result[0])[1:-1, 1:-1]
    T_direct = np.asarray(direct_result[0])[1:-1, 1:-1]

    afi_residual_history = afi_result[2]
    direct_residual_history = direct_result[2]

    # Compare final temperature fields
    difference = T_direct - T_afi

    absolute_error = np.linalg.norm(difference)
    relative_error = absolute_error / max(
        np.linalg.norm(T_afi), 1e-14
    )
    maximum_error = np.max(np.abs(difference))

    print("\n--- Direct vs AFI ---")
    print(f"AFI iterations: {afi_result[1][-1] if len(afi_result[1]) else 'N/A'}")
    print(
        "Direct iterations: "
        f"{direct_result[1][-1] if len(direct_result[1]) else 'N/A'}"
    )
    print(
        "AFI final residual: "
        f"{afi_residual_history[-1] if len(afi_residual_history) else 'N/A'}"
    )
    print(
        "Direct final residual: "
        f"{direct_residual_history[-1] if len(direct_residual_history) else 'N/A'}"
    )
    print(f"Relative L2 difference: {relative_error:.6e}")
    print(f"Maximum absolute difference: {maximum_error:.6e}")



def run_direct_time_step_test(
    time_step,
    mesh=(25, 10),
    inlet_temperature=None,
    residual_tolerance=None,
    max_iterations=None,
    reference_temperature=None,
    convection_coefficient=1.0,
):
    """
    Test the direct solver at a specified time step.

    Returns a dictionary containing convergence, computational-cost,
    residual, temperature-field, and optional reference-comparison data.
    """

    from time import perf_counter

    import numpy as np

    from src.config import (
        DOMAIN_LENGTH_X,
        DOMAIN_LENGTH_Y,
        PECLET_NUMBER,
        BRINKMAN_NUMBER,
        MEAN_VELOCITY,
        RESIDUAL_TOLERANCE,
        MAX_ITERATIONS,
    )
    from src.initialization import (
        initialize_temperature,
        initialize_velocity_fields,
    )
    from src.diagnostics import steady_state_residual_rms
    from src.solvers.direct import solve_with_direct

    # ---------------------------------------------------------
    # 1. Validate inputs and set defaults
    # ---------------------------------------------------------
    if time_step <= 0:
        raise ValueError("time_step must be positive.")

    if residual_tolerance is None:
        residual_tolerance = RESIDUAL_TOLERANCE

    if max_iterations is None:
        max_iterations = MAX_ITERATIONS

    if residual_tolerance <= 0:
        raise ValueError(
            "residual_tolerance must be positive for this test."
        )

    if max_iterations < 1:
        raise ValueError("max_iterations must be positive.")

    if inlet_temperature is None:
        inlet_temperature = lambda y: y

    nx, ny = mesh

    # ---------------------------------------------------------
    # 2. Initialize the problem
    # ---------------------------------------------------------
    initial_temperature = initialize_temperature(
        mesh=mesh,
        domain_length_y=DOMAIN_LENGTH_Y,
    )

    u_velocity, v_velocity = initialize_velocity_fields(
        mesh=mesh,
        domain_length_y=DOMAIN_LENGTH_Y,
        mean_velocity=MEAN_VELOCITY,
    )

    # ---------------------------------------------------------
    # 3. Run the direct solver
    # ---------------------------------------------------------
    print("\n" + "=" * 72)
    print("DIRECT SOLVER: TIME-STEP TEST")
    print("=" * 72)
    print(f"Mesh:               {nx} x {ny}")
    print(f"Time step:          {time_step:g}")
    print(f"Pe:                 {PECLET_NUMBER:g}")
    print(f"Br:                 {BRINKMAN_NUMBER:g}")
    print(f"Mean velocity:      {MEAN_VELOCITY:g}")
    print(f"Residual tolerance: {residual_tolerance:.3e}")
    print(f"Maximum iterations: {max_iterations}")
    print("-" * 72)

    start_time = perf_counter()

    try:
        result = solve_with_direct(
            initial_temperature=initial_temperature,
            time_step=time_step,
            mesh=mesh,
            inlet_temperature=inlet_temperature,
            peclet_number=PECLET_NUMBER,
            brinkman_number=BRINKMAN_NUMBER,
            u_velocity=u_velocity,
            v_velocity=v_velocity,
            domain_length_x=DOMAIN_LENGTH_X,
            domain_length_y=DOMAIN_LENGTH_Y,
            mean_velocity=MEAN_VELOCITY,
            residual_tolerance=residual_tolerance,
            max_iterations=max_iterations,
            snapshot_interval=max_iterations + 1,
            convection_coefficient=convection_coefficient,
        )

    except FloatingPointError as exc:
        elapsed_time = perf_counter() - start_time

        result = {
            "time_step": float(time_step),
            "mesh": mesh,
            "converged": False,
            "status": "numerical_failure",
            "error": str(exc),
            "wall_time": float(elapsed_time),
        }

        print("Status:             NUMERICAL FAILURE")
        print(f"Error:              {exc}")
        print(f"Elapsed time:       {elapsed_time:.6f} s")
        print("=" * 72)

        return result

    (
        temperature,
        iteration_history,
        residual_history,
        computation_time,
        temperature_history,
        snapshot_iterations,
    ) = result

    temperature = np.asarray(temperature)
    residual_history = np.asarray(residual_history, dtype=float)

    # ---------------------------------------------------------
    # 4. Independently evaluate the final steady-state residual
    # ---------------------------------------------------------
    final_residual = steady_state_residual_rms(
        mesh=mesh,
        temperature=temperature,
        peclet_number=PECLET_NUMBER,
        brinkman_number=BRINKMAN_NUMBER,
        u_velocity=u_velocity,
        v_velocity=v_velocity,
        domain_length_x=DOMAIN_LENGTH_X,
        domain_length_y=DOMAIN_LENGTH_Y,
        convection_coefficient=convection_coefficient,
    )

    physical_temperature = temperature[1:-1, 1:-1]
    all_finite = bool(np.all(np.isfinite(temperature)))

    converged = bool(
        all_finite
        and np.isfinite(final_residual)
        and final_residual <= residual_tolerance
    )

    iterations = len(iteration_history)

    # Residual trend: useful diagnostic, not a formal stability test.
    if len(residual_history) >= 2:
        residual_nonincreasing = bool(
            np.all(
                np.diff(residual_history)
                <= 1e-12 * np.maximum(1.0, residual_history[:-1])
            )
        )
        last_residual_ratio = float(
            residual_history[-1] / residual_history[-2]
        )
    else:
        residual_nonincreasing = None
        last_residual_ratio = None

    # ---------------------------------------------------------
    # 5. Optional comparison with a reference steady solution
    # ---------------------------------------------------------
    relative_reference_difference = None

    if reference_temperature is not None:
        reference = np.asarray(reference_temperature)

        if reference.shape == temperature.shape:
            reference = reference[1:-1, 1:-1]
        elif reference.shape != physical_temperature.shape:
            raise ValueError(
                "reference_temperature must have either the full "
                "temperature-array shape or the physical-cell shape."
            )

        difference_rms = float(
            np.sqrt(np.mean((physical_temperature - reference) ** 2))
        )
        reference_rms = float(np.sqrt(np.mean(reference ** 2)))

        relative_reference_difference = (
            difference_rms / reference_rms
            if reference_rms > 0
            else difference_rms
        )

    # ---------------------------------------------------------
    # 6. Collect and print results
    # ---------------------------------------------------------

    output = {
        "time_step": float(time_step),
        "mesh": tuple(mesh),
        "converged": converged,
        "status": "converged" if converged else "not_converged",
        "iterations": iterations,
        "max_iterations": max_iterations,
        "final_residual": float(final_residual),
        "residual_tolerance": float(residual_tolerance),
        "residual_history": residual_history.copy(),
        "residual_nonincreasing": residual_nonincreasing,
        "last_residual_ratio": last_residual_ratio,
        "computation_time": float(computation_time),
        "time_per_iteration": (
            float(computation_time / iterations)
            if iterations > 0 else None
        ),
        "temperature": temperature.copy(),
        "temperature_min": float(np.min(physical_temperature)),
        "temperature_max": float(np.max(physical_temperature)),
        "all_finite": all_finite,
        "temperature_history": temperature_history,
        "snapshot_iterations": snapshot_iterations,
        "relative_reference_difference": relative_reference_difference,
    }

    print(f"Status:             {output['status'].upper()}")
    print(f"Iterations:         {iterations}")
    print(f"Final residual:     {final_residual:.6e}")
    print(f"Computation time:   {computation_time:.6f} s")
    print(f"Time per iteration: {output['time_per_iteration']}")
    print(
        f"Temperature range:  "
        f"[{output['temperature_min']:.6g}, "
        f"{output['temperature_max']:.6g}]"
    )
    print(f"All values finite:  {all_finite}")
    print(f"Residual decreasing: {residual_nonincreasing}")

    if relative_reference_difference is not None:
        print(
            "Relative difference from reference: "
            f"{relative_reference_difference:.6e}"
        )

    print("=" * 72)

    return output