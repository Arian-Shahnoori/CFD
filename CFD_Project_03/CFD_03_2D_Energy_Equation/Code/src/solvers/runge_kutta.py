import time
import warnings

import numpy as np

from src.solvers.apf import build_rhs
from src.diagnostics import (
    steady_state_residual_rms,
)
from src.boundary_conditions import apply_temperature_boundary_conditions


# ---------------------------------------------------------------------
# Explicit Runge-Kutta coefficients
# ---------------------------------------------------------------------

# Explicit midpoint RK2:
# T_stage = T_n + dt * (1/2) * K1
# T_new   = T_n + dt * K2

RK2_A = (
    (),
    (0.5,),
)

RK2_B = (
    0.0,
    1.0,
)


# Classical four-stage RK4:
#
# K1 = F(T_n)
# K2 = F(T_n + dt*K1/2)
# K3 = F(T_n + dt*K2/2)
# K4 = F(T_n + dt*K3)
#
# T_new = T_n + dt*(K1/6 + K2/3 + K3/3 + K4/6)

RK4_A = (
    (),
    (0.5,),
    (0.0, 0.5),
    (0.0, 0.0, 1.0),
)

RK4_B = (
    1.0 / 6.0,
    1.0 / 3.0,
    1.0 / 3.0,
    1.0 / 6.0,
)


# ---------------------------------------------------------------------
# Spatial RHS evaluation
# ---------------------------------------------------------------------

def evaluate_rhs(
    temperature,
    mesh,
    peclet_number,
    brinkman_number,
    u_velocity,
    v_velocity,
    domain_length_x,
    domain_length_y,
    convection_coefficient=1.0,
):
    """
    Evaluate dT/dt = F(T) over the physical cells.

    The spatial discretization and viscous-heating source are taken
    directly from the existing AFI solver's build_rhs() function.

    Returns an array with the same shape as temperature. The RHS is
    populated in the physical interior; ghost-cell RHS values are zero.
    """

    cell_count_x, cell_count_y = mesh

    rhs_matrix = np.zeros_like(temperature, dtype=float)

    for y_index in range(1, cell_count_y + 1):
        rhs_matrix[:, y_index] = build_rhs(
            cell_counts=mesh,
            y_index=y_index,
            temperature=temperature,
            peclet_number=peclet_number,
            brinkman_number=brinkman_number,
            u_velocity=u_velocity,
            v_velocity=v_velocity,
            domain_length_x=domain_length_x,
            domain_length_y=domain_length_y,
            convection_coefficient=convection_coefficient,
        )

    return rhs_matrix


# ---------------------------------------------------------------------
# Shared explicit Runge-Kutta solver
# ---------------------------------------------------------------------

def _solve_with_explicit_rk(
    initial_temperature,
    time_step,
    mesh,
    inlet_temperature,
    peclet_number,
    brinkman_number,
    u_velocity,
    v_velocity,
    domain_length_x,
    domain_length_y,
    rk_a,
    rk_b,
    method_name,
    mean_velocity=3.0,
    residual_tolerance=1e-12,
    max_iterations=100000,
    snapshot_interval=10,
    convection_coefficient=1.0,
):
    """
    Advance the energy equation with an explicit Runge-Kutta method
    until the steady-state residual reaches the prescribed tolerance.

    rk_a and rk_b define the explicit Runge-Kutta method.

    The solver returns the same seven-element tuple as solve_with_afi().
    """

    if time_step <= 0.0:
        raise ValueError("time_step must be positive.")

    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1.")

    if snapshot_interval < 1:
        raise ValueError("snapshot_interval must be at least 1.")

    if len(rk_a) != len(rk_b):
        raise ValueError("rk_a and rk_b must define the same number of stages.")

    expected_shape = (mesh[0] + 2, mesh[1] + 2)

    if initial_temperature.shape != expected_shape:
        raise ValueError(
            f"initial_temperature must have shape {expected_shape}; "
            f"received {initial_temperature.shape}."
        )

    temperature = initial_temperature.copy()

    # Ensure the starting field has consistent ghost-cell values.
    temperature = apply_temperature_boundary_conditions(
        temperature=temperature,
        mesh=mesh,
        inlet_temperature=inlet_temperature,
        domain_length_y=domain_length_y,
    )

    iteration_history = []
    residual_history = []
    time_history = []

    # Store the initial temperature field.
    temperature_history = [temperature.copy()]
    snapshot_iterations = [0]

    physical_slice = np.s_[1:-1, 1:-1]

    start_time = time.perf_counter()
    converged = False

    for iteration in range(1, max_iterations + 1):

        # -------------------------------------------------------------
        # Runge-Kutta stages
        # -------------------------------------------------------------

        stage_rhs_list = []

        for stage_index, a_row in enumerate(rk_a):

            # Start each stage from the current time-level solution.
            stage_temperature = temperature.copy()

            # Construct the intermediate stage using the preceding RHS
            # evaluations. Only physical cells are advanced explicitly.
            if stage_index > 0:
                stage_increment = np.zeros_like(temperature)

                for coefficient, previous_rhs in zip(
                    a_row,
                    stage_rhs_list,
                ):
                    stage_increment[physical_slice] += (
                        coefficient * previous_rhs[physical_slice]
                    )

                stage_temperature[physical_slice] = (
                    temperature[physical_slice]
                    + time_step * stage_increment[physical_slice]
                )

            # Ghost cells must match the boundary conditions at every
            # stage, before evaluating the spatial RHS.
            stage_temperature = apply_temperature_boundary_conditions(
                temperature=stage_temperature,
                mesh=mesh,
                inlet_temperature=inlet_temperature,
                domain_length_y=domain_length_y,
            )

            stage_rhs = evaluate_rhs(
                temperature=stage_temperature,
                mesh=mesh,
                peclet_number=peclet_number,
                brinkman_number=brinkman_number,
                u_velocity=u_velocity,
                v_velocity=v_velocity,
                domain_length_x=domain_length_x,
                domain_length_y=domain_length_y,
                convection_coefficient=convection_coefficient,
            )

            stage_rhs_list.append(stage_rhs)

        # -------------------------------------------------------------
        # Combine the stages to obtain T^(n+1)
        # -------------------------------------------------------------

        total_increment = np.zeros_like(temperature)

        for weight, stage_rhs in zip(rk_b, stage_rhs_list):
            total_increment[physical_slice] += (
                weight * stage_rhs[physical_slice]
            )

        temperature[physical_slice] += (
            time_step * total_increment[physical_slice]
        )

        # Ensure the final solution has consistent ghost cells.
        temperature = apply_temperature_boundary_conditions(
            temperature=temperature,
            mesh=mesh,
            inlet_temperature=inlet_temperature,
            domain_length_y=domain_length_y,
        )

        # -------------------------------------------------------------
        # Convergence diagnostics
        # -------------------------------------------------------------

        residual = steady_state_residual_rms(
            mesh=mesh,
            temperature=temperature,
            peclet_number=peclet_number,
            brinkman_number=brinkman_number,
            u_velocity=u_velocity,
            v_velocity=v_velocity,
            domain_length_x=domain_length_x,
            domain_length_y=domain_length_y,
            convection_coefficient=convection_coefficient,
        )

        iteration_history.append(iteration)
        residual_history.append(residual)

        # Cumulative elapsed time at this iteration.
        time_history.append(time.perf_counter() - start_time)        

        # Save snapshots at the requested interval.
        if iteration % snapshot_interval == 0:
            temperature_history.append(temperature.copy())
            snapshot_iterations.append(iteration)

        if residual <= residual_tolerance:
            converged = True
            break

    computation_time = time.perf_counter() - start_time

    if not converged:
        warnings.warn(
            f"{method_name} did not reach the residual tolerance "
            f"after {max_iterations} iterations. "
            f"Final residual: {residual_history[-1]:.3e}",
            RuntimeWarning,
        )

    # Always include the final temperature field.
    if snapshot_iterations[-1] != iteration:
        temperature_history.append(temperature.copy())
        snapshot_iterations.append(iteration)

    return (
        temperature,
        iteration_history,
        residual_history,
        time_history,
        computation_time,
        temperature_history,
        snapshot_iterations,
    )


# ---------------------------------------------------------------------
# Public solver: explicit midpoint RK2
# ---------------------------------------------------------------------

def solve_with_rk2(
    initial_temperature,
    time_step,
    mesh,
    inlet_temperature,
    peclet_number,
    brinkman_number,
    u_velocity,
    v_velocity,
    domain_length_x,
    domain_length_y,
    mean_velocity=3.0,
    residual_tolerance=1e-12,
    max_iterations=100000,
    snapshot_interval=10,
    convection_coefficient=1.0,
):
    """Advance the energy equation using explicit midpoint RK2."""

    return _solve_with_explicit_rk(
        initial_temperature=initial_temperature,
        time_step=time_step,
        mesh=mesh,
        inlet_temperature=inlet_temperature,
        peclet_number=peclet_number,
        brinkman_number=brinkman_number,
        u_velocity=u_velocity,
        v_velocity=v_velocity,
        domain_length_x=domain_length_x,
        domain_length_y=domain_length_y,
        rk_a=RK2_A,
        rk_b=RK2_B,
        method_name="RK2",
        mean_velocity=mean_velocity,
        residual_tolerance=residual_tolerance,
        max_iterations=max_iterations,
        snapshot_interval=snapshot_interval,
        convection_coefficient=convection_coefficient,
    )


# ---------------------------------------------------------------------
# Public solver: classical RK4
# ---------------------------------------------------------------------

def solve_with_rk4(
    initial_temperature,
    time_step,
    mesh,
    inlet_temperature,
    peclet_number,
    brinkman_number,
    u_velocity,
    v_velocity,
    domain_length_x,
    domain_length_y,
    mean_velocity=3.0,
    residual_tolerance=1e-12,
    max_iterations=100000,
    snapshot_interval=10,
    convection_coefficient=1.0,
):
    """Advance the energy equation using classical explicit RK4."""

    return _solve_with_explicit_rk(
        initial_temperature=initial_temperature,
        time_step=time_step,
        mesh=mesh,
        inlet_temperature=inlet_temperature,
        peclet_number=peclet_number,
        brinkman_number=brinkman_number,
        u_velocity=u_velocity,
        v_velocity=v_velocity,
        domain_length_x=domain_length_x,
        domain_length_y=domain_length_y,
        rk_a=RK4_A,
        rk_b=RK4_B,
        method_name="RK4",
        mean_velocity=mean_velocity,
        residual_tolerance=residual_tolerance,
        max_iterations=max_iterations,
        snapshot_interval=snapshot_interval,
        convection_coefficient=convection_coefficient,
    )