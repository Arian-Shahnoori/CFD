import numpy as np
from ..boundary_conditions import apply_temperature_boundary_conditions
from .thomas import solve_tridiagonal_matrix
import time
import warnings

from src.diagnostics import (
    l2_error,
    steady_state_residual_rms,
)

# RHS vector construction for the energy equation
def build_rhs(
    cell_counts, y_index, temperature,
    peclet_number, brinkman_number,
    u_velocity, v_velocity, domain_length_x, domain_length_y,
    convection_coefficient: float = 1.0
):
    cell_count_x, cell_count_y = cell_counts

    dx = domain_length_x / cell_count_x
    dy = domain_length_y / cell_count_y

    rhs = np.zeros(cell_count_x + 2, dtype=float)

    for x_index in range(1, cell_count_x + 1):

        # Advection and diffusion in the x-direction
        x_flux_difference = (
            u_velocity[x_index + 1, y_index]
            * temperature[x_index + 1, y_index]
            - u_velocity[x_index - 1, y_index]
            * temperature[x_index - 1, y_index]
        )

        x_diffusion_difference = (
            temperature[x_index + 1, y_index]
            - 2 * temperature[x_index, y_index]
            + temperature[x_index - 1, y_index]
        )

        # Advection and diffusion in the y-direction
        y_flux_difference = (
            v_velocity[x_index, y_index + 1]
            * temperature[x_index, y_index + 1]
            - v_velocity[x_index, y_index - 1]
            * temperature[x_index, y_index - 1]
        )

        y_diffusion_difference = (
            temperature[x_index, y_index + 1]
            - 2 * temperature[x_index, y_index]
            + temperature[x_index, y_index - 1]
        )

        transport_rate = (
            -convection_coefficient * x_flux_difference / (2 * dx)
            + x_diffusion_difference / (peclet_number * dx**2)
            - convection_coefficient * y_flux_difference / (2 * dy)
            + y_diffusion_difference / (peclet_number * dy**2)
        )

        # Velocity gradients for viscous dissipation
        du_dx = (
            u_velocity[x_index + 1, y_index]
            - u_velocity[x_index - 1, y_index]
        ) / (2 * dx)

        dv_dy = (
            v_velocity[x_index, y_index + 1]
            - v_velocity[x_index, y_index - 1]
        ) / (2 * dy)

        dv_dx = (
            v_velocity[x_index + 1, y_index]
            - v_velocity[x_index - 1, y_index]
        ) / (2 * dx)

        du_dy = (
            u_velocity[x_index, y_index + 1]
            - u_velocity[x_index, y_index - 1]
        ) / (2 * dy)

        viscous_heating = (brinkman_number / peclet_number) * (
            2 * du_dx**2
            + 2 * dv_dy**2
            + (dv_dx + du_dy)**2
        )

        rhs[x_index] = transport_rate + viscous_heating

    return rhs


# Coefficient matrix for the x-direction solve
def build_x_matrix(
    cell_counts, y_index, time_step,
    peclet_number, u_velocity, domain_length_x,
    convection_coefficient: float = 1.0
):
    cell_count_x, cell_count_y = cell_counts
    dx = domain_length_x / cell_count_x

    coefficient_matrix = np.zeros(
        (cell_count_x + 2, cell_count_x + 2)
    )

    # Inlet: fixed-temperature correction
    coefficient_matrix[0, 0] = 1
    coefficient_matrix[0, 1] = 1

    # Outlet: zero-gradient correction
    coefficient_matrix[cell_count_x + 1, cell_count_x] = 1
    coefficient_matrix[cell_count_x + 1, cell_count_x + 1] = -1

    for x_index in range(1, cell_count_x + 1):
        coefficient_matrix[x_index, x_index] = (
            1 + 2 * time_step / (peclet_number * dx**2)
        )

        coefficient_matrix[x_index, x_index + 1] = (
            convection_coefficient
            * time_step * u_velocity[x_index + 1, y_index] / (2 * dx)
            - time_step / (peclet_number * dx**2)
        )

        coefficient_matrix[x_index, x_index - 1] = (
            -convection_coefficient
            * time_step * u_velocity[x_index - 1, y_index] / (2 * dx)
            - time_step / (peclet_number * dx**2)
        )
        

    return coefficient_matrix


# Coefficient matrix for the y-direction solve
def build_y_matrix(
    cell_counts, x_index, time_step,
    peclet_number, v_velocity, domain_length_y,
    convection_coefficient: float = 1.0
):
    cell_count_x, cell_count_y = cell_counts
    dy = domain_length_y / cell_count_y

    coefficient_matrix = np.zeros(
        (cell_count_y + 2, cell_count_y + 2)
    )

    # Bottom wall: fixed-temperature correction
    coefficient_matrix[0, 0] = 1
    coefficient_matrix[0, 1] = 1

    # Top wall: fixed-temperature correction
    coefficient_matrix[cell_count_y + 1, cell_count_y] = 1
    coefficient_matrix[cell_count_y + 1, cell_count_y + 1] = 1

    for y_index in range(1, cell_count_y + 1):
        coefficient_matrix[y_index, y_index] = (
            1 + 2 * time_step / (peclet_number * dy**2)
        )

        coefficient_matrix[y_index, y_index + 1] = (
            convection_coefficient
            * time_step * v_velocity[x_index, y_index + 1] / (2 * dy)
            - time_step / (peclet_number * dy**2)
        )

        coefficient_matrix[y_index, y_index - 1] = (
            -convection_coefficient
            * time_step * v_velocity[x_index, y_index - 1] / (2 * dy)
            - time_step / (peclet_number * dy**2)
        )

    return coefficient_matrix


def apf_step(
    temperature,
    mesh,
    inlet_temperature,
    time_step,
    peclet_number,
    brinkman_number,
    u_velocity,
    v_velocity,
    domain_length_x,
    domain_length_y,
    convection_coefficient: float = 1.0
):
    """
    Advance the temperature field by one approximate-factorization step.

    The temperature and velocity arrays include ghost cells and have
    shape (cell_count_x + 2, cell_count_y + 2).
    """
    cell_count_x, cell_count_y = mesh

    temperature = np.asarray(temperature, dtype=float).copy()

    expected_shape = (cell_count_x + 2, cell_count_y + 2)
    if temperature.shape != expected_shape:
        raise ValueError(
            f"Expected temperature shape {expected_shape}, "
            f"got {temperature.shape}."
        )

    # Enforce physical boundary conditions before evaluating the RHS.
    temperature = apply_temperature_boundary_conditions(
        temperature=temperature,
        mesh=mesh,
        inlet_temperature=inlet_temperature,
        domain_length_y=domain_length_y,
    )

    # ---------------------------------------------------------
    # Step 1: Solve in the x-direction.
    # A_x * delta_T_intermediate = time_step * RHS(T^n)
    # ---------------------------------------------------------
    intermediate_correction = np.zeros_like(temperature)

    for y_index in range(1, cell_count_y + 1):

        rhs = build_rhs(
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

        rhs = time_step * np.asarray(rhs, dtype=float)

        # Homogeneous correction boundary conditions.
        rhs[0] = 0.0
        rhs[-1] = 0.0

        x_matrix = build_x_matrix(
            cell_counts=mesh,
            y_index=y_index,
            time_step=time_step,
            peclet_number=peclet_number,
            u_velocity=u_velocity,
            domain_length_x=domain_length_x,
            convection_coefficient=convection_coefficient,
        )

        intermediate_correction[:, y_index] = solve_tridiagonal_matrix(
            x_matrix,
            rhs,
        )

    # ---------------------------------------------------------
    # Step 2: Solve in the y-direction.
    # A_y * delta_T = delta_T_intermediate
    # ---------------------------------------------------------
    temperature_correction = np.zeros_like(temperature)

    for x_index in range(1, cell_count_x + 1):

        y_matrix = build_y_matrix(
            cell_counts=mesh,
            x_index=x_index,
            time_step=time_step,
            peclet_number=peclet_number,
            v_velocity=v_velocity,
            domain_length_y=domain_length_y,
            convection_coefficient=convection_coefficient,
        )

        temperature_correction[x_index, :] = solve_tridiagonal_matrix(
            y_matrix,
            intermediate_correction[x_index, :],
        )

    # ---------------------------------------------------------
    # Step 3: Update temperature and reapply physical BCs.
    # ---------------------------------------------------------
    updated_temperature = temperature + temperature_correction

    updated_temperature = apply_temperature_boundary_conditions(
        temperature=updated_temperature,
        mesh=mesh,
        inlet_temperature=inlet_temperature,
        domain_length_y=domain_length_y,
    )

    return updated_temperature



def solve_with_apf(
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
    convection_coefficient=1,
):
    """Advance the energy equation with AFI until steady-state convergence."""

    temperature = initial_temperature.copy()

    iteration_history = []
    residual_history = []
    time_history = [] 

    # Store the initial temperature field.
    temperature_history = [temperature.copy()]
    snapshot_iterations = [0]



    start_time = time.perf_counter()
    converged = False

    for iteration in range(1, max_iterations + 1):

        temperature = apf_step(
            temperature=temperature,
            mesh=mesh,
            inlet_temperature=inlet_temperature,
            time_step=time_step,
            peclet_number=peclet_number,
            brinkman_number=brinkman_number,
            u_velocity=u_velocity,
            v_velocity=v_velocity,
            domain_length_x=domain_length_x,
            domain_length_y=domain_length_y,
            convection_coefficient=convection_coefficient,
        )

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

        # Store a snapshot every snapshot_interval iterations.
        if iteration % snapshot_interval == 0:
            temperature_history.append(temperature.copy())
            snapshot_iterations.append(iteration)
        

        iteration_history.append(iteration)
        residual_history.append(residual)

        # Cumulative elapsed time at this iteration.
        time_history.append(time.perf_counter() - start_time)

        if residual <= residual_tolerance:
            converged = True
            break

    computation_time = time.perf_counter() - start_time

    if not converged:
        warnings.warn(
            f"AFI did not reach the residual tolerance "
            f"after {max_iterations} iterations. "
            f"Final residual: {residual_history[-1]:.3e}"
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