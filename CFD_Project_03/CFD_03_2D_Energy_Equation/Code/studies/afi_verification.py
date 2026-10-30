import os

from src import config
from src.initialization import (
    inlet_temperature,
    initialize_temperature,
    initialize_velocity_fields,
)
from src.boundary_conditions import apply_temperature_boundary_conditions
from src.solvers.apf import solve_with_apf
from src.visualization import (
    plot_temperature,
    plot_convergence,
    plot_outlet_error,
)


def run_afi_verfication():

    mesh = (
        config.CELL_COUNT_X,
        config.CELL_COUNT_Y,
    )

    os.makedirs(config.OUTPUT_DIRECTORY, exist_ok=True)

    # Initialize temperature and velocity fields.
    temperature_initial = initialize_temperature(
        mesh=mesh,
        domain_length_y=config.DOMAIN_LENGTH_Y,
    )

    u_velocity, v_velocity = initialize_velocity_fields(
        mesh=mesh,
        domain_length_y=config.DOMAIN_LENGTH_Y,
        mean_velocity=config.MEAN_VELOCITY,
    )

    # Enforce physical boundary conditions on the initial field.
    temperature_initial = apply_temperature_boundary_conditions(
        temperature=temperature_initial,
        mesh=mesh,
        inlet_temperature=inlet_temperature,
        domain_length_y=config.DOMAIN_LENGTH_Y,
    )

    # Solve using AFI.
    (
        temperature,
        iteration_history,
        residual_history,
        computation_time,
        temperature_history,
        snapshot_iterations,
    ) = solve_with_apf(
        initial_temperature=temperature_initial,
        time_step=config.TIME_STEP,
        mesh=mesh,
        inlet_temperature=inlet_temperature,
        peclet_number=config.PECLET_NUMBER,
        brinkman_number=config.BRINKMAN_NUMBER,
        u_velocity=u_velocity,
        v_velocity=v_velocity,
        domain_length_x=config.DOMAIN_LENGTH_X,
        domain_length_y=config.DOMAIN_LENGTH_Y,
        mean_velocity=config.MEAN_VELOCITY,
        residual_tolerance=config.RESIDUAL_TOLERANCE,
        max_iterations=config.MAX_ITERATIONS,
        snapshot_interval=1,
        convection_coefficient=config.convection_coefficient,
    )

    print(f"Time steps: {len(iteration_history)}")
    print(f"Computation time: {computation_time:.2f} s")

    if residual_history:
        print(f"Final residual: {residual_history[-1]:.6e}")

    # Visualize results.
    plot_temperature(
    temperature=temperature,
    mesh=mesh,
    domain_length_x=config.DOMAIN_LENGTH_X,
    domain_length_y=config.DOMAIN_LENGTH_Y,
    reynolds_number=config.REYNOLDS_NUMBER,
    prandtl_number=config.PRANDTL_NUMBER,
    eckert_number=config.ECKERT_NUMBER,
    peclet_number=config.PECLET_NUMBER,
    brinkman_number=config.BRINKMAN_NUMBER,
)

    plot_convergence(
        iteration_history=iteration_history,
        residual_history=residual_history,
        output_directory=config.OUTPUT_DIRECTORY,
    )
