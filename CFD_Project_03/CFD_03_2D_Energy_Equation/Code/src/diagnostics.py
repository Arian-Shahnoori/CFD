import numpy as np

from src.solvers import apf


def l2_error(reference, numerical):
    """Calculate the root-mean-square difference between two arrays."""
    reference = np.asarray(reference, dtype=float)
    numerical = np.asarray(numerical, dtype=float)

    if reference.shape != numerical.shape:
        raise ValueError("Reference and numerical arrays must have the same shape.")

    return float(np.sqrt(np.mean((numerical - reference) ** 2)))


def steady_state_residual_rms(
    mesh,
    temperature,
    peclet_number,
    brinkman_number,
    u_velocity,
    v_velocity,
    domain_length_x,
    domain_length_y,
    convection_coefficient: float = 1.0,
):
    """Calculate the RMS of the discrete steady energy-equation residual."""
    cell_count_x, cell_count_y = mesh
    rhs_matrix = np.zeros_like(temperature, dtype=float)

    for y_index in range(1, cell_count_y + 1):
        rhs_matrix[:, y_index] = apf.build_rhs(
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

    residual = rhs_matrix[1:-1, 1:-1]

    return float(np.sqrt(np.mean(residual**2)))


