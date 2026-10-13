import numpy as np


def exact_solution(
    x: np.ndarray,
    t: float,
    u: float = 2.0,
) -> np.ndarray:
    """
    Analytical solution for the prescribed inlet and zero initial field.
    """
    if u <= 0.0:
        raise ValueError("This solution assumes u > 0.")

    x = np.asarray(x, dtype=float)

    return np.where(
        x <= u * t,
        np.sin(2.0 * np.pi * (t - x / u)),
        0.0,
    )