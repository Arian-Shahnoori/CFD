import numpy as np


def rms_l2_error(
    numerical: np.ndarray,
    exact: np.ndarray,
) -> float:
    """Calculate the normalized discrete L2 error."""
    numerical = np.asarray(numerical, dtype=float)
    exact = np.asarray(exact, dtype=float)

    if numerical.shape != exact.shape:
        raise ValueError("Numerical and exact arrays must have equal shapes.")

    return float(np.sqrt(np.mean((numerical - exact) ** 2)))


def observed_order(
    cell_counts: list[int],
    errors: list[float],
) -> float:
    """
    Estimate convergence order from a log-log fit of error versus cell count.
    """
    n_cells = np.asarray(cell_counts, dtype=float)
    errors = np.asarray(errors, dtype=float)

    if len(n_cells) != len(errors) or len(n_cells) < 2:
        raise ValueError("Provide at least two matching grid/error values.")
    if np.any(n_cells <= 0) or np.any(errors <= 0):
        raise ValueError("Cell counts and errors must be positive.")

    slope, _ = np.polyfit(np.log(n_cells), np.log(errors), 1)

    # Error ~ N_CV^(-p), so the positive convergence order is -slope.
    return float(-slope)