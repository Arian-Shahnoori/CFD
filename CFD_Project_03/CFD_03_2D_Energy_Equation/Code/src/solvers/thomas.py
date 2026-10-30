import numpy as np


def solve_tridiagonal(
    lower_diagonal,
    main_diagonal,
    upper_diagonal,
    rhs,
):
    """Solve a tridiagonal linear system using the Thomas algorithm."""

    lower = np.asarray(lower_diagonal, dtype=float).copy()
    diagonal = np.asarray(main_diagonal, dtype=float).copy()
    upper = np.asarray(upper_diagonal, dtype=float).copy()
    rhs = np.asarray(rhs, dtype=float).copy()

    size = len(diagonal)

    if rhs.size != size:
        raise ValueError("The RHS and diagonal must have the same length.")
    if lower.size != size - 1 or upper.size != size - 1:
        raise ValueError("The off-diagonals must have length size - 1.")
    if size == 0:
        raise ValueError("The system cannot be empty.")

    modified_upper = np.zeros(max(size - 1, 0), dtype=float)
    modified_rhs = np.zeros(size, dtype=float)

    if abs(diagonal[0]) < 1e-14:
        raise np.linalg.LinAlgError("Zero pivot in Thomas algorithm.")

    if size > 1:
        modified_upper[0] = upper[0] / diagonal[0]

    modified_rhs[0] = rhs[0] / diagonal[0]

    # Forward elimination.
    for i in range(1, size):
        pivot = diagonal[i] - lower[i - 1] * modified_upper[i - 1]

        if abs(pivot) < 1e-14:
            raise np.linalg.LinAlgError("Zero pivot in Thomas algorithm.")

        if i < size - 1:
            modified_upper[i] = upper[i] / pivot

        modified_rhs[i] = (
            rhs[i] - lower[i - 1] * modified_rhs[i - 1]
        ) / pivot

    # Back substitution.
    solution = np.zeros(size, dtype=float)
    solution[-1] = modified_rhs[-1]

    for i in range(size - 2, -1, -1):
        solution[i] = (
            modified_rhs[i] - modified_upper[i] * solution[i + 1]
        )

    return solution


def solve_tridiagonal_matrix(matrix, rhs):
    """Solve a tridiagonal system supplied as a matrix."""

    matrix = np.asarray(matrix, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("The coefficient matrix must be square.")

    return solve_tridiagonal(
        lower_diagonal=np.diag(matrix, k=-1),
        main_diagonal=np.diag(matrix),
        upper_diagonal=np.diag(matrix, k=1),
        rhs=rhs,
    )