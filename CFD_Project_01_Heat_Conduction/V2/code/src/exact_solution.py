import numpy as np


# =============================================================================
# Exact Solution
# =============================================================================

def exact_solution(X, Y):
    # X, Y: 2D coordinate matrices

    return (
        np.sin(np.pi * X)
        * np.sinh(np.pi * Y)
        / np.sinh(np.pi)
    )