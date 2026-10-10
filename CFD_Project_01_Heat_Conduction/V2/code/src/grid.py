import numpy as np


# =============================================================================
# Grid Generation
# =============================================================================

def create_grid(N):
    # N: number of control volumes in each direction

    # Generate x-coordinates of cell centers
    x = np.linspace(1/(2*N), 1 - 1/(2*N), N)

    # Generate 2D coordinate matrices using matrix-compatible (ij) indexing
    X, Y = np.meshgrid(x, x, indexing='ij')

    return x, X, Y