import numpy as np


# =============================================================================
# Temperature Initialization
# =============================================================================

def initialize_temperature(N):
    # N: number of control volumes in each direction

    # Initialize temperature field including one layer of ghost cells
    T = np.zeros((N + 2, N + 2))

    # Generate x-coordinates of cell centers
    x = np.linspace(1/(2*N), 1 - 1/(2*N), N)

    # Set ghost-cell values for the top Dirichlet boundary
    T[1:N+1, N+1] = 2 * np.sin(np.pi * x)

    return T