import numpy as np


# =============================================================================
# Boundary Conditions
# =============================================================================

def apply_boundary_conditions(T, N):
    # N: number of control volumes in each direction
    # T: temperature field including ghost cells

    x = np.linspace(1/(2*N), 1 - 1/(2*N), N)

    # Left / right boundaries: T = 0 (Dirichlet)
    T[0, 1:N+1] = -T[1, 1:N+1]
    T[N+1, 1:N+1] = -T[N, 1:N+1]

    # Bottom boundary: T = 0 (Dirichlet)
    T[1:N+1, 0] = -T[1:N+1, 1]

    # Top boundary: T = 2 sin(pi x) (Dirichlet)
    T[1:N+1, N+1] = (
        2 * np.sin(np.pi * x)
        - T[1:N+1, N]
    )