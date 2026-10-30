from pathlib import Path

import numpy as np
from scipy.linalg import eigvals
from scipy.sparse import lil_matrix

from src.config import (
    DOMAIN_LENGTH_X,
    DOMAIN_LENGTH_Y,
    PECLET_NUMBER,
    MEAN_VELOCITY,
    TIME_STEP,
)


# ============================================================
# Configuration
# ============================================================

GRID_LEVELS = [
    #(25, 10),
    (50, 20),
    #(100, 40),
]

SCHEMES = (
    "implicit_euler",
    "rk2",
    "rk4",
)

STABILITY_TOLERANCE = 1e-10
BISECTION_ITERATIONS = 80
MAX_DOUBLINGS = 80

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIRECTORY = PROJECT_ROOT / "results"


# ============================================================
# 1. Construct the source-free spatial matrix
# ============================================================

def build_spatial_matrix(
    nx,
    ny,
    length_x=DOMAIN_LENGTH_X,
    length_y=DOMAIN_LENGTH_Y,
    peclet_number=PECLET_NUMBER,
    mean_velocity=MEAN_VELOCITY,
):
    """
    Assemble the spatial matrix A for

        dT/dt = A @ T + b

    using second-order central differences.

    The source term and prescribed boundary-temperature
    contributions are excluded because they are independent
    of the physical-cell temperatures.

    Physical temperature cells are ordered consistently with

        temperature[1:-1, 1:-1].ravel(order="C")

    so the wall-normal index varies fastest.
    """

    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be positive.")

    if length_x <= 0 or length_y <= 0:
        raise ValueError("Domain lengths must be positive.")

    if peclet_number <= 0:
        raise ValueError("Peclet number must be positive.")

    dx = length_x / nx
    dy = length_y / ny

    # Nondimensional thermal diffusivity
    kappa = 1.0 / peclet_number

    # Diffusive stencil coefficients
    diffusion_x = kappa / dx**2
    diffusion_y = kappa / dy**2

    # Centre coefficient for the five-point stencil
    diagonal = -2.0 * (diffusion_x + diffusion_y)

    number_of_cells = nx * ny

    A = lil_matrix(
        (number_of_cells, number_of_cells),
        dtype=float,
    )

    def index(i, j):
        """Map physical cell (i, j), using zero-based indices, to A."""
        return i * ny + j

    # Physical wall-normal cell-centre coordinates
    y_centres = (np.arange(ny) + 0.5) * dy

    for i in range(nx):
        for j, y in enumerate(y_centres):

            # Actual parabolic velocity at wall-normal location y
            u = (
                6.0
                * mean_velocity
                * y
                * (1.0 - y)
            )

            # Spatial stencil coefficients:
            #
            # aE = diffusion_x - u / (2 dx)
            # aW = diffusion_x + u / (2 dx)
            # aN = diffusion_y
            # aS = diffusion_y
            # aP = -2 (diffusion_x + diffusion_y)

            a_east = diffusion_x - u / (2.0 * dx)
            a_west = diffusion_x + u / (2.0 * dx)
            a_north = diffusion_y
            a_south = diffusion_y

            k = index(i, j)

            A[k, k] += diagonal

            # ------------------------------------------------
            # East boundary: zero streamwise gradient
            #
            # T_ghost = T_adjacent
            # The ghost-cell contribution adds +aE to the
            # diagonal coefficient.
            # ------------------------------------------------
            if i + 1 < nx:
                A[k, index(i + 1, j)] += a_east
            else:
                A[k, k] += a_east

            # ------------------------------------------------
            # West boundary: prescribed inlet temperature
            #
            # T_ghost = 2*T_in - T_adjacent
            # The dependence on the adjacent physical cell
            # contributes -aW to the diagonal.
            # ------------------------------------------------
            if i > 0:
                A[k, index(i - 1, j)] += a_west
            else:
                A[k, k] -= a_west

            # ------------------------------------------------
            # North boundary: fixed top-wall temperature
            #
            # T_ghost = 2*T_top - T_adjacent
            # The temperature-dependent contribution is -aN.
            # ------------------------------------------------
            if j + 1 < ny:
                A[k, index(i, j + 1)] += a_north
            else:
                A[k, k] -= a_north

            # ------------------------------------------------
            # South boundary: fixed bottom-wall temperature
            #
            # T_ghost = 2*T_bottom - T_adjacent
            # The temperature-dependent contribution is -aS.
            # ------------------------------------------------
            if j > 0:
                A[k, index(i, j - 1)] += a_south
            else:
                A[k, k] -= a_south

    return A.tocsr()


# ============================================================
# 2. Temporal amplification factors
# ============================================================

def amplification_factor(z, scheme):
    """
    Evaluate the temporal amplification factor G(z).

    z = lambda * dt, where lambda is a spatial eigenvalue.
    """

    scheme = scheme.lower()

    if scheme in ("implicit_euler", "ie"):
        return 1.0 / (1.0 - z)

    if scheme == "rk2":
        # Explicit midpoint RK2
        return 1.0 + z + 0.5 * z**2

    if scheme == "rk4":
        # Classical four-stage, fourth-order Runge-Kutta
        return (
            1.0
            + z
            + 0.5 * z**2
            + z**3 / 6.0
            + z**4 / 24.0
        )

    raise ValueError(
        f"Unknown scheme: {scheme}. "
        "Use 'implicit_euler', 'rk2', or 'rk4'."
    )


# ============================================================
# 3. Stability check for a given time step
# ============================================================

def assess_timestep(eigenvalues, time_step, scheme):
    """
    Check the modal absolute-stability condition

        max_k |G(lambda_k * dt)| <= 1.
    """

    if time_step < 0:
        raise ValueError("The time step cannot be negative.")

    z = time_step * eigenvalues

    with np.errstate(over="ignore", invalid="ignore"):
        amplification = amplification_factor(z, scheme)
        amplification_magnitudes = np.abs(amplification)

    max_amplification = float(
        np.max(amplification_magnitudes)
    )

    stable = (
        np.isfinite(max_amplification)
        and max_amplification <= 1.0 + STABILITY_TOLERANCE
    )

    return max_amplification, stable


# ============================================================
# 4. Estimate maximum stable time step
# ============================================================

def maximum_stable_timestep(
    eigenvalues,
    scheme,
    initial_dt=TIME_STEP,
):
    """
    Estimate the first absolute-stability boundary encountered
    as the time step increases from zero.

    The search first brackets a stable and unstable time step,
    then applies bisection to refine the boundary.

    For standard implicit Euler, return infinity when all spatial
    eigenvalues have non-positive real parts.

    A finite result is a stability limit, not an accuracy limit.
    """

    scheme = scheme.lower()

    max_real_part = float(np.max(eigenvalues.real))

    # A positive real eigenvalue means that a mode in the
    # semi-discrete system itself grows exponentially.
    if max_real_part > STABILITY_TOLERANCE:
        return np.nan

    # Standard backward Euler is A-stable for this spectrum.
    if scheme in ("implicit_euler", "ie"):
        return np.inf

    def is_stable(dt):
        _, stable = assess_timestep(
            eigenvalues,
            dt,
            scheme,
        )
        return stable

    # Find a stable lower bound and unstable upper bound.
    low = 0.0
    high = max(float(initial_dt), 1e-12)

    bracket_found = False

    for _ in range(MAX_DOUBLINGS):

        if not is_stable(high):
            bracket_found = True
            break

        low = high
        high *= 2.0

    if not bracket_found:
        # No instability was found within the search range.
        return np.inf

    # Refine the first bracketed stability boundary.
    for _ in range(BISECTION_ITERATIONS):

        middle = 0.5 * (low + high)

        if is_stable(middle):
            low = middle
        else:
            high = middle

    # Return the stable side of the estimated limit.
    return low


# ============================================================
# 5. Analyse one mesh
# ============================================================

def analyze_mesh(nx, ny):
    """
    Construct the spatial matrix, calculate its full spectrum,
    and evaluate the stability limits for all selected schemes.
    """

    print("\n" + "=" * 72)
    print(f"Spatial stability analysis: nx={nx}, ny={ny}")
    print("=" * 72)

    dx = DOMAIN_LENGTH_X / nx
    dy = DOMAIN_LENGTH_Y / ny

    A = build_spatial_matrix(nx, ny)

    print(f"Matrix dimensions: {A.shape}")
    print(f"Nonzero entries: {A.nnz}")
    print(f"dx = {dx:.8g}")
    print(f"dy = {dy:.8g}")

    print("\nCalculating the full eigenvalue spectrum...")

    # Full spectrum: all nx * ny eigenvalues.
    # Dense eigensolution can be computationally expensive
    # for the finest grid.
    dense_matrix = A.toarray()

    eigenvalues = eigvals(
        dense_matrix,
        check_finite=False,
        overwrite_a=True,
    )

    del dense_matrix

    max_real_part = float(np.max(eigenvalues.real))
    max_eigenvalue_magnitude = float(
        np.max(np.abs(eigenvalues))
    )

    print(f"Eigenvalues calculated: {eigenvalues.size}")
    print(
        f"Maximum real part: "
        f"{max_real_part:.6e}"
    )
    print(
        f"Largest eigenvalue magnitude: "
        f"{max_eigenvalue_magnitude:.6e}"
    )

    if max_real_part > STABILITY_TOLERANCE:
        print(
            "\nWARNING: The spatial matrix has eigenvalues "
            "with positive real parts."
        )

    print(
        f"\nConfigured time step: dt = {TIME_STEP:g}"
    )
    print("-" * 72)

    results = {}

    for scheme in SCHEMES:

        # Stability at the configured time step
        max_g, stable = assess_timestep(
            eigenvalues,
            TIME_STEP,
            scheme,
        )

        # Maximum stable time step
        dt_max = maximum_stable_timestep(
            eigenvalues,
            scheme,
            initial_dt=TIME_STEP,
        )

        if stable:
            status = "STABLE"
        else:
            status = "UNSTABLE"

        print(f"\nScheme: {scheme}")
        print(f"  max |G| at configured dt = {max_g:.10g}")
        print(f"  Configured time step status: {status}")

        if np.isnan(dt_max):
            suggested_dt = np.nan

            print(
                "  Maximum stable dt: not determined because "
                "the spatial matrix has a growing mode."
            )

        elif np.isinf(dt_max):
            suggested_dt = np.inf

            print(
                "  Maximum stable dt: no finite absolute-stability "
                "limit for standard implicit Euler."
            )

        else:
            # A conservative practical choice below the estimated
            # absolute-stability boundary.
            suggested_dt = 0.9 * dt_max

            print(
                f"  Estimated maximum stable dt = {dt_max:.10g}"
            )
            print(
                f"  Suggested dt (90% of limit) = "
                f"{suggested_dt:.10g}"
            )

        results[scheme] = {
            "max_amplification": max_g,
            "stable_at_configured_dt": stable,
            "maximum_stable_timestep": dt_max,
            "suggested_timestep": suggested_dt,
        }

    # Save eigenvalues and metadata for this mesh.
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    output_path = (
        RESULTS_DIRECTORY
        / f"spatial_eigenvalues_{nx}x{ny}.npz"
    )

    np.savez_compressed(
        output_path,
        eigenvalues=eigenvalues,
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        peclet_number=PECLET_NUMBER,
        mean_velocity=MEAN_VELOCITY,
        configured_time_step=TIME_STEP,
    )

    print(f"\nSaved eigenvalues to:\n{output_path}")

    return {
        "nx": nx,
        "ny": ny,
        "dx": dx,
        "dy": dy,
        "max_real_part": max_real_part,
        "max_eigenvalue_magnitude": max_eigenvalue_magnitude,
        "results": results,
    }


# ============================================================
# 6. Run all meshes and summarize
# ============================================================

def run_stability_analysis():
    all_results = []

    for nx, ny in GRID_LEVELS:
        mesh_results = analyze_mesh(nx, ny)
        all_results.append(mesh_results)

    print("\n\n" + "=" * 90)
    print("SUMMARY: TIME-STEP STABILITY")
    print("=" * 90)

    for result in all_results:

        nx = result["nx"]
        ny = result["ny"]

        print(f"\nGrid: {nx} x {ny}")

        for scheme in SCHEMES:
            data = result["results"][scheme]
            dt_max = data["maximum_stable_timestep"]
            dt_suggested = data["suggested_timestep"]

            if np.isnan(dt_max):
                dt_max_text = "undefined"
                dt_suggested_text = "undefined"

            elif np.isinf(dt_max):
                dt_max_text = "no finite stability limit"
                dt_suggested_text = "not applicable"

            else:
                dt_max_text = f"{dt_max:.8g}"
                dt_suggested_text = f"{dt_suggested:.8g}"

            print(
                f"  {scheme:16s} | "
                f"dt_max = {dt_max_text:>22s} | "
                f"suggested dt = {dt_suggested_text}"
            )

    # Save a compact numerical summary.
    summary_path = RESULTS_DIRECTORY / "spatial_stability_summary.npz"

    summary = {
        "grid_levels": np.array(GRID_LEVELS, dtype=int),
        "configured_time_step": TIME_STEP,
    }

    for scheme in SCHEMES:
        summary[f"{scheme}_maximum_stable_dt"] = np.array(
            [
                item["results"][scheme]["maximum_stable_timestep"]
                for item in all_results
            ],
            dtype=float,
        )

        summary[f"{scheme}_suggested_dt"] = np.array(
            [
                item["results"][scheme]["suggested_timestep"]
                for item in all_results
            ],
            dtype=float,
        )

        summary[f"{scheme}_max_amplification"] = np.array(
            [
                item["results"][scheme]["max_amplification"]
                for item in all_results
            ],
            dtype=float,
        )

    np.savez_compressed(summary_path, **summary)

    print(f"\nSaved summary to:\n{summary_path}")