from src.initialization import initialize_temperature
from src.solvers.gauss_seidel import gauss_seidel
from src.grid import create_grid
from src.exact_solution import exact_solution
from src.visualization import plot_contours, plot_error, plot_line_cuts, plot_residuals



# =============================================================================
# verification of the solution
# =============================================================================
tolerance = 1e-10
def run_gs_verification(N):

    T = initialize_temperature(N)

    T, residual_history, time_history, iteration = gauss_seidel(
        T,
        N,
        tolerance
    )

    T_num = T[1:N+1, 1:N+1]

    x, X, Y = create_grid(N)

    T_exact = exact_solution(X, Y)

    error = T_exact - T_num

    plot_contours(X, Y, T_num, T_exact)

    plot_error(X, Y, error)

    plot_line_cuts(x, T_num, T_exact)

    plot_residuals(residual_history, time_history)