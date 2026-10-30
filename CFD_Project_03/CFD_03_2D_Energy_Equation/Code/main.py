"""Main entry point for CFD Project #3.

Uncomment the desired study in the corresponding section.
"""

# =========================================================
# Imports
# =========================================================

from src import config
from src.solvers.apf import solve_with_apf
from src.solvers.runge_kutta import solve_with_rk2, solve_with_rk4
from src.solvers.direct import solve_with_direct

from studies.afi_verification import run_afi_verfication
from studies.direct_verification import (
   run_direct_matrix_consistency,
   run_direct_implicit_step_verification,
   run_verify_direct_vs_afi,
   run_direct_time_step_test,
)
from studies.snapshots import run_snapshots
from studies.grid_convergence import run_grid_convergence_study
from studies.temporal_accuracy import run_temporal_accuracy_study
from studies.parameter_study import run_parameter_study
from studies.stability import run_stability_analysis
from studies.time_step_efficiency import run_time_step_efficiency_study
from studies.solver_comparison import run_solver_comparison

# =========================================================
# Part 1: Solution and spatial grid convergence
# =========================================================

# AFI
#run_afi_verfication()
#run_grid_convergence_study(
#    solver=solve_with_apf,
#    solver_name="APF",
#)

# RK2
#run_grid_convergence_study(
#    solver=solve_with_rk2,
#    solver_name="RK2",
#    time_step=0.003,
#)

# RK4
#run_grid_convergence_study(
#    solver=solve_with_rk4,
#    solver_name="RK4",
#    time_step=0.001,
#)

# Direct solver
#run_grid_convergence_study(
#    solver=solve_with_direct,
#    solver_name="Direct",
#    time_step=0.1,
#)


# =========================================================
# Part 2: Temporal accuracy
# =========================================================
'''
run_temporal_accuracy_study(
    solver=solve_with_apf,
    solver_name="APF",
)

run_temporal_accuracy_study(
    solver=solve_with_rk2,
    solver_name="RK2",
)

run_temporal_accuracy_study(
    solver=solve_with_rk4,
    solver_name="RK4",
)

run_temporal_accuracy_study(
    solver=solve_with_direct,
    solver_name="Direct",
)
'''

# =========================================================
# Part 3: Solver verification
# =========================================================

#run_afi_verfication()
#run_direct_matrix_consistency()
#run_direct_implicit_step_verification()
#run_verify_direct_vs_afi()
#run_direct_time_step_test(
#  1,
#  mesh=(200, 80),
#  inlet_temperature=None,
#  residual_tolerance=None,
#  max_iterations=None,
#  reference_temperature=None,
#  convection_coefficient=1.0,
#)


# =========================================================
# Part 4: Other studies
# =========================================================

#run_snapshots()
#run_stability_analysis()
run_parameter_study()


# =========================================================
# Part 5: Solver comparison
# =========================================================

#run_time_step_efficiency_study()
#run_solver_comparison()

if __name__ == "__main__":
    pass
