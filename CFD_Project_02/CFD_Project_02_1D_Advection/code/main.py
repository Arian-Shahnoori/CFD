from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent

# Make modules in src importable.
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from studies.study_01_rk2_second_order_upwind_convergence import (
    run_study_01_rk2_second_order_upwind_convergence,
)
from studies.study_02_rk2_second_order_upwind_stability import (
    run_study_02_rk2_second_order_upwind_stability,
)
from studies.study_03_rk2_second_order_upwind_cfl_validation import (
    run_study_03_rk2_second_order_upwind_cfl_validation,
)
from studies.study_04_rk2_second_order_central_convergence import (
    run_study_04_rk2_second_order_central_convergence,
)
from studies.study_05_rk2_central_upwind_comparison import (
    run_study_05_rk2_central_upwind_comparison,
)
from studies.study_06_rk4_second_order_upwind_stability import (
    run_study_06_rk4_second_order_upwind_stability,
)
from studies.study_07_rk2_rk4_accuracy_comparison import (
    run_study_07_rk2_rk4_accuracy_comparison
)

def main():
    run_study_01_rk2_second_order_upwind_convergence()
    run_study_02_rk2_second_order_upwind_stability()
    run_study_03_rk2_second_order_upwind_cfl_validation()
    run_study_04_rk2_second_order_central_convergence()
    run_study_05_rk2_central_upwind_comparison()
    run_study_06_rk4_second_order_upwind_stability()
    run_study_07_rk2_rk4_accuracy_comparison()


if __name__ == "__main__":
    main()