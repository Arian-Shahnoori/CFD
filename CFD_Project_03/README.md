# Two-Dimensional Channel-Flow Heat Transfer

A Python implementation of a cell-centred finite-volume solver for the two-dimensional energy equation in a channel flow. The project examines the effects of convection, thermal diffusion, and viscous dissipation on the steady-state temperature field. It also compares numerical methods for advancing the discretized equation toward steady state.

## 1. Project Overview

The computational domain is a rectangular channel with dimensions:

$$
L_x=5,\qquad L_y=1.
$$

The prescribed velocity field represents fully developed laminar channel flow:

$$
u(y)=18y(1-y),\qquad v=0,
$$

with a mean velocity of $\bar{u}=3$.

The energy equation accounts for convective transport, thermal diffusion, and viscous dissipation. Its parameterized form is:

$$
\frac{\partial T}{\partial t}
=
-c\nabla\cdot(\mathbf{u}T)
+\frac{1}{Pe}\nabla^2T
+\frac{Br}{Pe}\Phi,
$$

where $\Phi$ denotes the viscous-dissipation function without its multiplicative coefficient.

The dimensionless parameters are defined as:

$$
Pe=Re\,Pr,\qquad Br=Ec\,Pr.
$$

The coefficient $c$ controls the strength of convective transport.

## 2. Boundary and Initial Conditions

The temperature boundary conditions are:

| Boundary | Condition |
|---|---|
| Bottom wall | $T=0$ |
| Top wall | $T=1$ |
| Inlet | $T_{\mathrm{in}}(y)=y$ |
| Outlet | $\partial T/\partial x=0$ |

The initial temperature field is:

$$
T(x,y,0)=y.
$$

The velocity field remains prescribed throughout the calculation. The project therefore solves the energy equation without simultaneously solving the momentum equations.

## 3. Numerical Methods

The computational domain is discretized using a second-order, cell-centred finite-volume formulation on a structured Cartesian grid. Central-difference approximations are used for the spatial derivatives.

The resulting semi-discrete system is advanced in pseudo-time until the steady-state residual satisfies the prescribed convergence criterion.

Four numerical methods are implemented:

- **Approximate factorization method (APF):** An iterative method based on factorization of the spatial operator.
- **Sparse direct implicit method:** An implicit update based on solving a sparse linear system.
- **Explicit midpoint Runge–Kutta (RK2):** A second-order explicit time-integration method.
- **Classical Runge–Kutta (RK4):** A fourth-order explicit time-integration method.

The methods are compared using residual histories, iteration counts, and computation times. An eigenvalue-based stability analysis is also performed to estimate the time-step restrictions of the explicit methods.

## 4. Parameter Study

The parameter study investigates the effects of the Brinkman number, convection coefficient, and Peclet number on the steady-state temperature distribution.

### 4.1 Brinkman Number

The Brinkman number controls the strength of viscous dissipation relative to thermal diffusion. The following values are considered:

$$
Br=0,\quad 0.07,\quad 0.2,
$$

with $Pe=17.5$ and $c=1$ held constant.

### 4.2 Convection Coefficient

The convection coefficient is varied to examine the influence of convective transport:

$$
c=0.4,\quad 1,\quad 10,
$$

with $Pe=17.5$ and $Br=0.2$ held constant.

### 4.3 Peclet Number

The Peclet-number study considers:

$$
Pe=1,\quad 17.5,\quad 50.
$$

The ratio $Br/Pe$ is held constant at $0.004$. This keeps the coefficient of the viscous-dissipation source fixed while varying the thermal-diffusion coefficient $1/Pe$.

Temperature contours and wall-normal temperature profiles are used to compare the resulting fields.

## 5. Repository Structure

The main project components are organized as follows:

```text
project/
│
├── src/
│   ├── initialization.py
│   ├── boundary_condition.py
│   ├── diagnostics.py
│   ├── solvers/
│   │   ├── apf.py
│   │   └── ...
│   └── visualization/
│       └── ...
│
├── results/
│   ├── solver_comparison/
│   └── parameter_study/
│       ├── figures/
│       │   ├── parameter_groups/
│       │   └── cross_sections/
│       └── parameter_study_fields.npz
│
├── sections/
│   ├── 01_problem_Formulation.tex
│   ├── 02_Governing_Equation_and_Spatial_Discretization.tex
│   ├── 03_Temporal_Discretization.tex
│   └── ...
│
├── main.tex
├── packages.tex
└── README.md
```

This is a simplified representation of the project structure. Additional solver, visualization, and result files may be present.

## 6. Requirements

The Python implementation uses the following libraries:

- Python
- NumPy
- SciPy
- Matplotlib

Install the main dependencies with:

```bash
pip install numpy scipy matplotlib
```

The LaTeX report requires a LaTeX distribution and the packages specified in `packages.tex`.

## 7. Running the Simulations

### Parameter Study

The parameter-study script defines the simulation cases in `CASES` and executes them through `run_parameter_study()`.

Run the parameter-study script from the project root:

```bash
python path/to/parameter_study.py
```

The script solves each unique case, generates comparison figures, and saves the numerical fields and diagnostics.

The main configuration parameters are defined at the beginning of the script:

```python
CELL_COUNT_X = 100
CELL_COUNT_Y = 40

TIME_STEP = 0.05
MAX_ITERATIONS = 100_000
TOLERANCE = 1e-10

MEAN_VELOCITY = 3.0
```

### Solver Comparison

The solver-comparison script evaluates the implemented numerical methods using a common grid, convergence tolerance, and selected time steps. The corresponding configuration and execution are defined in the solver-comparison script.

The reported performance depends on the grid resolution, time step, convergence tolerance, implementation, and computational environment.

## 8. Generated Results

### Solver Comparison

The solver-comparison study generates:

- Residual versus iteration plots.
- Residual versus elapsed-time plots.
- Numerical data containing iteration counts, final residuals, and computation times.

### Parameter Study

The parameter study generates:

- Temperature-contour comparisons for the Brinkman-number cases.
- Temperature-contour comparisons for the convection-coefficient cases.
- Temperature-contour comparisons for the Peclet-number cases.
- Contour and cross-section figures for selected parameter combinations.
- Compressed numerical data containing temperature fields, parameter values, residuals, computation times, and iteration counts.

The parameter-study data are saved in:

```text
results/parameter_study/parameter_study_fields.npz
```

The data can be loaded in Python using:

```python
import numpy as np

data = np.load(
    "results/parameter_study/parameter_study_fields.npz"
)

print(data.files)
```

## 9. Report

The project report is written in LaTeX. It covers the governing equations, spatial and temporal discretization, stability analysis, numerical accuracy assessment, computational implementation, solver comparison, and parameter study.

Compile the main LaTeX file from the project root using:

```bash
latexmk -pdf main.tex
```

Alternatively, compile it using a LaTeX editor.

## 10. Scope and Limitations

The velocity field is prescribed and is not updated by solving the momentum equations. The study therefore focuses on heat transfer within an established channel-flow field.

Central-difference spatial discretization is used for the convective and diffusive terms. Its accuracy and stability characteristics depend on the grid resolution and the relative strengths of convection and diffusion.

The time variable is used for pseudo-time integration toward a steady solution. The numerical time steps and convergence histories should therefore be interpreted in the context of the steady-state solution procedure rather than as predictions of physical transient behaviour.
