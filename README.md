# Computational Fluid Dynamics and Numerical Methods Projects

This repository collects numerical modelling projects focused on the implementation, verification, and analysis of computational methods for fluid mechanics and heat transfer. The emphasis is on understanding how governing equations are discretized and solved, and on assessing numerical accuracy, convergence, stability, and computational behaviour against analytical solutions.

The projects use Python for numerical implementation and LaTeX for technical reporting.

## Projects

### Project 01 — Steady Two-Dimensional Heat Conduction

**Problem:** Solve the steady two-dimensional Laplace equation on a square domain using a cell-centred finite-volume method.

\[
\nabla^2 T = 0.
\]

The discretization uses a structured Cartesian grid and Dirichlet boundary conditions. Numerical solutions are verified against an analytical solution. The project documents the development from the original master's implementation to a modular solver framework and compares iterative and direct solution methods.

**Methods implemented in V2:**

- **Point Gauss–Seidel:** point-wise iterative solution.
- **Successive Over-Relaxation (SOR):** point-wise iteration with a relaxation factor.
- **Line-SOR:** line-based solution of temperature corrections using a tridiagonal system.
- **Fully implicit direct method:** assembly and direct solution of the discrete linear system.

The repository preserves the original implementation and subsequent versions so the evolution of the code and report remains traceable.

![Implicit solution: temperature contours for the steady heat-conduction problem](CFD_Project_01/CFD_Project_01_Heat_Conduction/V2/report/latex/figures/implicit_temperature_contours.png)

*Representative result: temperature contours obtained using the fully implicit solution method.*

[Open the Project 01 folder](CFD_Project_01/CFD_Project_01_Heat_Conduction/)

### Project 02 — One-Dimensional Linear Advection

**Problem:** Solve the one-dimensional linear advection equation with a prescribed time-dependent inlet condition.

\[
\frac{\partial T}{\partial t} + u\frac{\partial T}{\partial x}=0,
\qquad u=2.
\]

The solver separates spatial operators from time integrators, allowing numerical schemes to be combined without duplicating the time-marching algorithm.

**Methods and studies:**

- **Spatial discretization:** second-order upwind and second-order central schemes.
- **Time integration:** explicit midpoint Runge–Kutta (RK2) and classical fourth-order Runge–Kutta (RK4).
- **Verification:** comparison with the analytical solution, grid-refinement studies, and time-step refinement.
- **Stability analysis:** Fourier-mode analysis of the amplification factors and their dependence on the Courant number.
- **Error analysis:** RMS discrete \(L^2\) errors and observed convergence orders.

For the second-order upwind scheme, the analysis gives a limiting Courant number of \(C_{\max}=0.5\) with RK2 and approximately \(C_{\max}=0.6963\) with RK4. The analysis also shows that second-order central differencing combined with explicit midpoint RK2 does not satisfy the absolute-stability condition for nonzero Fourier modes at any positive Courant number.

![Numerical solution of the one-dimensional advection equation using RK2 and second-order upwind discretization](CFD_Project_02/CFD_Project_02_1D_Advection/report/latex/figures/rk2_2ndupwind_solution.png)

*Representative result: numerical and analytical solution profiles for the second-order upwind scheme advanced with RK2.*

[Open the Project 02 folder](CFD_Project_02/CFD_Project_02_1D_Advection/)

## Repository Layout

```text
.
├── README.md
├── CFD_Project_01/
│   └── CFD_Project_01_Heat_Conduction/
│       └── V2/
│           └── report/
│               └── latex/
│                   └── figures/
│                       └── implicit_temperature_contours.png
└── CFD_Project_02/
    └── CFD_Project_02_1D_Advection/
        └── report/
            └── latex/
                └── figures/
                    └── rk2_2ndupwind_solution.png
```

Each project folder contains its own implementation and report materials. Refer to the README and documentation within each project for version-specific details and instructions for running the code or compiling the report.

## Main Topics

- Finite-volume discretization of partial differential equations
- Iterative and direct numerical solution methods
- Modular scientific Python programming
- Analytical verification and grid-convergence studies
- Runge–Kutta time integration
- Fourier stability analysis and Courant-number limits
- Error estimation, visualization, and technical reporting
