# Steady Two-Dimensional Heat Conduction — Finite-Volume Solver

A Python project for solving the steady-state heat conduction equation on a square domain using the finite-volume method. The project documents the development of a numerical solver, from the original master's implementation to a modular framework for comparing iterative and direct solution methods.

## Problem Description

The governing equation is the two-dimensional Laplace equation,

$$
\nabla^2 T = 0,
$$

discretized using a second-order, cell-centered finite-volume formulation on a structured Cartesian grid. Dirichlet boundary conditions are imposed using ghost cells, and the numerical solutions are verified against an analytical solution.

The project investigates numerical solution methods, convergence behavior, solution accuracy, and computational cost.

## Repository Structure

### Original MSc Implementation

[`MSc_Original/`](CFD_Project_01_Heat_Conduction/MSc_Original/)

Contains the original Python code and report developed during my master's studies. These files are preserved in their original form to document the initial implementation and provide a reference for subsequent development.

### V1 — Point Gauss-Seidel and LaTeX Report

[`V1/`](CFD_Project_01_Heat_Conduction/V1/)

Contains the initial Point Gauss-Seidel implementation and the revised English project report developed in LaTeX. This version focuses on documenting the numerical formulation, implementation, analytical verification, and results in a structured technical report. The code retains the original, non-modular organization.

### V2 — Modular Solver Framework

[`V2/`](CFD_Project_01_Heat_Conduction/V2/)

Extends the project into a modular Python codebase, separating numerical solvers, shared computational functions, study definitions, and visualization.

Four solution methods are implemented:

* **Point Gauss-Seidel:** Sequential point-wise iterative solution.
* **Successive Over-Relaxation (SOR):** Point-wise iteration with a relaxation factor.
* **Line-SOR:** Line-based solution of temperature corrections using a tridiagonal coefficient matrix.
* **Fully implicit direct method:** Assembly and direct solution of the discrete linear system.

V2 also includes verification against the analytical solution, mesh-size studies, solver convergence comparisons, and CPU-time evaluations.

The updated report is available here: [CFD_Project_01_Heat_Conduction_V2.pdf](CFD_Project_01_Heat_Conduction/V2/report/CFD_Project_01_Heat_Conduction_V2.pdf).

## Purpose

The main objective is to develop a clear and extensible numerical implementation while examining how the choice of solution method affects convergence and computational efficiency. Keeping the original implementation and both subsequent versions makes the development process traceable and preserves the distinction between the original work, the report-focused revision, and the modular solver framework.

For implementation details and instructions for running the code, see the README in the corresponding version directory.
