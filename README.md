# Staggered Grid Fluid Simulation (Python)

A 2D Incompressible Fluid Simulation implemented in Python using the Marker-and-Cell (MAC) method on a staggered grid. This project demonstrates core computational fluid dynamics (CFD) concepts including semi-Lagrangian advection, pressure projection with internal static obstacles, and real-time visualization.

## Overview

The simulation solves the Incompressible Navier-Stokes equations for inviscid flow in the presence of arbitrary static obstacles:

1.  **Momentum Equation**: $\frac{\partial \mathbf{u}}{\partial t} = -(\mathbf{u} \cdot \nabla)\mathbf{u} - \frac{1}{\rho}\nabla p + \mathbf{F}$
2.  **Incompressibility Constraint**: $\nabla \cdot \mathbf{u} = 0$

### Key Features
-   **Staggered Grid (MAC)**: Velocity components ($u, v$) are stored at cell faces, and pressure ($p$) is stored at cell centers. This arrangement naturally couples pressure and velocity, preventing checkerboard numerical artifacts.
-   **Static Obstacles**: Supports arbitrary internal solid boundaries. The solver enforces no-slip/no-through conditions at solid interfaces and treats them as Neumann boundaries ($\partial p / \partial n = 0$) during the pressure projection step.
-   **Stable Advection**: Uses a Semi-Lagrangian scheme (backtracing characteristics) to unconditionally stabilize the advection step, allowing for larger time steps.
-   **Pressure Projection**: Enforces incompressibility by solving a Poisson equation ($\nabla^2 p = \frac{\rho}{\Delta t} \nabla \cdot \mathbf{u}^*$) using `scipy.sparse.linalg`.
-   **Passive Scalar Transport**: Simulates dye density to visualize the flow dynamics.
-   **Interactive Visualization**: Real-time animation using `matplotlib`.

## Getting Started

### Prerequisites

-   Python 3.8+
-   `numpy`
-   `scipy`
-   `matplotlib`

### Installation

1.  Clone the repository and navigate to the folder.
2.  Create a virtual environment (recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

Run the main simulation script:

```bash
python Euler_grid.py
```

The visualization displays:
-   **Cyan Arrows**: The velocity field vectors.
-   **Orange/Purple Field**: The density (dye) concentration.
-   **Gray Region**: The static obstacle.

The simulation initiates with a single impulse directed towards the obstacle, demonstrating flow separation and vortex shedding around the solid geometry.

## Project Structure

-   `fluid_solver.py`:
    -   Contains the `FluidSolver` class.
    -   Implements the physics engine: `advect`, `project` (pressure solve), `advect_density`, and boundary condition enforcement.
    -   Includes heavy comments explaining the mathematical steps and discretization.
-   `Euler_grid.py`:
    -   Configures the simulation domain, obstacles, and initial conditions.
    -   Initializes the fluid state (velocity impulse and dye block).
    -   Executes the `matplotlib` animation loop.

## Physics Implementation Details

### 1. Advection (Semi-Lagrangian)
To update a quantity $q$ at position $\mathbf{x}$, the solver traces back where a particle at $\mathbf{x}$ originated from one time step ago: $\mathbf{x}_{old} \approx \mathbf{x} - \mathbf{u}(\mathbf{x}) \Delta t$. The value of $q$ is then interpolated at $\mathbf{x}_{old}$. This is implemented using `scipy.ndimage.map_coordinates` for efficiency.

### 2. Projection (Pressure Solve)
After advection, the intermediate velocity field $\mathbf{u}^*$ may not be divergence-free. The Helmholtz-Hodge decomposition is applied to extract the solenoidal part. A Poisson equation is solved for pressure $p$, such that subtracting $\nabla p$ from $\mathbf{u}^*$ enforces zero divergence.

For internal obstacles, the discrete Laplacian stencil is modified to decouple pressure inside solid cells and treat the solid-fluid interface as a zero-flux (Neumann) boundary.

### 3. Boundary Conditions
-   **Domain Walls**: No-through flow is enforced at the simulation box edges.
-   **Obstacles**: Velocities on faces adjacent to solid cells are explicitly set to zero ($u \cdot n = 0$ for slip, $\mathbf{u}=0$ for no-slip).

## Future Improvements
-   Add viscosity (diffusion term).
-   Implement interactive user controls (mouse interaction).
-   Port to JAX or GPU-accelerated frameworks for higher resolution.

## License
MIT
