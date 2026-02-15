# Staggered Grid Fluid Simulation (Python)

A 2D Incompressible Fluid Simulation implemented in Python using the **Marker-and-Cell (MAC)** method on a staggered grid. This project demonstrates core computational fluid dynamics (CFD) concepts including semi-Lagrangian advection, pressure projection, and real-time visualization.

## 🌊 Overview

The simulation solves the **Incompressible Navier-Stokes equations** for inviscid flow:

1.  **Momentum Equation**: $\frac{\partial \mathbf{u}}{\partial t} = -(\mathbf{u} \cdot \nabla)\mathbf{u} - \frac{1}{\rho}\nabla p + \mathbf{F}$
2.  **Incompressibility Constraint**: $\nabla \cdot \mathbf{u} = 0$

### Key Features
-   **Staggered Grid (MAC)**: Velocity components ($u, v$) are stored at cell faces, and pressure ($p$) is stored at cell centers. This arrangement naturally couples pressure and velocity, preventing checkerboard numerical artifacts.
-   **Stable Advection**: Uses a **Semi-Lagrangian** scheme (backtracing characteristics) to unconditionally stabilize the advection step, allowing for larger time steps.
-   **Pressure Projection**: Enforces incompressibility by solving a Poisson equation ($\nabla^2 p = \frac{\rho}{\Delta t} \nabla \cdot \mathbf{u}^*$) using `scipy.sparse.linalg`.
-   **Passive Scalar Transport**: Simulates "dye" or "smoke" density to visualize the flow dynamics.
-   **Interactive Visualization**: Real-time animation using `matplotlib`.

## 🚀 Getting Started

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

You should see a window open displaying:
-   **Cyan Arrows**: The velocity field vectors.
-   **Orange/Purple Field**: The density (dye) concentration.
-   The simulation starts with an initial **impulse** in the center, creating a vortex pair that travels through the box.

## 📂 Project Structure

-   `fluid_solver.py`:
    -   Contains the `FluidSolver` class.
    -   Implements the physics engine: `advect`, `project` (pressure solve), and `advect_density`.
    -   Heavily commented to explain the mathematical steps.
-   `Euler_grid.py`:
    -   Sets up the simulation parameters (grid size, time step).
    -   Initializes the fluid state (initial velocity and dye).
    -   Runs the `matplotlib` animation loop.

## 🔬 Physics Implementation Details

### 1. Advection (Semi-Lagrangian)
To update a quantity $q$ at position $\mathbf{x}$, we trace back where a particle at $\mathbf{x}$ came from one time step ago: $\mathbf{x}_{old} \approx \mathbf{x} - \mathbf{u}(\mathbf{x}) \Delta t$. We then interpolate the value of $q$ at $\mathbf{x}_{old}$. This is implemented using `scipy.ndimage.map_coordinates` for efficiency.

### 2. Projection (Pressure Solve)
After advection, the velocity field $\mathbf{u}^*$ may not be divergence-free. We decompose it (Helmholtz-Hodge) into a divergence-free part and a gradient field. We solve for pressure $p$ such that subtracting $\nabla p$ from $\mathbf{u}^*$ makes the divergence zero. This involves solving a large sparse linear system (Discrete Laplacian), handled efficiently by `scipy`.

## 🔮 Future Improvements
-   Add **viscosity** (diffusion term).
-   Implement **interactive controls** (mouse interaction to push fluid).
-   Port to **JAX** or **GPU** for higher resolution.

## License
MIT
