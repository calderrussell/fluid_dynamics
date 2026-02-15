import numpy as np
import scipy.sparse # type: ignore
from scipy.sparse.linalg import spsolve # type: ignore

class FluidSolver:
    """
    A 2D Incompressible Fluid Solver using a Staggered Grid (MAC Grid).
    
    Grid Layout (Staggered / MAC):
    --------------------------------
    The grid separates pressure and velocity components to avoid checkerboard artifacts.
    
    - p (pressure) is stored at CELL CENTERS (i, j). Shape: (N_y, N_x)
    - u (horizontal velocity) is stored at VERTICAL FACES (i, j). Shape: (N_y, N_x + 1)
    - v (vertical velocity) is stored at HORIZONTAL FACES (i, j). Shape: (N_y + 1, N_x)
      
    Solver Steps (Operator Splitting):
    1. Advection: Solve du/dt = -(u . nabla) u
       Method: Semi-Lagrangian Backtracing (Unconditionally stable).
    2. External Forces: Add gravity or user interaction.
    3. Projection: Enforce nabla . u = 0
       Method: Solve Poisson equation for pressure -> Subtract pressure gradient.
    """

    def __init__(self, N_x, N_y, dt, h, rho=1.0, visc=0.0):
        self.N_x = N_x
        self.N_y = N_y
        self.dt = dt
        self.h = h
        self.rho = rho
        self.visc = visc

        # --- Initialize Staggered Grid Velocities ---
        self.u = np.zeros((self.N_y, self.N_x + 1), dtype=np.float64)
        self.v = np.zeros((self.N_y + 1, self.N_x), dtype=np.float64)
        
        # Pressure (Cell centers)
        self.p = np.zeros((self.N_y, self.N_x), dtype=np.float64)

        # Precompute the linear operator (Laplacian matrix) for the Pressure Poisson Equation.
        self._build_pressure_matrix()

    def _build_pressure_matrix(self):
        """
        Builds the sparse Laplacian matrix 'A' for the pressure Poisson equation: A * p = b.
        Discretization: Standard 5-point stencil on a regular grid with Neumann BCs.
        """
        n = self.N_x * self.N_y
        
        # Lists to construct COO matrix (row, col, value)
        data = []
        rows = []
        cols = []

        # Loop over every pressure cell
        for y in range(self.N_y):
            for x in range(self.N_x):
                k = y * self.N_x + x
                
                num_neighbors = 0
                
                # Note: We check neighbors and apply Dirichlet/Neumann BC logic implicitly here.
                # For fluid simulation in a box, flow through walls is 0 (Neumann for pressure).
                
                # Left Neighbor (x-1)
                if x > 0:
                    k_left = y * self.N_x + (x - 1)
                    rows.append(k); cols.append(k_left); data.append(-1.0)
                    num_neighbors += 1
                
                # Right Neighbor (x+1)
                if x < self.N_x - 1:
                    k_right = y * self.N_x + (x + 1)
                    rows.append(k); cols.append(k_right); data.append(-1.0)
                    num_neighbors += 1
                    
                # Top Neighbor (y-1)
                if y > 0:
                    k_up = (y - 1) * self.N_x + x
                    rows.append(k); cols.append(k_up); data.append(-1.0)
                    num_neighbors += 1
                
                # Bottom Neighbor (y+1)
                if y < self.N_y - 1:
                    k_down = (y + 1) * self.N_x + x
                    rows.append(k); cols.append(k_down); data.append(-1.0)
                    num_neighbors += 1
                
                # Diagonal Entry
                rows.append(k); cols.append(k); data.append(float(num_neighbors))

        # Convert to CSR format for efficient solving
        self.laplacian_matrix = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        
        # --- Handle Singularity ---
        # "Pin" one pressure value (p[0,0] = 0) to remove the rank deficiency.
        lil = self.laplacian_matrix.tolil()
        lil[0, :] = 0.0
        lil[0, 0] = 1.0
        self.laplacian_matrix = lil.tocsr()

    def advect(self):
        """
        Advection Step: Updates velocity by moving it along the flow field.
        Method: Semi-Lagrangian Backtracing.
        """
        from scipy.ndimage import map_coordinates
        
        # ---------------------------------------------------------------------
        # Advect U Component
        # ---------------------------------------------------------------------
        # We need V at U-locations.
        v_cen = 0.5 * (self.v[:-1, :] + self.v[1:, :]) 
        v_cen_padded = np.pad(v_cen, ((0, 0), (1, 1)), mode='constant')
        v_at_u = 0.5 * (v_cen_padded[:, :-1] + v_cen_padded[:, 1:])
        
        coords_y, coords_x = np.meshgrid(np.arange(self.N_y), np.arange(self.N_x + 1), indexing='ij')
        
        # Backtrace
        old_y = coords_y - (v_at_u * self.dt / self.h)
        old_x = coords_x - (self.u * self.dt / self.h)
        
        new_u = map_coordinates(self.u, [old_y, old_x], order=1, mode='nearest')
        
        # ---------------------------------------------------------------------
        # Advect V Component
        # ---------------------------------------------------------------------
        # We need U at V-locations.
        u_cen = 0.5 * (self.u[:, :-1] + self.u[:, 1:])
        u_cen_padded = np.pad(u_cen, ((1, 1), (0, 0)), mode='constant')
        u_at_v = 0.5 * (u_cen_padded[:-1, :] + u_cen_padded[1:, :])
        
        coords_y_v, coords_x_v = np.meshgrid(np.arange(self.N_y + 1), np.arange(self.N_x), indexing='ij')
        
        old_y_v = coords_y_v - (self.v * self.dt / self.h)
        old_x_v = coords_x_v - (u_at_v * self.dt / self.h)
        
        new_v = map_coordinates(self.v, [old_y_v, old_x_v], order=1, mode='nearest')
        
        # Update and Apply Boundary Conditions
        self.u = new_u
        self.v = new_v
        self.apply_boundary_conditions()

    def advect_density(self, density_field):
        """
        Advect a scalar field (like smoke density or dye) through the velocity field.
        
        Args:
            density_field: 2D numpy array (N_y, N_x) of scalar values.
            
        Returns:
            new_density: Advected density field.
        """
        from scipy.ndimage import map_coordinates
        
        # Density lives at cell centers (N_y, N_x)
        # We need velocity at cell centers to backtrace
        
        # Interpolate U and V to cell centers
        u_cen = 0.5 * (self.u[:, :-1] + self.u[:, 1:])
        v_cen = 0.5 * (self.v[:-1, :] + self.v[1:, :])
        
        coords_y, coords_x = np.meshgrid(np.arange(self.N_y), np.arange(self.N_x), indexing='ij')
        
        # Backtrace
        old_y = coords_y - (v_cen * self.dt / self.h)
        old_x = coords_x - (u_cen * self.dt / self.h)
        
        # Interpolate new density
        new_density = map_coordinates(density_field, [old_y, old_x], order=1, mode='constant', cval=0.0)
        
        return new_density

    def apply_boundary_conditions(self):
        """
        Enforce boundary conditions (No-through flow).
        """
        # Left and Right walls: u = 0
        self.u[:, 0] = 0.0
        self.u[:, -1] = 0.0
        
        # Top and Bottom walls: v = 0
        self.v[0, :] = 0.0
        self.v[-1, :] = 0.0

    def project(self):
        """
        Projection Step: Enforces incompressibility (divergence-free).
        """
        
        # 1. Calculate Divergence
        div = np.zeros((self.N_y, self.N_x))
        div += (self.u[:, 1:] - self.u[:, :-1]) # du part
        div += (self.v[1:, :] - self.v[:-1, :]) # dv part
        div /= self.h
        
        # 2. Setup RHS for Poisson solve
        rhs_scale = - (self.h ** 2) * (self.rho / self.dt)
        rhs = div * rhs_scale
        rhs_flat = rhs.flatten()
        rhs_flat[0] = 0.0 # Pinning correction for first row
        
        # 3. Solve Linear System
        p_flat = spsolve(self.laplacian_matrix, rhs_flat)
        self.p = p_flat.reshape((self.N_y, self.N_x))
        
        # 4. Correct Velocities (Subtract Gradient)
        const = (self.dt / self.rho) / self.h
        
        # Update interior U
        self.u[:, 1:-1] -= const * (self.p[:, 1:] - self.p[:, :-1])
        
        # Update interior V
        self.v[1:-1, :] -= const * (self.p[1:, :] - self.p[:-1, :])
        
        # Re-enforce boundary conditions
        self.apply_boundary_conditions()

    def step(self):
        """Advance the simulation by one time step."""
        self.advect()
        self.project()
