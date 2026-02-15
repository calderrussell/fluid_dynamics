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

    def __init__(self, N_x, N_y, dt, h, rho=1.0, visc=0.0, solid_mask=None):
        self.N_x = N_x
        self.N_y = N_y
        self.dt = dt
        self.h = h
        self.rho = rho
        self.visc = visc

        # --- Obstacles ---
        # solid_mask: 2D boolean array (N_y, N_x). True where solid.
        if solid_mask is None:
            self.solid_mask = np.zeros((N_y, N_x), dtype=bool)
        else:
            self.solid_mask = solid_mask

        # --- Initialize Staggered Grid Velocities ---
        self.u = np.zeros((self.N_y, self.N_x + 1), dtype=np.float64)
        self.v = np.zeros((self.N_y + 1, self.N_x), dtype=np.float64)
        
        # Pressure (Cell centers)
        self.p = np.zeros((self.N_y, self.N_x), dtype=np.float64)

        # Precompute the linear operator (Laplacian matrix) for the Pressure Poisson Equation.
        self._build_pressure_matrix()

    def _build_pressure_matrix(self):
        """
        Builds the sparse Laplacian matrix 'A'.
        Internal obstacles are treated as Neumann boundaries (flux=0).
        """
        n = self.N_x * self.N_y
        
        data = []
        rows = []
        cols = []

        # Loop over every pressure cell
        for y in range(self.N_y):
            for x in range(self.N_x):
                k = y * self.N_x + x
                
                # If this cell is solid, pressure is irrelevant (decoupled).
                # To keep matrix non-singular, set explicit dummy equation p=0.
                if self.solid_mask[y, x]:
                    rows.append(k); cols.append(k); data.append(1.0)
                    continue
                
                # Build Laplacian stencil for FLUID cells
                num_fluid_neighbors = 0
                
                # Check neighbors (Left, Right, Up, Down)
                # Connection exists ONLY if neighbor is also FLUID.
                
                # Left Neighbor (x-1)
                if x > 0:
                    if not self.solid_mask[y, x-1]:
                        k_left = y * self.N_x + (x - 1)
                        rows.append(k); cols.append(k_left); data.append(-1.0)
                        num_fluid_neighbors += 1
                
                # Right Neighbor (x+1)
                if x < self.N_x - 1:
                    if not self.solid_mask[y, x+1]:
                        k_right = y * self.N_x + (x + 1)
                        rows.append(k); cols.append(k_right); data.append(-1.0)
                        num_fluid_neighbors += 1
                    
                # Top Neighbor (y-1)
                if y > 0:
                    if not self.solid_mask[y-1, x]:
                        k_up = (y - 1) * self.N_x + x
                        rows.append(k); cols.append(k_up); data.append(-1.0)
                        num_fluid_neighbors += 1
                
                # Bottom Neighbor (y+1)
                if y < self.N_y - 1:
                    if not self.solid_mask[y+1, x]:
                        k_down = (y + 1) * self.N_x + x
                        rows.append(k); cols.append(k_down); data.append(-1.0)
                        num_fluid_neighbors += 1
                
                # Diagonal Entry
                rows.append(k); cols.append(k); data.append(float(num_fluid_neighbors))

        # Convert to CSR format for efficient solving
        self.laplacian_matrix = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        
        # --- Handle Singularity ---
        # "Pin" one fluid pressure value (find first fluid cell)
        # Check if there are any fluid cells
        if not np.all(self.solid_mask):
            # Find first fluid cell index
            flat_mask = self.solid_mask.flatten()
            first_fluid_idx = np.where(~flat_mask)[0][0]
            
            lil = self.laplacian_matrix.tolil()
            lil[first_fluid_idx, :] = 0.0
            lil[first_fluid_idx, first_fluid_idx] = 1.0
            self.laplacian_matrix = lil.tocsr()
            self.pinned_idx = first_fluid_idx
        else:
            self.pinned_idx = 0 # Fallback

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
        
        # Mask out density inside obstacles? 
        # Yes, density inside solids should be 0 (or at least irrelevant)
        new_density[self.solid_mask] = 0.0
        
        return new_density

    def apply_boundary_conditions(self):
        """
        Enforce boundary conditions (No-through flow) for Domain Walls AND Internal Obstacles.
        """
        # --- Domain Walls ---
        self.u[:, 0] = 0.0 # Left
        self.u[:, -1] = 0.0 # Right
        self.v[0, :] = 0.0 # Bottom
        self.v[-1, :] = 0.0 # Top
        
        # --- Internal Obstacles ---
        # U faces touching solid cells
        # u[j, i] is face between cell (j, i-1) and (j, i).
        # We need a mask of shape (N_y, N_x+1)
        
        # Let's construct mask_u of shape (N_y, N_x+1) filled with False
        mask_u = np.zeros_like(self.u, dtype=bool)
        
        # Internal faces i=1 to N_x-1
        # Block if left cell (i-1) is solid
        mask_u[:, 1:-1] = np.logical_or(mask_u[:, 1:-1], self.solid_mask[:, :-1])
        # Block if right cell (i) is solid
        mask_u[:, 1:-1] = np.logical_or(mask_u[:, 1:-1], self.solid_mask[:, 1:])
        
        self.u[mask_u] = 0.0
        
        # V faces touching solid cells
        # v[j, i] is face between (j-1, i) [bottom] and (j, i) [top]
        # Shape (N_y+1, N_x)
        mask_v = np.zeros_like(self.v, dtype=bool)
        
        # Internal faces j=1 to N_y-1
        # Block if bottom cell (j-1) is solid
        mask_v[1:-1, :] = np.logical_or(mask_v[1:-1, :], self.solid_mask[:-1, :])
        # Block if top cell (j) is solid
        mask_v[1:-1, :] = np.logical_or(mask_v[1:-1, :], self.solid_mask[1:, :])
        
        self.v[mask_v] = 0.0

    def project(self):
        """
        Projection Step: Enforces incompressibility (divergence-free).
        """
        
        # 1. Calculate Divergence
        div = np.zeros((self.N_y, self.N_x))
        div += (self.u[:, 1:] - self.u[:, :-1]) # du part
        div += (self.v[1:, :] - self.v[:-1, :]) # dv part
        div /= self.h
        
        # Zero out divergence in solid cells (irrelevant)
        div[self.solid_mask] = 0.0
        
        # 2. Setup RHS for Poisson solve
        rhs_scale = - (self.h ** 2) * (self.rho / self.dt)
        rhs = div * rhs_scale
        rhs_flat = rhs.flatten()
        
        # Pinning correction
        if hasattr(self, 'pinned_idx'):
            rhs_flat[self.pinned_idx] = 0.0 
        
        # Solid cells have dummy equation 1*p = 0, so RHS should be 0 there too?
        # Yes, we modify Laplacian diagonal to 1 for these.
        # Set RHS to 0 for solid cells to be safe (though div was 0).
        rhs_flat[self.solid_mask.flatten()] = 0.0
        
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
