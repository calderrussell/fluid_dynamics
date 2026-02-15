import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from fluid_solver import FluidSolver

# ==============================================================================
# Fluid Simulation Experiment
# ==============================================================================
# What this simulates:
# A 2D box of fluid. Initially, the fluid is at rest.
# We inject a "puff" of smoke (density) and a burst of velocity (impulse) 
# in the center of the box.
#
# We then watch how:
# 1. The velocity advects (moves) the density.
# 2. The pressure projection ensures the fluid doesn't compress, causing swirls 
#    and vortices (incompressibility).
# 3. The fluid settles over time as viscosity (numerical or explicit) slows it down.
# ==============================================================================

# --- Configuration ---
N_x = 64
N_y = 64
h = 1.0 / N_x 
dt = 0.01
rho = 1.0
visc = 0.0 # Numerical dissipation will act as slight viscosity

# --- Obstacle Setup ---
# Create a mask for solid objects. True = Wall.
solid_mask = np.zeros((N_y, N_x), dtype=bool)

# Define a square obstacle in the top-right quadrant
# Top-right is high x, high y indexes
obs_x_start, obs_x_end = int(0.6 * N_x), int(0.8 * N_x)
obs_y_start, obs_y_end = int(0.6 * N_y), int(0.8 * N_y)

solid_mask[obs_y_start:obs_y_end, obs_x_start:obs_x_end] = True

# --- Initialize Solver ---
solver = FluidSolver(N_x, N_y, dt, h, rho=rho, visc=visc, solid_mask=solid_mask)

# --- Initial Conditions (The "Impulse") ---
# Instead of continuous forcing, we set an initial state and let it evolve.

# 1. Create a Density field (Smoke/Dye)
# We place a blob of density in the middle
density = np.zeros((N_y, N_x))
cx, cy = N_x // 2, N_y // 2
density[cy-8:cy+8, cx-8:cx+8] = 1.0  # A square block of dye

# 2. Add an initial velocity impulse
# We push the fluid up and to the right initially (towards the obstacle)
solver.u[cy-8:cy+8, cx-8:cx+8] = 3.0
solver.v[cy-8:cy+8, cx-8:cx+8] = 3.0

print("Simulation initialized with a single center impulse.")
print(f"Obstacle placed at X:[{obs_x_start}-{obs_x_end}], Y:[{obs_y_start}-{obs_y_end}]")

# --- Visualization Setup ---
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_title("Staggered Grid Fluid Simulation\n(Density + Velocity + Obstacles)")
ax.set_xlim(0, N_x)
ax.set_ylim(0, N_y)

# 1. Plot Density (The "Smoke")
# We use imshow to show the scalar field
im = ax.imshow(density, origin='lower', cmap='inferno', vmin=0, vmax=1.0, extent=[0, N_x, 0, N_y])

# 2. Plot Obstacles
# Overlay the solid mask in gray
# We use a masked array to make non-solid parts transparent
obstacle_viz = np.ma.masked_where(~solid_mask, np.ones_like(solid_mask))
ax.imshow(obstacle_viz, origin='lower', cmap='gray', vmin=0, vmax=1, 
          extent=[0, N_x, 0, N_y], alpha=0.9, interpolation='nearest')

# 3. Plot Velocity (The "Flow")
# We use quiver with fewer arrows (stride) to keep it readable
stride = 2
x_grid, y_grid = np.meshgrid(np.arange(N_x) + 0.5, np.arange(N_y) + 0.5)
q = ax.quiver(x_grid[::stride, ::stride], y_grid[::stride, ::stride], 
              np.zeros((N_y//stride, N_x//stride)), np.zeros((N_y//stride, N_x//stride)), 
              scale=15, color='cyan', width=0.003, alpha=0.6)

def update(frame):
    global density
    
    # 1. Advect Density (Move the smoke)
    density = solver.advect_density(density)
    
    # 2. Step the Physics (Advect Velocity + Project Pressure)
    solver.step()
    
    # --- Update Plots ---
    
    # Update Density Image
    im.set_data(density)
    
    # Update Quiver Arrows
    # Interpolate staggered velocity to cell centers for visualization
    u_c = 0.5 * (solver.u[:, :-1] + solver.u[:, 1:])
    v_c = 0.5 * (solver.v[:-1, :] + solver.v[1:, :])
    
    q.set_UVC(u_c[::stride, ::stride], v_c[::stride, ::stride])
    
    return im, q

anim = FuncAnimation(fig, update, frames=200, interval=30, blit=False)

if __name__ == "__main__":
    try:
        print("Running visualization... Close window to stop.")
        plt.show()
    except KeyboardInterrupt:
        pass
