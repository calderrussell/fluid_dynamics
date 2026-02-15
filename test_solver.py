import numpy as np
from fluid_solver import FluidSolver

def test_solver():
    print("Initializing solver...")
    N_x = 32
    N_y = 32
    h = 1.0/32
    dt = 0.1
    solver = FluidSolver(N_x, N_y, dt, h)
    
    print("Setting initial conditions...")
    # Add a block of velocity
    solver.u[10:20, 10:20] = 1.0
    
    print("Running step 1...")
    solver.step()
    print("Step 1 complete.")
    
    print("Running step 2...")
    solver.step()
    print("Step 2 complete.")
    
    # Check for NaNs
    if np.any(np.isnan(solver.u)) or np.any(np.isnan(solver.v)):
        print("ERROR: NaNs detected in velocity field.")
    else:
        print("Velocity field is valid (no NaNs).")
        
    # Check Divergence (should be small)
    div = np.zeros((N_y, N_x))
    div += (solver.u[:, 1:] - solver.u[:, :-1]) / h
    div += (solver.v[1:, :] - solver.v[:-1, :]) / h
    max_div = np.max(np.abs(div))
    print(f"Max Divergence after projection: {max_div}")
    
    if max_div > 1e-3:
        print("WARNING: Divergence is relatively high.")
    else:
        print("Divergence is well constrained.")

if __name__ == "__main__":
    test_solver()
