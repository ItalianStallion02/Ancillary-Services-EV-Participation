import numpy as np
import cvxpy as cp
import os
import time

def create_sparse_halfspace_matrix(T, delta_t=0.25):
    """Constructs the lower bidiagonal difference operator L_inv."""
    D = np.eye(T) - np.eye(T, k=-1)
    L_inv = D / delta_t
    return L_inv

def generate_heterogeneous_fleet(N, T, delta_t=0.25):
    """Generates fleet bounds array h_i concatenated as [x_max, -x_min, u_max, -u_min]."""
    print(f" -> Synthesizing a heterogeneous fleet of {N} EVs over {T} horizons...")
    np.random.seed(42)  # Ensures strict reproducibility
    fleet_h = []
    eps = 1e-3  # Numerical cushion for idle/unplugged hours
    
    for i in range(N):
        a_i = np.random.randint(0, max(1, T // 3))
        d_i = np.random.randint(2 * T // 3, T + 1)
        R_i = np.random.uniform(7.0, 11.0)
        battery_capacity = 60.0
        
        stay_duration = d_i - a_i
        max_achievable_energy = R_i * stay_duration * delta_t
        E_i = min(np.random.uniform(max_achievable_energy * 0.4, max_achievable_energy * 0.6), battery_capacity * 0.75)
        
        u_max, u_min = np.ones(T) * eps, np.ones(T) * -eps
        u_max[a_i:d_i], u_min[a_i:d_i] = R_i, 0.0
        x_max, x_min = np.ones(T) * battery_capacity, np.zeros(T)
        
        if d_i < T: 
            x_min[d_i-1:] = E_i
        else: 
            x_min[-1] = E_i 
        
        fleet_h.append(np.concatenate([x_max, -x_min, u_max, -u_min]))
    return np.array(fleet_h)

def aggregate_flexibility_ah_polytope_structural(L_inv, fleet_h, N, T, delta_t=0.25):
    """
    Exploits the analytical sparsity of Lambda_i.
    Replaces 2D matrix variables with fast 1D vector representations.
    """
    h_0 = np.mean(fleet_h, axis=0)
    
    # Extract baseline fleet boundaries directly as 1D vectors
    x_0_max = h_0[0:T]
    x_0_min = -h_0[T:2*T]
    u_0_max = h_0[2*T:3*T]
    u_0_min = -h_0[3*T:4*T]
    
    gamma_agg_x = np.zeros(T)
    gamma_agg_diag_x = np.zeros(T)
        
    print("\n==========================================================")
    print(f" -> RUNNING STRUCTURAL LINEAR CONSTRAINTS PIPELINE ({N} EVs)")
    print("==========================================================")
    
    global_start = time.time()
    
    for i in range(N):
        h_i = fleet_h[i]
        x_i_max = h_i[0:T]
        x_i_min = -h_i[T:2*T]
        u_i_max = h_i[2*T:3*T]
        u_i_min = -h_i[3*T:4*T]
        
        # Decision tracking vectors
        gamma_i_x = cp.Variable(T)
        gamma_i_diag_x = cp.Variable(T, nonneg=True)
        
        # 1D Vector representations of the active Lambda sub-blocks
        lam_31 = cp.Variable(T, nonneg=True)
        lam_32 = cp.Variable(T, nonneg=True)
        lam_41 = cp.Variable(T, nonneg=True)
        lam_42 = cp.Variable(T, nonneg=True)
        
        # Analytical forward-shifts matching the bidiagonal structures of L_inv
        M_expr = cp.hstack([
            np.zeros(1),
            (gamma_i_diag_x[1:] - gamma_i_diag_x[:-1]) / delta_t
        ])
        d_expr = cp.hstack([
            gamma_i_diag_x[0:1],
            gamma_i_diag_x[:-1]
        ])
        
        # Map element-wise containment constraints directly
        constraints = [
            lam_31 - lam_32 == M_expr,
            lam_41 - lam_42 == -M_expr,
            
            # Block Rows 1 & 2: State Space (Energy Boundary) Containment
            cp.multiply(gamma_i_diag_x, x_0_max) <= x_i_max - gamma_i_x,
            cp.multiply(gamma_i_diag_x, x_0_min) >= x_i_min - gamma_i_x,
            
            # Block Rows 3 & 4: Input Space (Power Boundary) Containment
            cp.multiply(lam_31, x_0_max) - cp.multiply(lam_32, x_0_min) + cp.multiply(d_expr, u_0_max) <= u_i_max - (L_inv @ gamma_i_x),
            cp.multiply(lam_41, x_0_max) - cp.multiply(lam_42, x_0_min) - cp.multiply(d_expr, u_0_min) <= -u_i_min + (L_inv @ gamma_i_x)
        ]
        
        objective = cp.Maximize(cp.sum(gamma_i_diag_x))
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.CLARABEL, verbose=False)
        
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise ValueError(f"Sub-problem failed at EV {i+1}")
            
        gamma_agg_x += gamma_i_x.value
        gamma_agg_diag_x += gamma_i_diag_x.value
        
    print("==========================================================")
    print(f" -> SUCCESS! Total Structured Execution Time: {time.time() - global_start:.4f} seconds")
    print("==========================================================\n")
        
    return gamma_agg_diag_x, gamma_agg_x, h_0

def run_and_save_v2_pipeline(N=100, T=96, dt=0.25, output_dir="v2_polytope"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    L_inv = create_sparse_halfspace_matrix(T=T, delta_t=dt)
    fleet_profiles = generate_heterogeneous_fleet(N=N, T=T, delta_t=dt)
    
    gamma_agg_diag_x, gamma_agg_x, h_0 = aggregate_flexibility_ah_polytope_structural(
        L_inv=L_inv, fleet_h=fleet_profiles, N=N, T=T, delta_t=dt
    )
    
    # Save the optimized data securely
    np.save(os.path.join(output_dir, "gamma_agg_diag_x.npy"), gamma_agg_diag_x)
    np.save(os.path.join(output_dir, "gamma_agg_x.npy"), gamma_agg_x)
    np.save(os.path.join(output_dir, "h_0.npy"), h_0)
    np.save(os.path.join(output_dir, "L_inv.npy"), L_inv)
    
    print(f"[SUCCESS] Polytope data archived successfully in: '{output_dir}/'")

if __name__ == "__main__":
    run_and_save_v2_pipeline(N=100, T=96, dt=0.25)