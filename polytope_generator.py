# polytope_generator.py
import numpy as np
import cvxpy as cp
import os

def create_sparse_halfspace_matrix(T, delta_t=0.25):
    """
    Constructs the highly sparse net-energy space H_tilde matrix (H * L^-1).
    By Remark 1, using this matrix drastically accelerates dual containment LP.
    """
    I_T = np.eye(T)
    D = np.eye(T) - np.eye(T, k=-1)
    L_inv = D / delta_t
    
    # H = [L; -L; I; -I] -> H_tilde = H @ L_inv = [I; -I; L_inv; -L_inv]
    H_tilde = np.vstack([I_T, -I_T, L_inv, -L_inv])
    return H_tilde, L_inv

def generate_heterogeneous_fleet(N, T, delta_t=0.25):
    """
    Generates a fleet of individual EV polytopes h_i over a 96-period horizon.
    Keeps energy boundaries robustly bounded for optimization stability.
    """
    print(f" -> Synthesizing a heterogeneous fleet of {N} EVs over {T} horizons...")
    np.random.seed(42)  # For reproducibility across script components
    fleet_h = []
    eps = 1e-3  # Small cushion for numerical stability during inactive hours
    
    for i in range(N):
        a_i = np.random.randint(0, max(1, T // 3))
        d_i = np.random.randint(2 * T // 3, T + 1)
        
        R_i = np.random.uniform(7.0, 11.0)
        battery_capacity = 60.0
        
        stay_duration = d_i - a_i
        max_achievable_energy = R_i * stay_duration * delta_t
        
        E_i = np.random.uniform(max_achievable_energy * 0.4, max_achievable_energy * 0.6)
        E_i = min(E_i, battery_capacity * 0.75)
        
        # Power is active only during plug-in windows [a_i:d_i]
        u_max = np.ones(T) * eps
        u_min = np.ones(T) * -eps
        u_max[a_i:d_i] = R_i
        u_min[a_i:d_i] = 0.0
        
        # Energy capacity remains continuous to guarantee solver boundary boundedness
        x_max = np.ones(T) * battery_capacity
        x_min = np.zeros(T)
        
        # Enforce target departure requirement
        if d_i < T:
            x_min[d_i-1:] = E_i
        else:
            x_min[-1] = E_i 
        
        h_i = np.concatenate([x_max, -x_min, u_max, -u_min])
        fleet_h.append(h_i)
        
    return np.array(fleet_h)

def aggregate_flexibility_ah_polytope_lp_sparse(H_tilde, fleet_h, N, T):
    """
    Solves maximum-volume inner approximation using a Diagonal Scaling matrix
    reformulated in Net-Energy space.
    """
    m_H = H_tilde.shape[0]
    h_0 = np.mean(fleet_h, axis=0)
    
    gamma_agg_x = cp.Variable(T)
    gamma_agg_diag_x = cp.Variable(T, nonneg=True)
        
    constraints = []
    gamma_i_sum_x = np.zeros(T)
    gamma_i_diag_sum_x = np.zeros(T)
    
    for i in range(N):
        gamma_i_x = cp.Variable(T)
        gamma_i_diag_x = cp.Variable(T, nonneg=True)
        Lambda_i = cp.Variable((m_H, m_H), nonneg=True)
            
        gamma_i_sum_x = gamma_i_sum_x + gamma_i_x
        gamma_i_diag_sum_x = gamma_i_diag_sum_x + gamma_i_diag_x
        
        constraints.append(Lambda_i @ H_tilde == H_tilde @ cp.diag(gamma_i_diag_x))
        constraints.append(Lambda_i @ h_0 <= fleet_h[i] - H_tilde @ gamma_i_x)
        
    constraints.append(gamma_agg_x == gamma_i_sum_x)
    constraints.append(gamma_agg_diag_x == gamma_i_diag_sum_x)
    
    objective = cp.Maximize(cp.sum(gamma_agg_diag_x))
    prob = cp.Problem(objective, constraints)
    
    print("\n==========================================================")
    print(" -> INVOKING CLARABEL FOR SPARSIFIED FLEXIBILITY AGGREGATION")
    print("==========================================================")
    prob.solve(solver=cp.CLARABEL, verbose=True)
    print("==========================================================\n")
    
    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        raise ValueError(f"LP Optimization failed: {prob.status}")
        
    return gamma_agg_diag_x.value, gamma_agg_x.value, h_0

def run_and_save_polytope_pipeline(N=15, T=96, dt=0.25, output_dir="saved_polytope"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    H_tilde, L_inv = create_sparse_halfspace_matrix(T=T, delta_t=dt)
    
    # Track individual EV arrival AND departure timestamps explicitly for visualization
    np.random.seed(42)
    arrivals = []
    departures = []
    for i in range(N):
        a_i = np.random.randint(0, max(1, T // 3))
        d_i = np.random.randint(2 * T // 3, T + 1)
        arrivals.append(a_i * dt)
        departures.append(d_i * dt)
        
    np.save(os.path.join(output_dir, "arrivals.npy"), np.array(arrivals))
    np.save(os.path.join(output_dir, "departures.npy"), np.array(departures))
    
    fleet_profiles = generate_heterogeneous_fleet(N=N, T=T, delta_t=dt)
    
    gamma_agg_diag_x, gamma_agg_x, h_0 = aggregate_flexibility_ah_polytope_lp_sparse(
        H_tilde=H_tilde, fleet_h=fleet_profiles, N=N, T=T
    )
    
    np.save(os.path.join(output_dir, "gamma_agg_diag_x.npy"), gamma_agg_diag_x)
    np.save(os.path.join(output_dir, "gamma_agg_x.npy"), gamma_agg_x)
    np.save(os.path.join(output_dir, "h_0.npy"), h_0)
    np.save(os.path.join(output_dir, "L_inv.npy"), L_inv)
    
    print(f"[SUCCESS] Polytope data, arrivals, and departures successfully archived in: '{output_dir}/'")

if __name__ == "__main__":
    run_and_save_polytope_pipeline(N=15, T=96, dt=0.25)