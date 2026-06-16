import numpy as np
import cvxpy as cp
import os
import time

#
# This generates polytope data for every single scenario
#
# Every scenario is saved in its own folder
#

def create_sparse_halfspace_matrix(T, delta_t=0.25):
    """Constructs the lower bidiagonal difference operator L_inv."""
    D = np.eye(T) - np.eye(T, k=-1)
    L_inv = D / delta_t
    return L_inv

def generate_heterogeneous_fleet(N, T=96, delta_t=0.25, seed=42):
    """Generates fleet bounds array h_i using detailed custom statistical profiles."""
    print(f" -> Synthesizing a heterogeneous fleet of {N} EVs over {T} horizons (Seed: {seed})...")
    np.random.seed(seed)  # Ensures strict reproducibility per scenario
    fleet_h = []
    eps = 1e-3  # Numerical cushion for idle/unplugged hours
    
    # Time steps corresponding to hours of the day (assuming 96 steps -> 1 step = 15 mins)
    steps_per_hour = int(1 / delta_t)
    
    # Assign assignment distributions to the fleet population
    group_assignments = np.random.choice([1, 2, 3], size=N, p=[0.25, 0.50, 0.25])
    
    for i in range(N):
        group = group_assignments[i]
        battery_capacity = 60.0  # kWh
        R_i = np.random.uniform(7.0, 11.0)  # Max Charging Power (kW)
        
        # 1. Determine Arrival (a_i) and Departure (d_i) steps based on group profile
        if group == 1:
            # 25% Morning drivers: 7:00 AM to 10:30 AM
            a_hour = np.random.uniform(7.0, 10.5)
            a_i = int(a_hour * steps_per_hour)
            
            # Weibull distribution: mean = 8 hours, shape (k) = 1.0 -> scale = 8.0
            dwell_hours = np.random.weibull(a=1.0) * 8.0
            d_i = min(T, a_i + int(dwell_hours * steps_per_hour))
            
        elif group == 2:
            # 50% Evening drivers: 4:00 PM to 9:00 PM
            a_hour = np.random.uniform(16.0, 21.0)
            a_i = int(a_hour * steps_per_hour)

            d_hour = np.random.uniform(a_hour, 24.0)
            d_i = int(d_hour * steps_per_hour)

            # 40% leave by evening (before midnight), 60% stay until next morning
            '''if np.random.rand() < 0.40:
                d_hour = np.random.uniform(a_hour, 24.0)
                d_i = int(d_hour * steps_per_hour)
            else:
                d_i = T'''  # Capped at end of current scheduled day horizon
                
        else:
            # 25% Generic drivers: Any time of day
            a_i = np.random.randint(0, T)
            # Triangular distribution centered at 12 hours
            dwell_hours = np.random.triangular(0.5, 12.0, 24.0)
            d_i = min(T, a_i + int(dwell_hours * steps_per_hour))
        
        # Ensure a valid duration tracking interval
        if d_i <= a_i:
            d_i = min(T, a_i + 1)
        
        stay_duration_steps = d_i - a_i
        
        # 2. Derive SOC from Gaussian distributions and process targeted Energy
        soc_init = np.clip(np.random.normal(0.25, 0.17), 0.0, 1.0)
        soc_des  = np.clip(np.random.normal(0.90, 0.17), soc_init, 1.0)
        
        requested_energy = (soc_des - soc_init) * battery_capacity
        
        # 3. Explicit Feasibility Check (Energy vs Duration & Charging rate)
        max_achievable_energy = R_i * stay_duration_steps * delta_t
        E_i = min(requested_energy, max_achievable_energy)
        
        # 4. Set Up Constraints Vectors
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
    """Exploits the analytical sparsity of Lambda_i with 1D vector representations."""
    h_0 = np.mean(fleet_h, axis=0)
    
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
        
        gamma_i_x = cp.Variable(T)
        gamma_i_diag_x = cp.Variable(T, nonneg=True)
        
        lam_31 = cp.Variable(T, nonneg=True)
        lam_32 = cp.Variable(T, nonneg=True)
        lam_41 = cp.Variable(T, nonneg=True)
        lam_42 = cp.Variable(T, nonneg=True)
        
        M_expr = cp.hstack([
            np.zeros(1),
            (gamma_i_diag_x[1:] - gamma_i_diag_x[:-1]) / delta_t
        ])
        d_expr = cp.hstack([
            gamma_i_diag_x[0:1],
            gamma_i_diag_x[:-1]
        ])
        
        constraints = [
            lam_31 - lam_32 == M_expr,
            lam_41 - lam_42 == -M_expr,
            
            cp.multiply(gamma_i_diag_x, x_0_max) <= x_i_max - gamma_i_x,
            cp.multiply(gamma_i_diag_x, x_0_min) >= x_i_min - gamma_i_x,
            
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

def run_and_save_v2_multiple_scenarios(num_scenarios=5, N=100, T=96, dt=0.25, output_dir="experimental_scenario_v2"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # L_inv is dependent only on T and dt, so we construct it once outside the loop
    L_inv = create_sparse_halfspace_matrix(T=T, delta_t=dt)
    
    # Save L_inv once in the parent directory since it does not change across scenarios
    np.save(os.path.join(output_dir, "L_inv.npy"), L_inv)
    print(f"[INFO] Global matrix L_inv saved in root directory: '{output_dir}/L_inv.npy'")
    
    for s in range(num_scenarios):
        print(f"\n**********************************************************")
        print(f" PROCESSING SCENARIO {s+1} OF {num_scenarios}")
        print(f"**********************************************************")
        
        # Vary the seed per scenario to capture new unique fleet distributions
        current_seed = 42 + s
        fleet_profiles = generate_heterogeneous_fleet(N=N, T=T, delta_t=dt, seed=current_seed)
        
        gamma_agg_diag_x, gamma_agg_x, h_0 = aggregate_flexibility_ah_polytope_structural(
            L_inv=L_inv, fleet_h=fleet_profiles, N=N, T=T, delta_t=dt
        )
        
        # Create a dedicated subdirectory for the current scenario
        scenario_dir = os.path.join(output_dir, f"scenario_{s}")
        if not os.path.exists(scenario_dir):
            os.makedirs(scenario_dir)
            
        np.save(os.path.join(scenario_dir, "gamma_agg_diag_x.npy"), gamma_agg_diag_x)
        np.save(os.path.join(scenario_dir, "gamma_agg_x.npy"), gamma_agg_x)
        np.save(os.path.join(scenario_dir, "h_0.npy"), h_0)
        
        print(f"[SUCCESS] Polytope data for scenario {s} archived successfully in: '{scenario_dir}/'")

if __name__ == "__main__":
    # Generates 5 distinct fleet scenarios over 96 time horizons, saving them in 'experimental_scenario_v2'
    run_and_save_v2_multiple_scenarios(num_scenarios=5, N=100, T=96, dt=0.25, output_dir="experimental_scenario_v2")