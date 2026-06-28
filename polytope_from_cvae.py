import os
import numpy as np
import cvxpy as cp
import time
import matplotlib.pyplot as plt

def create_sparse_halfspace_matrix(T=96, delta_t=0.25):
    """Constructs the lower bidiagonal difference operator L_inv."""
    D = np.eye(T) - np.eye(T, k=-1)
    L_inv = D / delta_t
    return L_inv

def select_real_world_charging_speed(energy_kwh, arrival_step, departure_step, delta_t=0.25):
    """
    Selects the minimum standard real-world charging speed that guarantees 
    the energy request is feasible within the vehicle's dwell time.
    """
    dwell_hours = (departure_step - arrival_step) * delta_t
    if dwell_hours <= 0:
        return 7.4
    
    required_power = energy_kwh / dwell_hours
    real_world_tiers = [7.4, 11.0, 22.0, 50.0, 150.0, 350.0]
    for speed in real_world_tiers:
        if speed >= required_power:
            return speed
    return required_power

def format_cvae_profiles_for_polytope(fleet_real, T=96, delta_t=0.25):
    """
    Converts raw continuous cVAE profiles [Arrival, Energy, Departure] into 
    the bidiagonal halfspace structural matrix format (fleet_h).
    """
    fleet_h = []
    eps = 1e-3
    
    for i in range(len(fleet_real)):
        a_hour = fleet_real[i, 0]
        E_i = max(0.0, fleet_real[i, 1])
        d_hour = fleet_real[i, 2]
        
        a_i = int(np.clip(a_hour / delta_t, 0, T - 1))
        d_i = int(np.clip(d_hour / delta_t, a_i + 1, T))
        
        R_i = select_real_world_charging_speed(E_i, a_i, d_i, delta_t)
        battery_capacity = max(60.0, E_i * 1.1)
        
        u_max, u_min = np.ones(T) * eps, np.ones(T) * -eps
        u_max[a_i:d_i], u_min[a_i:d_i] = R_i, 0.0
        
        x_max, x_min = np.ones(T) * battery_capacity, np.zeros(T)
        if d_i < T:
            x_min[d_i-1:] = E_i
        else:
            x_min[-1] = E_i
            
        fleet_h.append(np.concatenate([x_max, -x_min, u_max, -u_min]))
        
    return np.array(fleet_h)

def aggregate_flexibility_ah_polytope_structural(L_inv, fleet_h, agent_weights, N, T, delta_t=0.25):
    """Exploits matrix structures to compute aggregate system polytopes, incorporating agent volumetric scaling weights."""
    h_0 = np.mean(fleet_h, axis=0)
    
    x_0_max = h_0[0:T]
    x_0_min = -h_0[T:2*T]
    u_0_max = h_0[2*T:3*T]
    u_0_min = -h_0[3*T:4*T]
    
    gamma_agg_x = np.zeros(T)
    gamma_agg_diag_x = np.zeros(T)
    
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
        
        M_expr = cp.hstack([np.zeros(1), (gamma_i_diag_x[1:] - gamma_i_diag_x[:-1]) / delta_t])
        d_expr = cp.hstack([gamma_i_diag_x[0:1], gamma_i_diag_x[:-1]])
        
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
            raise ValueError(f"Sub-problem optimization failed at internal driver profiling tracking index: {i+1}")
            
        # =====================================================================
        # CRITICAL VOLUMETRIC SCALE FIX: 
        # Multiply individual parameters by the true number of EVs represented by this agent profile
        # =====================================================================
        gamma_agg_x += agent_weights[i] * gamma_i_x.value
        gamma_agg_diag_x += agent_weights[i] * gamma_i_diag_x.value
        
    return gamma_agg_diag_x, gamma_agg_x, h_0

def plot_aggregate_polytope(gamma_agg_diag_x, gamma_agg_x, h_0, T=96, delta_t=0.25):
    """
    Generates a clear 2-panel figure showing the aggregate State of Charge (Energy) 
    and Charging Rate (Power) boundaries over a 24-hour horizon.
    """
    x_0_max = h_0[0:T]
    x_0_min = -h_0[T:2*T]
    u_0_max = h_0[2*T:3*T]
    u_0_min = -h_0[3*T:4*T]
    
    # Calculate scaled inner-approximated polytope hyperplanes
    agg_x_max = (gamma_agg_diag_x * x_0_max) + gamma_agg_x
    agg_x_min = (gamma_agg_diag_x * x_0_min) + gamma_agg_x
    
    L_inv = create_sparse_halfspace_matrix(T=T, delta_t=delta_t)
    agg_u_max = (gamma_agg_diag_x * u_0_max) + (L_inv @ gamma_agg_x)
    agg_u_min = (gamma_agg_diag_x * u_0_min) + (L_inv @ gamma_agg_x)
    
    time_hours = np.arange(T) * delta_t
    
    # --- Subplot 1: Energy Boundary Polytope (State of Charge) ---
    plt.subplot(2, 1, 1)
    plt.fill_between(time_hours, agg_x_min, agg_x_max, color='skyblue', alpha=0.4, label='Feasible Energy Space')
    plt.plot(time_hours, agg_x_max, color='blue', linestyle='--', linewidth=1.5, label='Max Aggregate Energy Target ($X_{max}$)')
    plt.plot(time_hours, agg_x_min, color='navy', linestyle='--', linewidth=1.5, label='Min Aggregate Energy Target ($X_{min}$)')
    plt.title("Aggregate EV Fleet Virtual Battery Flexibility Polytope", fontsize=12, fontweight='bold')
    plt.ylabel("Stored Energy Capacity (kWh)", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left')
    plt.xlim(0, 24)
    
    # --- Subplot 2: Power Boundary Polytope (Charging Rates) ---
    plt.subplot(2, 1, 2)
    plt.fill_between(time_hours, agg_u_min, agg_u_max, color='salmon', alpha=0.4, label='Feasible Power Capacity')
    plt.plot(time_hours, agg_u_max, color='red', linestyle='-', linewidth=1.5, label='Max Power Charging Limit ($U_{max}$)')
    plt.plot(time_hours, agg_u_min, color='darkred', linestyle='-', linewidth=1.5, label='Min Power Charging Limit ($U_{min}$)')
    plt.xlabel("Time of Day (Hours)", fontsize=10)
    plt.ylabel("Charging Power Limit (kW)", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left')
    plt.xlim(0, 24)
    
    plt.tight_layout()
    plt.savefig('fleet_polytope_flexibility.png', dpi=300)
    print("[VISUALIZATION] Complete flexibility diagram saved successfully as 'fleet_polytope_flexibility.png'!")

if __name__ == "__main__":
    input_dir = "cvae_data"
    output_dir = "polytope_output"
    T, dt = 96, 0.25
    
    profiles_path = os.path.join(input_dir, "fleet_real_profiles.npy")
    if not os.path.exists(profiles_path):
        raise FileNotFoundError(f"Target arrays not available at '{profiles_path}'. Execute cvae_stratified_sampling.py first.")
        
    print(f"\n[POLYTOPE LOADING LAYER] Sourcing compiled matrix details from folder: '{input_dir}/'")
    fleet_real = np.load(profiles_path)
    agent_weights = np.load(os.path.join(input_dir, "agent_weights.npy"))
    
    N_profiles = len(fleet_real)
    print(f"[POLYTOPE LOADING LAYER] Recovered {N_profiles} distinct vehicle arrays.")
    
    L_inv = create_sparse_halfspace_matrix(T=T, delta_t=dt)
    fleet_h = format_cvae_profiles_for_polytope(fleet_real, T=T, delta_t=dt)
    
    print("[POLYTOPE LOADING LAYER] Launching Inner-Approximation Convex Solvers...")
    start_time = time.time()
    gamma_agg_diag_x, gamma_agg_x, h_0 = aggregate_flexibility_ah_polytope_structural(
        L_inv=L_inv, fleet_h=fleet_h, agent_weights=agent_weights, N=N_profiles, T=T, delta_t=dt
    )
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    np.save(os.path.join(output_dir, "gamma_agg_diag_x.npy"), gamma_agg_diag_x)
    np.save(os.path.join(output_dir, "gamma_agg_x.npy"), gamma_agg_x)
    np.save(os.path.join(output_dir, "h_0.npy"), h_0)
    print(f"[SUCCESS] Polytope system completed successfully in {time.time() - start_time:.2f}s!")
    
    # --- Generating & Exporting the Flexibility Graph ---
    plot_aggregate_polytope(gamma_agg_diag_x, gamma_agg_x, h_0, T=T, delta_t=dt)