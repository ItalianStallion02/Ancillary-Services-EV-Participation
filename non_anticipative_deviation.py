import os
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

#
# This code generates the plot of two trajectories where:
# - one trajectory is a standard power profile with no activation planned 
# - the other trajectory is NOT anticipating a 200 MW downward activation so it does not correct it profile until the activation
#

def create_sparse_halfspace_matrix(T=96, delta_t=0.25):
    D = np.eye(T) - np.eye(T, k=-1)
    return D / delta_t

def solve_flexibility_activation_trajectories(input_dir="polytope_output", T=96, delta_t=0.25):
    # 1. Load or fall back to synthetic data
    gamma_agg_diag_x_path = os.path.join(input_dir, "gamma_agg_diag_x.npy")
    gamma_agg_x_path = os.path.join(input_dir, "gamma_agg_x.npy")
    h_0_path = os.path.join(input_dir, "h_0.npy")
    
    if not os.path.exists(h_0_path):
        # Scaled to neatly support a 200 MW sudden-onset non-anticipative problem
        x_0_max = np.ones(T) * 50000.0   # 50 MWh Max Capacity
        x_0_min = np.zeros(T)
        u_0_max = np.ones(T) * 25000.0   # 25 MW Max Power Rate
        u_0_min = np.zeros(T)
        h_0 = np.concatenate([x_0_max, -x_0_min, u_0_max, -u_0_min])
        gamma_agg_diag_x = np.ones(T) * 0.8
        gamma_agg_x = np.ones(T) * 100.0
    else:
        gamma_agg_diag_x = np.load(gamma_agg_diag_x_path)
        gamma_agg_x = np.load(gamma_agg_x_path)
        h_0 = np.load(h_0_path)
        
    x_0_max = h_0[0:T]
    x_0_min = -h_0[T:2*T]
    
    L_inv = create_sparse_halfspace_matrix(T=T, delta_t=delta_t)
    I_T = np.eye(T)
    H_tilde = np.vstack([I_T, -I_T, L_inv, -L_inv])
    
    # 2. Define tracking window
    seg_A_start = 36
    seg_A_end = seg_A_start + 8
    
    # 3. Optimization Variables
    x_0_1 = cp.Variable(T)
    x_0_2 = cp.Variable(T)
    
    # Map variables to physical aggregate expressions
    x_ch_1_expr = gamma_agg_x + cp.diag(gamma_agg_diag_x) @ x_0_1
    x_ch_2_expr = gamma_agg_x + cp.diag(gamma_agg_diag_x) @ x_0_2
    p_ch_1_expr = L_inv @ x_ch_1_expr
    p_ch_2_expr = L_inv @ x_ch_2_expr
    
    # 4. Enforce 200 MW (200,000 kW) Activation
    activation_delta_kw = 200000.0  
    
    constraints = [
        H_tilde @ x_0_1 <= h_0,
        H_tilde @ x_0_2 <= h_0,
        
        # Non-anticipation: strict overlap constraint up until the step event
        p_ch_1_expr[:seg_A_start] == p_ch_2_expr[:seg_A_start],
        
        # Enforce the target 200 MW step deviation during the window
        p_ch_1_expr[seg_A_start:seg_A_end] - p_ch_2_expr[seg_A_start:seg_A_end] == activation_delta_kw
    ]
    
    # 5. Objective Function with Smoothing Penalty
    state_distance = cp.sum_squares(x_0_1 - x_0_2)
    power_spike_penalty = 0.2 * cp.sum_squares(p_ch_1_expr - p_ch_2_expr)
    
    objective = cp.Minimize(state_distance + power_spike_penalty)
    
    # Solve Problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)
    
    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        print(f"[ERROR] Optimization failed with status: {prob.status}")
        return

    # 6. Extract optimized trajectories
    x_ch_1 = gamma_agg_x + gamma_agg_diag_x * x_0_1.value
    x_ch_2 = gamma_agg_x + gamma_agg_diag_x * x_0_2.value
    p_ch_1 = L_inv @ x_ch_1
    p_ch_2 = L_inv @ x_ch_2
    
    p_deviation = p_ch_1 - p_ch_2
    
    # Compute envelopes
    agg_x_max = (gamma_agg_diag_x * x_0_max) + gamma_agg_x
    agg_x_min = (gamma_agg_diag_x * x_0_min) + gamma_agg_x
    
    agg_u_max = np.zeros(T)
    agg_u_min = np.zeros(T)
    x_env = cp.Variable(T)
    p_env = L_inv @ (gamma_agg_x + cp.diag(gamma_agg_diag_x) @ x_env)
    env_constraints = [H_tilde @ x_env <= h_0]
    
    for t in range(T):
        prob_max = cp.Problem(cp.Maximize(p_env[t]), env_constraints)
        prob_max.solve(solver=cp.CLARABEL)
        agg_u_max[t] = prob_max.value
        
        prob_min = cp.Problem(cp.Minimize(p_env[t]), env_constraints)
        prob_min.solve(solver=cp.CLARABEL)
        agg_u_min[t] = prob_min.value

    # 7. Plotting & Display
    time_hours = np.arange(T) * delta_t
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 11))
    
    # --- Panel 1: Energy Profile ---
    ax1.fill_between(time_hours, agg_x_min, agg_x_max, color='skyblue', alpha=0.2, label='Feasible Energy Polytope')
    ax1.axvspan(seg_A_start*delta_t, seg_A_end*delta_t, color='yellow', alpha=0.15, label='200 MW Activation Window')
    ax1.plot(time_hours, x_ch_1, color='blue', linewidth=2, label='Trajectory 1 (Baseline)')
    ax1.plot(time_hours, x_ch_2, color='crimson', linewidth=1.5, linestyle='--', label='Trajectory 2 (Dispatched)')
    ax1.set_ylabel("Stored Energy (kWh)")
    ax1.set_title("Flexibility Tracking with Strict Non-Anticipation (200 MW Target)", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left')
    ax1.set_xlim(0, 24)
    ax1.set_ylim(np.min(agg_x_min) - 5000, np.max(agg_x_max) + 5000)
    
    # --- Panel 2: Power Profile ---
    ax2.fill_between(time_hours, agg_u_min, agg_u_max, color='lightgreen', alpha=0.2, label='Feasible Power Envelope')
    ax2.axvspan(seg_A_start*delta_t, seg_A_end*delta_t, color='yellow', alpha=0.15)
    ax2.plot(time_hours, p_ch_1, color='blue', linewidth=2, label='Trajectory 1 Power')
    ax2.plot(time_hours, p_ch_2, color='crimson', linewidth=1.5, linestyle='--', label='Trajectory 2 Power')
    ax2.set_ylabel("Charging Power (kW)")
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper left')
    ax2.set_xlim(0, 24)
    ax2.set_ylim(np.min(agg_u_min) - 25000, np.max(agg_u_max) + 25000)
    
    # --- Panel 3: Power Deviation Profile ---
    ax3.step(time_hours, p_deviation, where='mid', color='purple', linewidth=2.5, label='Power Deviation ($P_1 - P_2$)')
    ax3.set_xlim(0, 24)
    
    max_observed_deviation = np.max(np.abs(p_deviation))
    dynamic_limit = max(max_observed_deviation, 220000.0) * 1.20
    ax3.set_ylim(-dynamic_limit, dynamic_limit)  
    
    ax3.axvspan(seg_A_start*delta_t, seg_A_end*delta_t, color='yellow', alpha=0.2, label='Activation Window')
    ax3.axhline(200000.0, color='darkred', linestyle=':', alpha=0.7, label='Target Delta (+200 MW)')
    ax3.axhline(0.0, color='black', linestyle='-', alpha=0.4)
    
    ax3.set_title("Full-Horizon Power Deviation Profile (Forced 0 MW Diff until Hour 9)", fontsize=11, color='purple', fontweight='bold')
    ax3.set_xlabel("Time of Day (Hours)")
    ax3.set_ylabel("Power Difference (kW)")
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    solve_flexibility_activation_trajectories()