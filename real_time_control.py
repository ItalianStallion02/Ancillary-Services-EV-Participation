import numpy as np
import cvxpy as cp
import time
import os

# ---------------------------------------------------------------------------
# 1. Fleet Profile Synthesizer (Matches Polytope Generation Philosophy)
# ---------------------------------------------------------------------------
def generate_heterogeneous_fleet_mpc(N=100, T=96, delta_t=0.25, seed=35):
    """
    Generates operational bounds and charging requirements for N individual EVs.
    """
    print(f"-> Synthesizing a real-time fleet scenario of {N} EVs over {T} horizons...")
    np.random.seed(seed)
    fleet = []
    eps = 1e-3
    steps_per_hour = int(1 / delta_t)
    group_assignments = np.random.choice([1, 2, 3], size=N, p=[0.25, 0.50, 0.25])
    
    for i in range(N):
        group = group_assignments[i]
        battery_capacity = 60.0  # kWh
        R_i = np.random.uniform(7.0, 11.0)  # Max Power (kW)
        
        if group == 1:
            a_hour = np.random.uniform(7.0, 10.5)
            a_i = int(a_hour * steps_per_hour)
            dwell_hours = np.random.weibull(a=1.0) * 8.0
            d_i = min(T, a_i + int(dwell_hours * steps_per_hour))
        elif group == 2:
            a_hour = np.random.uniform(16.0, 21.0)
            a_i = int(a_hour * steps_per_hour)
            d_hour = np.random.uniform(a_hour, 24.0)
            d_i = int(d_hour * steps_per_hour)
        else:
            a_i = np.random.randint(0, T)
            dwell_hours = np.random.triangular(0.5, 12.0, 24.0)
            d_i = min(T, a_i + int(dwell_hours * steps_per_hour))
            
        if d_i <= a_i:
            d_i = min(T, a_i + 1)
            
        stay_duration_steps = d_i - a_i
        
        soc_init = np.clip(np.random.normal(0.25, 0.17), 0.0, 1.0)
        soc_des = np.clip(np.random.normal(0.90, 0.17), soc_init, 1.0)
        requested_energy = (soc_des - soc_init) * battery_capacity
        
        max_achievable_energy = R_i * stay_duration_steps * delta_t
        E_i = min(requested_energy, max_achievable_energy - eps)
        
        u_max = np.ones(T) * eps
        u_min = np.ones(T) * -eps
        u_max[a_i:d_i] = R_i
        u_min[a_i:d_i] = 0.0
        
        x_max = np.ones(T) * battery_capacity
        x_min = np.zeros(T)
        if d_i < T:
            x_min[d_i-1:] = E_i
        else:
            x_min[-1] = E_i
            
        fleet.append({
            'u_max': u_max, 'u_min': u_min,
            'x_max': x_max, 'x_min': x_min,
            'E_target': E_i, 'd_i': d_i
        })
    return fleet

# ---------------------------------------------------------------------------
# 2. Day-Ahead Commitment Loader
# ---------------------------------------------------------------------------
def load_day_ahead_schedule(input_dir="day_ahead_data"):
    """
    Loads saved day-ahead baseline power and AS capacities.
    """
    print(f"-> Loading Day-Ahead market schedules from folder '{input_dir}'...")
    p_DA = np.load(os.path.join(input_dir, "p_DA.npy"))
    r_up = np.load(os.path.join(input_dir, "r_up.npy"))
    r_down = np.load(os.path.join(input_dir, "r_down.npy"))
    
    # Validation check
    assert np.sum(r_up * r_down) == 0, "[WARNING] Time intervals for r_up and r_down overlap!"
    return p_DA, r_up, r_down

# ---------------------------------------------------------------------------
# 3. Main Rolling Horizon MPC Tracking Engine (OPTIMIZED)
# ---------------------------------------------------------------------------
def run_real_time_mpc(fleet, p_DA, r_up, r_down, lambda_imb=1.5, lambda_pen=4.0, T=96, dt=0.25):
    print(f"\n{'='*70}")
    print(f"         RUNNING CLOSED-LOOP REAL-TIME MPC FLEET CONTROLLER")
    print(f"{'='*70}")
    
    start_total_time = time.time()
    
    np.random.seed(7)
    rt_deployment_shocks = np.clip(np.random.normal(0, 0.4, size=T), -1.0, 1.0)
    
    N_evs = len(fleet)
    x_actual_past = np.zeros(N_evs)  
    
    u_max_full = np.vstack([ev['u_max'] for ev in fleet])
    u_min_full = np.vstack([ev['u_min'] for ev in fleet])
    x_max_full = np.vstack([ev['x_max'] for ev in fleet])
    x_min_full = np.vstack([ev['x_min'] for ev in fleet])
    
    history_u_total = []
    history_p_target = []
    history_imbalance = []
    history_penalty = []
    
    for t in range(T):
        rem_len = T - t  
        shock = rt_deployment_shocks[t]
        
        r_up_act_t = -shock * r_up[t] if (shock < 0 and r_up[t] > 0) else 0.0
        r_down_act_t = shock * r_down[t] if (shock > 0 and r_down[t] > 0) else 0.0
        p_target_reference = p_DA[t] + r_down_act_t - r_up_act_t
        
        r_up_ref_horizon = np.zeros(rem_len)
        r_down_ref_horizon = np.zeros(rem_len)
        
        r_up_ref_horizon[0] = r_up_act_t
        r_down_ref_horizon[0] = r_down_act_t
        if rem_len > 1:
            r_up_ref_horizon[1:] = r_up[t+1:T]
            r_down_ref_horizon[1:] = r_down[t+1:T]
            
        u_var = cp.Variable((N_evs, rem_len), name=f"u_step_{t}")
        x_var = cp.Variable((N_evs, rem_len), name=f"x_step_{t}")
        
        s_plus = cp.Variable(rem_len, nonneg=True)
        s_minus = cp.Variable(rem_len, nonneg=True)
        r_up_var = cp.Variable(rem_len, nonneg=True)
        r_down_var = cp.Variable(rem_len, nonneg=True)
        
        u_max_slice = u_max_full[:, t:T]
        u_min_slice = u_min_full[:, t:T]
        x_max_slice = x_max_full[:, t:T]
        x_min_slice = x_min_full[:, t:T]
        
        constraints = [
            u_var >= u_min_slice,
            u_var <= u_max_slice,
            x_var >= x_min_slice,
            x_var <= x_max_slice,
            
            x_var == cp.cumsum(u_var * dt, axis=1) + cp.reshape(x_actual_past, (N_evs, 1)),
            
            r_up_var <= r_up_ref_horizon,
            r_down_var <= r_down_ref_horizon,
            
            cp.sum(u_var, axis=0) == p_DA[t:T] + r_down_var - r_up_var + s_plus - s_minus
        ]
        
        obj_expr = cp.sum(lambda_imb * (s_plus + s_minus)) + \
                   cp.sum(lambda_pen * (r_up_ref_horizon - r_up_var)) + \
                   cp.sum(lambda_pen * (r_down_ref_horizon - r_down_var))
                   
        prob = cp.Problem(cp.Minimize(obj_expr), constraints)
        prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-4, tol_gap_rel=1e-4)
        
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print(f"[CRITICAL] Operational failure at interval {t}. MPC Status: {prob.status}")
            break
            
        u_opt_t = u_var[:, 0].value
        u_total_implemented = np.sum(u_opt_t)
        
        x_actual_past += u_opt_t * dt
        
        imb_err = abs(u_total_implemented - p_target_reference)
        pen_val = (r_up_ref_horizon[0] - r_up_var[0].value) + (r_down_ref_horizon[0] - r_down_var[0].value)
        
        history_u_total.append(u_total_implemented)
        history_p_target.append(p_target_reference)
        history_imbalance.append(imb_err)
        history_penalty.append(max(0.0, pen_val))
        
        print(f"Interval Q-{t:02d} | Target Ref: {p_target_reference:6.1f} kW | "
              f"Dispatched: {u_total_implemented:6.1f} kW | "
              f"Imbalance Error: {history_imbalance[-1]:5.2f} kW")
            
    total_execution_time = time.time() - start_total_time
    print(f"\n{'='*70}")
    print(f"                  MPC PERFORMANCE SUMMARY REPORT")
    print(f"{'='*70}")
    print(f"Total Framework Execution Time:         {total_execution_time:.3f} seconds")
    print(f"Total Daily Energy Imbalance Violation: {sum(history_imbalance) * dt:.2f} kWh")
    print(f"Total Daily Service Delivery Penalties: {sum(history_penalty):.2f} kW-steps")
    print(f"Max Instantaneous Imbalance Spike:      {max(history_imbalance):.2f} kW")
    print(f"{'='*70}\n")

# ---------------------------------------------------------------------------
# 4. Execution Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    T_HORIZON = 96
    TIME_STEP = 0.25
    NUM_EVS = 100
    
    IMBALANCE_FEE = 1.80  
    PENALTY_FEE   = 4.50  
    
    ev_fleet = generate_heterogeneous_fleet_mpc(N=NUM_EVS, T=T_HORIZON, delta_t=TIME_STEP)
    
    # Load optimized schedules from Day-Ahead script
    try:
        p_DA, r_up, r_down = load_day_ahead_schedule(input_dir="day_ahead_data")
    except FileNotFoundError:
        print("[ERROR] Could not find 'day_ahead_data'. Please run the Day-Ahead script first!")
        exit(1)
    
    run_real_time_mpc(
        fleet=ev_fleet, p_DA=p_DA, r_up=r_up, r_down=r_down,
        lambda_imb=IMBALANCE_FEE, lambda_pen=PENALTY_FEE,
        T=T_HORIZON, dt=TIME_STEP
    )