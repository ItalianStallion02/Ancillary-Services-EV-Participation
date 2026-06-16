import numpy as np
import cvxpy as cp
import os

#
# Best version (at the moment)
#
# Works fine with optimized polytope
#  
# Current polytopes are taken from folder 'experimental_scenario_v2'
#
#

# ---------------------------------------------------------------------------
# 1. Price Data Builder
# ---------------------------------------------------------------------------
def build_price_vectors(T):
    da_prices_24 = np.array([
        0.04, 0.03, 0.03, 0.02, 0.03, 0.05,
        0.07, 0.09, 0.08, 0.06, 0.05, 0.05,
        0.04, 0.04, 0.05, 0.06, 0.08, 0.11,
        0.12, 0.10, 0.08, 0.07, 0.05, 0.04
    ])
    da_prices              = np.repeat(da_prices_24, T // 24)
    price_AS_up_activation = da_prices * 1.5   # $/kWh — upward activation premium
    price_AS_down_rebate   = da_prices * 0.40   # $/kWh — downward rebate
    return da_prices, price_AS_up_activation, price_AS_down_rebate


# ---------------------------------------------------------------------------
# 2. Polytope Data Loader
# ---------------------------------------------------------------------------
def load_scenario_polytopes(input_root="experimental_scenario_v2"):
    """
    Loads all scenario sub-directories found under `input_root`.
    """
    L_inv = np.load(os.path.join(input_root, "L_inv.npy"))

    scenario_dirs = sorted([
        d for d in os.listdir(input_root)
        if d.startswith("scenario_") and
           os.path.isdir(os.path.join(input_root, d))
    ])

    if not scenario_dirs:
        raise FileNotFoundError(
            f"No scenario sub-directories found in '{input_root}'. "
            "Run scenario_polytope_generator.py first."
        )

    scenarios = []
    for sd in scenario_dirs:
        path = os.path.join(input_root, sd)
        scenarios.append({
            "name"             : sd,
            "gamma_agg_diag_x" : np.load(os.path.join(path, "gamma_agg_diag_x.npy")),
            "gamma_agg_x"      : np.load(os.path.join(path, "gamma_agg_x.npy")),
            "h_0"              : np.load(os.path.join(path, "h_0.npy")),
        })
        print(f"  Loaded polytope: {sd}")

    return scenarios, L_inv


# ---------------------------------------------------------------------------
# 3. Main Optimization Engine (With Block Market Constraints)
# ---------------------------------------------------------------------------
def solve_stochastic_joint_bidding(
    input_root          : str   = "scenario_polytopes",
    T                   : int   = 96,
    num_deployment_draws: int   = 5,
    C_slack             : float = 10.0,
    M                   : float = 200.0,
    dt                  : float = 0.25,
    deployment_seed     : int   = 7,
    H                   : int   = 8,      # Minimum commitment duration (4 periods = 1 hour)
    min_bid_kw          : float = 75.0,   # Minimum bid requirement
):
    print(f"\n{'='*60}")
    print(f"  STOCHASTIC MULTI-SCENARIO MARKET BIDDER")
    print(f"{'='*60}")
    
    # Load data via internal functions
    scenarios, L_inv = load_scenario_polytopes(input_root)

    K = len(scenarios)
    S = num_deployment_draws
    prob_ks = 1.0 / (K * S)

    da_prices, price_up, price_down = build_price_vectors(T)

    I_T     = np.eye(T)
    H_tilde = np.vstack([I_T, -I_T, L_inv, -L_inv])

    rng              = np.random.default_rng(deployment_seed)
    deployment_draws = rng.uniform(0.5, 1.0, size=S)

    # Decision variables
    p_DA   = cp.Variable(T, name="p_DA")        
    r_up   = cp.Variable(T, name="r_up")        
    r_down = cp.Variable(T, name="r_down")     
    
    # State Indicators
    u_up       = cp.Variable(T, boolean=True, name="u_up")       
    v_up_start = cp.Variable(T, boolean=True, name="v_up_start") 
    v_up_end   = cp.Variable(T, boolean=True, name="v_up_end")   
    
    u_dn       = cp.Variable(T, boolean=True, name="u_dn")       
    v_dn_start = cp.Variable(T, boolean=True, name="v_dn_start") 
    v_dn_end   = cp.Variable(T, boolean=True, name="v_dn_end")   

    # Stochastic profile trackers
    p_ch    = cp.Variable((K, S, T), name="p_ch")
    s_plus  = cp.Variable((K, S, T), name="s_plus",  nonneg=True)
    s_minus = cp.Variable((K, S, T), name="s_minus", nonneg=True)
    x_0     = cp.Variable((K, S, T), name="x_0")

    constraints = [
        p_DA >= 0.0,
        r_up <= p_DA,
        p_DA + r_down <= M,
    ]

    for t in range(T):
        # A. State evolution logic
        if t == 0:
            constraints.append(u_up[t] == v_up_start[t])
            constraints.append(u_dn[t] == v_dn_start[t])
        else:
            constraints.append(u_up[t] - u_up[t-1] == v_up_start[t] - v_up_end[t])
            constraints.append(u_dn[t] - u_dn[t-1] == v_dn_start[t] - v_dn_end[t])

        # B. Bid cannot start and end in the same interval
        constraints.append(v_up_start[t] + v_up_end[t] <= 1)
        constraints.append(v_dn_start[t] + v_dn_end[t] <= 1)

        # C. Mutual exclusion
        constraints.append(u_up[t] + u_dn[t] <= 1)

        # D. Min Capacity & Strict Zeroing
        constraints.append(r_up[t] >= min_bid_kw * u_up[t])
        constraints.append(r_up[t] <= M * u_up[t])
        
        constraints.append(r_down[t] >= min_bid_kw * u_dn[t])
        constraints.append(r_down[t] <= M * u_dn[t])

        # E. Minimum Duration H
        look_ahead = min(t + H, T)
        for k in range(t, look_ahead):
            constraints.append(u_up[k] >= v_up_start[t])
            constraints.append(u_dn[k] >= v_dn_start[t])

        # F. Constant bid profile over an active sequence
        if t < T - 1:
            slack_up = M * (2.0 - u_up[t] - u_up[t+1])
            slack_dn = M * (2.0 - u_dn[t] - u_dn[t+1])

            constraints.extend([
                r_up[t] - r_up[t+1] <= slack_up,
                r_up[t+1] - r_up[t] <= slack_up,
                r_down[t] - r_down[t+1] <= slack_dn,
                r_down[t+1] - r_down[t] <= slack_dn,
            ])

    # Scenario ledger loops
    expected_cost = 0.0
    for k, sc in enumerate(scenarios):
        gamma_x      = sc["gamma_agg_x"]       
        gamma_diag_x = sc["gamma_agg_diag_x"]  
        h_0_k        = sc["h_0"]               
        Gamma_diag_mat = np.diag(gamma_diag_x)  

        for s_idx, delta_s in enumerate(deployment_draws):
            constraints.append(
                p_ch[k, s_idx, :] ==
                    p_DA + r_down * delta_s - r_up * delta_s
                    + s_plus[k, s_idx, :] - s_minus[k, s_idx, :]
            )

            x_ch_ks = gamma_x + Gamma_diag_mat @ x_0[k, s_idx, :]
            constraints.append(p_ch[k, s_idx, :] == L_inv @ x_ch_ks)
            constraints.append(H_tilde @ x_0[k, s_idx, :] <= h_0_k)

            slack_penalty = C_slack * cp.sum(s_plus[k, s_idx, :] + s_minus[k, s_idx, :]) * dt
            added_cost   = cp.sum(cp.multiply(da_prices,  r_down * delta_s)) * dt
            down_rebate  = cp.sum(cp.multiply(price_down, r_down * delta_s)) * dt
            saved_credit = cp.sum(cp.multiply(da_prices,  r_up   * delta_s)) * dt
            up_revenue   = cp.sum(cp.multiply(price_up,   r_up   * delta_s)) * dt

            ks_cost = slack_penalty + added_cost - down_rebate - saved_credit - up_revenue
            expected_cost += prob_ks * ks_cost

    # Objective compilation
    da_energy_cost = cp.sum(cp.multiply(da_prices, p_DA)) * dt
    objective      = cp.Minimize(da_energy_cost + expected_cost)
    problem        = cp.Problem(objective, constraints)

    print("="*60)
    print(" INVOKING HiGHS SOLVER WITH COMPLEX MARKET CONTROLS ")
    print("="*60)
    problem.solve(solver=cp.HIGHS, verbose=True, highs_options={'mip_rel_gap': 0.05, 'primal_feasibility_tolerance': 1e-3})
    print("="*60 + "\n")

    if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        print(f"[ERROR] Optimisation failed. Status: {problem.status}")
        return

    print("=== Optimisation Successful ===")
    print(f"Total Expected Net Operating Cost: ${problem.value:.2f}\n")

    print(f"{'Int':<5} | {'DA Base (kW)':<12} | {'Market':<10} | {'Action':<8} | {'Up Bid (kW)':<11} | {'Down Bid (kW)':<11}")
    print("-" * 75)
    for t in range(T):
        if u_up.value[t] > 0.5:
            mkt = "UPWARD"
            action = "START" if v_up_start.value[t] > 0.5 else ("END" if v_up_end.value[t] > 0.5 else "HOLD")
        elif u_dn.value[t] > 0.5:
            mkt = "DOWNWARD"
            action = "START" if v_dn_start.value[t] > 0.5 else ("END" if v_dn_end.value[t] > 0.5 else "HOLD")
        else:
            mkt = "IDLE"
            action = "-"
            
        print(f"Q-{t:02d} | {p_DA.value[t]:12.2f} | {mkt:<10} | {action:<8} | "
              f"{r_up.value[t]:11.2f} | {r_down.value[t]:11.2f}")


# ---------------------------------------------------------------------------
# 4. Standard Python Execution Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    solve_stochastic_joint_bidding(
        input_root           = "experimental_scenario_v2",
        T                    = 96,
        num_deployment_draws = 5,
        C_slack              = 1,
        M                    = 410,
        dt                   = 0.25,
        deployment_seed      = 7,
        H                    = 8,   # 4 quarters = 1 hour minimum block run
        min_bid_kw           = 100.0 # 50 kW minimum size
    )