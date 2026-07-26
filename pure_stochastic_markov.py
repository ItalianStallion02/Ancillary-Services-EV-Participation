import numpy as np
import cvxpy as cp
import os

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
    price_AS_down_rebate   = da_prices * 1.2   # $/kWh — downward rebate
    return da_prices, price_AS_up_activation, price_AS_down_rebate


# ---------------------------------------------------------------------------
# 2. Activation Profile Generator (N Independent Paths)
# ---------------------------------------------------------------------------
def generate_activation_profiles(N, T, dt=0.2):
    """
    Generates N time-series activation profiles (N x T).
    - Upward Bids: 1% off-peak base up to 7% peak cap.
    - Downward Bids: 2% off-peak base up to 15% peak cap.
    - Persistence: 80% once activated.
    """
    hours = np.linspace(0, 24 - dt, T)
    morning_peak = np.exp(-0.5 * ((hours - 9.5) / 1.5) ** 2)
    evening_peak = np.exp(-0.5 * ((hours - 21.0) / 1.2) ** 2)
    peak_profile = morning_peak + evening_peak
    
    p_01_up = 0.01 + 0.06 * peak_profile  # Upward: 1% -> 7%
    p_01_dn = 0.02 + 0.13 * peak_profile  # Downward: 2% -> 15%
    p_10 = 0.20                           # 80% persistence rate
    
    delta_up = np.zeros((N, T))
    delta_dn = np.zeros((N, T))
    
    np.random.seed(42)
    for n in range(N):
        for t in range(T):
            if t == 0:
                delta_up[n, t] = 1.0 if np.random.rand() < p_01_up[0] else 0.0
                delta_dn[n, t] = 1.0 if np.random.rand() < p_01_dn[0] else 0.0
            else:
                if delta_up[n, t-1] == 0:
                    delta_up[n, t] = 1.0 if np.random.rand() < p_01_up[t] else 0.0
                else:
                    delta_up[n, t] = 0.0 if np.random.rand() < p_10 else 1.0
                    
                if delta_dn[n, t-1] == 0:
                    delta_dn[n, t] = 1.0 if np.random.rand() < p_01_dn[t] else 0.0
                else:
                    delta_dn[n, t] = 0.0 if np.random.rand() < p_10 else 1.0
                    
    return delta_up, delta_dn


# ---------------------------------------------------------------------------
# 3. Polytope Data Loader
# ---------------------------------------------------------------------------
def load_scenario_polytopes(input_root="experimental_scenario_v2"):
    L_inv = np.load(os.path.join(input_root, "L_inv.npy"))
    scenario_dirs = sorted([
        d for d in os.listdir(input_root)
        if d.startswith("scenario_") and os.path.isdir(os.path.join(input_root, d))
    ])
    if not scenario_dirs:
        raise FileNotFoundError(f"No scenario sub-directories found in '{input_root}'.")

    scenarios = []
    for sd in scenario_dirs:
        path = os.path.join(input_root, sd)
        scenarios.append({
            "name"             : sd,
            "gamma_agg_diag_x" : np.load(os.path.join(path, "gamma_agg_diag_x.npy")),
            "gamma_agg_x"      : np.load(os.path.join(path, "gamma_agg_x.npy")),
            "h_0"              : np.load(os.path.join(path, "h_0.npy")),
        })
    return scenarios, L_inv


# ---------------------------------------------------------------------------
# 4. Main Optimization Engine with Incumbent Fallback Strategy
# ---------------------------------------------------------------------------
def solve_cvar_cross_product(
    input_root    : str   = "experimental_scenario_v2",
    T             : int   = 96,
    N             : int   = 10,      # Number of market activation profiles
    C_slack       : float = 0.3,
    M             : float = 450.0,
    dt            : float = 0.2,
    H             : int   = 8,       # Minimum run-time block length
    min_bid_kw    : float = 100.0,
    alpha         : float = 0.95,
    cvar_max      : float = 1000.0,  # Max allowable CVaR limit
    time_limit_sec: float = 180.0    # Solver cutoff limit
):
    print(f"\n{'='*85}")
    print(f"  STOCHASTIC CVAR BIDDER (N={N} Profiles, CVaR_max=${cvar_max:.2f}, Cutoff={time_limit_sec}s)")
    print(f"{'='*85}\n")

    scenarios, L_inv = load_scenario_polytopes(input_root)
    K = len(scenarios)

    # Generate N Market Activation Profiles
    delta_up_nt, delta_dn_nt = generate_activation_profiles(N, T, dt)
    da_prices, price_up, price_down = build_price_vectors(T)

    I_T     = np.eye(T)
    H_tilde = np.vstack([I_T, -I_T, L_inv, -L_inv])

    # First-stage decision variables (Bids)
    p_DA   = cp.Variable(T, name="p_DA")        
    r_up   = cp.Variable(T, name="r_up")        
    r_down = cp.Variable(T, name="r_down")     
    
    u_up       = cp.Variable(T, boolean=True, name="u_up")       
    v_up_start = cp.Variable(T, boolean=True, name="v_up_start") 
    v_up_end   = cp.Variable(T, boolean=True, name="v_up_end")   
    
    u_dn       = cp.Variable(T, boolean=True, name="u_dn")       
    v_dn_start = cp.Variable(T, boolean=True, name="v_dn_start") 
    v_dn_end   = cp.Variable(T, boolean=True, name="v_dn_end")   

    # Second-stage recourse variables (Dimensions: K x N x T)
    p_ch    = cp.Variable((K, N, T), name="p_ch")
    s_plus  = cp.Variable((K, N, T), name="s_plus",  nonneg=True)
    s_minus = cp.Variable((K, N, T), name="s_minus", nonneg=True)
    x_0     = cp.Variable((K, N, T), name="x_0")

    zeta    = cp.Variable(name="VaR_threshold_zeta")              
    psi     = cp.Variable((K, N), nonneg=True, name="tail_psi")  

    constraints = [
        p_DA >= 0.0,
        r_up <= p_DA,
        p_DA + r_down <= M,
    ]

    # Commitment Logic & Minimum Run-Time Block Length (H)
    for t in range(T):
        if t == 0:
            constraints.append(u_up[t] == v_up_start[t])
            constraints.append(u_dn[t] == v_dn_start[t])
        else:
            constraints.append(u_up[t] - u_up[t-1] == v_up_start[t] - v_up_end[t])
            constraints.append(u_dn[t] - u_dn[t-1] == v_dn_start[t] - v_dn_end[t])

        constraints.append(v_up_start[t] + v_up_end[t] <= 1)
        constraints.append(v_dn_start[t] + v_dn_end[t] <= 1)
        constraints.append(u_up[t] + u_dn[t] <= 1)

        constraints.append(r_up[t] >= min_bid_kw * u_up[t])
        constraints.append(r_up[t] <= M * u_up[t])
        constraints.append(r_down[t] >= min_bid_kw * u_dn[t])
        constraints.append(r_down[t] <= M * u_dn[t])

        look_ahead = min(t + H, T)
        for k_h in range(t, look_ahead):
            constraints.append(u_up[k_h] >= v_up_start[t])
            constraints.append(u_dn[k_h] >= v_dn_start[t])

        if t < T - 1:
            is_transition_up = v_up_start[t+1] + v_up_end[t+1]
            is_transition_dn = v_dn_start[t+1] + v_dn_end[t+1]
            constraints.extend([
                r_up[t] - r_up[t+1] <= M * is_transition_up,
                r_up[t+1] - r_up[t] <= M * is_transition_up,
                r_down[t] - r_down[t+1] <= M * is_transition_dn,
                r_down[t+1] - r_down[t] <= M * is_transition_dn,
            ])

    da_energy_cost = cp.sum(cp.multiply(da_prices, p_DA)) * dt
    expected_recourse_cost = 0.0
    joint_prob = 1.0 / (K * N)

    # Outer Loop: Physical Polytopes (K)
    for k, sc in enumerate(scenarios):
        gamma_x        = sc["gamma_agg_x"]       
        gamma_diag_x   = sc["gamma_agg_diag_x"]  
        h_0_k          = sc["h_0"]               
        Gamma_diag_mat = np.diag(gamma_diag_x)  

        # Inner Loop: Market Activation Profiles (N)
        for n in range(N):
            constraints.append(
                p_ch[k, n, :] == p_DA 
                                 + cp.multiply(delta_dn_nt[n, :], r_down) 
                                 - cp.multiply(delta_up_nt[n, :], r_up) 
                                 + s_plus[k, n, :] - s_minus[k, n, :]
            )

            x_ch_kn = gamma_x + Gamma_diag_mat @ x_0[k, n, :]
            constraints.append(p_ch[k, n, :] == L_inv @ x_ch_kn)
            constraints.append(H_tilde @ x_0[k, n, :] <= h_0_k)

            # Cashflows for joint branch (k, n)
            slack_penalty = C_slack * cp.sum(s_plus[k, n, :] + s_minus[k, n, :]) * dt
            added_cost    = cp.sum(cp.multiply(da_prices,  cp.multiply(delta_dn_nt[n, :], r_down))) * dt
            down_rebate   = cp.sum(cp.multiply(price_down, cp.multiply(delta_dn_nt[n, :], r_down))) * dt
            saved_credit  = cp.sum(cp.multiply(da_prices,  cp.multiply(delta_up_nt[n, :], r_up))) * dt
            up_revenue   = cp.sum(cp.multiply(price_up,   cp.multiply(delta_up_nt[n, :], r_up))) * dt

            kn_cost = slack_penalty + added_cost - down_rebate - saved_credit - up_revenue
            expected_recourse_cost += joint_prob * kn_cost

            total_branch_cost = da_energy_cost + kn_cost
            constraints.append(psi[k, n] >= total_branch_cost - zeta)

    # CVaR Formulation over K x N Joint Scenarios
    cvar_expression     = zeta + (1.0 / (1.0 - alpha)) * (joint_prob * cp.sum(psi))
    total_expected_cost = da_energy_cost + expected_recourse_cost

    constraints.append(cvar_expression <= cvar_max)

    problem = cp.Problem(cp.Minimize(total_expected_cost), constraints)
    
    highs_options = {
        'mip_rel_gap': 0.01,
        'primal_feasibility_tolerance': 1e-3,
        'time_limit': float(time_limit_sec)
    }

    print("=" * 85)
    print(f"  RUNNING SOLVER ({K} Polytopes x {N} Profiles = {K*N} Total Joint Scenarios)")
    print("=" * 85)

    try:
        problem.solve(
            solver=cp.HIGHS, 
            verbose=True, 
            highs_options=highs_options
        )
    except Exception as e:
        print(f"\n[WARNING] Solver interrupted or threw exception: {e}")

    print("\n" + "=" * 85)
    
    # Check if a feasible solution (Incumbent) exists
    has_incumbent = p_DA.value is not None

    if problem.status == cp.OPTIMAL:
        print("  SOLVER STATUS: OPTIMAL (Target MIP Gap Reached)")
    elif problem.status in [cp.USER_LIMIT, cp.OPTIMAL_INACCURATE] or has_incumbent:
        print(f"  SOLVER STATUS: {problem.status} (USING FEASIBLE INCUMBENT SOLUTION)")
    else:
        print(f"  [ERROR] Optimization failed with status '{problem.status}' and no incumbent solution was found.")
        return

    # ---------------------------------------------------------------------------
    # Summary Metrics Output
    # ---------------------------------------------------------------------------
    print("=" * 85)
    print("  CVAR CONSTRAINED METRICS SUMMARY")
    print("=" * 85)
    print(f"  Expected Net Cost           : ${total_expected_cost.value:10.2f}")
    print(f"  Value-at-Risk (VaR, ζ)       : ${zeta.value:10.2f}")
    print(f"  Conditional VaR (CVaR)       : ${cvar_expression.value:10.2f} (Max Limit: ${cvar_max:.2f})")
    print("=" * 85 + "\n")

    # ---------------------------------------------------------------------------
    # Period-by-Period Schedule Output
    # ---------------------------------------------------------------------------
    print("  OPTIMIZED DAY-AHEAD BASELINE & ANCILLARY SERVICE OFFERS")
    print("=" * 85)
    print(f"{'Period':<7} | {'Hour':<6} | {'DA Base (kW)':<12} | {'Market':<10} | {'Action':<8} | {'Up Bid (kW)':<11} | {'Down Bid (kW)':<11}")
    print("-" * 85)
    
    for t in range(T):
        hour_val = t * dt
        hour_str = f"{int(hour_val):02d}:{int((hour_val % 1) * 60):02d}"

        if u_up.value[t] > 0.5:
            mkt = "UPWARD"
            action = "START" if v_up_start.value[t] > 0.5 else ("END" if v_up_end.value[t] > 0.5 else "HOLD")
        elif u_dn.value[t] > 0.5:
            mkt = "DOWNWARD"
            action = "START" if v_dn_start.value[t] > 0.5 else ("END" if v_dn_end.value[t] > 0.5 else "HOLD")
        else:
            mkt = "IDLE"
            action = "-"
            
        print(f"Q-{t:02d}    | {hour_str:<6} | {p_DA.value[t]:12.2f} | {mkt:<10} | {action:<8} | "
              f"{r_up.value[t]:11.2f} | {r_down.value[t]:11.2f}")


if __name__ == "__main__":
    solve_cvar_cross_product(N=10, cvar_max=2000.0, time_limit_sec=180.0)