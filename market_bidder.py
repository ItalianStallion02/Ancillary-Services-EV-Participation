# market_bidder.py
import numpy as np
import cvxpy as cp
import os

def solve_stochastic_joint_bidding(input_dir="saved_polytope", T=96, num_scenarios=5):
    """
    Executes a joint optimization dispatch for both Upward and Downward ancillary services.
    Uses a stochastic scenario loop to compute physical slacks under deployment uncertainty
    and enforces directional mutual exclusion using master binary variables via bundled ECOS_BB.
    """
    print(f"\n[DEBUG] Loading pre-calculated polytope constraints from '{input_dir}' directory...")
    
    # Load external state parameters
    try:
        gamma_agg_diag_x = np.load(os.path.join(input_dir, "gamma_agg_diag_x.npy"))
        gamma_agg_x = np.load(os.path.join(input_dir, "gamma_agg_x.npy"))
        h_0 = np.load(os.path.join(input_dir, "h_0.npy"))
        L_inv = np.load(os.path.join(input_dir, "L_inv.npy"))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Polytope metrics missing. Run your generator first. Error: {e}")

    print(f"[DEBUG] Initializing Stochastic MILP Bidding variables for T={T} steps...")
    
    da_prices_24 = np.array([
        0.04, 0.03, 0.03, 0.02, 0.03, 0.05, 
        0.07, 0.09, 0.08, 0.06, 0.05, 0.05,
        0.04, 0.04, 0.05, 0.06, 0.08, 0.11, 
        0.12, 0.10, 0.08, 0.07, 0.05, 0.04
    ])
    da_prices = np.repeat(da_prices_24, 4)
    
    # Deployment performance pricing structures ($/kWh)
    price_AS_up_activation = da_prices * 1.50
    price_AS_down_rebate = da_prices * 0.40
    
    prob_s = 1.0 / num_scenarios
    C_slack = 10.0  # Heavy penalty for violating battery boundaries
    dt = 0.25 
    M = 2000.0 # Big-M tracking ceiling constant
    
    # 1. MASTER STRATEGIC DECISION VARIABLES (Unified across all realities)
    p_DA = cp.Variable(T, name="p_DA")          # Day-Ahead Energy Baseline
    r_up = cp.Variable(T, name="r_up")          # Upward Reserve max bid capacity
    r_down = cp.Variable(T, name="r_down")      # Downward Reserve max bid capacity
    
    # MUTUAL EXCLUSION BINARY VARIABLE
    # z == 1 -> Downward Bid Allowed, z == 0 -> Upward Bid Allowed
    z = cp.Variable(T, boolean=True, name="z")
    
    # 2. SCENARIO-SPECIFIC PHYSICAL PROFILE TRACKS
    p_ch = cp.Variable((num_scenarios, T), name="p_ch")
    s_plus = cp.Variable((num_scenarios, T), name="s_plus")
    s_minus = cp.Variable((num_scenarios, T), name="s_minus")
    x_0 = cp.Variable((num_scenarios, T), name="x_0") # Al Taha base coordinates

    # 3. BASELINE MARKET COST
    da_energy_cost = cp.sum(cp.multiply(da_prices, p_DA)) * dt
    
    # 4. INITIALIZE RECOURSE COST VECTOR
    expected_recourse_cost = 0.0
    
    # 5. STATIC BOUNDARY & BIG-M MASTER CONSTRAINTS
    constraints = [
        p_DA >= 0.0,
        r_up >= 0.0,
        r_down >= 0.0,
        p_ch >= 0.0,
        s_plus >= 0.0,
        s_minus >= 0.0,
        
        # MUTUAL EXCLUSION ENFORCEMENT via BIG-M
        r_down <= M * z,
        r_up <= M * (1.0 - z),
        
        # Upper/lower operational boundaries
        r_up <= p_DA,
        p_DA + r_down <= M 
    ]
    
    # Polytope setup
    Gamma_agg_x_matrix = np.diag(gamma_agg_diag_x)
    I_T = np.eye(T)
    H_tilde = np.vstack([I_T, -I_T, L_inv, -L_inv])
    
    # 6. STOCHASTIC SCENARIO ACCOUNTING LOOP
    for s in range(num_scenarios):
        # Deployment factor: 0.0 means no deployment, 1.0 means full capacity requested
        realization_factor = np.random.uniform(0.0, 1.0)
        
        # UNIFIED STOCHASTIC LEDGER EQUATION
        # Downward requests increase physical load (+), Upward requests decrease physical load (-)
        constraints.append(
            p_ch[s, :] == p_DA + (r_down * realization_factor) - (r_up * realization_factor) + s_plus[s, :] - s_minus[s, :]
        )
        
        # Structure-preserving container equation (Al Taha Mapping)
        x_ch_s = gamma_agg_x + Gamma_agg_x_matrix @ x_0[s, :]
        constraints.append(p_ch[s, :] == L_inv @ x_ch_s)
        constraints.append(H_tilde @ x_0[s, :] <= h_0)
        
        # 7. FINANCIAL RECOURSE COST FOR THIS SCENARIO
        slack_penalty = C_slack * cp.sum(s_plus[s, :] + s_minus[s, :]) * dt
        
        # Financial adjustments based on active asset behavior in this scenario
        added_consumption_cost = cp.sum(cp.multiply(da_prices, r_down * realization_factor)) * dt
        energy_discount_credit = cp.sum(cp.multiply(price_AS_down_rebate, r_down * realization_factor)) * dt
        saved_consumption_credit = cp.sum(cp.multiply(da_prices, r_up * realization_factor)) * dt
        up_generation_revenue = cp.sum(cp.multiply(price_AS_up_activation, r_up * realization_factor)) * dt
        
        # Net operational recourse impact for scenario s
        scenario_cost = (
            slack_penalty 
            + added_consumption_cost 
            - energy_discount_credit 
            - saved_consumption_credit 
            - up_generation_revenue
        )
        
        expected_recourse_cost += prob_s * scenario_cost

    # Master Objective: Base contract expenses + expected uncertain performance adjustments
    objective = cp.Minimize(da_energy_cost + expected_recourse_cost)
    prob = cp.Problem(objective, constraints)
    
    print("\n==========================================================")
    print(" -> INVOKING NATIVE BUNDLED ECOS_BB SOLVER ENGINE")
    print("==========================================================")
    
    # ECOS_BB handles mixed-integer problems and is natively included inside your CVXPY library
    prob.solve(solver='ECOS_BB', verbose=True)
    
    print("==========================================================\n")
    
    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        print("=== Optimization Successful ===")
        print(f"Total Expected Net Operating Cost: ${prob.value:.2f}")
        
        print("\nInterval | DA Baseline (kW) | Selected Market | Up Bid (kW) | Down Bid (kW)")
        print("-" * 77)
        for t in range(0, T, 4):
            market_type = "DOWNWARD" if z.value[t] > 0.5 else "UPWARD  "
            print(f"Q-{t:02d}     | {p_DA.value[t]:16.2f} | {market_type}      | {r_up.value[t]:11.2f} | {r_down.value[t]:13.2f}")
            
        total_slack_kwh = np.sum(s_plus.value + s_minus.value) * dt
        print(f"\nTotal Realized Slack Violation Energy: {total_slack_kwh:.2f} kWh")
    else:
        print("[ERROR] Bidding framework optimization failed via ECOS_BB. Status:", prob.status)

if __name__ == "__main__":
    solve_stochastic_joint_bidding(input_dir="saved_polytope", T=96, num_scenarios=5)