import os
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

def visualize_v2_asymmetric_flexibility(data_dir="v2_polytope"):
    # ----------------------------------------------------
    # 1. LOAD ARCHIVED DATA FROM TARGET DIRECTORY
    # ----------------------------------------------------
    print(f" -> Loading optimized polytope data from '{data_dir}/'...")
    try:
        gamma_agg_diag_x = np.load(os.path.join(data_dir, "gamma_agg_diag_x.npy"))
        gamma_agg_x = np.load(os.path.join(data_dir, "gamma_agg_x.npy"))
        h_0 = np.load(os.path.join(data_dir, "h_0.npy"))
        L_inv = np.load(os.path.join(data_dir, "L_inv.npy"))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing files in '{data_dir}'. Please run 'polytope_generator_v2.py' first.") from e

    T = len(gamma_agg_diag_x)
    delta_t = 0.25
    
    # Re-calculate N dynamically from data (Minkowski scaling factor)
    # Using a 100 EV benchmark setup
    N = 100
    
    time_steps = np.arange(T) * delta_t
    time_energy = np.arange(1, T + 1) * delta_t

    # ----------------------------------------------------
    # 2. COMPUTE MINKOWSKI SUM OUTER BOUNDS
    # ----------------------------------------------------
    mink_x_max = N * h_0[:T]
    mink_x_min = N * (-h_0[T:2*T])
    mink_u_max = N * h_0[2*T:3*T]
    mink_u_min = N * (-h_0[3*T:])

    # ----------------------------------------------------
    # 3. RECONSTRUCT INNER ENVELOPES (ENERGY & POWER)
    # ----------------------------------------------------
    print(" -> Reconstructing operational envelopes...")
    
    # Calculate energy bounds using diagonal mapping alignment
    inner_x_max = gamma_agg_diag_x * h_0[:T] + gamma_agg_x
    inner_x_min = gamma_agg_diag_x * (-h_0[T:2*T]) + gamma_agg_x
    
    # Reconstruct H_tilde to properly map temporally coupled power limits
    I_T = np.eye(T)
    H_tilde = np.vstack([I_T, -I_T, L_inv, -L_inv])
    
    inner_u_max = np.zeros(T)
    inner_u_min = np.zeros(T)
    
    # Setup a rapid LP query system to resolve coupled bounds step-by-step
    x_0 = cp.Variable(T)
    x_agg = cp.multiply(gamma_agg_diag_x, x_0) + gamma_agg_x
    u_agg = L_inv @ x_agg
    constraints = [H_tilde @ x_0 <= h_0]
    
    for t in range(T):
        # Maximize power boundary profile at index t
        prob_max = cp.Problem(cp.Maximize(u_agg[t]), constraints)
        prob_max.solve(solver=cp.CLARABEL, verbose=False)
        inner_u_max[t] = u_agg[t].value
        
        # Minimize power boundary profile at index t
        prob_min = cp.Problem(cp.Minimize(u_agg[t]), constraints)
        prob_min.solve(solver=cp.CLARABEL, verbose=False)
        inner_u_min[t] = u_agg[t].value

    # Pad origins at t=0 for visual flow
    plot_time_energy = np.concatenate(([0], time_energy))
    plot_inner_x_max = np.concatenate(([0], inner_x_max))
    plot_inner_x_min = np.concatenate(([0], inner_x_min))
    plot_mink_x_max = np.concatenate(([0], mink_x_max))
    plot_mink_x_min = np.concatenate(([0], mink_x_min))

    # ----------------------------------------------------
    # 4. GENERATE INTERACTIVE GRAPH DISPLAY
    # ----------------------------------------------------
    print(" -> Processing interactive plotting dashboard...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # --- PLOT 1: ENERGY SPACE ---
    ax1.fill_between(plot_time_energy, plot_mink_x_min, plot_mink_x_max, color='gray', alpha=0.12, label='Minkowski Sum (Outer Ceiling)')
    ax1.fill_between(plot_time_energy, plot_inner_x_min, plot_inner_x_max, color='royalblue', alpha=0.35, label='AH Inner Approximation ($\mathbb{P}$)')
    ax1.plot(plot_time_energy, plot_inner_x_max, color='blue', linestyle='-', linewidth=1.0)
    ax1.plot(plot_time_energy, plot_inner_x_min, color='blue', linestyle='-', linewidth=1.0)
    
    ax1.set_title(f'24-Hour Fleet Energy Envelope (N = {N} Heterogeneous EVs - Structural Asymmetric Homothety)', fontsize=12, fontweight='bold') 
    ax1.set_ylabel('Cumulative Fleet Energy Capacity (kWh)', fontsize=10)
    ax1.set_xlim([0, 24])
    ax1.set_xticks(np.arange(0, 25, 2))
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left')
    
    # --- PLOT 2: POWER SPACE ---
    ax2.fill_between(time_steps, mink_u_min, mink_u_max, step='post', color='gray', alpha=0.12, label='Minkowski Sum (Outer Ceiling)')
    ax2.fill_between(time_steps, inner_u_min, inner_u_max, step='post', color='mediumseagreen', alpha=0.35, label='AH Inner Approximation ($\mathbb{P}$)')
    ax2.step(time_steps, inner_u_max, color='darkgreen', linestyle='-', linewidth=1.0, where='post')
    ax2.step(time_steps, inner_u_min, color='darkgreen', linestyle='-', linewidth=1.0, where='post')
    
    ax2.set_title('24-Hour Fleet Instantaneous Power Charging Envelope', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time of Day (Hours Horizon)', fontsize=10)
    ax2.set_ylabel('Total Fleet Grid Power Capacity (kW)', fontsize=10)
    ax2.set_xlim([0, 24])
    ax2.set_xticks(np.arange(0, 25, 2))
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    print("Launching plot viewer window...")
    plt.show()

if __name__ == "__main__":
    visualize_v2_asymmetric_flexibility()