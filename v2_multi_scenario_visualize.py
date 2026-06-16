import numpy as np
import matplotlib.pyplot as plt
import os

def load_scenario_data(scenario_dir):
    """Loads localized outer-approximation metrics for a specific scenario."""
    gamma_agg_diag_x = np.load(os.path.join(scenario_dir, "gamma_agg_diag_x.npy"))
    gamma_agg_x = np.load(os.path.join(scenario_dir, "gamma_agg_x.npy"))
    h_0 = np.load(os.path.join(scenario_dir, "h_0.npy"))
    L_inv = np.load(os.path.join(scenario_dir, "L_inv.npy"))
    return gamma_agg_diag_x, gamma_agg_x, h_0, L_inv

def plot_all_scenarios_flexibility(base_dir="experimental_scenario_v2", dt=0.25):
    # Find and sort all scenario subdirectories numerically
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"The specified directory '{base_dir}' does not exist.")
        
    scenarios = sorted(
        [d for d in os.listdir(base_dir) if d.startswith("scenario_")],
        key=lambda x: int(x.split("_")[1])
    )
    num_scenarios = len(scenarios)
    
    if num_scenarios == 0:
        print(f"No scenario directories found inside '{base_dir}'.")
        return

    print(f"-> Found {num_scenarios} scenarios to visualize inside '{base_dir}'.")
    
    # Initialize a multi-row grid layout: 1 row per scenario, 2 columns (Energy & Power)
    fig, axes = plt.subplots(num_scenarios, 2, figsize=(14, 4 * num_scenarios), sharex=True)
    
    # If there's only 1 scenario, adjust shape to make indexing consistent
    if num_scenarios == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, s_name in enumerate(scenarios):
        scenario_path = os.path.join(base_dir, s_name)
        gamma_diag, gamma_x, h_0, L_inv = load_scenario_data(scenario_path)
        
        T = len(gamma_diag)
        time_axis = np.arange(T) * dt
        
        # Unpack structural center profiles h_0
        x_0_max = h_0[0:T]
        x_0_min = -h_0[T:2*T]
        u_0_max = h_0[2*T:3*T]
        u_0_min = -h_0[3*T:4*T]
        
        # Compute exact inner-bounding aggregate polytope limits 
        # using Minkowski scaling metrics mapped back to physical boundaries
        agg_x_max = gamma_x + (gamma_diag * x_0_max)
        agg_x_min = gamma_x + (gamma_diag * x_0_min)
        
        # Reconstruct the tracking control sequence trajectory space limits
        agg_u_max = (L_inv @ gamma_x) + (gamma_diag * u_0_max)
        agg_u_min = (L_inv @ gamma_x) + (gamma_diag * u_0_min)
        
        # ----------------------------------------------------
        # Column 1: Aggregate State-of-Charge Energy Space (kWh)
        # ----------------------------------------------------
        ax_x = axes[idx, 0]
        ax_x.plot(time_axis, agg_x_max, 'r--', label='Agg Upper Boundary', alpha=0.85)
        ax_x.plot(time_axis, agg_x_min, 'b--', label='Agg Lower Boundary', alpha=0.85)
        ax_x.fill_between(time_axis, agg_x_min, agg_x_max, color='purple', alpha=0.15, label='Flexible Polytope Domain')
        
        ax_x.set_ylabel(f"{s_name.upper()}\nEnergy (kWh)", fontsize=10, fontweight='bold')
        ax_x.grid(True, linestyle=':', alpha=0.6)
        if idx == 0:
            ax_x.set_title("Aggregated Energy Boundary $x(t)$", fontsize=12, fontweight='bold')
            ax_x.legend(loc='upper left', fontsize=8)
            
        # ----------------------------------------------------
        # Column 2: Aggregate Charging Rate Power Space (kW)
        # ----------------------------------------------------
        ax_u = axes[idx, 1]
        ax_u.plot(time_axis, agg_u_max, 'r-', label='Max Fleet Cap', alpha=0.85)
        ax_u.plot(time_axis, agg_u_min, 'b-', label='Min Fleet Cap', alpha=0.85)
        ax_u.fill_between(time_axis, agg_u_min, agg_u_max, color='orange', alpha=0.15, label='Feasible Flow Space')
        
        ax_u.set_ylabel("Power (kW)", fontsize=10)
        ax_u.grid(True, linestyle=':', alpha=0.6)
        if idx == 0:
            ax_u.set_title("Aggregated Power Boundary $u(t)$", fontsize=12, fontweight='bold')
            ax_u.legend(loc='upper right', fontsize=8)

    # Label only the bottom-most subplots' X-axis
    for col in range(2):
        axes[-1, col].set_xlabel("Time Horizon (Hours)", fontsize=11, fontweight='bold')
        
    plt.tight_layout()
    
    # Save the output visualization directly in the root folder for easy access
    output_plot_path = os.path.join(base_dir, "scenarios_flexibility_comparison.png")
    plt.savefig(output_plot_path, dpi=300)
    print(f"\n[SUCCESS] Unified visualization plot created and stored at: '{output_plot_path}'")
    plt.show()

if __name__ == "__main__":
    # Assumes T=96 intervals -> 24 hours (dt = 0.25)
    plot_all_scenarios_flexibility(base_dir="experimental_scenario_v2", dt=0.25)