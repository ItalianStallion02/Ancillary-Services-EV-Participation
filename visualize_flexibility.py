# visualize_flexibility.py
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_saved_polytope(input_dir="saved_polytope", T=96, dt=0.25):
    print(f"[DEBUG] Loading polytope data for visualization...")
    try:
        gamma_agg_diag_x = np.load(os.path.join(input_dir, "gamma_agg_diag_x.npy"))
        gamma_agg_x = np.load(os.path.join(input_dir, "gamma_agg_x.npy"))
        h_0 = np.load(os.path.join(input_dir, "h_0.npy"))
        L_inv = np.load(os.path.join(input_dir, "L_inv.npy"))
        arrival_hours = np.load(os.path.join(input_dir, "arrivals.npy"))
        departure_hours = np.load(os.path.join(input_dir, "departures.npy"))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Polytope data missing. Run 'polytope_generator.py' first.")

    N_evs = 15 
    time_hours = np.arange(T) * dt
    
    # 1. Reconstruct baseline parameters from optimization h_0
    total_u_max = h_0[2*T:3*T] * N_evs
    total_u_min = -h_0[3*T:4*T] * N_evs
    total_x_min = -h_0[T:2*T] * N_evs

    # 2. CALCULATE THE TRUE CUMULATIVE DYNAMIC CEILING
    # We use your reference scaling logic, but modulate it via an arrival tracking loop
    max_possible_power = 11.0 * N_evs 
    occupancy_ratio = total_u_max / max_possible_power
    occupancy_ratio = np.clip(occupancy_ratio, 0.0, 1.0)
    
    # Reconstruct the dynamic ceiling using a step-indicator for arrivals.
    # A car's potential capacity allocation is unlocked the moment it plugs in 
    # and stays in the cumulative fleet network energy log indefinitely.
    cumulative_arrival_weight = np.zeros(T)
    for i in range(N_evs):
        arr_t = arrival_hours[i]
        # Turn indicator to 1.0 from the exact arrival time onward
        cumulative_arrival_weight[time_hours >= arr_t] += 1.0

    # Translate the active vehicle weights into a clean time-varying multiplier
    # scaled against the total fleet size (N_evs = 15)
    fleet_arrival_ratio = cumulative_arrival_weight / N_evs
    
    # Combine the charger occupancy ratio with the arrival activation history
    total_x_max = 900.0 * np.minimum(occupancy_ratio + (1.0 - occupancy_ratio) * fleet_arrival_ratio, fleet_arrival_ratio)
    
    # Ensure the capacity ceiling safely bounds the raw minimum target lines
    total_x_max = np.maximum(total_x_max, total_x_min)

    # 3. Compute Envelope Bounds
    agg_energy_min = gamma_agg_x + gamma_agg_diag_x * (total_x_min / N_evs)
    agg_energy_max = gamma_agg_x + gamma_agg_diag_x * (total_x_max / N_evs)
    
    agg_energy_min[0] = 0.0
    agg_energy_min = np.clip(agg_energy_min, total_x_min, total_x_max)
    agg_energy_max = np.clip(agg_energy_max, total_x_min, total_x_max)

    # 4. Center Profiles
    center_energy = 0.5 * (agg_energy_min + agg_energy_max)
    center_power = L_inv @ center_energy
    
    # 5. Power Limits
    agg_power_min = L_inv @ agg_energy_min
    agg_power_max = L_inv @ agg_energy_max
    
    agg_power_min[0], agg_power_max[0] = total_u_min[0], total_u_max[0]
    agg_power_min = np.clip(agg_power_min, total_u_min, total_u_max)
    agg_power_max = np.clip(agg_power_max, total_u_min, total_u_max)
    center_power = np.clip(center_power, total_u_min, total_u_max)

    # Add visual jitter to separate marker positions cleanly
    np.random.seed(42)
    arrival_jitter = arrival_hours + np.random.uniform(-0.15, 0.15, size=arrival_hours.shape)
    departure_jitter = departure_hours + np.random.uniform(-0.15, 0.15, size=departure_hours.shape)

    if hasattr(plt.style, 'available') and 'seaborn-v0_8-whitegrid' in plt.style.available:
        plt.style.use('seaborn-v0_8-whitegrid')
    else:
        plt.style.use('ggplot')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Subplot A: Net-Energy
    ax1.fill_between(time_hours, agg_energy_min, agg_energy_max, color='skyblue', alpha=0.4, label='Aggregated Inner Flexibility Volume')
    ax1.plot(time_hours, center_energy, color='blue', linewidth=2, label='Feasible Center Energy Profile')
    ax1.plot(time_hours, total_x_max, 'r--', alpha=0.8, linewidth=2, label='Dynamic Minkowski Capacity Upper Limit')
    ax1.plot(time_hours, total_x_min, 'g--', alpha=0.6, label='Minkowski Fleet Capacity Lower Bound')
    
    ax1.scatter(arrival_jitter, np.ones_like(arrival_hours) * 920, color='limegreen', marker='v', s=100, edgecolor='black', zorder=5, label='Individual EV Arrival')
    ax1.scatter(departure_jitter, np.ones_like(departure_hours) * 920, color='darkorange', marker='^', s=100, edgecolor='black', zorder=5, label='Individual EV Departure')

    ax1.set_title('Aggregate Fleet Net-Energy Flexibility Envelope (State of Charge)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Stored Energy Volume (kWh)', fontsize=12)
    ax1.set_xlim([0, 24])
    ax1.set_ylim([-50, 980]) 
    ax1.set_xticks(np.arange(0, 25, 2))
    ax1.legend(loc='upper left', frameon=True)

    # Subplot B: Charging Power
    ax2.fill_between(time_hours, agg_power_min, agg_power_max, color='salmon', alpha=0.4, label='Aggregated Inner Power Bandwidth')
    ax2.plot(time_hours, center_power, color='darkred', linewidth=1.5, drawstyle='steps-post', label='Feasible Center Charging Profile')
    ax2.plot(time_hours, total_u_max, 'r--', alpha=0.6, label='Absolute Fleet Max Power Bound')
    ax2.plot(time_hours, total_u_min, 'g--', alpha=0.6, label='Absolute Fleet Min Power Bound')
    
    ax2.scatter(arrival_jitter, np.ones_like(arrival_hours) * 175, color='limegreen', marker='v', s=100, edgecolor='black', zorder=5, label='Individual EV Arrival')
    ax2.scatter(departure_jitter, np.ones_like(departure_hours) * 175, color='darkorange', marker='^', s=100, edgecolor='black', zorder=5, label='Individual EV Departure')

    ax2.set_title('Aggregate Fleet Charging Power Flexibility Envelope', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Power Draw Capacity (kW)', fontsize=12)
    ax2.set_xlabel('Time of Day (Hours)', fontsize=12)
    ax2.set_xlim([0, 24])
    ax2.set_ylim([-20, 190]) 
    ax2.set_xticks(np.arange(0, 25, 2))
    ax2.legend(loc='upper left', frameon=True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_saved_polytope(input_dir="saved_polytope", T=96, dt=0.25)