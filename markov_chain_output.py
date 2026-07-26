import numpy as np

def generate_autocorrelated_scenarios(K=10, T=96, dt=0.2):
    """
    Generates time-series activation draws (K x T) using 2-State Markov Chains.
    - Upward Bids: 1% off-peak base up to 7% peak cap.
    - Downward Bids: 2% off-peak base up to 15% peak cap.
    - Morning (8-11 AM) & Evening (8-10 PM) peak windows.
    """
    hours = np.linspace(0, 24 - dt, T)
    
    # Double-Gaussian distribution for morning and evening peaks
    morning_peak = np.exp(-0.5 * ((hours - 9.5) / 1.5) ** 2)
    evening_peak = np.exp(-0.5 * ((hours - 21.0) / 1.2) ** 2)
    peak_profile = morning_peak + evening_peak
    
    # Calibrated Peak Activation Curves
    p_01_up = 0.01 + 0.06 * peak_profile  # Upward: 1% -> 7%
    p_01_dn = 0.02 + 0.13 * peak_profile  # Downward: 2% -> 15%
    
    # 80% Persistence once activated (20% chance of returning to OFF)
    p_10 = 0.20  
    
    delta_up = np.zeros((K, T))
    delta_dn = np.zeros((K, T))
    
    np.random.seed(42)  # For reproducible draws
    for k in range(K):
        for t in range(T):
            if t == 0:
                delta_up[k, t] = 1.0 if np.random.rand() < p_01_up[0] else 0.0
                delta_dn[k, t] = 1.0 if np.random.rand() < p_01_dn[0] else 0.0
            else:
                # Upward Markov Transitions
                if delta_up[k, t-1] == 0:
                    delta_up[k, t] = 1.0 if np.random.rand() < p_01_up[t] else 0.0
                else:
                    delta_up[k, t] = 0.0 if np.random.rand() < p_10 else 1.0
                    
                # Downward Markov Transitions
                if delta_dn[k, t-1] == 0:
                    delta_dn[k, t] = 1.0 if np.random.rand() < p_01_dn[t] else 0.0
                else:
                    delta_dn[k, t] = 0.0 if np.random.rand() < p_10 else 1.0
                    
    return delta_up, delta_dn


def print_matrices_summary(K=15, T=96):
    delta_up, delta_dn = generate_autocorrelated_scenarios(K=K, T=T)
    
    print("=" * 80)
    print(f"GENERATED MATRICES SHAPE: {delta_up.shape} (K={K} Scenarios x T={T} Time Steps)")
    print("=" * 80)

    # --- 1. Scenario Level Summary ---
    print("\n--- SCENARIO-BY-SCENARIO ACTIVATION SUMMARY ---")
    print(f"{'Scenario':<10} | {'Upward Active Steps':<20} | {'Downward Active Steps':<22} | {'Status'}")
    print("-" * 80)
    
    empty_up_count = 0
    empty_dn_count = 0

    for k in range(K):
        up_active = int(np.sum(delta_up[k, :]))
        dn_active = int(np.sum(delta_dn[k, :]))
        
        if up_active == 0:
            empty_up_count += 1
        if dn_active == 0:
            empty_dn_count += 1
            
        status = []
        if up_active == 0 and dn_active == 0:
            status.append("COMPLETELY EMPTY")
        else:
            if up_active > 0: status.append("UP Active")
            if dn_active > 0: status.append("DN Active")

        print(f"k = {k:<6} | {up_active:<20} | {dn_active:<22} | {', '.join(status)}")

    print("-" * 80)
    print(f"Total Empty UP Rows : {empty_up_count} / {K} ({empty_up_count/K*100:.1f}%)")
    print(f"Total Empty DN Rows : {empty_dn_count} / {K} ({empty_dn_count/K*100:.1f}%)")

    # --- 2. ASCII Visualizer Grid (Representing 24 hours) ---
    print("\n" + "=" * 80)
    print("ASCII VISUAL GRID (1 character = 15-min quarter step)")
    print("'.' = OFF (0.0), '█' = ON (1.0)")
    print("=" * 80)

    print("\n[ UPWARD ACTIVATIONS MATRICES (delta_up) ]")
    for k in range(K):
        grid_row = "".join(["█" if delta_up[k, t] == 1.0 else "." for t in range(T)])
        print(f"Row {k:02d} |{grid_row}|")

    print("\n[ DOWNWARD ACTIVATIONS MATRICES (delta_dn) ]")
    for k in range(K):
        grid_row = "".join(["█" if delta_dn[k, t] == 1.0 else "." for t in range(T)])
        print(f"Row {k:02d} |{grid_row}|")

    # --- 3. Raw Numpy Matrix Output (First 3 Scenarios) ---
    print("\n" + "=" * 80)
    print("RAW NUMPY ARRAYS (Displaying Scenarios k=0 to k=2)")
    print("=" * 80)
    print("\ndelta_up[:3, :] = \n", delta_up[:3, :])
    print("\ndelta_dn[:3, :] = \n", delta_dn[:3, :])


if __name__ == "__main__":
    # Generate and print for 15 scenarios across 96 time periods
    print_matrices_summary(K=15, T=96)