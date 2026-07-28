import joblib
import numpy as np
import pandas as pd


def fast_forward_selection(
    scenarios: np.ndarray, n_reduce: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Reduces N scenarios to `n_reduce` representative scenarios using Fast Forward Selection (Wasserstein / Kantorovich distance).

    :param scenarios: Binary matrix of shape (N_scenarios, N_periods)
    :param n_reduce: Number of scenarios to keep
    :return: (selected_indices, scenario_probabilities)
    """
    N, T = scenarios.shape
    if N <= n_reduce:
        return np.arange(N), np.full(N, 1.0 / N)

    # 1. Compute pairwise distance matrix (Euclidean/L1 distance between daily paths)
    # Shape: (N, N)
    diff = scenarios[:, np.newaxis, :] - scenarios[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

    # Initial uniform probabilities for all 1,000 scenarios
    p_orig = np.full(N, 1.0 / N)

    # 2. Step 1: Find the first scenario that minimizes total distance to all others
    total_dist = np.sum(dist_matrix, axis=1)
    first_idx = np.argmin(total_dist)

    selected = [first_idx]
    unselected = list(set(range(N)) - set(selected))

    # 3. Iteratively select remaining (n_reduce - 1) scenarios
    for _ in range(1, n_reduce):
        min_dists = np.min(dist_matrix[unselected][:, selected], axis=1)
        # Select candidate that maximizes min distance reduction
        best_candidate_idx = np.argmax(min_dists)
        best_idx = unselected[best_candidate_idx]

        selected.append(best_idx)
        unselected.remove(best_idx)

    selected = np.array(selected)

    # 4. Redistribute probabilities of unselected scenarios to their nearest selected scenario
    weights = np.zeros(n_reduce)
    nearest_selected = np.argmin(dist_matrix[:, selected], axis=1)

    for i in range(N):
        assigned_k = nearest_selected[i]
        weights[assigned_k] += p_orig[i]

    return selected, weights


def generate_reduced_voghera_scenarios(
    val_data_path: str = "val_msd_unit.parquet",
    pipeline_path: str = "markov_lgb_pipeline.pkl",
    target_unit: str = "UP_VOGHERA_1",
    n_initial: int = 1000,
    n_final: int = 10,
):
    """Generates 1,000 vectorized Markov scenarios and reduces them to 10 weighted scenarios."""
    # Load and filter validation data
    val_df = pd.read_parquet(val_data_path)
    voghera_df = val_df[val_df["UNIT_REFERENCE_NO"] == target_unit].copy()
    day_df = (
        voghera_df.head(96).sort_values(by="PERIOD").reset_index(drop=True)
    )

    date_str = (
        str(day_df["date"].iloc[0])
        if "date" in day_df.columns
        else "Simulated Day"
    )

    # Load LightGBM Markov artifacts
    artifacts = joblib.load(pipeline_path)
    model_off = artifacts["model_off"]
    model_on = artifacts["model_on"]
    calib_off = artifacts["calib_off"]
    calib_on = artifacts["calib_on"]
    feature_cols = artifacts["feature_cols"]

    n_periods = len(day_df)

    # Batch prediction step
    raw_off = model_off.predict(day_df[feature_cols])
    raw_on = model_on.predict(day_df[feature_cols])

    purposes = day_df["PURPOSE_CD"].astype(str).values
    p_off_vec = np.array([
        calib_off[p].predict([r])[0] if p in calib_off else r
        for r, p in zip(raw_off, purposes)
    ])
    p_on_vec = np.array([
        calib_on[p].predict([r])[0] if p in calib_on else r
        for r, p in zip(raw_on, purposes)
    ])

    # Generate 1,000 initial scenarios
    random_draws = np.random.uniform(size=(n_initial, n_periods))
    initial_scenarios = np.zeros((n_initial, n_periods), dtype=np.int32)

    for p_idx in range(n_periods):
        prob_if_off = p_off_vec[p_idx]
        prob_if_on = p_on_vec[p_idx]

        if p_idx == 0:
            current_probs = np.full(n_initial, prob_if_off)
        else:
            prev_states = initial_scenarios[:, p_idx - 1]
            current_probs = np.where(prev_states == 1, prob_if_on, prob_if_off)

        initial_scenarios[:, p_idx] = (
            random_draws[:, p_idx] < current_probs
        ).astype(np.int32)

    # Perform Fast Forward Scenario Reduction (1000 -> 10)
    print(
        f"Reducing {n_initial:,} scenarios down to {n_final} using Fast Forward Selection..."
    )
    selected_indices, weights = fast_forward_selection(
        initial_scenarios, n_reduce=n_final
    )
    reduced_scenarios = initial_scenarios[selected_indices]

    # Print ASCII Visualization
    print("\n" + "=" * 85)
    print(
        f" REDUCED SCENARIO SET (1,000 -> 10) | Unit: {target_unit} | Date: {date_str}"
    )
    print("=" * 85)

    header = (
        "Scen | Prob (π) | Hour: 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 | Total On"
    )
    print(header)
    print("-" * 85)

    for idx, (scen_idx, weight) in enumerate(
        zip(selected_indices, weights), start=1
    ):
        scen_row = initial_scenarios[scen_idx]
        hourly_blocks = []

        for h in range(24):
            hour_periods = scen_row[h * 4 : (h + 1) * 4]
            active_count = hour_periods.sum()

            if active_count == 4:
                hourly_blocks.append("██")
            elif active_count >= 2:
                hourly_blocks.append("▄▄")
            elif active_count >= 1:
                hourly_blocks.append("  ")
            else:
                hourly_blocks.append("··")

        timeline_str = " ".join(hourly_blocks)
        total_hours = scen_row.sum() / 4.0
        print(
            f"#{idx:02d}  |  {weight*100:5.1f}%  |       {timeline_str} | {total_hours:4.1f} hrs"
        )

    print("-" * 85)
    print(
        " Legend:  ██ = Full Hour On (4/4)  |  ▄▄ = Half Hour On  |  ·· = Offline"
    )
    print("=" * 85 + "\n")

    return reduced_scenarios, weights, selected_indices


if __name__ == "__main__":
    reduced_scens, scenario_weights, orig_indices = (
        generate_reduced_voghera_scenarios()
    )