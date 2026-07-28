import time
import joblib
import numpy as np
import pandas as pd

# Lombardia / Central-North Cluster
LOMBARDIA_UNITS = ["UP_VOGHERA_1", "UP_NPWRMNTOVA_2", "UP_TURBIGO_3"]


def balanced_fast_forward_selection(
    joint_scenarios: np.ndarray,
    n_reduce: int = 6,
    temperature: float = 2.0,
    max_single_weight: float = 0.50,
) -> tuple[np.ndarray, np.ndarray]:
    """Applies Balanced Fast Forward Selection with Softmax Mass Redistribution.

    Parameters:
    -----------
    joint_scenarios: np.ndarray (N, Dim)
        Raw 0/1 activation matrix across all units and 96 periods.
    n_reduce: int
        Number of target reduced scenarios (default 6).
    temperature: float
        Softmax smoothing factor for mass redistribution. Higher values = more balanced weights.
    max_single_weight: float
        Upper cap for the probability weight of any single scenario (e.g., 0.50 = 50% max).
    """
    N, Dim = joint_scenarios.shape

    # 1. Compute summary features per scenario to prevent sparse dominance
    # Features: Raw activation + 4-period moving sum (hourly ramps) + total energy run
    p_mean = np.mean(joint_scenarios, axis=0)
    p_std = np.std(joint_scenarios, axis=0)
    p_std = np.where(p_std < 1e-4, 1.0, p_std)

    # Z-score normalization
    z_scenarios = (joint_scenarios - p_mean) / p_std

    # Add total daily run time per unit as additional feature columns
    unit_totals = joint_scenarios.reshape(N, 3, 96).sum(axis=2)  # shape (N, 3)
    unit_totals_norm = (unit_totals - unit_totals.mean(axis=0)) / (
        unit_totals.std(axis=0) + 1e-4
    )

    # Expanded Feature Matrix (Z-scores + Portfolio Run Totals)
    feature_matrix = np.hstack([z_scenarios, unit_totals_norm * 2.0])

    # 2. Distance Matrix
    diff = feature_matrix[:, np.newaxis, :] - feature_matrix[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

    # 3. Iterative Medoid Selection (P-FFS)
    p_orig = np.full(N, 1.0 / N)

    # 1st medoid: Minimum weighted distance to all points
    total_dist = np.sum(dist_matrix, axis=1)
    first_idx = np.argmin(total_dist)

    selected = [first_idx]
    unselected = list(set(range(N)) - set(selected))

    for _ in range(1, n_reduce):
        # Maximize the minimum distance to already selected medoids
        min_dists = np.min(dist_matrix[unselected][:, selected], axis=1)
        best_candidate_idx = np.argmax(min_dists)
        best_idx = unselected[best_candidate_idx]

        selected.append(best_idx)
        unselected.remove(best_idx)

    selected = np.array(selected)

    # 4. Softmax Distance Mass Redistribution (Prevents 99% collapse)
    # Distance from all N original scenarios to the 6 selected medoids
    dists_to_selected = dist_matrix[:, selected]  # shape (N, 6)

    # Compute soft assignment probabilities using inverse distance Softmax
    # Subtract min dist per row for numerical stability
    scaled_dists = -dists_to_selected / (
        dists_to_selected.std() * temperature + 1e-6
    )
    exp_dists = np.exp(scaled_dists - np.max(scaled_dists, axis=1, keepdims=True))
    soft_assignments = exp_dists / np.sum(exp_dists, axis=1, keepdims=True)

    # Aggregate weights
    raw_weights = np.sum(soft_assignments * p_orig[:, np.newaxis], axis=0)

    # Apply Probability Cap (max_single_weight) and re-normalize
    capped_weights = np.minimum(raw_weights, max_single_weight)
    final_weights = capped_weights / np.sum(capped_weights)

    # Sort scenarios by probability weight descending
    sort_idx = np.argsort(final_weights)[::-1]
    selected = selected[sort_idx]
    final_weights = final_weights[sort_idx]

    return selected, final_weights


def print_balanced_area_scenarios(
    reduced_portfolio: dict[str, np.ndarray],
    weights: np.ndarray,
    target_units: list[str],
):
    """Visualizes 6 reduced area scenarios with balanced probability distribution."""
    num_scenarios = len(weights)
    n_periods = 96

    hour_header_str = "".join([f"{h:02d}  " for h in range(24)])
    sub_period_str = "".join(["1234" for _ in range(24)])

    print("\n" + "=" * 125)
    print(" LOMBARDIA AREA: BALANCED 6-SCENARIO DISTRIBUTION")
    print("=" * 125)

    unit_matrix = np.array([reduced_portfolio[u] for u in target_units])

    for s_idx in range(num_scenarios):
        weight_pct = weights[s_idx] * 100
        area_active_count = unit_matrix[:, s_idx, :].sum(axis=0)

        area_ascii_chars = []
        for count in area_active_count:
            if count == 0:
                area_ascii_chars.append("·")
            elif count == 1:
                area_ascii_chars.append("▄")
            elif count == 2:
                area_ascii_chars.append("█")
            else:
                area_ascii_chars.append("▓")

        area_ascii_str = "".join(area_ascii_chars)
        area_active_periods = (area_active_count > 0).sum()

        print(f"\n┌{"─"*123}┐")
        print(
            f"│ BALANCED SCENARIO #{s_idx+1:02d} (Probability Weight π = {weight_pct:5.1f}%)"
            + " " * 62
            + "│"
        )
        print(f"├{"─"*123}┤")
        print(
            f"│ Unit / Level       │ Hour:   {hour_header_str} │ Total Active │"
        )
        print(
            f"│                    │ Period: {sub_period_str} │ (Periods)    │"
        )
        print(f"├{"─"*123}┤")

        for unit in target_units:
            scen_periods = reduced_portfolio[unit][s_idx]
            u_ascii = "".join(["█" if p == 1 else "·" for p in scen_periods])
            print(
                f"│ {unit:<18} │         {u_ascii} │   {scen_periods.sum():02d}/96 p    │"
            )

        print(f"├{"─"*123}┤")
        print(
            f"│ [AREA CUMULATED]   │         {area_ascii_str} │   {area_active_periods:02d}/96 p    │"
        )
        print(f"└{"─"*123}┘")

    # Overall Expected Area Probability Curve
    cumulated_area_prob = np.zeros(n_periods)
    for s_idx in range(num_scenarios):
        is_active = (unit_matrix[:, s_idx, :].sum(axis=0) > 0).astype(float)
        cumulated_area_prob += weights[s_idx] * is_active

    print("\n" + "=" * 125)
    print(" WEIGHTED CUMULATED AREA ACTIVATION PROBABILITY (%) PER PERIOD")
    print("=" * 125)
    print(f" Hour:   {hour_header_str}")
    print(f" Period: {sub_period_str}")
    print("─" * 125)

    prob_ascii = []
    for p in cumulated_area_prob:
        if p >= 0.75:
            prob_ascii.append("▓")
        elif p >= 0.50:
            prob_ascii.append("█")
        elif p >= 0.25:
            prob_ascii.append("▄")
        elif p > 0.0:
            prob_ascii.append("░")
        else:
            prob_ascii.append("·")

    print(f" Prob:   {"".join(prob_ascii)}")
    print(
        " Legend: · = 0% | ░ = 1-25% | ▄ = 25-50% | █ = 50-75% | ▓ = 75-100% Probability"
    )
    print("=" * 125 + "\n")


def run_balanced_area_pipeline(
    val_data_path: str = "val_msd_unit.parquet",
    pipeline_path: str = "markov_lgb_pipeline.pkl",
    target_purpose: str = "BID",
    num_scenarios: int = 1000,
    reduced_count: int = 6,
):
    val_df = pd.read_parquet(val_data_path)
    filtered_df = val_df[
        (val_df["UNIT_REFERENCE_NO"].isin(LOMBARDIA_UNITS))
        & (val_df["PURPOSE_CD"].astype(str) == target_purpose)
    ].copy()

    artifacts = joblib.load(pipeline_path)
    model_off = artifacts["model_off"]
    model_on = artifacts["model_on"]
    calib_off = artifacts["calib_off"]
    calib_on = artifacts["calib_on"]
    feature_cols = artifacts["feature_cols"]

    unit_scenarios_dict = {}
    valid_units = []

    for unit in LOMBARDIA_UNITS:
        unit_df = filtered_df[filtered_df["UNIT_REFERENCE_NO"] == unit].copy()

        if len(unit_df) < 96:
            continue

        day_df = (
            unit_df.head(96).sort_values(by="PERIOD").reset_index(drop=True)
        )
        n_periods = len(day_df)

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

        random_draws = np.random.uniform(size=(num_scenarios, n_periods))
        scenarios = np.zeros((num_scenarios, n_periods), dtype=np.int32)

        for p_idx in range(n_periods):
            prob_if_off = p_off_vec[p_idx]
            prob_if_on = p_on_vec[p_idx]

            if p_idx == 0:
                current_probs = np.full(num_scenarios, prob_if_off)
            else:
                prev_states = scenarios[:, p_idx - 1]
                current_probs = np.where(
                    prev_states == 1, prob_if_on, prob_if_off
                )

            scenarios[:, p_idx] = (
                random_draws[:, p_idx] < current_probs
            ).astype(np.int32)

        unit_scenarios_dict[unit] = scenarios
        valid_units.append(unit)

    # Combine into joint matrix (1000 x 288)
    joint_matrix = np.hstack([unit_scenarios_dict[u] for u in valid_units])

    # Run Balanced FFS
    selected_indices, scenario_weights = balanced_fast_forward_selection(
        joint_matrix,
        n_reduce=reduced_count,
        temperature=2.0,
        max_single_weight=0.45,  # Capped at 45% max per scenario
    )

    reduced_portfolio = {
        unit: unit_scenarios_dict[unit][selected_indices]
        for unit in valid_units
    }

    print_balanced_area_scenarios(
        reduced_portfolio, scenario_weights, valid_units
    )

    return reduced_portfolio, scenario_weights, selected_indices


if __name__ == "__main__":
    run_balanced_area_pipeline()