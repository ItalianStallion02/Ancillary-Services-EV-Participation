import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_evening_bid_heatmap(
    model_path="msd_lgb_prob_model.pkl",
    calib_path="msd_stratified_calibrators.pkl",
    meta_path="model_metadata.pkl",
    output_image="evening_bid_probabilities.png"
):
    # 1. Load trained model, calibrators, and metadata
    print("[1/4] Loading trained model and Isotonic calibrator...")
    model = joblib.load(model_path)
    calibrators = joblib.load(calib_path)
    metadata = joblib.load(meta_path)
    
    feature_cols = metadata['feature_cols']
    bid_calibrator = calibrators.get('BID')

    # 2. Match exact price grid from your working script (€50 to €250, 41 steps)
    prices = np.linspace(50, 250, 41)

    # Define Evening Periods (Intervals 18 through 22 => Periods 69 through 88)
    evening_hours = [18, 19, 20, 21, 22]
    target_periods = []
    period_to_interval = {}
    
    for h in evening_hours:
        periods_in_hour = list(range((h - 1) * 4 + 1, h * 4 + 1))
        target_periods.extend(periods_in_hour)
        for p in periods_in_hour:
            period_to_interval[p] = h

    # Realistic varying PUN curve across 20 15-min evening periods (€/MWh)
    pun_curve = {
        # Interval 18 (18:00 - 19:00)
        69: 110.0, 70: 115.0, 71: 120.0, 72: 125.0,
        # Interval 19 (19:00 - 20:00)
        73: 130.0, 74: 140.0, 75: 150.0, 76: 155.0,
        # Interval 20 (20:00 - 21:00) - Peak ramp
        77: 160.0, 78: 165.0, 79: 165.0, 80: 160.0,
        # Interval 21 (21:00 - 22:00)
        81: 150.0, 82: 145.0, 83: 140.0, 84: 135.0,
        # Interval 22 (22:00 - 23:00)
        85: 130.0, 86: 125.0, 87: 120.0, 88: 115.0
    }

    # Baseline defaults (matching your setup)
    baseline_features = {
        'QUANTITY_NO': 100.0,
        'LOG_QUANTITY': np.log1p(100.0),
        'PURPOSE_CD': 'BID',
        'TYPE_CD': 'ND',        
        'MARKET_CD': 'MGP',     
        'day_of_week': 2,       # Wednesday
        'month': 5              # May
    }

    # 3. Build evaluation grid & predict
    print("[2/4] Simulating bid grid across varying evening PUN periods...")
    grid_rows = []
    for p_period in target_periods:
        interval_no = period_to_interval[p_period]
        period_pun = pun_curve[p_period]

        for price in prices:
            row = baseline_features.copy()
            row['INTERVAL_NO'] = interval_no
            row['PERIOD'] = p_period
            row['PUN'] = period_pun
            row['ENERGY_PRICE_NO'] = price
            
            # Dynamic price spread features relative to the period-specific PUN
            row['PRICE_SPREAD_PUN'] = price - period_pun
            row['SIGNED_SPREAD_PUN'] = price - period_pun
            row['PRICE_RATIO_PUN'] = price / max(period_pun, 1e-3)
            
            grid_rows.append(row)

    grid_df = pd.DataFrame(grid_rows)

    # Ensure categorical types match training metadata
    for c in metadata.get('cat_cols', []):
        if c in grid_df.columns:
            grid_df[c] = grid_df[c].astype('category')

    # Predict Raw -> Calibrate
    raw_probs = model.predict(grid_df[feature_cols])
    if bid_calibrator is not None:
        calibrated_probs = bid_calibrator.transform(raw_probs)
    else:
        calibrated_probs = raw_probs

    grid_df['calibrated_prob'] = calibrated_probs

    # 4. Pivot into Matrix for Plotting
    pivot_matrix = grid_df.pivot(
        index='ENERGY_PRICE_NO', 
        columns='PERIOD', 
        values='calibrated_prob'
    )
    pivot_matrix_pct = pivot_matrix * 100.0

    # 5. Plot Color-Coded Heatmap with Overlaid PUN Line
    print("[3/4] Generating color-coded plot with PUN overlay...")
    plt.figure(figsize=(14, 9))
    
    # Render Heatmap
    ax = sns.heatmap(
        pivot_matrix_pct, 
        cmap="YlGnBu", 
        annot=False, 
        cbar_kws={'label': 'Calibrated Acceptance Probability (%)'},
        linewidths=0.0
    )
    
    # Align heatmap coordinates (X: 0.5 to N-0.5)
    x_coords = np.arange(len(target_periods)) + 0.5
    
    # Map raw PUN values to Y-axis positions (50 to 250 across 41 rows)
    y_min, y_max = prices.min(), prices.max()
    num_y_bins = len(prices)
    pun_y_coords = [(pun_curve[p] - y_min) / (y_max - y_min) * (num_y_bins - 1) + 0.5 for p in target_periods]

    # Overlay PUN curve
    plt.plot(
        x_coords, 
        pun_y_coords, 
        color='crimson', 
        linewidth=2.5, 
        linestyle='--', 
        marker='o', 
        markersize=4, 
        label='Period PUN (€/MWh)'
    )

    plt.title("Evening BID Acceptance Probability (%) with Dynamic PUN Profile\n(Purpose: BID | Quantity: 100 MW)", fontsize=14, pad=15)
    plt.xlabel("15-Min Evening Period (PERIOD 69=18:00 to 88=22:45)", fontsize=12)
    plt.ylabel("Bid Energy Price (ENERGY_PRICE_NO)", fontsize=12)
    
    # Format X-axis Ticks
    plt.xticks(
        ticks=x_coords[::2], 
        labels=[f"P{p}\n(Int {period_to_interval[p]})" for p in target_periods[::2]], 
        rotation=0
    )
    
    plt.gca().invert_yaxis()
    plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"[4/4] Saved plot to '{output_image}'.")
    plt.show()

if __name__ == "__main__":
    generate_evening_bid_heatmap()