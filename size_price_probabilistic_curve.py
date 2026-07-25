import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_single_period_bid_heatmap(
    model_path="msd_lgb_prob_model.pkl",
    calib_path="msd_stratified_calibrators.pkl",
    meta_path="model_metadata.pkl",
    output_image="single_period_quantity_price_probabilities.png"
):
    # 1. Load trained model, calibrators, and metadata
    print("[1/4] Loading trained model and Isotonic calibrator...")
    model = joblib.load(model_path)
    calibrators = joblib.load(calib_path)
    metadata = joblib.load(meta_path)
    
    feature_cols = metadata['feature_cols']
    bid_calibrator = calibrators.get('BID')

    # 2. Fixed Evening Period & Interval Setup
    fixed_period = 73                     # Period 73 (19:00 - 19:15)
    fixed_interval = ((fixed_period - 1) // 4) + 1  # Interval 19
    fixed_pun = 130.0                     # Fixed PUN price (€/MWh)

    # 3. Define Evaluation Grid (Quantity vs. Price)
    quantities = np.linspace(10, 500, 50)  # Bid Size from 10 MW to 500 MW
    prices = np.linspace(50, 250, 41)      # Price from €50 to €250 / MWh

    # Baseline defaults matching training configuration
    baseline_features = {
        'PERIOD': fixed_period,
        'INTERVAL_NO': fixed_interval,
        'PUN': fixed_pun,
        'PURPOSE_CD': 'BID',
        'TYPE_CD': 'ND',        
        'MARKET_CD': 'MGP',     
        'day_of_week': 2,                 # Wednesday
        'month': 5                        # May
    }

    # 4. Build Evaluation Grid & Predict
    print(f"[2/4] Simulating bid grid for Period {fixed_period} (Interval {fixed_interval}) across Quantities & Prices...")
    grid_rows = []
    
    for q in quantities:
        log_q = np.log1p(q)
        for price in prices:
            row = baseline_features.copy()
            
            # Quantity Features
            row['QUANTITY_NO'] = q
            row['LOG_QUANTITY'] = log_q
            
            # Price Features
            row['ENERGY_PRICE_NO'] = price
            
            # Dynamic Price Spreads relative to Fixed PUN
            row['PRICE_SPREAD_PUN'] = price - fixed_pun
            row['SIGNED_SPREAD_PUN'] = price - fixed_pun
            row['PRICE_RATIO_PUN'] = price / max(fixed_pun, 1e-3)
            
            grid_rows.append(row)

    grid_df = pd.DataFrame(grid_rows)

    # Ensure categorical data types match model metadata
    for c in metadata.get('cat_cols', []):
        if c in grid_df.columns:
            grid_df[c] = grid_df[c].astype('category')

    # Predict Raw Probabilities -> Isotonic Calibration
    raw_probs = model.predict(grid_df[feature_cols])
    if bid_calibrator is not None:
        calibrated_probs = bid_calibrator.transform(raw_probs)
    else:
        calibrated_probs = raw_probs

    grid_df['calibrated_prob'] = calibrated_probs

    # 5. Pivot into Matrix for Heatmap (Prices vs. Quantities)
    pivot_matrix = grid_df.pivot(
        index='ENERGY_PRICE_NO', 
        columns='QUANTITY_NO', 
        values='calibrated_prob'
    )
    pivot_matrix_pct = pivot_matrix * 100.0

    # 6. Plot Color-Coded Heatmap with Fixed PUN Overlay Line
    print("[3/4] Generating Quantity vs. Price Heatmap...")
    plt.figure(figsize=(14, 9))
    
    # Render Heatmap
    ax = sns.heatmap(
        pivot_matrix_pct, 
        cmap="YlGnBu", 
        annot=False, 
        cbar_kws={'label': 'Calibrated Acceptance Probability (%)'},
        linewidths=0.0
    )

    # Map Fixed PUN Value onto Y-axis pixel coordinate
    y_min, y_max = prices.min(), prices.max()
    num_y_bins = len(prices)
    pun_y_coord = (fixed_pun - y_min) / (y_max - y_min) * (num_y_bins - 1) + 0.5

    # Plot Horizontal PUN Reference Line across the entire Quantity X-axis
    plt.axhline(
        y=pun_y_coord, 
        color='crimson', 
        linewidth=2.5, 
        linestyle='--', 
        label=f'Fixed Period PUN (€{fixed_pun:.1f}/MWh)'
    )

    plt.title(
        f"BID Acceptance Probability Matrix (%)\n"
        f"Period: {fixed_period} (Interval {fixed_interval}) | Fixed PUN: €{fixed_pun:.1f}/MWh", 
        fontsize=14, 
        pad=15
    )
    plt.xlabel("Bid Size / Quantity (MW)", fontsize=12)
    plt.ylabel("Bid Energy Price (ENERGY_PRICE_NO)", fontsize=12)
    
    # Format X-axis Ticks (Quantities)
    x_coords = np.arange(len(quantities)) + 0.5
    tick_step = 5  # Show label every 5th quantity bin
    plt.xticks(
        ticks=x_coords[::tick_step], 
        labels=[f"{q:.0f} MW" for q in quantities[::tick_step]], 
        rotation=0
    )
    
    plt.gca().invert_yaxis()
    plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"[4/4] Saved plot to '{output_image}'.")
    plt.show()


if __name__ == "__main__":
    generate_single_period_bid_heatmap()