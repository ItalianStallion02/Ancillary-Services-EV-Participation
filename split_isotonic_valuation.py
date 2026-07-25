import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

def compute_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i+1]
        in_bin = (y_prob >= low) & (y_prob <= high) if i == 0 else (y_prob > low) & (y_prob <= high)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    return ece

def print_fixed_decile_table(y_true, y_prob):
    print("\n" + "="*80)
    print(" 1. FIXED PROBABILITY RELIABILITY TABLE (0.0 to 1.0)")
    print("="*80)
    print(f" {'Bin Range':<15} | {'Count':<10} | {'Pct Total':<10} | {'Mean Pred P':<14} | {'Empirical P':<14} | {'Abs Error':<10}")
    print("-" * 80)
    
    bins = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90, 1.00]
    total_n = len(y_true)
    
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i+1]
        mask = (y_prob >= low) & (y_prob <= high) if i == 0 else (y_prob > low) & (y_prob <= high)
        count = np.sum(mask)
        
        if count > 0:
            pct_total = (count / total_n) * 100
            mean_pred = np.mean(y_prob[mask])
            actual_rate = np.mean(y_true[mask])
            abs_err = np.abs(mean_pred - actual_rate)
            print(f" ({low:.2f}, {high:.2f}]     | {count:<10,} | {pct_total:<9.2f}% | {mean_pred:<14.6f} | {actual_rate:<14.6f} | {abs_err:<10.6f}")
        else:
            print(f" ({low:.2f}, {high:.2f}]     | 0          | 0.00%     | N/A            | N/A            | N/A")

def print_purpose_breakdown(df):
    print("\n" + "="*80)
    print(" 2. MARKET PURPOSE CALIBRATION BREAKDOWN (PURPOSE_CD)")
    print("="*80)
    
    purpose = df.groupby('PURPOSE_CD', observed=False).agg(
        total_bids=('target', 'count'),
        accepted_bids=('target', 'sum'),
        true_rate=('target', 'mean'),
        mean_pred=('cal_prob', 'mean')
    ).reset_index()
    
    purpose['abs_err'] = np.abs(purpose['true_rate'] - purpose['mean_pred'])
    
    print(f" {'Purpose':<10} | {'Total Bids':<10} | {'Accepted':<10} | {'True Rate':<12} | {'Mean Pred P':<14} | {'Abs Error':<10}")
    print("-" * 80)
    for _, row in purpose.iterrows():
        print(f" {str(row['PURPOSE_CD']):<10} | {int(row['total_bids']):<10,} | {int(row['accepted_bids']):<10,} | {row['true_rate']:<12.6f} | {row['mean_pred']:<14.6f} | {row['abs_err']:<10.6f}")

def run_deep_validation(val_path: str, model_path: str, calib_path: str, meta_path: str):
    print("\n" + "="*80)
    print(" STRATIFIED MODEL VALIDATION ENGINE (ISOTONIC)")
    print("="*80)
    
    model = joblib.load(model_path)
    calibrators = joblib.load(calib_path)
    metadata = joblib.load(meta_path)
    val_df = pd.read_parquet(val_path)
    
    feature_cols = metadata['feature_cols']
    cat_cols = metadata.get('cat_cols', [])
    for c in cat_cols:
        if c in val_df.columns:
            val_df[c] = val_df[c].astype('category')

    y_val = val_df['target'].values
    
    # 1. Predict Raw Probabilities
    raw_probs = model.predict(val_df[feature_cols])
    val_df['cal_prob'] = raw_probs
    
    # 2. Apply Purpose-Stratified Isotonic Calibration
    for p, iso_model in calibrators.items():
        sub_mask = val_df['PURPOSE_CD'].astype(str) == p
        if np.sum(sub_mask) > 0:
            sub_raw = val_df.loc[sub_mask, 'cal_prob'].values
            cal_probs = iso_model.transform(sub_raw)
            val_df.loc[sub_mask, 'cal_prob'] = cal_probs

    safe_probs = np.clip(val_df['cal_prob'].values, 1e-6, 1.0 - 1e-6)

    print("\n" + "="*80)
    print(" GLOBAL EVALUATION SUMMARY")
    print("="*80)
    print(f" Log Loss:                  {log_loss(y_val, safe_probs):.6f}")
    print(f" Brier Score:               {brier_score_loss(y_val, safe_probs):.6f}")
    print(f" Expected Calibration Error: {compute_ece(y_val, safe_probs, n_bins=10):.6f}")
    print(f" ROC-AUC Score:             {roc_auc_score(y_val, safe_probs):.4f}")
    print(f" Mean Pred vs True Rate:    {np.mean(safe_probs):.6f} vs {np.mean(y_val):.6f}")

    print_fixed_decile_table(y_val, safe_probs)
    print_purpose_breakdown(val_df)

if __name__ == "__main__":
    run_deep_validation(
        val_path="val_msd.parquet",
        model_path="msd_lgb_prob_model.pkl",
        calib_path="msd_stratified_calibrators.pkl",
        meta_path="model_metadata.pkl"
    )