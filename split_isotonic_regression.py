import time
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

def load_data(train_path: str, calib_path: str, val_path: str):
    print("\n[1/3] Loading Train, Calibration, and Validation datasets...")
    train_df = pd.read_parquet(train_path)
    calib_df = pd.read_parquet(calib_path)
    val_df = pd.read_parquet(val_path)
    
    cat_cols = ['PURPOSE_CD', 'TYPE_CD', 'MARKET_CD']
    for col in cat_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].astype('category')
            calib_df[col] = calib_df[col].astype('category')
            val_df[col] = val_df[col].astype('category')
            
    print(f"      Train: {len(train_df):,} | Calib: {len(calib_df):,} | Val: {len(val_df):,}")
    return train_df, calib_df, val_df

def get_feature_lists(df: pd.DataFrame):
    potential_features = [
        'QUANTITY_NO', 'LOG_QUANTITY', 'ENERGY_PRICE_NO', 'PUN', 
        'PRICE_SPREAD_PUN', 'PRICE_RATIO_PUN', 'SIGNED_SPREAD_PUN',
        'PURPOSE_CD', 'TYPE_CD', 'MARKET_CD', 
        'INTERVAL_NO', 'PERIOD', 'day_of_week', 'month'
    ]
    feature_cols = [c for c in potential_features if c in df.columns]
    cat_cols = [c for c in feature_cols if df[c].dtype.name == 'category']
    return feature_cols, cat_cols

def train_isotonic_calibrated_pipeline(train_df: pd.DataFrame, calib_df: pd.DataFrame, feature_cols: list, cat_cols: list):
    print("\n" + "="*65)
    print(" [2/3] STEP 1: TRAINING LIGHTGBM BASE MODEL")
    print("="*65)

    X_train, y_train = train_df[feature_cols], train_df['target'].values
    X_calib, y_calib = calib_df[feature_cols], calib_df['target'].values

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    calib_data = lgb.Dataset(X_calib, label=y_calib, reference=train_data, categorical_feature=cat_cols)

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.015,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 100,
        'reg_alpha': 2.0,
        'reg_lambda': 5.0,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=5000,
        valid_sets=[train_data, calib_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=150, verbose=True),
            lgb.log_evaluation(period=200)
        ]
    )

    print("\n" + "="*65)
    print(" [3/3] STEP 2: FITTING STRATIFIED ISOTONIC CALIBRATORS")
    print("="*65)
    
    # 1. Predict raw probabilities on calibration split
    raw_calib_probs = model.predict(X_calib, num_iteration=model.best_iteration)
    calib_df_temp = calib_df.copy()
    calib_df_temp['raw_prob'] = raw_calib_probs

    calibrators = {}
    purposes = calib_df_temp['PURPOSE_CD'].astype(str).unique()

    for p in purposes:
        sub_mask = calib_df_temp['PURPOSE_CD'].astype(str) == p
        sub_probs = calib_df_temp.loc[sub_mask, 'raw_prob'].values
        sub_y = calib_df_temp.loc[sub_mask, 'target'].values
        
        # Isotonic regression mapping raw probabilities -> empirical probabilities
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
        iso.fit(sub_probs, sub_y)
        calibrators[p] = iso
        print(f"      [{p:<5}] Isotonic Calibrator Fitted on {len(sub_y):,} samples.")

    # Save artifacts
    joblib.dump(model, "msd_lgb_prob_model.pkl")
    joblib.dump(calibrators, "msd_stratified_calibrators.pkl")
    joblib.dump({'feature_cols': feature_cols, 'cat_cols': cat_cols}, "model_metadata.pkl")

    print("\n Saved Isotonic calibrators successfully.")
    return model, calibrators

if __name__ == "__main__":
    train_df, calib_df, val_df = load_data("train_msd.parquet", "calib_msd.parquet", "val_msd.parquet")
    features, categoricals = get_feature_lists(train_df)
    train_isotonic_calibrated_pipeline(train_df, calib_df, features, categoricals)