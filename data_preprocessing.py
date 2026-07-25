import time
import numpy as np
import pandas as pd

# Optional: try importing holidays package for dynamic Easter/Pasquetta calculation
try:
    import holidays
    HAS_HOLIDAYS_LIB = True
except ImportError:
    HAS_HOLIDAYS_LIB = False
    from dateutil.easter import easter

def get_italian_holidays(years):
    """
    Returns a set of holiday dates in 'YYYYMMDD' string format for the requested years.
    Includes fixed national holidays + variable Easter Monday (Pasquetta).
    """
    holiday_dates = set()
    
    if HAS_HOLIDAYS_LIB:
        # Uses python-holidays library if installed
        it_holidays = holidays.Italy(years=years)
        for date_obj in it_holidays.keys():
            holiday_dates.add(date_obj.strftime('%Y%m%d'))
    else:
        # Pure Python fallback for Italian National Holidays
        for yr in years:
            # Fixed Italian National Holidays
            fixed_dates = [
                f"{yr}0101", # Capodanno
                f"{yr}0106", # Epifania
                f"{yr}0425", # Liberazione
                f"{yr}0501", # Festa del Lavoro
                f"{yr}0602", # Festa della Repubblica
                f"{yr}0815", # Ferragosto
                f"{yr}1101", # Ognissanti
                f"{yr}1208", # Immacolata
                f"{yr}1225", # Natale
                f"{yr}1226", # Santo Stefano
            ]
            holiday_dates.update(fixed_dates)
            
            # Dynamic Pasquetta (Easter Monday)
            e_date = easter(yr)
            pasquetta = e_date + pd.Timedelta(days=1)
            holiday_dates.add(pasquetta.strftime('%Y%m%d'))
            
    return holiday_dates


def process_and_save_data(input_path: str, train_out: str, calib_out: str, val_out: str):
    print(f"[1/4] Loading Parquet dataset from: '{input_path}'...")
    start_time = time.time()
    
    df = pd.read_parquet(input_path)
    print(f"      Loaded {len(df):,} raw rows.")

    # 1. Filter out 0 MW bids
    df = df[df['QUANTITY_NO'] > 0.0].reset_index(drop=True)

    # 2. Strict Filter: ZONE_CD == 'NORD' and SCOPE == 'GR1'
    if 'ZONE_CD' in df.columns and 'SCOPE' in df.columns:
        df = df[(df['ZONE_CD'] == 'NORD') & (df['SCOPE'] == 'GR1')].reset_index(drop=True)
        print(f"      Rows after filtering NORD & GR1: {len(df):,}")

    # 3. Parse Dates
    if 'BID_OFFER_DATE_DT' in df.columns:
        # Format string date column as YYYYMMDD
        date_series_str = df['BID_OFFER_DATE_DT'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(8)
        df['date'] = pd.to_datetime(date_series_str, format='%Y%m%d', errors='coerce')
        df['day_of_week'] = df['date'].dt.dayofweek.fillna(-1).astype('int32') # 0 = Mon, 5 = Sat, 6 = Sun
        df['month'] = df['date'].dt.month.fillna(-1).astype('int32')

    # =========================================================================
    # NEW: Filter Weekends & Italian National Holidays
    # =========================================================================
    print("[2/4] Filtering Weekends and Italian Holidays...")
    initial_count = len(df)
    
    # a) Exclude Weekends (Saturday=5, Sunday=6)
    df = df[df['day_of_week'] < 5].reset_index(drop=True)
    weekend_removed = initial_count - len(df)
    
    # b) Exclude Italian Holidays
    unique_years = df['date'].dt.year.dropna().unique().astype(int)
    it_holiday_strings = get_italian_holidays(unique_years)
    
    # Mask out matching YYYYMMDD dates
    date_str_formatted = df['date'].dt.strftime('%Y%m%d')
    is_holiday = date_str_formatted.isin(it_holiday_strings)
    df = df[~is_holiday].reset_index(drop=True)
    
    holiday_removed = initial_count - weekend_removed - len(df)
    print(f"      Removed {weekend_removed:,} weekend rows & {holiday_removed:,} holiday rows.")
    print(f"      Remaining Working Days Rows: {len(df):,}")

    # 4. Binary Target
    df['target'] = (df['AWARDED_QUANTITY_NO'] > 0).astype('int32')

    # 5. Enhanced Feature Engineering
    if 'ENERGY_PRICE_NO' in df.columns and 'PUN' in df.columns:
        df['PRICE_SPREAD_PUN'] = df['ENERGY_PRICE_NO'] - df['PUN']
        df['PRICE_RATIO_PUN'] = df['ENERGY_PRICE_NO'] / (df['PUN'] + 1e-5)
        
        # Directional Spread (BID vs OFF)
        if 'PURPOSE_CD' in df.columns:
            is_bid = df['PURPOSE_CD'].astype(str).str.upper() == 'BID'
            df['SIGNED_SPREAD_PUN'] = np.where(
                is_bid,
                df['PUN'] - df['ENERGY_PRICE_NO'],
                df['ENERGY_PRICE_NO'] - df['PUN']
            )

    df['LOG_QUANTITY'] = np.log1p(df['QUANTITY_NO'])

    # 6. Categoricals (Exclude constant ZONE_CD and SCOPE)
    cat_cols = ['PURPOSE_CD', 'TYPE_CD', 'MARKET_CD']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # 7. Chronological Sort
    sort_cols = [c for c in ['date', 'INTERVAL_NO', 'PERIOD'] if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols).reset_index(drop=True)

    # 8. Three-way Chronological Split (70% Train / 15% Calib / 15% Val)
    print(f"[3/4] Performing 70% Train / 15% Calibration / 15% Validation split...")
    n = len(df)
    train_idx = int(n * 0.70)
    calib_idx = int(n * 0.85)

    train_df = df.iloc[:train_idx]
    calib_df = df.iloc[train_idx:calib_idx]
    val_df = df.iloc[calib_idx:]

    print(f"[4/4] Saving filtered splits to disk...")
    train_df.to_parquet(train_out, index=False)
    calib_df.to_parquet(calib_out, index=False)
    val_df.to_parquet(val_out, index=False)
    print(f"      Completed in {time.time() - start_time:.2f}s.")

if __name__ == "__main__":
    process_and_save_data(
        input_path="msd_with_pun.parquet",
        train_out="train_msd.parquet",
        calib_out="calib_msd.parquet",
        val_out="val_msd.parquet"
    )