"""
Phase 1: Robust Data Ingestion & Cleaning
==========================================
Loads train.csv and test.csv, handles missing values, clips outliers,
and creates foundational temporal columns.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def parse_timestamp(ts_str):
    """Parse 'H:M' timestamp strings into hour and minute."""
    parts = str(ts_str).split(':')
    hour = int(parts[0])
    minute = int(parts[1]) if len(parts) > 1 else 0
    return hour, minute


def load_and_clean(train_path='dataset/train.csv', test_path='dataset/test.csv',
                   clip_demand=True, clip_lower=0.01, clip_upper=0.99):
    """
    Load and clean train/test datasets.
    
    Parameters
    ----------
    train_path : str
        Path to training CSV.
    test_path : str
        Path to test CSV.
    clip_demand : bool
        Whether to clip demand outliers.
    clip_lower : float
        Lower percentile for clipping (0-1).
    clip_upper : float
        Upper percentile for clipping (0-1).
    
    Returns
    -------
    train : pd.DataFrame
        Cleaned training data with demand.
    test : pd.DataFrame
        Cleaned test data without demand.
    """
    print("=" * 60)
    print("PHASE 1: DATA INGESTION & CLEANING")
    print("=" * 60)
    
    # ------------------------------------------------------------------
    # 1. Load raw data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading raw data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(f"  Train: {train.shape}, Test: {test.shape}")
    
    # ------------------------------------------------------------------
    # 2. Parse timestamps -> hour, minute, time_slot, time_order
    # ------------------------------------------------------------------
    print("[2/5] Parsing timestamps...")
    for df in [train, test]:
        parsed = df['timestamp'].apply(lambda x: pd.Series(parse_timestamp(x)))
        df['hour'] = parsed[0].astype(np.int8)
        df['minute'] = parsed[1].astype(np.int8)
        df['time_slot'] = (df['hour'] * 4 + df['minute'] // 15).astype(np.int8)
        df['time_order'] = (df['day'] * 96 + df['time_slot']).astype(np.int32)
    
    print(f"  Train time_order range: {train['time_order'].min()} -> {train['time_order'].max()}")
    print(f"  Test  time_order range: {test['time_order'].min()} -> {test['time_order'].max()}")
    assert test['time_order'].min() > train['time_order'].max(), \
        "CRITICAL: Test data leaks into train temporal range!"
    print("  [OK] Confirmed: zero temporal overlap between train and test")
    
    # ------------------------------------------------------------------
    # 3. Handle missing values
    # ------------------------------------------------------------------
    print("[3/5] Imputing missing values...")
    
    # Temperature: median imputation (computed on train, applied to both)
    temp_median = train['Temperature'].median()
    train_temp_nulls = train['Temperature'].isnull().sum()
    test_temp_nulls = test['Temperature'].isnull().sum()
    train['Temperature'] = train['Temperature'].fillna(temp_median)
    test['Temperature'] = test['Temperature'].fillna(temp_median)
    print(f"  Temperature: filled {train_temp_nulls} train + {test_temp_nulls} test nulls with median={temp_median:.2f}")
    
    # RoadType: impute with mode ("Residential" — 89.5% of data)
    road_mode = train['RoadType'].mode()[0]
    train_road_nulls = train['RoadType'].isnull().sum()
    test_road_nulls = test['RoadType'].isnull().sum()
    train['RoadType'] = train['RoadType'].fillna(road_mode)
    test['RoadType'] = test['RoadType'].fillna(road_mode)
    print(f"  RoadType: filled {train_road_nulls} train + {test_road_nulls} test nulls with mode='{road_mode}'")
    
    # Weather: impute with mode ("Sunny")
    weather_mode = train['Weather'].mode()[0]
    train_weather_nulls = train['Weather'].isnull().sum()
    test_weather_nulls = test['Weather'].isnull().sum()
    train['Weather'] = train['Weather'].fillna(weather_mode)
    test['Weather'] = test['Weather'].fillna(weather_mode)
    print(f"  Weather: filled {train_weather_nulls} train + {test_weather_nulls} test nulls with mode='{weather_mode}'")
    
    # NumberofLanes: median (no nulls found, but defensive)
    if train['NumberofLanes'].isnull().any():
        lanes_median = train['NumberofLanes'].median()
        train['NumberofLanes'] = train['NumberofLanes'].fillna(lanes_median)
        test['NumberofLanes'] = test['NumberofLanes'].fillna(lanes_median)
    
    # ------------------------------------------------------------------
    # 4. Clip demand outliers
    # ------------------------------------------------------------------
    if clip_demand:
        print("[4/5] Clipping demand outliers...")
        lower_val = train['demand'].quantile(clip_lower)
        upper_val = train['demand'].quantile(clip_upper)
        before_mean = train['demand'].mean()
        train['demand'] = train['demand'].clip(lower=lower_val, upper=upper_val)
        after_mean = train['demand'].mean()
        print(f"  Clipped to [{lower_val:.6f}, {upper_val:.6f}]")
        print(f"  Mean demand: {before_mean:.6f} -> {after_mean:.6f}")
    else:
        print("[4/5] Skipping demand clipping (disabled)")
    
    # ------------------------------------------------------------------
    # 5. Data type optimization
    # ------------------------------------------------------------------
    print("[5/5] Optimizing data types...")
    cat_cols = ['geohash', 'RoadType', 'Weather', 'LargeVehicles', 'Landmarks']
    for col in cat_cols:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')
    
    # Final null check
    train_nulls = train.isnull().sum().sum()
    test_nulls = test.isnull().sum().sum()
    print(f"\n  Final null counts - Train: {train_nulls}, Test: {test_nulls}")
    
    print(f"\n[OK] Phase 1 complete. Train: {train.shape}, Test: {test.shape}")
    return train, test


if __name__ == '__main__':
    train, test = load_and_clean()
    print("\n--- Train columns ---")
    print(train.dtypes)
    print("\n--- Train head ---")
    print(train.head(3).to_string())
    print("\n--- Test head ---")
    print(test.head(3).to_string())
