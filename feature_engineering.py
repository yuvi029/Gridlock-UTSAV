"""
Phase 2: Advanced Feature Engineering
======================================
Geospatial decoding, cyclical temporal encoding, interaction features,
and leakage-safe target encoding.
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import KFold
warnings.filterwarnings('ignore')


# ==========================================================================
# Geohash decoder (pure Python, no external dependency)
# ==========================================================================
_BASE32 = '0123456789bcdefghjkmnpqrstuvwxyz'
_DECODEMAP = {c: i for i, c in enumerate(_BASE32)}


def decode_geohash(geohash_str):
    """Decode a geohash string into (latitude, longitude) coordinates."""
    if pd.isna(geohash_str):
        return np.nan, np.nan
    lat_interval = (-90.0, 90.0)
    lon_interval = (-180.0, 180.0)
    is_even = True
    for c in str(geohash_str):
        cd = _DECODEMAP.get(c, 0)
        for mask in [16, 8, 4, 2, 1]:
            if is_even:
                mid = (lon_interval[0] + lon_interval[1]) / 2
                if cd & mask:
                    lon_interval = (mid, lon_interval[1])
                else:
                    lon_interval = (lon_interval[0], mid)
            else:
                mid = (lat_interval[0] + lat_interval[1]) / 2
                if cd & mask:
                    lat_interval = (mid, lat_interval[1])
                else:
                    lat_interval = (lat_interval[0], mid)
            is_even = not is_even
    lat = (lat_interval[0] + lat_interval[1]) / 2
    lon = (lon_interval[0] + lon_interval[1]) / 2
    return lat, lon


def decode_geohashes_batch(series):
    """Efficient geohash decoding: decode unique values once, then map."""
    str_series = series.astype(str)
    unique_hashes = str_series.unique()
    # Decode only unique geohashes (much faster: ~1249 vs ~77299)
    decode_map = {}
    for gh in unique_hashes:
        lat, lon = decode_geohash(gh)
        decode_map[gh] = (lat, lon)
    lats = str_series.map(lambda x: decode_map.get(x, (np.nan, np.nan))[0])
    lons = str_series.map(lambda x: decode_map.get(x, (np.nan, np.nan))[1])
    return lats.values, lons.values


# ==========================================================================
# Feature engineering pipeline
# ==========================================================================
def engineer_features(train, test):
    """
    Apply all feature engineering to train and test DataFrames.
    
    IMPORTANT: Target encoding statistics are computed from train only
    and then mapped to test to prevent data leakage.
    
    Parameters
    ----------
    train : pd.DataFrame
        Cleaned training data (must have 'demand' column).
    test : pd.DataFrame
        Cleaned test data.
    
    Returns
    -------
    train : pd.DataFrame
        Feature-enriched training data.
    test : pd.DataFrame
        Feature-enriched test data.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: FEATURE ENGINEERING")
    print("=" * 60)
    
    train = train.copy()
    test = test.copy()
    
    # ------------------------------------------------------------------
    # 1. GEOSPATIAL FEATURES
    # ------------------------------------------------------------------
    print("\n[1/6] Decoding geohashes...")
    
    # Decode to lat/lon
    train['latitude'], train['longitude'] = decode_geohashes_batch(train['geohash'])
    test['latitude'], test['longitude'] = decode_geohashes_batch(test['geohash'])
    
    # Geohash hierarchies (coarser spatial groupings)
    train['geohash_4'] = train['geohash'].astype(str).str[:4]
    train['geohash_5'] = train['geohash'].astype(str).str[:5]
    test['geohash_4'] = test['geohash'].astype(str).str[:4]
    test['geohash_5'] = test['geohash'].astype(str).str[:5]
    
    # Distance to city center (median of all coordinates)
    center_lat = train['latitude'].median()
    center_lon = train['longitude'].median()
    for df in [train, test]:
        df['dist_to_center'] = np.sqrt(
            (df['latitude'] - center_lat) ** 2 + (df['longitude'] - center_lon) ** 2
        )
    print(f"  Center: ({center_lat:.4f}, {center_lon:.4f})")
    print(f"  Train geohash_4 groups: {train['geohash_4'].nunique()}")
    print(f"  Train geohash_5 groups: {train['geohash_5'].nunique()}")
    
    # ------------------------------------------------------------------
    # 2. CYCLICAL TEMPORAL ENCODING
    # ------------------------------------------------------------------
    print("[2/6] Creating cyclical temporal features...")
    
    for df in [train, test]:
        # Hour cyclical (period = 24)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
        
        # Minute cyclical (period = 60)
        df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60.0)
        df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60.0)
        
        # Time slot cyclical (period = 96 = full day)
        df['slot_sin'] = np.sin(2 * np.pi * df['time_slot'] / 96.0)
        df['slot_cos'] = np.cos(2 * np.pi * df['time_slot'] / 96.0)
        
        # Day of week proxy (day column is just 48, 49 — encode cyclically)
        # Since we only have 2 days, use day directly as well
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 7.0)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 7.0)
    
    # ------------------------------------------------------------------
    # 3. TEMPORAL FLAGS
    # ------------------------------------------------------------------
    print("[3/6] Creating temporal flags...")
    
    for df in [train, test]:
        # Rush hour (morning: 7-10, evening: 16-19)
        df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 10) | 
                               (df['hour'] >= 16) & (df['hour'] <= 19)).astype(np.int8)
        
        # Night hours (22-6)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(np.int8)
        
        # Early morning low-demand window (from data analysis: hours 17-20 lowest)
        df['is_low_demand_window'] = ((df['hour'] >= 17) & (df['hour'] <= 20)).astype(np.int8)
        
        # Peak demand window (hours 10-14 from data analysis)
        df['is_peak_window'] = ((df['hour'] >= 10) & (df['hour'] <= 14)).astype(np.int8)
        
        # Fractional hour for smooth temporal representation
        df['frac_hour'] = df['hour'] + df['minute'] / 60.0
    
    # ------------------------------------------------------------------
    # 4. INTERACTION FEATURES
    # ------------------------------------------------------------------
    print("[4/6] Creating interaction features...")
    
    for df in [train, test]:
        # Road + Weather interaction (categorical)
        df['road_weather'] = df['RoadType'].astype(str) + '_' + df['Weather'].astype(str)
        
        # Road + LargeVehicles interaction
        df['road_vehicles'] = df['RoadType'].astype(str) + '_' + df['LargeVehicles'].astype(str)
        
        # Road + Lanes (numeric interaction)
        road_map = {'Residential': 0, 'Street': 1, 'Highway': 2}
        df['road_encoded'] = df['RoadType'].astype(str).map(road_map).fillna(0).astype(np.int8)
        df['road_capacity'] = df['road_encoded'] * df['NumberofLanes']
        
        # Lanes squared (captures nonlinear relationship: 4-5 lanes = very different)
        df['lanes_sq'] = df['NumberofLanes'] ** 2
        
        # Is highway (strong binary signal from data analysis)
        df['is_highway'] = (df['RoadType'].astype(str) == 'Highway').astype(np.int8)
        
        # Is street
        df['is_street'] = (df['RoadType'].astype(str) == 'Street').astype(np.int8)
        
        # High capacity indicator (4+ lanes)
        df['is_high_capacity'] = (df['NumberofLanes'] >= 4).astype(np.int8)
        
        # Bad weather flag
        df['is_bad_weather'] = df['Weather'].astype(str).isin(['Rainy', 'Snowy', 'Foggy']).astype(np.int8)
        
        # Temperature x bad weather interaction
        df['temp_bad_weather'] = df['Temperature'] * df['is_bad_weather']
        
        # Temperature x road type
        df['temp_highway'] = df['Temperature'] * df['is_highway']
        
        # Landmarks binary
        df['has_landmark'] = (df['Landmarks'].astype(str) == 'Yes').astype(np.int8)
        
        # Large vehicles binary
        df['large_vehicles_allowed'] = (df['LargeVehicles'].astype(str) == 'Allowed').astype(np.int8)
        
        # Combined road score (captures the hierarchy: Highway > Street > Residential)
        df['road_score'] = df['road_encoded'] * 3 + df['NumberofLanes']
    
    # ------------------------------------------------------------------
    # 4.5 LAG FEATURES (The most important feature)
    # ------------------------------------------------------------------
    print("[4.5/6] Creating lag features...")
    
    # We create a lookup for exactly 1 day ago based on geohash and timestamp
    lookup = train[['day', 'geohash', 'timestamp', 'demand']].copy()
    lookup['day'] = lookup['day'] + 1  # Shift forward by 1 day to match tomorrow
    lookup.rename(columns={'demand': 'lag_demand_1d'}, inplace=True)
    
    # Merge onto train and test
    train = train.merge(lookup, on=['day', 'geohash', 'timestamp'], how='left')
    test = test.merge(lookup, on=['day', 'geohash', 'timestamp'], how='left')
    
    # For day 48 in train, it won't find day 47 (will be NaN). 
    # Tree models like CatBoost/LightGBM will handle the NaN naturally.
    
    # ------------------------------------------------------------------
    # 5. TARGET ENCODING (train-only statistics)
    # ------------------------------------------------------------------
    print("[5/6] Computing target encodings (leakage-safe)...")
    
    global_mean = train['demand'].mean()
    
    # Helper function for smoothed target encoding
    def target_encode(train_df, test_df, group_col, target='demand', 
                      min_samples=20, smoothing=10, n_splits=5):
        """
        Out-Of-Fold smoothed target encoding to prevent leakage.
        """
        feat_name = f'te_{group_col}'
        
        train_df[feat_name] = np.nan
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kf.split(train_df):
            X_tr, X_val = train_df.iloc[train_idx], train_df.iloc[val_idx]
            stats = X_tr.groupby(group_col)[target].agg(['mean', 'count'])
            smooth = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
            val_key = X_val[group_col]
            train_df.loc[val_idx, feat_name] = val_key.map(smooth).fillna(global_mean).values
            
        stats = train_df.groupby(group_col)[target].agg(['mean', 'count'])
        smooth = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
        test_key = test_df[group_col]
        test_df[feat_name] = test_key.map(smooth).fillna(global_mean).astype(np.float32)
        train_df[feat_name] = train_df[feat_name].astype(np.float32)
        
        return train_df, test_df
    
    # Geohash-level target encoding (the most important one)
    train, test = target_encode(train, test, 'geohash', smoothing=10)
    
    # Geohash_4 level (coarser spatial)
    train, test = target_encode(train, test, 'geohash_4', smoothing=20)
    
    # Geohash_5 level
    train, test = target_encode(train, test, 'geohash_5', smoothing=15)
    
    # RoadType target encoding
    train, test = target_encode(train, test, 'RoadType', smoothing=50)
    
    # Hour-level target encoding
    train['hour_str'] = train['hour'].astype(str)
    test['hour_str'] = test['hour'].astype(str)
    train, test = target_encode(train, test, 'hour_str', smoothing=50)
    train.drop('hour_str', axis=1, inplace=True)
    test.drop('hour_str', axis=1, inplace=True)
    train.rename(columns={'te_hour_str': 'te_hour'}, inplace=True)
    test.rename(columns={'te_hour_str': 'te_hour'}, inplace=True)
    
    # Geohash x hour interaction target encoding (captures per-location daily patterns)
    train['gh_hour'] = train['geohash'].astype(str) + '_' + train['hour'].astype(str)
    test['gh_hour'] = test['geohash'].astype(str) + '_' + test['hour'].astype(str)
    train, test = target_encode(train, test, 'gh_hour', smoothing=5)
    train.drop('gh_hour', axis=1, inplace=True)
    test.drop('gh_hour', axis=1, inplace=True)
    train.rename(columns={'te_gh_hour': 'te_geohash_hour'}, inplace=True)
    test.rename(columns={'te_gh_hour': 'te_geohash_hour'}, inplace=True)
    
    # RoadType x hour interaction
    train['rt_hour'] = train['RoadType'].astype(str) + '_' + train['hour'].astype(str)
    test['rt_hour'] = test['RoadType'].astype(str) + '_' + test['hour'].astype(str)
    train, test = target_encode(train, test, 'rt_hour', smoothing=20)
    train.drop('rt_hour', axis=1, inplace=True)
    test.drop('rt_hour', axis=1, inplace=True)
    train.rename(columns={'te_rt_hour': 'te_roadtype_hour'}, inplace=True)
    test.rename(columns={'te_rt_hour': 'te_roadtype_hour'}, inplace=True)
    
    # ------------------------------------------------------------------
    # 6. GEOHASH FREQUENCY ENCODING
    # ------------------------------------------------------------------
    print("[6/6] Frequency encoding geohashes...")
    
    geo_counts = train['geohash'].value_counts().to_dict()
    total = len(train)
    for df in [train, test]:
        df['geohash_freq'] = df['geohash'].astype(str).map(geo_counts).fillna(0) / total
    
    # ------------------------------------------------------------------
    # Convert new categorical columns
    # ------------------------------------------------------------------
    new_cats = ['geohash_4', 'geohash_5', 'road_weather', 'road_vehicles']
    for col in new_cats:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')
    
    print(f"\n[OK] Phase 2 complete.")
    print(f"  Train features: {train.shape[1]} columns")
    print(f"  Test features:  {test.shape[1]} columns")
    
    return train, test


if __name__ == '__main__':
    from data_loader import load_and_clean
    train, test = load_and_clean()
    train, test = engineer_features(train, test)
    print("\n--- Feature columns ---")
    for i, col in enumerate(train.columns):
        print(f"  {i+1:2d}. {col:30s} {str(train[col].dtype):>10s}")
    print(f"\n--- Train shape: {train.shape}, Test shape: {test.shape} ---")
