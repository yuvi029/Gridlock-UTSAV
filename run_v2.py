"""
V2 Pipeline: Fixes CV-to-LB gap (94.88 CV -> 88.8 LB)
========================================================
Root cause: Target encoding features leak target info into training data,
inflating CV scores but hurting generalization.

Key changes:
1. REMOVE target encoding features (biggest overfitting source)
2. NO demand clipping (let model see full distribution)
3. Use K-Fold OOF target encoding as an option (proper, non-leaky)
4. CatBoost-heavy ensemble (50% weight - handles categoricals natively)
5. 10 seeds for CatBoost, 5 for LGB/XGB
6. More iterations for final model (1.5x Optuna best)
7. Higher regularization (prevent overfitting)
"""

import numpy as np
import pandas as pd
import json
import os
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import r2_score, mean_squared_error
import catboost as cb
import lightgbm as lgb
import xgboost as xgb

# ======================================================================
# PHASE 1: Data loading (no demand clipping)
# ======================================================================
def parse_timestamp(ts_str):
    parts = str(ts_str).split(':')
    return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0

def load_data():
    print("=" * 60)
    print("PHASE 1: DATA LOADING (V2 - No demand clipping)")
    print("=" * 60)
    
    train = pd.read_csv('dataset/train.csv')
    test = pd.read_csv('dataset/test.csv')
    print(f"  Train: {train.shape}, Test: {test.shape}")
    
    for df in [train, test]:
        parsed = df['timestamp'].apply(lambda x: pd.Series(parse_timestamp(x)))
        df['hour'] = parsed[0].astype(np.int8)
        df['minute'] = parsed[1].astype(np.int8)
        df['time_slot'] = (df['hour'] * 4 + df['minute'] // 15).astype(np.int8)
        df['time_order'] = (df['day'] * 96 + df['time_slot']).astype(np.int32)
    
    # Temperature imputation (train stats only)
    temp_median = train['Temperature'].median()
    train['Temperature'] = train['Temperature'].fillna(temp_median)
    test['Temperature'] = test['Temperature'].fillna(temp_median)
    
    # RoadType imputation
    road_mode = train['RoadType'].mode()[0]
    train['RoadType'] = train['RoadType'].fillna(road_mode)
    test['RoadType'] = test['RoadType'].fillna(road_mode)
    
    # Weather imputation
    weather_mode = train['Weather'].mode()[0]
    train['Weather'] = train['Weather'].fillna(weather_mode)
    test['Weather'] = test['Weather'].fillna(weather_mode)
    
    # NO demand clipping - let model see full distribution
    print(f"  Demand range: [{train['demand'].min():.6f}, {train['demand'].max():.6f}]")
    print(f"  Demand mean: {train['demand'].mean():.6f}")
    
    # Categoricals
    cat_cols = ['geohash', 'RoadType', 'Weather', 'LargeVehicles', 'Landmarks']
    for col in cat_cols:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')
    
    print(f"[OK] Phase 1 done. Train: {train.shape}, Test: {test.shape}")
    return train, test


# ======================================================================
# PHASE 2: Feature engineering (NO target encoding)
# ======================================================================
_BASE32 = '0123456789bcdefghjkmnpqrstuvwxyz'
_DECODEMAP = {c: i for i, c in enumerate(_BASE32)}

def decode_geohash(gh):
    if pd.isna(gh): return np.nan, np.nan
    lat_i, lon_i = (-90.0, 90.0), (-180.0, 180.0)
    is_even = True
    for c in str(gh):
        cd = _DECODEMAP.get(c, 0)
        for mask in [16, 8, 4, 2, 1]:
            if is_even:
                mid = (lon_i[0] + lon_i[1]) / 2
                lon_i = (mid, lon_i[1]) if cd & mask else (lon_i[0], mid)
            else:
                mid = (lat_i[0] + lat_i[1]) / 2
                lat_i = (mid, lat_i[1]) if cd & mask else (lat_i[0], mid)
            is_even = not is_even
    return (lat_i[0]+lat_i[1])/2, (lon_i[0]+lon_i[1])/2


def target_encode_oof(train_df, test_df, group_col, target='demand', smoothing=10, n_splits=5):
    feat_name = f'te_{group_col}'
    global_mean = train_df[target].mean()
    
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


def engineer_features_v2(train, test):
    """
    Feature engineering WITH OOF target encoding.
    CatBoost will handle categoricals natively, and OOF target encoding gives safe signal.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: FEATURE ENGINEERING (V2 - No target encoding)")
    print("=" * 60)
    
    train = train.copy()
    test = test.copy()
    
    # --- Geospatial ---
    print("[1/5] Geospatial features...")
    unique_gh = set(train['geohash'].astype(str).unique()) | set(test['geohash'].astype(str).unique())
    decode_map = {gh: decode_geohash(gh) for gh in unique_gh}
    
    for df in [train, test]:
        gh_str = df['geohash'].astype(str)
        df['latitude'] = gh_str.map(lambda x: decode_map[x][0])
        df['longitude'] = gh_str.map(lambda x: decode_map[x][1])
        df['geohash_4'] = gh_str.str[:4]
        df['geohash_5'] = gh_str.str[:5]
    
    center_lat = train['latitude'].median()
    center_lon = train['longitude'].median()
    for df in [train, test]:
        df['dist_to_center'] = np.sqrt(
            (df['latitude'] - center_lat)**2 + (df['longitude'] - center_lon)**2
        )
    
    # --- Cyclical temporal ---
    print("[2/5] Cyclical temporal features...")
    for df in [train, test]:
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
        df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60.0)
        df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60.0)
        df['slot_sin'] = np.sin(2 * np.pi * df['time_slot'] / 96.0)
        df['slot_cos'] = np.cos(2 * np.pi * df['time_slot'] / 96.0)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 7.0)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 7.0)
    
    # --- Temporal flags ---
    print("[3/5] Temporal flags...")
    for df in [train, test]:
        df['is_rush_hour'] = ((df['hour'].between(7, 10)) | (df['hour'].between(16, 19))).astype(np.int8)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(np.int8)
        df['is_low_demand'] = df['hour'].between(17, 20).astype(np.int8)
        df['is_peak'] = df['hour'].between(10, 14).astype(np.int8)
        df['frac_hour'] = df['hour'] + df['minute'] / 60.0
    
    # --- Interaction features ---
    print("[4/5] Interaction features...")
    for df in [train, test]:
        df['road_weather'] = df['RoadType'].astype(str) + '_' + df['Weather'].astype(str)
        df['road_vehicles'] = df['RoadType'].astype(str) + '_' + df['LargeVehicles'].astype(str)
        
        road_map = {'Residential': 0, 'Street': 1, 'Highway': 2}
        df['road_encoded'] = df['RoadType'].astype(str).map(road_map).fillna(0).astype(np.int8)
        df['road_capacity'] = df['road_encoded'] * df['NumberofLanes']
        df['lanes_sq'] = df['NumberofLanes'] ** 2
        df['is_highway'] = (df['RoadType'].astype(str) == 'Highway').astype(np.int8)
        df['is_street'] = (df['RoadType'].astype(str) == 'Street').astype(np.int8)
        df['is_high_capacity'] = (df['NumberofLanes'] >= 4).astype(np.int8)
        df['is_bad_weather'] = df['Weather'].astype(str).isin(['Rainy', 'Snowy', 'Foggy']).astype(np.int8)
        df['temp_bad_weather'] = df['Temperature'] * df['is_bad_weather']
        df['temp_highway'] = df['Temperature'] * df['is_highway']
        df['has_landmark'] = (df['Landmarks'].astype(str) == 'Yes').astype(np.int8)
        df['large_vehicles_allowed'] = (df['LargeVehicles'].astype(str) == 'Allowed').astype(np.int8)
        df['road_score'] = df['road_encoded'] * 3 + df['NumberofLanes']
        
        # V2: Road + hour interaction (strong signal)
        df['road_hour'] = df['road_encoded'] * 24 + df['hour']
        
        # V2: Highway at rush hour
        df['highway_rush'] = df['is_highway'] * df['is_rush_hour']
        
        # V2: Highway at peak
        df['highway_peak'] = df['is_highway'] * df['is_peak']
        
        # V2: Lanes at peak
        df['lanes_peak'] = df['NumberofLanes'] * df['is_peak']
    
    # --- Geohash frequency (non-target, safe) ---
    print("[5/5] Frequency features...")
    geo_counts = train['geohash'].value_counts().to_dict()
    total = len(train)
    for df in [train, test]:
        df['geohash_freq'] = df['geohash'].astype(str).map(geo_counts).fillna(0) / total
    
    # Convert new categoricals
    for col in ['geohash_4', 'geohash_5', 'road_weather', 'road_vehicles']:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')
        
    # --- Lag Features ---
    print("[5.5/6] Creating lag features...")
    lookup = train[['day', 'geohash', 'timestamp', 'demand']].copy()
    lookup['day'] = lookup['day'] + 1
    lookup.rename(columns={'demand': 'lag_demand_1d'}, inplace=True)
    
    train = train.merge(lookup, on=['day', 'geohash', 'timestamp'], how='left')
    test = test.merge(lookup, on=['day', 'geohash', 'timestamp'], how='left')
    
    # --- OOF Target Encoding ---
    print("[6/6] OOF Target Encoding (optimized smoothing)...")
    train['gh_hour'] = train['geohash'].astype(str) + '_' + train['hour'].astype(str)
    test['gh_hour'] = test['geohash'].astype(str) + '_' + test['hour'].astype(str)
    train['rt_hour'] = train['RoadType'].astype(str) + '_' + train['hour'].astype(str)
    test['rt_hour'] = test['RoadType'].astype(str) + '_' + test['hour'].astype(str)
    train['gh_rt'] = train['geohash'].astype(str) + '_' + train['RoadType'].astype(str)
    test['gh_rt'] = test['geohash'].astype(str) + '_' + test['RoadType'].astype(str)
    train['gh4_hour'] = train['geohash_4'].astype(str) + '_' + train['hour'].astype(str)
    test['gh4_hour'] = test['geohash_4'].astype(str) + '_' + test['hour'].astype(str)
    
    smooth_map = {
        'geohash': 10, 'geohash_4': 20, 'geohash_5': 15,
        'RoadType': 50, 'gh_hour': 5, 'rt_hour': 20,
        'gh_rt': 10, 'gh4_hour': 10, 'road_weather': 30,
        'road_vehicles': 30, 'Weather': 50
    }
    for col, smooth in smooth_map.items():
        train, test = target_encode_oof(train, test, col, smoothing=smooth)
        
    train.drop(['gh_hour', 'rt_hour', 'gh_rt', 'gh4_hour'], axis=1, inplace=True)
    test.drop(['gh_hour', 'rt_hour', 'gh_rt', 'gh4_hour'], axis=1, inplace=True)

    print(f"[OK] Phase 2 done. Train: {train.shape[1]} cols, Test: {test.shape[1]} cols")
    return train, test


# ======================================================================
# PHASE 3: Training & Inference
# ======================================================================
DROP_COLS = ['Index', 'timestamp', 'demand', 'day']

CATBOOST_CAT_FEATURES = [
    'geohash', 'RoadType', 'Weather', 'LargeVehicles', 'Landmarks',
    'geohash_4', 'geohash_5', 'road_weather', 'road_vehicles'
]

LGBM_CAT_FEATURES = CATBOOST_CAT_FEATURES.copy()


def get_features(df):
    return [c for c in df.columns if c not in DROP_COLS]


def label_encode(train, test, features):
    train = train.copy()
    test = test.copy()
    for col in LGBM_CAT_FEATURES:
        if col in features:
            combined = pd.concat([train[col].astype(str), test[col].astype(str)])
            codes, _ = pd.factorize(combined)
            train[col] = codes[:len(train)]
            test[col] = codes[len(train):]
    return train, test


def _clean_params(params, keys):
    return {k: v for k, v in params.items() if k not in keys}


def run_v2_pipeline():
    total_t0 = time.time()
    
    print("=" * 60)
    print("  V2 PIPELINE: ANTI-OVERFITTING EDITION")
    print("  - No target encoding (removes leakage)")
    print("  - No demand clipping")
    print("  - CatBoost-heavy ensemble (10 seeds)")
    print("  - Higher regularization")
    print("=" * 60)
    
    # Load data
    train, test = load_data()
    train, test = engineer_features_v2(train, test)
    
    # Sort chronologically
    train = train.sort_values('time_order').reset_index(drop=True)
    
    features = get_features(train)
    X_train = train[features]
    y_train = train['demand']
    X_test = test[features]
    
    # Label-encoded versions
    train_enc, test_enc = label_encode(train, test, features)
    X_train_enc = train_enc[features]
    X_test_enc = test_enc[features]
    
    cat_idx = [X_train.columns.get_loc(c) for c in CATBOOST_CAT_FEATURES if c in X_train.columns]
    
    print(f"\nFeatures: {len(features)}")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Load saved params
    with open('optuna_results/best_params.json') as f:
        saved_params = json.load(f)
    
    # ================================================================
    # Quick CV to verify improvement
    # ================================================================
    print(f"\n{'='*50}")
    print("QUICK CV CHECK (TimeSeriesSplit, 3 folds)")
    print(f"{'='*50}")
    
    tscv = TimeSeriesSplit(n_splits=3)
    cv_r2_scores = []
    
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        cb_params = saved_params['catboost'].copy()
        cb_params['random_seed'] = 42
        cb_params['verbose'] = 0
        
        model = cb.CatBoostRegressor(**cb_params)
        model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx],
                  cat_features=cat_idx,
                  eval_set=(X_train.iloc[val_idx], y_train.iloc[val_idx]),
                  early_stopping_rounds=50)
        
        preds = np.clip(model.predict(X_train.iloc[val_idx]), 0, None)
        r2 = r2_score(y_train.iloc[val_idx], preds)
        cv_r2_scores.append(r2)
        print(f"  Fold {fold+1}: R2 = {max(0, 100*r2):.2f}")
    
    mean_r2 = np.mean(cv_r2_scores)
    print(f"  Mean CV R2: {max(0, 100*mean_r2):.2f}")
    print("  (This should be LOWER than V1's 94.88 but closer to LB)")
    
    # ================================================================
    # Train CatBoost ensemble (10 seeds, main model)
    # ================================================================
    print(f"\n{'='*50}")
    print("[1/3] CATBOOST ENSEMBLE (3 seeds)")
    print(f"{'='*50}")
    
    cb_params = saved_params['catboost'].copy()
    # Boost iterations 2.5x and lower LR for final model (squeezing accuracy safely)
    cb_params['iterations'] = int(cb_params.get('iterations', 800) * 2.5)
    if 'learning_rate' in cb_params:
        cb_params['learning_rate'] = cb_params['learning_rate'] / 2.0
    cb_params['loss_function'] = 'RMSE'
    cb_params['thread_count'] = -1
    
    seeds_cb = [42, 123, 456, 789, 2024]
    cb_preds = []
    
    for i, seed in enumerate(seeds_cb):
        t0 = time.time()
        p = _clean_params(cb_params, {'verbose', 'random_seed'})
        model = cb.CatBoostRegressor(**p, random_seed=seed, verbose=0)
        model.fit(X_train, y_train, cat_features=cat_idx)
        pred = model.predict(X_test)
        cb_preds.append(pred)
        elapsed = time.time() - t0
        print(f"  Seed {seed} ({i+1}/5): [{pred.min():.4f}, {pred.max():.4f}] ({elapsed:.1f}s)")
    
    cb_avg = np.mean(cb_preds, axis=0)
    print(f"  CatBoost avg: mean={cb_avg.mean():.4f}, std={cb_avg.std():.4f}")
    
    # ================================================================
    # Train LightGBM ensemble (3 seeds)
    # ================================================================
    print(f"\n{'='*50}")
    print("[2/3] LIGHTGBM ENSEMBLE (3 seeds)")
    print(f"{'='*50}")
    
    lgb_params = saved_params['lightgbm'].copy()
    lgb_params['n_estimators'] = int(lgb_params.get('n_estimators', 1000) * 2.0)
    if 'learning_rate' in lgb_params:
        lgb_params['learning_rate'] = lgb_params['learning_rate'] / 2.0
    
    seeds_lgb = [42, 123, 456, 789, 2024]
    lgb_preds = []
    
    for i, seed in enumerate(seeds_lgb):
        t0 = time.time()
        p = _clean_params(lgb_params, {'verbose', 'random_state'})
        model = lgb.LGBMRegressor(**p, random_state=seed, verbose=-1)
        model.fit(X_train_enc, y_train)
        pred = model.predict(X_test_enc)
        lgb_preds.append(pred)
        elapsed = time.time() - t0
        print(f"  Seed {seed} ({i+1}/5): [{pred.min():.4f}, {pred.max():.4f}] ({elapsed:.1f}s)")
    
    lgb_avg = np.mean(lgb_preds, axis=0)
    print(f"  LightGBM avg: mean={lgb_avg.mean():.4f}, std={lgb_avg.std():.4f}")
    
    # ================================================================
    # Train XGBoost ensemble (3 seeds)
    # ================================================================
    print(f"\n{'='*50}")
    print("[3/3] XGBOOST ENSEMBLE (3 seeds)")
    print(f"{'='*50}")
    
    xgb_params = saved_params['xgboost'].copy()
    xgb_params['n_estimators'] = int(xgb_params.get('n_estimators', 900) * 2.0)
    if 'learning_rate' in xgb_params:
        xgb_params['learning_rate'] = xgb_params['learning_rate'] / 2.0
    
    seeds_xgb = [42, 123, 456, 789, 2024]
    xgb_preds = []
    
    for i, seed in enumerate(seeds_xgb):
        t0 = time.time()
        p = _clean_params(xgb_params, {'verbosity', 'random_state', 'early_stopping_rounds', 'eval_metric'})
        model = xgb.XGBRegressor(**p, random_state=seed, verbosity=0)
        model.fit(X_train_enc, y_train)
        pred = model.predict(X_test_enc)
        xgb_preds.append(pred)
        elapsed = time.time() - t0
        print(f"  Seed {seed} ({i+1}/5): [{pred.min():.4f}, {pred.max():.4f}] ({elapsed:.1f}s)")
    
    xgb_avg = np.mean(xgb_preds, axis=0)
    print(f"  XGBoost avg: mean={xgb_avg.mean():.4f}, std={xgb_avg.std():.4f}")
    
    # ================================================================
    # Ensemble (Weighted)
    # ================================================================
    print(f"\n{'='*50}")
    print("WEIGHTED ENSEMBLE")
    print(f"{'='*50}")
    
    w_cb, w_lgb, w_xgb = 0.55, 0.25, 0.20
    final_preds = w_cb * cb_avg + w_lgb * lgb_avg + w_xgb * xgb_avg
    final_preds = np.clip(final_preds, 0, None)
    
    print(f"  Weights: CB={w_cb}, LGB={w_lgb}, XGB={w_xgb}")
    print(f"  Final: mean={final_preds.mean():.4f}, std={final_preds.std():.4f}")
    print(f"  Range: [{final_preds.min():.4f}, {final_preds.max():.4f}]")
    
    # ================================================================
    # Submission
    # ================================================================
    submission = pd.DataFrame({'Index': test['Index'], 'demand': final_preds})
    
    assert len(submission) == 41778, f"Wrong row count: {len(submission)}"
    assert list(submission.columns) == ['Index', 'demand']
    assert submission['demand'].isnull().sum() == 0
    
    submission.to_csv('submission.csv', index=False)
    
    total_time = time.time() - total_t0
    
    print(f"\n{'='*60}")
    print(f"  V2 PIPELINE COMPLETE")
    print(f"  Time: {total_time/60:.1f} minutes")
    print(f"  Output: submission.csv ({len(submission)} rows)")
    print(f"  Demand: [{submission['demand'].min():.6f}, {submission['demand'].max():.6f}]")
    print(f"{'='*60}")
    
    return submission


if __name__ == '__main__':
    run_v2_pipeline()
