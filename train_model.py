"""
Phase 3 & 4: Model Training with Optuna Hyperparameter Optimization
=====================================================================
Three models (CatBoost, LightGBM, XGBoost) tuned via Optuna using
TimeSeriesSplit for chronologically valid cross-validation.

Optimized for speed: 3-fold CV, capped iterations with early stopping.
"""

import numpy as np
import pandas as pd
import json
import os
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
import catboost as cb
import lightgbm as lgb
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ==========================================================================
# Feature selection
# ==========================================================================
DROP_COLS = ['Index', 'timestamp', 'demand', 'day']

CATBOOST_CAT_FEATURES = [
    'geohash', 'RoadType', 'Weather', 'LargeVehicles', 'Landmarks',
    'geohash_4', 'geohash_5', 'road_weather', 'road_vehicles'
]

LGBM_CAT_FEATURES = [
    'geohash', 'RoadType', 'Weather', 'LargeVehicles', 'Landmarks',
    'geohash_4', 'geohash_5', 'road_weather', 'road_vehicles'
]


def get_feature_columns(df):
    """Get list of feature columns."""
    return [c for c in df.columns if c not in DROP_COLS]


def prepare_for_lgbm_xgb(train, test, features):
    """Label-encode categorical features for LightGBM/XGBoost."""
    train = train.copy()
    test = test.copy()
    
    for col in LGBM_CAT_FEATURES:
        if col in features:
            combined = pd.concat([train[col].astype(str), test[col].astype(str)])
            codes, uniques = pd.factorize(combined)
            train[col] = codes[:len(train)]
            test[col] = codes[len(train):]
    
    return train, test


def _clean_params(params, keys_to_remove):
    """Remove conflicting keys from params dict."""
    return {k: v for k, v in params.items() if k not in keys_to_remove}


def evaluate_cv(model_type, params, X, y, n_splits=3):
    """
    Evaluate model using TimeSeriesSplit cross-validation.
    Uses 3 folds for speed, early stopping for efficiency.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmse_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        if model_type == 'catboost':
            cat_idx = [X.columns.get_loc(c) for c in CATBOOST_CAT_FEATURES if c in X.columns]
            clean_p = _clean_params(params, {'verbose'})
            model = cb.CatBoostRegressor(**clean_p, verbose=0)
            model.fit(X_train, y_train, cat_features=cat_idx,
                      eval_set=(X_val, y_val), early_stopping_rounds=30)
        
        elif model_type == 'lightgbm':
            clean_p = _clean_params(params, {'verbose'})
            model = lgb.LGBMRegressor(**clean_p, verbose=-1)
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
        
        elif model_type == 'xgboost':
            clean_p = _clean_params(params, {'verbosity', 'early_stopping_rounds'})
            model = xgb.XGBRegressor(**clean_p, verbosity=0, early_stopping_rounds=30)
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      verbose=False)
        
        preds = model.predict(X_val)
        preds = np.clip(preds, 0, None)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        rmse_scores.append(rmse)
    
    return np.mean(rmse_scores)


def evaluate_cv_full(model_type, params, X, y, n_splits=3):
    """Full evaluation returning both R2 and RMSE for final reporting."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    r2_scores = []
    rmse_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        if model_type == 'catboost':
            cat_idx = [X.columns.get_loc(c) for c in CATBOOST_CAT_FEATURES if c in X.columns]
            clean_p = _clean_params(params, {'verbose'})
            model = cb.CatBoostRegressor(**clean_p, verbose=0)
            model.fit(X_train, y_train, cat_features=cat_idx,
                      eval_set=(X_val, y_val), early_stopping_rounds=30)
        elif model_type == 'lightgbm':
            clean_p = _clean_params(params, {'verbose'})
            model = lgb.LGBMRegressor(**clean_p, verbose=-1)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
        elif model_type == 'xgboost':
            clean_p = _clean_params(params, {'verbosity', 'early_stopping_rounds'})
            model = xgb.XGBRegressor(**clean_p, verbosity=0, early_stopping_rounds=30)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        preds = np.clip(model.predict(X_val), 0, None)
        r2_scores.append(r2_score(y_val, preds))
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))
    
    return np.mean(r2_scores), np.mean(rmse_scores)


# ==========================================================================
# Optuna objectives (speed-optimized)
# ==========================================================================
def catboost_objective(trial, X, y):
    """Optuna objective for CatBoost - capped at 1000 iterations with early stopping."""
    params = {
        'iterations': trial.suggest_int('iterations', 300, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
        'depth': trial.suggest_int('depth', 4, 9),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 1.0, 20.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'loss_function': 'RMSE',
        'random_seed': 42,
        'thread_count': -1,
    }
    return evaluate_cv('catboost', params, X, y)


def lightgbm_objective(trial, X, y):
    """Optuna objective for LightGBM."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 9),
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'objective': 'regression',
        'metric': 'rmse',
        'random_state': 42,
        'n_jobs': -1,
    }
    return evaluate_cv('lightgbm', params, X, y)


def xgboost_objective(trial, X, y):
    """Optuna objective for XGBoost."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 9),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist',
        'early_stopping_rounds': 30,
    }
    return evaluate_cv('xgboost', params, X, y)


# ==========================================================================
# Main
# ==========================================================================
def run_optuna_tuning(train_df, n_trials=30, output_dir='optuna_results'):
    """
    Run Optuna hyperparameter optimization for all three models.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("PHASE 3 & 4: MODEL TRAINING + OPTUNA OPTIMIZATION")
    print("=" * 60)
    
    train_sorted = train_df.sort_values('time_order').reset_index(drop=True)
    features = get_feature_columns(train_sorted)
    X = train_sorted[features]
    y = train_sorted['demand']
    
    print(f"\nFeatures: {len(features)} columns")
    print(f"Training samples: {len(X)}")
    print(f"Target stats: mean={y.mean():.4f}, std={y.std():.4f}")
    print(f"Using TimeSeriesSplit with 3 folds + early stopping")
    
    best_params = {}
    cv_scores = {}
    
    # ------------------------------------------------------------------
    # 1. CatBoost
    # ------------------------------------------------------------------
    print(f"\n{'='*50}")
    print(f"[1/3] CATBOOST OPTIMIZATION ({n_trials} trials)")
    print(f"{'='*50}")
    t0 = time.time()
    
    cb_study = optuna.create_study(direction='minimize', study_name='catboost')
    cb_study.optimize(lambda trial: catboost_objective(trial, X, y), 
                       n_trials=n_trials, show_progress_bar=True)
    
    cb_best = cb_study.best_params
    cb_best['loss_function'] = 'RMSE'
    cb_best['thread_count'] = -1
    best_params['catboost'] = cb_best
    
    cb_time = time.time() - t0
    
    # Full evaluation with best params
    cb_r2, cb_rmse = evaluate_cv_full('catboost', {**cb_best, 'random_seed': 42, 'verbose': 0}, X, y)
    cv_scores['catboost'] = {'r2': cb_r2, 'rmse': cb_rmse}
    
    print(f"\n  Best CatBoost RMSE: {cb_study.best_value:.6f}")
    print(f"  CV R2 Score: {max(0, 100*cb_r2):.2f}")
    print(f"  Time: {cb_time:.1f}s ({cb_time/60:.1f}min)")
    print(f"  Best params: {json.dumps(cb_best, indent=2, default=str)}")
    
    # ------------------------------------------------------------------
    # 2. LightGBM
    # ------------------------------------------------------------------
    print(f"\n{'='*50}")
    print(f"[2/3] LIGHTGBM OPTIMIZATION ({n_trials} trials)")
    print(f"{'='*50}")
    t0 = time.time()
    
    train_lgb, _ = prepare_for_lgbm_xgb(train_sorted, train_sorted, features)
    X_lgb = train_lgb[features]
    
    lgb_study = optuna.create_study(direction='minimize', study_name='lightgbm')
    lgb_study.optimize(lambda trial: lightgbm_objective(trial, X_lgb, y),
                        n_trials=n_trials, show_progress_bar=True)
    
    lgb_best = lgb_study.best_params
    lgb_best['objective'] = 'regression'
    lgb_best['metric'] = 'rmse'
    lgb_best['n_jobs'] = -1
    best_params['lightgbm'] = lgb_best
    
    lgb_time = time.time() - t0
    
    lgb_r2, lgb_rmse = evaluate_cv_full('lightgbm', {**lgb_best, 'random_state': 42, 'verbose': -1}, X_lgb, y)
    cv_scores['lightgbm'] = {'r2': lgb_r2, 'rmse': lgb_rmse}
    
    print(f"\n  Best LightGBM RMSE: {lgb_study.best_value:.6f}")
    print(f"  CV R2 Score: {max(0, 100*lgb_r2):.2f}")
    print(f"  Time: {lgb_time:.1f}s ({lgb_time/60:.1f}min)")
    print(f"  Best params: {json.dumps(lgb_best, indent=2, default=str)}")
    
    # ------------------------------------------------------------------
    # 3. XGBoost
    # ------------------------------------------------------------------
    print(f"\n{'='*50}")
    print(f"[3/3] XGBOOST OPTIMIZATION ({n_trials} trials)")
    print(f"{'='*50}")
    t0 = time.time()
    
    xgb_study = optuna.create_study(direction='minimize', study_name='xgboost')
    xgb_study.optimize(lambda trial: xgboost_objective(trial, X_lgb, y),
                        n_trials=n_trials, show_progress_bar=True)
    
    xgb_best = xgb_study.best_params
    xgb_best['objective'] = 'reg:squarederror'
    xgb_best['eval_metric'] = 'rmse'
    xgb_best['n_jobs'] = -1
    xgb_best['tree_method'] = 'hist'
    xgb_best['early_stopping_rounds'] = 30
    best_params['xgboost'] = xgb_best
    
    xgb_time = time.time() - t0
    
    xgb_r2, xgb_rmse = evaluate_cv_full('xgboost', {**xgb_best, 'random_state': 42, 'verbosity': 0}, X_lgb, y)
    cv_scores['xgboost'] = {'r2': xgb_r2, 'rmse': xgb_rmse}
    
    print(f"\n  Best XGBoost RMSE: {xgb_study.best_value:.6f}")
    print(f"  CV R2 Score: {max(0, 100*xgb_r2):.2f}")
    print(f"  Time: {xgb_time:.1f}s ({xgb_time/60:.1f}min)")
    print(f"  Best params: {json.dumps(xgb_best, indent=2, default=str)}")
    
    # ------------------------------------------------------------------
    # Compute optimal ensemble weights based on inverse RMSE
    # ------------------------------------------------------------------
    inv_rmse = {k: 1.0 / v['rmse'] for k, v in cv_scores.items()}
    total_inv = sum(inv_rmse.values())
    weights = {k: v / total_inv for k, v in inv_rmse.items()}
    best_params['ensemble_weights'] = weights
    
    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    params_path = os.path.join(output_dir, 'best_params.json')
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=2, default=str)
    print(f"\n[OK] Best parameters saved to {params_path}")
    
    print(f"\n{'='*60}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"  CatBoost  RMSE: {cv_scores['catboost']['rmse']:.6f}  R2: {max(0,100*cv_scores['catboost']['r2']):.2f}")
    print(f"  LightGBM  RMSE: {cv_scores['lightgbm']['rmse']:.6f}  R2: {max(0,100*cv_scores['lightgbm']['r2']):.2f}")
    print(f"  XGBoost   RMSE: {cv_scores['xgboost']['rmse']:.6f}  R2: {max(0,100*cv_scores['xgboost']['r2']):.2f}")
    print(f"\n  Ensemble weights (inverse RMSE):")
    print(f"    CatBoost:  {weights['catboost']:.3f}")
    print(f"    LightGBM:  {weights['lightgbm']:.3f}")
    print(f"    XGBoost:   {weights['xgboost']:.3f}")
    
    return best_params


if __name__ == '__main__':
    from data_loader import load_and_clean
    from feature_engineering import engineer_features
    
    train, test = load_and_clean()
    train, test = engineer_features(train, test)
    best_params = run_optuna_tuning(train, n_trials=30)
