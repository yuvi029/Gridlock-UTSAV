"""
Phase 5: Inference & Submission Generation
============================================
Trains final models on 100% of data using best Optuna parameters,
generates ensemble predictions, and outputs submission.csv.
"""

import numpy as np
import pandas as pd
import json
import os
import time
import warnings
warnings.filterwarnings('ignore')

import catboost as cb
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import r2_score


# Import project modules
from train_model import (
    get_feature_columns, prepare_for_lgbm_xgb,
    CATBOOST_CAT_FEATURES, LGBM_CAT_FEATURES, DROP_COLS
)


def load_best_params(params_path='optuna_results/best_params.json'):
    """Load best hyperparameters from Optuna results."""
    with open(params_path, 'r') as f:
        return json.load(f)


def train_final_ensemble(train_df, test_df, best_params, n_seeds=5):
    """
    Train final ensemble: multi-seed averaging across 3 model types.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Feature-engineered training data.
    test_df : pd.DataFrame
        Feature-engineered test data.
    best_params : dict
        Best hyperparameters for each model.
    n_seeds : int
        Number of random seeds per model for variance reduction.
    
    Returns
    -------
    final_preds : np.ndarray
        Final ensemble predictions for test data.
    model_preds : dict
        Individual model predictions (for analysis).
    """
    print("\n" + "=" * 60)
    print("PHASE 5: FINAL TRAINING & INFERENCE")
    print("=" * 60)
    
    # Sort train chronologically
    train_sorted = train_df.sort_values('time_order').reset_index(drop=True)
    
    features = get_feature_columns(train_sorted)
    X_train = train_sorted[features]
    y_train = train_sorted['demand']
    X_test = test_df[features]
    
    # Prepare label-encoded versions for LightGBM/XGBoost
    train_enc, test_enc = prepare_for_lgbm_xgb(train_sorted, test_df, features)
    X_train_enc = train_enc[features]
    X_test_enc = test_enc[features]
    
    print(f"\nTraining features: {len(features)}")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    seeds = [42, 123, 456, 789, 2024][:n_seeds]
    model_preds = {'catboost': [], 'lightgbm': [], 'xgboost': []}
    
    # ------------------------------------------------------------------
    # 1. CatBoost ensemble
    # ------------------------------------------------------------------
    print(f"\n[1/3] Training CatBoost ({n_seeds} seeds)...")
    cb_params = best_params.get('catboost', {}).copy()
    # Remove any non-CatBoost params
    cb_params.pop('random_state', None)
    cat_idx = [X_train.columns.get_loc(c) for c in CATBOOST_CAT_FEATURES if c in X_train.columns]
    
    for i, seed in enumerate(seeds):
        t0 = time.time()
        cb_params_seed = {**cb_params, 'random_seed': seed, 'verbose': 0}
        model = cb.CatBoostRegressor(**cb_params_seed)
        model.fit(X_train, y_train, cat_features=cat_idx)
        preds = model.predict(X_test)
        model_preds['catboost'].append(preds)
        elapsed = time.time() - t0
        print(f"  Seed {seed} ({i+1}/{n_seeds}): range=[{preds.min():.4f}, {preds.max():.4f}] ({elapsed:.1f}s)")
    
    cb_avg = np.mean(model_preds['catboost'], axis=0)
    print(f"  CatBoost avg pred: mean={cb_avg.mean():.4f}, std={cb_avg.std():.4f}")
    
    # ------------------------------------------------------------------
    # 2. LightGBM ensemble
    # ------------------------------------------------------------------
    print(f"\n[2/3] Training LightGBM ({n_seeds} seeds)...")
    lgb_params = best_params.get('lightgbm', {}).copy()
    
    for i, seed in enumerate(seeds):
        t0 = time.time()
        lgb_params_seed = {**lgb_params, 'random_state': seed, 'verbose': -1}
        model = lgb.LGBMRegressor(**lgb_params_seed)
        model.fit(X_train_enc, y_train)
        preds = model.predict(X_test_enc)
        model_preds['lightgbm'].append(preds)
        elapsed = time.time() - t0
        print(f"  Seed {seed} ({i+1}/{n_seeds}): range=[{preds.min():.4f}, {preds.max():.4f}] ({elapsed:.1f}s)")
    
    lgb_avg = np.mean(model_preds['lightgbm'], axis=0)
    print(f"  LightGBM avg pred: mean={lgb_avg.mean():.4f}, std={lgb_avg.std():.4f}")
    
    # ------------------------------------------------------------------
    # 3. XGBoost ensemble
    # ------------------------------------------------------------------
    print(f"\n[3/3] Training XGBoost ({n_seeds} seeds)...")
    xgb_params = best_params.get('xgboost', {}).copy()
    # Remove early_stopping_rounds from params (not needed for final training without eval_set)
    xgb_params.pop('early_stopping_rounds', None)
    xgb_params.pop('eval_metric', None)
    
    for i, seed in enumerate(seeds):
        t0 = time.time()
        xgb_params_seed = {**xgb_params, 'random_state': seed, 'verbosity': 0}
        model = xgb.XGBRegressor(**xgb_params_seed)
        model.fit(X_train_enc, y_train)
        preds = model.predict(X_test_enc)
        model_preds['xgboost'].append(preds)
        elapsed = time.time() - t0
        print(f"  Seed {seed} ({i+1}/{n_seeds}): range=[{preds.min():.4f}, {preds.max():.4f}] ({elapsed:.1f}s)")
    
    xgb_avg = np.mean(model_preds['xgboost'], axis=0)
    print(f"  XGBoost avg pred: mean={xgb_avg.mean():.4f}, std={xgb_avg.std():.4f}")
    
    # ------------------------------------------------------------------
    # 4. Weighted ensemble
    # ------------------------------------------------------------------
    print("\n[ENSEMBLE] Computing weighted average...")
    
    # Use weights computed from inverse CV RMSE during Optuna tuning
    weights = best_params.get('ensemble_weights', 
                               {'catboost': 0.45, 'lightgbm': 0.30, 'xgboost': 0.25})
    w_cb = weights.get('catboost', 0.45)
    w_lgb = weights.get('lightgbm', 0.30)
    w_xgb = weights.get('xgboost', 0.25)
    
    final_preds = w_cb * cb_avg + w_lgb * lgb_avg + w_xgb * xgb_avg
    
    # Post-processing: clip to valid range
    final_preds = np.clip(final_preds, 0, None)
    
    print(f"  Weights: CatBoost={w_cb}, LightGBM={w_lgb}, XGBoost={w_xgb}")
    print(f"  Final pred stats: mean={final_preds.mean():.4f}, std={final_preds.std():.4f}")
    print(f"  Final pred range: [{final_preds.min():.4f}, {final_preds.max():.4f}]")
    
    return final_preds, model_preds


def generate_submission(test_df, predictions, output_path='submission.csv'):
    """
    Generate submission CSV in the exact required format.
    
    Parameters
    ----------
    test_df : pd.DataFrame
        Original test data (with Index column).
    predictions : np.ndarray
        Predicted demand values.
    output_path : str
        Path to save submission CSV.
    """
    print(f"\n{'='*60}")
    print("GENERATING SUBMISSION")
    print(f"{'='*60}")
    
    submission = pd.DataFrame({
        'Index': test_df['Index'],
        'demand': predictions
    })
    
    # Validation
    assert len(submission) == 41778, f"Expected 41778 rows, got {len(submission)}"
    assert list(submission.columns) == ['Index', 'demand'], f"Wrong columns: {submission.columns.tolist()}"
    assert submission['Index'].nunique() == 41778, "Duplicate Index values!"
    assert submission['demand'].isnull().sum() == 0, "NaN predictions found!"
    assert (submission['demand'] >= 0).all(), "Negative predictions found!"
    
    submission.to_csv(output_path, index=False)
    
    print(f"  Saved to: {output_path}")
    print(f"  Shape: {submission.shape}")
    print(f"  Demand stats:")
    print(f"    Min:    {submission['demand'].min():.6f}")
    print(f"    Max:    {submission['demand'].max():.6f}")
    print(f"    Mean:   {submission['demand'].mean():.6f}")
    print(f"    Median: {submission['demand'].median():.6f}")
    print(f"\n  Head:")
    print(submission.head().to_string())
    
    return submission


if __name__ == '__main__':
    from data_loader import load_and_clean
    from feature_engineering import engineer_features
    
    # Load and prepare data
    train, test = load_and_clean()
    train, test = engineer_features(train, test)
    
    # Load best parameters
    best_params = load_best_params()
    print(f"\nLoaded best parameters from optuna_results/best_params.json")
    
    # Train final ensemble and predict
    final_preds, model_preds = train_final_ensemble(train, test, best_params, n_seeds=5)
    
    # Generate submission
    submission = generate_submission(test, final_preds)
    
    print("\n[OK] Pipeline complete!")
