"""
Master Pipeline: Traffic Demand Prediction
============================================
Runs all phases sequentially:
  Phase 1: Data loading & cleaning
  Phase 2: Feature engineering
  Phase 3-4: Optuna hyperparameter tuning
  Phase 5: Final training & submission

Usage:
  python run_pipeline.py                   # Full pipeline (tuning + inference)
  python run_pipeline.py --skip-tuning     # Skip Optuna, use saved params
  python run_pipeline.py --trials 25       # Customize number of Optuna trials
"""

import sys
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

from data_loader import load_and_clean
from feature_engineering import engineer_features
from train_model import run_optuna_tuning
from inference import load_best_params, train_final_ensemble, generate_submission


def main():
    parser = argparse.ArgumentParser(description='Traffic Demand Prediction Pipeline')
    parser.add_argument('--skip-tuning', action='store_true',
                       help='Skip Optuna tuning, use saved params from optuna_results/')
    parser.add_argument('--trials', type=int, default=50,
                       help='Number of Optuna trials per model (default: 50)')
    parser.add_argument('--seeds', type=int, default=5,
                       help='Number of random seeds for ensemble (default: 5)')
    parser.add_argument('--output', type=str, default='submission.csv',
                       help='Output submission file path (default: submission.csv)')
    args = parser.parse_args()
    
    total_start = time.time()
    
    print("=" * 60)
    print("  TRAFFIC DEMAND PREDICTION - MASTER PIPELINE")
    print("  3-Model Ensemble | Optuna Tuning | TimeSeriesSplit")
    print("=" * 60)
    
    # ========================================
    # PHASE 1: Data Loading & Cleaning
    # ========================================
    t0 = time.time()
    train, test = load_and_clean()
    print(f"  Phase 1 time: {time.time()-t0:.1f}s")
    
    # ========================================
    # PHASE 2: Feature Engineering
    # ========================================
    t0 = time.time()
    train, test = engineer_features(train, test)
    print(f"  Phase 2 time: {time.time()-t0:.1f}s")
    
    # ========================================
    # PHASE 3-4: Optuna Tuning
    # ========================================
    if args.skip_tuning:
        print("\n[SKIP] Loading saved hyperparameters...")
        try:
            best_params = load_best_params()
            print("  Loaded from optuna_results/best_params.json")
        except FileNotFoundError:
            print("  ERROR: No saved params found. Running tuning instead...")
            best_params = run_optuna_tuning(train, n_trials=args.trials)
    else:
        t0 = time.time()
        best_params = run_optuna_tuning(train, n_trials=args.trials)
        print(f"  Phase 3-4 time: {time.time()-t0:.1f}s")
    
    # ========================================
    # PHASE 5: Final Training & Submission
    # ========================================
    t0 = time.time()
    final_preds, model_preds = train_final_ensemble(
        train, test, best_params, n_seeds=args.seeds
    )
    submission = generate_submission(test, final_preds, output_path=args.output)
    print(f"  Phase 5 time: {time.time()-t0:.1f}s")
    
    # ========================================
    # SUMMARY
    # ========================================
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Output: {args.output}")
    print(f"  Rows: {len(submission)}")
    print(f"  Demand range: [{submission['demand'].min():.6f}, {submission['demand'].max():.6f}]")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
