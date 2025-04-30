# deployment/model_trainer.py
# VERSION WITH OPTUNA HYPERPARAMETER TUNING
# MODIFIED TO PRINT FULL FEATURE IMPORTANCE

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split # Fallback split
from sklearn.metrics import mean_squared_error
import joblib
import os
import optuna # Import optuna

# Import functions from sibling modules
try:
    from .data_loader import load_training_data
    from .preprocessing import preprocess_data
    from .feature_engineering import feature_engineering
except ImportError:
    # Fallback for running the script directly
    from data_loader import load_training_data
    from preprocessing import preprocess_data
    from feature_engineering import feature_engineering

# Define constants
TARGET_COLUMN = 'Price'
MODEL_FILENAME = 'lgbm_price_model.joblib'
FEATURES_FILENAME = 'model_features.joblib'
VALIDATION_MONTHS = 2 # Use last N months for validation
RANDOM_SEED = 42 # For reproducibility
OPTUNA_N_TRIALS = 30 # Number of tuning trials (adjust based on available time)

# --- Global variables for Optuna objective function ---
# These need to be accessible by the objective function
X_train_global, y_train_global = None, None
X_val_global, y_val_global = None, None
categorical_features_global = None
features_global = None # Store feature list globally

def objective(trial):
    """Optuna objective function to minimize validation RMSE."""
    global X_train_global, y_train_global, X_val_global, y_val_global, categorical_features_global

    # Define hyperparameter search space
    params = {
        'objective': 'rmse',
        'metric': 'rmse',
        'random_state': RANDOM_SEED,
        'n_jobs': -1,
        'n_estimators': 1000, # Keep high, use early stopping
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True), # L1 regularization
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True), # L2 regularization
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0), # Feature fraction
        'subsample': trial.suggest_float('subsample', 0.6, 1.0), # Data fraction (bagging)
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 7), # Frequency for bagging
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50), # Min data in leaf
        'verbosity': -1 # Suppress verbose LightGBM logs during tuning trials
    }

    model = lgb.LGBMRegressor(**params)

    early_stopping_callback = lgb.early_stopping(stopping_rounds=50, verbose=False) # Less verbose during tuning

    model.fit(
        X_train_global, y_train_global,
        eval_set=[(X_val_global, y_val_global)],
        eval_metric='rmse',
        callbacks=[early_stopping_callback],
        categorical_feature=categorical_features_global
    )

    preds = model.predict(X_val_global)
    rmse = np.sqrt(mean_squared_error(y_val_global, preds))

    return rmse # Optuna tries to minimize this value


def train_model_with_tuning(df):
    """
    Performs data splitting, Optuna tuning, and final model training.
    """
    global X_train_global, y_train_global, X_val_global, y_val_global, categorical_features_global, features_global

    if df is None or not isinstance(df, pd.DataFrame):
        print("Error: Invalid input DataFrame for model training.")
        return None, None, None

    # 1. Handle NaNs
    initial_rows = len(df)
    df_cleaned = df.dropna()
    rows_dropped = initial_rows - len(df_cleaned)
    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} rows containing NaN values (due to lags/rolling windows).")
        print(f"Remaining rows for training/validation: {len(df_cleaned)}")
    if len(df_cleaned) == 0:
        print("Error: No data left after dropping NaNs.")
        return None, None, None

    # 2. Define features (X) and target (y)
    y = df_cleaned[TARGET_COLUMN]
    features_global = [col for col in df_cleaned.columns if col not in [TARGET_COLUMN, 'Date']]
    X = df_cleaned[features_global].copy() # Create a copy to avoid modifying original df_cleaned
    categorical_features_global = X.select_dtypes(include='category').columns.tolist()
    print(f"Identified categorical features: {categorical_features_global}")

    # Ensure categorical features have the correct type for LightGBM if needed
    # (Usually handled automatically by passing names to categorical_feature param)
    # for col in categorical_features_global:
    #     X[col] = X[col].astype('category') # Ensure it's category type

    # 3. Time-based Split (Assign to global vars for Optuna)
    # Ensure sorting before splitting
    df_cleaned = df_cleaned.sort_values(by='Date')
    # Re-assign X and y from the sorted df_cleaned to ensure alignment
    y = df_cleaned[TARGET_COLUMN]
    X = df_cleaned[features_global] # Use the sorted df_cleaned

    last_date = df_cleaned['Date'].max()
    split_date = last_date - pd.DateOffset(months=VALIDATION_MONTHS)
    train_mask = df_cleaned['Date'] <= split_date
    val_mask = df_cleaned['Date'] > split_date

    # Use boolean masks directly on X and y derived from the sorted df_cleaned
    X_train_global = X[train_mask]
    y_train_global = y[train_mask]
    X_val_global = X[val_mask]
    y_val_global = y[val_mask]

    if len(X_train_global) == 0 or len(X_val_global) == 0:
        print("Error: Not enough data for train/validation split. Cannot tune.")
        # Use the full X and y before masking for the fallback split
        print("Warning: Falling back to random split. Time series order not guaranteed.")
        X_train_global, X_val_global, y_train_global, y_val_global = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED
        )
    else:
        print(f"Time-based split: Train data up to {split_date} ({len(X_train_global)} rows), Validation data after ({len(X_val_global)} rows)")

    # 4. Optuna Study
    print(f"\nStarting Optuna hyperparameter search ({OPTUNA_N_TRIALS} trials)...")
    # Add exception handling for Optuna study
    try:
        study = optuna.create_study(direction='minimize') # We want to minimize RMSE
        study.optimize(objective, n_trials=OPTUNA_N_TRIALS, timeout=None) # Set timeout in seconds if needed

        print("Optuna study finished.")
        print(f"Best trial number: {study.best_trial.number}")
        print(f"Best validation RMSE: {study.best_value:.4f}")
        print("Best hyperparameters found:")
        best_params = study.best_params
        print(best_params)

    except Exception as e:
        print(f"An error occurred during Optuna study: {e}")
        print("Proceeding without hyperparameter tuning, using default parameters.")
        # Define default parameters if tuning fails
        best_params = {
            'objective': 'rmse',
            'metric': 'rmse',
            'random_state': RANDOM_SEED,
            'n_jobs': -1,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1,
            'n_estimators': 1000 # Use default n_estimators if tuning fails
        }


    # 5. Train Final Model with Best Parameters (or defaults if tuning failed)
    print("\nTraining final model with best hyperparameters found (or defaults)...")
    # Ensure essential params are included even if using defaults
    best_params['objective'] = 'rmse'
    best_params['metric'] = 'rmse'
    best_params['random_state'] = RANDOM_SEED
    best_params['n_jobs'] = -1
    # Ensure n_estimators is set, increase for final model if tuning was successful
    if 'n_estimators' not in best_params:
         best_params['n_estimators'] = 1500 if study and hasattr(study, 'best_params') else 1000


    final_model = lgb.LGBMRegressor(**best_params)

    # Use a slightly longer patience for the final fit
    final_early_stopping_callback = lgb.early_stopping(stopping_rounds=100, verbose=True)

    # Train on the same train/val split used during tuning
    final_model.fit(
        X_train_global, y_train_global,
        eval_set=[(X_val_global, y_val_global)],
        eval_metric='rmse',
        callbacks=[final_early_stopping_callback],
        categorical_feature=categorical_features_global
    )

    # 6. Final Evaluation (on the same validation set)
    print("\nEvaluating final model...")
    y_pred_final_val = final_model.predict(X_val_global)
    final_validation_rmse = np.sqrt(mean_squared_error(y_val_global, y_pred_final_val))
    print(f"Final Model Validation RMSE: {final_validation_rmse:.4f}")

    # Feature Importance of the final model
    try:
        # Ensure features_global list matches the columns used in training (X_train_global)
        final_feature_importance_df = pd.DataFrame({
            'feature': X_train_global.columns, # Use columns from actual training data
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nFull Feature Importances (Final Model):") # <<< CHANGED TEXT
        # Print the entire DataFrame
        with pd.option_context('display.max_rows', None, 'display.max_columns', None): # Ensure all rows/cols print
             print(final_feature_importance_df) # <<< CHANGED TO PRINT ALL
    except Exception as e:
        print(f"Could not retrieve feature importances for final model: {e}")

    # Return the final trained model, features used, and its validation RMSE
    return final_model, features_global, final_validation_rmse

# Main execution block
if __name__ == "__main__":
    project_root = "."
    print(f"Running model training test with Optuna tuning from base path: {os.path.abspath(project_root)}")

    # Data Pipeline
    raw_train_data = load_training_data(base_path=project_root)
    if raw_train_data is None: exit()
    print("\nRaw data loaded.")
    processed_train_data = preprocess_data(raw_train_data)
    if processed_train_data is None: exit()
    print("\nPreprocessing complete.")
    featured_train_data = feature_engineering(processed_train_data) # Uses the reverted feature engineering
    if featured_train_data is None: exit()
    print("\nFeature engineering complete.")

    # Model Training with Tuning
    model, features_used, val_rmse = train_model_with_tuning(featured_train_data)

    # Save Model
    if model is not None and features_used is not None:
        model_save_path = os.path.join(project_root, "deployment", MODEL_FILENAME)
        features_save_path = os.path.join(project_root, "deployment", FEATURES_FILENAME)
        try:
            joblib.dump(model, model_save_path)
            print(f"\nTrained final tuned model saved successfully to: {model_save_path}")
            # Save the correct list of features used in the final model
            final_features_list = list(X_train_global.columns) # Get features from the actual training data
            joblib.dump(final_features_list, features_save_path)
            print(f"Feature list ({len(final_features_list)} features) saved successfully to: {features_save_path}")
        except Exception as e:
            print(f"\nError saving final model or feature list: {e}")
    else:
        print("\nModel training with tuning failed. Model not saved.")

