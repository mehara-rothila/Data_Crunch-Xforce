# deployment/feature_engineering.py
# VERSION THAT PRODUCED RMSE ~84.38

import pandas as pd
import numpy as np
import os
import warnings

# Import functions from sibling modules
try:
    from .data_loader import load_training_data
    from .preprocessing import preprocess_data
except ImportError:
    # Fallback for running the script directly
    from data_loader import load_training_data
    from preprocessing import preprocess_data

def create_time_features(df):
    """Creates time-based features from the 'Date' column."""
    if 'Date' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['Date']):
        print("Error: 'Date' column not found or not datetime type.")
        return df # Return original df if date column is missing or wrong type

    df_copy = df.copy() # Work on a copy

    df_copy['year'] = df_copy['Date'].dt.year
    df_copy['month'] = df_copy['Date'].dt.month
    df_copy['day'] = df_copy['Date'].dt.day
    df_copy['day_of_week'] = df_copy['Date'].dt.dayofweek # Monday=0, Sunday=6
    df_copy['day_of_year'] = df_copy['Date'].dt.dayofyear
    # Use ISO week date standard
    df_copy['week_of_year'] = df_copy['Date'].dt.isocalendar().week.astype(int)
    df_copy['is_weekend'] = df_copy['day_of_week'].isin([5, 6]).astype(int) # 1 if weekend, 0 otherwise
    print("Created time features: year, month, day, day_of_week, day_of_year, week_of_year, is_weekend")
    return df_copy

def create_lag_features(df, group_cols, target_col='Price', lags=None):
    """
    Creates lag features for the target column, grouped by specified columns.
    Ensures data is sorted by Date within each group before lagging.
    """
    if lags is None:
        # Since we predict 4 weeks (28 days) ahead, we need lags beyond 28 days
        # Let's use lags corresponding to 4, 5, 6 weeks
        lags = [28, 35, 42] # Keep these relevant lags

    print(f"Creating lag features for '{target_col}' with lags: {lags} grouped by {group_cols}")

    df_copy = df.sort_values(by=group_cols + ['Date']).copy() # Sort and copy

    # Suppress the FutureWarning for groupby observed=False
    # and explicitly set observed=False
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)

        for lag in lags:
            lag_col_name = f'{target_col}_lag_{lag}'
            # Calculate lag within each group
            df_copy[lag_col_name] = df_copy.groupby(group_cols, observed=False)[target_col].shift(lag)
            print(f"Created lag feature: {lag_col_name}")

    return df_copy

def create_rolling_window_features(df, group_cols, target_col='Price', windows=None, stats=None):
    """
    Creates rolling window features for the target column, grouped by specified columns.
    Ensures data is sorted by Date within each group before calculating.
    """
    if windows is None:
        windows = [7, 14, 28] # Example window sizes (days)
    if stats is None:
        stats = ['mean', 'std'] # Example statistics

    print(f"Creating rolling window features for '{target_col}' with windows: {windows}, stats: {stats}, grouped by {group_cols}")

    df_copy = df.sort_values(by=group_cols + ['Date']).copy() # Sort and copy

    # Suppress the FutureWarning for groupby observed=False
    # and explicitly set observed=False
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)

        # Group once
        grouped = df_copy.groupby(group_cols, observed=False)[target_col]

        for window in windows:
            # Shift by 1 because rolling includes the current observation by default,
            # but we want features based on *past* data only for prediction.
            shifted_group = grouped.shift(1)

            for stat in stats:
                roll_col_name = f'{target_col}_roll_{stat}_{window}d'
                try:
                    # Calculate rolling statistic on the shifted data
                    rolling_result = shifted_group.rolling(window=window, min_periods=max(1, window // 2)).agg(stat) # Require at least half the window
                    df_copy[roll_col_name] = rolling_result
                    print(f"Created rolling feature: {roll_col_name}")
                except Exception as e:
                    print(f"Could not create rolling feature {roll_col_name}: {e}")

    return df_copy


def feature_engineering(df):
    """
    Applies feature engineering steps to the DataFrame.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        print("Error: Invalid input DataFrame for feature engineering.")
        return None

    df_featured = df.copy() # Start with a copy of the preprocessed data

    # Define grouping columns used repeatedly
    group_columns = ['Region', 'Commodity']

    # 1. Create time-based features
    df_featured = create_time_features(df_featured)

    # 2. Create lag features for Price
    df_featured = create_lag_features(df_featured, group_cols=group_columns, target_col='Price')

    # 3. Create rolling window features for Price ONLY
    df_featured = create_rolling_window_features(df_featured, group_cols=group_columns, target_col='Price')

    print("Feature engineering steps (time, price lags, price rolling windows) complete.")
    return df_featured


# Example usage when running this script directly
if __name__ == "__main__":
    project_root = "."
    print(f"Running feature engineering test from base path: {os.path.abspath(project_root)}")

    # Load data
    raw_train_data = load_training_data(base_path=project_root)
    if raw_train_data is not None:
        print("\nRaw data loaded.")
        # Preprocess data
        processed_train_data = preprocess_data(raw_train_data)
        if processed_train_data is not None:
            print("\nPreprocessing complete. Starting feature engineering...")
            # Create features
            featured_train_data = feature_engineering(processed_train_data)

            if featured_train_data is not None:
                print("\nFeature engineering completed successfully!")
                print("Featured Data Info:")
                featured_train_data.info()
                print("\nNaN counts introduced by lags/rolling windows:")
                print(featured_train_data.isnull().sum())
                print("\nFirst 5 rows of featured data (showing new columns):")
                # Select relevant columns to display, including new ones
                display_cols = ['Date', 'Region', 'Commodity', 'Price', 'year', 'week_of_year',
                                'Price_lag_28', 'Price_roll_mean_7d', 'Price_roll_std_28d']
                # Check if all display_cols exist before printing
                display_cols_exist = [col for col in display_cols if col in featured_train_data.columns]
                print(featured_train_data[display_cols_exist].head())

            else:
                print("\nFeature engineering failed.")
        else:
             print("\nPreprocessing failed.")
    else:
        print("\nFailed to load raw data.")