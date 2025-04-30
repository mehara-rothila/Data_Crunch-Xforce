# deployment/predictor.py
# Corrected version v4: Added ffill() for lag/rolling features

import pandas as pd
import numpy as np
import joblib
import os
from datetime import timedelta

# Import necessary functions from sibling modules
try:
    # Import the specific functions needed from the correct feature_engineering module
    from .feature_engineering import create_time_features, create_lag_features, create_rolling_window_features
except ImportError:
    # Fallback for running the script directly
    from feature_engineering import create_time_features, create_lag_features, create_rolling_window_features


# Define constants relative to this script's expected location
MODEL_DIR = os.path.dirname(__file__) # Gets the directory where predictor.py is saved
MODEL_FILENAME = os.path.join(MODEL_DIR, 'lgbm_price_model.joblib')
FEATURES_FILENAME = os.path.join(MODEL_DIR, 'model_features.joblib')

# Define weather columns needed for features
WEATHER_COLS = ['Temperature_K', 'Rainfall_mm', 'Humidity_pct', 'CropYieldImpactScore']
# Define lag/rolling columns that might have NaNs at the start
LAG_ROLLING_COLS_PREFIX = ['Price_lag_', 'Price_roll_']


class Predictor:
    """Loads the model and makes predictions."""

    def __init__(self):
        """Loads the model and feature list."""
        self.model = None
        self.features = None
        try:
            print(f"Loading model from: {MODEL_FILENAME}")
            self.model = joblib.load(MODEL_FILENAME)
            print("Model loaded successfully.")

            print(f"Loading feature list from: {FEATURES_FILENAME}")
            self.features = joblib.load(FEATURES_FILENAME)
            print(f"Feature list loaded successfully. Expecting {len(self.features)} features.")

        except FileNotFoundError as e:
            print(f"Error loading model or features: {e}. Ensure training was run first.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred during model loading: {e}")
            raise

    def generate_future_features(self, historical_data, future_dates, region, commodity, type_):
        """
        Generates features for future dates using historical context.
        Includes forward-filling weather data and handling initial NaNs in lags/rolling features.

        Args:
            historical_data (pd.DataFrame): DataFrame of recent historical data
                                             for the specific region/commodity.
                                             Should be preprocessed & include weather cols.
            future_dates (pd.DatetimeIndex): Dates for which to generate features.
            region (str): The region.
            commodity (str): The commodity.
            type_ (str): 'Fruit' or 'Vegetable'. (Renamed from 'type' to avoid keyword conflict)

        Returns:
            pd.DataFrame: DataFrame with features for the future_dates.
                          Returns None if features cannot be generated.
        """
        if historical_data is None or historical_data.empty:
             print("Error: No historical data provided for feature generation.")
             return None

        # Ensure hist data is sorted and has necessary columns
        required_hist_cols = ['Date', 'Region', 'Commodity', 'Type', 'Price'] + WEATHER_COLS
        if not all(col in historical_data.columns for col in required_hist_cols):
             print(f"Error: Historical data is missing required columns. Need: {required_hist_cols}")
             return None

        historical_data = historical_data.sort_values('Date')

        # --- Prepare future DataFrame ---
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Region': region,
            'Commodity': commodity,
            'Type': type_ # Use the renamed argument
        })

        # --- Forward-fill weather data ---
        last_known_weather = historical_data[['Date'] + WEATHER_COLS].ffill().iloc[-1][WEATHER_COLS]
        for col in WEATHER_COLS:
            future_df[col] = last_known_weather[col]
        print(f"Forward-filled weather data for future dates using values from {historical_data['Date'].max().date()}")

        # --- Combine historical + future for feature calculation ---
        hist_for_features = historical_data[required_hist_cols].copy()
        combined_df = pd.concat([hist_for_features, future_df], ignore_index=True)
        combined_df['Date'] = pd.to_datetime(combined_df['Date'])
        for col in ['Region', 'Commodity', 'Type']:
             combined_df[col] = combined_df[col].astype('category')
        for col in WEATHER_COLS + ['Price']:
             combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

        # --- Apply Feature Engineering Consistently ---
        combined_df = combined_df.sort_values(by=['Region', 'Commodity', 'Date'])
        group_columns = ['Region', 'Commodity']
        combined_df = create_time_features(combined_df)
        combined_df = create_lag_features(combined_df, group_cols=group_columns, target_col='Price')
        combined_df = create_rolling_window_features(combined_df, group_cols=group_columns, target_col='Price')
        print("Applied feature engineering (time, lags, rolling) to combined data.")

        # --- Finalize Future Features ---
        future_features_df = combined_df[combined_df['Date'].isin(future_dates)].copy()

        # *** NEW STEP: Forward-fill NaNs in lag/rolling features ***
        # Identify lag/rolling columns dynamically
        cols_to_ffill = [col for col in future_features_df.columns
                         if any(col.startswith(prefix) for prefix in LAG_ROLLING_COLS_PREFIX)]
        if cols_to_ffill:
             print(f"Forward-filling potential NaNs in columns: {cols_to_ffill}")
             # Need to sort by date within the future_df slice before ffill
             future_features_df = future_features_df.sort_values('Date')
             future_features_df[cols_to_ffill] = future_features_df[cols_to_ffill].ffill()
             # Optional: Backward fill any remaining NaNs at the very start
             future_features_df[cols_to_ffill] = future_features_df[cols_to_ffill].bfill()
             # Optional: Fill any remaining NaNs (e.g., if entire series was NaN) with 0
             future_features_df[cols_to_ffill] = future_features_df[cols_to_ffill].fillna(0)


        # --- Select and Align Features ---
        try:
            missing_cols = set(self.features) - set(future_features_df.columns)
            if missing_cols:
                print(f"Error: Critical features missing after generation: {missing_cols}")
                return None
            future_features_df = future_features_df[self.features]
        except Exception as e:
            print(f"Error aligning features: {e}")
            return None

        # --- Final NaN Check ---
        if future_features_df.isnull().any().any():
            print("Error: NaNs still found in final future features after fill attempts. Cannot proceed.")
            # print(future_features_df[future_features_df.isnull().any(axis=1)]) # Debug: Print rows with NaNs
            return None

        print("Future features generated successfully.")
        return future_features_df


    def predict(self, input_data_future_features):
        """
        Makes predictions using the loaded model.
        """
        if self.model is None or self.features is None:
            print("Error: Model or feature list not loaded.")
            return None
        if input_data_future_features is None or input_data_future_features.empty:
             print("Error: No input data provided for prediction.")
             return None

        # Ensure no NaNs before prediction
        if input_data_future_features.isnull().any().any():
            print("Error: Input data contains NaNs before prediction.")
            return None

        try:
            input_aligned = input_data_future_features[self.features]
            categorical_features = input_aligned.select_dtypes(include='category').columns.tolist()
            # No need to explicitly handle categories for LightGBM prediction if names passed during fit

            predictions = self.model.predict(input_aligned)
            return predictions

        except KeyError as e:
            print(f"Error: Missing expected feature during prediction: {e}")
            return None
        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            return None

# Example usage (conceptual - requires access to historical data)
if __name__ == "__main__":
    print("Running Predictor class test...")
    try:
        predictor = Predictor()
        print("\n--- Conceptual Prediction Example ---")
        print("NOTE: The following requires loading actual historical data, skipping for simple test.")

    except Exception as e:
        print(f"Failed to initialize or test Predictor: {e}")
