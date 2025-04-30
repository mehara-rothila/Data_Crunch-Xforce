# deployment/predictor.py

import pandas as pd
import numpy as np
import joblib
import os
from datetime import timedelta

# Import necessary functions from sibling modules
try:
    # Assuming these functions are designed to work on a DataFrame
    # and return a DataFrame with the same index + new columns
    from .preprocessing import preprocess_data
    from .feature_engineering import feature_engineering, create_time_features, create_lag_features # Might need individual components
    from .data_loader import load_training_data # May need this to get historical data context
except ImportError:
    # Fallback for running the script directly
    from preprocessing import preprocess_data
    from feature_engineering import feature_engineering, create_time_features, create_lag_features
    from data_loader import load_training_data # Assuming availability if run directly


# Define constants relative to this script's expected location
MODEL_DIR = os.path.dirname(__file__) # Gets the directory where predictor.py is saved
MODEL_FILENAME = os.path.join(MODEL_DIR, 'lgbm_price_model.joblib')
FEATURES_FILENAME = os.path.join(MODEL_DIR, 'model_features.joblib')

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
            raise  # Re-raise the exception to prevent predictor use
        except Exception as e:
            print(f"An unexpected error occurred during model loading: {e}")
            raise

    def generate_future_features(self, historical_data, future_dates, region, commodity, type):
        """
        Generates features for future dates using historical context.

        Args:
            historical_data (pd.DataFrame): DataFrame of recent historical data
                                             for the specific region/commodity.
                                             Should be preprocessed.
            future_dates (pd.DatetimeIndex): Dates for which to generate features.
            region (str): The region.
            commodity (str): The commodity.
            type (str): 'Fruit' or 'Vegetable'.

        Returns:
            pd.DataFrame: DataFrame with features for the future_dates.
                          Returns None if features cannot be generated.
        """
        if historical_data is None or historical_data.empty:
             print("Warning: No historical data provided for feature generation.")
             # Cannot generate lags without history
             return None

        # Ensure hist data is sorted
        historical_data = historical_data.sort_values('Date')

        # Create a DataFrame for future predictions
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Region': region,
            'Commodity': commodity,
            'Type': type
        })

        # Combine historical + future for lag/rolling feature calculation
        # Only keep necessary columns from historical data (Price, weather if needed for features)
        # NOTE: Assuming feature_engineering only needs 'Price' from hist for lags
        # If weather lags/rolling windows are added, those columns are needed too.
        hist_for_features = historical_data[['Date', 'Region', 'Commodity', 'Type', 'Price']].copy()

        combined_df = pd.concat([hist_for_features, future_df], ignore_index=True)
        combined_df = combined_df.sort_values(by=['Region', 'Commodity', 'Date'])

        # --- Apply Feature Engineering ---
        # 1. Apply basic preprocessing (like type conversion, column renaming if needed)
        #    Assuming historical_data is already preprocessed, only apply to combined_df if necessary
        #    Make sure categorical types match those used in training
        for col in ['Region', 'Commodity', 'Type']:
             combined_df[col] = combined_df[col].astype('category')

        # 2. Create time features (can be done just on future_df or combined)
        combined_df = create_time_features(combined_df)

        # 3. Create lag features (Needs price from historical data)
        #    Crucially, this calculates lags based on the combined history + future structure
        group_columns = ['Region', 'Commodity']
        combined_df = create_lag_features(combined_df, group_cols=group_columns, target_col='Price')
        # --- Add other feature engineering steps consistently here ---

        # Filter to keep only the rows corresponding to the future dates
        future_features_df = combined_df[combined_df['Date'].isin(future_dates)].copy()

        # Select and align columns with the trained model's features
        try:
            # Ensure all expected features exist, fill missing ones (e.g., weather if not forecast)
            missing_cols = set(self.features) - set(future_features_df.columns)
            for c in missing_cols:
                print(f"Warning: Feature '{c}' not generated for future dates. Filling with 0.")
                future_features_df[c] = 0 # Or use mean/median imputation if appropriate

            # Keep only the features the model was trained on, in the correct order
            future_features_df = future_features_df[self.features]
        except Exception as e:
            print(f"Error aligning features: {e}")
            return None

        # Check for NaNs in the final future features (should ideally be handled above)
        if future_features_df.isnull().any().any():
            print("Warning: NaNs found in final future features. Check feature generation logic.")
            # Optionally, fill NaNs here as a last resort, e.g., future_features_df.fillna(0, inplace=True)


        return future_features_df


    def predict(self, input_data_future_features):
        """
        Makes predictions using the loaded model.

        Args:
            input_data_future_features (pd.DataFrame): DataFrame containing the *exact*
                                                      features the model was trained on,
                                                      for the future dates.

        Returns:
            np.ndarray: Array of predictions, or None if prediction fails.
        """
        if self.model is None or self.features is None:
            print("Error: Model or feature list not loaded.")
            return None
        if input_data_future_features is None or input_data_future_features.empty:
             print("Error: No input data provided for prediction.")
             return None

        try:
            # Ensure columns match the order used during training
            input_aligned = input_data_future_features[self.features]

            # Ensure categorical features are handled correctly for prediction
            categorical_features = input_aligned.select_dtypes(include='category').columns.tolist()
            for col in categorical_features:
                input_aligned[col] = input_aligned[col].cat.as_ordered()


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

        # --- This part is conceptual and needs real data access ---
        # In a real scenario (like the API), you would:
        # 1. Get the prediction request (region, crop)
        # 2. Define the future dates (e.g., next 28 days/4 weeks)
        today = pd.Timestamp('2043-06-30') # Example: last date in training data
        future_prediction_dates = pd.date_range(start=today + timedelta(days=1), periods=28, freq='D')

        # 3. Load relevant recent historical data for that region/crop
        #    (This requires loading and filtering the original dataset)
        #    Example: Load last 60 days of processed data for 'Arcadia', 'Loquat'
        #    historical_df = load_and_filter_hist_data(region='Arcadia', crop='Loquat', days=60)
        #    if historical_df is not None: historical_df = preprocess_data(historical_df) # Ensure preprocessed
        print("\n--- Conceptual Prediction Example ---")
        print("NOTE: The following requires loading actual historical data, skipping for simple test.")
        # conceptual_historical_data = pd.DataFrame(...) # Replace with actual loading
        # if predictor.model and predictor.features: # Check if loaded
        #    future_features = predictor.generate_future_features(
        #        historical_data=None, # conceptual_historical_data, # Pass actual data here
        #        future_dates=future_prediction_dates,
        #        region='Arcadia',
        #        commodity='Loquat',
        #        type='Fruit'
        #    )
        #    if future_features is not None:
        #        predictions = predictor.predict(future_features)
        #        if predictions is not None:
        #            print("\nConceptual Predictions:")
        #            print(pd.DataFrame({'Date': future_prediction_dates, 'Prediction': predictions}))
        #        else:
        #            print("Conceptual prediction failed.")
        #    else:
        #         print("Conceptual future feature generation failed (likely needs historical data).")
        # --- End Conceptual Example ---

    except Exception as e:
        print(f"Failed to initialize or test Predictor: {e}")