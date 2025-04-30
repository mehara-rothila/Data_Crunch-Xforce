# deployment/preprocessing.py

import pandas as pd
import os

# Import the loader function from the sibling module
try:
    from .data_loader import load_training_data
except ImportError:
    # Fallback for running the script directly
    from data_loader import load_training_data


def preprocess_data(df):
    """
    Performs basic preprocessing on the merged DataFrame.

    Args:
        df (pandas.DataFrame): The merged DataFrame from data_loader.

    Returns:
        pandas.DataFrame: The preprocessed DataFrame. Returns None if input is invalid.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        print("Error: Invalid input DataFrame.")
        return None

    # Make a copy to avoid modifying the original DataFrame
    df_processed = df.copy()

    # Rename columns for easier access
    df_processed = df_processed.rename(columns={
        'Price per Unit (Silver Drachma/kg)': 'Price'
    })
    print("Renamed 'Price per Unit (Silver Drachma/kg)' to 'Price'.")

    # Convert categorical columns to 'category' dtype for efficiency
    categorical_cols = ['Region', 'Commodity', 'Type']
    for col in categorical_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype('category')
            print(f"Converted column '{col}' to category dtype.")
        else:
            print(f"Warning: Categorical column '{col}' not found.")

    # Ensure numeric types (already seem okay from data_loader, but good practice)
    numeric_cols = ['Price', 'Temperature_K', 'Rainfall_mm', 'Humidity_pct', 'CropYieldImpactScore']
    for col in numeric_cols:
         if col in df_processed.columns:
            # Convert to float32 for potential memory savings, if precision allows
            # For now, keep float64 as loaded, unless memory becomes an issue
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
            # print(f"Ensured column '{col}' is numeric.") # Optional: uncomment if needed
         else:
             print(f"Warning: Numeric column '{col}' not found.")

    # Drop rows where conversion to numeric might have failed (if any)
    initial_rows = len(df_processed)
    df_processed.dropna(subset=numeric_cols, inplace=True)
    if len(df_processed) < initial_rows:
        print(f"Dropped {initial_rows - len(df_processed)} rows due to failed numeric conversion.")


    print("Basic preprocessing complete.")
    return df_processed

# Example usage when running this script directly
if __name__ == "__main__":
    # Assume running from project root
    project_root = "."
    print(f"Running preprocessing test from base path: {os.path.abspath(project_root)}")

    # Load data first
    raw_train_data = load_training_data(base_path=project_root)

    if raw_train_data is not None:
        print("\nRaw data loaded successfully. Starting preprocessing...")
        # Preprocess the loaded data
        processed_train_data = preprocess_data(raw_train_data)

        if processed_train_data is not None:
            print("\nPreprocessing completed successfully!")
            print("Processed Data Info:")
            processed_train_data.info()
            print("\nFirst 5 rows of processed data:")
            print(processed_train_data.head())
        else:
            print("\nPreprocessing failed.")
    else:
        print("\nFailed to load raw data, preprocessing skipped.")