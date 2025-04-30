# deployment/data_loader.py

import pandas as pd
import os

def load_training_data(base_path="."):
    """
    Loads the training price and weather datasets, merges them,
    and performs initial date parsing.

    Args:
        base_path (str): The root directory of the project where
                         the 'Datasets' folder resides. Defaults to current dir.

    Returns:
        pandas.DataFrame: A merged DataFrame containing price and weather data,
                          sorted by date, region, and commodity. Returns None if
                          files are not found or an error occurs.
    """
    price_data_path = os.path.join(base_path, "Datasets", "PriceData", "train_data.csv")
    weather_data_path = os.path.join(base_path, "Datasets", "WeatherData", "train_data.csv")

    print(f"Attempting to load price data from: {price_data_path}")
    print(f"Attempting to load weather data from: {weather_data_path}")

    try:
        # Load the datasets
        # Ensure 'Date' column is parsed as datetime objects
        price_df = pd.read_csv(price_data_path, parse_dates=['Date'])
        weather_df = pd.read_csv(weather_data_path, parse_dates=['Date'])

        print(f"Loaded price data: {price_df.shape[0]} rows")
        print(f"Loaded weather data: {weather_df.shape[0]} rows")

        # Rename weather columns slightly to avoid potential conflicts if merging adds suffixes
        weather_df = weather_df.rename(columns={
            'Temperature (K)': 'Temperature_K',
            'Rainfall (mm)': 'Rainfall_mm',
            'Humidity (%)': 'Humidity_pct',
            'Crop Yield Impact Score': 'CropYieldImpactScore'
        })

        # Check for duplicates in weather data based on Date and Region before merge
        initial_weather_rows = len(weather_df)
        weather_df = weather_df.drop_duplicates(subset=['Date', 'Region'], keep='first')
        if len(weather_df) < initial_weather_rows:
            print(f"Dropped {initial_weather_rows - len(weather_df)} duplicate Date/Region rows from weather data.")


        # Merge the datasets based on Date and Region
        # Use a left merge to keep all price records and add corresponding weather data
        merged_df = pd.merge(price_df, weather_df, on=['Date', 'Region'], how='left')

        print(f"Merged data shape: {merged_df.shape}")

        # Handle potential missing weather data (if left merge resulted in NaNs)
        # We might need a strategy later (e.g., imputation), but for now, just note it.
        missing_weather = merged_df['Temperature_K'].isnull().sum()
        if missing_weather > 0:
            print(f"Warning: {missing_weather} rows have missing weather data after merge.")

        # Sort data for consistency (useful for time series)
        merged_df = merged_df.sort_values(by=['Date', 'Region', 'Commodity'])

        return merged_df

    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Please ensure the CSV files exist at the specified paths.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        return None

# Example usage when running this script directly
if __name__ == "__main__":
    # Assuming the script is run from the 'deployment' directory,
    # the base path needs to go up one level to find 'Datasets'
    # If running from the project root, base_path="." would work.
    # Let's assume running from project root for consistency with main.py later
    project_root = "." # Adjust if necessary, e.g., ".." if running from deployment/

    print(f"Running data loader test from base path: {os.path.abspath(project_root)}")
    train_data = load_training_data(base_path=project_root)

    if train_data is not None:
        print("\nTraining data loaded successfully!")
        print("First 5 rows:")
        print(train_data.head())
        print("\nLast 5 rows:")
        print(train_data.tail())
        print("\nData Info:")
        train_data.info()
    else:
        print("\nFailed to load training data.")