# main.py

import pandas as pd
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field # For request/response models
from typing import List, Dict, Any, Optional
from datetime import date, timedelta
import uvicorn
import os
import numpy as np

# --- Project Modules Import ---
# Assuming predictor, preprocessing, etc. are in the 'deployment' folder
try:
    from deployment.predictor import Predictor
    # Import necessary functions for historical data handling if needed
    from deployment.data_loader import load_training_data
    from deployment.preprocessing import preprocess_data
    # Feature engineering functions might be implicitly called by Predictor
except ImportError as e:
    print(f"Error importing deployment modules: {e}")
    print("Ensure main.py is run from the project root directory (D:\\Rothila\\Data_Crunch).")
    exit()

# --- Configuration ---
# Define how much historical data to load for context (enough for max lag)
HISTORICAL_DAYS_NEEDED = 60 # e.g., need at least 42 days for Price_lag_42

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Crop Price Prediction API",
    version="1.0.0",
    description="API to predict crop prices and accept new data entries."
)

# --- Global Variables / Resources ---
# Load predictor (loads model and features)
try:
    predictor = Predictor()
except Exception as e:
    print(f"Fatal error: Could not initialize Predictor: {e}")
    # In a real app, might retry or enter a safe mode. Here, we exit.
    predictor = None # Set to None to handle errors in endpoints
    # exit() # Or exit if predictor is absolutely essential

# Load historical data for feature generation context
# Load the base training data ONCE at startup. Filter within endpoint.
print("Loading historical data for prediction context...")
# Assuming data_loader/preprocessor handle paths correctly relative to project root
raw_hist_data = load_training_data(base_path=".")
if raw_hist_data is not None:
    processed_hist_data = preprocess_data(raw_hist_data)
    if processed_hist_data is not None:
        # Keep only data needed for lags (optional optimization)
        latest_date_in_history = processed_hist_data['Date'].max()
        cutoff_date = latest_date_in_history - timedelta(days=HISTORICAL_DAYS_NEEDED)
        historical_context_df = processed_hist_data[processed_hist_data['Date'] > cutoff_date].copy()
        print(f"Historical data loaded and processed. Using data after {cutoff_date} for context.")
        # Ensure categorical types are set correctly for potential filtering/merging later
        for col in ['Region', 'Commodity', 'Type']:
             if col in historical_context_df.columns:
                  historical_context_df[col] = historical_context_df[col].astype('category')
    else:
        print("Error: Failed to preprocess historical data.")
        historical_context_df = None
else:
    print("Error: Failed to load raw historical data.")
    historical_context_df = None


# --- Pydantic Models (matching api.yml) ---

class PredictionRequest(BaseModel):
    crop: str = Field(..., example="Cantaloupe")
    region: str = Field(..., example="Valhalla")

class PredictionItem(BaseModel):
    prediction_index: int
    date: date
    price: float

class PredictionResponse(BaseModel):
    crop: str
    region: str
    predictions: List[PredictionItem]

class WeatherData(BaseModel):
    rainfall: Optional[float] = Field(None, example=5.2)
    humidity: Optional[float] = Field(None, example=78.3)
    temp: Optional[float] = Field(None, example=29.4)

class WeatherSubmitRequest(BaseModel):
    date: date
    region: str = Field(..., example="Valhalla")
    weatherData: WeatherData

class PriceData(BaseModel):
    price: float = Field(..., example=86.4)

class PriceSubmitRequest(BaseModel):
    date: date # Assuming date is part of the main object, not priceData as per sample
    crop: str = Field(..., example="Cantaloupe")
    region: str = Field(..., example="Valhalla")
    priceData: PriceData # Contains the price field


# --- API Endpoints ---

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Crop Price Prediction API!"}

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_prices(request: PredictionRequest):
    """Predict future prices for a given crop and region for the next 4 weeks."""
    if predictor is None or historical_context_df is None:
         raise HTTPException(status_code=503, detail="Service Unavailable: Model or historical data not loaded.")

    print(f"Received prediction request for Crop: {request.crop}, Region: {request.region}")

    # 1. Determine prediction dates (next 4 weeks / 28 days from last known date)
    if latest_date_in_history is None:
         raise HTTPException(status_code=500, detail="Cannot determine prediction start date.")

    start_prediction_date = latest_date_in_history + timedelta(days=1)
    prediction_dates = pd.date_range(start=start_prediction_date, periods=28, freq='D')

    # 2. Filter historical data for the specific crop/region
    # Use .loc for safe filtering, especially with category types
    hist_filtered = historical_context_df.loc[
        (historical_context_df['Region'] == request.region) &
        (historical_context_df['Commodity'] == request.crop)
    ].copy()

    if hist_filtered.empty:
         print(f"Warning: No recent historical data found for {request.crop} in {request.region}.")
         # Cannot generate lags without history for this specific item
         # Return empty predictions or an error
         # raise HTTPException(status_code=404, detail=f"No historical data for {request.crop}/{request.region}")
         # Or return empty list:
         return PredictionResponse(crop=request.crop, region=request.region, predictions=[])


    # Need the 'Type' (Fruit/Vegetable) for feature generation
    item_type = hist_filtered['Type'].iloc[0] if not hist_filtered.empty else None
    if item_type is None:
         # Try fetching type from full dataset if missing in recent history (edge case)
         # Or make a reasonable default/raise error
         print(f"Warning: Could not determine type for {request.crop}. Check historical data.")
         item_type = "Unknown" # Placeholder, adjust as needed

    # 3. Generate features for future dates
    future_features = predictor.generate_future_features(
        historical_data=hist_filtered,
        future_dates=prediction_dates,
        region=request.region,
        commodity=request.crop,
        type=item_type # Pass the determined type
    )

    if future_features is None or future_features.empty:
        print("Failed to generate features for future dates.")
        raise HTTPException(status_code=500, detail="Feature generation failed.")

    # 4. Make predictions
    predictions_array = predictor.predict(future_features)

    if predictions_array is None:
        print("Prediction generation failed.")
        raise HTTPException(status_code=500, detail="Prediction failed.")

    # Ensure predictions are non-negative
    predictions_array = np.maximum(0, predictions_array)


    # 5. Format response
    response_items = []
    for i, (pred_date, pred_price) in enumerate(zip(prediction_dates, predictions_array)):
        response_items.append(
            PredictionItem(
                prediction_index=i,
                date=pred_date.date(), # Convert timestamp to date
                price=round(pred_price, 2) # Round to 2 decimal places
            )
        )

    return PredictionResponse(
        crop=request.crop,
        region=request.region,
        predictions=response_items
    )


@app.post("/api/data/weather", status_code=200)
async def submit_weather_data(data: WeatherSubmitRequest):
    """Accept new weather data (placeholder)."""
    # In a real application, this data would be stored and potentially trigger
    # model updates or be used in future predictions if forecasts are incorporated.
    print(f"Received weather data for Date: {data.date}, Region: {data.region}")
    print(f"Data: {data.weatherData}")
    # For now, just acknowledge receipt
    return {"message": "Weather data received successfully (placeholder implementation)."}


@app.post("/api/data/prices", status_code=200)
async def submit_price_data(data: PriceSubmitRequest):
    """Accept new price data (placeholder)."""
    # Similar to weather data, store/use this in a real application.
    print(f"Received price data for Date: {data.date}, Crop: {data.crop}, Region: {data.region}")
    print(f"Price: {data.priceData.price}")
    # For now, just acknowledge receipt
    return {"message": "Price data received successfully (placeholder implementation)."}


# --- Run API Server ---
if __name__ == "__main__":
    # Get port from environment variable or default to 8000
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Uvicorn server on http://localhost:{port}")
    # Use reload=True for development, disable for production/container
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)