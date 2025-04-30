# main.py
# Final version: Redirects root '/' to '/docs'

import pandas as pd
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import RedirectResponse # Import RedirectResponse
from pydantic import BaseModel, Field # For request/response models
from typing import List, Dict, Any, Optional
from datetime import date, timedelta
import uvicorn
import os
import numpy as np

# --- Project Modules Import ---
try:
    from deployment.predictor import Predictor
    from deployment.data_loader import load_training_data
    from deployment.preprocessing import preprocess_data
except ImportError as e:
    print(f"Error importing deployment modules: {e}")
    print("Ensure main.py is run from the project root directory.")
    exit()

# --- Configuration ---
HISTORICAL_DAYS_NEEDED = 60

# --- FastAPI App Initialization ---
# Add root_path if running behind a proxy, otherwise leave as default
# Note: Setting docs_url=None, redoc_url=None temporarily disables default docs
# if we only want the redirect, but it's better to keep them available.
app = FastAPI(
    title="Crop Price Prediction API",
    version="1.0.0",
    description="API to predict crop prices and accept new data entries."
    # Optional: docs_url="/swagger", redoc_url="/redoc" # To move docs if needed
)

# --- Global Variables / Resources ---
predictor = None
historical_context_df = None
latest_date_in_history = None

# Use lifespan context manager for loading resources (recommended over on_event)
# Note: This requires FastAPI 0.90.0+ and Uvicorn 0.17.3+
# If using older versions, stick to @app.on_event("startup")
# from contextlib import asynccontextmanager
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Load the ML model and data
#     global predictor, historical_context_df, latest_date_in_history
#     print("Lifespan startup: Loading resources...")
#     try:
#         predictor = Predictor()
#     except Exception as e:
#         print(f"Fatal error: Could not initialize Predictor: {e}")
#     # ... (rest of the loading logic) ...
#     print("Lifespan startup: Resources loaded.")
#     yield
#     # Clean up resources if needed on shutdown
#     print("Lifespan shutdown: Cleaning up.")
# app = FastAPI(lifespan=lifespan, ...) # Pass lifespan to FastAPI app

# Using on_event for broader compatibility for now
@app.on_event("startup")
async def startup_event():
    global predictor, historical_context_df, latest_date_in_history
    try:
        predictor = Predictor()
    except Exception as e:
        print(f"Fatal error: Could not initialize Predictor: {e}")

    print("Loading historical data for prediction context...")
    raw_hist_data = load_training_data(base_path=".")
    if raw_hist_data is not None:
        processed_hist_data = preprocess_data(raw_hist_data)
        if processed_hist_data is not None:
            latest_date_in_history = processed_hist_data['Date'].max()
            cutoff_date = latest_date_in_history - timedelta(days=HISTORICAL_DAYS_NEEDED)
            historical_context_df = processed_hist_data[processed_hist_data['Date'] > cutoff_date].copy()
            print(f"Historical data loaded and processed. Using data after {cutoff_date} for context.")
            for col in ['Region', 'Commodity', 'Type']:
                 if col in historical_context_df.columns:
                      historical_context_df[col] = historical_context_df[col].astype('category')
        else:
            print("Error: Failed to preprocess historical data.")
            historical_context_df = None
            latest_date_in_history = None
    else:
        print("Error: Failed to load raw historical data.")
        historical_context_df = None
        latest_date_in_history = None


# --- Pydantic Models ---
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
    date: date
    crop: str = Field(..., example="Cantaloupe")
    region: str = Field(..., example="Valhalla")
    priceData: PriceData


# --- API Endpoints ---

# *** Redirect root path ('/') to '/docs' ***
@app.get("/", include_in_schema=False) # Exclude from OpenAPI schema
async def root_redirect():
    """Redirects the base URL to the API documentation."""
    return RedirectResponse(url="/docs")

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_prices(request: PredictionRequest):
    """Predict future prices for a given crop and region for the next 4 weeks."""
    global predictor, historical_context_df, latest_date_in_history

    if predictor is None or historical_context_df is None or latest_date_in_history is None:
         raise HTTPException(status_code=503, detail="Service Unavailable: Model or historical data not loaded correctly.")

    print(f"Received prediction request for Crop: {request.crop}, Region: {request.region}")

    start_prediction_date = latest_date_in_history + timedelta(days=1)
    prediction_dates = pd.date_range(start=start_prediction_date, periods=28, freq='D')

    hist_filtered = historical_context_df.loc[
        (historical_context_df['Region'] == request.region) &
        (historical_context_df['Commodity'] == request.crop)
    ].copy()

    if hist_filtered.empty:
         print(f"Warning: No recent historical data found for {request.crop} in {request.region}.")
         return PredictionResponse(crop=request.crop, region=request.region, predictions=[])

    item_type = hist_filtered['Type'].iloc[0] if not hist_filtered.empty else None
    if item_type is None:
         print(f"Error: Could not determine type for {request.crop}. Check historical data.")
         raise HTTPException(status_code=500, detail=f"Could not determine type for crop {request.crop}")

    # Call predictor to generate features
    future_features = predictor.generate_future_features(
        historical_data=hist_filtered,
        future_dates=prediction_dates,
        region=request.region,
        commodity=request.crop,
        type_=item_type # Use the correct argument name 'type_'
    )

    if future_features is None or future_features.empty:
        print("Failed to generate features for future dates.")
        raise HTTPException(status_code=500, detail="Feature generation failed.")

    # Make predictions
    predictions_array = predictor.predict(future_features)

    if predictions_array is None:
        print("Prediction generation failed.")
        raise HTTPException(status_code=500, detail="Prediction failed.")

    # Ensure predictions are non-negative
    predictions_array = np.maximum(0, predictions_array)

    # Format response
    response_items = []
    for i, (pred_date, pred_price) in enumerate(zip(prediction_dates, predictions_array)):
        response_items.append(
            PredictionItem(
                prediction_index=i,
                date=pred_date.date(),
                price=round(pred_price, 2)
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
    print(f"Received weather data for Date: {data.date}, Region: {data.region}")
    print(f"Data: {data.weatherData}")
    return {"message": "Weather data received successfully (placeholder implementation)."}


@app.post("/api/data/prices", status_code=200)
async def submit_price_data(data: PriceSubmitRequest):
    """Accept new price data (placeholder)."""
    print(f"Received price data for Date: {data.date}, Crop: {data.crop}, Region: {data.region}")
    print(f"Price: {data.priceData.price}")
    return {"message": "Price data received successfully (placeholder implementation)."}


# --- Run API Server ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Uvicorn server on http://localhost:{port}")
    # Set reload=False for production/container or final local testing
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

