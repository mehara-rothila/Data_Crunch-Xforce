# Dockerfile (Updated to include libgomp1)

# 1. Use an official Python runtime as a parent image
FROM python:3.13-slim

# 2. Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 3. Set the working directory in the container
WORKDIR /app

# 4. Install system dependencies (including libgomp1) BEFORE Python deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# 5. Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copy project files into the container
COPY main.py .
COPY ./deployment ./deployment
COPY ./Datasets/PriceData/train_data.csv ./Datasets/PriceData/train_data.csv
COPY ./Datasets/WeatherData/train_data.csv ./Datasets/WeatherData/train_data.csv

# 7. Expose the port the app runs on
EXPOSE 8000

# 8. Define the command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]