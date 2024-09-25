#!/usr/bin/env python
# coding: utf-8

# In[6]:


import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import requests
import pickle
from datetime import datetime, timedelta
from typing import List


class BatchPredictionRequest(BaseModel):
    company_names: List[str]

def fetch_latest_data(api_url):
    response = requests.get(api_url)
    if response.status_code == 200:
        return response.json().get("markets", [])
    else:
        print("Failed to fetch data from API")
        return []

def generate_synthetic_data(end_date):
    # Generate synthetic data for the previous two months
    dates = pd.date_range(end=end_date - timedelta(days=1), periods=60)
    synthetic_data = pd.DataFrame({
        'Date': dates,
        'Name': 'AMEN BANK',  # Use the company name for which you want to generate data
        'open': np.random.randint(30, 50, size=len(dates)),
        'close': np.random.randint(30, 50, size=len(dates)),
        'low': np.random.randint(30, 50, size=len(dates)),
        'high': np.random.randint(30, 50, size=len(dates)),
        'NB_TRANSACTION': np.random.randint(5000, 20000, size=len(dates)),
        'volume': np.random.randint(5000, 50000, size=len(dates))
    })
    synthetic_data['Date'] = synthetic_data['Date'].dt.strftime('%Y-%m-%d')  # Format date as string
    return synthetic_data

def create_dataframe_cota(data, company_name):
    if not data or not isinstance(data, list):
        print("Invalid data format or missing data")
        return None

    columns = ["Date", "Name", "open", "close", "low", "high", "NB_TRANSACTION", "volume"]
    rows = []

    for entry in data:
        name = entry.get("referentiel", {}).get("stockName", "")
        if name == company_name:
            date = entry.get("arabSeance", "")
            open_price = entry.get("open", 0)
            close_price = entry.get("close", 0)
            low_price = entry.get("low", 0)
            high_price = entry.get("high", 0)
            nb_transactions = entry.get("trvolume", 0)
            volume = entry.get("volume")

            rows.append([date, name, open_price, close_price, low_price, high_price, nb_transactions, volume])

    df = pd.DataFrame(rows, columns=columns)
    df['Date'] = df['Date'].apply(arabic_to_english_month)  # Convert Arabic month names to English
    df['Date'] = pd.to_datetime(df['Date'])  # Convert 'Date' column to datetime
    df.set_index('Date', inplace=True)  # Set 'Date' as index
    df.sort_index(inplace=True)  # Sort DataFrame by index
    
    # Convert numeric columns to numeric type
    numeric_cols = ['open', 'close', 'low', 'high', 'NB_TRANSACTION', 'volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    return df

def arabic_to_english_month(arabic_date):
    # Dictionary to map Arabic month names to English
    arabic_months = {
        "يناير": "January",
        "فبراير": "February",
        "مارس": "March",
        "إبريل": "April",
        "مايو": "May",
        "يونيو": "June",
        "يوليو": "July",
        "أغسطس": "August",
        "سبتمبر": "September",
        "أكتوبر": "October",
        "نوفمبر": "November",
        "ديسمبر": "December"
    }
    
    for arabic, english in arabic_months.items():
        if arabic in arabic_date:
            arabic_date = arabic_date.replace(arabic, english)
            break
    
    return arabic_date

# Function to calculate moving averages
def calculate_moving_averages(data):
    data['moving_average_5'] = data['close'].rolling(window=5).mean()
    data['moving_average_10'] = data['close'].rolling(window=10).mean()
    data['moving_average_20'] = data['close'].rolling(window=20).mean()

# Function to calculate MACD
def calculate_macd(data):
    data['ema_12'] = data['close'].ewm(span=12, min_periods=0, adjust=False, ignore_na=False).mean()
    data['ema_26'] = data['close'].ewm(span=26, min_periods=0, adjust=False, ignore_na=False).mean()
    data['macd'] = data['ema_12'] - data['ema_26']

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))

# Function to calculate volatility
def calculate_volatility(data, window=20):
    data['volatility'] = data['close'].rolling(window=window).std()

# Function to calculate liquidity
def calculate_liquidity(data):
    data['liquidity'] = data['volume'] * data['close']

def calculate_features(data):
    calculate_moving_averages(data)
    calculate_macd(data)
    calculate_rsi(data)
    calculate_volatility(data)
    calculate_liquidity(data)

app = FastAPI()

# Load models and scalers
companies = ['AMEN BANK', 'ATL', 'ATTIJARI LEASING', 'BIAT', 'BNA', 'BT', 'CITY CARS', 'ENNAKL AUTOMOBILES']
trained_models = {}
scalers = {}
for company_name in companies:
    # Load model
    model = load_model(f"{company_name}_model.h5")
    model.compile(optimizer='rmsprop', loss='mean_squared_error')
    trained_models[company_name] = model
    # Load scaler
    with open(f"{company_name}_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    scalers[company_name] = scaler

# Prediction endpoint
@app.post("/predict_batch")
async def predict_batch(request: BatchPredictionRequest):
    predictions = []
    for company_name in request.company_names:
        # Fetch latest data for the specified company
        api_url = "https://bvmt.com.tn/rest_api/rest/market/groups/11,12,52,95,99"
        latest_data = fetch_latest_data(api_url)
        if not latest_data:
            return {"error": f"Failed to fetch data from the API for company {company_name}"}

        # Generate synthetic data for the previous two months
        today = datetime.now().date()
        synthetic_data = generate_synthetic_data(today)
        
        # Combine latest data with synthetic data
        combined_data = pd.concat([synthetic_data, create_dataframe_cota(latest_data, company_name)])

        # Calculate features
        calculate_features(combined_data)
        
        # Extract features for the latest date
        latest_features = combined_data.iloc[-1][['open', 'close', 'low', 'high', 'volume', 'NB_TRANSACTION',
                                                  'moving_average_5', 'moving_average_10', 'moving_average_20',
                                                  'macd', 'rsi', 'volatility', 'liquidity']].values

        # Scale features
        scaler = scalers[company_name]
        scaled_features = scaler.transform([latest_features])

        # Make prediction
        prediction = trained_models[company_name].predict(np.expand_dims(scaled_features, axis=0))

        predictions.append({"company": company_name, "predicted_price": float(prediction[0][0])})

    return predictions




# In[ ]:




