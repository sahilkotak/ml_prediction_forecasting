from fastapi import FastAPI, HTTPException
from joblib import load
import pandas as pd
import datetime

app = FastAPI()

# Load the model, encoders, and recent data
model = load('models/xgb_model.joblib')
encoders_dict = load('encoders_dict.joblib')
recent_data = pd.read_parquet('recent_data.parquet')
sell_price_df = pd.read_parquet('weekly_sell_price.parquet')

@app.get("/")
def read_root():
    return {
        "description": "Retail Sales Prediction and Forecasting",
        "endpoints": ["/", "/health", "/sales/stores/items"],
        "github": "https://github.com/sahilkotak/ml-pred-forecasting"
    }

@app.get("/health/")
def health_check():
    return {"message": "Welcome to our Sales Prediction API!"}

@app.get("/sales/stores/items/")
def get_sales_prediction(item_id: str, store_id: str, date: str, event_name: str = 'NoEvent', event_type: str = 'NoEvent'):
    # Create a dataframe with input data
    df = pd.DataFrame({
        'item_id': [item_id],
        'store_id': [store_id],
        'date': [pd.to_datetime(date)],
        'event_name': [event_name],
        'event_type': [event_type]
    })

    # Compute wm_yr_wk
    df['wm_yr_wk'] = df['date'].dt.year * 100 + df['date'].dt.week
    
    # Look up the sell_price
    price_lookup = sell_price_df.query('item_id == @item_id and store_id == @store_id and wm_yr_wk == @df.wm_yr_wk.iat[0]')
    if not price_lookup.empty:
        df['sell_price'] = price_lookup['sell_price'].iat[0]
    else:
        # Use last known price if future date not available
        last_known_price = sell_price_df.query('item_id == @item_id and store_id == @store_id')['sell_price'].iloc[-1]
        df['sell_price'] = last_known_price

    # Compute other required features
    df['id'] = df['item_id'] + "_" + df['store_id'] + "_evaluation"
    df['dept_id'] = "_".join(df['item_id'].split("_")[:2])
    df['cat_id'] = df['item_id'].split("_")[0]
    df['state_id'] = df['store_id'].split("_")[0]
    df['wm_yr_wk'] = df['date'].dt.year * 100 + df['date'].dt.week
    df['event_name'] = 'NoEvent'
    df['event_type'] = 'NoEvent'

    # Update the placeholders using recent data
    relevant_data = recent_data[(recent_data['item_id'] == item_id) & (recent_data['store_id'] == store_id)].sort_values(by='date', ascending=False).head(1)
    for col in ['sales_lag_7', 'rolling_mean_7_7', 'rolling_mean_7_28', 
                'sales_lag_28', 'rolling_mean_28_7', 'rolling_mean_28_28', 'sales_trend']:
        if not relevant_data.empty:
            df[col] = relevant_data[col].values[0]
        else:
            df[col] = 0

    df['is_weekend'] = df['date'].dt.weekday >= 5

    # Encode categorical features using the encoders
    for col, encoder in encoders_dict.items():
        df[col] = encoder.transform(df[col])

    # Predict
    prediction = model.predict(df)

    return {"sales_prediction": prediction[0]}