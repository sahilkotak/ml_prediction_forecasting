from fastapi import FastAPI, HTTPException
from joblib import load
import pandas as pd
import datetime
import xgboost as xgb

app = FastAPI()

# Load the model, encoders, and recent data
model = load('models/xgb_model_sample.joblib')
encoders_dict = load('src/encoders_dict.joblib')
recent_data = pd.read_parquet('src/recent_data.parquet')
sell_price_df = pd.read_parquet('src/weekly_sell_price.parquet')
feature_order = load('src/feature_order.joblib')

# Load the Prophet model
prophet_model = load('models/prophet_model_with_events.joblib')
prophet_data = pd.read_csv("src/prophet_ready_data.csv")
prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])  # Convert 'ds' column to datetime format


@app.get("/")
def read_root():
    return {
        "description": "This is a Retail Sales Prediction and Forecasting API. It uses machine learning models to predict sales for a given item in a specific store on a particular date. It also forecasts future sales using the Prophet model.",
        "endpoints": {
            "/": "Displays this information.",
            "/health": "Checks the health of the API.",
            "/sales/stores/items": "Predicts sales for a given item in a specific store on a particular date. Expected input parameters are 'item_id', 'store_id', 'date', 'event_name', and 'event_type'. The output is a JSON object with the predicted sales.",
            "/sales/national/": "Returns the forecasted sales volume for the next 7 days from the input date for all stores and items combined at a national level."
        },
        "github": "https://github.com/sahilkotak/ml-pred-forecasting",
    }

@app.get("/health/")
def health_check():
    return {"message": "Welcome to our Sales Prediction API!"}, 200

@app.get("/sales/stores/items/")
def get_sales_prediction(item_id: str, store_id: str, date: str, event_name: str = 'NoEvent', event_type: str = 'NoEvent'):
    # Create a dataframe with input data
    df = pd.DataFrame({
        'item_id': [item_id],
        'store_id': [store_id],
        'date': [pd.to_datetime(date)],
        'event_name': [event_name],
        'event_type': [event_type],
    })
    print("Initial DataFrame:\n", df)

    # Compute wm_yr_wk
    df['wm_yr_wk'] = df['date'].dt.strftime('11%y').astype(int) * 100 + \
                    (df['date'].dt.strftime('%W').astype(int) - 4)
    print("After computing wm_yr_wk:\n", df)
    
    # Look up the sell_price
    price_lookup = sell_price_df.query('item_id == @item_id and store_id == @store_id and wm_yr_wk == @df.wm_yr_wk.iat[0]')
    if not price_lookup.empty:
        df['sell_price'] = price_lookup['sell_price'].iat[0]
    else:
        # Use last known price if future date not available
        last_known_price = sell_price_df.query('item_id == @item_id and store_id == @store_id')['sell_price'].iloc[-1]
        df['sell_price'] = last_known_price
    print("After looking up the sell_price:\n", df)

    # Compute other required features
    df['id'] = df['item_id'] + "_" + df['store_id'] + "_evaluation"
    df['dept_id'] = df['item_id'].str.split("_").str[:2].str.join("_")
    df['cat_id'] = df['item_id'].str.split("_").str[0]
    df['state_id'] = df['store_id'].str.split("_").str[0]
    print("After computing other required features:\n", df)

    # Update the placeholders using recent data
    relevant_data = recent_data[(recent_data['item_id'] == item_id) & (recent_data['store_id'] == store_id)].sort_values(by='date', ascending=False).head(1)
    for col in ['sales_lag_7', 'rolling_mean_7_7', 'rolling_mean_7_28', 
                'sales_lag_28', 'rolling_mean_28_7', 'rolling_mean_28_28', 'sales_trend']:
        if not relevant_data.empty:
            df[col] = relevant_data[col].values[0]
        else:
            df[col] = 0
    print("After updating the placeholders using recent data:\n", df)

    df['is_weekend'] = df['date'].dt.weekday >= 5
    print("After computing is_weekend:\n", df)

    # Apply label encoding
    for col, encoder in encoders_dict.items():
        try:
            df[col] = encoder.transform(df[col])
        except ValueError as e:
            print(f"Error in column: {col}")
        # Handle or log the error
    print("After applying label encoding:\n", df)

    # Drop the 'date' column and ensure 'wm_yr_wk' is of type int
    df.drop(columns=['date'], inplace=True)
    df['wm_yr_wk'] = df['wm_yr_wk'].astype(int)
    print("After dropping the 'date' column and ensuring 'wm_yr_wk' is of type int:\n", df)

    #Re-ordering the features
    df = df[feature_order]
    print("After re-ordering the features:\n", df)

    
    # Convert the dataframe to DMatrix
    data_dmatrix = xgb.DMatrix(df)
    print("After converting the dataframe to DMatrix:\n", data_dmatrix)

    # Predict
    prediction = model.predict(data_dmatrix)
    
    print("Raw prediction:", prediction[0])

    return {"prediction": float(prediction[0])}

@app.get("/sales/national/")
def get_sales_forecast(date: str):
    print("Starting sales forecast...")
    
    # Try to convert the string date to datetime format using the specified format
    try:
        input_date = datetime.datetime.strptime(date, '%d/%m/%Y')
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Please use dd/mm/yyyy format.")
    
    print("Input date:", input_date)

    # Create a dataframe with the input date and the next 7 days
    future_dates = [input_date + datetime.timedelta(days=i) for i in range(8)]
    future = pd.DataFrame({'ds': future_dates})

    # Append the historical data from the CSV to the future dataframe
    future = pd.concat([prophet_data[['ds']], future], ignore_index=True).drop_duplicates()
    print("Future dates with historical data:\n", future)

    # Predict using Prophet
    forecast = prophet_model.predict(future)
    print("Forecast:\n", forecast)
    
    # Extract the required forecast data
    forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(8)
    print("Forecast data:\n", forecast_data)
    
    # Convert the forecast data to the desired dictionary format
    forecast_dict = {row['ds'].strftime('%d/%m/%Y'): round(row['yhat'], 2) for row in forecast_data.to_dict(orient='records')}
    
    print("Forecast dictionary:\n", forecast_dict)
    
    print("Finished sales forecast.")
    return forecast_dict
