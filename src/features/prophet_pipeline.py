import pandas as pd
from prophet import Prophet
from joblib import dump

def aggregate_data(data):
    data_agg = data.groupby('date')['sales_revenue'].sum().reset_index()
    data_agg.to_csv("aggregated_data.csv", index=False)  # Save for inspection
    return data_agg

def prepare_data_for_prophet(data):
    data_prophet = data.rename(columns={'date': 'ds', 'sales_revenue': 'y'})
    data_prophet['ds'] = pd.to_datetime(data_prophet['ds'])
    # Create a DataFrame with all dates in the range
    all_dates = pd.DataFrame({'ds': pd.date_range(start=data_prophet.ds.min(), end=data_prophet.ds.max())})
    # Merge the data with all dates and fill missing values with zeros
    data_prophet = pd.merge(all_dates, data_prophet, on='ds', how='left').fillna(0)
    data_prophet.to_csv("prophet_ready_data.csv", index=False)
    return data_prophet

def extract_holidays(data):
    holidays = data[data['event_name'] != "NoEvent"][['date', 'event_name', 'event_type']]
    holidays = holidays.rename(columns={'date': 'ds', 'event_name': 'holiday'})
    holidays.drop(columns=['event_type'], inplace=True)
    holidays_unique = holidays.drop_duplicates()  # Drop duplicates
    holidays_unique.to_csv("holidays_data.csv", index=False)  # Save for inspection
    return holidays_unique

def train_prophet_model(data, holidays):
    model = Prophet(yearly_seasonality=True, daily_seasonality=True, holidays=holidays)
    model.fit(data)
    return model

def main_pipeline(train_data):
    print(f"Initial data shape: {train_data.shape}")

    aggregated_data = aggregate_data(train_data)
    print(f"Data shape after aggregation: {aggregated_data.shape}")

    prophet_data = prepare_data_for_prophet(aggregated_data)
    print(f"Data shape after preparing for Prophet: {prophet_data.shape}")

    holidays = extract_holidays(train_data)
    print(f"Number of unique holidays/events extracted: {holidays.shape[0]}")

    model = train_prophet_model(prophet_data, holidays)

    dump(model, 'prophet_model_with_events.joblib')
    print("Model trained and saved!")