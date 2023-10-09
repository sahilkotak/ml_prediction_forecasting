import pandas as pd

def create_lagged_and_rolling_features(df, lags=[7, 28], windows=[7, 28]):
    """
    Create lagged sales features and rolling averages.
    
    Parameters:
    - df: DataFrame with sales data.
    - lags: List of lags to create.
    - windows: List of rolling windows to compute.
    
    Returns:
    - DataFrame with lagged and rolling features.
    """
    for lag in lags:
        df[f'sales_lag_{lag}'] = df.groupby(['id', 'item_id', 'store_id'])['sales'].shift(lag)
        for window in windows:
            df[f'rolling_mean_{lag}_{window}'] = df.groupby(['id', 'item_id', 'store_id'])[f'sales_lag_{lag}'].transform(lambda x: x.rolling(window).mean())
    
    return df

def create_trend_and_special_day_features(df):
    """
    Create trend and special day features.
    
    Parameters:
    - df: DataFrame with sales data.
    
    Returns:
    - DataFrame with trend and special day features.
    """
    df['sales_trend'] = df['sales_lag_7'] - df['sales_lag_28']
    df['date'] = pd.to_datetime(df['date'])
    df['is_weekend'] = df['date'].dt.weekday >= 5
    
    return df
