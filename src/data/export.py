import pandas as pd

def export_to_parquet(df, days=90, filename='recent_data.parquet'):
    """
    Exports the most recent data with essential columns to a Parquet file.
    
    Parameters:
    - df: DataFrame
        The input dataset with all columns.
    - days: int, optional, default = 90
        Number of most recent days to export.
    - filename: str, optional, default = 'recent_data.parquet'
        Name of the Parquet file to export to.
        
    Returns:
    None
    """
    
    # Ensure the dataframe is sorted by date
    df = df.sort_values(by='date', ascending=False)
    
    # Select the most recent data
    df_recent = df.head(days)
    
    # Select essential columns
    essential_cols = ['item_id', 'store_id', 'sales', 'date', 'sales_trend'] + \
                    [col for col in df.columns if 'sales_lag_' in col or 'rolling_mean_' in col]
    df_recent = df_recent[essential_cols]
    
    # Export to Parquet
    df_recent.to_parquet(filename, index=False)
    
    print(f"Data exported to {filename}")