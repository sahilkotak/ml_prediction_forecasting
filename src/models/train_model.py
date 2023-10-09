import pandas as pd
import numpy as np
import xgboost as xgb

def train_val_split(data, days=28):
    """
    Splits the data into training and validation sets.
    The last 'days' days data is used as validation set.
    
    Parameters:
    - data: DataFrame
        The input dataset with a 'date' column.
    - days: int, optional, default = 28
        Number of days to be used for the validation set.
        
    Returns:
    - train: DataFrame
        Training dataset.
    - val: DataFrame
        Validation dataset.
    """
    
    # Determine the split date
    max_date = data['date'].max()
    split_date = max_date - pd.Timedelta(days=days)
    
    # Split the data
    train = data[data['date'] <= split_date]
    val = data[data['date'] > split_date]
    
    return train, val

def train_xgb_model(train_data, val_data, features, target):
    """
    Trains an XGBoost model.
    
    Parameters:
    - train_data: DataFrame
        Training dataset.
    - val_data: DataFrame
        Validation dataset.
    - features: list
        List of feature columns.
    - target: str
        Target variable.
        
    Returns:
    - model: XGBoost model
        Trained XGBoost model.
    """
    
    # Prepare data in DMatrix format
    dtrain = xgb.DMatrix(train_data[features], label=train_data[target])
    dval = xgb.DMatrix(val_data[features], label=val_data[target])
    
    # Hyperparameters
    params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.05,
        'max_depth': 10,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'eval_metric': 'rmse'
    }
    
    evals = [(dtrain, 'train'), (dval, 'eval')]
    
    model = xgb.train(params, 
                      dtrain, 
                      num_boost_round=1000, 
                      evals=evals, 
                      early_stopping_rounds=50, 
                      verbose_eval=100)
    
    return model