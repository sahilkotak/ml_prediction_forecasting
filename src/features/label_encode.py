from sklearn.preprocessing import LabelEncoder

def label_encode_data(df, cat_cols):
    """
    Label encodes the categorical columns of the dataframe.
    
    Parameters:
    - df: DataFrame, input dataframe with categorical columns to be encoded.
    - cat_cols: list, list of categorical column names to be encoded.
    
    Returns:
    - df: DataFrame, dataframe with categorical columns label encoded.
    - encoders: dict, dictionary of label encoders for each column.
    """
    # Drop the 'd' column
    df.drop('d', axis=1, inplace=True)
    
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    return df, encoders