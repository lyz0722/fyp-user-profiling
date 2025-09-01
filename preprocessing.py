import pandas as pd
import numpy as np

def preprocess_behavior(df_behavior):
    """
    Behavior data preprocessing: Encode behavior types, convert timestamps.
    """
    behavior_map = {'pv': 1, 'fav': 2, 'cart': 3, 'buy': 4} # Map behavior types to integers
    df_behavior['behavior_type_code'] = df_behavior['behavior_type'].map(behavior_map)
    df_behavior['datetime'] = pd.to_datetime(df_behavior['timestamp'])  
    df_behavior['relative_day'] = (df_behavior['timestamp'] - df_behavior['timestamp'].min()) // (60 * 60 * 24)  # Calculate relative days



    return df_behavior


def preprocess_item(df_item):
    """
    Data preprocessing of products: Handling outliers in the prices.
    """
    df_item['price'] = df_item['price'].replace(-1, np.nan) # Replace -1 with NaN
    df_item['price'] = df_item['price'].fillna(df_item['price'].median()) # Fill NaN prices with median
    return df_item


def merge_all(df_behavior, df_user, df_item):
    """
    Merge the behavior logs, user information and product information to generate complete behavior records.
    """
    df = df_behavior.merge(df_user, on='user_id', how='left') # Merge behavior with user data
    df = df.merge(df_item, on='item_id', how='left') # Merge with item data
    return df




