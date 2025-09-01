import pandas as pd
import matplotlib.pyplot as plt

def construct_time_series(df_behavior_clean):
    """
    Construct daily behavior counts for each user using relative_day as time index.
    """
    print("[TimeSeries] Constructing user daily behavior timeline...")

    df_behavior_clean['relative_day'] = (df_behavior_clean['timestamp'] - df_behavior_clean['timestamp'].min()) // (60 * 60 * 24)
    ts = df_behavior_clean.groupby(['user_id', 'relative_day', 'behavior_type']).size().unstack(fill_value=0).reset_index()

    print(f"Time series shape: {ts.shape}")
    print("Sample:\n", ts.head())
    return ts



def plot_user_behavior_trend(ts, user_id):
    """
    Plot daily behavior trend for a specific user using relative days.
    """
    user_ts = ts[ts['user_id'] == user_id].set_index('relative_day')


    # Ensure all four behavior types exist (in case some are missing)
    for col in ['pv', 'fav', 'cart', 'buy']:
        if col not in user_ts.columns:
            user_ts[col] = 0

    user_ts = user_ts[['pv', 'fav', 'cart', 'buy']]

    user_ts.plot(kind='line', figsize=(10, 4))
    plt.title(f"User {user_id} - Daily Behavior Trend (Relative Days)")
    plt.xlabel("Day")
    plt.ylabel("Action Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

