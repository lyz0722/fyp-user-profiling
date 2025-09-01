import numpy as np
import pandas as pd
import random

# Define mapping from action labels to descriptive recommendations
ACTION_MAP = {
    'recommend_A': 'Recommend popular discounted items',
    'recommend_B': "Recommend user's favorite category",
    'recommend_C': 'Recommend newly released items'
}


def build_user_state(df_time_series, user_id, day_col='relative_day',
                     include_cluster=False, cluster_col='cluster_skm',
                     one_hot=False, n_clusters=9):
    user_ts = df_time_series[df_time_series['user_id'] == user_id]
    user_states = {}
    for _, row in user_ts.iterrows():
        day = row[day_col]
        base = [row.get('pv', 0), row.get('fav', 0), row.get('cart', 0), row.get('buy', 0)]
        if include_cluster and (cluster_col in row):
            cid = int(row[cluster_col]) if pd.notna(row[cluster_col]) else -1
            if one_hot:
                oh = [0]*n_clusters
                if 0 <= cid < n_clusters:
                    oh[cid] = 1
                state = base + oh
            else:
                state = base + [cid]
        else:
            state = base
        user_states[day] = np.array(state)
    return user_states


def q_learning_recommend(user_states, n_days=15, epsilon=0.2, alpha=0.5, gamma=0.9):
    """
    Run a simple Q-learning algorithm to learn optimal recommendation policy over n_days.
    States: user behavior [pv, fav, cart, buy]
    Actions: predefined recommendation strategies
    Reward: buy × 5 + cart × 2 (from next state)
    """
    actions = list(ACTION_MAP.keys())
    Q = {}

    for day in range(n_days - 1):
        state = tuple(user_states.get(day, [0, 0, 0, 0]))
        if state not in Q:
            Q[state] = {a: 0 for a in actions}

        # Epsilon-greedy policy
        action = random.choice(actions) if random.random() < epsilon else max(Q[state], key=Q[state].get)

        next_state = tuple(user_states.get(day + 1, [0, 0, 0, 0]))
        reward = next_state[3] * 5 + next_state[2] * 2

        if next_state not in Q:
            Q[next_state] = {a: 0 for a in actions}

        Q[state][action] += alpha * (reward + gamma * max(Q[next_state].values()) - Q[state][action])

    return Q

def get_recommendation(Q, current_state):
    """
    Recommend the best action key (e.g., 'recommend_A') based on Q-table.
    """
    state = tuple(current_state)
    if state not in Q:
        return None  # Return None for unseen states

    best_action = max(Q[state], key=Q[state].get)
    return best_action

# rl_training.py

def build_state_kmeans(row):
    """State = (KMeans cluster, activity_7d)"""
    return (int(row['kmeans_cluster']), int(row.get('act7d', 0)))

def build_state_gmm(row):
    """State = (GMM cluster, activity_7d)"""
    return (int(row['gmm_cluster']), int(row.get('act7d', 0)))


def train_q_learning(df, state_fn, actions, episodes, gamma, alpha, epsilon):
    """
    Generic Q-learning trainer.
    df: DataFrame with user features
    state_fn: function(row) -> discrete state tuple
    actions: list of available actions
    """
    Q = {}
    for ep in range(episodes):
        for _, row in df.iterrows():
            state = state_fn(row)
            if state not in Q:
                Q[state] = {a: 0 for a in actions}

            # choose action
            import random
            action = random.choice(actions) if random.random() < epsilon else max(Q[state], key=Q[state].get)

            # simulate next state and reward
            reward = row.get('total_buy', 0) * 5 + row.get('total_cart', 0) * 2

            Q[state][action] += alpha * (reward + gamma * max(Q[state].values()) - Q[state][action])

    return Q

import pandas as pd

def get_user_day_row(df: pd.DataFrame, user_id: int, day: int) -> pd.Series:
    """
    Return the single row for a given user and day from df.
    Expected columns: 'user_id' and 'day'.
    Raises ValueError if the row does not exist.
    """
    if 'user_id' not in df.columns or 'day' not in df.columns:
        raise KeyError("DataFrame must contain 'user_id' and 'day' columns.")
    row = df[(df['user_id'] == user_id) & (df['day'] == day)]
    if row.empty:
        raise ValueError(f"No row found for user_id={user_id}, day={day}.")
    return row.iloc[0]



