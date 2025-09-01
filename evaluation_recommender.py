
from typing import Callable, Dict, List, Set, Tuple, Optional
import numpy as np
import pandas as pd

# Ranking Metrics (per user-day)

def precision_at_k(recs: List[int], gt: Set[int], k: int) -> float:
    if k == 0:
        return 0.0
    topk = recs[:k]
    if not topk:
        return 0.0
    hits = sum(1 for i in topk if i in gt)
    return hits / min(k, len(topk))

def recall_at_k(recs: List[int], gt: Set[int], k: int) -> float:
    if not gt:
        return 0.0
    topk = recs[:k]
    hits = sum(1 for i in topk if i in gt)
    return hits / len(gt)

def dcg_at_k(recs: List[int], gt: Set[int], k: int) -> float:
    dcg = 0.0
    for idx, item in enumerate(recs[:k], start=1):
        rel = 1.0 if item in gt else 0.0
        if rel > 0:
            dcg += rel / np.log2(idx + 1)
    return dcg

def ndcg_at_k(recs: List[int], gt: Set[int], k: int) -> float:
    ideal_hits = min(len(gt), k)
    if ideal_hits == 0:
        return 0.0
    ideal_dcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg_at_k(recs, gt, k) / ideal_dcg

def average_precision_at_k(recs: List[int], gt: Set[int], k: int) -> float:
    # AP = mean of precision@i at each hit position i
    if not gt:
        return 0.0
    hits = 0
    ap_sum = 0.0
    for i, item in enumerate(recs[:k], start=1):
        if item in gt:
            hits += 1
            ap_sum += hits / i
    if hits == 0:
        return 0.0
    return ap_sum / min(len(gt), k)

# For aggregation across user-days
def safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0



# Offline evaluator for policy + recsys
# recommend_fn signature:
# recommend_fn(user_id: int, day: int, action_key: str, k: int, df_behavior: pd.DataFrame, df_item: pd.DataFrame) -> List[int]

def offline_evaluate_policy(
    Q_table: Dict[Tuple[int, int, int, int], Dict[str, float]],
    user_states: Dict[int, np.ndarray],
    df_behavior: pd.DataFrame,
    df_item: pd.DataFrame,
    user_id: int,
    k: int,
    n_days: int,
    recommend_fn: Callable[[int, int, str, int, pd.DataFrame, pd.DataFrame], List[int]],
    reward_weights: Tuple[float, float] = (5.0, 2.0),
) -> Dict[str, float]:
    """
    Evaluate a learned Q policy by generating recommendations each day and comparing to next-day purchases.
    Returns averaged metrics over available days for this user.
    """
    from recommendation import get_recommendation  # lazy import to avoid circular deps

    precs, recs, ndcgs, maps, hits = [], [], [], [], []
    rewards = []

    for day in range(n_days - 1):
        state = user_states.get(day, np.array([0, 0, 0, 0]))
        action_key = get_recommendation(Q_table, state)
        if action_key is None:
            continue

        # Generate top-k recommendations for this (user, day, action)
        rec_items = recommend_fn(user_id, day, action_key, k, df_behavior, df_item)

        # Ground-truth: items the user buys on next day
        nextday_purchases = df_behavior[
            (df_behavior['user_id'] == user_id)
            & (df_behavior['relative_day'] == day + 1)
            & (df_behavior['behavior_type'] == 'buy')
        ]['item_id'].unique().tolist()
        gt = set(nextday_purchases)

        # Metrics
        precs.append(precision_at_k(rec_items, gt, k))
        recs.append(recall_at_k(rec_items, gt, k))
        ndcgs.append(ndcg_at_k(rec_items, gt, k))
        maps.append(average_precision_at_k(rec_items, gt, k))
        hits.append(1.0 if any(i in gt for i in rec_items[:k]) else 0.0)

        # RL-style reward (optional): from next-day state we can compute counts, but here approximate using purchases only
        # If behavior table has carts, you can also include them:
        nextday_cart = df_behavior[
            (df_behavior['user_id'] == user_id)
            & (df_behavior['relative_day'] == day + 1)
            & (df_behavior['behavior_type'] == 'cart')
        ]['item_id'].nunique()
        buy_w, cart_w = reward_weights
        rewards.append(buy_w * len(gt) + cart_w * nextday_cart)

    return {
        f'Precision@{k}': safe_mean(precs),
        f'Recall@{k}': safe_mean(recs),
        f'NDCG@{k}': safe_mean(ndcgs),
        f'MAP@{k}': safe_mean(maps),
        'HitRate': safe_mean(hits),
        'AvgReward': safe_mean(rewards),
        'EvaluatedDays': len(precs),
    }


def offline_evaluate_many_users(
    Q_tables: Dict[int, Dict],
    user_states_dict: Dict[int, Dict[int, np.ndarray]],
    df_behavior: pd.DataFrame,
    df_item: pd.DataFrame,
    user_ids: List[int],
    k: int,
    n_days: int,
    recommend_fn: Callable[[int, int, str, int, pd.DataFrame, pd.DataFrame], List[int]],
) -> Dict[str, float]:
    """
    Aggregate metrics across multiple users.
    Q_tables: mapping user_id -> Q_table (since you train per user in current pipeline)
    user_states_dict: user_id -> {day -> state vector}
    """
    metrics_list = []
    for uid in user_ids:
        Q = Q_tables.get(uid)
        states = user_states_dict.get(uid, {})
        if Q is None or not states:
            continue
        user_metrics = offline_evaluate_policy(Q, states, df_behavior, df_item, uid, k, n_days, recommend_fn)
        metrics_list.append(user_metrics)

    # Aggregate
    if not metrics_list:
        return {}

    keys = metrics_list[0].keys()
    agg = {}
    for kname in keys:
        vals = [m[kname] for m in metrics_list]
        agg[kname] = float(np.mean(vals))
    return agg
