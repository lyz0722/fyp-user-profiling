
from typing import List
import pandas as pd
import numpy as np

# Helper to get day boundaries from relative_day
def _day_mask(df_behavior: pd.DataFrame, user_id: int, day: int):
    return (df_behavior['user_id'] == user_id) & (df_behavior['relative_day'] <= day)

def recommend_items_simple(
    user_id: int,
    day: int,
    action_key: str,
    k: int,
    df_behavior: pd.DataFrame,
    df_item: pd.DataFrame,
) -> List[int]:
    """
    Very simple, offline recommenders per action:
    - recommend_A: globally popular & cheaper items (proxy for 'popular discounted')
    - recommend_B: user's favorite category (most viewed historically up to 'day'), then popular within that category
    - recommend_C: 'newly released' approximated as items first seen recently in behavior logs (up to day), then popular among them
    Returns a ranked list of item_ids (length <= k).
    """
    hist = df_behavior[_day_mask(df_behavior, user_id, day)]
    # Global popularity up to 'day'
    global_pop = hist.groupby('item_id').size().sort_values(ascending=False)

    if action_key == 'recommend_A':
        # Popular & lower price
        pop_df = global_pop.reset_index(name='cnt')
        merged = pop_df.merge(df_item[['item_id', 'price']], on='item_id', how='left')
        merged = merged.sort_values(['cnt', 'price'], ascending=[False, True])
        return merged['item_id'].head(k).tolist()

    elif action_key == 'recommend_B':
        # Favorite category by views
        views = hist[hist['behavior_type'] == 'pv']
        fav_cat = (views.merge(df_item[['item_id', 'category_id']], on='item_id', how='left')
                        .groupby('category_id').size().sort_values(ascending=False))
        if fav_cat.empty or fav_cat.index.isnull().all():
            return global_pop.head(k).index.tolist()
        top_cat = fav_cat.index[0]
        in_cat = (hist.merge(df_item[['item_id', 'category_id']], on='item_id', how='left'))
        in_cat = in_cat[in_cat['category_id'] == top_cat].groupby('item_id').size().sort_values(ascending=False)
        return in_cat.head(k).index.tolist()

    elif action_key == 'recommend_C':
        # Newly released ~ items that first appeared near 'day' in overall logs
        first_seen = (df_behavior.groupby('item_id')['relative_day'].min())
        recent_items = first_seen[first_seen >= (day - 7)].index  # seen first within last 7 days
        recent_pop = hist[hist['item_id'].isin(recent_items)].groupby('item_id').size().sort_values(ascending=False)
        if recent_pop.empty:
            return global_pop.head(k).index.tolist()
        return recent_pop.head(k).index.tolist()

    else:
        # Fallback: global popular
        return global_pop.head(k).index.tolist()
