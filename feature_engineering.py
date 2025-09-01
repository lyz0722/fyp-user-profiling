import pandas as pd
import numpy as np

def extract_user_features(df_behavior_clean: pd.DataFrame) -> pd.DataFrame:
    
    print("[extract_user_features] Start extracting user features...")

    
    behavior_counts = df_behavior_clean.pivot_table(
        index="user_id", columns="behavior_type", aggfunc="size", fill_value=0
    )
    for col in ["pv", "fav", "cart", "buy"]:
        if col not in behavior_counts.columns:
            behavior_counts[col] = 0
    behavior_counts.columns = [f"total_{c}" for c in behavior_counts.columns]
    behavior_counts = behavior_counts.reset_index()

    
    behavior_counts["total_actions"] = behavior_counts[
        ["total_pv", "total_fav", "total_cart", "total_buy"]
    ].sum(axis=1)
    behavior_counts["buy_rate"] = (
        behavior_counts["total_buy"] / behavior_counts["total_actions"].replace(0, np.nan)
    ).fillna(0.0)

    
    if "relative_day" in df_behavior_clean.columns:
        grp = df_behavior_clean.groupby("user_id")["relative_day"]
        active_days = grp.nunique()
        min_day = grp.min()
        max_day = grp.max()
        span_days = (max_day - min_day + 1).clip(lower=1)

        global_last_day = df_behavior_clean["relative_day"].max()
        recency_days = (global_last_day - max_day).clip(lower=0)

        time_feat = pd.DataFrame({
            "user_id": active_days.index,
            "active_days": active_days.values,
            "span_days": span_days.values,
            "recency_days": recency_days.values,
        })
    elif "datetime" in df_behavior_clean.columns and np.issubdtype(df_behavior_clean["datetime"].dtype, np.datetime64):
        tmp = (
            df_behavior_clean.assign(date=df_behavior_clean["datetime"].dt.date)
            .groupby("user_id")
            .agg(
                active_days=("date", "nunique"),
                first_time=("datetime", "min"),
                last_time=("datetime", "max"),
            )
            .reset_index()
        )
        span_days = (tmp["last_time"] - tmp["first_time"]).dt.days + 1
        span_days = span_days.clip(lower=1)

        data_end = df_behavior_clean["datetime"].max()
        recency_days = (data_end - tmp["last_time"]).dt.days.clip(lower=0)

        time_feat = tmp[["user_id"]].copy()
        time_feat["active_days"] = tmp["active_days"].astype(int)
        time_feat["span_days"] = span_days.astype(int)
        time_feat["recency_days"] = recency_days.astype(int)
    else:
        
        time_feat = pd.DataFrame({
            "user_id": behavior_counts["user_id"],
            "active_days": 1,
            "span_days": 1,
            "recency_days": 0,
        })

    df_feat = behavior_counts.merge(time_feat, on="user_id", how="left")

   
    df_feat["avg_daily_actions_active"] = (
        df_feat["total_actions"] / df_feat["active_days"].replace(0, np.nan)
    ).fillna(0.0)
    df_feat["avg_daily_actions_span"] = (
        df_feat["total_actions"] / df_feat["span_days"].replace(0, np.nan)
    ).fillna(0.0)

    
    item_div = df_behavior_clean.groupby("user_id")["item_id"].nunique().rename("unique_items")
    buy_items = (
        df_behavior_clean.loc[df_behavior_clean["behavior_type"] == "buy"]
        .groupby("user_id")["item_id"].nunique()
        .rename("unique_buy_items")
    )
    df_feat = df_feat.merge(item_div, on="user_id", how="left")
    df_feat = df_feat.merge(buy_items, on="user_id", how="left")
    df_feat[["unique_items", "unique_buy_items"]] = df_feat[
        ["unique_items", "unique_buy_items"]
    ].fillna(0).astype(int)

    rep_buy = (df_feat["total_buy"] - df_feat["unique_buy_items"]).clip(lower=0)
    df_feat["repeat_buy_rate"] = (
        rep_buy / df_feat["total_buy"].replace(0, np.nan)
    ).fillna(0.0)

   
    df_feat["pv_to_cart_rate"] = (
        df_feat["total_cart"] / df_feat["total_pv"].replace(0, np.nan)
    ).fillna(0.0)
    df_feat["pv_to_fav_rate"] = (
        df_feat["total_fav"] / df_feat["total_pv"].replace(0, np.nan)
    ).fillna(0.0)
    df_feat["pv_to_buy_rate"] = (
        df_feat["total_buy"] / df_feat["total_pv"].replace(0, np.nan)
    ).fillna(0.0)

    df_feat["cart_to_buy_rate"] = (
        df_feat["total_buy"] / df_feat["total_cart"].replace(0, np.nan)
    ).fillna(0.0)
    df_feat["fav_to_buy_rate"] = (
        df_feat["total_buy"] / df_feat["total_fav"].replace(0, np.nan)
    ).fillna(0.0)

    
    for c in ["total_pv", "total_fav", "total_cart", "total_buy"]:
        df_feat[f"{c}_share"] = (
            df_feat[c] / df_feat["total_actions"].replace(0, np.nan)
        ).fillna(0.0)

    df_feat = df_feat.fillna(0)

    print("[extract_user_features] Finished.")
    print(f"Feature shape: {df_feat.shape}")
    print("Preview:\n", df_feat.head())

    return df_feat


