#import matplotlib
#matplotlib.use("Agg")  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
os.environ["OMP_NUM_THREADS"] = "1"

# header）
df_behavior = pd.read_csv("user_behavior_train.csv", header=None, names=["user_id", "item_id", "behavior_type", "timestamp"])
df_item = pd.read_csv("item_train.csv", header=None, names=["item_id", "category_id", "brand_id", "price"])
df_user = pd.read_csv("user_train.csv", header=None, names=["user_id", "gender", "age", "user_level"])

# Check shape and head
print("user_behavior_train.csv loaded:", df_behavior.shape)
print(df_behavior.head(), "\n")

print("item_train.csv loaded:", df_item.shape)
print(df_item.head(), "\n")

print("user_train.csv loaded:", df_user.shape)
print(df_user.head())

from preprocessing import preprocess_behavior, preprocess_item, merge_all

df_behavior_clean = preprocess_behavior(df_behavior)
df_item_clean = preprocess_item(df_item)
df_merged = merge_all(df_behavior_clean, df_user, df_item_clean)

print("MergeData：", df_merged.shape)
print(df_merged.head())

print(df_behavior[['behavior_type', 'behavior_type_code']].head(10))

from feature_engineering import extract_user_features

# Step: Feature extraction
df_user_features = extract_user_features(df_behavior_clean)

print("Data type of each dataset feature:\n")
print("Feature:\t\tData Type:")
print(df_user_features.dtypes)

# Check missing values
print("\nThe number of missing values per feature is:\n")
print("Feature:\t\tMissing Values:")
print(df_user_features.isnull().sum())

from eda import plot_boxplot_iqr, winsorize_iqr, plot_feature_distributions

# Step: Feature extraction
df_user_features = extract_user_features(df_behavior_clean)

# Outlier check and visualization
outlier_info = plot_boxplot_iqr(df_user_features, exclude_cols=["user_id","item_id"])

# Outlier removal (IQR method)
df_user_features, summary = winsorize_iqr(df_user_features, exclude_cols=["user_id","item_id"])
print("\nWinsorize summary:")
print(summary)


plot_feature_distributions(
    df_user_features,
    cols=[
        "total_buy","total_cart","total_fav","total_actions",
        "avg_daily_actions_active","repeat_buy_rate","cart_to_buy_rate","pv_to_buy_rate"
    ],
    bins=30
)

from eda import compute_corr, plot_corr_matrix, get_numeric_cols


numeric_cols = get_numeric_cols(df_user_features, exclude_cols=["user_id", "item_id"])


corr_df, labels = compute_corr(
    df_user_features,
    cols=numeric_cols,         
    exclude_cols=None,
    method="pearson"
)


plot_corr_matrix(
    corr_df,
    labels=labels,
    title="Correlation Matrix for Cleaned User Features",
    use_seaborn=False   
)


from clustering import pca_kmeans_pipeline, plot_pca_scatter_2d


cluster_features = ["total_buy", "total_cart", "total_fav", "total_actions", "buy_rate"]

print("\n[Step] PCA + KMeans (on PCA features)")
best_k_pca, X_pca, pca_model, scaler_used = pca_kmeans_pipeline(
    df_user_features,
    features=cluster_features,
    k_range=(2, 8),
    n_components=3,     
    scale=True,
    use_minibatch=True,
    sample_size=10000,  
    random_state=42,
    show_plots=True     
)
print(f"[PCA] Suggested k by silhouette on PCA space: {best_k_pca}")


from sklearn.cluster import MiniBatchKMeans
km_vis = MiniBatchKMeans(n_clusters=best_k_pca, random_state=42, n_init="auto").fit(X_pca)
plot_pca_scatter_2d(X_pca, labels=km_vis.labels_, title="PCA(2D) scatter colored by KMeans")



#from eda import (
#    plot_kmeans_elbow_sklearn,
#    plot_kmeans_silhouette_sklearn,
#)


#cluster_features = ["total_buy", "total_cart", "total_fav", "total_pv", "total_actions", "buy_rate"]


#ks_elbow, inertias = plot_kmeans_elbow_sklearn(
#    df_user_features,
#    features=cluster_features,
#    k_range=(2, 10),
#    scale=True,
#    random_state=1,
#)


#ks_sil, silhouettes = plot_kmeans_silhouette_sklearn(
#    df_user_features,
#    features=cluster_features,
#    k_range=(2, 10),
#    scale=True,
#    random_state=1,
#)


#best_k = ks_sil[int(np.argmax(silhouettes))]
#print(f"[Info] Suggested k by silhouette: {best_k}")



# Optional: Save to CSV (if needed)
# df_user_features.to_csv("user_features.csv", index=False)

# Optional: Confirm it's done
print("User feature extraction complete in main.py")

from clustering import (
    kmeans_elbow_inertia, kmeans_silhouette_curve,
    gmm_bic_curve, gmm_silhouette_curve,
    perform_kmeans_clustering, perform_gmm_clustering,
    plot_cluster_distribution,
)

# Step 4: Define features to use for clustering
cluster_features = ["total_buy", "total_cart", "total_fav", "total_actions", "buy_rate"]
# delete total_pv because it is same as total_actions


# 1) first use elbow method to find k, bic for GMM
km_k_elbow, _, _ = kmeans_elbow_inertia(
    df_user_features, cluster_features, k_range=(2, 8), scale=True, show_plot=True
)
gmm_k_bic,  _, _, _ = gmm_bic_curve(
    df_user_features, cluster_features, k_range=(2, 8), scale=True, show_plot=True,
    covariance_type="diag", n_init=1, max_iter=200, sample_size=8000  # quick parameter
)
print(f"[Quick] KMeans elbow={km_k_elbow} | GMM BIC={gmm_k_bic}")

# 2) then use silhouette method to refine k
km_k_sil, _, _ = kmeans_silhouette_curve(
    df_user_features, cluster_features,
    k_range=(max(2, km_k_elbow-2), km_k_elbow+2),
    scale=True, sample_size=10000, use_minibatch=True, show_plot=True,  # key parameter
    
)

gmm_k_sil, _, _ = gmm_silhouette_curve(
    df_user_features, cluster_features,
    k_range=(max(2, gmm_k_bic-2), gmm_k_bic+2),
    scale=True, sample_size=8000, covariance_type="diag", n_init=1, max_iter=200, show_plot=True
)
print(f"[Silhouette] KMeans={km_k_sil} | GMM={gmm_k_sil}")

# 2) perform clustering with the best k
df_user_features_clustered = perform_kmeans_clustering(df_user_features, n_clusters=km_k_elbow)
df_user_features_clustered = perform_gmm_clustering(df_user_features_clustered, n_clusters=gmm_k_bic)

# 3)visualize clustering results
plot_cluster_distribution(df_user_features_clustered, "kmeans_cluster", title="KMeans Cluster Distribution")
plot_cluster_distribution(df_user_features_clustered, "gmm_cluster",    title="GMM Cluster Distribution")


#km_k_sil, _, _  = kmeans_silhouette_curve(
#    df_user_features, cluster_features,
#    k_range=(max(2, km_k_elbow-2), km_k_elbow+2),
#    scale=True, sample_size=10000, use_minibatch=True, show_plot=True, 
#    verbose=True
#)
#gmm_k_sil, _, _ = gmm_silhouette_curve(
#    df_user_features, cluster_features,
#    k_range=(max(2, gmm_k_bic-2), gmm_k_bic+2),
#    scale=True, sample_size=8000, covariance_type="diag",
#    n_init=1, max_iter=200, show_plot=True,
#    verbose=True
#)
#print(f"[Silhouette] KMeans={km_k_sil} | GMM={gmm_k_sil}")

from clustering import spherical_kmeans_silhouette

cos_k, ks, sils = spherical_kmeans_silhouette(
    df_user_features,
    features=["total_buy","total_cart","total_fav","total_actions","buy_rate",
              "repeat_buy_rate","cart_to_buy_rate","pv_to_buy_rate"],
    k_range=(2,10),
    scaler="quantile",
    pca_components=5,
    show_plot=True,
    sample_size=60000,       
    use_minibatch=True,      
    max_iter=100, n_init=1,  
    verbose=True
)
print("[Cosine-KMeans] best k=", cos_k, "best sil=", max(sils))

# —— Final spherical KMeans labeling on FULL data (use the same preprocessing) ——
from sklearn.preprocessing import QuantileTransformer, normalize
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import numpy as np

FINAL_K = cos_k  

skm_features = ["total_buy","total_cart","total_fav","total_actions","buy_rate",
                "repeat_buy_rate","cart_to_buy_rate","pv_to_buy_rate"]

X = df_user_features[skm_features].to_numpy()

# Quantile -> PCA(5) -> L2 normalize
qt = QuantileTransformer(output_distribution="normal", random_state=42)
Xs = qt.fit_transform(X)
pca = PCA(n_components=5, random_state=42)
Xp = pca.fit_transform(Xs)
Xn = normalize(Xp)

km = MiniBatchKMeans(n_clusters=FINAL_K, random_state=42, n_init=1, max_iter=100, batch_size=8192)
df_user_features["cluster_skm"] = km.fit_predict(Xn)

print("[Final SKM] labeled all users with k=", FINAL_K,
      "| label counts:", df_user_features["cluster_skm"].value_counts().to_dict())

# ==== PCA(2D) scatter colored by spherical K-Means labels (k=FINAL_K) ====
from sklearn.preprocessing import QuantileTransformer, normalize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


skm_features = ["total_buy","total_cart","total_fav","total_actions","buy_rate",
                "repeat_buy_rate","cart_to_buy_rate","pv_to_buy_rate"]
X = df_user_features[skm_features].to_numpy()

qt_2d = QuantileTransformer(output_distribution="normal", random_state=42)
Xs_2d = qt_2d.fit_transform(X)
pca_2d = PCA(n_components=2, random_state=42)
Xp_2d = pca_2d.fit_transform(Xs_2d)

labels = df_user_features["cluster_skm"].to_numpy()

plt.figure(figsize=(7,6))
idx = np.random.RandomState(42).choice(len(Xp_2d), size=min(100000, len(Xp_2d)), replace=False)
plt.scatter(Xp_2d[idx,0], Xp_2d[idx,1], c=labels[idx], s=4, alpha=0.6, cmap="tab10")
plt.title(f"PCA(2D) scatter colored by Spherical K-Means (k={FINAL_K})")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.tight_layout(); plt.show()


#from clustering import (
#    dbscan_preview_counts, perform_dbscan, filter_by_clusters, plot_cluster_boxplots
#)


#dbscan_features = [
#    "cart_to_buy_rate","fav_to_buy_rate","pv_to_buy_rate",
#    "avg_daily_actions_active","repeat_buy_rate",
#    "total_buy_share","total_cart_share","recency_days"
#]


#_ = dbscan_preview_counts(
#    df_user_features,
#    features=dbscan_features,
#    eps=0.8,              
#    min_samples=10,
#    scale=True,
#    sample_size=30000     
#)


#drop_list = [-1, 7]      
# min_size = 20          


#df_user_features = perform_dbscan(
#    df_user_features,
#    features=dbscan_features,
#    eps=0.7, min_samples=10,
#    algorithm="ball_tree",  
#    leaf_size=40,
#    n_jobs=1,
#    to_float32=True
#)



#from clustering import optics_dbscan_labels_fast

#df_filtered, _ = optics_dbscan_labels_fast(
#    df_user_features,
#    features=dbscan_features,
#    eps=0.6,               # DBSCAN 的 eps
#    min_samples=20,
#    xi=0.03,
#    scale=True, to_float32=True, n_jobs=1,
#    label_col="dbscan_cluster_refit",
    
#    max_eps=0.6,           
#    pca_components=2,      
#    sample_size=15000,     
#    random_state=42,
#    verbose=True
#)



#df_filtered, removed = filter_by_clusters(
#    df_filtered,
#    label_col="dbscan_cluster_refit",
#    drop_labels=[-1],    
#    # min_size=20,
#)





#plot_cluster_boxplots(
#    df_filtered,
#    label_col="dbscan_cluster_refit",
#    feature_cols=["total_buy","total_cart","total_pv"],   # 换成你要看的列
#    title_prefix="Refit: "
#)



# Step 7: Time series modeling
print("\nSTEP 7: Time Series Modeling")

from timeseries_modeling import construct_time_series, plot_user_behavior_trend

# Construct user daily behavior timeline
df_time_series = construct_time_series(df_behavior_clean)

# construct time series
df_time_series = construct_time_series(df_behavior_clean)  

# merge cluster labels
df_time_series = df_time_series.merge(
    df_user_features[["user_id", "cluster_skm"]],
    on="user_id", how="left"
)


ts_cluster = (df_time_series
              .groupby(["cluster_skm","relative_day"])[["pv","fav","cart","buy"]]
              .sum()
              .reset_index())
print("[TS] per-cluster daily totals:", ts_cluster.shape)


# Plot behavior trend for a selected user
plot_user_behavior_trend(df_time_series, user_id=1000003)

print("\nSTEP 8: Reinforcement Learning Recommendation")

from recommendation import build_user_state, q_learning_recommend, get_recommendation, ACTION_MAP

# Choose a target user for testing
user_id = 1000003
n_days = 30
current_day = 13

# Build user state sequence from time series
user_states = build_user_state(df_time_series, user_id,
                               include_cluster=True,  # let Q-learning perceive clusters
                               one_hot=False,         # set to True for more stability with 9D one-hot
                               n_clusters=9)


# Train a simple Q-learning policy over user behavior states
Q_table = q_learning_recommend(user_states, n_days=n_days)

# Get current state of the user on a given day
current_state = user_states.get(current_day, [0, 0, 0, 0])

# Generate recommendation
recommendation_key = get_recommendation(Q_table, current_state)

if recommendation_key:
    recommendation_text = ACTION_MAP.get(recommendation_key, "No description available")
    print(f"User {user_id} on Day {current_day}, Recommendation: {recommendation_key}")
    print(f"→ Action Description: {recommendation_text}")
else:
    print(f"User {user_id} on Day {current_day}, Recommendation: No recommendation (unseen state)")

#from evaluation_recommender import offline_evaluate_policy
#from recommendation_strategies import recommend_items_simple

# user_states = build_user_state(df_time_series, user_id)
# Q_table = q_learning_recommend(user_states, n_days=n_days)

#k = 10  
#metrics = offline_evaluate_policy(
#    Q_table=Q_table,
#    user_states=user_states,
#    df_behavior=df_behavior_clean,
#    df_item=df_item_clean,
#    user_id=user_id,
#    k=k,
#    n_days=n_days,
#    recommend_fn=recommend_items_simple,   
#    reward_weights=(5.0, 2.0),             
#)

#print(f"[Offline Evaluation@K={k}] → {metrics}")
