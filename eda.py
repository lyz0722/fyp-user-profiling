import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def infer_feature_groups(df: pd.DataFrame):
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    rate_cols  = [c for c in num_cols if c.endswith("_rate") or c.endswith("_share")]
   
    count_cols = [c for c in num_cols if c not in rate_cols]
    return count_cols, rate_cols


def plot_boxplot_iqr(dataset, exclude_cols=None, auto_exclude_rates=True):
    import numpy as np, matplotlib.pyplot as plt
    numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
    if exclude_cols:
        numeric_cols = [c for c in numeric_cols if c not in set(exclude_cols)]
    if auto_exclude_rates:
        _, rate_cols = infer_feature_groups(dataset)
        numeric_cols = [c for c in numeric_cols if c not in rate_cols]

    print("Boxplot numeric columns:", numeric_cols)
    if not numeric_cols:
        print("No numeric columns to plot.")
        return pd.DataFrame()

    plt.figure(figsize=(12, 3 * max(1, len(numeric_cols))))
    outlier_summary = []
    for i, col in enumerate(numeric_cols, 1):
        col_data = dataset[col].dropna()
        Q1 = col_data.quantile(0.25); Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1; lower = Q1 - 1.5*IQR; upper = Q3 + 1.5*IQR
        n_out = ((col_data < lower) | (col_data > upper)).sum()

        plt.subplot(len(numeric_cols), 1, i)
        plt.boxplot(col_data, vert=False)
        plt.title(f'Boxplot of {col} (Outliers: {n_out})'); plt.xlabel(col)

        outlier_summary.append({"feature":col,"Q1":Q1,"Q3":Q3,"IQR":IQR,
                                "lower_bound":lower,"upper_bound":upper,"n_outliers":int(n_out)})
    plt.tight_layout(); plt.show()
    return pd.DataFrame(outlier_summary).sort_values("n_outliers", ascending=False)


def winsorize_iqr(dataset, exclude_cols=None, auto_exclude_rates=True):
    data = dataset.copy()
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if exclude_cols:
        numeric_cols = [c for c in numeric_cols if c not in set(exclude_cols)]
    if auto_exclude_rates:
        _, rate_cols = infer_feature_groups(data)
        numeric_cols = [c for c in numeric_cols if c not in rate_cols]

    summary = []
    for col in numeric_cols:
        s = data[col].dropna()
        Q1 = s.quantile(0.25); Q3 = s.quantile(0.75)
        IQR = Q3 - Q1; lower = Q1 - 1.5*IQR; upper = Q3 + 1.5*IQR
        data[col] = data[col].clip(lower, upper)
        n_out = ((s < lower) | (s > upper)).sum()
        summary.append({"feature":col,"Q1":Q1,"Q3":Q3,"IQR":IQR,"lower":lower,"upper":upper,"n_outliers":int(n_out)})
    return data, pd.DataFrame(summary).sort_values("n_outliers", ascending=False)


import matplotlib.pyplot as plt

def plot_feature_distributions(dataset, cols, bins=30, log_for_counts=True):
    _, rate_cols = infer_feature_groups(dataset)
    for col in cols:
        if col not in dataset.columns:
            print(f"Warning: column {col} not in dataset"); continue
        x = dataset[col].dropna().values
        plt.figure(figsize=(10,6))
        if col in rate_cols:
            plt.hist(x, bins=20, range=(0,1), edgecolor="black")
        else:
            if log_for_counts:
                x = np.log1p(x)  
                plt.title(f'Distribution of {col} (log1p)')
            plt.hist(x, bins=bins, edgecolor='black')
        if "(log1p)" not in plt.gca().get_title():
            plt.title(f'Distribution of {col}')
        plt.xlabel(col); plt.ylabel('Frequency'); plt.tight_layout(); plt.show()



def get_numeric_cols(df: pd.DataFrame, exclude_cols=None):
    
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if exclude_cols:
        cols = [c for c in cols if c not in set(exclude_cols)]
    return cols

def compute_corr(df: pd.DataFrame, cols=None, exclude_cols=None, method="pearson"):
    
    use_cols = cols or get_numeric_cols(df, exclude_cols=exclude_cols)
    corr = df[use_cols].corr(method=method)
    return corr, use_cols

def plot_corr_matrix(corr_df: pd.DataFrame, labels=None, title="Correlation Matrix", use_seaborn=False):
    
    labels = labels or list(corr_df.columns)


    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_df.values, aspect='auto')
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_title(title)
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr_df.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    plt.tight_layout()
    plt.show()


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def plot_kmeans_elbow_sklearn(df, features, k_range=(2, 10), scale=True, random_state=1):
    
    X = df[features].to_numpy()
    if scale:
        X = StandardScaler().fit_transform(X)

    ks, inertias = [], []
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        km.fit(X)
        ks.append(k)
        inertias.append(km.inertia_)

    plt.figure(figsize=(7, 5))
    plt.plot(ks, inertias, marker="o")
    plt.title("Elbow Method (SSE vs k)")
    plt.xlabel("k (number of clusters)")
    plt.ylabel("SSE / Inertia")
    plt.xticks(ks)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return ks, inertias


def plot_kmeans_silhouette_sklearn(df, features, k_range=(2, 10), scale=True, random_state=1):
    
    X = df[features].to_numpy()
    if scale:
        X = StandardScaler().fit_transform(X)

    ks, sils = [], []
    for k in range(max(2, k_range[0]), k_range[1] + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        ks.append(k)
        sils.append(sil)

    plt.figure(figsize=(7, 5))
    plt.plot(ks, sils, marker="o")
    plt.title("Silhouette Score vs k")
    plt.xlabel("k (number of clusters)")
    plt.ylabel("Average Silhouette Score")
    plt.xticks(ks)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return ks, sils

