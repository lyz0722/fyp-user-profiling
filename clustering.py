import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, MiniBatchKMeans
import time
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS, cluster_optics_dbscan


def _elbow_from_inertia(ks, inertias):
    
    ks = np.array(ks, dtype=float)
    ys = np.array(inertias, dtype=float)
    x = (ks - ks.min()) / (ks.max() - ks.min() + 1e-12)
    y = (ys - ys.min()) / (ys.max() - ys.min() + 1e-12)
    p1, p2 = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    v = p2 - p1; v /= (np.linalg.norm(v) + 1e-12)
    dists = [np.linalg.norm(np.array([xi, yi]) - (p1 + np.dot(np.array([xi, yi]) - p1, v) * v))
             for xi, yi in zip(x, y)]
    return int(ks[int(np.argmax(dists))])

def kmeans_elbow_inertia(df, features, k_range=(2,10), scale=True, random_state=1, show_plot=True):
    
    X = df[features].to_numpy()
    if scale:
        X = StandardScaler().fit_transform(X)
    ks, inertias = [], []
    for k in range(k_range[0], k_range[1]+1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto").fit(X)
        ks.append(k); inertias.append(km.inertia_)
    best_k = _elbow_from_inertia(ks, inertias)
    if show_plot:
        plt.figure(figsize=(7,5))
        plt.plot(ks, inertias, marker="o"); 
        idx = ks.index(best_k)
        plt.scatter([best_k],[inertias[idx]], s=120, edgecolors="black")
        plt.annotate(f"Elbow k={best_k}", xy=(best_k,inertias[idx]),
                     xytext=(best_k+0.2,inertias[idx]*1.05),
                     arrowprops=dict(arrowstyle="->"))
        plt.title("Elbow Method (SSE vs k)"); plt.xlabel("k"); plt.ylabel("SSE/Inertia")
        plt.xticks(ks); plt.grid(alpha=.3); plt.tight_layout(); plt.show()
    return best_k, ks, inertias

def kmeans_silhouette_curve(
    df, features, k_range=(2, 10), scale=True, random_state=1,
    show_plot=True, sample_size=10000, use_minibatch=True, verbose=True
):
    
    X = df[features].to_numpy()
    if scale:
        X = StandardScaler().fit_transform(X)

    # sample
    if X.shape[0] > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X.shape[0], size=sample_size, replace=False)
        X_eval = X[idx]
    else:
        X_eval = X

    Cls = MiniBatchKMeans if use_minibatch else KMeans
    ks, sils = [], []

    if verbose:
        print(f"[KMeans/sil] eval_n={X_eval.shape[0]}, k_range={k_range}, minibatch={use_minibatch}")

    for k in range(max(2, k_range[0]), k_range[1] + 1):
        t0 = time.time()
        km = Cls(n_clusters=k, random_state=random_state, n_init="auto")
        labels = km.fit_predict(X_eval)
        sil = silhouette_score(X_eval, labels)
        dt = time.time() - t0
        ks.append(k); sils.append(sil)
        if verbose:
            print(f"  k={k:2d} | silhouette={sil:.4f} | {dt:.2f}s")

    best_k = ks[int(np.argmax(sils))]

    if show_plot:
        plt.figure(figsize=(7, 5))
        plt.plot(ks, sils, marker="o")
        plt.title("Silhouette Score vs k (KMeans)")
        plt.xlabel("k (number of clusters)"); plt.ylabel("Average Silhouette Score")
        plt.xticks(ks); plt.grid(alpha=0.3); plt.tight_layout()
        plt.show()

    return best_k, ks, sils

def gmm_bic_curve(
    df, features, k_range=(2, 10), scale=True, random_state=1, show_plot=True,
    covariance_type="diag", n_init=1, max_iter=200, reg_covar=1e-6,
    sample_size=None, verbose=True
):
    X = df[features].to_numpy()
    if scale:
        X = StandardScaler().fit_transform(X)

    # 可选抽样
    if sample_size and X.shape[0] > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X.shape[0], size=sample_size, replace=False)
        X_use = X[idx]
    else:
        X_use = X

    ks, bics, aics = [], [], []
    if verbose:
        print(f"[GMM/BIC] n={X_use.shape[0]}, k_range={k_range}, cov={covariance_type}, n_init={n_init}, max_iter={max_iter}")

    for k in range(k_range[0], k_range[1] + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,   
            reg_covar=reg_covar,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
            init_params="kmeans",
        ).fit(X_use)
        ks.append(k)
        bics.append(gmm.bic(X_use))
        aics.append(gmm.aic(X_use))
        if verbose:
            print(f"  k={k:2d} | BIC={bics[-1]:.1f} | AIC={aics[-1]:.1f}")

    best_k = ks[int(np.argmin(bics))]

    if show_plot:
        plt.figure(figsize=(7,5))
        plt.plot(ks, bics, marker="o", label="BIC")
        plt.plot(ks, aics, marker="o", linestyle="--", label="AIC")
        idx = ks.index(best_k)
        plt.scatter([best_k],[bics[idx]], s=120, edgecolors="black")
        plt.annotate(f"Best k (BIC)={best_k}", xy=(best_k,bics[idx]),
                     xytext=(best_k+0.2,bics[idx]*1.02), arrowprops=dict(arrowstyle="->"))
        plt.title("GMM Model Selection (lower is better)")
        plt.xlabel("k"); plt.ylabel("Criterion"); plt.xticks(ks); plt.grid(alpha=.3); plt.legend()
        plt.tight_layout(); plt.show()

    return best_k, ks, bics, aics


def gmm_silhouette_curve(
    df, features, k_range=(2, 10), scale=True, random_state=1,
    show_plot=True, sample_size=8000, covariance_type="diag",
    n_init=1, max_iter=200, verbose=True
):
    
    X = df[features].to_numpy()
    if scale:
        X = StandardScaler().fit_transform(X)

    if X.shape[0] > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X.shape[0], size=sample_size, replace=False)
        X_eval = X[idx]
    else:
        X_eval = X

    ks, sils = [], []

    if verbose:
        print(f"[GMM/sil] eval_n={X_eval.shape[0]}, k_range={k_range}, cov={covariance_type}, n_init={n_init}, max_iter={max_iter}")

    for k in range(max(2, k_range[0]), k_range[1] + 1):
        t0 = time.time()
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,  
            reg_covar=1e-6,                  
            random_state=random_state,
            n_init=n_init, max_iter=max_iter,
            init_params="kmeans",
        )
        labels = gmm.fit_predict(X_eval)
        sil = silhouette_score(X_eval, labels)
        dt = time.time() - t0
        ks.append(k); sils.append(sil)
        if verbose:
            print(f"  k={k:2d} | silhouette={sil:.4f} | {dt:.2f}s")

    best_k = ks[int(np.argmax(sils))]

    if show_plot:
        plt.figure(figsize=(7, 5))
        plt.plot(ks, sils, marker="o")
        plt.title("Silhouette Score vs k (GMM)")
        plt.xlabel("k (number of components)"); plt.ylabel("Average Silhouette Score")
        plt.xticks(ks); plt.grid(alpha=0.3); plt.tight_layout()
        plt.show()

    return best_k, ks, sils


def perform_kmeans_clustering(df_features, n_clusters=5):
    """
    Apply K-Means clustering to user features.
    """
    print("[KMeans] Start clustering...")

    
    numeric_cols = (
        df_features.select_dtypes(include=[np.number])
        .columns.difference(['user_id', 'kmeans_cluster', 'gmm_cluster'])
        .tolist()
    )
    feature_data = df_features[numeric_cols].fillna(0)

    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_data)

    # Run K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    cluster_labels = kmeans.fit_predict(scaled_features)

    # Append labels
    df_features['kmeans_cluster'] = cluster_labels

    print("[KMeans] Clustering complete.")
    print("Cluster distribution:\n", df_features['kmeans_cluster'].value_counts().sort_index())

    return df_features


def perform_gmm_clustering(df_features, n_clusters=5):
    """
    Apply Gaussian Mixture Model clustering to user features.
    """
    print("[GMM] Start clustering...")

    
    numeric_cols = (
        df_features.select_dtypes(include=[np.number])
        .columns.difference(['user_id', 'kmeans_cluster', 'gmm_cluster'])
        .tolist()
    )
    feature_data = df_features[numeric_cols].fillna(0)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_data)

    
    gmm = GaussianMixture(n_components=n_clusters, random_state=42, covariance_type="full", reg_covar=1e-6)
    cluster_labels = gmm.fit_predict(scaled_features)

    df_features['gmm_cluster'] = cluster_labels

    print("[GMM] Clustering complete.")
    print("Cluster distribution:\n", df_features['gmm_cluster'].value_counts().sort_index())

    return df_features



def plot_cluster_distribution(df, cluster_column, title=None):
    """
    Plot bar chart of cluster distribution.
    """
    counts = df[cluster_column].value_counts().sort_index()
    counts.plot(kind='bar')

    plot_title = title if title else f"{cluster_column} distribution"
    plt.title(plot_title)
    plt.xlabel("Cluster")
    plt.ylabel("Number of users")
    plt.tight_layout()
    plt.show()

def pca_fit_transform(df, features, n_components=None, scale=True, random_state=42):
    
    X = df[features].to_numpy()
    scaler = StandardScaler() if scale else None
    X_scaled = scaler.fit_transform(X) if scale else X

    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca, scaler, X_scaled


def plot_pca_variance(pca, title="PCA Explained Variance"):
    
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    xs = np.arange(1, len(evr) + 1)

    plt.figure(figsize=(7,5))
    plt.plot(xs, evr, marker="o", label="Explained variance ratio")
    plt.plot(xs, cum, marker="o", linestyle="--", label="Cumulative explained variance")
    plt.xlabel("Principal component")
    plt.ylabel("Ratio")
    plt.title(title)
    plt.xticks(xs)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def kmeans_silhouette_on_matrix(
    X, k_range=(2,10), use_minibatch=True, sample_size=10000, random_state=42, show_plot=True
):
    
    
    if X.shape[0] > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X.shape[0], size=sample_size, replace=False)
        X_eval = X[idx]
    else:
        X_eval = X

    Cls = MiniBatchKMeans if use_minibatch else KMeans
    ks, sils = [], []
    for k in range(max(2, k_range[0]), k_range[1] + 1):
        km = Cls(n_clusters=k, random_state=random_state, n_init="auto")
        labels = km.fit_predict(X_eval)
        sil = silhouette_score(X_eval, labels)
        ks.append(k); sils.append(sil)

    best_k = ks[int(np.argmax(sils))]

    if show_plot:
        plt.figure(figsize=(7,5))
        plt.plot(ks, sils, marker="o")
        plt.title("Silhouette Score vs k (KMeans on PCA)")
        plt.xlabel("k (number of clusters)")
        plt.ylabel("Average Silhouette Score")
        plt.xticks(ks)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    return best_k, ks, sils


def plot_pca_scatter_2d(X_pca, labels=None, title="PCA 2D Scatter"):
    
    if X_pca.shape[1] < 2:
        raise ValueError("X_pca must have at least 2 components for 2D scatter.")
    plt.figure(figsize=(7,6))
    if labels is None:
        plt.scatter(X_pca[:,0], X_pca[:,1], s=8, alpha=0.6)
    else:
        
        for lab in np.unique(labels):
            mask = (labels == lab)
            plt.scatter(X_pca[mask,0], X_pca[mask,1], s=8, alpha=0.7, label=f"Cluster {lab}")
        plt.legend()
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def pca_kmeans_pipeline(
    df, features, k_range=(2,10), n_components=3, scale=True,
    use_minibatch=True, sample_size=10000, random_state=42, show_plots=True
):
    
    # 1) PCA
    X_pca, pca, scaler, X_scaled = pca_fit_transform(
        df, features, n_components=n_components, scale=scale, random_state=random_state
    )
    if show_plots:
        plot_pca_variance(pca, title="PCA Explained Variance (on cluster features)")

    # 2) silhouette 
    best_k, ks, sils = kmeans_silhouette_on_matrix(
        X_pca, k_range=k_range, use_minibatch=use_minibatch,
        sample_size=sample_size, random_state=random_state, show_plot=show_plots
    )

    return best_k, X_pca, pca, scaler

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# ---------- 通用：多种稳健缩放 ----------
def _scale_matrix(X: np.ndarray, how: str = "standard"):
    if how == "standard":
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
    elif how == "robust":
        scaler = RobustScaler()
        Xs = scaler.fit_transform(X)
    elif how == "quantile":
        scaler = QuantileTransformer(output_distribution="normal", random_state=42)
        Xs = scaler.fit_transform(X)
    else:
        raise ValueError("how must be one of {'standard','robust','quantile'}")
    return Xs, scaler

# ---------- 1) Spherical KMeans（余弦） ----------
def spherical_kmeans_silhouette(
    df: pd.DataFrame,
    features: List[str],
    k_range: Tuple[int,int] = (2,10),
    scaler: str = "quantile",
    pca_components: Optional[int] = None,
    show_plot: bool = True,
    random_state: int = 42,
    sample_size: int = 60000,         # ✅ 新增：抽样，默认 6 万
    use_minibatch: bool = True,       # ✅ 新增：默认用 MiniBatch
    max_iter: int = 100,              # ✅ 新增：收敛上限
    n_init: int = 1,                  # ✅ 新增：初始化次数
    verbose: bool = True,
):
    X = df[features].to_numpy()
    Xs, _ = _scale_matrix(X, how=scaler)
    if pca_components:
        pca = PCA(n_components=pca_components, random_state=random_state)
        Xs = pca.fit_transform(Xs)

    # 余弦：L2 归一化
    Xn = normalize(Xs)

    # ✅ 抽样评估，避免对千万级样本全量跑
    if Xn.shape[0] > (sample_size or 0):
        rng = np.random.default_rng(random_state)
        idx = rng.choice(Xn.shape[0], size=sample_size, replace=False)
        X_eval = Xn[idx]
    else:
        X_eval = Xn

    Cls = MiniBatchKMeans if use_minibatch else KMeans
    ks, sils = [], []
    if verbose:
        print(f"[Spherical-KMeans] eval_n={X_eval.shape[0]}, k_range={k_range}, minibatch={use_minibatch}")

    for k in range(max(2, k_range[0]), k_range[1]+1):
        t0 = time.time()
        km = Cls(n_clusters=k, random_state=random_state, n_init=n_init, max_iter=max_iter)
        labels = km.fit_predict(X_eval)
        sil = silhouette_score(X_eval, labels)
        ks.append(k); sils.append(sil)
        if verbose:
            print(f"  k={k:2d} | silhouette={sil:.4f} | {time.time()-t0:.2f}s", flush=True)

    best_k = ks[int(np.argmax(sils))]
    best_score = max(sils)

    if show_plot:
        plt.figure(figsize=(7,5))
        plt.plot(ks, sils, marker="o")
        plt.title("Silhouette (Spherical KMeans / cosine)")
        plt.xlabel("k"); plt.ylabel("silhouette")
        plt.xticks(ks); plt.grid(alpha=.3); plt.tight_layout(); plt.show()

    print(f"[Spherical-KMeans] best k={best_k}, silhouette={best_score:.3f}")
    return best_k, ks, sils



import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1) 只做一次“预览”，不改动原 df：返回每个簇（含 -1）的计数
def dbscan_preview_counts(df, features, eps, min_samples, scale=True, sample_size=None, random_state=42):
    X = df[features].to_numpy()
    if scale:
        X = StandardScaler().fit_transform(X)

    # 可选抽样（大数据更快）
    if sample_size and X.shape[0] > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X.shape[0], size=sample_size, replace=False)
        X_use = X[idx]
    else:
        X_use = X

    labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(X_use)
    sizes = pd.Series(labels).value_counts().sort_index().to_frame(name="DBSCAN_size")
    print("\n[Preview] DBSCAN_size (includes -1 noise):")
    print(sizes)
    return sizes  # 只返回计数，方便你决定删哪些簇


# 2) 实际聚类：把标签写回 df[label_col]
def perform_dbscan(
    df, features, eps, min_samples, scale=True, label_col="dbscan_cluster",
    algorithm="ball_tree",   # ← 省内存：优先试 ball_tree；还不行换 "brute"
    leaf_size=40,            # ← 控制树的块大小
    n_jobs=1,                # ← 避免并行复制大内存
    to_float32=True          # ← 降精度，省一半内存
):
    X = df[features].to_numpy()
    scaler = StandardScaler() if scale else None
    X_scaled = scaler.fit_transform(X) if scale else X
    if to_float32:
        X_scaled = X_scaled.astype(np.float32)  # ← 直接减半内存占用

    model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        algorithm=algorithm,   # ← 'ball_tree' 或 'brute'
        leaf_size=leaf_size,
        n_jobs=n_jobs
    )
    labels = model.fit_predict(X_scaled)

    df[label_col] = labels
    vc = pd.Series(labels).value_counts().sort_index()
    print("\n[DBSCAN] label distribution (includes -1):")
    print(vc)
    return df

def perform_dbscan_fast(
    df, features, *,
    eps=0.6, min_samples=20,
    scale=True, to_float32=True,
    algorithm="brute",      # brute 更省内存，速度稳定
    leaf_size=40,
    n_jobs=1,               # 避免并行复制大矩阵
    label_col="dbscan_cluster_refit",
    verbose=True
):
    X = df[features].to_numpy()
    scaler = StandardScaler() if scale else None
    X_scaled = scaler.fit_transform(X) if scale else X
    if to_float32:
        X_scaled = X_scaled.astype(np.float32)

    if verbose:
        print(f"\n[DBSCAN-fast] n={X_scaled.shape[0]}, d={X_scaled.shape[1]}, "
              f"eps={eps}, min_samples={min_samples}, alg={algorithm}")

    t0 = time.time()
    model = DBSCAN(
        eps=eps, min_samples=min_samples,
        algorithm=algorithm, leaf_size=leaf_size,
        n_jobs=n_jobs
    )
    labels = model.fit_predict(X_scaled)
    dt = time.time() - t0

    df[label_col] = labels
    vc = pd.Series(labels).value_counts().sort_index()
    print("[DBSCAN-fast] label distribution:")
    print(vc)
    print(f"[DBSCAN-fast] finished in {dt:.2f}s")

    return df

# 3) 过滤工具：按“指定簇列表”或“最小簇大小”删除样本
def filter_by_clusters(df, label_col, drop_labels=None, min_size=None, verbose=True):
    """
    - drop_labels: 要删除的簇标签列表（例如 [-1, 7]）
    - min_size: 小于该大小的簇一并删除（和 drop_labels 叠加）
    返回：过滤后的 df、被删的标签集合
    """
    s = df[label_col].value_counts()
    to_drop = set()

    if drop_labels:
        to_drop |= set(drop_labels)

    if min_size is not None:
        small = s[s < int(min_size)].index.tolist()
        to_drop |= set(small)

    if verbose:
        print(f"\n[Filter] removing clusters: {sorted(to_drop)}")
    df_new = df[~df[label_col].isin(to_drop)].copy()
    if verbose:
        print(f"[Filter] kept {len(df_new)}/{len(df)} rows")
    return df_new, sorted(to_drop)


# 4) 画箱线图（只用 matplotlib）
def plot_cluster_boxplots(df, label_col, feature_cols, title_prefix=""):
    clusters = sorted(df[label_col].unique())
    n = len(feature_cols)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
    if n == 1:
        axes = [axes]
    for i, feat in enumerate(feature_cols):
        data = [df[df[label_col]==c][feat].dropna().values for c in clusters]
        axes[i].boxplot(data, labels=clusters)
        axes[i].set_title(f"{title_prefix}{feat} by {label_col}")
        axes[i].set_xlabel("Cluster"); axes[i].set_ylabel(feat)
    plt.tight_layout(); plt.show()

def optics_dbscan_labels(
    df, features, *,
    eps=0.7,                 # 想要的 DBSCAN 半径
    min_samples=10,          # 同 DBSCAN
    xi=0.03,                 # OPTICS 的分段灵敏度，默认即可
    min_cluster_size=None,   # 可不填
    scale=True,
    to_float32=True,
    n_jobs=1,
    label_col="dbscan_from_optics",
    verbose=True
):
    X = df[features].to_numpy()
    scaler = StandardScaler() if scale else None
    Xs = scaler.fit_transform(X) if scale else X
    if to_float32:
        Xs = Xs.astype(np.float32)

    if verbose:
        print(f"\n[OPTICS] n={Xs.shape[0]}, d={Xs.shape[1]}, min_samples={min_samples}, xi={xi}")

    t0 = time.time()
    optics = OPTICS(
        min_samples=min_samples,
        xi=xi,
        min_cluster_size=min_cluster_size,
        n_jobs=n_jobs
    )
    optics.fit(Xs)
    dt = time.time() - t0
    if verbose:
        print(f"[OPTICS] fitted in {dt:.2f}s; now extract DBSCAN labels with eps={eps}")

    labels = cluster_optics_dbscan(
        reachability=optics.reachability_,
        core_distances=optics.core_distances_,
        ordering=optics.ordering_,
        eps=eps
    )
    df[label_col] = labels
    vc = pd.Series(labels).value_counts().sort_index()
    print("[OPTICS→DBSCAN] label distribution:")
    print(vc)
    return df, optics

def optics_dbscan_labels_fast(
    df, features, *,
    eps=0.6,
    min_samples=20,
    xi=0.03,
    scale=True,
    to_float32=True,
    n_jobs=1,
    label_col="dbscan_cluster_refit",
    verbose=True,
    # 新增的关键参数 ↓↓↓
    max_eps=0.6,              # 限制 OPTICS 的最大半径（非常重要）
    pca_components=2,         # 先PCA降到低维再聚类
    sample_size=15000,         # 例如 60000；不填则全量
    random_state=42,
):
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import OPTICS, cluster_optics_dbscan

    X = df[features].to_numpy()
    scaler = StandardScaler() if scale else None
    Xs = scaler.fit_transform(X) if scale else X
    if to_float32:
        Xs = Xs.astype(np.float32)

    # PCA 降维
    if pca_components is not None:
        pca = PCA(n_components=pca_components, random_state=random_state)
        Xs = pca.fit_transform(Xs)

    n_total = Xs.shape[0]
    if verbose:
        print(f"\n[OPTICS-fast] n={n_total}, d={Xs.shape[1]}, "
              f"min_samples={min_samples}, xi={xi}, max_eps={max_eps}, eps(final)={eps}")

    # 可选抽样
    if sample_size is not None and n_total > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n_total, size=sample_size, replace=False)
        X_sub = Xs[idx]
        if verbose:
            print(f"[OPTICS-fast] using subsample: {len(X_sub)} rows")

        optics = OPTICS(
            min_samples=min_samples,
            xi=xi,
            max_eps=max_eps,        # ★ 限制半径，上墙提速
            n_jobs=n_jobs,
        )
        optics.fit(X_sub)
        labels_sub = cluster_optics_dbscan(
            reachability=optics.reachability_,
            core_distances=optics.core_distances_,
            ordering=optics.ordering_,
            eps=eps
        )

        # 把未抽样的样本用最近子样本的标签“跟簇”
        # 把未抽样的样本用最近子样本的标签“跟簇”
        nn = NearestNeighbors(n_neighbors=1, n_jobs=n_jobs, algorithm="ball_tree", leaf_size=40)
        nn.fit(X_sub)

        N = Xs.shape[0]
        labels_full = np.empty(N, dtype=np.int32)

        batch = 20000  # 分批大小，可以根据内存调整
        for start in range(0, N, batch):
            end = min(N, start + batch)
            if verbose:
                print(f"[OPTICS-fast] propagate {start}:{end}/{N} ...", flush=True)
            ind = nn.kneighbors(Xs[start:end], return_distance=False)
            labels_full[start:end] = labels_sub[ind.ravel()]

        df[label_col] = labels_full
        vc = pd.Series(labels_full).value_counts().sort_index()
        print("[OPTICS-fast] label distribution (propagated):")
        print(vc)
        return df, optics


    # 全量（有 max_eps + PCA 也会快很多）
    optics = OPTICS(
        min_samples=min_samples,
        xi=xi,
        max_eps=max_eps,          # ★ 限制半径
        n_jobs=n_jobs,
    )
    optics.fit(Xs)
    labels = cluster_optics_dbscan(
        reachability=optics.reachability_,
        core_distances=optics.core_distances_,
        ordering=optics.ordering_,
        eps=eps
    )
    df[label_col] = labels
    vc = pd.Series(labels).value_counts().sort_index()
    print("[OPTICS-fast] label distribution:")
    print(vc)
    return df, optics

