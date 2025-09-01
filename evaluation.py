from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import numpy as np

def evaluate_kmeans_silhouette(df_features, cluster_labels, sample_size=10000):
    """
    Calculate silhouette score on PCA-reduced data.
    """
    print("[KMeans] Calculating Silhouette Score (with PCA)...")

    if sample_size is not None and len(df_features) > sample_size:#if sample_size is specified and df_features is larger than sample_size
        df_features = df_features.sample(sample_size, random_state=42)#if df_features is larger than sample_size, sample it
        cluster_labels = cluster_labels.loc[df_features.index]

    X = df_features.drop(columns=['user_id'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2) # Reduce to 2D for calculation
    X_pca = pca.fit_transform(X_scaled) 

    score = silhouette_score(X_pca, cluster_labels)
    print(f"Silhouette Score (PCA): {score:.4f}")
    return score



def evaluate_gmm_bic(df_features, n_components=5, sample_size=10000):
    """
    Compute BIC for GMM clustering with PCA preprocessing.
    """
    print("[GMM] Calculating BIC (with PCA)...")

    if sample_size is not None and len(df_features) > sample_size:
        df_features = df_features.sample(sample_size, random_state=42)

    X = df_features.drop(columns=['user_id'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        max_iter=50,
        tol=1e-3,
        random_state=42
    )
    gmm.fit(X_scaled)

    bic_score = gmm.bic(X_scaled)
    print(f"BIC Score: {bic_score:.2f}")
    return bic_score




