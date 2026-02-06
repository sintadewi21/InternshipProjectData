import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

def perform_kmeans(df, features, n_clusters):
    """
    Melakukan K-Means Clustering pada data.
    
    Args:
        df: pandas DataFrame
        features: list nama kolom fitur
        n_clusters: jumlah cluster (K)
        
    Returns:
        tuple: (DataFrame dengan label cluster, model KMeans)
    """
    if df is None or not features:
        return None, None
        
    X = df[features].copy()
    X = X.dropna()
    
    if X.empty:
        return None, None
        
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    
    result_df = df.copy()
    result_df.loc[X.index, 'Cluster'] = kmeans.labels_
    
    result_df['Cluster'] = result_df['Cluster'].astype('Int64').astype(str)
    
    return result_df, kmeans

def calculate_metrics(df, features, max_k=10):
    """
    Menghitung Inertia (Elbow Method) dan Silhouette Score untuk rentang K.
    
    Args:
        df: pandas DataFrame
        features: list nama kolom fitur
        max_k: maksimum K yang akan dicek
        
    Returns:
        dict: {'k': list, 'inertia': list, 'silhouette': list}
    """
    if df is None or not features:
        return {}
        
    X = df[features].dropna()
    
    if len(X) < 2:
        return {}
        
    inertia = []
    silhouette = []
    k_values = range(2, min(max_k + 1, len(X)))
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
        
        if len(X) > k:
            score = silhouette_score(X, kmeans.labels_)
            silhouette.append(score)
        else:
            silhouette.append(0)
            
    return {
        'k': list(k_values),
        'inertia': inertia,
        'silhouette': silhouette
    }