import pandas as pd
import numpy as np
from utils import clustering, visualization
import plotly.graph_objects as go

def verify_clustering():
    print("Verifying Clustering Implementation...")
    
    # Create dummy data
    data = {
        'A': [1, 1.5, 3, 5, 3.5, 1, 3.2],
        'B': [1, 2, 4, 7, 5, 1.5, 4.5]
    }
    df = pd.DataFrame(data)
    features = ['A', 'B']
    
    # Test perform_kmeans
    print("Testing perform_kmeans...")
    try:
        res_df, model = clustering.perform_kmeans(df, features, n_clusters=2)
        if 'Cluster' in res_df.columns:
            print("✓ perform_kmeans successful")
            print(res_df.head())
        else:
            print("✗ perform_kmeans failed: Cluster column missing")
    except Exception as e:
        print(f"✗ perform_kmeans error: {e}")
        
    # Test calculate_metrics
    print("\nTesting calculate_metrics...")
    try:
        metrics = clustering.calculate_metrics(df, features, max_k=5)
        if 'k' in metrics and 'inertia' in metrics and 'silhouette' in metrics:
            print("✓ calculate_metrics successful")
            print(f"K values: {metrics['k']}")
            print(f"Inertia: {metrics['inertia']}")
            print(f"Silhouette: {metrics['silhouette']}")
        else:
            print("✗ calculate_metrics failed: Missing keys")
    except Exception as e:
        print(f"✗ calculate_metrics error: {e}")

    # Test Visualization (just checking for no errors)
    print("\nTesting Visualization functions...")
    try:
        fig1 = visualization.plot_elbow_curve(metrics['k'], metrics['inertia'])
        print("✓ plot_elbow_curve created")
        
        fig2 = visualization.plot_silhouette_curve(metrics['k'], metrics['silhouette'])
        print("✓ plot_silhouette_curve created")
        
        fig3 = visualization.plot_clustering_2d(res_df, 'A', 'B', 'Cluster')
        print("✓ plot_clustering_2d created")
        
    except Exception as e:
        print(f"✗ Visualization error: {e}")
        
if __name__ == "__main__":
    verify_clustering()