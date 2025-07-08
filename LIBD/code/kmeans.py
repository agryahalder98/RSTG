import argparse
import scanpy as sc
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from kneed import KneeLocator
import os
import celery as cel

def main(args):
    # Load the data
    data = sc.read(args.input_path)
    print("Size of spatial data [observations x genes]:", data.X.shape)

    # Visualize genes as image and centralize
    cel.getGeneImg(data, emptypixel=0, obsset=args.genes)
    data_expanded = np.expand_dims(data.GeneImg, axis=1)
    data_centralized = cel.centralize(data_expanded.copy())

    # Flatten image data
    flattened = [data_centralized[i, 0, :, :].flat for i in range(data_centralized.shape[0])]
    data_flat = np.stack(flattened).astype(np.float32)

    # If optimal_k is not given, determine it using the elbow method
    if args.optimal_k is None:
        print("Finding optimal number of clusters using the elbow method...")
        inertia = []
        cluster_range = range(args.min_clusters, args.max_clusters + 1)
        for k in tqdm(cluster_range):
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(data_flat)
            inertia.append(kmeans.inertia_)
        knee = KneeLocator(cluster_range, inertia, curve='convex', direction='decreasing')
        args.optimal_k = knee.elbow
        print(f"The optimal number of clusters is: {args.optimal_k}")
    else:
        print(f"Using user-specified number of clusters: {args.optimal_k}")

    # Clustering
    kmeans = KMeans(n_clusters=args.optimal_k, random_state=0)
    kmeans.fit(data_flat)

    # Save cluster labels
    filename = f"cluster_k_{args.optimal_k}_{os.path.basename(args.input_path).split('_')[-1].split('.')[0]}.npy"
    output_file = os.path.join(args.output_path, filename)
    #np.save(output_file, kmeans.labels_)
    print(f"Cluster labels saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster spatial gene images from .h5ad files.")
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input .h5ad file')
    parser.add_argument('--output_path', type=str, required=True, help='Directory to save the clustering output')
    parser.add_argument('--genes', nargs='+', required=True, help='List of genes to visualize (e.g., x2 x3)')
    parser.add_argument('--optimal_k', type=int, default=None, help='Optional: number of clusters to use (if not given, elbow method is used)')
    parser.add_argument('--min_clusters', type=int, default=2, help='Minimum clusters to try for elbow method')
    parser.add_argument('--max_clusters', type=int, default=100, help='Maximum clusters to try for elbow method')

    args = parser.parse_args()
    main(args)

