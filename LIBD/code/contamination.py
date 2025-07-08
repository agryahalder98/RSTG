# add_data_noise.py
import numpy as np
import scanpy as sc
import argparse
import os

def apply_dropout_noise(X,indices,frac_anom=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed)
    #n_samples = int(frac_anom * X.shape[0])
    #indices = np.random.choice(X.shape[0], n_samples, replace=False)
    X[indices] = X[indices] * np.random.binomial(1, 0.9, size=X[indices].shape)
    return X

def apply_batch_effect(X,indices,frac_anom=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed)
    #n_samples = int(frac_anom * X.shape[0])
    #indices = np.random.choice(X.shape[0], n_samples, replace=False)
    gene_bias = np.random.normal(loc=0.1, scale=0.05, size=X.shape[1])
    X[indices] += gene_bias
    return np.maximum(X, 0)

def apply_white_noise_contamination(X,indices,frac_anom=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed)
    X_noisy = X.copy()
    #n_samples = int(frac_anom * X.shape[0])
    #indices = np.random.choice(X.shape[0], n_samples, replace=False)
    white_noise = np.random.normal(loc=0.5, scale=1.0, size=X_noisy[indices].shape)
    X_noisy[indices] = white_noise
    return X_noisy

def apply_gaussian_noise_with_outliers(adata, frac_anom=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed)
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X.copy()

    idn1 = adata.obs['Layer'][adata.obs['Layer'].isin([1, 2, 3, 4])]
    data_outliers = adata[idn1.index]

    Nsamp = int(np.rint(adata.shape[0] * frac_anom)) + 1
    outlier_sample_count = min(Nsamp, data_outliers.shape[0])

    X[:Nsamp] = data_outliers.X[:outlier_sample_count].copy()
    adata.obs.loc[adata.obs.index[:Nsamp], "Layer"] = 8

    noise = np.random.normal(loc=0.0, scale=0.1, size=X.shape)
    return np.maximum(X + noise, 0), adata

def save_modified_data(adata,input_h5ad,output_path,filename,suffix):
    if not filename:
        filename = os.path.splitext(os.path.basename(input_h5ad))[0]
    #print(filename)
    output_file = os.path.join(os.path.dirname(output_path), f"{filename}_{suffix}.h5ad")
    #adata.write(output_file)
    print(f"Saved noisy data to: {output_file}")

def parse_args():
    parser = argparse.ArgumentParser(description="Add a single type of anomaly to spatial transcriptomic data")
    parser.add_argument('--input_h5ad', type=str, required=True, help='Path to the input .h5ad file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the noisy data')
    parser.add_argument('--frac_anom', type=float, default=0.05, help='Fraction of cells to contaminate')
    parser.add_argument('--noise_type', type=str, choices=['dropout', 'batcheffects', 'white', 'gaussian'], required=True, help='Type of noise to apply')
    parser.add_argument('--filename', type=str,default=None, help='Name of output file')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    return parser.parse_args()

def main():
    args = parse_args()
    adata = sc.read(args.input_h5ad)
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X.copy()

    n_samples = int(args.frac_anom * X.shape[0])
    indices = np.random.choice(X.shape[0], n_samples, replace=False)
    #print()

    if args.noise_type == 'dropout':
        X = apply_dropout_noise(X,indices=indices,frac_anom=args.frac_anom, seed=args.seed)
        suffix = f"dropout_{args.frac_anom}"
    elif args.noise_type == 'batcheffects':
        X = apply_batch_effect(X,indices=indices,frac_anom=args.frac_anom, seed=args.seed)
        suffix = f"batch_{args.frac_anom}"
    elif args.noise_type == 'white':
        X = apply_white_noise_contamination(X,indices=indices,frac_anom=args.frac_anom, seed=args.seed)
        suffix = f"white_{args.frac_anom}"
    elif args.noise_type == 'gaussian':
        X, adata = apply_gaussian_noise_with_outliers(adata, frac_anom=args.frac_anom, seed=args.seed)
        suffix = f"gaussian_{args.frac_anom}"

    adata.X = X
    save_modified_data(adata,args.input_h5ad,args.output_path,args.filename, suffix)

if __name__ == '__main__':
    main()
