
# RSTG: Robust Generation of High-quality ST data using Beta Divergence-based AutoEncoder


Spatio-transcriptomics (ST) has emerged as a transformative technology in biomedical research, enabling the spatial mapping of gene expression across tissues. However, generating high-quality spatial transcriptomic data remains challenging due to sample scarcity, technical noise, and biological variability.

Here, we introduce RSTG (Robust Spatial Transcriptomic Generator)—a robust generative framework designed to produce realistic and reliable ST data while maintaining structural fidelity even under noisy conditions.

RSTG operates in two key stages: first, it employs a robust autoencoder framework with β-ELBO loss to denoise and generate high-fidelity synthetic spatial transcriptomic samples by learning the data’s latent distribution. In the second stage, the generated data is used to train a downstream neural network for accurate recovery of 2D spatial coordinates and spatial domain, enhancing performance even under data contamination.


## Getting Started

## System requirements
### Hardware requirements
The running of RSTG requires only a standard computer(limited performance) with enough RAM to support the operations defined by a user. For optimal performance, we recommend a computer with the following specs:

RAM: 16+ GB
CPU: 4+ cores, 3.3+ GHz/core
GPU: 16GB(RTX series)
### Software requirements
The package development version is tested on Linux and Windows operating systems

Python (>3.8) support packages: torch>=1.8, pandas>=1.4, numpy>=1.20, scipy, tqdm, scanpy>=1.5, anndata, sklearn, scikit-image, CuDNN, CUDA

## **Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgement](#cknowledgement)


## **Installation**
`conda` virtual environment is recommended. 
```
conda create -n pytorch-gpu python=3.9
conda activate pytorch-gpu
pip install -r requirements.txt
```

## **Usage**
Change the working directory to ```cd /path-to-dataset/```

### Augmentation
Training the Augmentation Network of RSTG.
```
python aug.py \
  --input_h5ad "path-to-dataset" \
  --gene_img_path "path-to-geneimage" \
  --cluster_path "path-to-cluster information" \
  --output_dir "path-to-Generation Folder" \
  --optimal_k value-of-k \
  --noise_type type-of-noise \
  --frac_anom frac-anom \
  --beta beta-value \
  --sigma sigma-value \
  --epochs num-steps \
  --batch_size batch-size \
  --nsample sample_size
  ```
  Example:
  ```
  python aug.py \
  --input_h5ad "C:/Users/Agrya/Documents/code/data/data_151673.h5ad" \
  --gene_img_path "C:/Users/Agrya/Documents/code/output/LIBD/full_geneimg.npy" \
  --cluster_path "C:/Users/Agrya/Documents/code/output/LIBD/cluster_k_20_673.npy" \
  --output_dir "C:/Users/Agrya/Documents/code/output/LIBD/Generation/" \
  --optimal_k 20 \
  --noise_type 'white' \
  --frac_anom 0.05 \
  --beta 0.03 \
  --sigma 0.5 \
  --epochs 50 \
  --batch_size 64 \
  --nsample 2
  ```
### Layer/Class Prediction
Train:
```
python train_prediction.py \
  --input_h5ad "Path to input .h5ad file" \
  --input_npy "Path to generated data .npy file" \
  --output_dir "path-to-save model Folder" \
  --beta beta-value \
  --nrep Repetition number \
  --epochs num-steps 
```

Test: 

```
python test_prediction.py
```

## **Acknowledgement**
The code base is built with [CeLEry](https://github.com/QihuangZhang/CeLEry).

Thanks for the great implementations!


