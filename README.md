
# 🌊 SST Autoencoder – Self-Supervised Learning for Sea Surface Temperature Patterns

This project applies a **self-supervised deep learning** pipeline to daily NOAA OISST (Optimum Interpolation Sea Surface Temperature) data. Using a convolutional autoencoder, it extracts latent representations of SST patterns, clusters them, and visualizes their spatial distributions.

---

## 🧠 Overview

- Download and explore NOAA SST data
- Extract 64×64 spatial patches
- Train a convolutional autoencoder to compress and reconstruct SST fields
- Reduce embeddings to 2D with UMAP
- Cluster patch embeddings to identify recurring SST patterns
- Map cluster occurrences geographically

---

## 📂 Project Structure

```

sst-autoencoder/
├── data/               # SST netCDF + extracted patches (not versioned)
├── notebooks/          # Ordered Jupyter notebooks (from EDA to analysis)
├── outputs/            # Trained model and embeddings
├── results/figures/    # Final visual outputs
├── src/                # Autoencoder + training + data utilities
├── requirements.txt
├── .gitignore
└── README.md

````

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/sst-autoencoder.git
cd sst-autoencoder
````

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install requirements

```bash
pip install -r requirements.txt
```

### 4. Download SST data

Use the helper script:

```bash
python src/download_data.py
```

This downloads `sst_2024.nc` from NOAA and saves it under `data/`.

---

## 📓 Notebooks

| Notebook                                    | Description                                |
| ------------------------------------------- | ------------------------------------------ |
| `01_explore_sst.ipynb`                      | Load and visualize global SST data         |
| `02_extract_patches.ipynb`                  | Extract 64×64 patches from SST fields      |
| `03_visualize_reconstructions.ipynb`        | Compare original and reconstructed patches |
| `04_visualize_embeddings.ipynb`             | UMAP projection of latent space            |
| `05_cluster_and_interpret_embeddings.ipynb` | Cluster the UMAP embeddings                |
| `06_random_cluster_patch_visualizer.ipynb`  | Visualize sample patches per cluster       |
| `07_map_clusters_globally.ipynb`            | Global map of cluster distributions        |
| `08_cluster_stats.ipynb`                    | Statistical analysis of cluster features   |

---


## 🛠 Model

The autoencoder is a 3-layer convolutional encoder-decoder architecture implemented in `src/autoencoder.py`. It compresses SST patches to latent vectors and reconstructs them during training.

---

## 📜 License

This project is open-source and licensed under the MIT License.

---


## 🧠 Acknowledgments

* NOAA Physical Sciences Laboratory – OISST v2.1
  [https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html](https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html)

---

## 🙋‍♂️ Author

*Created and maintained by Michael Giannopoulos — climate scientist, geologist, and data analyst.*
\:sweden: Working with real-world oceanographic datasets
\:computer: Python | PyTorch | Jupyter | UMAP | Clustering



