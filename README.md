# Autoencoder from Scratch — PyTorch / MNIST

A self-contained reference implementation of a fully connected autoencoder trained on MNIST, designed for reproducibility and structured experimentation. The codebase is intentionally minimal to serve as a clean baseline for extensions such as denoising autoencoders, variational autoencoders (VAEs), and convolutional variants.

---

## Overview

Autoencoders learn a compressed latent representation of input data by optimizing a reconstruction objective. This implementation covers the core unsupervised representation learning pipeline: encoding high-dimensional observations into a bottleneck embedding and decoding them back to pixel space using only dense (MLP) layers.

All components — model definition, training loop, evaluation, and latent-space diagnostics — are contained in a single annotated notebook.

---

## Architecture

The model is a symmetric encoder–decoder with ReLU nonlinearities in the encoder and a Sigmoid output in the decoder to match the `[0, 1]` pixel range produced by `torchvision.transforms.ToTensor`.

```
Input (784)
    │
    ▼
Linear(784 → hidden_dim) → ReLU
    │
    ▼
Linear(hidden_dim → latent_dim) → ReLU     ← latent code z
    │
    ▼
Linear(latent_dim → hidden_dim) → ReLU
    │
    ▼
Linear(hidden_dim → 784) → Sigmoid

Output (784 → 28×28)
```

| Hyperparameter | Default |
|---|---|
| `latent_dim` | 32 |
| `hidden_dim` | 256 |
| Input dimension | 784 (28 × 28, flattened) |
| Output activation | Sigmoid |

---

## Training Configuration

| Setting | Value |
|---|---|
| Dataset | MNIST (training split, 60 000 samples) |
| Batch size | 128 |
| Optimizer | Adam |
| Learning rate | 1 × 10⁻³ |
| Loss function | MSELoss (pixel-wise) |
| Epochs | 5 |
| Device | CUDA if available, else CPU |

---

## Repository Structure

```
.
├── coding Autoencoder from scratch.ipynb   # Full pipeline: train → evaluate → visualize
└── README.md
```

---

## Requirements

Python ≥ 3.9.

```bash
pip install torch torchvision matplotlib scikit-learn numpy
```

Optional:

```bash
pip install jupyter   # for Jupyter Lab / Notebook interface
```

---

## Reproducing Results

**VS Code (recommended)**

1. Open `coding Autoencoder from scratch.ipynb` in VS Code.
2. Select a Python kernel with the required packages.
3. Run all cells top-to-bottom (`Ctrl+Shift+P` → *Run All Cells*).

**Jupyter**

```bash
jupyter notebook "coding Autoencoder from scratch.ipynb"
```

Execute cells sequentially. First run downloads MNIST (~11 MB); internet access is required.

---

## Outputs

A complete run produces:

- **Training log** — per-epoch reconstruction loss printed to stdout.
- **Reconstruction figure** — side-by-side grid of original and decoded MNIST digits.
- **Latent-space diagnostic** — visualization modality selected automatically by dimensionality:

| `latent_dim` | Visualization |
|---|---|
| 1 | Histogram of scalar codes |
| 2 | 2D scatter, colored by class label |
| 3 | Rotatable 3D scatter |
| > 3 (default: 32) | PCA projection to 2D, colored by class label |

---

## Experimental Variables

The following hyperparameters are exposed in the notebook for systematic ablation:

| Variable | Values to explore |
|---|---|
| `latent_dim` | 2, 3, 8, 16, 32, 64 |
| `hidden_dim` | 128, 256, 512 |
| `epochs` | 5, 10, 20 |

Reducing `latent_dim` (e.g., to 2 or 3) allows direct geometric inspection of the learned manifold without PCA projection.

---

## Design Decisions

**Flattened input.** Convolutions are omitted deliberately to isolate autoencoder fundamentals from spatial inductive biases. This makes the reconstruction objective and the latent structure easier to reason about.

**Sigmoid decoder output.** MNIST pixels lie in `[0, 1]` after `ToTensor`. Sigmoid is the natural output nonlinearity for MSE in this range and avoids unbounded reconstructions.

**MSELoss.** Pixel-wise mean squared error is a stable, interpretable reconstruction loss and a standard baseline prior to adopting binary cross-entropy or perceptual losses.

---

## Potential Extensions

- **Convolutional autoencoder** — replace MLP blocks with `Conv2d` / `ConvTranspose2d` layers for spatially aware feature extraction.
- **Denoising autoencoder (DAE)** — corrupt inputs with Gaussian or dropout noise; train to reconstruct the clean target.
- **Variational autoencoder (VAE)** — replace the deterministic bottleneck with a learned Gaussian posterior; add KL divergence to the loss for generative modeling.
- **Disentangled representations** — apply β-VAE or FactorVAE regularization to encourage axis-aligned latent factors.

---

## Known Issues

| Symptom | Resolution |
|---|---|
| MNIST download fails | Ensure internet access on first run; retry the dataset cell |
| Slow training on CPU | Reduce `epochs` during experimentation; reduce `batch_size` if memory-constrained |
| Missing packages in kernel | Confirm installation in the same Python environment selected as the notebook kernel |

---

## License

No license file is currently present. If you intend to distribute or build upon this work, add an open-source license (MIT recommended) to clarify usage rights.
