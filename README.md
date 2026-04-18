# Autoencoder From Scratch (PyTorch + MNIST)

This repository contains a notebook-first implementation of a fully connected autoencoder built from scratch with PyTorch and trained on MNIST.

Project file:
- `coding Autoencoder from scratch.ipynb`

The notebook walks through the full pipeline:
1. Import libraries and configure device (CPU/GPU).
2. Load MNIST with torchvision and create a DataLoader.
3. Define an `Autoencoder` class (encoder + decoder).
4. Train using reconstruction loss (`MSELoss`) and Adam.
5. Visualize original vs reconstructed digits.
6. Extract latent vectors and inspect latent structure with plotting logic.
7. For latent dimensions greater than 3, use PCA projection to visualize in 2D.

## What Is Implemented

### Model Architecture

The autoencoder is a dense (MLP) architecture operating on flattened MNIST images:

- Input: `28 x 28 = 784`
- Encoder:
  - `Linear(784, hidden_dim)`
  - `ReLU`
  - `Linear(hidden_dim, latent_dim)`
  - `ReLU`
- Decoder:
  - `Linear(latent_dim, hidden_dim)`
  - `ReLU`
  - `Linear(hidden_dim, 784)`
  - `Sigmoid`

Default constructor values in the notebook:
- `latent_dim = 32`
- `hidden_dim = 256`

### Training Setup

- Dataset: MNIST training split
- Batch size: `128`
- Optimizer: `Adam(lr=1e-3)`
- Loss: `MSELoss`
- Epochs: `5`
- Device: CUDA if available, else CPU

### Visualizations

The notebook includes two key visual checks:

- **Reconstruction quality**: side-by-side original and decoded images.
- **Latent space analysis**:
  - `1D`: histogram
  - `2D`: scatter
  - `3D`: rotatable 3D scatter
  - `>3D`: PCA to 2D (this is used by default because latent dim is 32)

## Repository Structure

```
.
├── coding Autoencoder from scratch.ipynb
└── README.md
```

## Requirements

Python 3.9+ recommended.

Install dependencies:

```bash
pip install torch torchvision matplotlib scikit-learn numpy
```

Optional for a cleaner notebook experience:

```bash
pip install jupyter
```

## How To Run

### Option 1: VS Code Notebook (recommended for this repo)

1. Open `coding Autoencoder from scratch.ipynb` in VS Code.
2. Select a Python kernel with required packages installed.
3. Run cells from top to bottom.

### Option 2: Jupyter Lab/Notebook

```bash
jupyter notebook
```

Then open the notebook file and execute all cells sequentially.

## Expected Outputs

When the notebook is run successfully, you should see:

- Epoch-by-epoch training loss printed in the training cell.
- A figure showing original digits and their reconstructions.
- A latent-space plot. With default settings (`latent_dim=32`), this appears as a PCA-projected 2D scatter plot colored by digit labels.

## Notes On Design Choices

- **Why flatten images?**
  - Keeps implementation simple and focused on autoencoder fundamentals.
- **Why `Sigmoid` at decoder output?**
  - MNIST pixel values are in `[0, 1]` after `ToTensor()`, so sigmoid naturally matches this range.
- **Why MSE for reconstruction?**
  - Common and stable baseline for image reconstruction tasks.

## Common Issues And Fixes

### 1) Dataset download errors

- Ensure internet access on first run.
- Retry the notebook cell that creates `datasets.MNIST(..., download=True, ...)`.

### 2) Slow training on CPU

- Use fewer epochs while experimenting.
- Reduce batch size if memory is constrained.

### 3) Notebook kernel missing packages

- Confirm package installation in the same Python environment as the selected kernel.

## Suggested Experiments

Try changing these values in the notebook and compare reconstruction/latent plots:

- `latent_dim`: 2, 3, 8, 16, 32, 64
- `hidden_dim`: 128, 256, 512
- `epochs`: 5, 10, 20

You can also extend this baseline with:

- A convolutional autoencoder for better image quality.
- Denoising objective (corrupt input, reconstruct clean target).
- Variational autoencoder (VAE) for generative latent modeling.

## License

No license file is currently present in this repository.
If you plan to share or distribute this code, add a license file (for example, MIT) to clarify usage rights.