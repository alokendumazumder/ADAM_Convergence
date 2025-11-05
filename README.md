# Training Experiments – README

This repo contains data loaders, model architectures, learning‑rate schedulers, and experiment utilities to run **full‑batch**, **mini‑batch** training on common vision datasets (MNIST, CIFAR‑10/100, ImageNet). It also produces basic plots for **gradient norm**, **validation loss**, and **validation accuracy**.

## Repository layout (key files)

- **get_data.py**  
  Loader utilities for **CIFAR‑10**, **CIFAR‑100**, **MNIST**, and **ImageNet**.  
  Typical responsibilities:
  - Apply standard train/validation transforms.
  - Build training/validation/test splits.
  - Return dataset/dataloader objects configurable by batch size, workers, etc.

- **schedulers.py**  
  Collection of learning‑rate schedulers.  
  Includes:
  - Wrappers around **PyTorch** schedulers.
  - **Custom schedulers** implemented from the literature (exact list in the file).

- **models.py**  
  Model architectures used in experiments:
  - Simple **Linear** classifier
  - **LeNet**
  - **VGG‑9**
  - **ResNet‑18**
  - **MobileNet‑V2**

- **utils.py**  
  Experiment helper classes and shared utilities.  
  Provides class implementations for:
  - **Full‑batch** experiments
  - **Mini‑batch** experiments
  - **Step-size** experiments  
  These classes typically encapsulate the training/eval loops and common bookkeeping (e.g., logging/metrics).

- **experiments.py**  
  High‑level functions that **instantiate** the experiment classes from `utils.py`, **run** training/evaluation, and **generate plots** for:
  - Gradient norm
  - Validation loss
  - Validation accuracy

## Quick start

1. **Run the experiment:**
   ```bash
   python main.py
   ```
   - Edit arguments or config inside `main.py` (or use CLI flags).

## Outputs

- Training/evaluation logs
- Plots for **gradient norm**, **validation loss**, and **validation accuracy** (saved to a `results/`).

## Notes

- The exact function/class names and options live in the respective files; check their docstrings/comments for the authoritative interface.
- ImageNet typically requires you to point to the dataset directory on disk; set this path where indicated in `get_data.py`.
