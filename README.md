# 🖼️ CIFAR-10 Image Classification — From Scratch to Fine-Tuned

A systematic deep learning project that benchmarks multiple model architectures on the **CIFAR-10** dataset — from a plain MLP all the way to fine-tuned ResNet and MobileNet — with a focus on understanding *why* each model performs the way it does.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Experiments & Architecture Journey](#experiments--architecture-journey)
- [Model Comparison](#model-comparison)
- [Key Findings](#key-findings)
- [Techniques Used](#techniques-used)
- [Installation & Usage](#installation--usage)
- [Requirements](#requirements)

---

## Overview

This project explores image classification on CIFAR-10 through a series of progressively more powerful architectures. Each experiment builds on the lessons of the previous one — tracking not just accuracy, but also model efficiency, generalization, and robustness to simple image shifts.

**Goal:** Understand the tradeoffs between model complexity, parameter count, accuracy, and spatial invariance.

---

## Dataset

**CIFAR-10** — 60,000 color images (32×32 pixels) across 10 classes.

| Split | Samples |
|-------|---------|
| Train | 40,000  |
| Validation | 10,000 |
| Test  | 10,000  |

**Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

**Preprocessing:**
- Normalized per channel: Mean `(0.4914, 0.4822, 0.4465)`, Std `(0.2470, 0.2435, 0.2616)`
- Training augmentations: Random crop (32×32, padding=4), Random horizontal flip

---

## Project Structure

```
├── notebooks/
│   ├── 01_data_eda.ipynb           # Exploratory data analysis
│   ├── 02_neural_baseline.ipynb    # MLP baseline
│   ├── 03_cnn_baseline.ipynb       # Standard CNN
│   ├── 04_finetune_resnet_arch_.ipynb  # ResNet architecture experiments (failed)
│   ├── 05_finetune_resnet.ipynb    # ResNet fine-tuning (Kaggle GPU)
│   ├── 06_efficient_learning.ipynb # Depthwise Separable CNN
│   └── 07_mobilenet_train.ipynb    # MobileNetV2 from scratch (Kaggle GPU)
│
├── src/
│   ├── models/
│   │   ├── neural.py               # MLP, CNN, DS-CNN architectures
│   │   └── vision_finetuned.py     # ResNet fine-tuning variants
│   ├── data/
│   │   └── utils.py                # Data loading & dataset classes
│   └── eval/
│       └── metrics.py              # Evaluation utilities
│
├── configs/
│   └── base_config.yaml            # Hyperparameters
│
├── artifacts/                      # Saved model weights (.pth)
└── data/
    └── raw/                        # CIFAR-10 raw data
```

---

## Experiments & Architecture Journey

### Notebook 01 — Exploratory Data Analysis
- Visualized sample images per class
- Plotted per-class mean images to study visual patterns
- Computed pixel distribution histograms per RGB channel
- Confirmed channel statistics for normalization

---

### Notebook 02 — MLP Baseline (`MLPv01`)

A fully connected network — the simplest possible approach.

```
Input (32×32×3 = 3072)
  → Linear(3072, 512) + BN + LeakyReLU + Dropout(0.2)
  → Linear(512, 512) + BN + LeakyReLU + Dropout(0.2)
  → Linear(512, 512) + BN + LeakyReLU + Dropout(0.2)
  → Linear(512, 10)
```

**Key experiment:** Tested robustness by shifting images 4 pixels horizontally. The MLP performed noticeably worse on shifted images — demonstrating its lack of spatial invariance.

---

### Notebook 03 — CNN Baseline (`CNNv01`)

A standard convolutional network that preserves spatial structure.

```
Conv2d(3→32, k=3) + AvgPool  →  16×16
Conv2d(32→64, k=3) + AvgPool →  8×8
Conv2d(64→128, k=3) + AvgPool → 4×4
Flatten
Linear(2048, 128) + BN + LeakyReLU + Dropout(0.2)
Linear(128, 128)  + BN + LeakyReLU + Dropout(0.2)
Linear(128, 10)
```

Trained two variants: 10 epochs and 30 epochs. CNN outperformed the MLP — especially on shifted images — due to translation-invariant feature extraction.

---

### Notebook 04 — ResNet Fine-Tuning Architecture Experiments *(Failed)*

Explored three ResNet-18 fine-tuning strategies:

| Variant | Change from Base ResNet-18 |
|---------|---------------------------|
| `ResNetFTv01` | Replaced `conv1` with 3×3/stride-1, removed MaxPool, froze backbone except `conv1` |
| `ResNetFTv02` | Kept original `conv1`, froze entire backbone |
| `ResNetFTv03` | Replaced `conv1` with 3×3/stride-1, removed MaxPool, froze backbone |

All three variants append a custom head:
```
Linear(512, 128) + BN + LeakyReLU
Linear(128, 128) + BN + LeakyReLU
Linear(128, 10)
```

> ⚠️ **Failed experiment**: The architecture exploration revealed issues with naive fine-tuning — leading to the more structured approach in Notebook 05.

---

### Notebook 05 — ResNet Fine-Tuning with Layer-wise LR (`ResNetFTv03`)
> Trained on Kaggle with GPU

**Key insight:** Different layers should learn at different speeds. Later layers need faster updates (they're more task-specific), while earlier layers benefit from smaller updates (they already have good low-level features).

**Layer-wise learning rates:**

| Layer | Learning Rate |
|-------|--------------|
| `conv1` | 1e-3 |
| `layer1` | 5e-4 |
| `layer2` | 2e-4 |
| `layer3` | 1e-4 |
| `layer4` | 5e-5 |
| Head | 1e-3 |

**Two-phase training:**
- **Phase 1 (5 epochs):** Freeze layers 1 & 2, train only layers 3, 4, and head
- **Phase 2 (continued):** Gradually unfreeze and fine-tune all layers

**Weight initialization:** Kaiming uniform on custom conv1 and head linear layers.

---

### Notebook 06 — Efficient Learning with Depthwise Separable CNN (`DSCNNv01`)

Same architecture as `CNNv01`, but standard convolutions are replaced with **Depthwise Separable Convolutions** — dramatically cutting parameter count.

**How it works:**

```
Standard Conv:   K × K × Cin × Cout  params
DS Conv:         (K × K × Cin) + (Cin × Cout)  params
```

**Parameter savings example (128→256 channels):**

| Type | Parameters |
|------|-----------|
| Standard Conv 3×3 | 294,912 |
| Depthwise Separable | 33,920 |
| **Reduction** | **~88%** |

Each DS block:
```
DepthwiseConv(Cin→Cin, groups=Cin)  ← "where" to look
PointwiseConv(Cin→Cout, 1×1)        ← "how" to mix channels
```

---

### Notebook 07 — MobileNetV2 from Scratch (`MobileNetv01`)
> Trained on Kaggle with GPU

Used `mobilenet_v2(weights=None)` with a custom classification head:

```
MobileNetV2 backbone (from scratch)
  → Dropout(0.2)
  → Linear(1280, 128) + BN + LeakyReLU
  → Linear(128, 128)  + BN + LeakyReLU
  → Linear(128, 10)
```

**Training setup:**
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR (T_max=100)
- Epochs: 100
- Best model saved via checkpoint during training

---

## Model Comparison

| Model | Architecture | Params (approx) | Training Epochs | Notes |
|-------|-------------|-----------------|-----------------|-------|
| `MLPv01` | Fully Connected | ~1.6M | 30 | No spatial awareness |
| `CNNv01` | Standard CNN | ~300K | 30 | Good baseline |
| `DSCNNv01` | Depthwise Separable CNN | ~40K | 30 | ~88% fewer params vs CNNv01 |
| `ResNetFTv03` | ResNet-18 Fine-tuned | ~11M total / head only trainable | 15 (2-phase) | Layer-wise LR, pretrained |
| `MobileNetv01` | MobileNetV2 from scratch | ~3.4M | 100 | Cosine LR, no pretraining |

### Robustness to Image Shift (4px horizontal shift)

| Model | Normal Accuracy | Shifted Accuracy | Drop |
|-------|----------------|-----------------|------|
| MLPv01 | Moderate | Lower | High ↓ |
| CNNv01 | Better | Moderate | Smaller ↓ |
| DSCNNv01 | Similar to CNN | Similar | Smaller ↓ |
| ResNetFTv03 | High | High | Minimal ↓ |

> CNNs are more robust to spatial shifts than MLPs due to local receptive fields and pooling.

---

## Key Findings

**1. MLPs are spatially blind**
Flattening images loses all positional information. A 4-pixel shift caused a measurable accuracy drop — something CNNs handle much better.

**2. Depthwise Separable Convolutions are a free lunch**
`DSCNNv01` achieves similar accuracy to `CNNv01` with ~88% fewer parameters. A practical choice whenever memory or speed is a constraint.

**3. Naive fine-tuning doesn't always work**
Notebook 04 (the failed experiments) showed that simply swapping the classifier head on a frozen ResNet-18 is not enough for CIFAR-10's 32×32 images. The original ResNet-18 was designed for 224×224 images — the early stride-2 conv and MaxPool aggressively downsample small images, losing too much spatial information too early.

**4. Architecture adaptation matters for fine-tuning**
Replacing `conv1` (7×7, stride 2) with a smaller (3×3, stride 1) kernel and removing MaxPool preserves more spatial detail for small images — a crucial modification for CIFAR-10.

**5. Layer-wise learning rates improve fine-tuning**
Applying higher learning rates to task-specific later layers and lower rates to general early features leads to better convergence than a uniform learning rate.

---

## Techniques Used

- **Optimizer:** AdamW with weight decay
- **Loss:** Cross-Entropy
- **Augmentation:** Random crop, random horizontal flip
- **Normalization:** Per-channel mean/std normalization
- **Regularization:** Dropout, BatchNorm, weight decay
- **Weight Init:** Kaiming Uniform for Linear and Conv layers
- **LR Scheduling:** CosineAnnealingLR (MobileNet), ReduceLROnPlateau / manual (ResNet)
- **Reproducibility:** Fixed seeds, `cudnn.deterministic = True`
- **Evaluation:** Confusion matrix, classification report (precision, recall, F1), shift robustness test

---

## Installation & Usage

```bash
# Clone the repo
git clone https://github.com/your-username/cifar10-classification.git
cd cifar10-classification

# Install dependencies
pip install -r requirements.txt

# Download CIFAR-10 data and place in:
# data/raw/cifar-10-python.tar/cifar-10-python/cifar-10-batches-py/

# Run notebooks in order
jupyter notebook notebooks/01_data_eda.ipynb
```

> Notebooks 05 and 07 were trained on **Kaggle** with a GPU. To run locally, change the `PATH` variable to your local data path and set `GPU = 'cuda'` or `'cpu'` accordingly.

---

## Requirements

```
torch
torchvision
polars
numpy
matplotlib
scikit-learn
scipy
opencv-python
tqdm
pyyaml
pandas
jupyter
```

Install all at once:
```bash
pip install torch torchvision polars numpy matplotlib scikit-learn scipy opencv-python tqdm pyyaml pandas jupyter
```

---

## Acknowledgements

- Dataset: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) by Alex Krizhevsky
- Pretrained weights via `torchvision.models`
- GPU training via [Kaggle Notebooks](https://www.kaggle.com/)