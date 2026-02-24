# CIFAKE: Real and AI-Generated Synthetic Images — Technical Report

<div align="center">

![CIFAKE](https://img.shields.io/badge/Dataset-CIFAKE-c8320a?style=flat-square&logo=kaggle&logoColor=white)
![IEEE Access](https://img.shields.io/badge/Paper-IEEE%20Access%202024-1a4f8a?style=flat-square&logo=ieee&logoColor=white)
![Images](https://img.shields.io/badge/Total%20Images-120%2C000-2d7a4f?style=flat-square)
![Classes](https://img.shields.io/badge/Classes-10-orange?style=flat-square)
![Accuracy](https://img.shields.io/badge/CNN%20Accuracy-92.98%25-blue?style=flat-square)

**A technical deep-dive into the CIFAKE dataset — benchmarking binary classification of real photographs vs. AI-generated synthetic images using CNNs.**

[📄 Dataset on Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) &nbsp;·&nbsp; [📑 Original Paper (IEEE)](https://doi.org/10.1109/ACCESS.2024.3356122)

</div>

---

## Overview

CIFAKE is a dataset published in **IEEE Access (2024)** by Jordan J. Bird and Ahmad Lotfi. It contains **120,000 labelled images** — half real (sourced from CIFAR-10) and half synthetically generated using **Stable Diffusion v1.4** — designed as a benchmark for detecting AI-generated imagery.

As generative models improve, distinguishing real images from synthetic ones becomes increasingly difficult even for human observers. CIFAKE provides a standardised, balanced corpus to train and evaluate automated detectors.

---

## Dataset Details

| Property | Value |
|----------|-------|
| Total Images | 120,000 |
| Real Images | 60,000 (from CIFAR-10) |
| Fake Images | 60,000 (Stable Diffusion v1.4) |
| Image Resolution | 32 × 32 px (RGB) |
| Format | JPEG |
| Training Split | 100,000 (50K real + 50K fake) |
| Test Split | 20,000 (10K real + 10K fake) |
| Classes | 10 (balanced) |
| Task | Binary Classification (REAL / FAKE) |

---

## Classes

Both REAL and FAKE subsets are evenly distributed across the same 10 semantic categories inherited from CIFAR-10:

```
airplane · automobile · bird · cat · deer
dog · frog · horse · ship · truck
```

Each class contains **6,000 training** and **1,000 test** images per label (REAL/FAKE), ensuring no class imbalance.

---

## Data Generation Pipeline

**REAL images** — sourced directly from the CIFAR-10 dataset (Krizhevsky & Hinton, 2009), converted to JPEG.

**FAKE images** — generated using Stable Diffusion v1.4 conditioned on class-label text prompts (e.g., `"a photo of a horse"`), then resized to 32×32 to match CIFAR-10 dimensions.

```
Text Prompt (class label)
        ↓
Stable Diffusion v1.4
        ↓
Generated Image (high-res)
        ↓
Resize → 32×32 px
        ↓
FAKE subset (60,000 images)
```

---

## Model — CNN Architecture

The authors performed a **hyperparameter search across 36 CNN topologies**, varying filter counts, layer depth, and dropout rates. The best-performing architecture follows a standard Conv → Pool → Conv → Pool → Dense → Output structure:

```
Input         →  32×32×3 RGB tensor
Conv2D + ReLU →  32 filters, 3×3 kernel
MaxPooling    →  2×2
Conv2D + ReLU →  32 filters, 3×3 kernel
MaxPooling    →  2×2
Dense + ReLU  →  Fully connected layer
Output        →  2 neurons (Softmax) → REAL / FAKE
```

### Results

| Model | Accuracy |
|-------|----------|
| LeNet-5 | ~73% |
| Multi-Layer Perceptron | ~80% |
| VGG16 (fine-tuned) | ~88–90% |
| **Optimal CNN (Bird & Lotfi)** | **92.98%** |
| Human perception | < ~85% |

The optimised CNN **outperforms human-level performance** on this binary classification task.

---

## Explainability — Grad-CAM Analysis

Gradient-weighted Class Activation Mapping (Grad-CAM) was applied to understand *what* the model actually learns.

**Key finding:** The model focuses primarily on **background imperfections and texture artefacts** rather than the foreground object itself. Latent diffusion models like Stable Diffusion produce subtly irregular backgrounds that CNNs exploit as the main discriminative signal.

This has two important implications:
- **Adversarial vulnerability** — an attacker who smooths out background textures could potentially fool the classifier
- **Generalisation risk** — detectors trained at 32×32 may not transfer to high-resolution outputs from newer, cleaner generative models

---

## Quickstart

### Load with PyTorch

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Directory structure:
# /train/REAL/*.jpg   /train/FAKE/*.jpg
# /test/REAL/*.jpg    /test/FAKE/*.jpg

train_dataset = datasets.ImageFolder('./cifake/train', transform=transform)
test_dataset  = datasets.ImageFolder('./cifake/test',  transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

# Classes: ['FAKE', 'REAL']  (alphabetical → FAKE=0, REAL=1)
```

### Load with TensorFlow / Keras

```python
import tensorflow as tf

train_ds = tf.keras.utils.image_dataset_from_directory(
    './cifake/train',
    label_mode='binary',
    image_size=(32, 32),
    batch_size=64,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    './cifake/test',
    label_mode='binary',
    image_size=(32, 32),
    batch_size=64,
    shuffle=False,
)

# Normalise to [0, 1]
normalise = tf.keras.layers.Rescaling(1./255)
train_ds  = train_ds.map(lambda x, y: (normalise(x), y))
```

---

## Limitations

- **Low resolution (32×32)** — chosen to match CIFAR-10, but far below modern AI-generated image resolutions (e.g., SD-XL at 1024×1024). Models trained here may not generalise to high-res forgery detection without domain adaptation.
- **Single generator (SD v1.4)** — results may differ against newer models such as DALL-E 3, Midjourney, or Stable Diffusion XL.
- **Background bias** — Grad-CAM reveals the model exploits background artefacts, making it potentially fragile against post-processing attacks.

---

## Citation

If you use this dataset or reference this work, please cite:

```bibtex
@article{bird2024cifake,
  title   = {CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images},
  author  = {Bird, Jordan J. and Lotfi, Ahmad},
  journal = {IEEE Access},
  year    = {2024},
  doi     = {10.1109/ACCESS.2024.3356122}
}

@techreport{krizhevsky2009learning,
  title       = {Learning Multiple Layers of Features from Tiny Images},
  author      = {Krizhevsky, Alex and Hinton, Geoffrey},
  year        = {2009},
  institution = {University of Toronto}
}
```

---

## Author

**Rakesh Raut**
- 🔗 [LinkedIn](https://www.linkedin.com/in/rakeshkumarraut/)
- 🐙 [GitHub](https://github.com/Rex1671/)

*Technical report compiled from the CIFAKE dataset and IEEE Access paper for educational and portfolio purposes.*
