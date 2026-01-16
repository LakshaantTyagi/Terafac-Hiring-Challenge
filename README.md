# Terafac Hiring Challenge – CIFAR-10 Image Classification

Google Collab link : https://colab.research.google.com/drive/15uDWHWPYz0VeDs1rVrQ5I_33tom3wY8Q?usp=sharing

This repository contains my submission for the Terafac AI/Computer Vision/ML
Hiring Challenge using the CIFAR-10 dataset.

The project is structured into progressive levels, demonstrating increasing
model sophistication, analysis depth, and engineering judgment.

---

## Dataset
- CIFAR-10 (60,000 images, 10 classes)
- Image size: 32×32 RGB
- Train / Validation / Test split: 80 / 10 / 10

---

## Level-wise Overview

### Level 1 – Baseline Model
- Transfer learning using ResNet-18
- Cross-entropy loss, Adam optimizer
- Achieved ~89–90% test accuracy

### Level 2 – Intermediate Techniques
- Data augmentation (random crop, horizontal flip)
- Regularization and tuning
- Ablation study: with vs without augmentation
- Improved generalization performance

### Level 3 – Advanced Architecture
- SE-ResNet-18 with channel attention
- Grad-CAM for model interpretability
- Per-class performance analysis
- Achieved ~90–91% test accuracy

### Level 4 – Expert Techniques
- Ensemble learning (soft-voting) design
- Combination of ResNet-18 and SE-ResNet-18
- Methodology, formulation, and analysis provided
- Execution constrained due to checkpoint availability

---

## Setup Instructions

```bash
pip install -r requirements.txt
