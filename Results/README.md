# Results and Visualizations

This document contains accuracy metrics and qualitative visualizations for all
levels of the CIFAR-10 classification task.

---

## Level 1 – Baseline Results

Accuracy Curve

**Test Accuracy:** ~89%

---

## Level 2 – Augmentation & Ablation


**Observation:** Augmentation improves validation stability and generalization.

---

## Level 3 – SE-ResNet + Grad-CAM

Grad-CAM highlights semantically relevant regions, indicating that the attention
mechanism focuses on meaningful features.

---

## Level 4 – Ensemble Strategy


The ensemble combines predictions from ResNet-18 and SE-ResNet-18 using soft voting.

