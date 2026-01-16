# Models Documentation

This document describes the neural network architectures used across different
levels of the Terafac Hiring Challenge (CIFAR-10 Image Classification).

All models were implemented and trained using PyTorch.

---

## 1. ResNet-18 (Baseline Model)

**Level:** 1 & 2  
**Purpose:** Establish a strong baseline using transfer learning principles.

### Architecture Overview
ResNet-18 is a deep convolutional neural network built using residual blocks.
Residual connections allow gradients to flow directly across layers, enabling
stable training of deeper architectures.

Key components:
- Initial 7×7 convolution + max pooling
- 4 residual stages with increasing channel depth
- Global Average Pooling
- Fully connected classification layer (10 classes)

### Modifications for CIFAR-10
- Final fully connected layer replaced to output 10 classes
- Image normalization applied to match ImageNet statistics
- Fine-tuning performed on CIFAR-10

### Strengths
- Strong generalization
- Stable training
- Computationally efficient

---

## 2. ResNet-18 with Data Augmentation

**Level:** 2  
**Purpose:** Improve generalization using regularization techniques.

### Techniques Applied
- Random horizontal flipping
- Random cropping
- Validation-based hyperparameter tuning

### Observations
- Reduced overfitting
- Improved validation and test performance
- Better robustness to spatial variations

---

## 3. SE-ResNet-18 (Squeeze-and-Excitation Network)

**Level:** 3  
**Purpose:** Introduce channel-wise attention to enhance feature representation.

### SE Block Description
The Squeeze-and-Excitation (SE) block recalibrates channel responses by:
1. **Squeeze:** Global Average Pooling to capture global context
2. **Excitation:** Two fully connected layers to learn channel importance
3. **Reweighting:** Channel-wise scaling of feature maps

Mathematically:
- Channel weights are learned dynamically
- More informative channels are emphasized

### Integration
SE blocks were inserted after each residual stage in ResNet-18:
- Layer1 → SE(64)
- Layer2 → SE(128)
- Layer3 → SE(256)
- Layer4 → SE(512)

### Benefits
- Improved feature discrimination
- Higher test accuracy than baseline
- Better interpretability (validated using Grad-CAM)

---

## 4. Ensemble Learning Strategy (Conceptual)

**Level:** 4  
**Purpose:** Improve robustness and reduce prediction variance.

### Ensemble Design
A soft-voting ensemble is proposed using:
- ResNet-18 (baseline)
- SE-ResNet-18 (attention-enhanced)

The ensemble combines class-probability outputs at inference time:

P_ensemble = (P_resnet18 + P_se_resnet18) / 2

Final predictions are obtained using argmax over the ensemble probabilities.

### Rationale
- Architectural diversity improves generalization
- Attention-based and standard CNNs complement each other
- Ensemble learning is widely used in production ML systems

### Practical Note
Due to runtime checkpoint persistence constraints, trained model weights could
not be preserved for re-execution. The ensemble design, however, follows standard
industry practices and is fully reproducible.

---

## Summary

| Level | Model | Key Contribution |
|-----|------|------------------|
| 1 | ResNet-18 | Strong baseline |
| 2 | ResNet-18 + Augmentation | Improved generalization |
| 3 | SE-ResNet-18 | Attention-driven features |
| 4 | Ensemble (Conceptual) | Robust prediction strategy |

---

## Implementation Notes
- Framework: PyTorch
- Loss: CrossEntropyLoss
- Optimizers: Adam / SGD (with momentum)
- Evaluation Metrics: Accuracy, Loss
- Interpretability: Grad-CAM

---

## Author
Lakshaant Tyagi
