# M²FNet: Multimodal Medical Fusion Network

[![TensorFlow 2.10+](https://img.shields.io/badge/TensorFlow-2.10+-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Official PyTorch/TensorFlow implementation of **M²FNet: A Multimodal Fusion Network for HAS-Negative Large Vessel Occlusion in Acute Ischemic Stroke**.

## 📖 Overview
The accurate diagnosis of large vessel occlusion (LVO) using Non-Contrast Computed Tomography (NCCT) is often impeded by the subtle nature of early radiological signs, particularly in 'HAS-negative' patients. **M²FNet** is a unified framework explicitly designed to bridge the intrinsic heterogeneity gap between high-dimensional radiological imaging and low-dimensional clinical informatics.

### Core Architectural Innovations:
1. **Domain-Adaptive Stem**: Aligns the grayscale medical image manifold with the semantic feature space of frozen pre-trained backbones (ResNet50).
2. **Relation-Aware Tabular Encoder**: A Transformer-based module that re-conceptualizes scalar clinical variables as a relational pseudo-sequence.
3. **Tri-Pathway Synergistic Fusion**: Dynamically orchestrates bidirectional cross-attention, gated modality weighting, and direct feature preservation.

[//]: # (## ⚙️ Installation)

[//]: # (```bash)

[//]: # (git clone [https://github.com/YourUsername/M2FNet.git]&#40;https://github.com/YourUsername/M2FNet.git&#41;)

[//]: # (cd M2FNet)

[//]: # (pip install -r requirements.txt)