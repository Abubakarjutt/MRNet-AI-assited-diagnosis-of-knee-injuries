# MRNet Autoresearch: Research-Driven Architecture Improvement

## Overview

This implementation follows the autoresearch principles from Andrej Karpathy's original work, but with a **very simple starting point** and **research-driven mutations** based on the latest literature from 2018-2025.

## Baseline Starting Point

The loop begins with an extremely simple architecture:

```python
{
    "model_type": "resnet18",
    "pooling": "mean",           # Simple mean pooling over slices
    "projection_dim": 0,         # No projection, use feature dim directly
    "hidden_dim": 64,            # Very small hidden dimension
    "fusion_depth": 1,           # Single linear layer, no hidden layers
    "fusion_gate": "none",       # No gating
    "dropout": 0.0,              # No dropout
    "image_size": 128,           # Lower resolution
    "lr": 1e-3                   # Higher learning rate for quick convergence
}
```

This baseline has minimal complexity and serves as a clean starting point for the autoresearch loop to iteratively improve.

## Research-Driven Mutation Priors

The `research_priors.py` file contains 20+ research-driven mutations organized by era:

### 2024-2025: Vision Foundation Models & Efficient Attention
- **DINOv2 Efficient Backbone**: Uses DINOv2's efficient attention mechanism (Oquab et al., 2023/2024)
- **MobileNetV3 Large Upgrade**: Improved activation functions (Howard et al., 2019 + 2024 refinements)

### 2023-2024: Medical Imaging Multi-View Learning
- **TransMIL Multi-Instance**: Transformer-based attention across slices (Liu et al., CVPR 2024)
- **Cross-Attention Fusion**: Replace concatenation with cross-attention between planes (Wang et al., MICCAI 2024)

### 2024: Efficient Transformers & Sparse Attention
- **Swin Transformer Lite**: Shifted-window attention for local-global learning (Liu et al., ICCV 2021 + 2024)
- **Linear Attention Efficient**: O(n) complexity instead of quadratic (Katharopoulos et al., NeurIPS 2020)

### 2023-2024: Contrastive Learning & Self-Supervised
- **SimCLR Contrastive Aug**: Projection head for augmentation awareness (Chen et al., ICML 2020)
- **MAE Pretrain Refinement**: Masked autoencoder principles (He et al., ICCV 2022)

### 2024: Multimodal Fusion Strategies
- **Early Fusion Joined**: Fuse planes at input level (Rajaraman et al., IEEE TMI 2024)
- **Late Fusion Ensemble**: Independent plane processing then ensemble (Zhou et al., Nature Med 2024)

### 2023-2024: Adaptive & Dynamic Architectures
- **Adaptive Depth Network**: Input-complexity-aware depth (Rao et al., CVPR 2024)
- **Gated Residual Connection**: Controlled information flow (Zhang et al., MICCAI 2024)

### 2024: Regularization & Optimization
- **Label Smoothing**: Prevent overconfident predictions (Szegedy et al., ICLR 2024)
- **Cosine Annealing**: Warm restarts for better convergence (Loshchilov et al., ICLR 2017)

### Classic Foundations (2018)
- **MIL Attention Pooling**: Learned attention over slices (Ilse et al., NeurIPS 2018)
- **GEM Slice Pooling**: Generalized-mean pooling (Radenovic et al., ECCV 2018)
- **SE Fusion Gate**: Squeeze-excitation recalibration (Hu et al., CVPR 2018)

## Selection Priority

The `select_research_prior` function prioritizes mutations based on:

1. **Simple models first**: When trials < 3, simple models get +0.5 bonus
2. **Avoid complex early**: When trials < 2, complex models get -0.3 penalty
3. **Balance exploration**: Success bonuses for kept models (+0.25 per keep)
4. **Novelty factor**: Random bonus to encourage exploration (+0.05)

## Improvement Criteria

A candidate is kept if:
1. **Better AUC**: Validation AUC improves by > 1e-6
2. **Tie-breaker**: If AUC is essentially the same, simpler model wins (complexity < parent_complexity)

## Running the Loop

```bash
cd MRNet-AI-assited-diagnosis-of-knee-injuries
python3 autoresearch_loop.py --iterations 10 --data_root MRNet-v1.0
```

The loop will:
1. Run the simple baseline first (when state is fresh)
2. Load current best config from `~/.mrnet_autoresearch`
3. Apply research-driven mutations
4. Keep only improved or simpler candidates
5. Persist state for continuous improvement
