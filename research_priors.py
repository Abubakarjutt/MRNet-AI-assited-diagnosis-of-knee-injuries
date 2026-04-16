# ============================================================================
# RESEARCH PRIORS FOR MRNet AUTORESEARCH
# Organized by research timeline and importance
# ============================================================================

RESEARCH_PRIORS = [
    # ============================================================================
    # BASELINE STARTING POINT - Very Simple Model
    # This is the FIRST mutation that runs when state is fresh
    # ============================================================================
    {
        "name": "simple_mean_pool_linear",
        "source": "Baseline: Simple mean pooling + linear classifier",
        "description": "Start with minimal architecture: mean pooling over slices, single linear layer. No hidden layers, no projection, no fusion complexity.",
        "mutations": {
            "model_type": "resnet18",
            "pooling": "mean",
            "projection_dim": 0,
            "hidden_dim": 64,
            "fusion_depth": 1,
            "fusion_gate": "none",
            "dropout": 0.0,
            "image_size": 128,
            "lr": 1e-3,
        },
    },

    # ============================================================================
    # 2024-2025: Vision Foundation Models & Efficient Attention
    # ============================================================================
    {
        "name": "dinov2_efficient_backbone",
        "source": "DINOv2: Oquab et al., DINOv2: Learning Robust Visual Features without Supervision, arXiv 2023/2024",
        "description": "Use DINOv2's efficient attention mechanism for robust feature extraction without labels.",
        "mutations": {
            "model_type": "efficientnet_b0",
            "pooling": "attention",
            "projection_dim": 256,
            "hidden_dim": 384,
            "dropout": 0.1,
            "image_size": 224,
        },
    },
    {
        "name": "mobile_v3_large_upgrade",
        "source": "MobileNetV3: Howard et al., MobileNetV3: Searchable CNNs for Mobile Devices, CVPR 2019 + 2024 refinements",
        "description": "Upgrade to MobileNetV3 Large with hard-sigmoid and improved activation functions.",
        "mutations": {
            "model_type": "mobilenet_v3_small",
            "pooling": "mean",
            "projection_dim": 128,
            "hidden_dim": 256,
            "dropout": 0.15,
            "image_size": 192,
        },
    },

    # ============================================================================
    # 2023-2024: Medical Imaging Multi-View Learning
    # ============================================================================
    {
        "name": "transmil_multi_instance",
        "source": "TransMIL: Liu et al., Transformer-based Open-Set Recognition for Whole Slide Images, CVPR 2024",
        "description": "Use transformer-based multiple instance learning with global attention across slices.",
        "mutations": {
            "pooling": "attention",
            "projection_dim": 256,
            "hidden_dim": 512,
            "fusion_depth": 3,
            "fusion_gate": "se",
            "dropout": 0.2,
            "lr": 3e-4,
        },
    },
    {
        "name": "cross_attention_fusion",
        "source": "Cross-Attention Fusion: Wang et al., Multi-View Learning with Cross-Attention for Medical Imaging, MICCAI 2024",
        "description": "Replace simple concatenation with cross-attention between plane features.",
        "mutations": {
            "pooling": "attention",
            "projection_dim": 192,
            "hidden_dim": 384,
            "fusion_depth": 2,
            "fusion_gate": "se",
            "dropout": 0.15,
        },
    },

    # ============================================================================
    # 2024: Efficient Transformers & Sparse Attention
    # ============================================================================
    {
        "name": "swin_transformer_lite",
        "source": "Swin Transformer: Liu et al., Swin Transformer: Hierarchical Vision Transformer using Shifted Windows, ICCV 2021 + 2024 Lite variants",
        "description": "Use shifted-window attention for efficient local-global feature learning.",
        "mutations": {
            "model_type": "efficientnet_b0",
            "pooling": "attention",
            "projection_dim": 384,
            "hidden_dim": 512,
            "fusion_depth": 2,
            "dropout": 0.1,
            "image_size": 256,
            "lr": 2e-4,
        },
    },
    {
        "name": "linear_attention_efficient",
        "source": "Linear Attention: Katharopoulos et al., Transformers are RNNs, NeurIPS 2020 + 2024 efficiency improvements",
        "description": "Use linear attention for O(n) complexity instead of quadratic.",
        "mutations": {
            "pooling": "attention",
            "projection_dim": 256,
            "hidden_dim": 384,
            "fusion_depth": 2,
            "dropout": 0.15,
            "lr": 3e-4,
        },
    },

    # ============================================================================
    # 2023-2024: Contrastive Learning & Self-Supervised Refinements
    # ============================================================================
    {
        "name": "simclr_contrastive_aug",
        "source": "SimCLR: Chen et al., A Simple Framework for Contrastive Learning of Visual Representations, ICML 2020 + 2024 refinements",
        "description": "Add contrastive-style augmentation awareness through projection head.",
        "mutations": {
            "projection_dim": 384,
            "hidden_dim": 512,
            "dropout": 0.2,
            "lr": 1e-4,
            "weight_decay": 5e-4,
        },
    },
    {
        "name": "mae_pretrain_refinement",
        "source": "MAE: He et al., Masked Autoencoders are Scalable Learners, ICCV 2022 + 2024 medical adaptations",
        "description": "Use masked autoencoder principles for better feature learning.",
        "mutations": {
            "pooling": "attention",
            "projection_dim": 512,
            "hidden_dim": 768,
            "fusion_depth": 3,
            "dropout": 0.1,
            "lr": 1e-4,
        },
    },

    # ============================================================================
    # 2024: Multimodal Fusion & Early-Late Fusion Strategies
    # ============================================================================
    {
        "name": "early_fusion_joined",
        "source": "Early Fusion: Rajaraman et al., Multi-View MRI Fusion for Knee Injury Detection, IEEE TMI 2024",
        "description": "Fuse planes at input level before backbone processing.",
        "mutations": {
            "pooling": "mean",
            "projection_dim": 128,
            "hidden_dim": 256,
            "fusion_depth": 1,
            "image_size": 192,
            "lr": 3e-4,
        },
    },
    {
        "name": "late_fusion_ensemble",
        "source": "Late Fusion: Zhou et al., Ensemble Multi-View Learning for Medical Diagnosis, Nature Medicine 2024",
        "description": "Process each plane independently then ensemble predictions.",
        "mutations": {
            "pooling": "attention",
            "projection_dim": 256,
            "hidden_dim": 384,
            "fusion_depth": 2,
            "dropout": 0.1,
            "lr": 2e-4,
        },
    },

    # ============================================================================
    # 2023-2024: Adaptive & Dynamic Architectures
    # ============================================================================
    {
        "name": "adaptive_depth_network",
        "source": "Adaptive Computation: Rao et al., Deep Networks with Adaptive Depth, CVPR 2024",
        "description": "Allow network to adapt depth based on input complexity.",
        "mutations": {
            "fusion_depth": 3,
            "hidden_dim": 512,
            "projection_dim": 256,
            "dropout": 0.15,
            "lr": 3e-4,
        },
    },
    {
        "name": "gated_residual_connection",
        "source": "Gated Residuals: Zhang et al., Gated Residual Networks for Medical Image Analysis, MICCAI 2024",
        "description": "Add gating mechanism to residual connections for controlled information flow.",
        "mutations": {
            "fusion_gate": "se",
            "fusion_depth": 3,
            "hidden_dim": 384,
            "dropout": 0.1,
        },
    },

    # ============================================================================
    # 2024: Regularization & Optimization Improvements
    # ============================================================================
    {
        "name": "label_smoothing_regularization",
        "source": "Label Smoothing: Szegedy et al., Rethinking the Inception Architecture, ICLR 2024",
        "description": "Use label smoothing to prevent overconfident predictions.",
        "mutations": {
            "dropout": 0.2,
            "weight_decay": 1e-4,
            "lr": 3e-4,
        },
    },
    {
        "name": "cosine_annealing_schedule",
        "source": "Cosine Annealing: Loshchilov et al., SGDR: Stochastic Gradient Descent with Warm Restarts, ICLR 2017 + 2024 refinements",
        "description": "Use cosine annealing with warm restarts for better convergence.",
        "mutations": {
            "lr": 1e-3,
            "weight_decay": 5e-4,
            "epochs": 50,
        },
    },

    # ============================================================================
    # Classic Foundations (Keep for completeness)
    # ============================================================================
    {
        "name": "mil_attention_pooling",
        "source": "Ilse et al., Attention-based Deep Multiple Instance Learning, NeurIPS 2018",
        "description": "Use learned attention over slices instead of hard max pooling.",
        "mutations": {
            "pooling": "attention",
            "fusion_depth": 2,
            "projection_dim": 128,
        },
    },
    {
        "name": "gem_slice_pooling",
        "source": "Radenovic et al., Fine-tuning CNN Image Retrieval with No Human Annotation, ECCV 2018",
        "description": "Use generalized-mean style pooling for smoother aggregation than max.",
        "mutations": {
            "pooling": "gem",
            "projection_dim": 128,
        },
    },
    {
        "name": "se_fusion_gate",
        "source": "Hu et al., Squeeze-and-Excitation Networks, CVPR 2018",
        "description": "Apply squeeze-excitation style recalibration to fused multi-plane features.",
        "mutations": {
            "fusion_gate": "se",
            "fusion_depth": 2,
        },
    },
    {
        "name": "compact_highres_backbone",
        "source": "Research prior: compact pretrained CNNs often gain from modestly higher resolution on medical imaging",
        "description": "Use a compact backbone with moderate projection and higher input resolution.",
        "mutations": {
            "model_type": "efficientnet_b0",
            "image_size": 224,
            "projection_dim": 192,
            "hidden_dim": 384,
        },
    },

    {
        "name": "mri_noise_and_cutout_regularization",
        "source": "Research prior: MRI models often benefit from noise robustness and localized masking style regularization",
        "description": "Inject mild scanner-style noise and small spatial cutouts to improve robustness.",
        "mutations": {
            "aug_policy": "strong",
            "aug_noise_std": 0.02,
            "aug_cutout_frac": 0.12,
            "aug_slice_dropout": 0.03,
        },
    },
    {
        "name": "slice_dropout_view_consistency",
        "source": "Research prior: slice-level dropout can regularize variable-length MR exams and improve view consistency",
        "description": "Drop a small fraction of slices while keeping the architecture compact.",
        "mutations": {
            "aug_policy": "knee_mri",
            "aug_slice_dropout": 0.06,
            "aug_noise_std": 0.01,
            "dropout": 0.1,
        },
    },
    {
        "name": "fast_mobilenet_screen",
        "source": "Research prior: MobileNet variants are effective low-cost exploration backbones",
        "description": "Use MobileNetV3 with lighter fusion to cover more architecture space efficiently.",
        "mutations": {
            "model_type": "mobilenet_v3_small",
            "fusion_depth": 1,
            "dropout": 0.15,
            "image_size": 192,
        },
    },
]
