RESEARCH_PRIORS = [
    {
        "name": "mil_attention_pooling",
        "source": "Ilse et al., Attention-based Deep Multiple Instance Learning, 2018",
        "description": "Use learned attention over slices instead of hard max pooling.",
        "mutations": {
            "pooling": "attention",
            "fusion_depth": 2,
            "projection_dim": 128,
        },
    },
    {
        "name": "gem_slice_pooling",
        "source": "Radenovic et al., Fine-tuning CNN Image Retrieval with No Human Annotation, 2018",
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
