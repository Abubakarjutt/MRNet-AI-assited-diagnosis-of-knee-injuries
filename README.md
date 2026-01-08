# MRNet: AI-Assisted Diagnosis of Knee Injuries

This repository contains an implementation of the research paper [MRNet: Deep-learning-assisted diagnosis for knee magnetic resonance](https://stanfordmlgroup.github.io/projects/mrnet/).

## Project Overview

This project introduces an advanced deep learning model for diagnosing knee injuries from Magnetic Resonance Imaging (MRI) scans. It builds upon the foundational research of the [original MRNet paper](https://stanfordmlgroup.github.io/projects/mrnet/) but introduces a key architectural enhancement: a **Vision Transformer (ViT)**.

Instead of training separate models for each MRI plane (sagittal, coronal, and axial), this implementation uses a **single, end-to-end model** that processes all three planes simultaneously. This ViT-based approach not only streamlines the training process but also achieves high diagnostic accuracy for multiple conditions, including abnormalities, ACL tears, and meniscus tears. This updated architecture represents a significant step forward, reducing training time by approximately 5x while maintaining robust performance.

## Model Architecture

The model uses a Vision Transformer (ViT) pretrained on the ImageNet dataset (*ViT-B/16*). The architecture is designed to handle the multi-plane nature of MRI data effectively:

1.  **Input Processing**: The model takes three MRI series as input: sagittal, coronal, and axial. Each series consists of multiple slices.
2.  **Feature Extraction**: The pretrained ViT backbone extracts features from every slice of all three planes.
3.  **Aggregation**: A max-pooling layer aggregates the slice-level features across all planes to create a single feature vector that represents the entire MRI exam.
4.  **Classification**: A final linear layer with dropout classifies the aggregated features to predict the probabilities for the three diagnostic labels (abnormal, ACL tear, meniscus tear).

While the primary model is ViT-based, the repository also includes implementations for `AlexNet` and `ResNet` as alternative architectures.

## Dataset: MRNet

The data comes from the Stanford ML Group research lab. It consists of 1,370 knee MRI exams performed at Stanford University Medical Center to study the presence of Anterior Cruciate Ligament (ACL) tears and Meniscus tears.

*   **Input**: Multi-view MRI scans (Sagittal, Coronal, Axial).
*   **Task**: Multi-label, multi-class classification.
*   **Labels**: Abnormal, ACL tear, Meniscus tear.

You can request the dataset from [this link](https://stanfordmlgroup.github.io/competitions/mrnet/).

## Setup

### Prerequisites

*   **Python**: 3.11+
*   **PyTorch**: Ensure you have a version of PyTorch compatible with your hardware (e.g., CUDA for NVIDIA GPUs, MPS for Apple Silicon).

### Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/your-repo/mrnet.git
    cd mrnet
    ```

2.  **Install Dependencies**:
    It is recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

### Directory Structure

Ensure your project is structured as follows:

```
.
├── MRNet-v1.0/              # Dataset directory
│   ├── train/
│   ├── valid/
│   └── ...
├── models/                  # Directory for saved model checkpoints
├── logs/                    # Directory for Tensorboard logs
├── train.py                 # Main training script
├── dataloader.py            # Data loading logic
├── vit.py                   # Vision Transformer model
├── resnet.py                # ResNet model (alternative)
├── alexnet.py               # AlexNet model (alternative)
├── utils.py                 # Helper functions
├── requirements.txt         # Project dependencies
└── README.md
```

## Usage

To train the model, use the `train.py` script.

### Basic Usage

A typical training command looks like this:

```bash
python train.py --prefix_name "vit_experiment_1" --epochs 50 --lr 1e-5
```

### Command-Line Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--prefix_name` | **Required**. A unique name for the experiment, used for saving models and logs. | N/A |
| `--task` | The diagnostic task. Since the model is multi-label, this is primarily for organizing logs. | `acl` |
| `--epochs` | The total number of training epochs. | `50` |
| `--lr` | The initial learning rate for the Adam optimizer. | `1e-5` |
| `--patience` | Number of epochs with no improvement in validation loss before early stopping. | `20` |
| `--save_model` | Set to `1` to save the best model based on validation AUC. Set to `0` to disable. | `1` |
| `--flush_history` | Set to `1` to delete previous logs for the same task before starting a new run. | `0` |
| `--log_every` | The frequency (in batches) of logging training progress. | `100` |

### Example

Train a model for 100 epochs with a specific learning rate and a custom prefix:

```bash
python train.py --prefix_name "MRNet_ViT_Run1" --epochs 100 --lr 0.0001
```

## Results

The model's performance is evaluated using the Area Under the Receiver Operating Characteristic Curve (AUC), a standard metric for binary and multi-label classification tasks. The ViT-based model demonstrates strong performance, comparable to the original MRNet paper, with the advantage of an end-to-end training pipeline.

*   **Validation AUC**: The model typically achieves a validation AUC of approximately **0.915**.
*   **Training AUC**: The training AUC reaches around **0.99**, indicating that the model fits the training data well.

These results confirm the effectiveness of the Vision Transformer architecture for this diagnostic task.

## Contributions

Contributions are welcome! If you feel that some functionalities or improvements could be added to the project, don't hesitate to submit a pull request.
