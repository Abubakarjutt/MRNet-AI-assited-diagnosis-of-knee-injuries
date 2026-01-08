# MRNet: AI-Assisted Diagnosis of Knee Injuries

This repository contains an implementation of the research paper [MRNet: Deep-learning-assisted diagnosis for knee magnetic resonance](https://stanfordmlgroup.github.io/projects/mrnet/).

## Project Overview

The original paper trains three separate models (one for each plane: sagittal, coronal, axial) for each class (abnormal, ACL, meniscus) and then predicts the class using the maximum value among the predictions of all three models.

**This implementation differs by training an end-to-end single model** that takes all three planes as input and predicts class probabilities for all three classes simultaneously. This approach significantly reduces training time (approx. 5x) while maintaining high performance.

## Dataset: MRNet

The data comes from the Stanford ML Group research lab. It consists of 1,370 knee MRI exams performed at Stanford University Medical Center to study the presence of Anterior Cruciate Ligament (ACL) tears and Meniscus tears.

*   **Input**: Multi-view MRI scans (Sagittal, Coronal, Axial).
*   **Task**: Multi-label, multi-class classification.
*   **Labels**: Abnormal, ACL tear, Meniscus tear.

You can request the dataset from [this link](https://stanfordmlgroup.github.io/competitions/mrnet/).

## Setup

### Prerequisites

*   Python 3.11+
*   PyTorch
*   Torchvision
*   TensorboardX
*   Scikit-learn
*   Pandas
*   Geomloss

### Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install torch torchvision tensorboardX scikit-learn pandas geomloss
    ```
    *(Note: Ensure you have the correct PyTorch version for your hardware, e.g., CUDA for NVIDIA GPUs or MPS for Apple Silicon).*

### Directory Structure

Ensure your project directory looks like this before running:

```
.
├── MRNet-v1.0/              # Dataset directory
│   ├── train/
│   ├── valid/
│   ├── train-abnormal.csv
│   ├── ...
├── models/                  # Directory for saving model checkpoints (created automatically)
├── logs/                    # Directory for Tensorboard logs (created automatically)
├── train.py                 # Main training script
├── dataloader.py            # Data loading logic
├── alexnet.py               # Model architecture
├── utils.py                 # Helper functions
└── README.md
```

## Usage

To train the model, use the `train.py` script.

### Basic Usage

```bash
python train.py --prefix_name my_experiment
```

### Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--prefix_name` | **Required**. A prefix for the saved model file name. | N/A |
| `--task` | The task to train on (abnormal, acl, meniscus). *Note: The current implementation trains on all, this arg might be for logging/naming.* | `acl` |
| `--plane` | The plane configuration. | `Segittal_Coronal_and_Axial` |
| `--epochs` | Number of training epochs. | `50` |
| `--lr` | Learning rate. | `1e-5` |
| `--patience` | Early stopping patience (epochs without improvement). | `20` |
| `--save_model` | Whether to save the best model (1 for yes, 0 for no). | `1` |
| `--flush_history` | Whether to delete previous logs (1 for yes, 0 for no). | `0` |
| `--log_every` | Frequency of logging training status (in batches). | `100` |

### Example

Train a model for 100 epochs with a specific learning rate:

```bash
python train.py --prefix_name MRNet_Run1 --epochs 100 --lr 0.00001
```

## Results

The end-to-end multi-label multi-class model achieves competitive performance:

*   **Train AUC**: ~0.99
*   **Validation AUC**: ~0.915

## Contributions

Contributions are welcome! If you feel that some functionalities or improvements could be added to the project, don't hesitate to submit a pull request.
