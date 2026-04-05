# MRNet: AI-Assisted Diagnosis of Knee Injuries

This repository contains an implementation of the research paper [MRNet: Deep-learning-assisted diagnosis for knee magnetic resonance](https://stanfordmlgroup.github.io/projects/mrnet/).

## Project Overview

This project trains a single end-to-end model over all three MRI planes instead of training separate models per plane. The current codebase now supports both the original heavier transformer-style path and a faster autoresearch-oriented path built around lighter shared encoders, short experiment budgets, and Apple Silicon friendly preprocessing.

## Dataset: MRNet

The data comes from the Stanford ML Group research lab. It consists of 1,370 knee MRI exams performed at Stanford University Medical Center to study the presence of Anterior Cruciate Ligament (ACL) tears and Meniscus tears.

- **Input**: Multi-view MRI scans (Sagittal, Coronal, Axial).
- **Task**: Multi-label, multi-class classification.
- **Labels**: Abnormal, ACL tear, Meniscus tear.

You can request the dataset from [this link](https://stanfordmlgroup.github.io/competitions/mrnet/).

## Setup

### Prerequisites

- Python 3.11+
- PyTorch compatible with your hardware, including MPS for Apple Silicon if available

### Installation

```bash
git clone https://github.com/your-repo/mrnet.git
cd mrnet
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Directory Structure

```text
.
├── MRNet-v1.0/
├── models/
├── logs/
├── train.py
├── dataloader.py
├── advanced_vit.py
├── lightweight_models.py
├── experiment_runner.py
├── experiment_configs.py
└── README.md
```

## Usage

`train.py` is the canonical training entrypoint in this workspace.

### Basic Usage

```bash
python train.py --prefix_name my_experiment --model_type resnet18
```

### Command-Line Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--prefix_name` | Required experiment prefix used for logs and model names. | N/A |
| `--task` | Task name used for logging. | `acl` |
| `--plane` | Plane configuration label. | `Segittal_Coronal_and_Axial` |
| `--epochs` | Number of training epochs. | `30` |
| `--lr` | Learning rate. | `3e-4` |
| `--patience` | Early stopping patience. | `8` |
| `--save_model` | Save best checkpoint. | `1` |
| `--flush_history` | Clear previous logs. | `0` |
| `--log_every` | Logging interval in batches. | `25` |
| `--data_root` | Path to dataset root. | `MRNet-v1.0` |
| `--num_workers` | DataLoader workers. | `2` |
| `--model_type` | Includes `resnet18`, `mobilenet_v3_small`, `efficientnet_b0`, `advanced`, and `multiscale`. | `resnet18` |
| `--time_budget_minutes` | Optional wall-clock budget for experiment runs. | unset |
| `--max_train_batches` | Optional cap for train batches. | unset |
| `--max_val_batches` | Optional cap for validation batches. | unset |

### Example

```bash
python train.py --prefix_name r18_trial --model_type resnet18
python train.py --prefix_name mobile_trial --model_type mobilenet_v3_small
python train.py --prefix_name vit_trial --model_type advanced --vit_model vit_b_16
```

### Quick Smoke Test

```bash
python train.py \
  --prefix_name smoke_test \
  --epochs 1 \
  --pretrained 0 \
  --save_model 0 \
  --max_train_batches 2 \
  --max_val_batches 2
```

## Faster Training Path

The faster MRNet path is aimed at short automatic experiments and Apple Silicon:

- A single exam-level dataset keeps sagittal, coronal, and axial scans synchronized.
- `.npy` volumes are memory-mapped and optionally cached.
- Resize and normalization happen in one batched operation on the device.
- Lightweight pretrained backbones are available through `train.py`.

Recommended first runs:

```bash
python train.py --prefix_name r18_trial --model_type resnet18
python train.py --prefix_name mobile_trial --model_type mobilenet_v3_small
python train.py --prefix_name effb0_trial --model_type efficientnet_b0
```

## Autoresearch-Style Loop

An experiment harness inspired by Karpathy's `autoresearch` is available:

```bash
python experiment_runner.py --group short --run_tag mrnet_mps
```

Edit `experiment_configs.py` to change the search queue, and read `program.md` for the intended workflow. Results are appended to `results.tsv`, with per-run logs written to `experiment_logs/`.

Available experiment groups:

- `short`: default 5-minute automatic experiments
- `full`: longer 20-minute comparison runs
- `all`: both groups in sequence

Example with runtime overrides:

```bash
python experiment_runner.py --group short --set data_root=/absolute/path/to/MRNet-v1.0 num_workers=4
```

## GitHub Actions

A GitHub Actions workflow is included at `.github/workflows/mrnet-autoresearch.yml`.

This is designed for a self-hosted Apple Silicon runner because:

- the MRNet dataset is local and not available on GitHub-hosted runners
- Apple MPS is only available on your Mac, not on standard hosted runners

Suggested setup:

1. Register your Mac as a self-hosted Actions runner with labels `self-hosted`, `macOS`, and `ARM64`.
2. Set the repository variable `MRNET_DATA_ROOT` to the local dataset path on that runner.
3. Trigger the workflow manually from Actions, or use the built-in nightly schedule.

The workflow uploads `ci_results/` and `experiment_logs/` as artifacts after each run.

## Results

The model is evaluated primarily with validation AUC. The compact shared-encoder path is intended to trade a small amount of peak accuracy for much faster iteration, which makes it a better fit for automatic experimentation on a Mac.

## Contributions

Contributions are welcome. If you have ideas for model improvements, data pipeline optimizations, or better experiment strategies, feel free to open a pull request.
