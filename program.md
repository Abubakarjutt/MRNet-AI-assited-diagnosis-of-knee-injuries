# MRNet Autoresearch

This folder now supports an `autoresearch`-style workflow for the MRNet dataset, including GitHub Actions triggers.

## Scope

The fastest iteration path is:

1. Edit [`experiment_configs.py`](/Users/Apple/Workdir/mrnet/MRNet-AI-assited-diagnosis-of-knee-injuries/experiment_configs.py) to define which runs to try.
2. Keep [`train.py`](/Users/Apple/Workdir/mrnet/MRNet-AI-assited-diagnosis-of-knee-injuries/train.py) as the canonical training entrypoint.
3. Launch [`experiment_runner.py`](/Users/Apple/Workdir/mrnet/MRNet-AI-assited-diagnosis-of-knee-injuries/experiment_runner.py) to run the queue and append results to `results.tsv`.

## Short Runs First

The default experiment group is `short`, not `full`.

- `short`: compact 5-minute experiments meant for automatic iteration.
- `full`: longer 20-minute comparison runs.
- `all`: both sets in sequence.

## What Changed

- Training now uses a single synchronized exam dataset instead of zipping three independent loaders.
- MRI volumes are memory-mapped and optionally cached in the dataset.
- Resize, channel expansion, and ImageNet normalization now happen as one batched device-side operation, which is much friendlier to Apple MPS.
- Lighter pretrained shared-encoder models are available:
  - `resnet18`
  - `mobilenet_v3_small`
  - `efficientnet_b0`

## Suggested Default Search Space

Start with the compact CNNs before touching the ViT models:

- `resnet18`: best balance of speed and stability.
- `mobilenet_v3_small`: strongest throughput candidate.
- `efficientnet_b0`: slightly slower, often a good accuracy compromise.

## Running Experiments

Single training run:

```bash
python train.py --prefix_name resnet18_trial --model_type resnet18
```

Batch experiment loop:

```bash
python experiment_runner.py --group short --run_tag mrnet_mps
```

Specific override example:

```bash
python experiment_runner.py --group short --set data_root=/absolute/path/to/MRNet-v1.0 num_workers=4
```

## GitHub Actions Trigger

The workflow file is [`mrnet-autoresearch.yml`](/Users/Apple/Workdir/mrnet/.github/workflows/mrnet-autoresearch.yml).

Important constraint:

- Use a self-hosted macOS Apple Silicon runner.
- GitHub-hosted runners will not have your MRNet dataset.
- GitHub-hosted runners also will not give you Apple MPS.

Recommended setup:

1. Register your Mac as a self-hosted GitHub Actions runner with labels `self-hosted`, `macOS`, and `ARM64`.
2. Set the repository variable `MRNET_DATA_ROOT` to the dataset path on that machine.
3. Trigger the workflow manually with the `short` group, or let the nightly schedule run it.

## Logging

`experiment_runner.py` writes:

- `results.tsv`: tab-separated experiment summary.
- `experiment_logs/*.log`: full stdout/stderr for each run.

## Notes For Apple Silicon

- Leave `--amp 1` and `--channels_last 1` enabled unless a run becomes numerically unstable.
- `--num_workers 2` is a reasonable default on a MacBook; tune it if worker startup or memory pressure becomes an issue.
- The shared lightweight CNNs are expected to be much faster than `vit_b_16` while remaining competitive enough to use as your main search baseline.
