# MRNet Autoresearch Program

This folder now supports a true autoresearch-style improvement loop for MRNet.

## Goal

Continuously improve the MRNet architecture through fixed-budget experiments.

Each experiment should:

1. Start from the current best known architecture.
2. Mutate a small number of architecture or training knobs.
3. Run a fixed training budget.
4. Keep the new configuration only if validation AUC improves.
5. Use the kept winner as the parent for the next experiment.

## Source Of Truth

- [`autoresearch_loop.py`](/Users/Apple/Workdir/mrnet/MRNet-AI-assited-diagnosis-of-knee-injuries/autoresearch_loop.py) is the autonomous controller.
- [`train.py`](/Users/Apple/Workdir/mrnet/MRNet-AI-assited-diagnosis-of-knee-injuries/train.py) is the canonical training entrypoint.
- [`lightweight_models.py`](/Users/Apple/Workdir/mrnet/MRNet-AI-assited-diagnosis-of-knee-injuries/lightweight_models.py) contains the searchable architecture family.

## Search Strategy

The loop mutates the current best config over:

- backbone: `resnet18`, `mobilenet_v3_small`, `efficientnet_b0`
- pooling: `max`, `mean`, `lse`
- projection dimension
- hidden dimension
- fusion depth
- dropout
- learning rate
- weight decay
- image size
- cache size
- worker count

## Persistent State

The best architecture is persisted outside the repo checkout so scheduled GitHub Actions runs can keep improving over time on the same Mac.

Default state directory:

```bash
~/.mrnet_autoresearch
```

Important files there:

- `best_config.json`
- `state.json`
- `results.tsv`
- `logs/`
- `configs/`

## Local Usage

Run one improvement iteration:

```bash
python3 autoresearch_loop.py --iterations 1 --data_root MRNet-v1.0
```

## GitHub Actions

The workflow at [`mrnet-autoresearch.yml`](/Users/Apple/Workdir/mrnet/.github/workflows/mrnet-autoresearch.yml) now runs this continuous improvement loop on the self-hosted Apple Silicon runner.

Each dispatch or scheduled run:

1. loads the persistent best config from the runner state directory
2. runs several short mutations
3. promotes only improved candidates
4. uploads the latest summary and best config as artifacts

The default per-candidate budget is now 60 minutes.
