# MRNet Autoresearch Program

This folder now supports a true autoresearch-style improvement loop for MRNet.

## Goal

Continuously improve the MRNet architecture through fixed-budget experiments.

The loop follows the same core principles as the original `autoresearch` repository:

1. Keep the evaluation harness fixed.
2. Use a fixed wall-clock budget per candidate.
3. Establish a baseline first.
4. Optimize validation AUC first, and use simplicity only to break true AUC ties.
5. Run autonomously without waiting for human input between iterations.

Each experiment should:

1. Run the baseline configuration first when the state directory is fresh.
2. Start from the current best known architecture after that baseline.
3. Mutate a small number of architecture or training knobs.
4. Run a fixed training budget.
5. Keep the new configuration only if validation AUC improves, or if AUC is effectively identical while the model is simpler.
6. Use the kept winner as the parent for the next experiment.

## Source Of Truth

- [`autoresearch_loop.py`](/Users/Apple/Workdir/mrnet/MRNet-AI-assited-diagnosis-of-knee-injuries/autoresearch_loop.py) is the autonomous controller.
- [`train.py`](/Users/Apple/Workdir/mrnet/MRNet-AI-assited-diagnosis-of-knee-injuries/train.py) is the canonical training entrypoint.
- [`lightweight_models.py`](/Users/Apple/Workdir/mrnet/MRNet-AI-assited-diagnosis-of-knee-injuries/lightweight_models.py) contains the searchable architecture family.

## Fixed Harness

The autonomous loop intentionally keeps these parts fixed during ordinary iterations:

- dataset and label construction in [`dataloader.py`](/Users/Apple/Workdir/mrnet/MRNet-AI-assited-diagnosis-of-knee-injuries/dataloader.py)
- training/evaluation metric extraction in [`train.py`](/Users/Apple/Workdir/mrnet/MRNet-AI-assited-diagnosis-of-knee-injuries/train.py)
- persistent experiment logging in [`autoresearch_loop.py`](/Users/Apple/Workdir/mrnet/MRNet-AI-assited-diagnosis-of-knee-injuries/autoresearch_loop.py)

This is the MRNet analogue of `autoresearch` keeping `prepare.py` fixed and iterating on the research surface only.

## Search Strategy

The loop mutates the current best config over:

- backbone: `resnet18`, `mobilenet_v3_small`, `efficientnet_b0`
- pooling: `max`, `mean`, `lse`, `attention`, `gem`
- projection dimension
- hidden dimension
- fusion depth
- fusion gate
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
2. runs the baseline first if the state directory is fresh
3. runs research-driven mutations after that
4. promotes only improved or simplicity-winning candidates
5. uploads the latest summary, best config, and candidate logs as artifacts

The default per-candidate budget is now 60 minutes.
