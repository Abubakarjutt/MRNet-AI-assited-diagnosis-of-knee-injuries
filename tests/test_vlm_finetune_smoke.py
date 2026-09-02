"""Fast-suite coverage for vlm_finetune's pure helpers + a @slow end-to-end smoke.

The pure helpers (arg parsing, one-best-per-run adapter cleanup, naming, trainable
count) need no model, so they run with ``pytest -m "not slow"``. The full ``run()``
train + eval_only smoke needs peft + a real/tiny model and is the ``@slow`` test.
"""

import os

import pytest
import torch
import torch.nn as nn

import vlm_finetune


# --- parse_arguments --------------------------------------------------------- #
def test_parse_arguments_defaults(monkeypatch):
    monkeypatch.setattr("sys.argv",
                        ["vlm_finetune.py", "--prefix_name", "smoke",
                         "--data_root", "MRNet-v1.0"])
    args = vlm_finetune.parse_arguments()
    assert args.prefix_name == "smoke"
    assert args.batch_size == 1
    assert args.slice_strategy in ("uniform", "center")
    assert args.mmap in (0, 1)


# --- one-best-per-run adapter cleanup --------------------------------------- #
def test_adapter_dir_naming():
    name = vlm_finetune._adapter_dir("run_x", 0.9123)
    assert name == os.path.join("models", "run_x_medgemma_lora_valauc_0.9123")


def test_prune_prior_adapter_dirs_removes_matching(tmp_path, monkeypatch):
    models = tmp_path / "models"
    keep = models / "other_medgemma_lora_valauc_0.9000"
    drop = models / "run_x_medgemma_lora_valauc_0.8800"
    keep.mkdir(parents=True)
    drop.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    vlm_finetune._prune_prior_adapter_dirs("run_x")
    assert drop.is_dir() is False
    assert keep.is_dir() is True          # a different prefix must be untouched


def test_prune_is_noop_when_no_models_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    vlm_finetune._prune_prior_adapter_dirs("run_x")      # must not raise


# --- trainable count --------------------------------------------------------- #
def test_trainable_param_count_counts_only_grad():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    for p in model[0].parameters():
        p.requires_grad_(False)
    expected = 4 * 4 + 4               # second layer: weights(16) + bias(4)
    assert vlm_finetune.trainable_param_count(model) == expected


# --- End-to-end smoke (needs peft + a model) --------------------------------- #
def _args(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["vlm_finetune.py", *argv])
    return vlm_finetune.parse_arguments()


@pytest.mark.slow
def test_run_train_and_eval_only_smoke(mrnet_fixture, tmp_path, monkeypatch):
    """A 2-step train run writes an adapter dir; --eval_only reloads it and prints
    the section-5.5 metrics block with a parseable best_val_auc."""
    if not vlm_finetune._HAVE_DEPS:
        pytest.skip("peft/transformers not installed")

    train_args = _args(
        monkeypatch,
        ["--prefix_name", "smoke_train", "--data_root", str(mrnet_fixture),
         "--epochs", "1", "--max_train_batches", "2", "--max_val_batches", "2",
         "--lora_r", "4", "--lora_alpha", "8", "--grad_accum", "1"],
    )
    vlm_finetune.run(train_args)

    saved = [d for d in os.listdir("models") if "smoke_train_medgemma_lora" in d]
    assert saved, "no adapter dir was written"

    eval_args = _args(
        monkeypatch,
        ["--prefix_name", "smoke_train", "--data_root", str(mrnet_fixture),
         "--eval_only", os.path.join("models", saved[0]), "--max_val_batches", "2"],
    )
    auc = vlm_finetune.run(eval_args)
    assert 0.0 <= float(auc) <= 1.0
