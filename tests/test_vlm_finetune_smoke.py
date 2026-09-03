"""Fast-suite coverage for vlm_finetune's pure helpers + a @slow end-to-end smoke.

The pure helpers (arg parsing, one-best-per-run adapter cleanup, naming, trainable
count) need no model, so they run with ``pytest -m "not slow"``. The full ``run()``
train + eval_only smoke needs peft + a real/tiny model and is the ``@slow`` test.
"""

import os

import pytest
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

import vlm_finetune


# --- Local fakes for the fast run()/train_one_epoch/score_exam coverage ------- #
class _FakeOut:
    def __init__(self, loss, logits):
        self.loss, self.logits = loss, logits


class _FakeVLM(nn.Module):
    """Trainable stand-in: a real Linear param so backward populates grads, a
    forward that returns .loss (depends on the param) and .logits, plus the
    attributes vlm_finetune / vlm_common read (device, dtype, save_pretrained)."""

    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.device = torch.device("cpu")
        self.dtype = dtype
        self.forward_dtypes = []

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None,
                labels=None, **kw):
        if pixel_values is not None:
            self.forward_dtypes.append(pixel_values.dtype)
        t = self.lin(torch.ones(4))
        loss = (t ** 2).sum() + self.lin.bias.pow(2).sum()
        seq = int(input_ids.shape[-1]) if input_ids is not None else 8
        logits = torch.zeros(1, seq, 64)
        return _FakeOut(loss, logits)

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "adapter_config.json"), "w") as fh:
            fh.write("{}")


def _patch_common(monkeypatch, model):
    monkeypatch.setattr(vlm_finetune, "apply_lora", lambda m, a: m)
    monkeypatch.setattr(
        vlm_finetune, "get_cosine_schedule_with_warmup",
        lambda opt, num_warmup_steps, num_training_steps: LambdaLR(opt, lambda s: 1.0),
        raising=False,
    )


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


def test_prune_keeps_the_freshly_saved_dir(tmp_path, monkeypatch):
    models = tmp_path / "models"
    fresh = models / "run_x_medgemma_lora_valauc_0.9100"
    stale = models / "run_x_medgemma_lora_valauc_0.8800"
    fresh.mkdir(parents=True)
    stale.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    vlm_finetune._prune_prior_adapter_dirs("run_x", keep=str(fresh))
    assert fresh.is_dir() is True          # the just-saved best survives its own prune
    assert stale.is_dir() is False


# --- resume-from-checkpoint (chained one-epoch runs) ----------------------- #
def test_parse_valauc_recovers_score_from_dir_name():
    assert vlm_finetune._parse_valauc(
        "models/medgemma_full_medgemma_lora_valauc_0.7877") == 0.7877
    assert vlm_finetune._parse_valauc(
        "/abs/path/run_x_medgemma_lora_valauc_0.9100/") == 0.9100


def test_parse_valauc_is_zero_without_a_tag():
    assert vlm_finetune._parse_valauc(None) == 0.0
    assert vlm_finetune._parse_valauc("models/some_hand_named_adapter") == 0.0


def test_resume_adapter_arg_defaults_none_and_parses(monkeypatch):
    monkeypatch.setattr("sys.argv",
                        ["vlm_finetune.py", "--prefix_name", "smoke"])
    assert vlm_finetune.parse_arguments().resume_adapter is None
    monkeypatch.setattr("sys.argv",
                        ["vlm_finetune.py", "--prefix_name", "smoke",
                         "--resume_adapter", "models/x_valauc_0.80"])
    assert vlm_finetune.parse_arguments().resume_adapter == "models/x_valauc_0.80"


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
    if not os.environ.get("HF_TOKEN"):
        pytest.skip("HF_TOKEN not set (gated google/medgemma-4b-it)")

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


# --- Fast run()/train_one_epoch/score_exam coverage (NOT @slow) -------------- #
def test_train_run_actually_updates_weights(mrnet_fixture, fake_processor, tmp_path, monkeypatch):
    """C1: with the step-order bug the optimizer never applies an update. Fixed, the
    LoRA (here: the fake's Linear) weights must change after a 2-batch epoch."""
    monkeypatch.chdir(tmp_path)
    model = _FakeVLM()
    monkeypatch.setattr(vlm_finetune, "load_base", lambda base, device: (fake_processor, model))
    _patch_common(monkeypatch, model)

    before = model.lin.weight.detach().clone()
    args = _args(monkeypatch, [
        "--prefix_name", "c1", "--data_root", str(mrnet_fixture),
        "--epochs", "1", "--max_train_batches", "2", "--max_val_batches", "2",
        "--grad_accum", "1",
    ])
    vlm_finetune.run(args)
    assert not torch.equal(before, model.lin.weight), "optimizer never stepped (C1)"
    assert [d for d in os.listdir("models") if "c1_medgemma_lora" in d], "no adapter dir"


def test_train_batch_reaches_model_in_model_dtype(mrnet_fixture, fake_processor, tmp_path, monkeypatch):
    """C3: the training batch must be moved + pixel_values cast to model.dtype."""
    monkeypatch.chdir(tmp_path)
    model = _FakeVLM(dtype=torch.float64)          # distinct, CPU-safe
    monkeypatch.setattr(vlm_finetune, "load_base", lambda base, device: (fake_processor, model))
    _patch_common(monkeypatch, model)
    args = _args(monkeypatch, [
        "--prefix_name", "c3", "--data_root", str(mrnet_fixture),
        "--epochs", "1", "--max_train_batches", "1", "--max_val_batches", "1",
        "--grad_accum", "1",
    ])
    vlm_finetune.run(args)
    assert model.forward_dtypes, "model never saw pixel_values"
    assert all(d == torch.float64 for d in model.forward_dtypes), model.forward_dtypes


def test_eval_only_does_not_build_model(mrnet_fixture, fake_processor, tmp_path, monkeypatch):
    """C5: --eval_only must not call build_model (which would load the 8 GB base a
    second time)."""
    monkeypatch.chdir(tmp_path)
    model = _FakeVLM()

    def _boom(*a, **k):
        raise AssertionError("build_model must not run in --eval_only (C5)")

    monkeypatch.setattr(vlm_finetune, "build_model", _boom)
    monkeypatch.setattr(vlm_finetune, "_rebuild_with_adapter",
                        lambda args, device: (fake_processor, model, 1.0))
    _patch_common(monkeypatch, model)
    adapter = tmp_path / "some_adapter"
    adapter.mkdir()
    args = _args(monkeypatch, [
        "--prefix_name", "c5", "--data_root", str(mrnet_fixture),
        "--eval_only", str(adapter), "--max_val_batches", "2",
    ])
    auc = vlm_finetune.run(args)
    assert 0.0 <= float(auc) <= 1.0
