"""@slow real-model checks for MedGemma-4B LoRA (needs HF_TOKEN + MPS/CPU model).

Skipped automatically when peft/transformers are missing or HF_TOKEN is unset, so the
fast suite stays green with no download. These guard the parts the fake fixtures cannot:
the real tokenizer's digit tokens and one real teacher-forced scoring pass.
"""

import os

import numpy as np
import pytest
import torch

import vlm_finetune
import vlm_common
from dataloader import TASKS

_BASE = "google/medgemma-4b-it"


def _skip_if_unavailable():
    if not vlm_finetune._HAVE_DEPS:
        pytest.skip("peft/transformers not installed")
    if not os.environ.get("HF_TOKEN"):
        pytest.skip("HF_TOKEN not set")


@pytest.mark.slow
def test_real_digit_token_ids_are_single_tokens():
    _skip_if_unavailable()
    processor, _ = vlm_finetune.load_base(_BASE, torch.device("cpu"))
    zero_id, one_id = vlm_common.digit_token_ids(processor)
    assert zero_id != one_id
    assert processor.tokenizer.decode([zero_id]).strip() == "0"
    assert processor.tokenizer.decode([one_id]).strip() == "1"


@pytest.mark.slow
def test_real_score_exam_returns_finite_probs(mrnet_fixture):
    _skip_if_unavailable()
    processor, model, _ = vlm_finetune.build_model(
        vlm_finetune.parse_arguments(["--prefix_name", "slow-real"]),
        torch.device("cpu"),
    )
    import vlm_dataset
    ds = vlm_dataset.MRVLMDataset(
        str(mrnet_fixture), train=False, k=6, grid=(3, 2), cell=448,
        mmap=False, cache_size=4,
     )
    probs = vlm_common.score_exam(model, processor, ds[0]["images"])
    assert probs.shape == (len(TASKS),)
    assert np.all(np.isfinite(probs))
    assert np.all((probs >= 0.0) & (probs <= 1.0))
