"""Fast-suite checks that LoRA attaches to the language model only.

The load-bearing guarantees are pure (a regex + a frozen-param assert), so they run
with ``pytest -m "not slow"`` and no model download. The real ``get_peft_model``
wiring against a tiny/real MedGemma is the ``@slow`` test at the bottom.
"""

import re

import pytest
import torch
import torch.nn as nn

import vlm_finetune


# --- The language-only target regex ----------------------------------------- #
# PEFT matches target_modules with re.fullmatch, so these tests use fullmatch too
# (re.search masked C2: it matched regardless of the leading `model.` wrapper).
_PAT = vlm_finetune.LORA_TARGET_MODULES


@pytest.mark.parametrize("name", [
    "model.language_model.layers.0.self_attn.q_proj",       # real Gemma-3 layout
    "language_model.model.layers.0.self_attn.q_proj",       # older/no-wrapper layout
])
@pytest.mark.parametrize("proj", ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"])
def test_lora_regex_fullmatches_language_proj(name, proj):
    name = name.rsplit(".", 1)[0] + "." + proj
    assert re.fullmatch(_PAT, name) is not None


@pytest.mark.parametrize("name", [
    "model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj",
    "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj",
    "model.multi_modal_projector.mm_input_projection_weight",
    "model.language_model.norm",                            # not a proj
    "model.language_model.layers.0.self_attn.q_proj.weight",  # param, not module name
])
def test_lora_regex_rejects_non_targets(name):
    assert re.fullmatch(_PAT, name) is None


# --- assert_freeze: the no-leak guarantee ------------------------------------ #
class _TinyGemma(nn.Module):
    """Hand-built stand-in whose module names mirror Gemma-3's real layout."""

    def __init__(self, vision_trainable=False):
        super().__init__()
        self.vision_tower = nn.Linear(8, 8)
        self.multi_modal_projector = nn.Linear(8, 8)
        self.language_model = nn.Module()
        self.language_model.q_proj = nn.Linear(8, 8)
        self.language_model.v_proj = nn.Linear(8, 8)
        if not vision_trainable:
            for m in (self.vision_tower, self.multi_modal_projector):
                for p in m.parameters():
                    p.requires_grad_(False)


def test_assert_freeze_passes_when_vision_frozen():
    model = _TinyGemma(vision_trainable=False)
    trainable_M = vlm_finetune.assert_freeze(model)
    assert trainable_M > 0.0
    assert trainable_M < vlm_finetune._TRAINABLE_TRIPWIRE_M


def test_assert_freeze_raises_when_vision_trainable():
    model = _TinyGemma(vision_trainable=True)
    with pytest.raises(AssertionError):
        vlm_finetune.assert_freeze(model)


def test_trainable_param_count_counts_only_grad():
    model = _TinyGemma(vision_trainable=False)
    expected = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert vlm_finetune.trainable_param_count(model) == expected


# --- Real wiring (needs peft + a model) -------------------------------------- #
@pytest.mark.slow
def test_get_peft_model_targets_language_only():
    """Slow: attach LoRA to a real/tiny MedGemma and assert the vision tower +
    projector stay frozen while language LoRA params train."""
    if not vlm_finetune._HAVE_DEPS:
        pytest.skip("peft/transformers not installed")
    # The real model load is the @slow part; it needs HF_TOKEN + MPS.
    from peft import LoraConfig, get_peft_model

    import vlm_finetune as vf
    model = vf.load_base("google/medgemma-4b-it", torch.device("cpu"))[1]
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=vf.LORA_TARGET_MODULES))
    vf.assert_freeze(model)
