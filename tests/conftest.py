import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import csv
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from dataloader import PLANES, TASKS

_SEED = 20260901
_SPLIT_SIZES = {"train": 6, "valid": 4}


def _write_split(root, split, n, rng):
    exam_ids = [f"{i:04d}" for i in range(n)]

    per_task_labels = {}
    for task in TASKS:
        column = [0, 1] + [rng.randint(0, 1) for _ in range(n - 2)]
        rng.shuffle(column)
        per_task_labels[task] = column

    for task in TASKS:
        with open(root / f"{split}-{task}.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            for exam_id, label in zip(exam_ids, per_task_labels[task]):
                writer.writerow([exam_id, label])

    for plane in PLANES:
        plane_dir = root / split / plane
        plane_dir.mkdir(parents=True, exist_ok=True)
        for exam_id in exam_ids:
            slice_count = rng.randint(12, 28)
            array = np.random.default_rng(rng.randint(0, 2**31 - 1)).random(
                 (slice_count, 256, 256), dtype=np.float32
             ) * 255.0
            np.save(plane_dir / f"{exam_id}.npy", array)


@pytest.fixture
def mrnet_fixture(tmp_path):
    rng = random.Random(_SEED)
    root = tmp_path / "MRNet-v1.0"
    root.mkdir(parents=True, exist_ok=True)
    for split, n in _SPLIT_SIZES.items():
         _write_split(root, split, n, rng)
    return root


class _StubTower(nn.Module):
    def __init__(self, feature_dim=1152):
        super().__init__()
        self.proj = nn.Linear(3, feature_dim)

    def forward(self, pixel_values):
        pooled = self.proj(pixel_values.mean(dim=(2, 3)))   # [N,3] -> [N,feature_dim]
        return SimpleNamespace(pooler_output=pooled)


class _StubSiglip(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = _StubTower()


class _FakeAutoModel:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return _StubSiglip()


@pytest.fixture
def stub_medsiglip(monkeypatch):
    monkeypatch.setattr("medical_encoders.AutoModel", _FakeAutoModel)
    return _FakeAutoModel


# --------------------------------------------------------------------------- #
# MedGemma-4B LoRA (Path 2) - hand-built fake processor + fake model.          #
# No network, no MPS: the real google/medgemma-4b-it is unavailable here        #
# (no HF_TOKEN, MPS off). The fake exercises the load-bearing scoring logic      #
# (digit_token_ids / locate_answer_slots / score_exam) on the fast suite; the    #
# real model path is covered by the @slow test_vlm_real.py.                     #
# --------------------------------------------------------------------------- #
_NIMG_TOKENS = 8          # placeholder tokens each image expands to
_VOCAB = 512             # must exceed every id we emit
_ROLE = {"system": 100, "user": 101, "assistant": 102}
_BOS, _EOS, _IMG = 1, 2, 3


class _FakeTokenizer:
    """Char-level tokenizer: every distinct char maps to a unique id in [10, ...).
    '0' and '1' therefore render as distinct single tokens - the property that
    digit_token_ids relies on."""

    def __init__(self):
        self._chars = sorted({chr(c) for c in range(32, 127)})
        self.char2id = {c: 10 + i for i, c in enumerate(self._chars)}
        self.id2char = {v: k for k, v in self.char2id.items()}
        for i in (_BOS, _EOS, _IMG):
            self.id2char.setdefault(i, "")
        for r in _ROLE.values():
            self.id2char.setdefault(r, "")

    def encode(self, text, add_special_tokens=True):
        ids = [_BOS] if add_special_tokens else []
        ids.extend(self.char2id[c] for c in str(text))
        if add_special_tokens:
            ids.append(_EOS)
        return ids

    def __call__(self, text, add_special_tokens=True, **_):
        return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}

    def decode(self, ids, skip_special_tokens=False, **_):
        return "".join(self.id2char.get(int(i), "") for i in ids)


class _FakeBatch(dict):
    """Dict subclass that supports .to(device) and **batch splatting."""

    def to(self, device):
        for k, v in list(self.items()):
            if torch.is_tensor(v):
                self[k] = v.to(device)
        return self


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()

    def apply_chat_template(self, conversation, add_generation_prompt=False,
                            tokenize=True, return_dict=True, return_tensors="pt",
                            do_pan_and_scan=False, **_):
        ids = [_BOS]
        for msg in conversation:
            role = msg["role"]
            ids.append(_ROLE[role])
            content = msg["content"]
            parts = content if isinstance(content, list) else [{"type": "text", "text": content}]
            for part in parts:
                if part["type"] == "image":
                    ids.extend([_IMG] * _NIMG_TOKENS)
                else:
                    ids.extend(self.tokenizer.encode(part["text"], add_special_tokens=False))
        if add_generation_prompt:
            ids.append(_ROLE["assistant"])

        input_ids = torch.tensor([ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        n_images = sum(
             1
            for m in conversation
            for p in (m["content"] if isinstance(m["content"], list)
                      else [{"type": "text", "text": m["content"]}])
            if p["type"] == "image"
          )
        pixel_values = torch.randn(max(n_images, 1), 3, 32, 32)
        batch = _FakeBatch(
             input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
          )
        if add_generation_prompt:
            batch["assistant_start"] = int(input_ids.shape[1]) - 1
        return batch


class _FakeModel(nn.Module):
    """Minimal stand-in: forward returns SimpleNamespace(logits=[1,T,V]).
    `logits` is monkeypatchable so tests can force P(one_id)->1 at every position."""

    def __init__(self, vocab_size=_VOCAB, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.vocab_size = vocab_size
        self._logits = None
        self.proj = nn.Linear(8, 8)   # a real parameter so it is a legitimate module

    def forward(self, input_ids, attention_mask=None, pixel_values=None, **_):
        t = input_ids.shape[-1]
        if self._logits is not None:
            logits = self._logits
        else:
            logits = torch.zeros(1, t, self.vocab_size, device=self.device)
        return SimpleNamespace(logits=logits)


@pytest.fixture
def fake_processor():
    return _FakeProcessor()


@pytest.fixture
def fake_model():
    return _FakeModel(device="cpu")
