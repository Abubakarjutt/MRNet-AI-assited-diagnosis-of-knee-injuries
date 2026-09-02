"""End-to-end smoke: the full medsiglip run() path on synthetic data + stub tower.

Exercises build_model -> iterate_epoch (train + val) -> checkpoint-on-best -> summary
block in one shot. ~5s. No network, no real weights (stub_medsiglip), CPU.
"""
import glob
import os

import torch

import train


def test_medsiglip_run_end_to_end(mrnet_fixture, stub_medsiglip, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # logs/ and models/ land under tmp
    # get_device() would pick MPS on this machine; the stub tower is a plain
    # float32 nn.Linear and the run only needs to prove the wiring, so pin CPU.
    monkeypatch.setattr("utils.get_device", lambda: torch.device("cpu"))
    argv = [
        "train.py",
        "--prefix_name", "medsiglip_smoke",
        "--model_type", "medsiglip",
        "--pooling", "gem",
        "--plane_fusion", "plane_attention",
        "--fusion_depth", "2",
        "--data_root", str(mrnet_fixture),
        "--epochs", "1",
        "--max_train_batches", "2",
        "--max_val_batches", "2",
        "--num_workers", "0",
        "--medsiglip_chunk", "4",
    ]
    monkeypatch.setattr("sys.argv", argv)
    args = train.parse_arguments()
    train.run(args)

    checkpoints = glob.glob(os.path.join("models", "*medsiglip_smoke*.pth"))
    assert checkpoints, "run() wrote no checkpoint"
    # run() saves a bare model.state_dict() (train.py) so weights_only=True is safe,
    # matching tests/test_medsiglip_backbone.py.
    state = torch.load(checkpoints[0], map_location="cpu", weights_only=True)
    assert state, "checkpoint state_dict is empty"
    assert not any(k.startswith("encoder.tower") for k in state), \
        "frozen tower leaked into the checkpoint"
    assert any(k.startswith("classifier") for k in state), \
        "classifier head missing from checkpoint"
