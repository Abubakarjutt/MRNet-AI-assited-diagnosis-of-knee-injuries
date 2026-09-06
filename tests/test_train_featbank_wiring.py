from types import SimpleNamespace
import train


def test_run_dispatches_to_run_featbank(monkeypatch):
    seen = {}
    monkeypatch.setattr("featbank_train.run_featbank", lambda a: seen.setdefault("args", a) or {"per_task_mean": 0.9})
    train.run(SimpleNamespace(model_type="featbank", batch_size=1, select_metric="per_task_mean"))
    assert seen["args"].model_type == "featbank"


def test_parser_has_featbank_choice_and_args():
    args = train.parse_args([
        "--prefix_name", "x", "--model_type", "featbank",
        "--feature_cache", "/tmp/fc",
    ])
    assert args.model_type == "featbank"
    assert args.feature_cache == "/tmp/fc"
    assert args.encoders == "medsiglip,dinov2"
    assert args.slices_used == 24 and args.d_model == 256
    assert args.seed == 0


def test_select_metric_defaults_by_model_type():
    fb = train.parse_args(["--prefix_name", "x", "--model_type", "featbank"])
    assert fb.select_metric == "per_task_mean"
    cnn = train.parse_args(["--prefix_name", "x", "--model_type", "resnet18"])
    assert cnn.select_metric == "loss"
