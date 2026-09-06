# tests/test_featbank_train.py
from types import SimpleNamespace
import numpy as np
import featbank_train as ft


def test_make_folds_deterministic_partition_disjoint():
    a = ft.make_folds(37, 5, seed=0)
    b = ft.make_folds(37, 5, seed=0)
    assert [list(x) for x in a] == [list(y) for y in b]
    allidx = np.concatenate(a)
    assert sorted(allidx.tolist()) == list(range(37))
    c = ft.make_folds(37, 5, seed=1)
    assert sorted(np.concatenate(c).tolist()) == list(range(37))


def _args(cache, mrnet_fixture, **over):
    base = dict(
        feature_cache=str(cache), data_root=str(mrnet_fixture),
        encoders="fake", cache_variants="clean,hflip", slices_used=4,
        slices_used_meniscus=6, d_model=8, dropout=0.1, epochs=1, lr=1e-3,
        weight_decay=0.0, ema_decay=0.0, focal_gamma=2.0, loss_type="focal",
        label_smoothing=0.0, patience=0, select_metric="per_task_mean",
        featbank_batch_size=2, eval_tta_variants="clean", prefix_name="tfb",
        save_model=0, cv_folds=0, cv_fold=-1, seed=0, time_budget_minutes=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_run_featbank_returns_metrics_dict(feature_cache_fixture, mrnet_fixture):
    out = ft.run_featbank(_args(feature_cache_fixture, mrnet_fixture))
    assert set(out) == {"per_task_auc", "per_task_mean", "pooled_auc", "best_epoch"}
    assert len(out["per_task_auc"]) == 3
    assert 0.0 <= out["per_task_mean"] <= 1.0


def test_run_featbank_cv_fold_runs(feature_cache_fixture, mrnet_fixture):
    out = ft.run_featbank(_args(feature_cache_fixture, mrnet_fixture, cv_folds=2, cv_fold=0))
    assert "per_task_mean" in out


def test_select_metric_loss_is_accepted(feature_cache_fixture, mrnet_fixture):
    out = ft.run_featbank(_args(feature_cache_fixture, mrnet_fixture, select_metric="loss"))
    assert out["best_epoch"] >= 1
