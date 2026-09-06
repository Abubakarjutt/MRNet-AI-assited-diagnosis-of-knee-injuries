# tests/test_cv_select.py
from types import SimpleNamespace
import csv
from scripts import cv_select


def _base(cache, mrnet_fixture):
    return SimpleNamespace(
        feature_cache=str(cache), data_root=str(mrnet_fixture), encoders="fake",
        cache_variants="clean,hflip", slices_used=4, slices_used_meniscus=6, d_model=8,
        dropout=0.1, epochs=1, lr=1e-3, weight_decay=0.0, ema_decay=0.0, focal_gamma=2.0,
        loss_type="focal", label_smoothing=0.0, patience=0, select_metric="per_task_mean",
        featbank_batch_size=2, eval_tta_variants="clean", prefix_name="cv", save_model=0,
        seed=0, time_budget_minutes=None,
    )


def test_cv_select_writes_tsv_with_expected_rows(feature_cache_fixture, mrnet_fixture, tmp_path):
    out = tmp_path / "cv.tsv"
    rows = cv_select.cv_select(_base(feature_cache_fixture, mrnet_fixture),
                               k=2, seed=0, d_model_grid=[8], out_tsv=str(out))
    assert len(rows) == 2                                   # 1 d_model x 2 folds
    with open(out) as fh:
        reader = list(csv.DictReader(fh, delimiter="\t"))
    assert len(reader) == 2
    assert {"d_model", "fold", "abnormal_auc", "acl_auc", "meniscus_auc", "per_task_mean"} <= set(reader[0])


def test_cv_select_sweeps_d_model(feature_cache_fixture, mrnet_fixture, tmp_path):
    rows = cv_select.cv_select(_base(feature_cache_fixture, mrnet_fixture),
                               k=2, seed=0, d_model_grid=[8, 12], out_tsv=str(tmp_path / "s.tsv"))
    assert sorted({r["d_model"] for r in rows}) == [8, 12]
    assert len(rows) == 4
