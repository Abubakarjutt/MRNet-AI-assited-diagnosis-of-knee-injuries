"""5-fold CV over cached features for Path 3 hyperparameter / threshold selection
(spec §6). Writes cv_results.tsv; the 120-val number is produced separately by a
plain `python train.py --model_type featbank` run (no --cv_fold)."""

import argparse
import copy
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from featbank_train import run_featbank                     # noqa: E402

_FIELDS = ["d_model", "fold", "abnormal_auc", "acl_auc", "meniscus_auc", "per_task_mean"]


def cv_select(base_args, *, k=5, seed=0, d_model_grid=None, out_tsv="cv_results.tsv"):
    grid = d_model_grid or [getattr(base_args, "d_model", 256)]
    rows = []
    for dm in grid:
        for fold in range(k):
            a = copy.copy(base_args)
            a.d_model = dm
            a.cv_folds = k
            a.cv_fold = fold
            a.seed = seed
            res = run_featbank(a)
            rows.append({
                "d_model": dm, "fold": fold,
                "abnormal_auc": round(res["per_task_auc"][0], 4),
                "acl_auc": round(res["per_task_auc"][1], 4),
                "meniscus_auc": round(res["per_task_auc"][2], 4),
                "per_task_mean": round(res["per_task_mean"], 4),
            })
    with open(out_tsv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_cache", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--encoders", default="medsiglip,dinov2")
    ap.add_argument("--cache_variants", default="clean,hflip,rotp,rotn,slicesB,slicesC")
    ap.add_argument("--slices_used", type=int, default=24)
    ap.add_argument("--slices_used_meniscus", type=int, default=32)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--ema_decay", type=float, default=0.0)
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--loss_type", default="focal")
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--select_metric", default="per_task_mean")
    ap.add_argument("--featbank_batch_size", type=int, default=16)
    ap.add_argument("--eval_tta_variants", default="clean")
    ap.add_argument("--prefix_name", default="cvsel")
    ap.add_argument("--save_model", type=int, default=0)
    ap.add_argument("--time_budget_minutes", type=float, default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep_d_model", type=int, default=0)
    ap.add_argument("--out", default="cv_results.tsv")
    a = ap.parse_args(argv)
    grid = [192, 256, 384] if a.sweep_d_model else [a.d_model]
    rows = cv_select(a, k=a.k, seed=a.seed, d_model_grid=grid, out_tsv=a.out)
    for dm in sorted({r["d_model"] for r in rows}):
        sub = [r["per_task_mean"] for r in rows if r["d_model"] == dm]
        print(f"d_model={dm}: mean per_task_mean over {len(sub)} folds = {sum(sub) / len(sub):.4f}")


if __name__ == "__main__":
    main()
