"""Feature-bank training loop for MRNet Path 3 (spec §6).

Isolated from train.iterate_epoch (which is volume-tensor shaped). Reuses
train.ModelEMA / compute_auc / compute_loss and the checkpoint-naming convention.
"""

import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

import utils
from dataloader import FeatureCacheDataset, featbank_collate, TASKS
from lightweight_models import FeatBankMRNet
from train import ModelEMA, compute_auc, compute_loss


def make_folds(n, k, seed):
    order = np.random.default_rng(seed).permutation(n)
    return [np.sort(chunk) for chunk in np.array_split(order, k)]


def _g(args, name, default=None):
    return getattr(args, name, default)


def _move(payload, device):
    return {
        key: {e: {p: t.to(device) for p, t in planes.items()}
              for e, planes in payload[key].items()}
        for key in ("pooled", "patch")
    }


def _select_value(metric, val_loss, pooled_auc, per_task_mean):
    if metric == "loss":
        return -val_loss                       # higher is better everywhere
    if metric == "pooled_auc":
        return pooled_auc
    return per_task_mean


def run_featbank(args):
    device = utils.get_device()
    variants = _g(args, "cache_variants", "clean").split(",")
    encoders = _g(args, "encoders", "fake").split(",")

    full = FeatureCacheDataset(
        _g(args, "feature_cache"), "train", encoders=encoders, variants=variants,
        data_root=_g(args, "data_root"), slices_used=_g(args, "slices_used", 24),
        slices_used_meniscus=_g(args, "slices_used_meniscus", 32),
        train=True, seed=_g(args, "seed", 0),
    )
    val_variants = _g(args, "eval_tta_variants", "clean").split(",")

    if _g(args, "cv_folds", 0) and _g(args, "cv_fold", -1) >= 0:
        folds = make_folds(len(full), args.cv_folds, _g(args, "seed", 0))
        val_idx = folds[args.cv_fold]
        train_idx = np.concatenate([f for j, f in enumerate(folds) if j != args.cv_fold])
        train_ds = Subset(full, train_idx.tolist())
        val_ds = Subset(
            FeatureCacheDataset(_g(args, "feature_cache"), "train", encoders=encoders,
                                variants=val_variants, data_root=_g(args, "data_root"),
                                slices_used=_g(args, "slices_used", 24),
                                slices_used_meniscus=_g(args, "slices_used_meniscus", 32),
                                train=False, seed=_g(args, "seed", 0)),
            val_idx.tolist(),
        )
        weights = full.weights
    else:
        train_ds = full
        val_ds = FeatureCacheDataset(
            _g(args, "feature_cache"), "valid", encoders=encoders, variants=val_variants,
            data_root=_g(args, "data_root"), slices_used=_g(args, "slices_used", 24),
            slices_used_meniscus=_g(args, "slices_used_meniscus", 32),
            train=False, seed=_g(args, "seed", 0),
        )
        weights = full.weights

    bs = _g(args, "featbank_batch_size", 16)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=featbank_collate)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=featbank_collate)

    model = FeatBankMRNet(full.encoder_dims, full.patch_dims,
                          d_model=_g(args, "d_model", 256),
                          dropout=_g(args, "dropout", 0.15)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=_g(args, "lr", 3e-4),
                                  weight_decay=_g(args, "weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.3)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights.to(device))
    ema = ModelEMA(model, args.ema_decay) if _g(args, "ema_decay", 0.0) > 0 else None

    best_select = -float("inf")
    best = {"per_task_auc": [0.0, 0.0, 0.0], "per_task_mean": 0.0,
            "pooled_auc": 0.0, "best_epoch": 0}
    patience = _g(args, "patience", 0)
    bad_epochs = 0
    model_dir = os.path.abspath(os.environ.get("MRNET_MODEL_DIR", "models"))
    t0 = time.time()

    for epoch in range(_g(args, "epochs", 30)):
        budget = _g(args, "time_budget_minutes", None)
        if budget is not None and (time.time() - t0) / 60.0 >= budget:
            break
        if isinstance(train_ds, Subset):
            train_ds.dataset.set_epoch(epoch)
        else:
            train_ds.set_epoch(epoch)

        model.train()
        for payload, labels, _w, _ids in train_loader:
            payload = _move(payload, device)
            labels = labels.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits = model(payload)
            loss = compute_loss(logits, labels, criterion, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if ema is not None:
                ema.update(model)

        if ema is not None:
            ema.apply_to(model)
        model.eval()
        yt, yp, losses = [], [], []
        with torch.no_grad():
            for payload, labels, _w, _ids in val_loader:
                payload = _move(payload, device)
                labels_d = labels.to(device=device, dtype=torch.float32)
                logits = model(payload)
                losses.append(compute_loss(logits, labels_d, criterion, args).item())
                yp.append(torch.sigmoid(logits).cpu().numpy())
                yt.append(labels.numpy())
        if ema is not None:
            ema.restore(model)

        yt = np.concatenate(yt, axis=0)
        yp = np.concatenate(yp, axis=0)
        val_loss = float(np.mean(losses)) if losses else float("nan")
        per_task = [
            roc_auc_score(yt[:, k], yp[:, k]) if len(np.unique(yt[:, k])) > 1 else 0.5
            for k in range(3)
        ]
        per_task_mean = float(np.mean(per_task))
        pooled_auc = compute_auc(yt.reshape(-1).astype(int).tolist(), yp.reshape(-1).tolist())
        scheduler.step(val_loss)

        print(f"[featbank epoch {epoch + 1}] val_loss {val_loss:.4f} "
              f"| abnormal {per_task[0]:.4f} acl {per_task[1]:.4f} meniscus {per_task[2]:.4f} "
              f"| per_task_mean {per_task_mean:.4f} | pooled {pooled_auc:.4f}")

        select = _select_value(_g(args, "select_metric", "per_task_mean"),
                               val_loss, pooled_auc, per_task_mean)
        if select > best_select:
            best_select = select
            best = {"per_task_auc": per_task, "per_task_mean": per_task_mean,
                    "pooled_auc": pooled_auc, "best_epoch": epoch + 1}
            bad_epochs = 0
            if _g(args, "save_model", 0):
                prefix = _g(args, "prefix_name", "featbank")
                os.makedirs(model_dir, exist_ok=True)
                fname = f"model_{prefix}_featbank_ptmean_{per_task_mean:.4f}_epoch_{epoch + 1}.pth"
                for existing in os.listdir(model_dir):
                    if prefix in existing:
                        os.remove(os.path.join(model_dir, existing))
                torch.save(model.state_dict(), os.path.join(model_dir, fname))
        else:
            bad_epochs += 1
            if patience and bad_epochs >= patience:
                break

    print(f"best_val_per_task_mean: {best['per_task_mean']:.6f}")
    print(f"best_val_per_task:      {best['per_task_auc']}")
    print(f"best_val_pooled_auc:    {best['pooled_auc']:.6f}")
    return best
