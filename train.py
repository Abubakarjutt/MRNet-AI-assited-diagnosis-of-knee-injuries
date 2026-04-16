import argparse
import json
import os
import shutil
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics

try:
    from tensorboardX import SummaryWriter
except ImportError:
    class SummaryWriter:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass

import advanced_vit
import lightweight_models
import utils
from dataloader import MRMultiPlaneDataset, MRVolumeAugmentor


torch.set_float32_matmul_precision("high")


def safe_confusion_counts(y_true, y_pred):
    rounded_preds = np.array(y_pred).round()
    if len(np.unique(y_true)) < 2:
        return "n/a (single class observed in sampled labels)"

    confusion = metrics.confusion_matrix(y_true, rounded_preds, labels=[0, 1])
    if confusion.size != 4:
        return "n/a (confusion matrix shape mismatch)"

    tn, fp, fn, tp = confusion.ravel()
    return f"tn={tn} fp={fp} fn={fn} tp={tp}"


def maybe_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def build_dataloader(dataset, shuffle, args):
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "drop_last": False,
    }

    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    if torch.cuda.is_available():
        loader_kwargs["pin_memory"] = True

    return torch.utils.data.DataLoader(dataset, **loader_kwargs)




def build_train_augmentor(args):
    return MRVolumeAugmentor(
        policy=args.aug_policy,
        noise_std=args.aug_noise_std,
        cutout_frac=args.aug_cutout_frac,
        slice_dropout=args.aug_slice_dropout,
    )

def prepare_inputs(volumes, device, args):
    channels_last = bool(args.channels_last)
    sagittal, coronal, axial = (
        utils.prepare_volume_batch(
            volume,
            device=device,
            image_size=args.image_size,
            channels_last=channels_last,
        )
        for volume in volumes
    )
    return sagittal, coronal, axial


def compute_auc(y_true, y_pred):
    try:
        if len(np.unique(y_true)) > 1:
            return metrics.roc_auc_score(y_true, y_pred)
    except ValueError:
        pass
    return 0.5


def iterate_epoch(
    model,
    loader,
    criterion,
    optimizer,
    writer,
    epoch,
    num_epochs,
    current_lr,
    device,
    args,
    is_train,
    global_step,
):
    if is_train:
        model.train()
    else:
        model.eval()

    y_preds = []
    y_trues = []
    losses = []
    phase = "Train" if is_train else "Val"
    max_batches = args.max_train_batches if is_train else args.max_val_batches
    amp_context = utils.get_amp_context(device, enabled=bool(args.amp))

    iterator = enumerate(loader)
    for batch_index, (volumes, label, _, _) in iterator:
        if max_batches is not None and batch_index >= max_batches:
            break

        label = label.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
        sagittal, coronal, axial = prepare_inputs(volumes, device, args)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with amp_context:
                prediction = model(sagittal, coronal, axial)
                loss = criterion(prediction, label)

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        probas = torch.sigmoid(prediction).detach().cpu().numpy().reshape(-1)
        truth = label.detach().cpu().numpy().reshape(-1)

        y_preds.extend(probas.tolist())
        y_trues.extend(truth.astype(int).tolist())
        losses.append(loss.item())
        auc = compute_auc(y_trues, y_preds)

        writer.add_scalar(f"{phase}/Loss", loss.item(), global_step)
        writer.add_scalar(f"{phase}/AUC", auc, global_step)

        if batch_index > 0 and batch_index % args.log_every == 0:
            print(
                f"[Epoch: {epoch + 1} / {num_epochs} | batch: {batch_index} / {len(loader)}] "
                f"| avg {phase.lower()} loss {np.mean(losses):.4f} "
                f"| {phase.lower()} auc: {auc:.4f} | lr: {current_lr:.6g}"
            )

        global_step += 1

    epoch_loss = float(np.mean(losses)) if losses else float("nan")
    epoch_auc = compute_auc(y_trues, y_preds) if y_preds else float("nan")
    return epoch_loss, epoch_auc, y_trues, y_preds, global_step


def build_model(args):
    if args.model_type == "advanced":
        model = advanced_vit.AdvancedMRNetViT(
            num_classes=3,
            model_name=args.vit_model,
            pretrained=bool(args.pretrained),
        )
    elif args.model_type == "multiscale":
        model = advanced_vit.MultiScaleMRNetViT(
            num_classes=3,
            pretrained=bool(args.pretrained),
        )
    else:
        backbone_name = "resnet18" if args.model_type == "basic" else args.model_type
        model = lightweight_models.FastMRNet(
            backbone_name=backbone_name,
            num_classes=3,
            pretrained=bool(args.pretrained),
            dropout=args.dropout,
            pooling=args.pooling,
            projection_dim=args.projection_dim,
            hidden_dim=args.hidden_dim,
            fusion_depth=args.fusion_depth,
            fusion_gate=args.fusion_gate,
        )

    return model


def maybe_load_search_config(args):
    if not args.search_config:
        return args

    with open(args.search_config, "r") as handle:
        config = json.load(handle)

    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)

    return args


def maybe_load_init_checkpoint(model, checkpoint_path):
    if not checkpoint_path:
        return

    if not os.path.isfile(checkpoint_path):
        print(f"Skipping init checkpoint because it does not exist: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    if not isinstance(checkpoint, dict):
        print(f"Skipping init checkpoint because it is not a state_dict: {checkpoint_path}")
        return

    model_state = model.state_dict()
    compatible = {}
    skipped_shape = []

    for key, value in checkpoint.items():
        if key not in model_state:
            continue
        if model_state[key].shape != value.shape:
            skipped_shape.append(key)
            continue
        compatible[key] = value

    if not compatible:
        print(f"No compatible tensors found in init checkpoint: {checkpoint_path}")
        return

    load_result = model.load_state_dict(compatible, strict=False)
    print(
        f"Warm-started from {checkpoint_path} with {len(compatible)} tensors; "
        f"missing={len(load_result.missing_keys)} "
        f"unexpected={len(load_result.unexpected_keys)} "
        f"shape_mismatch={len(skipped_shape)}"
    )


def run(args):
    if args.batch_size != 1:
        raise ValueError("MRNet training currently expects --batch_size 1 because slice counts vary by exam.")

    device = utils.get_device()
    print(f"Using device: {device.type}")

    log_root_folder = f"./logs/{args.task}/{args.plane}/"
    os.makedirs(log_root_folder, exist_ok=True)

    if args.flush_history == 1:
        for item in os.listdir(log_root_folder):
            item_path = os.path.join(log_root_folder, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)

    now = datetime.now()
    run_name = args.prefix_name or now.strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(log_root_folder, now.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    data_root = args.data_root.rstrip("/") + "/"
    train_dataset = MRMultiPlaneDataset(
        data_root,
        train=True,
        mmap=bool(args.mmap),
        cache_size=args.cache_size,
        transform=build_train_augmentor(args),
    )
    val_dataset = MRMultiPlaneDataset(
        data_root,
        train=False,
        mmap=bool(args.mmap),
        cache_size=max(1, args.cache_size // 2),
    )

    train_loader = build_dataloader(train_dataset, shuffle=True, args=args)
    val_loader = build_dataloader(val_dataset, shuffle=False, args=args)

    model = build_model(args)
    maybe_load_init_checkpoint(model, args.init_checkpoint)
    model = utils.maybe_channels_last(model, device)
    model.to(device)

     # Log model architecture info for tracking
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model_type}, Parameters: {num_params/1e6:.2f}M, Trainable: {trainable_params/1e6:.2f}M")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        factor=0.3,
        threshold=1e-4,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=train_dataset.weights.to(device))

    best_val_loss = float("inf")
    best_val_auc = 0.0
    best_epoch = -1
    patience_counter = 0
    global_train_step = 0
    global_val_step = 0
    model_output_dir = os.path.abspath(os.environ.get("MRNET_MODEL_DIR", "models"))

    num_epochs = args.epochs
    patience = args.patience
    t_start_training = time.time()
    avg_epoch_seconds = []

    for epoch in range(num_epochs):
        elapsed_minutes = (time.time() - t_start_training) / 60.0
        if args.time_budget_minutes is not None and elapsed_minutes >= args.time_budget_minutes:
            print(f"Stopping early after reaching the {args.time_budget_minutes:.2f} minute budget.")
            break

        current_lr = utils.get_lr(optimizer)
        maybe_sync(device)
        t_start_epoch = time.time()

        train_loss, train_auc, train_y_trues, train_y_preds, global_train_step = iterate_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            writer=writer,
            epoch=epoch,
            num_epochs=num_epochs,
            current_lr=current_lr,
            device=device,
            args=args,
            is_train=True,
            global_step=global_train_step,
        )

        val_loss, val_auc, val_y_trues, val_y_preds, global_val_step = iterate_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            writer=writer,
            epoch=epoch,
            num_epochs=num_epochs,
            current_lr=current_lr,
            device=device,
            args=args,
            is_train=False,
            global_step=global_val_step,
        )

        scheduler.step(val_loss)

        maybe_sync(device)
        epoch_seconds = time.time() - t_start_epoch
        avg_epoch_seconds.append(epoch_seconds)

        print(
            f"train loss: {train_loss:.4f} | train auc: {train_auc:.4f} "
            f"| val loss: {val_loss:.4f} | val auc: {val_auc:.4f} "
            f"| elapsed time: {epoch_seconds:.2f} s"
        )
        print("train confusion:", safe_confusion_counts(train_y_trues, train_y_preds))
        print("val confusion:", safe_confusion_counts(val_y_trues, val_y_preds))
        print("-" * 30)

        if val_auc > best_val_auc:
            best_val_auc = float(val_auc)
            best_epoch = epoch + 1
            if bool(args.save_model):
                os.makedirs(model_output_dir, exist_ok=True)
                file_name = (
                    f"model_{run_name}_{args.model_type}_val_auc_{val_auc:0.4f}_"
                    f"train_auc_{train_auc:0.4f}_epoch_{epoch + 1}.pth"
                )
                for existing_file in os.listdir(model_output_dir):
                    if run_name in existing_file:
                        os.remove(os.path.join(model_output_dir, existing_file))
                torch.save(model.state_dict(), os.path.join(model_output_dir, file_name))
                print(f"Best model saved with validation AUC: {val_auc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping after {patience_counter} epochs without validation loss improvement."
                )
                break

    maybe_sync(device)
    training_seconds = time.time() - t_start_training
    writer.close()

    num_params = sum(parameter.numel() for parameter in model.parameters())
    avg_epoch_time = float(np.mean(avg_epoch_seconds)) if avg_epoch_seconds else float("nan")

     # Compute model complexity score for autoresearch tracking
    model_complexity = 0.0
    if args.model_type == "mobilenet_v3_small":
        model_complexity = 0.0
    elif args.model_type == "resnet18":
        model_complexity = 0.2
    elif args.model_type == "efficientnet_b0":
        model_complexity = 0.4

    pooling_complexity = {"max": 0.0, "mean": 0.0, "lse": 0.2, "gem": 0.3, "attention": 0.5}.get(args.pooling, 0.3)
    augmentation_complexity = {"none": 0.0, "light": 0.05, "strong": 0.12, "knee_mri": 0.18}.get(args.aug_policy, 0.08)
    fusion_complexity = 0.2 * max(int(args.fusion_depth) - 1, 0)
    gate_complexity = 0.15 * (1 if args.fusion_gate == "se" else 0)
    total_complexity = model_complexity + pooling_complexity + fusion_complexity + gate_complexity + augmentation_complexity

    print("---")
    print(f"best_val_auc:       {best_val_auc:.6f}")
    print(f"best_val_loss:      {best_val_loss:.6f}")
    print(f"training_seconds:   {training_seconds:.2f}")
    print(f"avg_epoch_seconds:  {avg_epoch_time:.2f}")
    print(f"epochs_ran:         {len(avg_epoch_seconds)}")
    print(f"best_epoch:         {best_epoch}")
    print(f"num_params_M:       {num_params / 1e6:.2f}")
    print(f"model_type:         {args.model_type}")
    print(f"model_complexity:   {total_complexity:.2f}")
    print(f"device:             {device.type}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        choices=["abnormal", "acl", "meniscus"],
        default="acl",
    )
    parser.add_argument(
        "-p",
        "--plane",
        type=str,
        choices=["Segittal_Coronal_and_Axial"],
        default="Segittal_Coronal_and_Axial",
    )
    parser.add_argument("--prefix_name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--projection_dim", type=int, default=0)
    parser.add_argument("--fusion_depth", type=int, default=2)
    parser.add_argument(
        "--pooling",
        type=str,
        default="max",
        choices=["max", "mean", "lse", "attention", "gem"],
        help="Slice aggregation strategy for the lightweight searchable models.",
    )
    parser.add_argument(
        "--fusion_gate",
        type=str,
        default="none",
        choices=["none", "se"],
        help="Optional fusion recalibration block for research-driven mutations.",
    )
    parser.add_argument("--flush_history", type=int, choices=[0, 1], default=0)
    parser.add_argument("--save_model", type=int, choices=[0, 1], default=1)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument(
        "--model_type",
        type=str,
        default="resnet18",
        choices=[
            "basic",
            "advanced",
            "multiscale",
            "resnet18",
            "mobilenet_v3_small",
            "efficientnet_b0",
        ],
        help="Model family to use. Lighter CNNs are much faster than the ViT variants.",
    )
    parser.add_argument(
        "--vit_model",
        type=str,
        default="vit_b_16",
        choices=["vit_b_16", "vit_l_16", "vit_h_14"],
        help="Only used when model_type=advanced.",
    )
    parser.add_argument(
        "--pretrained",
        type=int,
        choices=[0, 1],
        default=1,
        help="Use ImageNet pretrained weights.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="MRNet-v1.0",
        help="Path to the MRNet dataset root folder.",
    )
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--cache_size", type=int, default=32)
    parser.add_argument(
        "--aug_policy",
        type=str,
        default="none",
        choices=["none", "light", "strong", "knee_mri"],
        help="Training-time augmentation policy searched by the autoresearch loop.",
    )
    parser.add_argument("--aug_noise_std", type=float, default=0.0)
    parser.add_argument("--aug_cutout_frac", type=float, default=0.0)
    parser.add_argument("--aug_slice_dropout", type=float, default=0.0)
    parser.add_argument("--mmap", type=int, choices=[0, 1], default=1)
    parser.add_argument("--amp", type=int, choices=[0, 1], default=1)
    parser.add_argument("--channels_last", type=int, choices=[0, 1], default=1)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument(
        "--time_budget_minutes",
        type=float,
        default=None,
        help="Optional wall-clock budget for experiment runs.",
    )
    parser.add_argument(
        "--search_config",
        type=str,
        default=None,
        help="Optional JSON file containing an autoresearch-selected model/training configuration.",
    )
    parser.add_argument(
        "--init_checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path used to warm-start compatible layers.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_arguments()
    maybe_load_search_config(parsed_args)
    run(parsed_args)
