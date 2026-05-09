"""sigimsae Event Detection 5-Fold Cross-Validation training script.

Runs the event detection model using the **same split method** as
train_kfold.py (classification): per-song sklearn KFold with a fixed seed.

- split key: ``label_file.stem`` (= same as ``meta["file"]`` on the classification side)
- KFold(n_splits=5, shuffle=True, random_state=seed)
- Initialize the model with the same model-seed every fold -> measure only data-split variation

Usage:
    python -m src.event.train_event_kfold --model EDTCN --feature-type f0 --class-scheme 17cat
    python -m src.event.train_event_kfold --model MERTHead --feature-type mert_hidden \\
        --class-scheme 17cat --n-folds 5 --model-seed 123 --seed 42
"""

import argparse
import fcntl
import json
import numpy as np
import torch
import matplotlib
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import KFold

matplotlib.use("Agg")
matplotlib.rcParams['font.family'] = 'NanumGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

from event_dataset import SigimsaeEventDataset, DONT_CARE
from models_event import (
    EVENT_MODEL_REGISTRY, DontCareCrossEntropyLoss, DontCareFocalLoss,
)
from metrics_event import evaluate_model_events, print_event_metrics
from train_event import (
    _align_logits_labels,
    frame_evaluate,
    compute_macro_f1,
    plot_curves,
    plot_confusion_matrix,
)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ─── fully fix seed ────────────────────────────────────────

def set_deterministic(seed: int):
    """Fully fix PyTorch + NumPy + cuDNN seeds."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── per-song K-Fold split (same logic as classification) ─

def make_event_kfold_splits(file_to_chunks, n_folds=5, seed=42):
    """Same method as train_kfold.py::make_song_kfold_splits.

    Sort the keys of ``file_to_chunks`` (=label_file.stem), then use sklearn KFold
    to pick the test fold, and split off the last 1/(n_folds-1) of the remainder as val.
    With the same seed and the same file set, this matches the classification-side split.
    """
    file_names = sorted(file_to_chunks.keys())
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    folds = []
    for fold_idx, (trainval_file_idx, test_file_idx) in enumerate(kf.split(file_names)):
        test_files = set(file_names[i] for i in test_file_idx)
        trainval_files = [file_names[i] for i in trainval_file_idx]
        n_val = max(1, len(trainval_files) // (n_folds - 1))
        val_files = set(trainval_files[-n_val:])
        train_files = set(trainval_files[:-n_val])

        split_indices = {"train": [], "val": [], "test": []}
        for stem in train_files:
            split_indices["train"].extend(file_to_chunks[stem])
        for stem in val_files:
            split_indices["val"].extend(file_to_chunks[stem])
        for stem in test_files:
            split_indices["test"].extend(file_to_chunks[stem])
        for k in split_indices:
            split_indices[k] = sorted(split_indices[k])

        folds.append({
            **split_indices,
            "train_files": train_files,
            "val_files": val_files,
            "test_files": test_files,
        })
    return folds


def apply_fold_to_dataset(ds, fold):
    """Overwrite the split indices of SigimsaeEventDataset with fold."""
    ds._split_indices = {
        "train": fold["train"],
        "val": fold["val"],
        "test": fold["test"],
    }
    ds._split_files = {
        "train": fold["train_files"],
        "val": fold["val_files"],
        "test": fold["test_files"],
    }


# ─── model builder (extracted from train_event.py main()) ─

def build_model(args, ds, device):
    ModelClass = EVENT_MODEL_REGISTRY[args.model]
    n_classes = ds.n_classes
    n_freq = ds.n_freq

    if args.model == "MERTHead":
        extra = {}
        if args.gru_hidden is not None:
            extra["gru_hidden"] = args.gru_hidden
        model = ModelClass(n_classes=n_classes, mert_dim=n_freq, **extra).to(device)
        hop_sec = model.hop_sec
    elif args.model == "EDTCN":
        periodic_pad = 2 if args.feature_type == "chroma" else 0
        model = ModelClass(
            n_freq=n_freq, n_classes=n_classes, periodic_pad=periodic_pad,
        ).to(device)
        hop_sec = ds.hop_sec
    else:
        extra = {}
        if args.gru_hidden is not None:
            extra["gru_hidden"] = args.gru_hidden
        model = ModelClass(n_freq=n_freq, n_classes=n_classes, **extra).to(device)
        hop_sec = ds.hop_sec

    return model, hop_sec


# ─── single fold training ──────────────────────────────────

def train_one_fold(args, ds, fold_idx, fold, model_seed, run_dir, device):
    from sklearn.metrics import (
        f1_score as sk_f1,
        precision_score as sk_precision,
        recall_score as sk_recall,
        confusion_matrix as sk_confusion,
        classification_report as sk_report,
    )

    apply_fold_to_dataset(ds, fold)
    n_classes = ds.n_classes
    class_names = ds.class_names

    # initialize the model identically for each fold
    set_deterministic(model_seed)

    train_loader = ds.get_dataloader("train", batch_size=args.batch_size, shuffle=True)
    val_loader = ds.get_dataloader("val", batch_size=args.batch_size)
    test_loader = ds.get_dataloader("test", batch_size=args.batch_size)

    model, hop_sec = build_model(args, ds, device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    class_weights = ds.get_class_weights("train").to(device)

    if args.loss == "focal":
        criterion = DontCareFocalLoss(gamma=args.focal_gamma, weight=class_weights)
    else:
        criterion = DontCareCrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6,
        )

    fold_dir = run_dir / f"fold{fold_idx+1}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    model_path = fold_dir / "best_model.pt"

    print(f"\n  Fold {fold_idx+1}: "
          f"Train={len(fold['train'])} chunks ({len(fold['train_files'])} files) / "
          f"Val={len(fold['val'])} ({len(fold['val_files'])} files) / "
          f"Test={len(fold['test'])} ({len(fold['test_files'])} files)")
    print(f"  Class weights: {class_weights.cpu().numpy().round(2)}")

    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []
    final_epoch = 0

    for epoch in range(1, args.epochs + 1):
        final_epoch = epoch
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            logits = _align_logits_labels(logits, Y_batch)
            loss = criterion(logits, Y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            mask = Y_batch != DONT_CARE
            if mask.sum() > 0:
                preds = logits.argmax(dim=1)
                train_correct += (preds[mask] == Y_batch[mask]).sum().item()
                train_total += mask.sum().item()
            train_loss += loss.item() * mask.sum().item()

        train_loss = train_loss / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        val_result = frame_evaluate(model, val_loader, criterion, device, n_classes)
        val_f1 = compute_macro_f1(val_result["preds"], val_result["targets"], n_classes)

        if args.scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_f1)

        current_lr = optimizer.param_groups[0]["lr"]
        improved = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            improved = " *"
        else:
            patience_counter += 1

        if epoch % 10 == 0 or improved or patience_counter >= args.patience:
            print(f"    Ep {epoch:3d}: TrLoss={train_loss:.4f} "
                  f"VaLoss={val_result['loss']:.4f} VaAcc={val_result['accuracy']:.4f} "
                  f"VaF1={val_f1:.4f} Best={best_val_f1:.4f} "
                  f"LR={current_lr:.6f} Pat={patience_counter}{improved}")

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_result["loss"], 4),
            "val_acc": round(val_result["accuracy"], 4),
            "val_macro_f1": round(val_f1, 4),
            "lr": round(current_lr, 8),
        })

        if patience_counter >= args.patience:
            print(f"    Early stopping at epoch {epoch}")
            break

    # ─── test evaluation ───────────────────────────────────
    model.load_state_dict(torch.load(model_path, weights_only=True))

    test_result = frame_evaluate(model, test_loader, criterion, device, n_classes)
    targets = test_result["targets"]
    preds = test_result["preds"]

    test_macro_f1 = sk_f1(targets, preds, average="macro", zero_division=0)
    test_weighted_f1 = sk_f1(targets, preds, average="weighted", zero_division=0)
    test_macro_prec = sk_precision(targets, preds, average="macro", zero_division=0)
    test_macro_rec = sk_recall(targets, preds, average="macro", zero_division=0)
    per_class_frame = sk_report(
        targets, preds, target_names=class_names,
        output_dict=True, zero_division=0,
    )
    frame_cm = sk_confusion(targets, preds).tolist()

    event_results = evaluate_model_events(
        model, test_loader, device, hop_sec, n_classes=n_classes,
        median_kernel=args.median_kernel,
        merge_gap_sec=args.merge_gap_sec,
    )

    # per-class event F1 (200ms collar)
    per_class_event = {}
    for cls_idx_str, v in event_results.get("per_class_event", {}).items():
        cls_idx = int(cls_idx_str)
        if cls_idx < len(class_names):
            per_class_event[class_names[cls_idx]] = round(float(v.get("f1", 0)), 4)

    # visualization
    plot_curves(history, best_epoch, best_val_f1, fold_dir / "curves.png")
    plot_confusion_matrix(
        targets, preds, class_names,
        fold_dir / "confusion.png",
        test_acc=test_result["accuracy"], macro_f1=test_macro_f1,
    )

    print(f"    ✓ Fold {fold_idx+1}: Best Ep={best_epoch} "
          f"| FrameAcc={test_result['accuracy']:.4f} FrameF1={test_macro_f1:.4f} "
          f"| Event200ms F1={float(event_results.get('event_200ms', {}).get('f1', 0)):.4f} "
          f"Onset100ms F1={float(event_results.get('onset_100ms', {}).get('f1', 0)):.4f} "
          f"IoU0.5 F1={float(event_results.get('iou_0.5', {}).get('f1', 0)):.4f}")

    fold_result = {
        "fold": fold_idx + 1,
        "best_epoch": best_epoch,
        "epochs_run": final_epoch,
        "best_val_macro_f1": round(best_val_f1, 4),
        "test_frame_accuracy": round(test_result["accuracy"], 4),
        "test_frame_macro_f1": round(float(test_macro_f1), 4),
        "test_frame_weighted_f1": round(float(test_weighted_f1), 4),
        "test_frame_macro_precision": round(float(test_macro_prec), 4),
        "test_frame_macro_recall": round(float(test_macro_rec), 4),
        "event_f1_200ms": round(float(event_results.get("event_200ms", {}).get("f1", 0)), 4),
        "event_f1_200ms_agnostic": round(float(
            event_results.get("event_200ms_agnostic", {}).get("f1", 0)), 4),
        "event_f1_100ms": round(float(event_results.get("event_100ms", {}).get("f1", 0)), 4),
        "onset_f1_100ms": round(float(event_results.get("onset_100ms", {}).get("f1", 0)), 4),
        "onset_f1_50ms": round(float(event_results.get("onset_50ms", {}).get("f1", 0)), 4),
        "iou_f1_0.5": round(float(event_results.get("iou_0.5", {}).get("f1", 0)), 4),
        "iou_f1_0.5_agnostic": round(float(
            event_results.get("iou_0.5_agnostic", {}).get("f1", 0)), 4),
        "iou_f1_0.3": round(float(event_results.get("iou_0.3", {}).get("f1", 0)), 4),
        "n_pred_events": int(event_results.get("n_pred_events", 0)),
        "n_ref_events": int(event_results.get("n_ref_events", 0)),
        "per_class_frame": {
            cls_name: {
                "precision": round(per_class_frame[cls_name]["precision"], 4),
                "recall": round(per_class_frame[cls_name]["recall"], 4),
                "f1": round(per_class_frame[cls_name]["f1-score"], 4),
                "support": per_class_frame[cls_name]["support"],
            }
            for cls_name in class_names if cls_name in per_class_frame
        },
        "per_class_event_f1_200ms": per_class_event,
        "frame_confusion_matrix": frame_cm,
        "n_train": len(fold["train"]),
        "n_val": len(fold["val"]),
        "n_test": len(fold["test"]),
        "n_train_files": len(fold["train_files"]),
        "n_val_files": len(fold["val_files"]),
        "n_test_files": len(fold["test_files"]),
    }

    with open(fold_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with open(fold_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(fold_result, f, ensure_ascii=False, indent=2)

    # also save the full event_results (including per_class_event)
    def _serialize(obj):
        if isinstance(obj, dict):
            return {str(k): _serialize(v) for k, v in obj.items()}
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return round(float(obj), 6)
        if isinstance(obj, float):
            return round(obj, 6)
        return obj

    with open(fold_dir / "event_metrics.json", "w", encoding="utf-8") as f:
        json.dump(_serialize(event_results), f, ensure_ascii=False, indent=2)

    return fold_result, param_count, hop_sec


# ─── result aggregation ────────────────────────────────────

def aggregate_results(fold_results, class_names):
    metrics = [
        "test_frame_accuracy",
        "test_frame_macro_f1",
        "test_frame_weighted_f1",
        "test_frame_macro_precision",
        "test_frame_macro_recall",
        "event_f1_200ms",
        "event_f1_200ms_agnostic",
        "event_f1_100ms",
        "onset_f1_100ms",
        "onset_f1_50ms",
        "iou_f1_0.5",
        "iou_f1_0.5_agnostic",
        "iou_f1_0.3",
    ]

    agg = {}
    for m in metrics:
        values = [r[m] for r in fold_results]
        agg[m] = {
            "mean": round(float(np.mean(values)), 4),
            "std": round(float(np.std(values)), 4),
            "values": values,
        }

    # per-class frame aggregation
    per_class_frame = {}
    for cls_name in class_names:
        cls_agg = {}
        for metric in ["precision", "recall", "f1"]:
            values = [r["per_class_frame"][cls_name][metric]
                      for r in fold_results if cls_name in r["per_class_frame"]]
            if values:
                cls_agg[metric] = {
                    "mean": round(float(np.mean(values)), 4),
                    "std": round(float(np.std(values)), 4),
                    "values": values,
                }
        if cls_agg:
            per_class_frame[cls_name] = cls_agg
    agg["per_class_frame"] = per_class_frame

    # per-class event aggregation
    per_class_event = {}
    for cls_name in class_names:
        values = [r["per_class_event_f1_200ms"][cls_name]
                  for r in fold_results
                  if cls_name in r.get("per_class_event_f1_200ms", {})]
        if values:
            per_class_event[cls_name] = {
                "mean": round(float(np.mean(values)), 4),
                "std": round(float(np.std(values)), 4),
                "values": values,
            }
    agg["per_class_event_f1_200ms"] = per_class_event

    return agg


def print_summary(agg, class_names, model_name, feature_type, scheme_tag, n_folds):
    print(f"\n{'='*72}")
    print(f"  Event Detection — {model_name} / {feature_type} / {scheme_tag}")
    print(f"  {n_folds}-Fold Cross-Validation Results")
    print(f"{'='*72}")

    print(f"\n  Frame-level Metrics (mean ± std):")
    for m in ["test_frame_accuracy", "test_frame_macro_f1",
              "test_frame_weighted_f1",
              "test_frame_macro_precision", "test_frame_macro_recall"]:
        label = m.replace("test_frame_", "").replace("_", " ").title()
        mean = agg[m]["mean"]; std = agg[m]["std"]; vals = agg[m]["values"]
        print(f"    {label:24s}: {mean:.4f} ± {std:.4f}  {vals}")

    print(f"\n  Event-level Metrics (mean ± std):")
    for m in ["event_f1_200ms", "event_f1_200ms_agnostic", "event_f1_100ms",
              "onset_f1_100ms", "onset_f1_50ms",
              "iou_f1_0.5", "iou_f1_0.5_agnostic", "iou_f1_0.3"]:
        label = m.replace("_", " ").title()
        mean = agg[m]["mean"]; std = agg[m]["std"]; vals = agg[m]["values"]
        print(f"    {label:24s}: {mean:.4f} ± {std:.4f}  {vals}")

    print(f"\n  Per-class Frame F1 (mean ± std):")
    print(f"    {'Class':16s}  {'Precision':>14s}  {'Recall':>14s}  {'F1':>14s}")
    print(f"    {'-'*62}")
    for cls_name in class_names:
        if cls_name not in agg["per_class_frame"]:
            continue
        ca = agg["per_class_frame"][cls_name]
        p = f"{ca['precision']['mean']:.4f}±{ca['precision']['std']:.4f}"
        r = f"{ca['recall']['mean']:.4f}±{ca['recall']['std']:.4f}"
        f = f"{ca['f1']['mean']:.4f}±{ca['f1']['std']:.4f}"
        print(f"    {cls_name:16s}  {p:>14s}  {r:>14s}  {f:>14s}")

    if agg.get("per_class_event_f1_200ms"):
        print(f"\n  Per-class Event F1 @200ms (mean ± std):")
        for cls_name in class_names:
            if cls_name not in agg["per_class_event_f1_200ms"]:
                continue
            ca = agg["per_class_event_f1_200ms"][cls_name]
            print(f"    {cls_name:16s}  {ca['mean']:.4f} ± {ca['std']:.4f}")

    print(f"{'='*72}")


# ─── Main ──────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="sigimsae Event Detection K-Fold CV")

    # model & feature
    p.add_argument("--model", type=str, default="EDTCN",
                   choices=list(EVENT_MODEL_REGISTRY.keys()))
    p.add_argument("--feature-type", type=str, default="chroma",
                   choices=["chroma", "mel", "waveform", "mert_hidden", "f0", "mel_f0"])
    p.add_argument("--feature-dir", type=str, default=None)
    p.add_argument("--class-scheme", type=str, default=None,
                   choices=["7cat", "9cat", "17cat", "total"])
    p.add_argument("--label-dir", type=str, default="precheck_sigimsae_labels")
    p.add_argument("--chunk-sec", type=float, default=10.0)
    p.add_argument("--no-dont-care", action="store_true")
    p.add_argument("--jitter-sec", type=float, default=0.5)

    # training
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--scheduler", type=str, default="cosine",
                   choices=["cosine", "plateau"])
    p.add_argument("--loss", type=str, default="focal", choices=["ce", "focal"])
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--gru-hidden", type=int, default=None)

    # post-processing
    p.add_argument("--median-kernel", type=int, default=11)
    p.add_argument("--merge-gap-sec", type=float, default=0.05)

    # K-Fold
    p.add_argument("--n-folds", type=int, default=5,
                   help="number of K-Folds (default: 5)")
    p.add_argument("--seed", type=int, default=42,
                   help="seed for data K-Fold split (same as classification, default: 42)")
    p.add_argument("--model-seed", type=int, default=123,
                   help="model initialization seed (same for all folds, default: 123)")

    # wandb
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", type=str,
                   default="sigimsae-event-detection-kfold")

    return p.parse_args()


def main():
    args = parse_args()

    DEVICE = (
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"Model: {args.model}  |  Feature: {args.feature_type}  |  "
          f"Class: {args.class_scheme or '9cat(legacy)'}")
    print(f"K-Fold: {args.n_folds}  |  Data seed: {args.seed}  |  "
          f"Model seed: {args.model_seed}")
    print(f"Device: {DEVICE}")

    # ─── load dataset (only once) ─────────────────────────
    print("\nLoading dataset...")
    use_dont_care = not args.no_dont_care

    # the initial split will be overwritten by kfold, so use default ratio
    ds = SigimsaeEventDataset(
        label_dir=args.label_dir,
        feature_type=args.feature_type,
        feature_dir=args.feature_dir,
        class_scheme=args.class_scheme,
        chunk_sec=args.chunk_sec,
        seed=args.seed,
        use_dont_care=use_dont_care,
        jitter_sec=args.jitter_sec,
    )
    ds.summary()

    class_names = ds.class_names
    n_freq = ds.n_freq
    n_classes = ds.n_classes
    n_files = len(ds.file_to_chunks)
    n_chunks = len(ds.chunks)

    print(f"\nTotal chunks: {n_chunks}  |  Files: {n_files}")

    # ─── K-Fold split (same logic as classification) ──────
    folds = make_event_kfold_splits(
        ds.file_to_chunks, n_folds=args.n_folds, seed=args.seed,
    )

    # ─── run directory ────────────────────────────────────
    RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
    scheme_tag = args.class_scheme or "9cat"
    dc_tag = "dc" if use_dont_care else "nodc"
    RUN_NAME = (
        f"event_{args.model}_{args.feature_type}_{scheme_tag}_{dc_tag}"
        f"_{args.n_folds}fold_{RUN_TS}"
    )
    run_dir = Path(f"models/{RUN_NAME}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run: {run_dir}")

    # ─── wandb ────────────────────────────────────────────
    use_wandb = HAS_WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=RUN_NAME,
            config={
                **vars(args),
                "n_freq": n_freq,
                "n_classes": n_classes,
                "n_files": n_files,
                "n_chunks": n_chunks,
                "device": DEVICE,
            },
        )

    # ─── K-Fold training ──────────────────────────────────
    fold_results = []
    param_count = None
    hop_sec = None

    for fold_idx, fold in enumerate(folds):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold_idx+1}/{args.n_folds}")
        print(f"{'='*60}")

        result, param_count, hop_sec = train_one_fold(
            args, ds, fold_idx, fold,
            model_seed=args.model_seed,
            run_dir=run_dir,
            device=DEVICE,
        )
        fold_results.append(result)

        if use_wandb:
            log_dict = {
                f"fold{fold_idx+1}/frame_accuracy": result["test_frame_accuracy"],
                f"fold{fold_idx+1}/frame_macro_f1": result["test_frame_macro_f1"],
                f"fold{fold_idx+1}/event_f1_200ms": result["event_f1_200ms"],
                f"fold{fold_idx+1}/onset_f1_100ms": result["onset_f1_100ms"],
                f"fold{fold_idx+1}/iou_f1_0.5": result["iou_f1_0.5"],
                f"fold{fold_idx+1}/best_epoch": result["best_epoch"],
            }
            wandb.log(log_dict)

    # ─── aggregation ──────────────────────────────────────
    agg = aggregate_results(fold_results, class_names)
    print_summary(agg, class_names, args.model, args.feature_type,
                   scheme_tag, args.n_folds)

    full_results = {
        "model": args.model,
        "feature_type": args.feature_type,
        "feature_dir": args.feature_dir,
        "class_scheme": scheme_tag,
        "n_classes": n_classes,
        "n_freq": n_freq,
        "param_count": param_count,
        "hop_sec": round(float(hop_sec), 6) if hop_sec is not None else None,
        "use_dont_care": use_dont_care,
        "jitter_sec": args.jitter_sec,
        "median_kernel": args.median_kernel,
        "merge_gap_sec": args.merge_gap_sec,
        "n_folds": args.n_folds,
        "data_seed": args.seed,
        "model_seed": args.model_seed,
        "class_names": class_names,
        "aggregate": agg,
        "folds": fold_results,
        "run_name": RUN_NAME,
        "timestamp": RUN_TS,
    }

    results_path = run_dir / "kfold_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved: {results_path}")

    # append to the consolidated results file (safe for parallel runs: lockfile + flock)
    CV_RESULTS_PATH = Path("results/event_kfold_results.json")
    CV_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_path = CV_RESULTS_PATH.with_suffix(".lock")
    with open(lock_path, "w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        all_cv = []
        if CV_RESULTS_PATH.exists():
            with open(CV_RESULTS_PATH, encoding="utf-8") as f:
                all_cv = json.load(f)
        all_cv.append(full_results)
        with open(CV_RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(all_cv, f, ensure_ascii=False, indent=2)
    print(f"Consolidated CV results: {CV_RESULTS_PATH} ({len(all_cv)} experiments)")

    # wandb final
    if use_wandb:
        for m in ["test_frame_accuracy", "test_frame_macro_f1",
                  "test_frame_weighted_f1",
                  "event_f1_200ms", "event_f1_200ms_agnostic",
                  "onset_f1_100ms", "onset_f1_50ms",
                  "iou_f1_0.5", "iou_f1_0.5_agnostic"]:
            wandb.summary[f"{m}_mean"] = agg[m]["mean"]
            wandb.summary[f"{m}_std"] = agg[m]["std"]
        wandb.finish()

    print(f"\nDone! All fold results: {run_dir}")


if __name__ == "__main__":
    main()
