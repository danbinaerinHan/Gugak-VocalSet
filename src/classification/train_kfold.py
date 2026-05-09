"""
sigimsae classification 5-Fold Cross-Validation training script

Splits the data using Song-level Stratified K-Fold and fixes the model seed
so that only variation due to data splitting is measured.

Usage:
    python train_kfold.py --config configs/b2_f0cnn.yaml
    python train_kfold.py --config configs/b3_melcnn.yaml
    python train_kfold.py --config configs/b4_resnet18.yaml
    python train_kfold.py --config configs/b5_mert_emb.yaml
    python train_kfold.py --config configs/b2_f0cnn.yaml --n-folds 5 --model-seed 123
"""

import argparse
import json
import numpy as np
import torch
import yaml
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from datetime import datetime
from collections import Counter
from sklearn.model_selection import KFold

matplotlib.use("Agg")
matplotlib.rcParams['font.family'] = 'NanumGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

import wandb

from dataset import SigimsaeF0Dataset, SigimsaeMelDataset, SigimsaeEmbeddingDataset, SigimsaeChromaDataset
from prepare_f0_dataset import CLASS_SCHEMES
from models import (
    MODEL_REGISTRY,
    MODEL_CONFIGS,
    FocalLoss,
    compute_class_weights, compute_macro_f1,
    evaluate, print_classification_report,
)
from train import load_config, plot_confusion_matrix, plot_curves


# ─── full seed fixing ─────────────────────────────────────

def set_deterministic(seed: int):
    """Fully fix the PyTorch + NumPy + cuDNN seeds."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── genre extraction helper ─────────────────────────────────────

def _get_genre(file_stem: str) -> str:
    """Extract the genre (genreMajor) from a filename. e.g., '001_정악_풍류음악' → '정악'
    Normalize to NFC because the filesystem may use NFD encoding."""
    import unicodedata
    return unicodedata.normalize("NFC", file_stem.split("_")[1])


# ─── song-level K-Fold split ────────────────────────────────

def make_song_kfold_splits(meta, n_folds=5, seed=42):
    """Generate a song (file) level K-Fold split.

    Returns:
        list of dicts: [{
            "train": [indices], "val": [indices], "test": [indices],
            "train_files": set, "val_files": set, "test_files": set,
        }, ...]
    """
    file_names = sorted({m["file"] for m in meta})
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    folds = []
    for fold_idx, (trainval_file_idx, test_file_idx) in enumerate(kf.split(file_names)):
        test_files = set(file_names[i] for i in test_file_idx)

        # split off the last 1/(n_folds-1) fraction of trainval as val
        trainval_files = [file_names[i] for i in trainval_file_idx]
        n_val = max(1, len(trainval_files) // (n_folds - 1))
        val_files = set(trainval_files[-n_val:])
        train_files = set(trainval_files[:-n_val])

        # index mapping
        split_indices = {"train": [], "val": [], "test": []}
        for i, m in enumerate(meta):
            f = m["file"]
            if f in train_files:
                split_indices["train"].append(i)
            elif f in val_files:
                split_indices["val"].append(i)
            elif f in test_files:
                split_indices["test"].append(i)

        folds.append({
            **split_indices,
            "train_files": train_files,
            "val_files": val_files,
            "test_files": test_files,
        })

    return folds


def make_cross_genre_splits(
    meta, train_genre, n_folds=5, seed=42,
    source_size=None, target_size=None, sample_seed=None,
):
    """Generate a split for Cross-genre training.

    Construct train/val by K-Fold splitting source-genre files, and
    use target-genre files as the cross-genre test set.

    Args:
        source_size: use only N randomly sampled source-genre songs (None=all)
        target_size: use only N randomly sampled target-genre songs (None=all)
        sample_seed: seed for reproducing source/target subsampling (use seed if None)

    Returns:
        list of dicts: [{
            "train", "val", "test" (within-genre),
            "test_cross" (target genre subset),
            + corresponding *_files sets
        }, ...]
    """
    import random as _random

    all_files = sorted({m["file"] for m in meta})
    source_files_full = sorted([f for f in all_files if _get_genre(f) == train_genre])
    target_genre = [g for g in {"정악", "민속악"} - {train_genre}][0]
    target_files_full = sorted([f for f in all_files if _get_genre(f) == target_genre])

    # Subsample (reproducibility ensured by seed)
    sub_seed = sample_seed if sample_seed is not None else seed
    rng = _random.Random(sub_seed)

    if source_size is not None and source_size < len(source_files_full):
        source_files = sorted(rng.sample(source_files_full, source_size))
        print(f"  Source subsample: {train_genre} {len(source_files_full)} → {len(source_files)} songs (sample_seed={sub_seed})")
    else:
        source_files = source_files_full

    if target_size is not None and target_size < len(target_files_full):
        target_files = sorted(rng.sample(target_files_full, target_size))
        print(f"  Target subsample: {target_genre} {len(target_files_full)} → {len(target_files)} songs (sample_seed={sub_seed})")
    else:
        target_files = target_files_full

    print(f"  Cross-genre split: source({train_genre}) {len(source_files)} songs → "
          f"target({target_genre}) {len(target_files)} songs")

    # file → index mapping
    file_to_indices = {}
    for i, m in enumerate(meta):
        file_to_indices.setdefault(m["file"], []).append(i)

    target_indices = []
    for f in target_files:
        target_indices.extend(file_to_indices.get(f, []))

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    folds = []
    for fold_idx, (trainval_file_idx, test_file_idx) in enumerate(kf.split(source_files)):
        test_within_files = set(source_files[i] for i in test_file_idx)
        trainval_files_list = [source_files[i] for i in trainval_file_idx]

        n_val = max(1, len(trainval_files_list) // (n_folds - 1))
        val_files = set(trainval_files_list[-n_val:])
        train_files = set(trainval_files_list[:-n_val])

        train_idx, val_idx, test_within_idx = [], [], []
        for f in train_files:
            train_idx.extend(file_to_indices.get(f, []))
        for f in val_files:
            val_idx.extend(file_to_indices.get(f, []))
        for f in test_within_files:
            test_within_idx.extend(file_to_indices.get(f, []))

        folds.append({
            "train": train_idx,
            "val": val_idx,
            "test": test_within_idx,
            "test_cross": target_indices,
            "train_files": train_files,
            "val_files": val_files,
            "test_files": test_within_files,
            "test_cross_files": set(target_files),
        })

    return folds


def make_mixed_genre_splits(
    meta, total_size=102, ratio=0.5, n_folds=5, seed=42, sample_seed=None,
):
    """Split for mixed training combining jeongak + minsokak.

    Compose a total of total_size songs as jeongak n_j = round(total_size * ratio)
    songs + minsokak n_m songs. Run KFold independently per genre so that each
    fold's test/val contains both genres in proportion (= genre-stratified
    file-level split).

    Test is held-out per fold; during evaluation, jeongak/minsokak indices are
    aggregated separately and provided as ``test_jeongak`` / ``test_minsokak``.

    Returns:
        list of dicts: [{
            "train", "val", "test" (indices),
            "test_jeongak", "test_minsokak" (per-genre held-out indices),
            "train_files", "val_files", "test_files",
            "test_jeongak_files", "test_minsokak_files",
        }, ...]
    """
    import random as _random

    all_files = sorted({m["file"] for m in meta})
    jeongak_full = sorted([f for f in all_files if _get_genre(f) == "정악"])
    minsokak_full = sorted([f for f in all_files if _get_genre(f) == "민속악"])

    n_jeongak = int(round(total_size * ratio))
    n_minsokak = total_size - n_jeongak
    assert n_jeongak <= len(jeongak_full), \
        f"requested {n_jeongak} jeongak songs but only {len(jeongak_full)} exist"
    assert n_minsokak <= len(minsokak_full), \
        f"requested {n_minsokak} minsokak songs but only {len(minsokak_full)} exist"

    sub_seed = sample_seed if sample_seed is not None else seed
    rng = _random.Random(sub_seed)

    jeongak_files = sorted(rng.sample(jeongak_full, n_jeongak))
    minsokak_files = sorted(rng.sample(minsokak_full, n_minsokak))

    print(f"  Mixed split: jeongak {n_jeongak} songs + minsokak {n_minsokak} songs "
          f"= {total_size} songs (sample_seed={sub_seed})")

    file_to_indices = {}
    for i, m in enumerate(meta):
        file_to_indices.setdefault(m["file"], []).append(i)

    # independent per-genre KFold → preserve both-genre proportions in each fold
    kf_j = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    kf_m = KFold(n_splits=n_folds, shuffle=True, random_state=seed + 1)
    j_folds = list(kf_j.split(jeongak_files))
    m_folds = list(kf_m.split(minsokak_files))

    folds = []
    for fold_idx in range(n_folds):
        j_trainval_idx, j_test_idx = j_folds[fold_idx]
        m_trainval_idx, m_test_idx = m_folds[fold_idx]

        j_test_files = [jeongak_files[i] for i in j_test_idx]
        m_test_files = [minsokak_files[i] for i in m_test_idx]
        j_trainval = [jeongak_files[i] for i in j_trainval_idx]
        m_trainval = [minsokak_files[i] for i in m_trainval_idx]

        # split train/val per genre (preserving the same ratio as cross-genre)
        n_val_j = max(1, len(j_trainval) // (n_folds - 1))
        j_val = set(j_trainval[-n_val_j:])
        j_train = set(j_trainval[:-n_val_j])
        n_val_m = max(1, len(m_trainval) // (n_folds - 1))
        m_val = set(m_trainval[-n_val_m:])
        m_train = set(m_trainval[:-n_val_m])

        train_files = j_train | m_train
        val_files = j_val | m_val
        test_files = set(j_test_files) | set(m_test_files)

        train_idx, val_idx = [], []
        test_j_idx, test_m_idx = [], []
        for f in train_files:
            train_idx.extend(file_to_indices.get(f, []))
        for f in val_files:
            val_idx.extend(file_to_indices.get(f, []))
        for f in j_test_files:
            test_j_idx.extend(file_to_indices.get(f, []))
        for f in m_test_files:
            test_m_idx.extend(file_to_indices.get(f, []))

        folds.append({
            "train": train_idx,
            "val": val_idx,
            "test": test_j_idx + test_m_idx,
            "test_jeongak": test_j_idx,
            "test_minsokak": test_m_idx,
            "train_files": train_files,
            "val_files": val_files,
            "test_files": test_files,
            "test_jeongak_files": set(j_test_files),
            "test_minsokak_files": set(m_test_files),
        })

    return folds


def apply_fold_to_dataset(ds, fold):
    """Overwrite the dataset's _split_indices and _split_files with the fold."""
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


# ─── single-fold training ─────────────────────────────────────

def train_one_fold(args, ds, fold_idx, fold, model_seed, run_dir, device):
    """Train one fold and return the result."""
    from sklearn.metrics import (
        f1_score as sk_f1,
        precision_score as sk_precision,
        recall_score as sk_recall,
        confusion_matrix as sk_confusion,
        classification_report as sk_report,
    )

    # apply fold
    apply_fold_to_dataset(ds, fold)
    class_names = ds.class_names
    n_classes = len(class_names)

    # fix model seed (same initialization every fold)
    set_deterministic(model_seed)

    # data loaders
    input_type = MODEL_CONFIGS.get(args.model, {}).get("input_type", "f0")
    is_mel = (input_type == "mel")

    if input_type == "embedding":
        train_ds = ds.get_torch_dataset("train")
        val_ds = ds.get_torch_dataset("val")
        test_ds = ds.get_torch_dataset("test")
    elif is_mel or input_type == "chroma":
        train_ds = ds.get_torch_dataset(
            "train", augment=True, aug_prob=args.aug_prob,
            freq_mask_param=args.freq_mask_param,
            time_mask_param=args.time_mask_param,
        )
        val_ds = ds.get_torch_dataset("val", augment=False)
        test_ds = ds.get_torch_dataset("test", augment=False)
    else:
        aug_kwargs = dict(
            aug_pitch_shift_range=args.aug_pitch_shift,
            aug_time_stretch_range=(args.aug_time_stretch_lo, args.aug_time_stretch_hi),
            aug_noise_std=args.aug_noise_std,
            aug_prob=args.aug_prob,
        )
        train_ds = ds.get_torch_dataset("train", augment=True, **aug_kwargs)
        val_ds = ds.get_torch_dataset("val", augment=False)
        test_ds = ds.get_torch_dataset("test", augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(model_seed))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    print(f"\n  Fold {fold_idx+1}: Train={len(fold['train'])} "
          f"({len(fold['train_files'])} files) / "
          f"Val={len(fold['val'])} ({len(fold['val_files'])} files) / "
          f"Test={len(fold['test'])} ({len(fold['test_files'])} files)")

    # build model (with seed fixed)
    ModelClass = MODEL_REGISTRY[args.model]
    if input_type == "embedding":
        n_features = ds.X.shape[1]
        model = ModelClass(n_features=n_features, n_classes=n_classes).to(device)
    elif is_mel or input_type == "chroma":
        try:
            model = ModelClass(num_classes=n_classes).to(device)
        except TypeError:
            model = ModelClass(n_classes=n_classes).to(device)
    else:
        n_features = ds.X.shape[-1]
        model = ModelClass(n_features=n_features, n_classes=n_classes).to(device)

    # loss / optimizer / scheduler
    train_idx = fold["train"]
    class_weights = compute_class_weights(ds.y[train_idx], n_classes=n_classes).to(device)
    criterion = FocalLoss(gamma=args.focal_gamma, weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6,
        )
    elif args.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr,
            steps_per_epoch=len(train_loader), epochs=args.epochs,
        )

    # training paths
    fold_dir = run_dir / f"fold{fold_idx+1}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    model_path = fold_dir / f"best_model.pt"

    # training loop
    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y_batch)
            train_correct += (logits.argmax(1) == y_batch).sum().item()
            train_total += len(y_batch)

            if args.scheduler == "onecycle":
                scheduler.step()

        train_loss /= train_total
        train_acc = train_correct / train_total

        val_result = evaluate(model, val_loader, criterion, device)
        val_f1 = compute_macro_f1(val_result["preds"], val_result["targets"], n_classes)

        if args.scheduler == "cosine":
            scheduler.step()
        elif args.scheduler == "plateau":
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
                  f"VaF1={val_f1:.4f} Best={best_val_f1:.4f} "
                  f"Pat={patience_counter}{improved}")

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

    # test evaluation
    model.load_state_dict(torch.load(model_path, weights_only=True))
    test_result = evaluate(model, test_loader, criterion, device)

    targets = test_result["targets"]
    preds = test_result["preds"]

    test_macro_f1 = sk_f1(targets, preds, average="macro", zero_division=0)
    test_weighted_f1 = sk_f1(targets, preds, average="weighted", zero_division=0)
    test_macro_precision = sk_precision(targets, preds, average="macro", zero_division=0)
    test_macro_recall = sk_recall(targets, preds, average="macro", zero_division=0)

    per_class_report = sk_report(
        targets, preds, labels=list(range(n_classes)),
        target_names=class_names, output_dict=True, zero_division=0,
    )
    cm = sk_confusion(targets, preds, labels=list(range(n_classes))).tolist()

    print(f"    ✓ Fold {fold_idx+1} done: Best Epoch={best_epoch}, "
          f"Test Acc={test_result['accuracy']:.4f}, "
          f"Macro F1={test_macro_f1:.4f}, "
          f"Weighted F1={test_weighted_f1:.4f}")

    # save visualizations
    plot_curves(history, best_epoch, best_val_f1, fold_dir / "curves.png")
    plot_confusion_matrix(
        targets, preds, class_names,
        fold_dir / "confusion.png",
        test_acc=test_result["accuracy"], macro_f1=test_macro_f1,
    )

    # Cross-genre test (if present)
    cross_result_data = {}
    combined_result_data = {}
    if fold.get("test_cross"):
        cross_indices = fold["test_cross"]
        X_cross = torch.FloatTensor(ds.X[cross_indices])
        y_cross = torch.LongTensor(ds.y[cross_indices])
        cross_td = TensorDataset(X_cross, y_cross)
        cross_loader = DataLoader(cross_td, batch_size=args.batch_size, shuffle=False)
        cross_eval = evaluate(model, cross_loader, criterion, device)

        cross_targets = cross_eval["targets"]
        cross_preds = cross_eval["preds"]

        cross_macro_f1 = sk_f1(cross_targets, cross_preds, average="macro", zero_division=0)
        cross_weighted_f1 = sk_f1(cross_targets, cross_preds, average="weighted", zero_division=0)
        cross_macro_precision = sk_precision(cross_targets, cross_preds, average="macro", zero_division=0)
        cross_macro_recall = sk_recall(cross_targets, cross_preds, average="macro", zero_division=0)

        cross_per_class = sk_report(
            cross_targets, cross_preds, labels=list(range(n_classes)),
            target_names=class_names, output_dict=True, zero_division=0,
        )
        cross_cm = sk_confusion(cross_targets, cross_preds, labels=list(range(n_classes))).tolist()

        cross_result_data = {
            "test_accuracy": round(cross_eval["accuracy"], 4),
            "test_macro_f1": round(cross_macro_f1, 4),
            "test_weighted_f1": round(cross_weighted_f1, 4),
            "test_macro_precision": round(cross_macro_precision, 4),
            "test_macro_recall": round(cross_macro_recall, 4),
            "per_class": {
                cls_name: {
                    "precision": round(cross_per_class[cls_name]["precision"], 4),
                    "recall": round(cross_per_class[cls_name]["recall"], 4),
                    "f1": round(cross_per_class[cls_name]["f1-score"], 4),
                    "support": cross_per_class[cls_name]["support"],
                }
                for cls_name in class_names if cls_name in cross_per_class
            },
            "confusion_matrix": cross_cm,
            "n_test": len(cross_indices),
            "n_test_files": len(fold["test_cross_files"]),
        }

        plot_confusion_matrix(
            cross_targets, cross_preds, class_names,
            fold_dir / "confusion_cross.png",
            test_acc=cross_eval["accuracy"], macro_f1=cross_macro_f1,
        )

        # Combined (within + cross)
        all_targets = np.concatenate([targets, cross_targets])
        all_preds = np.concatenate([preds, cross_preds])
        comb_macro_f1 = sk_f1(all_targets, all_preds, average="macro", zero_division=0)
        comb_weighted_f1 = sk_f1(all_targets, all_preds, average="weighted", zero_division=0)
        comb_acc = float((all_targets == all_preds).mean())

        combined_result_data = {
            "test_accuracy": round(comb_acc, 4),
            "test_macro_f1": round(comb_macro_f1, 4),
            "test_weighted_f1": round(comb_weighted_f1, 4),
            "n_test": len(all_targets),
        }

        print(f"    ✓ Cross-genre: Acc={cross_eval['accuracy']:.4f}, "
              f"Macro F1={cross_macro_f1:.4f}")
        print(f"    ✓ Combined:    Acc={comb_acc:.4f}, "
              f"Macro F1={comb_macro_f1:.4f}")

    # fold result
    fold_result = {
        "fold": fold_idx + 1,
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_val_f1, 4),
        "test_accuracy": round(test_result["accuracy"], 4),
        "test_macro_f1": round(test_macro_f1, 4),
        "test_weighted_f1": round(test_weighted_f1, 4),
        "test_macro_precision": round(test_macro_precision, 4),
        "test_macro_recall": round(test_macro_recall, 4),
        "per_class": {
            cls_name: {
                "precision": round(per_class_report[cls_name]["precision"], 4),
                "recall": round(per_class_report[cls_name]["recall"], 4),
                "f1": round(per_class_report[cls_name]["f1-score"], 4),
                "support": per_class_report[cls_name]["support"],
            }
            for cls_name in class_names if cls_name in per_class_report
        },
        "confusion_matrix": cm,
        "n_train": len(fold["train"]),
        "n_val": len(fold["val"]),
        "n_test": len(fold["test"]),
        "n_train_files": len(fold["train_files"]),
        "n_val_files": len(fold["val_files"]),
        "n_test_files": len(fold["test_files"]),
    }
    if cross_result_data:
        fold_result["cross_genre"] = cross_result_data
        fold_result["combined"] = combined_result_data

    # Mixed-genre: aggregate held-out test separately per genre
    if fold.get("test_jeongak") is not None and fold.get("test_minsokak") is not None:
        per_genre_result = {}
        for gname, gkey in [("정악", "test_jeongak"), ("민속악", "test_minsokak")]:
            g_idx = fold[gkey]
            if len(g_idx) == 0:
                continue
            X_g = torch.FloatTensor(ds.X[g_idx])
            y_g = torch.LongTensor(ds.y[g_idx])
            g_loader = DataLoader(
                TensorDataset(X_g, y_g), batch_size=args.batch_size, shuffle=False,
            )
            g_eval = evaluate(model, g_loader, criterion, device)
            g_tgt, g_prd = g_eval["targets"], g_eval["preds"]
            g_macro_f1 = sk_f1(g_tgt, g_prd, average="macro", zero_division=0)
            g_weighted_f1 = sk_f1(g_tgt, g_prd, average="weighted", zero_division=0)
            g_macro_prec = sk_precision(g_tgt, g_prd, average="macro", zero_division=0)
            g_macro_rec = sk_recall(g_tgt, g_prd, average="macro", zero_division=0)
            g_report = sk_report(
                g_tgt, g_prd, labels=list(range(n_classes)),
                target_names=class_names,
                output_dict=True, zero_division=0,
            )
            g_cm = sk_confusion(g_tgt, g_prd, labels=list(range(n_classes))).tolist()

            per_genre_result[gname] = {
                "test_accuracy": round(g_eval["accuracy"], 4),
                "test_macro_f1": round(g_macro_f1, 4),
                "test_weighted_f1": round(g_weighted_f1, 4),
                "test_macro_precision": round(g_macro_prec, 4),
                "test_macro_recall": round(g_macro_rec, 4),
                "per_class": {
                    cls_name: {
                        "precision": round(g_report[cls_name]["precision"], 4),
                        "recall": round(g_report[cls_name]["recall"], 4),
                        "f1": round(g_report[cls_name]["f1-score"], 4),
                        "support": g_report[cls_name]["support"],
                    }
                    for cls_name in class_names if cls_name in g_report
                },
                "confusion_matrix": g_cm,
                "n_test": len(g_idx),
                "n_test_files": len(fold[f"{gkey}_files"]),
            }
            print(f"    ✓ Mixed→{gname}: Acc={g_eval['accuracy']:.4f}, "
                  f"Macro F1={g_macro_f1:.4f} (n_files={len(fold[f'{gkey}_files'])})")

        if per_genre_result:
            fold_result["per_genre"] = per_genre_result

    # save history
    with open(fold_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with open(fold_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(fold_result, f, ensure_ascii=False, indent=2)

    return fold_result


# ─── result aggregation ───────────────────────────────────────────

def aggregate_results(fold_results, class_names):
    """Aggregate per-fold results and compute mean ± std."""
    metrics = [
        "test_accuracy", "test_macro_f1", "test_weighted_f1",
        "test_macro_precision", "test_macro_recall",
    ]

    agg = {}
    for m in metrics:
        values = [r[m] for r in fold_results]
        agg[m] = {
            "mean": round(np.mean(values), 4),
            "std": round(np.std(values), 4),
            "values": values,
        }

    # per-class aggregation
    per_class_agg = {}
    for cls_name in class_names:
        cls_agg = {}
        for metric in ["precision", "recall", "f1"]:
            values = [r["per_class"][cls_name][metric]
                      for r in fold_results if cls_name in r["per_class"]]
            cls_agg[metric] = {
                "mean": round(np.mean(values), 4),
                "std": round(np.std(values), 4),
                "values": values,
            }
        per_class_agg[cls_name] = cls_agg

    agg["per_class"] = per_class_agg

    # Cross-genre aggregation (if present)
    if "cross_genre" in fold_results[0]:
        cross_agg = {}
        for m in metrics:
            values = [r["cross_genre"][m] for r in fold_results]
            cross_agg[m] = {
                "mean": round(np.mean(values), 4),
                "std": round(np.std(values), 4),
                "values": values,
            }
        cross_per_class = {}
        for cls_name in class_names:
            cls_a = {}
            for metric in ["precision", "recall", "f1"]:
                values = [r["cross_genre"]["per_class"][cls_name][metric]
                          for r in fold_results
                          if cls_name in r["cross_genre"]["per_class"]]
                cls_a[metric] = {
                    "mean": round(np.mean(values), 4),
                    "std": round(np.std(values), 4),
                    "values": values,
                }
            cross_per_class[cls_name] = cls_a
        cross_agg["per_class"] = cross_per_class
        agg["cross_genre"] = cross_agg

        # Combined aggregation
        comb_metrics = ["test_accuracy", "test_macro_f1", "test_weighted_f1"]
        comb_agg = {}
        for m in comb_metrics:
            values = [r["combined"][m] for r in fold_results]
            comb_agg[m] = {
                "mean": round(np.mean(values), 4),
                "std": round(np.std(values), 4),
                "values": values,
            }
        agg["combined"] = comb_agg

    # Mixed-genre: per-genre aggregation
    if "per_genre" in fold_results[0]:
        pg_agg = {}
        genres_present = sorted({g for r in fold_results for g in r.get("per_genre", {})})
        for gname in genres_present:
            g_agg = {}
            for m in metrics:
                values = [r["per_genre"][gname][m]
                          for r in fold_results if gname in r.get("per_genre", {})]
                g_agg[m] = {
                    "mean": round(np.mean(values), 4),
                    "std": round(np.std(values), 4),
                    "values": values,
                }
            g_per_class = {}
            for cls_name in class_names:
                cls_a = {}
                for metric in ["precision", "recall", "f1"]:
                    values = [r["per_genre"][gname]["per_class"][cls_name][metric]
                              for r in fold_results
                              if gname in r.get("per_genre", {})
                              and cls_name in r["per_genre"][gname]["per_class"]]
                    if values:
                        cls_a[metric] = {
                            "mean": round(np.mean(values), 4),
                            "std": round(np.std(values), 4),
                            "values": values,
                        }
                if cls_a:
                    g_per_class[cls_name] = cls_a
            g_agg["per_class"] = g_per_class
            pg_agg[gname] = g_agg
        agg["per_genre"] = pg_agg

    return agg


def print_summary(agg, class_names, model_name, n_folds):
    """Pretty-print the aggregated results."""
    print(f"\n{'='*70}")
    print(f"  {model_name} — {n_folds}-Fold Cross-Validation results")
    print(f"{'='*70}")

    print(f"\n  Overall Metrics (mean ± std):")
    for m in ["test_accuracy", "test_macro_f1", "test_weighted_f1",
              "test_macro_precision", "test_macro_recall"]:
        label = m.replace("test_", "").replace("_", " ").title()
        mean = agg[m]["mean"]
        std = agg[m]["std"]
        vals = agg[m]["values"]
        print(f"    {label:20s}: {mean:.4f} ± {std:.4f}  {vals}")

    print(f"\n  Per-class F1 (mean ± std):")
    print(f"    {'Class':12s}  {'Precision':>14s}  {'Recall':>14s}  {'F1':>14s}")
    print(f"    {'-'*58}")
    for cls_name in class_names:
        ca = agg["per_class"][cls_name]
        p = f"{ca['precision']['mean']:.4f}±{ca['precision']['std']:.4f}"
        r = f"{ca['recall']['mean']:.4f}±{ca['recall']['std']:.4f}"
        f = f"{ca['f1']['mean']:.4f}±{ca['f1']['std']:.4f}"
        print(f"    {cls_name:12s}  {p:>14s}  {r:>14s}  {f:>14s}")

    # Cross-genre results (if present)
    if "cross_genre" in agg:
        print(f"\n  ── Cross-genre Test (mean ± std) ──")
        for m in ["test_accuracy", "test_macro_f1", "test_weighted_f1",
                   "test_macro_precision", "test_macro_recall"]:
            label = m.replace("test_", "").replace("_", " ").title()
            mean = agg["cross_genre"][m]["mean"]
            std = agg["cross_genre"][m]["std"]
            vals = agg["cross_genre"][m]["values"]
            print(f"    {label:20s}: {mean:.4f} ± {std:.4f}  {vals}")

        print(f"\n  Cross-genre Per-class F1 (mean ± std):")
        print(f"    {'Class':12s}  {'Precision':>14s}  {'Recall':>14s}  {'F1':>14s}")
        print(f"    {'-'*58}")
        for cls_name in class_names:
            ca = agg["cross_genre"]["per_class"][cls_name]
            p = f"{ca['precision']['mean']:.4f}±{ca['precision']['std']:.4f}"
            r = f"{ca['recall']['mean']:.4f}±{ca['recall']['std']:.4f}"
            f = f"{ca['f1']['mean']:.4f}±{ca['f1']['std']:.4f}"
            print(f"    {cls_name:12s}  {p:>14s}  {r:>14s}  {f:>14s}")

        print(f"\n  ── Combined (Within + Cross) ──")
        for m in ["test_accuracy", "test_macro_f1", "test_weighted_f1"]:
            label = m.replace("test_", "").replace("_", " ").title()
            mean = agg["combined"][m]["mean"]
            std = agg["combined"][m]["std"]
            print(f"    {label:20s}: {mean:.4f} ± {std:.4f}")

    # Mixed-genre per-genre results (if present)
    if "per_genre" in agg:
        for gname, g_agg in agg["per_genre"].items():
            print(f"\n  ── Mixed → {gname} (held-out, mean ± std) ──")
            for m in ["test_accuracy", "test_macro_f1", "test_weighted_f1",
                       "test_macro_precision", "test_macro_recall"]:
                label = m.replace("test_", "").replace("_", " ").title()
                mean = g_agg[m]["mean"]
                std = g_agg[m]["std"]
                vals = g_agg[m]["values"]
                print(f"    {label:20s}: {mean:.4f} ± {std:.4f}  {vals}")

            print(f"\n  Mixed → {gname} Per-class F1 (mean ± std):")
            print(f"    {'Class':12s}  {'Precision':>14s}  {'Recall':>14s}  {'F1':>14s}")
            print(f"    {'-'*58}")
            for cls_name in class_names:
                if cls_name not in g_agg["per_class"]:
                    continue
                ca = g_agg["per_class"][cls_name]
                p = f"{ca['precision']['mean']:.4f}±{ca['precision']['std']:.4f}"
                r = f"{ca['recall']['mean']:.4f}±{ca['recall']['std']:.4f}"
                f = f"{ca['f1']['mean']:.4f}±{ca['f1']['std']:.4f}"
                print(f"    {cls_name:12s}  {p:>14s}  {r:>14s}  {f:>14s}")

    print(f"{'='*70}")


# ─── main ───────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="sigimsae classification K-Fold CV")

    p.add_argument("--config", type=str, default=None,
                   help="YAML config file path")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--class-scheme", type=str, default=None)
    p.add_argument("--label-dir", type=str, default=None)
    p.add_argument("--f0-method", type=str, default=None)
    p.add_argument("--f0-dir", type=str, default=None)
    p.add_argument("--beat-dir", type=str, default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--pad-ms", type=int, default=None)
    p.add_argument("--seed", type=int, default=None,
                   help="Data K-Fold split seed (default: 42)")
    p.add_argument("--embedding-path", type=str, default=None)
    p.add_argument("--audio-dir", type=str, default=None)
    p.add_argument("--sr", type=int, default=None)
    p.add_argument("--n-mels", type=int, default=None)
    p.add_argument("--n-fft", type=int, default=None)
    p.add_argument("--hop-length", type=int, default=None)
    p.add_argument("--max-mel-frames", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--scheduler", type=str, default=None)
    p.add_argument("--focal-gamma", type=float, default=None)
    p.add_argument("--aug-pitch-shift", type=float, default=None)
    p.add_argument("--aug-time-stretch-lo", type=float, default=None)
    p.add_argument("--aug-time-stretch-hi", type=float, default=None)
    p.add_argument("--aug-noise-std", type=float, default=None)
    p.add_argument("--aug-prob", type=float, default=None)
    p.add_argument("--freq-mask-param", type=int, default=None)
    p.add_argument("--time-mask-param", type=int, default=None)
    p.add_argument("--norm-mode", type=str, default=None,
                   choices=["segment_median", "song_median", "midi_abs"],
                   help="F0 normalization mode (F0 CNN only)")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default=None)

    # K-Fold specific
    p.add_argument("--n-folds", type=int, default=5,
                   help="Number of K-Folds (default: 5)")
    p.add_argument("--model-seed", type=int, default=123,
                   help="Model initialization seed (same for all folds, default: 123)")

    # Cross-genre
    p.add_argument("--cross-genre", action="store_true",
                   help="Cross-genre training mode: train on one genre, test on the other")
    p.add_argument("--train-genre", type=str, default=None,
                   choices=["정악", "민속악"],
                   help="Genre to train on in cross-genre mode")
    p.add_argument("--source-size", type=int, default=None,
                   help="(Cross-genre) use only N randomly sampled source-genre songs. default=all")
    p.add_argument("--target-size", type=int, default=None,
                   help="(Cross-genre) use only N randomly sampled target-genre songs. default=all")
    p.add_argument("--sample-seed", type=int, default=None,
                   help="(Cross-genre/Mixed) source/target subsample seed. default=--seed")

    # Mixed-genre (jeongak+minsokak combined baseline)
    p.add_argument("--mixed-genre", action="store_true",
                   help="Mixed training mode: train on a fixed number of jeongak+minsokak songs, "
                        "and aggregate the held-out test separately per genre")
    p.add_argument("--mixed-total-size", type=int, default=None,
                   help="(Mixed) total training songs. default=102")
    p.add_argument("--mixed-ratio", type=float, default=None,
                   help="(Mixed) jeongak ratio. default=0.5 → 51 jeongak + 51 minsokak")

    cli_args = p.parse_args()

    DEFAULTS = {
        "model": "SigimsaeCNN",
        "class_scheme": "7cat",
        "label_dir": "precheck_sigimsae_labels",
        "f0_method": "rmvpe",
        "f0_dir": None,
        "beat_dir": None,
        "max_frames": 300,
        "pad_ms": 0,
        "seed": 42,
        "audio_dir": "KVocSet",
        "chroma_dir": "features/chroma_output",
        "sr": 22050,
        "n_mels": 128,
        "n_chroma": 120,
        "n_fft": 2048,
        "hop_length": 512,
        "max_mel_frames": 130,
        "max_chroma_frames": 171,
        "batch_size": 256,
        "epochs": 100,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "patience": 15,
        "scheduler": "cosine",
        "focal_gamma": 1.0,
        "aug_pitch_shift": 0.1,
        "aug_time_stretch_lo": 0.85,
        "aug_time_stretch_hi": 1.15,
        "aug_noise_std": 0.015,
        "aug_prob": 0.3,
        "freq_mask_param": 16,
        "time_mask_param": 16,
        "no_wandb": False,
        "wandb_project": "ismir2026-sigimsae-kfold",
        "n_folds": 5,
        "model_seed": 123,
        "norm_mode": "segment_median",
        "cross_genre": False,
        "train_genre": None,
        "source_size": None,
        "target_size": None,
        "sample_seed": None,
        "mixed_genre": False,
        "mixed_total_size": 102,
        "mixed_ratio": 0.5,
    }

    final = dict(DEFAULTS)
    if cli_args.config:
        yaml_cfg = load_config(cli_args.config)
        for k, v in yaml_cfg.items():
            final[k.replace("-", "_")] = v

    for key, val in vars(cli_args).items():
        if key == "config":
            continue
        if val is not None:
            final[key.replace("-", "_")] = val

    if final["f0_dir"] is None:
        final["f0_dir"] = f"pitch_output/{final['f0_method']}"

    FLOAT_KEYS = {"lr", "weight_decay", "focal_gamma", "aug_pitch_shift",
                  "aug_time_stretch_lo", "aug_time_stretch_hi", "aug_noise_std",
                  "aug_prob", "mixed_ratio"}
    INT_KEYS = {"batch_size", "epochs", "patience", "max_frames", "max_mel_frames",
                "max_chroma_frames", "n_mels", "n_chroma", "n_fft", "hop_length",
                "sr", "seed", "pad_ms",
                "freq_mask_param", "time_mask_param", "n_folds", "model_seed",
                "mixed_total_size"}
    for k in FLOAT_KEYS:
        if k in final and final[k] is not None:
            final[k] = float(final[k])
    for k in INT_KEYS:
        if k in final and final[k] is not None:
            final[k] = int(final[k])

    return argparse.Namespace(**final)


def main():
    args = parse_args()

    DEVICE = (
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model_name = args.model
    assert model_name in MODEL_REGISTRY, \
        f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}"
    input_type = MODEL_CONFIGS.get(model_name, {}).get("input_type", "f0")
    is_mel = (input_type == "mel")

    # validate Cross-genre / Mixed-genre mode
    is_cross_genre = getattr(args, "cross_genre", False)
    is_mixed_genre = getattr(args, "mixed_genre", False)
    train_genre = getattr(args, "train_genre", None)
    assert not (is_cross_genre and is_mixed_genre), \
        "--cross-genre and --mixed-genre cannot be used together"
    if is_cross_genre:
        assert train_genre in ("정악", "민속악"), \
            "--cross-genre mode requires --train-genre (정악 or 민속악)"
        target_genre = [g for g in {"정악", "민속악"} - {train_genre}][0]
        genre_tag_map = {"정악": "J", "민속악": "M"}
        cross_tag = f"_cross{genre_tag_map[train_genre]}2{genre_tag_map[target_genre]}"
        print(f"Model: {model_name}  |  Input: {input_type}  |  Class: {args.class_scheme}")
        print(f"Cross-genre: {train_genre} → {target_genre}")
        print(f"K-Fold: {args.n_folds} (within source genre)  |  Data seed: {args.seed}  |  Model seed: {args.model_seed}")
    elif is_mixed_genre:
        mtot = int(getattr(args, "mixed_total_size", 102))
        mratio = float(getattr(args, "mixed_ratio", 0.5))
        n_j = int(round(mtot * mratio))
        n_m = mtot - n_j
        cross_tag = f"_mixed{mtot}j{n_j}m{n_m}"
        print(f"Model: {model_name}  |  Input: {input_type}  |  Class: {args.class_scheme}")
        print(f"Mixed-genre: jeongak {n_j} songs + minsokak {n_m} songs = {mtot} songs")
        print(f"K-Fold: {args.n_folds} (per-genre stratified)  |  Data seed: {args.seed}  |  Model seed: {args.model_seed}")
    else:
        cross_tag = ""
        print(f"Model: {model_name}  |  Input: {input_type}  |  Class: {args.class_scheme}")
        print(f"K-Fold: {args.n_folds}  |  Data seed: {args.seed}  |  Model seed: {args.model_seed}")
    print(f"Device: {DEVICE}")
    print("Loading dataset...")

    # load dataset (auto-split is ignored, so load with the default seed)
    if input_type == "embedding":
        embedding_path = getattr(args, 'embedding_path', None)
        assert embedding_path, \
            "MERTEmbeddingClassifier requires --embedding-path"
        ds = SigimsaeEmbeddingDataset(
            embedding_path=embedding_path,
            seed=args.seed,
            class_scheme=args.class_scheme,
        )
    elif is_mel:
        ds = SigimsaeMelDataset(
            label_dir=args.label_dir,
            audio_dir=args.audio_dir,
            sr=args.sr,
            n_mels=args.n_mels,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            max_mel_frames=args.max_mel_frames,
            pad_ms=args.pad_ms,
            seed=args.seed,
            class_scheme=args.class_scheme,
        )
    elif input_type == "chroma":
        ds = SigimsaeChromaDataset(
            label_dir=args.label_dir,
            chroma_dir=getattr(args, 'chroma_dir', 'features/chroma_output'),
            sr=args.sr,
            n_chroma=getattr(args, 'n_chroma', 120),
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            max_chroma_frames=getattr(args, 'max_chroma_frames', 171),
            pad_ms=args.pad_ms,
            seed=args.seed,
            class_scheme=args.class_scheme,
        )
    else:
        ds = SigimsaeF0Dataset(
            label_dir=args.label_dir,
            f0_dir=args.f0_dir,
            max_frames=args.max_frames,
            pad_ms=args.pad_ms,
            seed=args.seed,
            beat_dir=args.beat_dir,
            class_scheme=args.class_scheme,
            norm_mode=args.norm_mode,
        )

    ds.summary()

    class_names = ds.class_names
    print(f"\nTotal segments: {len(ds.X)}  |  Songs: {len({m['file'] for m in ds.meta})}")

    # generate K-Fold split
    if is_cross_genre:
        folds = make_cross_genre_splits(
            ds.meta, train_genre=train_genre,
            n_folds=args.n_folds, seed=args.seed,
            source_size=getattr(args, "source_size", None),
            target_size=getattr(args, "target_size", None),
            sample_seed=getattr(args, "sample_seed", None),
        )
    elif is_mixed_genre:
        folds = make_mixed_genre_splits(
            ds.meta,
            total_size=int(getattr(args, "mixed_total_size", 102)),
            ratio=float(getattr(args, "mixed_ratio", 0.5)),
            n_folds=args.n_folds, seed=args.seed,
            sample_seed=getattr(args, "sample_seed", None),
        )
    else:
        folds = make_song_kfold_splits(ds.meta, n_folds=args.n_folds, seed=args.seed)

    # run directory
    RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
    norm_tag = ""
    if input_type == "f0" and getattr(args, "norm_mode", None) and args.norm_mode != "segment_median":
        norm_tag = f"_{args.norm_mode}"
    sub_tag = ""
    if is_cross_genre:
        ssize = getattr(args, "source_size", None)
        tsize = getattr(args, "target_size", None)
        sseed = getattr(args, "sample_seed", None)
        if ssize is not None or tsize is not None:
            sub_tag = f"_s{ssize or 'all'}t{tsize or 'all'}_ss{sseed if sseed is not None else args.seed}"
    elif is_mixed_genre:
        sseed = getattr(args, "sample_seed", None)
        sub_tag = f"_ss{sseed if sseed is not None else args.seed}"
    RUN_NAME = f"{model_name}_{args.class_scheme}{norm_tag}{cross_tag}{sub_tag}_{args.n_folds}fold_{RUN_TS}"
    run_dir = Path(f"models/{RUN_NAME}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run: {run_dir}")

    # wandb
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=RUN_NAME,
            config={
                "model": model_name,
                "input_type": input_type,
                "class_scheme": args.class_scheme,
                "n_folds": args.n_folds,
                "data_seed": args.seed,
                "model_seed": args.model_seed,
                "n_segments": len(ds.X),
                "n_songs": len({m["file"] for m in ds.meta}),
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "scheduler": args.scheduler,
                "focal_gamma": args.focal_gamma,
                "patience": args.patience,
                "device": DEVICE,
                "cross_genre": is_cross_genre,
                "train_genre": train_genre if is_cross_genre else "all",
                "mixed_genre": is_mixed_genre,
                "mixed_total_size": int(getattr(args, "mixed_total_size", 102)) if is_mixed_genre else None,
                "mixed_ratio": float(getattr(args, "mixed_ratio", 0.5)) if is_mixed_genre else None,
            },
        )

    # K-Fold training
    fold_results = []
    for fold_idx, fold in enumerate(folds):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold_idx+1}/{args.n_folds}")
        print(f"{'='*60}")

        result = train_one_fold(
            args, ds, fold_idx, fold,
            model_seed=args.model_seed,
            run_dir=run_dir,
            device=DEVICE,
        )
        fold_results.append(result)

        if use_wandb:
            log_dict = {
                f"fold{fold_idx+1}/test_accuracy": result["test_accuracy"],
                f"fold{fold_idx+1}/test_macro_f1": result["test_macro_f1"],
                f"fold{fold_idx+1}/test_weighted_f1": result["test_weighted_f1"],
                f"fold{fold_idx+1}/best_epoch": result["best_epoch"],
            }
            if "cross_genre" in result:
                log_dict[f"fold{fold_idx+1}/cross_test_accuracy"] = result["cross_genre"]["test_accuracy"]
                log_dict[f"fold{fold_idx+1}/cross_test_macro_f1"] = result["cross_genre"]["test_macro_f1"]
            if "per_genre" in result:
                for gname, g_res in result["per_genre"].items():
                    gtag = "J" if gname == "정악" else "M"
                    log_dict[f"fold{fold_idx+1}/mixed2{gtag}_accuracy"] = g_res["test_accuracy"]
                    log_dict[f"fold{fold_idx+1}/mixed2{gtag}_macro_f1"] = g_res["test_macro_f1"]
            wandb.log(log_dict)

    # aggregate
    agg = aggregate_results(fold_results, class_names)
    print_summary(agg, class_names, model_name, args.n_folds)

    # save full results
    full_results = {
        "model": model_name,
        "class_scheme": args.class_scheme,
        "n_folds": args.n_folds,
        "data_seed": args.seed,
        "model_seed": args.model_seed,
        "class_names": class_names,
        "input_type": input_type,
        "cross_genre": is_cross_genre,
        "train_genre": train_genre if is_cross_genre else "all",
        "mixed_genre": is_mixed_genre,
        "mixed_total_size": int(getattr(args, "mixed_total_size", 102)) if is_mixed_genre else None,
        "mixed_ratio": float(getattr(args, "mixed_ratio", 0.5)) if is_mixed_genre else None,
        "aggregate": agg,
        "folds": fold_results,
        "run_name": RUN_NAME,
        "timestamp": RUN_TS,
    }

    results_path = run_dir / "kfold_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved: {results_path}")

    # combined results file
    CV_RESULTS_PATH = Path("results/kfold_results.json")
    CV_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_cv = []
    if CV_RESULTS_PATH.exists():
        with open(CV_RESULTS_PATH, encoding="utf-8") as f:
            all_cv = json.load(f)
    all_cv.append(full_results)
    with open(CV_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_cv, f, ensure_ascii=False, indent=2)
    print(f"Combined CV results: {CV_RESULTS_PATH} ({len(all_cv)} experiments)")

    # wandb final
    if use_wandb:
        for m in ["test_accuracy", "test_macro_f1", "test_weighted_f1",
                   "test_macro_precision", "test_macro_recall"]:
            wandb.summary[f"{m}_mean"] = agg[m]["mean"]
            wandb.summary[f"{m}_std"] = agg[m]["std"]
        if "cross_genre" in agg:
            for m in ["test_accuracy", "test_macro_f1", "test_weighted_f1"]:
                wandb.summary[f"cross_{m}_mean"] = agg["cross_genre"][m]["mean"]
                wandb.summary[f"cross_{m}_std"] = agg["cross_genre"][m]["std"]
        if "per_genre" in agg:
            for gname, g_agg in agg["per_genre"].items():
                gtag = "J" if gname == "정악" else "M"
                for m in ["test_accuracy", "test_macro_f1", "test_weighted_f1"]:
                    wandb.summary[f"mixed2{gtag}_{m}_mean"] = g_agg[m]["mean"]
                    wandb.summary[f"mixed2{gtag}_{m}_std"] = g_agg[m]["std"]
        wandb.finish()

    print(f"\nDone! All fold results: {run_dir}")


if __name__ == "__main__":
    main()
