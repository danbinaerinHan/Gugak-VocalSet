"""
Sigimsae classification models and training utilities

Models:
    - SigimsaeCNN: F0-based 1D CNN (input: batch, seq_len, n_features)
    - SigimsaeMelCNN: Mel Spectrogram 2D CNN (input: batch, 1, 64, n_frames)
    - MERTEmbeddingClassifier: MARBLE-style probing head over pre-extracted MERT/CultureMERT embeddings

Utils:
    - FocalLoss
    - compute_class_weights
    - compute_macro_f1
    - evaluate
    - print_classification_report
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare_f0_dataset import CLASS_NAMES


# ─── Loss ──────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal Loss: down-weights easy samples and focuses on hard ones."""

    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ─── Model: F0 1D CNN ──────────────────────────────────

class SigimsaeCNN(nn.Module):
    """1D CNN for F0-based sigimsae classification.

    Input: (batch, seq_len, n_features) — [norm_f0, voicing, delta_f0, ...]
    Output: (batch, n_classes)
    """

    def __init__(self, n_features=3, n_classes=len(CLASS_NAMES)):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, n_features, seq_len)
        x = self.conv_blocks(x)
        x = x.squeeze(-1)       # (batch, 256)
        return self.classifier(x)


# ─── Model: Mel 2D CNN ─────────────────────────────────

class SigimsaeMelCNN(nn.Module):
    """2D CNN for mel spectrogram classification.

    Input: (batch, 1, 64, n_frames)
    Output: (batch, n_classes)
    """

    def __init__(self, n_classes=len(CLASS_NAMES)):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)  # (batch, 256)
        return self.classifier(x)


# ─── Model: MERT Embedding Classifier (uses pre-extracted embeddings) ──

class MERTEmbeddingClassifier(nn.Module):
    """MARBLE probing head: 1 hidden layer (512) + ReLU + Dropout + linear.
    Matches CultureMERT/MARBLE protocol for sequence-level probing.

    Input: (batch, embed_dim) — pooled embedding extracted via extract_mert_embeddings.py
    Output: (batch, n_classes) logits
    """

    def __init__(self, n_features: int = 768, n_classes: int = 7, hidden: int = 512):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.classifier(x)


# ─── Model registry ───────────────────────────────────

MODEL_REGISTRY = {
    "SigimsaeCNN": SigimsaeCNN,
    "SigimsaeMelCNN": SigimsaeMelCNN,
    "MERTEmbeddingClassifier": MERTEmbeddingClassifier,
}


# ─── Per-model input type metadata ─────────────────────
MODEL_CONFIGS = {
    "SigimsaeCNN":      {"input_type": "f0"},
    "SigimsaeMelCNN":   {"input_type": "mel"},
    "MERTEmbeddingClassifier": {"input_type": "embedding"},
}


# ─── Utilities ──────────────────────────────────────────

def compute_class_weights(y_train, n_classes=len(CLASS_NAMES)):
    """Class-imbalance correction weights (sqrt smoothing)."""
    counts = np.bincount(y_train, minlength=n_classes)
    total = counts.sum()
    raw_weights = total / (n_classes * counts.astype(np.float64))
    weights = np.sqrt(raw_weights)
    return torch.FloatTensor(weights)


def compute_macro_f1(preds, targets, n_classes):
    """Compute macro F1 score."""
    f1_sum = 0.0
    for i in range(n_classes):
        tp = int(((preds == i) & (targets == i)).sum())
        fp = int(((preds == i) & (targets != i)).sum())
        fn = int(((preds != i) & (targets == i)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1_sum += f1
    return f1_sum / n_classes


def evaluate(model, loader, criterion, device):
    """Evaluate the model, returning loss, accuracy, preds, targets."""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "preds": np.array(all_preds),
        "targets": np.array(all_targets),
    }


def print_classification_report(targets, preds, class_names):
    """Print per-class precision, recall, f1; return macro_f1."""
    n_classes = len(class_names)
    print(f"\n{'Class':15s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'Support':>8s}")
    print("-" * 45)

    precisions, recalls, f1s, supports = [], [], [], []
    for i in range(n_classes):
        tp = int(((preds == i) & (targets == i)).sum())
        fp = int(((preds == i) & (targets != i)).sum())
        fn = int(((preds != i) & (targets == i)).sum())
        support = int((targets == i).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(support)

        print(f"{class_names[i]:15s} {prec:6.3f} {rec:6.3f} {f1:6.3f} {support:8d}")

    print("-" * 45)
    macro_prec = np.mean(precisions)
    macro_rec = np.mean(recalls)
    macro_f1 = np.mean(f1s)
    print(f"{'Macro avg':15s} {macro_prec:6.3f} {macro_rec:6.3f} {macro_f1:6.3f} {sum(supports):8d}")

    total = sum(supports)
    w_prec = sum(p * s for p, s in zip(precisions, supports)) / total
    w_rec = sum(r * s for r, s in zip(recalls, supports)) / total
    w_f1 = sum(f * s for f, s in zip(f1s, supports)) / total
    print(f"{'Weighted avg':15s} {w_prec:6.3f} {w_rec:6.3f} {w_f1:6.3f} {total:8d}")

    return macro_f1
