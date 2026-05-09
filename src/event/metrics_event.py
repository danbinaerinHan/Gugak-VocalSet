"""sigimsae Event Detection evaluation metrics

Multi-layer evaluation scheme for paper Table X:

1. Frame-level: Accuracy, Macro F1, Per-class P/R/F1
2. Onset F1: onset only (@50ms, @100ms tolerance)
3. Event F1: onset+offset both evaluated (collar=100ms, 200ms)
4. IoU F1: temporal overlap ratio based (@0.3, @0.5)

Each metric supports 3-level evaluation conditions:
  - class_agnostic: only sigimsae presence/absence (binary)
  - class_aware: sigimsae type must also match

Usage:
    from metrics_event import evaluate_all_metrics, frames_to_events

    pred_events = frames_to_events(pred_frames, hop_sec)
    ref_events = frames_to_events(ref_frames, hop_sec)
    results = evaluate_all_metrics(pred_events, ref_events)
"""

import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict


DONT_CARE = -1


# ─── Frame → Event conversion ────────────────────────────

def frames_to_events(frame_labels, hop_sec, ignore_class=0):
    """Convert consecutive same-class frames into events (onset, offset, class).

    ignore_class (default 0=no_ornament) is not converted into events.
    don't care (-1) frames are also ignored.

    Returns:
        list of {"onset": float, "offset": float, "class": int}
    """
    events = []
    if len(frame_labels) == 0:
        return events

    current_class = frame_labels[0]
    start_frame = 0

    for i in range(1, len(frame_labels)):
        if frame_labels[i] != current_class:
            if current_class != ignore_class and current_class != DONT_CARE:
                events.append({
                    "onset": start_frame * hop_sec,
                    "offset": i * hop_sec,
                    "class": int(current_class),
                })
            current_class = frame_labels[i]
            start_frame = i

    # last segment
    if current_class != ignore_class and current_class != DONT_CARE:
        events.append({
            "onset": start_frame * hop_sec,
            "offset": len(frame_labels) * hop_sec,
            "class": int(current_class),
        })

    return events


# ─── Matching utilities ──────────────────────────────────

def _match_events_onset(pred_events, ref_events, tolerance_sec,
                        class_agnostic=False):
    """Onset-based matching: |pred_onset - ref_onset| <= tolerance.

    Returns:
        (tp, matched_pred_indices, matched_ref_indices)
    """
    matched_ref = set()
    matched_pred = set()

    for pi, pe in enumerate(pred_events):
        for ri, re in enumerate(ref_events):
            if ri in matched_ref:
                continue
            class_ok = class_agnostic or (pe["class"] == re["class"])
            if class_ok and abs(pe["onset"] - re["onset"]) <= tolerance_sec:
                matched_pred.add(pi)
                matched_ref.add(ri)
                break

    return len(matched_ref), matched_pred, matched_ref


def _match_events_collar(pred_events, ref_events, collar_sec,
                         class_agnostic=False):
    """Collar-based matching: both onset and offset within collar.

    Returns:
        (tp, matched_pred_indices, matched_ref_indices)
    """
    matched_ref = set()
    matched_pred = set()

    for pi, pe in enumerate(pred_events):
        for ri, re in enumerate(ref_events):
            if ri in matched_ref:
                continue
            class_ok = class_agnostic or (pe["class"] == re["class"])
            onset_ok = abs(pe["onset"] - re["onset"]) <= collar_sec
            offset_ok = abs(pe["offset"] - re["offset"]) <= collar_sec
            if class_ok and onset_ok and offset_ok:
                matched_pred.add(pi)
                matched_ref.add(ri)
                break

    return len(matched_ref), matched_pred, matched_ref


def _compute_iou(pred_event, ref_event):
    """Temporal IoU (Intersection over Union) of two events."""
    inter_start = max(pred_event["onset"], ref_event["onset"])
    inter_end = min(pred_event["offset"], ref_event["offset"])
    intersection = max(0, inter_end - inter_start)

    union = ((pred_event["offset"] - pred_event["onset"])
             + (ref_event["offset"] - ref_event["onset"])
             - intersection)

    if union <= 0:
        return 0.0
    return intersection / union


def _match_events_iou(pred_events, ref_events, iou_threshold,
                      class_agnostic=False):
    """IoU-based matching: IoU >= threshold.

    Returns:
        (tp, matched_pred_indices, matched_ref_indices)
    """
    matched_ref = set()
    matched_pred = set()

    # greedy matching in order of highest IoU
    pairs = []
    for pi, pe in enumerate(pred_events):
        for ri, re in enumerate(ref_events):
            class_ok = class_agnostic or (pe["class"] == re["class"])
            if class_ok:
                iou = _compute_iou(pe, re)
                if iou >= iou_threshold:
                    pairs.append((iou, pi, ri))

    pairs.sort(key=lambda x: -x[0])  # IoU descending

    for iou, pi, ri in pairs:
        if pi in matched_pred or ri in matched_ref:
            continue
        matched_pred.add(pi)
        matched_ref.add(ri)

    return len(matched_ref), matched_pred, matched_ref


# ─── P/R/F1 computation ──────────────────────────────────

def _prf(tp, n_pred, n_ref):
    """Compute Precision/Recall/F1 from TP, number of pred, number of ref."""
    precision = tp / n_pred if n_pred > 0 else 0.0
    recall = tp / n_ref if n_ref > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1}


# ─── Individual metric functions ─────────────────────────

def onset_f1(pred_events, ref_events, tolerance_sec=0.1,
             class_agnostic=False):
    """Onset F1: matching by onset position only.

    Parameters:
        tolerance_sec: onset tolerance (default 100ms)
        class_agnostic: if True, ignore class (binary detection)
    """
    if not pred_events and not ref_events:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    tp, _, _ = _match_events_onset(
        pred_events, ref_events, tolerance_sec, class_agnostic)
    return _prf(tp, len(pred_events), len(ref_events))


def collar_event_f1(pred_events, ref_events, collar_sec=0.2,
                    class_agnostic=False):
    """Collar-based Event F1: both onset and offset within collar.

    Parameters:
        collar_sec: onset/offset tolerance (default 200ms)
        class_agnostic: if True, ignore class (binary detection)
    """
    if not pred_events and not ref_events:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    tp, _, _ = _match_events_collar(
        pred_events, ref_events, collar_sec, class_agnostic)
    return _prf(tp, len(pred_events), len(ref_events))


def iou_event_f1(pred_events, ref_events, iou_threshold=0.5,
                 class_agnostic=False):
    """IoU-based Event F1: temporal IoU >= threshold.

    Parameters:
        iou_threshold: IoU threshold (default 0.5)
        class_agnostic: if True, ignore class (binary detection)
    """
    if not pred_events and not ref_events:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    tp, _, _ = _match_events_iou(
        pred_events, ref_events, iou_threshold, class_agnostic)
    return _prf(tp, len(pred_events), len(ref_events))


# ─── Per-class Event F1 ──────────────────────────────────

def per_class_event_f1(pred_events, ref_events, n_classes,
                       metric_fn, **metric_kwargs):
    """Compute per-class event-level F1.

    Parameters:
        n_classes: total number of classes (including no_ornament)
        metric_fn: one of onset_f1, collar_event_f1, iou_event_f1
        **metric_kwargs: additional arguments to pass to metric_fn

    Returns:
        dict: {class_idx: {"precision", "recall", "f1", "support"}}
    """
    results = {}
    for cls in range(1, n_classes):  # exclude 0=no_ornament
        cls_pred = [e for e in pred_events if e["class"] == cls]
        cls_ref = [e for e in ref_events if e["class"] == cls]
        prf = metric_fn(cls_pred, cls_ref, class_agnostic=True, **metric_kwargs)
        prf["support"] = len(cls_ref)
        results[cls] = prf
    return results


# ─── Frame-level metrics ─────────────────────────────────

def frame_metrics(preds, targets, n_classes):
    """Frame-level Accuracy, Macro F1, Per-class P/R/F1.

    Assumes don't care frames are already excluded.

    Returns:
        dict with "accuracy", "macro_f1", "per_class"
    """
    if len(preds) == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "per_class": {}}

    accuracy = float((preds == targets).sum()) / len(preds)

    per_class = {}
    f1_list = []
    for i in range(n_classes):
        tp = int(((preds == i) & (targets == i)).sum())
        fp = int(((preds == i) & (targets != i)).sum())
        fn = int(((preds != i) & (targets == i)).sum())
        support = int((targets == i).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)
        per_class[i] = {
            "precision": prec, "recall": rec, "f1": f1, "support": support,
        }

    macro_f1 = float(np.mean(f1_list))
    return {"accuracy": accuracy, "macro_f1": macro_f1, "per_class": per_class}


# ─── Combined evaluation function ────────────────────────

def evaluate_all_metrics(pred_events, ref_events, n_classes=None):
    """Compute all event-level metrics at once.

    Parameters:
        pred_events: list of {"onset", "offset", "class"}
        ref_events: list of {"onset", "offset", "class"}
        n_classes: required for per-class computation (skip per-class if None)

    Returns:
        dict with all metric results
    """
    results = {}

    # ── Onset F1 ──
    for tol_ms in [50, 100, 200, 300]:
        tol = tol_ms / 1000.0
        r = onset_f1(pred_events, ref_events, tol, class_agnostic=False)
        results[f"onset_{tol_ms}ms"] = r
        r_ag = onset_f1(pred_events, ref_events, tol, class_agnostic=True)
        results[f"onset_{tol_ms}ms_agnostic"] = r_ag

    # ── Collar Event F1 ──
    for collar_ms in [50, 100, 200, 300]:
        collar = collar_ms / 1000.0
        r = collar_event_f1(pred_events, ref_events, collar, class_agnostic=False)
        results[f"event_{collar_ms}ms"] = r
        r_ag = collar_event_f1(pred_events, ref_events, collar, class_agnostic=True)
        results[f"event_{collar_ms}ms_agnostic"] = r_ag

    # ── IoU F1 ──
    for iou_thr in [0.3, 0.5]:
        r = iou_event_f1(pred_events, ref_events, iou_thr, class_agnostic=False)
        results[f"iou_{iou_thr}"] = r
        r_ag = iou_event_f1(pred_events, ref_events, iou_thr, class_agnostic=True)
        results[f"iou_{iou_thr}_agnostic"] = r_ag

    # ── Per-class (based on collar 200ms) ──
    if n_classes is not None:
        results["per_class_event"] = per_class_event_f1(
            pred_events, ref_events, n_classes,
            collar_event_f1, collar_sec=0.2,
        )

    return results


# ─── Model evaluation wrapper ────────────────────────────

def _median_filter_frames(frames, kernel_size=11):
    """Apply median filter to frame predictions to remove flickering."""
    from scipy.ndimage import median_filter
    return median_filter(frames, size=kernel_size).astype(frames.dtype)


def _merge_short_gaps(events, max_gap_sec=0.05):
    """Merge same-class events that are separated by a short gap."""
    if len(events) <= 1:
        return events
    events = sorted(events, key=lambda e: e["onset"])
    merged = [events[0].copy()]
    for e in events[1:]:
        prev = merged[-1]
        if (e["class"] == prev["class"]
                and (e["onset"] - prev["offset"]) <= max_gap_sec):
            prev["offset"] = e["offset"]
        else:
            merged.append(e.copy())
    return merged


def evaluate_model_events(model, loader, device, hop_sec, n_classes=None,
                          median_kernel=0, merge_gap_sec=0.0):
    """Model inference → event conversion → compute all metrics.

    Parameters:
        model: model that outputs frame-level logits
        loader: DataLoader (X, Y)
        device: torch device
        hop_sec: feature hop (seconds)
        n_classes: for per-class computation
        median_kernel: median filter size (0 to disable, odd recommended)
        merge_gap_sec: gap threshold for merging same-class events (seconds, 0 to disable)

    Returns:
        dict with all metrics + "n_pred_events", "n_ref_events"
    """
    import torch

    model.eval()
    all_pred_events = []
    all_ref_events = []

    with torch.no_grad():
        for X_batch, Y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            # output frame count and label frame count may differ (e.g. in MERT)
            T_label = Y_batch.shape[1]
            if logits.shape[2] != T_label:
                logits = torch.nn.functional.interpolate(
                    logits, size=T_label, mode="linear", align_corners=False,
                )
            preds = logits.argmax(dim=1).cpu().numpy()  # (B, T)
            targets = Y_batch.numpy()  # (B, T)

            for b in range(preds.shape[0]):
                # post-processing: median filter
                pred_b = preds[b]
                if median_kernel > 0:
                    pred_b = _median_filter_frames(pred_b, median_kernel)

                pred_ev = frames_to_events(pred_b, hop_sec)

                # post-processing: gap merge
                if merge_gap_sec > 0:
                    pred_ev = _merge_short_gaps(pred_ev, merge_gap_sec)

                ref_ev = frames_to_events(targets[b], hop_sec)
                all_pred_events.extend(pred_ev)
                all_ref_events.extend(ref_ev)

    results = evaluate_all_metrics(all_pred_events, all_ref_events, n_classes)
    results["n_pred_events"] = len(all_pred_events)
    results["n_ref_events"] = len(all_ref_events)
    return results


# ─── Output utilities ────────────────────────────────────

def print_event_metrics(results, class_names=None):
    """Pretty-print the evaluate_all_metrics result."""

    print("\n" + "=" * 65)
    print("  Event Detection Metrics")
    print("=" * 65)

    # Onset F1
    print("\n  Onset F1:")
    print(f"  {'Metric':30s} {'P':>7s} {'R':>7s} {'F1':>7s}")
    print(f"  {'-'*51}")
    for key in ["onset_50ms", "onset_100ms", "onset_200ms", "onset_300ms",
                "onset_50ms_agnostic", "onset_100ms_agnostic",
                "onset_200ms_agnostic", "onset_300ms_agnostic"]:
        if key in results:
            r = results[key]
            label = key.replace("_agnostic", " (agnostic)")
            print(f"  {label:30s} {r['precision']:7.4f} {r['recall']:7.4f} {r['f1']:7.4f}")

    # Event F1 (collar)
    print("\n  Event F1 (collar):")
    print(f"  {'Metric':30s} {'P':>7s} {'R':>7s} {'F1':>7s}")
    print(f"  {'-'*51}")
    for key in ["event_50ms", "event_100ms", "event_200ms", "event_300ms",
                "event_50ms_agnostic", "event_100ms_agnostic",
                "event_200ms_agnostic", "event_300ms_agnostic"]:
        if key in results:
            r = results[key]
            label = key.replace("_agnostic", " (agnostic)")
            print(f"  {label:30s} {r['precision']:7.4f} {r['recall']:7.4f} {r['f1']:7.4f}")

    # IoU F1
    print("\n  IoU F1:")
    print(f"  {'Metric':30s} {'P':>7s} {'R':>7s} {'F1':>7s}")
    print(f"  {'-'*51}")
    for key in ["iou_0.3", "iou_0.5",
                "iou_0.3_agnostic", "iou_0.5_agnostic"]:
        if key in results:
            r = results[key]
            label = key.replace("_agnostic", " (agnostic)")
            print(f"  {label:30s} {r['precision']:7.4f} {r['recall']:7.4f} {r['f1']:7.4f}")

    # Per-class
    if "per_class_event" in results and class_names is not None:
        print("\n  Per-class Event F1 (collar=200ms):")
        print(f"  {'Class':20s} {'P':>7s} {'R':>7s} {'F1':>7s} {'Support':>9s}")
        print(f"  {'-'*55}")
        pc = results["per_class_event"]
        f1_list = []
        for cls_idx in sorted(pc.keys()):
            r = pc[cls_idx]
            name = class_names[cls_idx] if cls_idx < len(class_names) else f"cls_{cls_idx}"
            print(f"  {name:20s} {r['precision']:7.4f} {r['recall']:7.4f} "
                  f"{r['f1']:7.4f} {r['support']:9d}")
            f1_list.append(r["f1"])
        if f1_list:
            macro = np.mean(f1_list)
            print(f"  {'-'*55}")
            print(f"  {'Macro avg':20s} {'':>7s} {'':>7s} {macro:7.4f}")

    # event counts
    if "n_pred_events" in results:
        print(f"\n  Pred events: {results['n_pred_events']}, "
              f"Ref events: {results['n_ref_events']}")

    print("=" * 65)
