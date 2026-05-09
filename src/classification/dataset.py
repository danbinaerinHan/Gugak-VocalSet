"""sigimsae F0 dataset class

Extracts segments from F0 contour + sigimsae label JSON and
manages them by splitting into train/val/test.

Usage:
    from dataset import SigimsaeF0Dataset

    ds = SigimsaeF0Dataset()
    ds.summary()

    X_train, y_train = ds.get_split("train")
    X_val, y_val = ds.get_split("val")
"""
import json
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset

from prepare_f0_dataset import (
    CLASS_NAMES,
    CLASS_NAMES2,
    LABEL_TO_GROUP,
    YOSEONG_LABELS,
    resolve_class,
    resolve_class2,
    FINE_CLASS_NAMES,
    resolve_class_fine,
    CLASS_NAMES_7CAT,
    resolve_class_7cat,
    CLASS_NAMES_17CAT,
    resolve_class_17cat,
    CLASS_SCHEMES,
    extract_f0_segment as _extract_f0_segment_base,
    pad_or_truncate,
    HOP_SEC,
)


def _compute_beat_phase(beats: np.ndarray, start_sec: float, n_frames: int,
                        hop_sec: float) -> np.ndarray:
    """Compute per-frame beat phase. (0~1, position within the current beat period)"""
    phase = np.zeros(n_frames, dtype=np.float32)
    if len(beats) < 2:
        return phase
    for t in range(n_frames):
        frame_time = start_sec + t * hop_sec
        idx = np.searchsorted(beats, frame_time, side="right")
        if idx == 0 or idx >= len(beats):
            # outside the beat range, set to 0
            continue
        prev_beat = beats[idx - 1]
        next_beat = beats[idx]
        interval = next_beat - prev_beat
        if interval > 0:
            phase[t] = (frame_time - prev_beat) / interval
    return phase


def _load_f0(f0_dir: Path, file_id: str, suffix: str) -> Optional[np.ndarray]:
    """Load the file corresponding to file_id from the F0 directory."""
    matches = list(f0_dir.glob(f"{file_id}_*_{suffix}.npy"))
    if not matches:
        return None
    return np.load(matches[0])


class SigimsaeF0Dataset:
    """sigimsae F0 segment dataset

    Extracts feature arrays of shape (N, max_frames, 3) from label JSON
    and RMVPE F0 files, and splits into train/val/test on a file basis.

    Parameters
    ----------
    label_dir : sigimsae label JSON directory
    f0_dir : RMVPE F0 numpy directory
    max_frames : fixed segment length (number of frames)
    pad_ms : onset/offset front/back padding (milliseconds)
    seed : random seed
    train_ratio, val_ratio, test_ratio : split ratios
    verbose : whether to print progress
    """

    def __init__(
        self,
        label_dir: str = "sigimsae_labels",
        f0_dir: str = "pitch_output/rmvpe",
        max_frames: int = 200,
        pad_ms: int = 0,
        seed: int = 42,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        verbose: bool = True,
        fine_grained: bool = False,
        group2: bool = False,
        split_by: str = "song",
        beat_dir: Optional[str] = None,
        class_scheme: Optional[str] = None,
        norm_mode: str = "segment_median",
    ):
        self.label_dir = Path(label_dir)
        self.f0_dir = Path(f0_dir)
        self.max_frames = max_frames
        self.pad_ms = pad_ms
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.verbose = verbose
        self.split_by = split_by
        self.beat_dir = Path(beat_dir) if beat_dir else None
        self.norm_mode = norm_mode

        # If class_scheme is specified, use it; otherwise fall back to fine_grained/group2 compatibility
        if class_scheme is not None:
            assert class_scheme in CLASS_SCHEMES, \
                f"class_scheme must be one of {list(CLASS_SCHEMES.keys())}, got '{class_scheme}'"
            self.class_scheme = class_scheme
        elif fine_grained:
            self.class_scheme = "17cat"
        elif group2:
            self.class_scheme = "9cat"
        else:
            self.class_scheme = "9cat"

        scheme = CLASS_SCHEMES[self.class_scheme]
        self.class_names = scheme["class_names"]
        self._resolve_fn = scheme["resolve_fn"]
        self.n_classes = scheme["n_classes"]

        # For backward compatibility (referenced by existing code)
        self.fine_grained = (self.class_scheme == "17cat")
        self.group2 = False

        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.meta: List[Dict] = []
        self.class_counter = Counter()
        self.skip_counter = Counter()

        self._split_indices: Dict[str, List[int]] = {}
        self._split_files: Dict[str, set] = {}

        self._extract_segments()
        self._split()

    # ── Internal methods ──────────────────────────────────────

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _extract_f0_segment(self, f0, times, start_sec, end_sec, song_ref_f0=None):
        """Extend front/back by pad_ms, then extract the segment."""
        pad_sec = self.pad_ms / 1000.0
        adj_start = max(0.0, start_sec - pad_sec)
        adj_end = end_sec + pad_sec
        if pad_sec > 0:
            return _extract_f0_segment_base(
                f0, times, adj_start, adj_end,
                ref_start=start_sec, ref_end=end_sec,
                norm_mode=self.norm_mode, song_ref_f0=song_ref_f0,
            )
        return _extract_f0_segment_base(
            f0, times, adj_start, adj_end,
            norm_mode=self.norm_mode, song_ref_f0=song_ref_f0,
        )

    def _load_beats(self, file_stem: str) -> Optional[np.ndarray]:
        """Load the beats array from beat_output."""
        if self.beat_dir is None:
            return None
        beat_file = self.beat_dir / f"{file_stem}.json"
        if not beat_file.exists():
            return None
        with open(beat_file, encoding="utf-8") as f:
            beat_data = json.load(f)
        return np.array(beat_data.get("beats", []), dtype=np.float64)

    def _extract_segments(self):
        np.random.seed(self.seed)

        label_files = sorted(self.label_dir.glob("*.json"))
        self._log(f"Label files: {len(label_files)}")
        self._log(f"F0 padding: {self.pad_ms}ms")
        if self.beat_dir:
            self._log(f"Beat directory: {self.beat_dir}")

        all_features: List[np.ndarray] = []
        all_labels: List[int] = []

        for i, label_file in enumerate(label_files):
            file_id = label_file.stem.split("_")[0]

            f0 = _load_f0(self.f0_dir, file_id, "f0")
            times = _load_f0(self.f0_dir, file_id, "times")
            if f0 is None or times is None:
                self.skip_counter["f0_missing"] += 1
                continue

            # song-level reference f0 (voiced median over the whole song)
            song_ref_f0 = None
            if self.norm_mode == "song_median":
                voiced_song = f0[f0 > 0]
                if len(voiced_song) >= 2:
                    song_ref_f0 = float(np.median(voiced_song))

            beats = self._load_beats(label_file.stem)

            with open(label_file, encoding="utf-8") as f:
                data = json.load(f)
            regions = data.get("annotations", {}).get("sigimsaeRegions", [])

            for region in regions:
                labels = region.get("sigimsae", [])
                if not labels:
                    self.skip_counter["no_label"] += 1
                    continue

                cls = self._resolve_fn(labels)
                if cls is None:
                    self.skip_counter["unknown_label"] += 1
                    continue

                features = self._extract_f0_segment(
                    f0, times, region["start_sec"], region["end_sec"],
                    song_ref_f0=song_ref_f0,
                )
                if features is None:
                    self.skip_counter["too_few_voiced"] += 1
                    continue

                # add beat_phase channel
                if self.beat_dir is not None:
                    n_frames = features.shape[0]
                    if beats is not None:
                        pad_sec = self.pad_ms / 1000.0
                        adj_start = max(0.0, region["start_sec"] - pad_sec)
                        bp = _compute_beat_phase(beats, adj_start, n_frames, HOP_SEC)
                    else:
                        bp = np.zeros(n_frames, dtype=np.float32)
                    features = np.column_stack([features, bp])

                features = pad_or_truncate(features, self.max_frames)
                all_features.append(features)
                all_labels.append(cls)
                self.meta.append({
                    "file": label_file.stem,
                    "sigimsae_id": region.get("sigimsae_id", ""),
                    "start_sec": region["start_sec"],
                    "end_sec": region["end_sec"],
                    "duration": round(region["end_sec"] - region["start_sec"], 4),
                    "original_labels": labels,
                    "class": cls,
                    "class_name": self.class_names[cls],
                })
                self.class_counter[cls] += 1

            if self.verbose and (i + 1) % 50 == 0:
                self._log(
                    f"  [{i+1}/{len(label_files)}] "
                    f"extracting {len(all_features)} segments..."
                )

        if all_features:
            self.X = np.array(all_features, dtype=np.float32)
            self.y = np.array(all_labels, dtype=np.int64)
        else:
            self.X = np.empty((0,), dtype=np.float32)
            self.y = np.empty((0,), dtype=np.int64)

        self._log(f"\nTotal valid segments: {len(self.X)}  |  Shape: {self.X.shape}")
        self._log(f"Skipped: {dict(self.skip_counter)}")

    def _split(self):
        if len(self.meta) == 0:
            for k in ("train", "val", "test"):
                self._split_indices[k] = []
                self._split_files[k] = set()
            return

        np.random.seed(self.seed)

        if self.split_by == "random":
            # segment-level random split
            indices = np.arange(len(self.meta))
            np.random.shuffle(indices)
            n_total = len(indices)
            n_train = int(n_total * self.train_ratio)
            n_val = int(n_total * self.val_ratio)

            train_idx = indices[:n_train].tolist()
            val_idx = indices[n_train:n_train + n_val].tolist()
            test_idx = indices[n_train + n_val:].tolist()

            self._split_indices = {"train": train_idx, "val": val_idx, "test": test_idx}
            self._split_files = {
                k: {self.meta[i]["file"] for i in idx}
                for k, idx in self._split_indices.items()
            }
        else:
            # song-level (file-level) split (default)
            file_names = sorted({m["file"] for m in self.meta})
            np.random.shuffle(file_names)

            n_files = len(file_names)
            n_train = int(n_files * self.train_ratio)
            n_val = int(n_files * self.val_ratio)

            train_files = set(file_names[:n_train])
            val_files = set(file_names[n_train:n_train + n_val])
            test_files = set(file_names[n_train + n_val:])

            self._split_files = {
                "train": train_files,
                "val": val_files,
                "test": test_files,
            }
            self._split_indices = {
                "train": [i for i, m in enumerate(self.meta) if m["file"] in train_files],
                "val": [i for i, m in enumerate(self.meta) if m["file"] in val_files],
                "test": [i for i, m in enumerate(self.meta) if m["file"] in test_files],
            }

    # ── Public API ─────────────────────────────────────────

    def get_split(self, split: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return split data as (X, y) numpy arrays."""
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        idx = self._split_indices[split]
        return self.X[idx], self.y[idx]

    def get_split_meta(self, split: str) -> List[Dict]:
        """Return the metadata list for the split."""
        assert split in ("train", "val", "test")
        return [self.meta[i] for i in self._split_indices[split]]

    def get_split_files(self, split: str) -> set:
        """Return the set of file names included in the split."""
        assert split in ("train", "val", "test")
        return self._split_files[split]

    @property
    def n_segments(self) -> int:
        return len(self.X)

    @property
    def feature_shape(self) -> Tuple[int, ...]:
        return self.X.shape[1:] if self.X.ndim > 1 else (0,)

    def summary(self):
        """Print dataset summary information."""
        print(f"\n{'='*55}")
        print(f"  SigimsaeF0Dataset")
        print(f"{'='*55}")
        print(f"  Total segments: {self.n_segments}  |  Shape: {self.X.shape}")
        print(f"  max_frames={self.max_frames}  pad_ms={self.pad_ms}  seed={self.seed}")
        print(f"  Skipped: {dict(self.skip_counter)}")

        print(f"\n  Class distribution:")
        for cls_idx in range(self.n_classes):
            cnt = self.class_counter.get(cls_idx, 0)
            pct = cnt / self.n_segments * 100 if self.n_segments > 0 else 0
            print(f"    [{cls_idx}] {self.class_names[cls_idx]:10s}: {cnt:6d} ({pct:5.1f}%)")

        print(f"\n  Split (ratio={self.train_ratio}/{self.val_ratio}/{self.test_ratio}):")
        for split in ("train", "val", "test"):
            idx = self._split_indices[split]
            files = self._split_files[split]
            print(f"    {split.capitalize():5s}: {len(idx):6d} segments ({len(files)} files)")

        for split in ("train", "val", "test"):
            idx = self._split_indices[split]
            if not idx:
                continue
            split_labels = self.y[idx]
            print(f"\n    {split.capitalize()} class distribution:")
            for cls_idx in range(self.n_classes):
                cnt = int((split_labels == cls_idx).sum())
                print(f"      [{cls_idx}] {self.class_names[cls_idx]:10s}: {cnt}")
        print(f"{'='*55}")

    def get_torch_dataset(self, split: str, augment: bool = False,
                          aug_pitch_shift_range: float = 0.1,
                          aug_time_stretch_range: Tuple[float, float] = (0.85, 1.15),
                          aug_noise_std: float = 0.015,
                          aug_prob: float = 0.3) -> "SigimsaeAugDataset":
        """Return a Dataset for PyTorch DataLoader."""
        X_np, y_np = self.get_split(split)
        X_t = torch.FloatTensor(X_np)
        y_t = torch.LongTensor(y_np)
        return SigimsaeAugDataset(
            X_t, y_t, augment=augment,
            aug_pitch_shift_range=aug_pitch_shift_range,
            aug_time_stretch_range=aug_time_stretch_range,
            aug_noise_std=aug_noise_std,
            aug_prob=aug_prob,
        )

    def to_metadata_dict(self) -> Dict:
        """Return metadata as a dictionary (for JSON saving)."""
        return {
            "class_names": self.class_names,
            "max_frames": self.max_frames,
            "hop_sec": HOP_SEC,
            "pad_ms": self.pad_ms,
            "n_features": 3,
            "feature_names": ["norm_f0_cent", "voicing", "delta_f0"],
            "total_segments": self.n_segments,
            "train_segments": len(self._split_indices["train"]),
            "val_segments": len(self._split_indices["val"]),
            "test_segments": len(self._split_indices["test"]),
            "train_files": sorted(self._split_files.get("train", set())),
            "val_files": sorted(self._split_files.get("val", set())),
            "test_files": sorted(self._split_files.get("test", set())),
            "class_distribution": {
                self.class_names[k]: v
                for k, v in sorted(self.class_counter.items())
            },
        }


class SigimsaeAugDataset(TorchDataset):
    """Dataset for PyTorch DataLoader (with augmentation).

    Can be created via SigimsaeF0Dataset.get_torch_dataset()
    or directly by passing X (FloatTensor) and y (LongTensor).
    """

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        augment: bool = False,
        aug_pitch_shift_range: float = 0.1,
        aug_time_stretch_range: Tuple[float, float] = (0.85, 1.15),
        aug_noise_std: float = 0.015,
        aug_prob: float = 0.3,
    ):
        self.X = X
        self.y = y
        self.augment = augment
        self.aug_pitch_shift_range = aug_pitch_shift_range
        self.aug_time_stretch_range = aug_time_stretch_range
        self.aug_noise_std = aug_noise_std
        self.aug_prob = aug_prob

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx]
        if self.augment:
            x = self._augment(x)
        return x, y

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        # 1) Pitch shift: shift voiced norm_f0 by ±range
        if torch.rand(1).item() < self.aug_prob:
            shift = (torch.rand(1).item() * 2 - 1) * self.aug_pitch_shift_range
            voiced = x[:, 1] > 0.5
            x[voiced, 0] = torch.clamp(x[voiced, 0] + shift, -1.0, 1.0)

        # 2) Time stretch: stretch or shrink the sequence by rate, then resize to original length
        if torch.rand(1).item() < self.aug_prob:
            lo, hi = self.aug_time_stretch_range
            rate = lo + torch.rand(1).item() * (hi - lo)
            seq_len = x.shape[0]
            n_ch = x.shape[1]
            x_t = x.permute(1, 0).unsqueeze(0)
            new_len = int(round(seq_len * rate))
            x_t = F.interpolate(x_t, size=new_len, mode="linear", align_corners=False)
            x_t = x_t.squeeze(0).permute(1, 0)
            if x_t.shape[0] >= seq_len:
                start = (x_t.shape[0] - seq_len) // 2
                x = x_t[start:start + seq_len]
            else:
                padded = torch.zeros(seq_len, n_ch)
                padded[:x_t.shape[0]] = x_t
                x = padded
            # Beat channels (idx>=3): continuous values (phase), so apply clamp only
            if n_ch > 3:
                x[:, 3:] = x[:, 3:].clamp(0, 1)

        # 3) Gaussian noise: add small noise to voiced norm_f0
        if torch.rand(1).item() < self.aug_prob:
            voiced = x[:, 1] > 0.5
            noise = torch.randn(voiced.sum()) * self.aug_noise_std
            x[voiced, 0] = torch.clamp(x[voiced, 0] + noise, -1.0, 1.0)

        return x


# ══════════════════════════════════════════════════════════
#  Mel Spectrogram dataset
# ══════════════════════════════════════════════════════════

class SigimsaeMelDataset:
    """sigimsae Mel Spectrogram dataset

    Computes mel spectrograms from audio files, then slices segments
    aligned to label time intervals to form shape (N, 1, n_mels, max_mel_frames).

    Parameters
    ----------
    label_dir : sigimsae label JSON directory
    audio_dir : audio file directory
    sr : audio sample rate
    n_mels : number of mel filter banks
    n_fft : FFT window size
    hop_length : mel hop length (samples)
    max_mel_frames : fixed segment frame count
    pad_ms : onset/offset front/back padding (milliseconds)
    seed, train_ratio, val_ratio, test_ratio : split settings
    fine_grained : if True, 18-class; if False, 9-class
    """

    def __init__(
        self,
        label_dir: str = "sigimsae_labels",
        audio_dir: str = "KVocSet",
        mel_dir: str = "features/mel_output",
        sr: int = 22050,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 256,
        max_mel_frames: int = 256,
        pad_ms: int = 0,
        seed: int = 42,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        verbose: bool = True,
        fine_grained: bool = False,
        group2: bool = False,
        split_by: str = "song",
        class_scheme: Optional[str] = None,
    ):
        self.label_dir = Path(label_dir)
        self.audio_dir = Path(audio_dir)
        self.mel_dir = Path(mel_dir)
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_mel_frames = max_mel_frames
        self.pad_ms = pad_ms
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.verbose = verbose
        self.split_by = split_by

        # If class_scheme is specified, use it; otherwise fall back to fine_grained/group2 compatibility
        if class_scheme is not None:
            assert class_scheme in CLASS_SCHEMES, \
                f"class_scheme must be one of {list(CLASS_SCHEMES.keys())}, got '{class_scheme}'"
            self.class_scheme = class_scheme
        elif fine_grained:
            self.class_scheme = "17cat"
        elif group2:
            self.class_scheme = "9cat"
        else:
            self.class_scheme = "9cat"

        scheme = CLASS_SCHEMES[self.class_scheme]
        self.class_names = scheme["class_names"]
        self._resolve_fn = scheme["resolve_fn"]
        self.n_classes = scheme["n_classes"]

        # For backward compatibility
        self.fine_grained = (self.class_scheme == "17cat")
        self.group2 = False

        self.mel_hop_sec = hop_length / sr

        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.meta: List[Dict] = []
        self.class_counter = Counter()
        self.skip_counter = Counter()

        self._split_indices: Dict[str, List[int]] = {}
        self._split_files: Dict[str, set] = {}

        self._extract_segments()
        self._split()

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _compute_mel(self, audio_path: Path) -> Optional[np.ndarray]:
        """Compute log-mel spectrogram from an audio file.

        Returns: (n_mels, T) float32 array
        """
        import librosa
        try:
            y, _ = librosa.load(audio_path, sr=self.sr)
        except Exception as e:
            self._log(f"  Audio load failed: {audio_path.name} ({e})")
            return None

        mel = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_fft=self.n_fft,
            hop_length=self.hop_length, n_mels=self.n_mels,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)  # (n_mels, T)
        # Normalize to [0, 1] range
        log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)
        return log_mel.astype(np.float32)

    def _load_precomputed_mel(self, file_id: str) -> Optional[np.ndarray]:
        """Load a pre-extracted mel spectrogram .npy file."""
        config_name = f"{self.sr}_{self.n_mels}_{self.n_fft}_{self.hop_length}"
        mel_path = self.mel_dir / config_name / f"{file_id}.npy"
        if mel_path.exists():
            return np.load(mel_path)
        return None

    def _slice_mel(self, mel: np.ndarray, start_sec: float, end_sec: float) -> np.ndarray:
        """Slice a time interval from the mel spectrogram and pad/crop to fixed length.

        Returns: (n_mels, max_mel_frames) array
        """
        pad_sec = self.pad_ms / 1000.0
        adj_start = max(0.0, start_sec - pad_sec)
        adj_end = end_sec + pad_sec

        start_frame = int(round(adj_start / self.mel_hop_sec))
        end_frame = int(round(adj_end / self.mel_hop_sec))

        start_frame = max(0, start_frame)
        end_frame = min(mel.shape[1], end_frame)

        if end_frame <= start_frame:
            return None

        segment = mel[:, start_frame:end_frame]  # (n_mels, n_frames)
        n_frames = segment.shape[1]

        if n_frames == self.max_mel_frames:
            return segment
        elif n_frames > self.max_mel_frames:
            # center crop
            start = (n_frames - self.max_mel_frames) // 2
            return segment[:, start:start + self.max_mel_frames]
        else:
            # zero pad (at the end)
            padded = np.zeros((self.n_mels, self.max_mel_frames), dtype=np.float32)
            padded[:, :n_frames] = segment
            return padded

    def _extract_segments(self):
        np.random.seed(self.seed)

        label_files = sorted(self.label_dir.glob("*.json"))
        self._log(f"Label files: {len(label_files)}")
        self._log(f"Mel config: sr={self.sr}, n_mels={self.n_mels}, "
                   f"n_fft={self.n_fft}, hop={self.hop_length}, "
                   f"max_frames={self.max_mel_frames}")

        all_features: List[np.ndarray] = []
        all_labels: List[int] = []

        # audio file cache (file_id -> mel)
        mel_cache: Dict[str, Optional[np.ndarray]] = {}

        for i, label_file in enumerate(label_files):
            file_id = label_file.stem.split("_")[0]

            # check/compute mel cache (prefer pre-extracted; fall back to computing from audio)
            if file_id not in mel_cache:
                precomputed = self._load_precomputed_mel(file_id)
                if precomputed is not None:
                    mel_cache[file_id] = precomputed
                else:
                    audio_matches = list(self.audio_dir.glob(f"{file_id}_*.mp3"))
                    if not audio_matches:
                        audio_matches = list(self.audio_dir.glob(f"{file_id}_*.wav"))
                    if not audio_matches:
                        mel_cache[file_id] = None
                    else:
                        mel_cache[file_id] = self._compute_mel(audio_matches[0])

            mel = mel_cache[file_id]
            if mel is None:
                self.skip_counter["audio_missing"] += 1
                continue

            with open(label_file, encoding="utf-8") as f:
                data = json.load(f)
            regions = data.get("annotations", {}).get("sigimsaeRegions", [])

            for region in regions:
                labels = region.get("sigimsae", [])
                if not labels:
                    self.skip_counter["no_label"] += 1
                    continue

                cls = self._resolve_fn(labels)
                if cls is None:
                    self.skip_counter["unknown_label"] += 1
                    continue

                segment = self._slice_mel(mel, region["start_sec"], region["end_sec"])
                if segment is None:
                    self.skip_counter["empty_segment"] += 1
                    continue

                all_features.append(segment)
                all_labels.append(cls)
                self.meta.append({
                    "file": label_file.stem,
                    "sigimsae_id": region.get("sigimsae_id", ""),
                    "start_sec": region["start_sec"],
                    "end_sec": region["end_sec"],
                    "duration": round(region["end_sec"] - region["start_sec"], 4),
                    "original_labels": labels,
                    "class": cls,
                    "class_name": self.class_names[cls],
                })
                self.class_counter[cls] += 1

            if self.verbose and (i + 1) % 50 == 0:
                self._log(
                    f"  [{i+1}/{len(label_files)}] "
                    f"extracting {len(all_features)} segments..."
                )

        if all_features:
            # (N, n_mels, max_mel_frames) -> (N, 1, n_mels, max_mel_frames)
            stacked = np.array(all_features, dtype=np.float32)
            self.X = stacked[:, np.newaxis, :, :]
            self.y = np.array(all_labels, dtype=np.int64)
        else:
            self.X = np.empty((0, 1, self.n_mels, self.max_mel_frames), dtype=np.float32)
            self.y = np.empty((0,), dtype=np.int64)

        self._log(f"\nTotal valid segments: {len(self.X)}  |  Shape: {self.X.shape}")
        self._log(f"Skipped: {dict(self.skip_counter)}")

    def _split(self):
        if len(self.meta) == 0:
            for k in ("train", "val", "test"):
                self._split_indices[k] = []
                self._split_files[k] = set()
            return

        np.random.seed(self.seed)

        if self.split_by == "random":
            # segment-level random split
            indices = np.arange(len(self.meta))
            np.random.shuffle(indices)
            n_total = len(indices)
            n_train = int(n_total * self.train_ratio)
            n_val = int(n_total * self.val_ratio)

            train_idx = indices[:n_train].tolist()
            val_idx = indices[n_train:n_train + n_val].tolist()
            test_idx = indices[n_train + n_val:].tolist()

            self._split_indices = {"train": train_idx, "val": val_idx, "test": test_idx}
            self._split_files = {
                k: {self.meta[i]["file"] for i in idx}
                for k, idx in self._split_indices.items()
            }
        else:
            # song-level (file-level) split (default)
            file_names = sorted({m["file"] for m in self.meta})
            np.random.shuffle(file_names)

            n_files = len(file_names)
            n_train = int(n_files * self.train_ratio)
            n_val = int(n_files * self.val_ratio)

            train_files = set(file_names[:n_train])
            val_files = set(file_names[n_train:n_train + n_val])
            test_files = set(file_names[n_train + n_val:])

            self._split_files = {"train": train_files, "val": val_files, "test": test_files}
            self._split_indices = {
                "train": [i for i, m in enumerate(self.meta) if m["file"] in train_files],
                "val": [i for i, m in enumerate(self.meta) if m["file"] in val_files],
                "test": [i for i, m in enumerate(self.meta) if m["file"] in test_files],
            }

    # ── Public API ─────────────────────────────────────────

    def get_split(self, split: str) -> Tuple[np.ndarray, np.ndarray]:
        assert split in ("train", "val", "test")
        idx = self._split_indices[split]
        return self.X[idx], self.y[idx]

    @property
    def n_segments(self) -> int:
        return len(self.X)

    def summary(self):
        print(f"\n{'='*55}")
        print(f"  SigimsaeMelDataset")
        print(f"{'='*55}")
        print(f"  Total segments: {self.n_segments}  |  Shape: {self.X.shape}")
        print(f"  sr={self.sr}  n_mels={self.n_mels}  n_fft={self.n_fft}  "
              f"hop={self.hop_length}  max_frames={self.max_mel_frames}")
        print(f"  mel_hop_sec={self.mel_hop_sec:.4f}s  pad_ms={self.pad_ms}  seed={self.seed}")
        print(f"  Skipped: {dict(self.skip_counter)}")

        print(f"\n  Class distribution:")
        for cls_idx in range(self.n_classes):
            cnt = self.class_counter.get(cls_idx, 0)
            pct = cnt / self.n_segments * 100 if self.n_segments > 0 else 0
            print(f"    [{cls_idx}] {self.class_names[cls_idx]:10s}: {cnt:6d} ({pct:5.1f}%)")

        print(f"\n  Split (ratio={self.train_ratio}/{self.val_ratio}/{self.test_ratio}):")
        for split in ("train", "val", "test"):
            idx = self._split_indices[split]
            files = self._split_files[split]
            print(f"    {split.capitalize():5s}: {len(idx):6d} segments ({len(files)} files)")

        for split in ("train", "val", "test"):
            idx = self._split_indices[split]
            if not idx:
                continue
            split_labels = self.y[idx]
            print(f"\n    {split.capitalize()} class distribution:")
            for cls_idx in range(self.n_classes):
                cnt = int((split_labels == cls_idx).sum())
                print(f"      [{cls_idx}] {self.class_names[cls_idx]:10s}: {cnt}")
        print(f"{'='*55}")

    def get_torch_dataset(self, split: str, augment: bool = False,
                          freq_mask_param: int = 8,
                          time_mask_param: int = 16,
                          aug_prob: float = 0.3) -> "SigimsaeMelAugDataset":
        X_np, y_np = self.get_split(split)
        X_t = torch.FloatTensor(X_np)
        y_t = torch.LongTensor(y_np)
        return SigimsaeMelAugDataset(
            X_t, y_t, augment=augment,
            freq_mask_param=freq_mask_param,
            time_mask_param=time_mask_param,
            aug_prob=aug_prob,
        )


class SigimsaeMelAugDataset(TorchDataset):
    """PyTorch Dataset for mel spectrograms (with SpecAugment).

    Augmentation:
      - Frequency masking: mask a random frequency band to 0
      - Time masking: mask a random time interval to 0
    """

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        augment: bool = False,
        freq_mask_param: int = 8,
        time_mask_param: int = 16,
        aug_prob: float = 0.3,
    ):
        self.X = X  # (N, 1, n_mels, n_frames)
        self.y = y
        self.augment = augment
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.aug_prob = aug_prob

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx]
        if self.augment:
            x = self._augment(x)
        return x, y

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """SpecAugment: frequency masking + time masking."""
        # x shape: (1, n_mels, n_frames)
        _, n_mels, n_frames = x.shape

        # Frequency masking
        if torch.rand(1).item() < self.aug_prob:
            f = int(torch.randint(0, self.freq_mask_param + 1, (1,)).item())
            f0 = int(torch.randint(0, max(1, n_mels - f), (1,)).item())
            x[:, f0:f0 + f, :] = 0

        # Time masking
        if torch.rand(1).item() < self.aug_prob:
            t = int(torch.randint(0, self.time_mask_param + 1, (1,)).item())
            t0 = int(torch.randint(0, max(1, n_frames - t), (1,)).item())
            x[:, :, t0:t0 + t] = 0

        return x


# ══════════════════════════════════════════════════════════
#  Waveform dataset (for pre-trained models such as MERT)
# ══════════════════════════════════════════════════════════

class SigimsaeWaveformDataset:
    """sigimsae raw waveform dataset (for pre-trained models such as MERT)

    Extracts raw waveforms of sigimsae intervals from audio files.
    Uses the same label/split logic as the Mel dataset.

    Parameters
    ----------
    label_dir : sigimsae label JSON directory
    audio_dir : audio file directory
    sr : target sample rate (MERT: 24000)
    duration : segment length (seconds)
    pad_ms : onset/offset front/back padding (milliseconds)
    class_scheme : class scheme ("7cat", "9cat", "17cat", "total")
    """

    def __init__(
        self,
        label_dir: str = "precheck_sigimsae_labels",
        audio_dir: str = "KVocSet",
        sr: int = 24000,
        duration: float = 3.0,
        pad_ms: int = 0,
        seed: int = 42,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        verbose: bool = True,
        split_by: str = "song",
        class_scheme: Optional[str] = "7cat",
        fine_grained: bool = False,
        group2: bool = False,
        cache_dir: str = "waveform_cache",
    ):
        self.label_dir = Path(label_dir)
        self.audio_dir = Path(audio_dir)
        self.sr = sr
        self.duration = duration
        self.target_samples = int(sr * duration)
        self.pad_ms = pad_ms
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.verbose = verbose
        self.split_by = split_by
        self.cache_dir = Path(cache_dir)

        if class_scheme is not None:
            assert class_scheme in CLASS_SCHEMES, \
                f"class_scheme must be one of {list(CLASS_SCHEMES.keys())}"
            self.class_scheme = class_scheme
        elif fine_grained:
            self.class_scheme = "17cat"
        else:
            self.class_scheme = "9cat"

        scheme = CLASS_SCHEMES[self.class_scheme]
        self.class_names = scheme["class_names"]
        self._resolve_fn = scheme["resolve_fn"]
        self.n_classes = scheme["n_classes"]

        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.meta: List[Dict] = []
        self.class_counter = Counter()
        self.skip_counter = Counter()

        self._split_indices: Dict[str, List[int]] = {}
        self._split_files: Dict[str, set] = {}

        # If a cache file exists, load from cache; otherwise extract from original audio
        if self._load_from_cache():
            self._log("Loaded from cache")
        else:
            self._log("No cache found — extracting from original audio")
            self._extract_segments()
        self._split()

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _get_cache_path(self) -> Path:
        return self.cache_dir / f"waveform_{self.sr}hz_{self.duration}s_{self.class_scheme}.npz"

    def _get_cache_meta_path(self) -> Path:
        return self.cache_dir / f"waveform_{self.sr}hz_{self.duration}s_{self.class_scheme}_meta.json"

    def _load_from_cache(self) -> bool:
        """If a cache file exists, load it and populate self.X, self.y, self.meta."""
        cache_path = self._get_cache_path()
        meta_path = self._get_cache_meta_path()

        if not cache_path.exists() or not meta_path.exists():
            return False

        self._log(f"Cache load: {cache_path}")
        data = np.load(cache_path)
        self.X = data["X"]
        self.y = data["y"]

        with open(meta_path, encoding="utf-8") as f:
            meta_json = json.load(f)

        self.meta = meta_json.get("segments", [])
        # restore class_counter
        for m in self.meta:
            self.class_counter[m["class"]] += 1

        self._log(f"  Shape: {self.X.shape}  |  Segments: {len(self.X)}")
        return True

    def _load_audio(self, audio_path: Path) -> Optional[np.ndarray]:
        """Load an audio file and return as a mono waveform (numpy)."""
        import librosa
        try:
            y, _ = librosa.load(audio_path, sr=self.sr, mono=True)
            return y
        except Exception as e:
            self._log(f"  Audio load failed: {audio_path.name} ({e})")
            return None

    def _slice_waveform(self, waveform: np.ndarray,
                        start_sec: float, end_sec: float) -> np.ndarray:
        """Slice a time interval from the waveform and pad/crop to fixed length.

        Returns: (target_samples,) array
        """
        pad_sec = self.pad_ms / 1000.0
        adj_start = max(0.0, start_sec - pad_sec)
        adj_end = end_sec + pad_sec

        start_sample = int(round(adj_start * self.sr))
        end_sample = int(round(adj_end * self.sr))

        start_sample = max(0, start_sample)
        end_sample = min(len(waveform), end_sample)

        if end_sample <= start_sample:
            return None

        segment = waveform[start_sample:end_sample]
        n_samples = len(segment)

        if n_samples == self.target_samples:
            return segment
        elif n_samples > self.target_samples:
            # center crop
            start = (n_samples - self.target_samples) // 2
            return segment[start:start + self.target_samples]
        else:
            # zero pad (at the end)
            padded = np.zeros(self.target_samples, dtype=np.float32)
            padded[:n_samples] = segment
            return padded

    def _extract_segments(self):
        np.random.seed(self.seed)

        label_files = sorted(self.label_dir.glob("*.json"))
        self._log(f"Label files: {len(label_files)}")
        self._log(f"Waveform config: sr={self.sr}, duration={self.duration}s, "
                   f"target_samples={self.target_samples}")

        all_features: List[np.ndarray] = []
        all_labels: List[int] = []

        # audio file cache (file_id -> waveform)
        audio_cache: Dict[str, Optional[np.ndarray]] = {}

        for i, label_file in enumerate(label_files):
            file_id = label_file.stem.split("_")[0]

            if file_id not in audio_cache:
                audio_matches = list(self.audio_dir.glob(f"{file_id}_*.mp3"))
                if not audio_matches:
                    audio_matches = list(self.audio_dir.glob(f"{file_id}_*.wav"))
                if not audio_matches:
                    audio_cache[file_id] = None
                else:
                    audio_cache[file_id] = self._load_audio(audio_matches[0])

            waveform = audio_cache[file_id]
            if waveform is None:
                self.skip_counter["audio_missing"] += 1
                continue

            with open(label_file, encoding="utf-8") as f:
                data = json.load(f)
            regions = data.get("annotations", {}).get("sigimsaeRegions", [])

            for region in regions:
                labels = region.get("sigimsae", [])
                if not labels:
                    self.skip_counter["no_label"] += 1
                    continue

                cls = self._resolve_fn(labels)
                if cls is None:
                    self.skip_counter["unknown_label"] += 1
                    continue

                segment = self._slice_waveform(
                    waveform, region["start_sec"], region["end_sec"]
                )
                if segment is None:
                    self.skip_counter["too_short"] += 1
                    continue

                all_features.append(segment)
                all_labels.append(cls)
                self.meta.append({
                    "file": label_file.stem,
                    "sigimsae_id": region.get("sigimsae_id", ""),
                    "start_sec": region["start_sec"],
                    "end_sec": region["end_sec"],
                    "duration": round(region["end_sec"] - region["start_sec"], 4),
                    "original_labels": labels,
                    "class": cls,
                    "class_name": self.class_names[cls],
                })
                self.class_counter[cls] += 1

            if self.verbose and (i + 1) % 50 == 0:
                self._log(
                    f"  [{i+1}/{len(label_files)}] "
                    f"extracting {len(all_features)} segments..."
                )

        if all_features:
            self.X = np.array(all_features, dtype=np.float32)  # (N, target_samples)
            self.y = np.array(all_labels, dtype=np.int64)
        else:
            self.X = np.empty((0,), dtype=np.float32)
            self.y = np.empty((0,), dtype=np.int64)

        self._log(f"\nTotal valid segments: {len(self.X)}  |  Shape: {self.X.shape}")
        self._log(f"Skipped: {dict(self.skip_counter)}")

    def _split(self):
        if len(self.meta) == 0:
            for k in ("train", "val", "test"):
                self._split_indices[k] = []
                self._split_files[k] = set()
            return

        np.random.seed(self.seed)

        if self.split_by == "random":
            indices = np.arange(len(self.meta))
            np.random.shuffle(indices)
            n_train = int(len(indices) * self.train_ratio)
            n_val = int(len(indices) * self.val_ratio)
            self._split_indices = {
                "train": indices[:n_train].tolist(),
                "val": indices[n_train:n_train + n_val].tolist(),
                "test": indices[n_train + n_val:].tolist(),
            }
        else:
            file_names = sorted({m["file"] for m in self.meta})
            np.random.shuffle(file_names)
            n_files = len(file_names)
            n_train = int(n_files * self.train_ratio)
            n_val = int(n_files * self.val_ratio)
            train_files = set(file_names[:n_train])
            val_files = set(file_names[n_train:n_train + n_val])
            test_files = set(file_names[n_train + n_val:])
            self._split_files = {
                "train": train_files, "val": val_files, "test": test_files,
            }
            self._split_indices = {
                "train": [i for i, m in enumerate(self.meta) if m["file"] in train_files],
                "val": [i for i, m in enumerate(self.meta) if m["file"] in val_files],
                "test": [i for i, m in enumerate(self.meta) if m["file"] in test_files],
            }

    def get_split(self, split: str):
        indices = self._split_indices[split]
        return self.X[indices], self.y[indices]

    def get_torch_dataset(self, split: str, augment: bool = False,
                          aug_prob: float = 0.3):
        X_np, y_np = self.get_split(split)
        X_t = torch.FloatTensor(X_np)
        y_t = torch.LongTensor(y_np)
        return SigimsaeWaveformAugDataset(X_t, y_t, augment=augment, aug_prob=aug_prob)

    def summary(self):
        self._log(f"\n{'='*50}")
        self._log(f"SigimsaeWaveformDataset (class_scheme={self.class_scheme})")
        self._log(f"  sr={self.sr}, duration={self.duration}s, target_samples={self.target_samples}")
        self._log(f"  Total segments: {len(self.X)}")
        self._log(f"  Classes ({self.n_classes}): {self.class_names}")
        for split, indices in self._split_indices.items():
            self._log(f"  {split}: {len(indices)} segments")
        self._log(f"{'='*50}")


class SigimsaeWaveformAugDataset(TorchDataset):
    """PyTorch Dataset for waveforms (with light augmentation)."""

    def __init__(self, X: torch.Tensor, y: torch.Tensor,
                 augment: bool = False, aug_prob: float = 0.3):
        self.X = X  # (N, target_samples)
        self.y = y
        self.augment = augment
        self.aug_prob = aug_prob

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx]
        if self.augment:
            x = self._augment(x)
        return x, y

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Waveform augmentation: gain perturbation + additive noise."""
        # Gain perturbation (±3dB)
        if torch.rand(1).item() < self.aug_prob:
            gain_db = (torch.rand(1).item() - 0.5) * 6.0  # -3 to +3 dB
            x = x * (10 ** (gain_db / 20.0))

        # Additive Gaussian noise
        if torch.rand(1).item() < self.aug_prob:
            noise_std = torch.rand(1).item() * 0.005
            x = x + torch.randn_like(x) * noise_std

        return x


class AddBeatDataset(SigimsaeF0Dataset):
    """SigimsaeF0Dataset + beat/downbeat as a 4th channel.

    Channels: [norm_f0_cent, voicing, delta_f0, beat_info]
    beat_info: 0=none, 1=beat, 2=downbeat

    Parameters
    ----------
    beat_dir : beat_output JSON directory
    **kwargs : arguments forwarded to SigimsaeF0Dataset
    """

    def __init__(self, beat_dir="beat_output", **kwargs):
        self.beat_dir = Path(beat_dir)
        self._beat_cache = {}
        super().__init__(**kwargs)
        self._add_beat_channel()

    def _load_beats(self, stem):
        if stem not in self._beat_cache:
            path = self.beat_dir / f"{stem}.json"
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    self._beat_cache[stem] = json.load(f)
            else:
                self._beat_cache[stem] = None
        return self._beat_cache[stem]

    def _add_beat_channel(self):
        if self.X is None or len(self.X) == 0:
            return

        n_samples = len(self.X)
        beat_array = np.zeros((n_samples, self.max_frames, 1), dtype=np.float32)
        pad_sec = self.pad_ms / 1000.0
        n_with_beats = 0

        for i, m in enumerate(self.meta):
            beat_data = self._load_beats(m["file"])
            if beat_data is None:
                continue

            start_sec = m["start_sec"]
            end_sec = m["end_sec"]
            adj_start = max(0.0, start_sec - pad_sec)
            adj_end = end_sec + pad_sec

            # estimate the original frame count -> reproduce pad_or_truncate logic
            raw_frames = max(1, int(round((adj_end - adj_start) / HOP_SEC)))

            if raw_frames > self.max_frames:
                # center crop
                crop_offset = (raw_frames - self.max_frames) // 2
                eff_start = adj_start + crop_offset * HOP_SEC
                content_frames = self.max_frames
            else:
                # zero pad
                eff_start = adj_start
                content_frames = raw_frames

            has_beat = False
            for t in beat_data.get("beats", []):
                idx = int(round((t - eff_start) / HOP_SEC))
                if 0 <= idx < content_frames:
                    beat_array[i, idx, 0] = 1.0
                    has_beat = True

            for t in beat_data.get("downbeats", []):
                idx = int(round((t - eff_start) / HOP_SEC))
                if 0 <= idx < content_frames:
                    beat_array[i, idx, 0] = 2.0
                    has_beat = True

            if has_beat:
                n_with_beats += 1

        self.X = np.concatenate([self.X, beat_array], axis=-1)
        self._log(
            f"Beat channel added: Shape {self.X.shape}  "
            f"(segments with beats: {n_with_beats}/{n_samples})"
        )


# ─── MERT embedding dataset ─────────────────────────────

class SigimsaeEmbeddingDataset:
    """Dataset that loads pre-extracted MERT embeddings.

    Loads .npz files extracted by extract_mert_embeddings.py.

    Parameters
    ----------
    embedding_path : path to the embedding .npz file
    seed, train_ratio, val_ratio, test_ratio, split_by : split settings
    """

    def __init__(
        self,
        embedding_path: str,
        seed: int = 42,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        split_by: str = "song",
        class_scheme: str = "7cat",
        verbose: bool = True,
    ):
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.split_by = split_by
        self.class_scheme = class_scheme
        self.verbose = verbose

        scheme = CLASS_SCHEMES[class_scheme]
        self.class_names = scheme["class_names"]
        self.n_classes = scheme["n_classes"]

        emb_path = Path(embedding_path)
        meta_path = emb_path.with_name(emb_path.stem + "_meta.json")

        self._log(f"Embedding load: {emb_path}")
        data = np.load(emb_path)
        self.X = data["X"]  # (N, embed_dim)
        self.y = data["y"]  # (N,)

        with open(meta_path, encoding="utf-8") as f:
            meta_json = json.load(f)
        self.meta = meta_json.get("segments", [])

        self._log(f"  Shape: {self.X.shape}  |  Segments: {len(self.X)}")

        self._split_indices: Dict[str, List[int]] = {}
        self._split_files: Dict[str, set] = {}
        self._split()

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _split(self):
        if len(self.meta) == 0:
            for k in ("train", "val", "test"):
                self._split_indices[k] = []
                self._split_files[k] = set()
            return

        np.random.seed(self.seed)

        if self.split_by == "random":
            indices = np.arange(len(self.meta))
            np.random.shuffle(indices)
            n_train = int(len(indices) * self.train_ratio)
            n_val = int(len(indices) * self.val_ratio)
            self._split_indices = {
                "train": indices[:n_train].tolist(),
                "val": indices[n_train:n_train + n_val].tolist(),
                "test": indices[n_train + n_val:].tolist(),
            }
        else:
            file_names = sorted({m["file"] for m in self.meta})
            np.random.shuffle(file_names)
            n_files = len(file_names)
            n_train = int(n_files * self.train_ratio)
            n_val = int(n_files * self.val_ratio)
            train_files = set(file_names[:n_train])
            val_files = set(file_names[n_train:n_train + n_val])
            test_files = set(file_names[n_train + n_val:])
            self._split_files = {
                "train": train_files, "val": val_files, "test": test_files,
            }
            self._split_indices = {
                "train": [i for i, m in enumerate(self.meta) if m["file"] in train_files],
                "val": [i for i, m in enumerate(self.meta) if m["file"] in val_files],
                "test": [i for i, m in enumerate(self.meta) if m["file"] in test_files],
            }

    def get_split(self, split: str):
        indices = self._split_indices[split]
        return self.X[indices], self.y[indices]

    def get_torch_dataset(self, split: str, **kwargs):
        X_np, y_np = self.get_split(split)
        X_t = torch.FloatTensor(X_np)
        y_t = torch.LongTensor(y_np)
        return torch.utils.data.TensorDataset(X_t, y_t)

    def summary(self):
        self._log(f"\n{'='*50}")
        self._log(f"SigimsaeEmbeddingDataset (class_scheme={self.class_scheme})")
        self._log(f"  embed_dim={self.X.shape[1]}")
        self._log(f"  Total segments: {len(self.X)}")
        self._log(f"  Classes ({self.n_classes}): {self.class_names}")
        for split, indices in self._split_indices.items():
            self._log(f"  {split}: {len(indices)} segments")
        self._log(f"{'='*50}")


# ══════════════════════════════════════════════════════════
#  Chromagram dataset (pre-extracted 120-bin CQT chromagram)
# ══════════════════════════════════════════════════════════

class SigimsaeChromaDataset:
    """sigimsae chromagram dataset

    Slices label intervals from pre-extracted 120-bin CQT chromagram .npy files
    to form shape (N, 1, n_chroma, max_chroma_frames).

    Parameters
    ----------
    label_dir : sigimsae label JSON directory
    chroma_dir : chromagram .npy directory (e.g., chroma_output)
    sr : audio sample rate
    n_chroma : number of chromagram bins
    n_fft : FFT window size
    hop_length : hop length (samples)
    max_chroma_frames : fixed segment frame count
    pad_ms : onset/offset front/back padding (milliseconds)
    """

    def __init__(
        self,
        label_dir: str = "precheck_sigimsae_labels",
        chroma_dir: str = "features/chroma_output",
        sr: int = 22050,
        n_chroma: int = 120,
        n_fft: int = 4096,
        hop_length: int = 386,
        max_chroma_frames: int = 171,
        pad_ms: int = 0,
        seed: int = 42,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        verbose: bool = True,
        split_by: str = "song",
        class_scheme: Optional[str] = None,
    ):
        self.label_dir = Path(label_dir)
        self.chroma_dir = Path(chroma_dir)
        self.sr = sr
        self.n_chroma = n_chroma
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_chroma_frames = max_chroma_frames
        self.pad_ms = pad_ms
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.verbose = verbose
        self.split_by = split_by

        if class_scheme is not None:
            assert class_scheme in CLASS_SCHEMES, \
                f"class_scheme must be one of {list(CLASS_SCHEMES.keys())}, got '{class_scheme}'"
            self.class_scheme = class_scheme
        else:
            self.class_scheme = "9cat"

        scheme = CLASS_SCHEMES[self.class_scheme]
        self.class_names = scheme["class_names"]
        self._resolve_fn = scheme["resolve_fn"]
        self.n_classes = scheme["n_classes"]

        self.chroma_hop_sec = hop_length / sr

        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.meta: List[Dict] = []
        self.class_counter = Counter()
        self.skip_counter = Counter()

        self._split_indices: Dict[str, List[int]] = {}
        self._split_files: Dict[str, set] = {}

        self._extract_segments()
        self._split()

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _load_chroma(self, file_id: str) -> Optional[np.ndarray]:
        """Load a pre-extracted chromagram .npy file."""
        config_name = f"{self.sr}_{self.n_chroma}_{self.n_fft}_{self.hop_length}"
        path = self.chroma_dir / config_name / f"{file_id}.npy"
        if path.exists():
            return np.load(path)
        return None

    def _slice_chroma(self, chroma: np.ndarray, start_sec: float, end_sec: float) -> Optional[np.ndarray]:
        """Slice a time interval from the chromagram and pad/crop to fixed length.

        Returns: (n_chroma, max_chroma_frames) array
        """
        pad_sec = self.pad_ms / 1000.0
        adj_start = max(0.0, start_sec - pad_sec)
        adj_end = end_sec + pad_sec

        start_frame = int(round(adj_start / self.chroma_hop_sec))
        end_frame = int(round(adj_end / self.chroma_hop_sec))

        start_frame = max(0, start_frame)
        end_frame = min(chroma.shape[1], end_frame)

        if end_frame <= start_frame:
            return None

        segment = chroma[:, start_frame:end_frame]  # (n_chroma, n_frames)
        n_frames = segment.shape[1]

        if n_frames == self.max_chroma_frames:
            return segment
        elif n_frames > self.max_chroma_frames:
            start = (n_frames - self.max_chroma_frames) // 2
            return segment[:, start:start + self.max_chroma_frames]
        else:
            padded = np.zeros((self.n_chroma, self.max_chroma_frames), dtype=np.float32)
            padded[:, :n_frames] = segment
            return padded

    def _extract_segments(self):
        np.random.seed(self.seed)

        label_files = sorted(self.label_dir.glob("*.json"))
        self._log(f"Label files: {len(label_files)}")
        self._log(f"Chroma config: sr={self.sr}, n_chroma={self.n_chroma}, "
                   f"n_fft={self.n_fft}, hop={self.hop_length}, "
                   f"max_frames={self.max_chroma_frames}")

        all_features: List[np.ndarray] = []
        all_labels: List[int] = []

        chroma_cache: Dict[str, Optional[np.ndarray]] = {}

        for i, label_file in enumerate(label_files):
            file_id = label_file.stem.split("_")[0]

            if file_id not in chroma_cache:
                chroma_cache[file_id] = self._load_chroma(file_id)

            chroma = chroma_cache[file_id]
            if chroma is None:
                self.skip_counter["chroma_missing"] += 1
                continue

            with open(label_file, encoding="utf-8") as f:
                data = json.load(f)
            regions = data.get("annotations", {}).get("sigimsaeRegions", [])

            for region in regions:
                labels = region.get("sigimsae", [])
                if not labels:
                    self.skip_counter["no_label"] += 1
                    continue

                cls = self._resolve_fn(labels)
                if cls is None:
                    self.skip_counter["unknown_label"] += 1
                    continue

                segment = self._slice_chroma(chroma, region["start_sec"], region["end_sec"])
                if segment is None:
                    self.skip_counter["empty_segment"] += 1
                    continue

                all_features.append(segment)
                all_labels.append(cls)
                self.meta.append({
                    "file": label_file.stem,
                    "sigimsae_id": region.get("sigimsae_id", ""),
                    "start_sec": region["start_sec"],
                    "end_sec": region["end_sec"],
                    "duration": round(region["end_sec"] - region["start_sec"], 4),
                    "original_labels": labels,
                    "class": cls,
                    "class_name": self.class_names[cls],
                })
                self.class_counter[cls] += 1

            if self.verbose and (i + 1) % 50 == 0:
                self._log(
                    f"  [{i+1}/{len(label_files)}] "
                    f"extracting {len(all_features)} segments..."
                )

        if all_features:
            stacked = np.array(all_features, dtype=np.float32)
            self.X = stacked[:, np.newaxis, :, :]  # (N, 1, n_chroma, max_frames)
            self.y = np.array(all_labels, dtype=np.int64)
        else:
            self.X = np.empty((0, 1, self.n_chroma, self.max_chroma_frames), dtype=np.float32)
            self.y = np.empty((0,), dtype=np.int64)

        self._log(f"\nTotal valid segments: {len(self.X)}  |  Shape: {self.X.shape}")
        self._log(f"Skipped: {dict(self.skip_counter)}")

    def _split(self):
        if len(self.meta) == 0:
            for k in ("train", "val", "test"):
                self._split_indices[k] = []
                self._split_files[k] = set()
            return

        np.random.seed(self.seed)

        if self.split_by == "random":
            indices = np.arange(len(self.meta))
            np.random.shuffle(indices)
            n_total = len(indices)
            n_train = int(n_total * self.train_ratio)
            n_val = int(n_total * self.val_ratio)

            train_idx = indices[:n_train].tolist()
            val_idx = indices[n_train:n_train + n_val].tolist()
            test_idx = indices[n_train + n_val:].tolist()

            self._split_indices = {"train": train_idx, "val": val_idx, "test": test_idx}
            self._split_files = {
                k: {self.meta[i]["file"] for i in idx}
                for k, idx in self._split_indices.items()
            }
        else:
            file_names = sorted({m["file"] for m in self.meta})
            np.random.shuffle(file_names)

            n_files = len(file_names)
            n_train = int(n_files * self.train_ratio)
            n_val = int(n_files * self.val_ratio)

            train_files = set(file_names[:n_train])
            val_files = set(file_names[n_train:n_train + n_val])
            test_files = set(file_names[n_train + n_val:])

            self._split_files = {"train": train_files, "val": val_files, "test": test_files}
            self._split_indices = {
                "train": [i for i, m in enumerate(self.meta) if m["file"] in train_files],
                "val": [i for i, m in enumerate(self.meta) if m["file"] in val_files],
                "test": [i for i, m in enumerate(self.meta) if m["file"] in test_files],
            }

    def get_split(self, split: str) -> Tuple[np.ndarray, np.ndarray]:
        assert split in ("train", "val", "test")
        idx = self._split_indices[split]
        return self.X[idx], self.y[idx]

    @property
    def n_segments(self) -> int:
        return len(self.X)

    def summary(self):
        print(f"\n{'='*55}")
        print(f"  SigimsaeChromaDataset")
        print(f"{'='*55}")
        print(f"  Total segments: {self.n_segments}  |  Shape: {self.X.shape}")
        print(f"  sr={self.sr}  n_chroma={self.n_chroma}  n_fft={self.n_fft}  "
              f"hop={self.hop_length}  max_frames={self.max_chroma_frames}")
        print(f"  chroma_hop_sec={self.chroma_hop_sec:.5f}s  pad_ms={self.pad_ms}  seed={self.seed}")
        print(f"  Skipped: {dict(self.skip_counter)}")

        print(f"\n  Class distribution:")
        for cls_idx in range(self.n_classes):
            cnt = self.class_counter.get(cls_idx, 0)
            pct = cnt / self.n_segments * 100 if self.n_segments > 0 else 0
            print(f"    [{cls_idx}] {self.class_names[cls_idx]:10s}: {cnt:6d} ({pct:5.1f}%)")

        print(f"\n  Split (ratio={self.train_ratio}/{self.val_ratio}/{self.test_ratio}):")
        for split in ("train", "val", "test"):
            idx = self._split_indices[split]
            files = self._split_files[split]
            print(f"    {split.capitalize():5s}: {len(idx):6d} segments ({len(files)} files)")

        for split in ("train", "val", "test"):
            idx = self._split_indices[split]
            if not idx:
                continue
            split_labels = self.y[idx]
            print(f"\n    {split.capitalize()} class distribution:")
            for cls_idx in range(self.n_classes):
                cnt = int((split_labels == cls_idx).sum())
                print(f"      [{cls_idx}] {self.class_names[cls_idx]:10s}: {cnt}")
        print(f"{'='*55}")

    def get_torch_dataset(self, split: str, augment: bool = False,
                          freq_mask_param: int = 8,
                          time_mask_param: int = 16,
                          aug_prob: float = 0.3):
        X_np, y_np = self.get_split(split)
        X_t = torch.FloatTensor(X_np)
        y_t = torch.LongTensor(y_np)
        return SigimsaeMelAugDataset(
            X_t, y_t, augment=augment,
            freq_mask_param=freq_mask_param,
            time_mask_param=time_mask_param,
            aug_prob=aug_prob,
        )
