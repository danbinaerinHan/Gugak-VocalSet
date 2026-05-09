"""sigimsae Event Detection dataset

Paper approach: split the full audio into fixed-length chunks and
assign per-frame labels to each frame to perform event detection.

Key points:
  - Don't care (†) labeling: sigimsae cut by chunk boundaries are marked as don't care
  - The next chunk starts from the onset of the cut sigimsae (overlap)
  - The number of classes is determined by class_scheme (7cat, 9cat, 17cat, etc.)
  - The input representation is determined by feature_type (chroma, mel)

Usage:
    from event_dataset import SigimsaeEventDataset
    ds = SigimsaeEventDataset(class_scheme="7cat", feature_type="chroma")
    ds = SigimsaeEventDataset(class_scheme="7cat", feature_type="mel")
    ds.summary()
    train_loader = ds.get_dataloader("train", batch_size=32)
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Optional, Dict, List, Tuple

import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader

from prepare_f0_dataset import (
    CLASS_SCHEMES,
    CLASS_NAMES2,
    resolve_class2,
)


# ─── Configuration ────────────────────────────────────────
DONT_CARE = -1
NO_ORNAMENT = 0

# Default settings per feature type
FEATURE_DEFAULTS = {
    "chroma": {
        "dir": "features/chroma_output/22050_120_4096_386",
        "hop_sec": 386 / 22050,   # ≈ 0.01752s
    },
    "mel": {
        "dir": "features/mel_output/22050_128_2048_512",
        "hop_sec": 512 / 22050,   # ≈ 0.02322s
    },
    "waveform": {
        "dir": "KVocSet",          # mp3 audio file directory
        "hop_sec": 320 / 24000,    # MERT CNN hop ≈ 0.01333s
        "sr": 24000,               # sample rate required by MERT
    },
    "mert_hidden": {
        "dir": "mert_embedding_cache/frame_features",  # pre-extracted MERT hidden states
        "hop_sec": 320 / 24000,    # MERT CNN hop ≈ 0.01333s
    },
    "f0": {
        "dir": "features/pitch_output/rmvpe",  # RMVPE-extracted F0 (song-level)
        "hop_sec": 0.01,           # RMVPE native hop = 10ms
    },
    "mel_f0": {
        "dir": "features/mel_output/22050_128_2048_512",  # mel-based hop
        "hop_sec": 512 / 22050,    # ≈ 0.02322s (mel hop; f0 is resampled)
        "f0_dir": "features/pitch_output/rmvpe",
    },
}

# Legacy compatibility: default 9cat class scheme (when class_scheme is not specified)
EVENT_CLASS_NAMES = ["no_ornament"] + list(CLASS_NAMES2)  # 10 classes total
NUM_EVENT_CLASSES = len(EVENT_CLASS_NAMES)


def _get_event_class_info(class_scheme: Optional[str] = None):
    """Return event detection class information based on class_scheme.

    Returns:
        (class_names, n_classes, resolve_fn)
        class_names: ["no_ornament"] + list of sigimsae class names
        n_classes: total number of classes including no_ornament
        resolve_fn: function mapping label list → class index
    """
    if class_scheme is None:
        # Legacy compatibility: 9cat (CLASS_NAMES2)
        return EVENT_CLASS_NAMES, NUM_EVENT_CLASSES, resolve_class2

    scheme = CLASS_SCHEMES[class_scheme]
    class_names = ["no_ornament"] + list(scheme["class_names"])
    n_classes = scheme["n_classes"] + 1  # +1 for no_ornament
    resolve_fn = scheme["resolve_fn"]
    return class_names, n_classes, resolve_fn


def _load_feature(feature_dir: Path, file_id: str) -> Optional[np.ndarray]:
    """Load a pre-extracted feature .npy (chromagram or mel spectrogram)."""
    path = feature_dir / f"{file_id}.npy"
    if path.exists():
        return np.load(path)
    return None


def _load_f0_song(f0_dir: Path, file_id: str) -> Optional[np.ndarray]:
    """Load RMVPE F0 for one song and convert it to a (3, T) normalized feature.

    Channels:
      [0] norm_f0:  cents relative to song median (±2400 → [-1, 1]), unvoiced=-2.0
      [1] voicing:  {0, 1}
      [2] delta_f0: norm_f0[t] - norm_f0[t-1] (0 at unvoiced boundaries)

    RMVPE file format: {file_id}_{genre}_*_f0.npy  (shape (T,), hop=10ms)
    """
    f0_matches = list(f0_dir.glob(f"{file_id}_*_f0.npy"))
    if not f0_matches:
        return None
    f0 = np.load(f0_matches[0]).astype(np.float32)

    voiced_mask = f0 > 0
    if voiced_mask.sum() < 2:
        return None

    song_ref_f0 = float(np.median(f0[voiced_mask]))
    if song_ref_f0 <= 0:
        return None

    T = len(f0)
    voicing = voiced_mask.astype(np.float32)

    norm_f0 = np.zeros(T, dtype=np.float32)
    cents = 1200.0 * np.log2(f0[voiced_mask] / song_ref_f0)
    norm_f0[voiced_mask] = np.clip(cents, -2400.0, 2400.0) / 2400.0
    norm_f0[~voiced_mask] = -2.0  # distinct from the voiced range ([-1,1])

    delta_f0 = np.zeros(T, dtype=np.float32)
    delta_f0[1:] = norm_f0[1:] - norm_f0[:-1]
    delta_f0[~voiced_mask] = 0.0
    unvoiced_shift = np.roll(voiced_mask, 1)
    unvoiced_shift[0] = True
    delta_f0[~unvoiced_shift] = 0.0
    # At 10ms hop, the raw std of delta (~0.016) is ~11× smaller than the voiced std of norm_f0 (~0.2),
    # so the first Conv may largely ignore the delta channel → scale to match magnitudes across channels.
    # Clip is used to remove outliers (RMVPE artifacts near boundaries, changes > 3.6 semitones in 10ms).
    delta_f0 = np.clip(delta_f0, -0.15, 0.15) * 10.0

    return np.stack([norm_f0, voicing, delta_f0], axis=0).astype(np.float32)  # (3, T)


def _load_mel_f0_song(mel_dir: Path, f0_dir: Path, file_id: str,
                      mel_hop_sec: float = 512/22050,
                      f0_hop_sec: float = 0.01) -> Optional[np.ndarray]:
    """Mel spectrogram + F0 (3-channel) fusion feature.

    Returns (128+3=131, T_mel) with f0 resampled to mel hop via nearest-frame.
    """
    mel = _load_feature(mel_dir, file_id)  # (128, T_mel) or None
    if mel is None:
        return None
    f0_feat = _load_f0_song(f0_dir, file_id)  # (3, T_f0) or None
    if f0_feat is None:
        return None

    T_mel = mel.shape[1]
    T_f0  = f0_feat.shape[1]

    # Resample f0 to match the mel hop (nearest-frame, preserving temporal alignment).
    # Time of mel frame i = i * mel_hop_sec, corresponding f0 index = round(i * mel_hop_sec / f0_hop_sec)
    ratio = mel_hop_sec / f0_hop_sec   # ≈ 2.32
    f0_indices = np.clip(np.round(np.arange(T_mel) * ratio).astype(np.int64), 0, T_f0 - 1)
    f0_resampled = f0_feat[:, f0_indices]  # (3, T_mel)

    fused = np.concatenate([mel.astype(np.float32), f0_resampled], axis=0)  # (131, T_mel)
    return fused


def _load_waveform(audio_dir: Path, file_id: str, sr: int = 24000) -> Optional[np.ndarray]:
    """Load an audio file and resample it to the target sr.

    Returns:
        (samples,) float32 numpy array, or None
    """
    import librosa
    # Find audio files starting with file_id
    for ext in ["*.mp3", "*.wav", "*.flac"]:
        matches = list(audio_dir.glob(f"{file_id}_{ext[1:]}")) + list(audio_dir.glob(f"{file_id}.{ext[2:]}"))
        if not matches:
            matches = list(audio_dir.glob(f"{file_id}_*{ext[1:]}"))
        if matches:
            y, _ = librosa.load(matches[0], sr=sr, mono=True)
            return y.astype(np.float32)
    return None


def _build_frame_labels(
    regions: List[Dict],
    total_frames: int,
    hop_sec: float,
    chunk_start: float,
    chunk_end: float,
    resolve_fn=None,
    use_dont_care: bool = True,
) -> np.ndarray:
    """Generate labels for every frame within a chunk.

    Args:
        use_dont_care: If True, mark sigimsae cut at the boundary as DONT_CARE(-1).
                       If False, assign the actual class label even to cut sigimsae.

    Returns:
        (total_frames,) int array — 0=no_ornament, 1~N=sigimsae, -1=don't care (when use_dont_care)
    """
    if resolve_fn is None:
        resolve_fn = resolve_class2

    labels = np.zeros(total_frames, dtype=np.int32)  # default: no_ornament

    for region in regions:
        sigimsae_list = region.get("sigimsae", [])
        if not sigimsae_list:
            continue

        cls = resolve_fn(sigimsae_list)
        if cls is None:
            continue

        onset = region["start_sec"]
        offset = region["end_sec"]

        # Skip if it does not overlap the chunk
        if offset <= chunk_start or onset >= chunk_end:
            continue

        # Frame indices (relative position within the chunk)
        rel_onset = onset - chunk_start
        rel_offset = offset - chunk_start

        start_frame = max(0, int(round(rel_onset / hop_sec)))
        end_frame = min(total_frames, int(round(rel_offset / hop_sec)))

        if end_frame <= start_frame:
            continue

        # Fully contained inside the chunk → actual label
        if onset >= chunk_start and offset <= chunk_end:
            labels[start_frame:end_frame] = cls + 1  # 1-indexed (0=no_ornament)
        else:
            # Cut at the boundary
            if use_dont_care:
                labels[start_frame:end_frame] = DONT_CARE
            else:
                # don't care disabled: assign the actual class label even when cut
                labels[start_frame:end_frame] = cls + 1

    return labels


def _chunk_audio_with_dont_care(
    regions: List[Dict],
    total_duration: float,
    chunk_sec: float,
    hop_sec: float,
    frames_per_chunk: int,
    resolve_fn=None,
    use_dont_care: bool = True,
) -> List[Dict]:
    """Don't care chunking: algorithm from paper Section V-A.

    Args:
        use_dont_care: If True, mark boundary-cut sigimsae as DONT_CARE.
                       If False, assign the actual class label even to cut sigimsae.

    Returns:
        list of {start_sec, end_sec, labels}
    """
    chunks = []
    t = 0.0

    while t < total_duration:
        chunk_start = t
        chunk_end = min(t + chunk_sec, total_duration)
        actual_frames = min(frames_per_chunk, int(round((chunk_end - chunk_start) / hop_sec)))

        if actual_frames < 10:  # skip last chunk if too short
            break

        labels = _build_frame_labels(
            regions, actual_frames, hop_sec, chunk_start, chunk_end,
            resolve_fn=resolve_fn,
            use_dont_care=use_dont_care,
        )

        chunks.append({
            "start_sec": chunk_start,
            "end_sec": chunk_end,
            "labels": labels,
        })

        # Determine the next chunk start: if there is a cut sigimsae, start from its onset
        next_start = chunk_end
        for region in regions:
            onset = region["start_sec"]
            offset = region["end_sec"]
            # Sigimsae cut at the end of the chunk
            if onset < chunk_end and offset > chunk_end and onset >= chunk_start:
                next_start = min(next_start, onset)

        # Force advance if no progress is made
        if next_start <= t:
            next_start = chunk_end

        t = next_start

    return chunks


class SigimsaeEventDataset:
    """sigimsae Event Detection dataset

    Full audio → 10-second chunks → per-frame feature + label.

    Parameters
    ----------
    label_dir : sigimsae label JSON directory
    feature_type : input representation ("chroma" or "mel")
    feature_dir : pre-extracted feature directory (uses feature_type default if None)
    class_scheme : class scheme ("7cat", "9cat", "17cat", "total", or None=legacy 9cat)
    chunk_sec : chunk length (seconds)
    hop_sec : feature hop (seconds; uses feature_type default if None)
    seed, train_ratio, val_ratio, test_ratio : split settings
    """

    def __init__(
        self,
        label_dir: str = "precheck_sigimsae_labels",
        feature_type: str = "chroma",
        feature_dir: str = None,
        class_scheme: str = None,
        chunk_sec: float = 10.0,
        hop_sec: float = None,
        seed: int = 42,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        verbose: bool = True,
        use_dont_care: bool = True,
        jitter_sec: float = 0.0,
        # Legacy compatibility: chroma_dir parameter
        chroma_dir: str = None,
    ):
        self.label_dir = Path(label_dir)
        self.feature_type = feature_type
        self.chunk_sec = chunk_sec
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.verbose = verbose
        self.use_dont_care = use_dont_care
        self.jitter_sec = jitter_sec

        # Determine feature directory (including legacy compatibility)
        if chroma_dir is not None:
            # Legacy: chroma_dir specified directly
            self.feature_dir = Path(chroma_dir)
            self.hop_sec = hop_sec if hop_sec is not None else FEATURE_DEFAULTS["chroma"]["hop_sec"]
        elif feature_dir is not None:
            self.feature_dir = Path(feature_dir)
            self.hop_sec = hop_sec if hop_sec is not None else FEATURE_DEFAULTS[feature_type]["hop_sec"]
        else:
            defaults = FEATURE_DEFAULTS[feature_type]
            self.feature_dir = Path(defaults["dir"])
            self.hop_sec = hop_sec if hop_sec is not None else defaults["hop_sec"]

        # Determine class scheme
        self.class_scheme = class_scheme
        self.class_names, self.n_classes, self._resolve_fn = _get_event_class_info(class_scheme)

        self.n_freq = None  # number of feature frequency bins; set on first load
        self.frames_per_chunk = int(round(chunk_sec / self.hop_sec))

        # Legacy compatibility attribute
        self.n_chroma = None

        # Chunk data
        self.chunks: List[Dict] = []  # {file, file_id, features, labels}
        self.file_to_chunks: Dict[str, List[int]] = {}  # file_stem → chunk indices
        self._file_data: Dict[str, Dict] = {}  # for jitter: file_stem → {features, regions, total_duration}

        self._split_indices: Dict[str, List[int]] = {}
        self._split_files: Dict[str, set] = {}

        self.class_counter = Counter()
        self.skip_counter = Counter()

        self._build_chunks()
        self._split()

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _build_chunks(self):
        label_files = sorted(self.label_dir.glob("*.json"))
        scheme_name = self.class_scheme or "9cat(legacy)"
        self._log(f"Label files: {len(label_files)}")
        self._log(f"Feature: {self.feature_type} ({self.feature_dir})")
        self._log(f"Class scheme: {scheme_name} ({self.n_classes} classes)")
        self._log(f"Chunk length: {self.chunk_sec}s, hop: {self.hop_sec:.5f}s, "
                   f"frames/chunk: {self.frames_per_chunk}")

        is_waveform = (self.feature_type == "waveform")
        if is_waveform:
            waveform_sr = FEATURE_DEFAULTS["waveform"]["sr"]
            samples_per_chunk = int(self.chunk_sec * waveform_sr)
            self.n_freq = None  # waveform has no freq axis
            self.n_chroma = None

        for i, label_file in enumerate(label_files):
            file_stem = label_file.stem
            file_id = file_stem.split("_")[0]

            if is_waveform:
                # Load waveform
                wav = _load_waveform(self.feature_dir, file_id, sr=waveform_sr)
                if wav is None:
                    self.skip_counter["feature_missing"] += 1
                    continue
                total_duration = len(wav) / waveform_sr
            elif self.feature_type == "f0":
                # Load RMVPE f0 + per-song normalization → (3, T)
                feat = _load_f0_song(self.feature_dir, file_id)
                if feat is None:
                    self.skip_counter["feature_missing"] += 1
                    continue
                if self.n_freq is None:
                    self.n_freq = feat.shape[0]  # = 3
                    self.n_chroma = self.n_freq
                    self._log(f"Feature bins: {self.n_freq} (f0/voicing/delta)")
                total_duration = feat.shape[1] * self.hop_sec
            elif self.feature_type == "mel_f0":
                # Mel (128) + F0 (3) channel concat → (131, T_mel)
                f0_dir = Path(FEATURE_DEFAULTS["mel_f0"]["f0_dir"])
                feat = _load_mel_f0_song(self.feature_dir, f0_dir, file_id,
                                         mel_hop_sec=self.hop_sec,
                                         f0_hop_sec=FEATURE_DEFAULTS["f0"]["hop_sec"])
                if feat is None:
                    self.skip_counter["feature_missing"] += 1
                    continue
                if self.n_freq is None:
                    self.n_freq = feat.shape[0]  # = 131
                    self.n_chroma = self.n_freq
                    self._log(f"Feature bins: {self.n_freq} (128 mel + 3 f0)")
                total_duration = feat.shape[1] * self.hop_sec
            else:
                # Load feature (chroma, mel, or mert_hidden)
                feat = _load_feature(self.feature_dir, file_id)
                if feat is None:
                    self.skip_counter["feature_missing"] += 1
                    continue
                # mert_hidden is stored as (T, D) → transpose to (D, T)
                # D depends on the model (95M=768, 330M=1024, CultureMERT-95M=768)
                if self.feature_type == "mert_hidden" and feat.ndim == 2:
                    feat = feat.T.astype(np.float32)
                if self.n_freq is None:
                    self.n_freq = feat.shape[0]
                    self.n_chroma = self.n_freq  # legacy compatibility
                    self._log(f"Feature bins: {self.n_freq}")
                total_duration = feat.shape[1] * self.hop_sec

            # Load labels
            with open(label_file, encoding="utf-8") as f:
                data = json.load(f)
            regions = data.get("annotations", {}).get("sigimsaeRegions", [])

            # Keep the full file data for jitter
            if self.jitter_sec > 0:
                self._file_data[file_stem] = {
                    "features": wav if is_waveform else feat,
                    "regions": regions,
                    "total_duration": total_duration,
                }

            # Chunking (pass resolve_fn)
            total_frames = int(round(total_duration / self.hop_sec))
            chunk_infos = _chunk_audio_with_dont_care(
                regions, total_duration, self.chunk_sec, self.hop_sec,
                self.frames_per_chunk, resolve_fn=self._resolve_fn,
                use_dont_care=self.use_dont_care,
            )

            chunk_indices_for_file = []
            for ci in chunk_infos:
                # Label processing (common)
                labels = ci["labels"]
                if len(labels) < self.frames_per_chunk:
                    pad_val = DONT_CARE if self.use_dont_care else NO_ORNAMENT
                    padded_labels = np.full(self.frames_per_chunk, pad_val, dtype=np.int32)
                    padded_labels[:len(labels)] = labels
                    labels = padded_labels

                if self.use_dont_care:
                    actual_label_frames = int((labels != DONT_CARE).sum())
                    if actual_label_frames < 10:
                        continue

                if is_waveform:
                    # waveform slice (sample units)
                    start_sample = int(round(ci["start_sec"] * waveform_sr))
                    end_sample = min(start_sample + samples_per_chunk, len(wav))
                    wav_chunk = wav[start_sample:end_sample]

                    # Fixed-length padding
                    if len(wav_chunk) < samples_per_chunk:
                        padded = np.zeros(samples_per_chunk, dtype=np.float32)
                        padded[:len(wav_chunk)] = wav_chunk
                        wav_chunk = padded

                    feat_chunk = wav_chunk  # (samples_per_chunk,) 1D
                else:
                    # feature slice (frame units)
                    start_frame = int(round(ci["start_sec"] / self.hop_sec))
                    end_frame = min(start_frame + self.frames_per_chunk,
                                    feat.shape[1] if not is_waveform else total_frames)
                    feat_chunk = feat[:, start_frame:end_frame]

                    # Fixed-length padding
                    if feat_chunk.shape[1] < self.frames_per_chunk:
                        padded = np.zeros(
                            (self.n_freq, self.frames_per_chunk), dtype=np.float32
                        )
                        padded[:, :feat_chunk.shape[1]] = feat_chunk
                        feat_chunk = padded

                # Per-frame class count
                for lbl in labels:
                    if lbl != DONT_CARE:
                        self.class_counter[lbl] += 1

                chunk_idx = len(self.chunks)
                self.chunks.append({
                    "file": file_stem,
                    "file_id": file_id,
                    "features": feat_chunk,
                    "chroma": feat_chunk,      # legacy compatibility
                    "labels": labels,
                    "start_sec": ci["start_sec"],
                    "end_sec": ci["end_sec"],
                })
                chunk_indices_for_file.append(chunk_idx)

            if chunk_indices_for_file:
                self.file_to_chunks[file_stem] = chunk_indices_for_file

            if self.verbose and (i + 1) % 50 == 0:
                self._log(f"  [{i+1}/{len(label_files)}] generated {len(self.chunks)} chunks...")

        self._log(f"\nTotal chunks: {len(self.chunks)} (from {len(self.file_to_chunks)} files)")

    def _split(self):
        """Per-file train/val/test split (based on file_id).

        Splits by file_id (numeric prefix); files with the same file_id
        but different names are assigned to the same split.
        """
        if not self.file_to_chunks:
            for k in ("train", "val", "test"):
                self._split_indices[k] = []
                self._split_files[k] = set()
            return

        # Group by file_id
        file_id_to_stems = {}
        for stem in self.file_to_chunks:
            fid = stem.split("_")[0]
            if fid not in file_id_to_stems:
                file_id_to_stems[fid] = []
            file_id_to_stems[fid].append(stem)

        np.random.seed(self.seed)
        file_ids = sorted(file_id_to_stems.keys())
        np.random.shuffle(file_ids)

        n_files = len(file_ids)
        n_train = int(n_files * self.train_ratio)
        n_val = int(n_files * self.val_ratio)

        train_ids = set(file_ids[:n_train])
        val_ids = set(file_ids[n_train:n_train + n_val])
        test_ids = set(file_ids[n_train + n_val:])

        self._split_files = {"train": set(), "val": set(), "test": set()}
        self._split_indices = {"train": [], "val": [], "test": []}

        for split, id_set in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
            for fid in id_set:
                for stem in file_id_to_stems[fid]:
                    self._split_files[split].add(stem)
                    self._split_indices[split].extend(self.file_to_chunks[stem])
            self._split_indices[split] = sorted(self._split_indices[split])

    def summary(self):
        scheme_name = self.class_scheme or "9cat(legacy)"
        print(f"\n{'='*60}")
        print(f"  SigimsaeEventDataset (Frame-level Event Detection)")
        print(f"{'='*60}")
        print(f"  Total chunks: {len(self.chunks)}  |  Files: {len(self.file_to_chunks)}")
        print(f"  Chunk length: {self.chunk_sec}s  |  frames/chunk: {self.frames_per_chunk}")
        if self.feature_type == "waveform":
            print(f"  Feature: waveform (sr={FEATURE_DEFAULTS['waveform']['sr']})")
        else:
            print(f"  Feature: {self.feature_type} ({self.n_freq} bins)")
        print(f"  Class scheme: {scheme_name}")
        print(f"  Number of classes: {self.n_classes} (0=no_ornament + {self.n_classes-1} sigimsae)")
        print(f"  Skipped: {dict(self.skip_counter)}")

        total_frames = sum(self.class_counter.values())
        print(f"\n  Per-frame class distribution (excluding don't care):")
        for cls_idx in range(self.n_classes):
            cnt = self.class_counter.get(cls_idx, 0)
            pct = cnt / total_frames * 100 if total_frames > 0 else 0
            print(f"    [{cls_idx}] {self.class_names[cls_idx]:15s}: {cnt:10d} ({pct:5.1f}%)")

        dc_count = sum(
            int((c["labels"] == DONT_CARE).sum()) for c in self.chunks
        )
        print(f"    [DC] don't care          : {dc_count:10d}")

        print(f"\n  Splits:")
        for split in ("train", "val", "test"):
            idx = self._split_indices[split]
            files = self._split_files[split]
            print(f"    {split.capitalize():5s}: {len(idx):5d} chunks ({len(files)} files)")
        print(f"{'='*60}")

    def get_dataloader(
        self,
        split: str,
        batch_size: int = 32,
        shuffle: bool = None,
        num_workers: int = 0,
    ) -> DataLoader:
        """Return a PyTorch DataLoader."""
        if shuffle is None:
            shuffle = (split == "train")
        ds = self._make_torch_dataset(split)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True)

    def _make_torch_dataset(self, split: str) -> TorchDataset:
        indices = self._split_indices[split]

        # train + jitter enabled → apply start-point jitter on every access
        if split == "train" and self.jitter_sec > 0 and self._file_data:
            chunk_metas = [self.chunks[idx] for idx in indices]
            is_waveform = (self.feature_type == "waveform")
            config = {
                "frames_per_chunk": self.frames_per_chunk,
                "hop_sec": self.hop_sec,
                "chunk_sec": self.chunk_sec,
                "jitter_frames": int(round(self.jitter_sec / self.hop_sec)),
                "resolve_fn": self._resolve_fn,
                "use_dont_care": self.use_dont_care,
                "is_waveform": is_waveform,
            }
            if is_waveform:
                config["waveform_sr"] = FEATURE_DEFAULTS["waveform"]["sr"]
                config["samples_per_chunk"] = int(self.chunk_sec * config["waveform_sr"])
            return JitteredEventDataset(chunk_metas, self._file_data, config)

        # val/test or jitter disabled → existing fixed chunks
        features = []
        labels_list = []
        for idx in indices:
            c = self.chunks[idx]
            features.append(c["features"])
            labels_list.append(c["labels"])

        X = np.stack(features, axis=0)
        Y = np.stack(labels_list, axis=0)
        return EventChunkDataset(
            torch.FloatTensor(X),
            torch.LongTensor(Y),
        )

    def get_class_weights(self, split: str = "train") -> torch.Tensor:
        """Compute per-frame class weights for the training split."""
        indices = self._split_indices[split]
        counts = np.zeros(self.n_classes, dtype=np.float64)
        for idx in indices:
            labels = self.chunks[idx]["labels"]
            for cls_idx in range(self.n_classes):
                counts[cls_idx] += (labels == cls_idx).sum()

        total = counts.sum()
        # inverse frequency with sqrt smoothing
        raw_weights = total / (self.n_classes * counts.clip(min=1))
        weights = np.sqrt(raw_weights)
        return torch.FloatTensor(weights)


class JitteredEventDataset(TorchDataset):
    """Dataset that applies random jitter to the chunk start point during training.

    On every __getitem__ call, the start point is shifted by ±jitter_frames
    to mitigate the boundary bias of fixed chunking.
    The full-file feature is referenced and re-sliced at the jittered position with labels regenerated.
    """

    def __init__(self, chunk_metas, file_data, config):
        self.chunk_metas = chunk_metas
        self.file_data = file_data
        self.cfg = config

    def __len__(self):
        return len(self.chunk_metas)

    def __getitem__(self, idx):
        meta = self.chunk_metas[idx]
        fd = self.file_data[meta["file"]]
        cfg = self.cfg

        # Random jitter (frame units)
        jitter = np.random.randint(-cfg["jitter_frames"], cfg["jitter_frames"] + 1)

        orig_start = meta["start_sec"]
        new_start = orig_start + jitter * cfg["hop_sec"]
        # Clamp to valid range
        max_start = max(0.0, fd["total_duration"] - cfg["chunk_sec"])
        new_start = max(0.0, min(new_start, max_start))
        new_end = min(new_start + cfg["chunk_sec"], fd["total_duration"])

        if cfg["is_waveform"]:
            sr = cfg["waveform_sr"]
            spc = cfg["samples_per_chunk"]
            s0 = int(round(new_start * sr))
            wav_chunk = fd["features"][s0:s0 + spc]
            if len(wav_chunk) < spc:
                padded = np.zeros(spc, dtype=np.float32)
                padded[:len(wav_chunk)] = wav_chunk
                wav_chunk = padded
            feat_out = wav_chunk
        else:
            fpc = cfg["frames_per_chunk"]
            s0 = int(round(new_start / cfg["hop_sec"]))
            feat_chunk = fd["features"][:, s0:s0 + fpc]
            if feat_chunk.shape[1] < fpc:
                padded = np.zeros((fd["features"].shape[0], fpc), dtype=np.float32)
                padded[:, :feat_chunk.shape[1]] = feat_chunk
                feat_chunk = padded
            feat_out = feat_chunk

        # Regenerate labels for the jittered interval
        actual_frames = min(cfg["frames_per_chunk"],
                            int(round((new_end - new_start) / cfg["hop_sec"])))
        labels = _build_frame_labels(
            fd["regions"], actual_frames, cfg["hop_sec"],
            new_start, new_end,
            resolve_fn=cfg["resolve_fn"],
            use_dont_care=cfg["use_dont_care"],
        )
        if len(labels) < cfg["frames_per_chunk"]:
            pad_val = DONT_CARE if cfg["use_dont_care"] else NO_ORNAMENT
            padded_labels = np.full(cfg["frames_per_chunk"], pad_val, dtype=np.int32)
            padded_labels[:len(labels)] = labels
            labels = padded_labels

        return torch.FloatTensor(feat_out), torch.LongTensor(labels)


class EventChunkDataset(TorchDataset):
    """Simple Tensor-wrapping Dataset."""

    def __init__(self, X: torch.Tensor, Y: torch.Tensor):
        self.X = X  # (N, n_chroma, T)
        self.Y = Y  # (N, T)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
