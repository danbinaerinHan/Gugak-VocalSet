"""Render 20 s CQT tiles per track for the viewer's background toggle."""
import math
import numpy as np
import librosa
from PIL import Image

TILE_SEC = 20
FMIN = 65.41          # C2
BPO = 36
N_BINS = 216          # 6 octaves
HOP = 512             # ~23 ms/col at 22050 Hz -> ~860 px per tile
PX_HEIGHT = 216       # 1 px per bin


def render_tiles(y: np.ndarray, sr: int, out_dir, track_id: str) -> dict:
    C = np.abs(librosa.cqt(y, sr=sr, fmin=FMIN, n_bins=N_BINS, bins_per_octave=BPO, hop_length=HOP))
    db = librosa.amplitude_to_db(C, ref=np.max)                     # (N_BINS, T), <= 0
    img = np.clip((db + 80.0) / 80.0, 0, 1)                          # -80..0 dB -> 0..1
    img = (img * 255).astype(np.uint8)[::-1, :]                      # low freq at bottom
    cols_per_tile = int(round(TILE_SEC * sr / HOP))
    n_tiles = math.ceil(img.shape[1] / cols_per_tile)
    for i in range(n_tiles):
        tile = img[:, i * cols_per_tile:(i + 1) * cols_per_tile]
        Image.fromarray(tile, mode="L").save(out_dir / f"{track_id}_{i:03d}.png", optimize=True)
    return {"tile_sec": TILE_SEC, "n_tiles": n_tiles, "fmin": FMIN, "bpo": BPO,
            "n_bins": N_BINS, "px_per_sec": sr / HOP, "height_px": PX_HEIGHT}
