import numpy as np
from tools.demo.cqt_tiles import render_tiles

def test_tiles_from_sine(tmp_path):
    sr = 22050
    y = 0.5 * np.sin(2 * np.pi * 440.0 * np.arange(sr * 45) / sr)  # 45 s of A4
    manifest = render_tiles(y, sr, tmp_path, track_id="TEST")
    assert manifest["tile_sec"] == 20 and manifest["n_tiles"] == 3   # 20+20+5
    assert manifest["fmin"] == 65.41 and manifest["n_bins"] == 216
    assert (tmp_path / "TEST_000.png").exists() and (tmp_path / "TEST_002.png").exists()
