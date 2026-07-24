import unicodedata
from pathlib import Path
from tools.demo.tracks import load_track_index

ROOT = Path(__file__).resolve().parents[2]


def test_index_covers_all_305_tracks():
    idx = load_track_index(ROOT)
    assert len(idx) == 305


def test_sample_track_resolves_everything():
    idx = load_track_index(ROOT)
    t = idx["KC_TM_JC_PR_P000001"]
    assert t["s_id"] == "KC_TM_JC_PR_S000001"
    assert t["singer"] == "홍창남"
    assert unicodedata.normalize("NFC", t["audio_path"].name) == "001_정악_풍류음악.mp3" and t["audio_path"].exists()
    assert t["f0_path"].name.endswith("_f0.npy") and t["f0_path"].exists()
    assert t["annotation_path"].name == "KC_TM_JC_PR_P000001.json" and t["annotation_path"].exists()
