import numpy as np
from tools.demo.track_json import downsample_f0, build_track_payload


def test_downsample_f0_halves_rate_and_nulls_unvoiced():
    f0 = np.array([0.0, 0.0, 220.0, 221.0, 222.0, 0.0], dtype=np.float32)  # 10 ms hop
    out = downsample_f0(f0, factor=2)
    assert out == [None, 220.0, 222.0]          # every 2nd frame, 0 -> None


def test_build_track_payload_shapes():
    annotation = {
        "annotation": {
            "lyrics_regions": [{"lyrics_id": "L001", "start_sec": 1.0, "end_sec": 2.0, "lyrics": "가"}],
            "sigimsage_regions": [{"sigimsage_id": "S001", "start_sec": 1.2, "end_sec": 1.5,
                                   "sigimsage_types": ["추성"]}],
            "caption_ko": "ko", "caption_en": "en",
        }
    }
    row = {"곡명": "t", "장르": "정악", "하위장르": "풍류음악", "세부장르(optional)": "가곡",
           "가창자": "s", "가창자 성별": "남창", "장단": "j", "최종 서양조성": "k",
           "최종 템포": "100", "오디오_길이(초)": "3.0"}
    p = build_track_payload("X_P000001", row, annotation, np.array([220.0] * 300, dtype=np.float32))
    assert p["id"] == "X_P000001" and p["f0"]["hop_sec"] == 0.02
    assert len(p["f0"]["hz"]) == 150
    assert p["sigimsae_regions"][0]["types"] == ["추성"]
    assert p["lyrics_regions"][0]["text"] == "가"
