"""Build the per-track demo JSON consumed by docs/js/viewer.js."""
import numpy as np

F0_SRC_HOP = 0.01   # RMVPE hop
DS_FACTOR = 2       # -> 50 Hz for the demo JSON


def downsample_f0(f0_hz: np.ndarray, factor: int = DS_FACTOR) -> list:
    ds = f0_hz[::factor]
    return [None if v <= 0 else round(float(v), 1) for v in ds]


def build_track_payload(p_id: str, row: dict, annotation: dict, f0_hz: np.ndarray) -> dict:
    ann = annotation["annotation"]
    return {
        "id": p_id,
        "title": row["곡명"],
        "genre": row["장르"], "subgenre": row["하위장르"], "subsub": row["세부장르(optional)"],
        "singer": row["가창자"], "gender": row["가창자 성별"],
        "jangdan": row["장단"], "key": row["최종 서양조성"],
        "tempo": float(row["최종 템포"] or 0),
        "duration_sec": float(row["오디오_길이(초)"] or 0),
        "caption_ko": ann.get("caption_ko", ""), "caption_en": ann.get("caption_en", ""),
        "f0": {"hop_sec": F0_SRC_HOP * DS_FACTOR, "hz": downsample_f0(f0_hz)},
        "lyrics_regions": [
            {"id": r["lyrics_id"], "start": round(r["start_sec"], 3),
             "end": round(r["end_sec"], 3), "text": r["lyrics"]}
            for r in ann["lyrics_regions"]
        ],
        "sigimsae_regions": [
            {"id": r["sigimsage_id"], "start": round(r["start_sec"], 3),
             "end": round(r["end_sec"], 3), "types": r["sigimsage_types"]}
            for r in ann["sigimsage_regions"]
        ],
    }
