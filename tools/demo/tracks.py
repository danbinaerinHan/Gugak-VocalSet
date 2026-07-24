"""Resolve track IDs to metadata rows, audio, F0, and annotation paths."""
import csv
import unicodedata
from pathlib import Path


def p_id_from_s_id(s_id: str) -> str:
    head, tail = s_id.rsplit("_", 1)          # tail = "S000001"
    return f"{head}_P{tail[1:]}"


def _resolve_existing(root_dir: Path, filename: str) -> Path:
    """Resolve a file path, trying both NFC and NFD forms if needed.

    Returns the path that actually exists on disk, or the NFC path as best-effort fallback.
    This handles filesystem encoding variations (e.g., Linux ext4 with NFD filenames).
    """
    p = root_dir / filename
    if p.exists():
        return p
    p_nfd = root_dir / unicodedata.normalize("NFD", filename)
    return p_nfd if p_nfd.exists() else p


def load_track_index(root: Path) -> dict:
    """Return {p_id: {s_id, singer, row, audio_path, f0_path, annotation_path}} for all tracks."""
    index = {}
    with open(root / "total_metadata.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            s_id = row["정제데이터 파일명"].strip()
            if not s_id:
                continue
            p_id = p_id_from_s_id(s_id)
            base = f"{row['Old ID'].strip()}_{row['장르'].strip()}_{row['하위장르'].strip()}"

            # Resolve audio path (handles NFC/NFD filesystem variations)
            audio_path = _resolve_existing(root / "KVocSet", f"{base}.mp3")

            # Resolve F0 path (handles NFC/NFD filesystem variations)
            f0_path = _resolve_existing(root / "features/pitch_output/rmvpe", f"{base}_f0.npy")
            f0_path = f0_path if f0_path.exists() else None

            index[p_id] = {
                "s_id": s_id,
                "singer": row["가창자"].strip(),
                "row": row,
                "audio_path": audio_path,
                "f0_path": f0_path,
                "annotation_path": root / "raw_json" / f"{p_id}.json",
            }
    return index
