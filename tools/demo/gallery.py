"""Select and cut gallery exemplar snippets for the 17 sigimsae types."""
import subprocess
from statistics import median

PAD_SEC = 0.5
MIN_VOICED = 0.8


def select_exemplars(candidates_by_type: dict, per_type: int = 2, overrides: dict = None) -> dict:
    """candidates_by_type: {kr_type: [ {track, id, start, end, types, voiced_ratio} ]}"""
    overrides = overrides or {}
    out = {}
    for kr, cands in candidates_by_type.items():
        if kr in overrides:
            keys = set(overrides[kr])
            out[kr] = [c for c in cands if (c["track"], c["id"]) in keys]
            continue
        ok = [c for c in cands if len(c["types"]) == 1 and c["voiced_ratio"] >= MIN_VOICED]
        if not ok:
            out[kr] = []
            continue
        med = median(c["end"] - c["start"] for c in ok)
        ok = [c for c in ok if 0.5 * med <= (c["end"] - c["start"]) <= 1.5 * med]
        ok.sort(key=lambda c: abs((c["end"] - c["start"]) - med))
        picked, used_tracks = [], set()
        for c in ok:                                   # pass 1: distinct tracks
            if len(picked) == per_type:
                break
            if c["track"] not in used_tracks:
                picked.append(c); used_tracks.add(c["track"])
        for c in ok:                                   # pass 2: fill up if needed
            if len(picked) == per_type:
                break
            if c not in picked:
                picked.append(c)
        out[kr] = picked
    return out


def cut_snippet(audio_path, start: float, end: float, out_path) -> None:
    s, dur = max(0.0, start - PAD_SEC), (end - start) + 2 * PAD_SEC
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{s:.3f}", "-t", f"{dur:.3f}",
         "-i", str(audio_path), "-codec:a", "libmp3lame", "-qscale:a", "4", str(out_path)],
        check=True)
