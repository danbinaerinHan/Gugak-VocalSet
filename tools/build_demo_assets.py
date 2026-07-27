# tools/build_demo_assets.py
"""Generate all demo-page assets into docs/assets/. Run locally from repo root:

    python3 -m tools.build_demo_assets --stage all

Stages: tracks (demo JSONs + audio previews), cqt (viewer tiles), gallery, ontology.
"""
import argparse, json, shutil, subprocess
from pathlib import Path
import numpy as np
import librosa

from tools.demo.constants import (SIGIMSAE_TYPES, GROUPS, GROUP_COLORS, GROUP_LABELS_EN,
                                  CONTOUR_H_CM)
from tools.demo.tracks import load_track_index
from tools.demo.track_json import build_track_payload
from tools.demo.cqt_tiles import render_tiles
from tools.demo.gallery import select_exemplars, cut_snippet

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
SAMPLE_AUDIO = ROOT / "sample" / "audio"
PAPER_FIGURES = ROOT / "ismir2026paper" / "figures"
CM_PX = 74   # web rendering of the paper's typeset heights (0.6 cm contour -> ~44 px)
DEMO_TRACKS = [   # the 5 public sample tracks
    "KC_TM_JC_GJ_P000074", "KC_TM_JC_PR_P000001", "KC_TM_MF_PS_P000285",
    "KC_TM_MF_MY_P000174", "KC_TM_MF_MY_P000103",
]
# Swap bad auto-picks here after listening: {"꺾어내기": [("KC_TM_..._P000123", "S042")]}
GALLERY_OVERRIDES = {}

F0_HOP = 0.01   # RMVPE hop (10 ms)
# Only the first PREVIEW_SEC of each demo track may be published until the dataset is
# cleared for release. Everything derived from full-song audio (sample/audio, docs audio,
# CQT tiles, the F0/regions in the track JSONs) is cut to this window.
PREVIEW_SEC = 40.0
MAX_LAG_MS = 5.0   # published preview must start on the same sample as the source


def cut_preview(src: Path, dst: Path, sec: float = PREVIEW_SEC) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-t", f"{sec:.3f}",
         "-codec:a", "libmp3lame", "-qscale:a", "2", str(dst)], check=True)


def preview_lag_ms(src: Path, preview: Path, sr: int = 22050, probe_sec: float = 10.0) -> float:
    """Offset of the re-encoded preview against the source, in ms (should be ~0).

    A silent shift here would desync the F0 contour and the region boundaries from the
    audio the browser plays, so the build refuses to ship one.
    """
    a, _ = librosa.load(src, sr=sr, mono=True, duration=probe_sec)
    b, _ = librosa.load(preview, sr=sr, mono=True, duration=probe_sec)
    n = min(len(a), len(b))
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    max_lag = int(sr * MAX_LAG_MS / 1000) * 4
    xc = np.correlate(np.pad(b, max_lag), a, mode="valid")   # index max_lag == zero lag
    return (int(np.argmax(xc)) - max_lag) / sr * 1000        # >0: preview lags the source


def voiced_ratio(f0_hz, start, end, hop=F0_HOP):
    seg = f0_hz[int(start / hop):max(int(start / hop) + 1, int(end / hop))]
    return float((seg > 0).mean()) if len(seg) else 0.0


def validate_demo_tracks(index):
    """KVocSet audio is the single source for every published artifact, previews included."""
    problems = []
    for p_id in DEMO_TRACKS:
        if p_id not in index:
            problems.append(f"{p_id}: missing from total_metadata.csv index")
            continue
        if not index[p_id]["audio_path"].exists():
            problems.append(f"{p_id}: KVocSet audio missing ({index[p_id]['audio_path']})")
    if problems:
        raise SystemExit("demo track validation failed:\n  " + "\n  ".join(problems))


def stage_tracks(index):
    (ASSETS / "tracks").mkdir(parents=True, exist_ok=True)
    (ASSETS / "audio").mkdir(parents=True, exist_ok=True)
    SAMPLE_AUDIO.mkdir(parents=True, exist_ok=True)
    for p_id in DEMO_TRACKS:
        t = index[p_id]
        ann = json.loads(t["annotation_path"].read_text(encoding="utf-8"))
        f0 = np.load(t["f0_path"]).astype(np.float32)
        payload = build_track_payload(p_id, t["row"], ann, f0, preview_sec=PREVIEW_SEC)
        (ASSETS / "tracks" / f"{p_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        preview = ASSETS / "audio" / f"{p_id}.mp3"
        cut_preview(t["audio_path"], preview)
        lag = preview_lag_ms(t["audio_path"], preview)
        if abs(lag) > MAX_LAG_MS:
            raise SystemExit(f"{p_id}: preview is {lag:+.1f} ms off the source audio")
        shutil.copy(preview, SAMPLE_AUDIO / f"{p_id}.mp3")   # sample/ ships the same excerpt
        print(f"tracks: {p_id} {len(payload['sigimsae_regions'])} regions, "
              f"{PREVIEW_SEC:.0f}s preview ({lag:+.2f} ms lag)")
    (ASSETS / "tracks" / "manifest.json").write_text(
        json.dumps(DEMO_TRACKS), encoding="utf-8")


def stage_cqt(index):
    out = ASSETS / "cqt"; out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.png"):        # shorter previews leave orphan tiles behind
        stale.unlink()
    manifests = {}
    for p_id in DEMO_TRACKS:
        y, sr = librosa.load(index[p_id]["audio_path"], sr=22050, mono=True, duration=PREVIEW_SEC)
        manifests[p_id] = render_tiles(y, sr, out, track_id=p_id)
        print("cqt:", p_id, manifests[p_id]["n_tiles"], "tiles")
    (out / "manifest.json").write_text(json.dumps(manifests), encoding="utf-8")


def stage_gallery(index):
    out = ASSETS / "gallery"; out.mkdir(parents=True, exist_ok=True)
    cands, counts = {k: [] for k in SIGIMSAE_TYPES}, {k: 0 for k in SIGIMSAE_TYPES}
    for p_id, t in index.items():
        if t["f0_path"] is None or not t["annotation_path"].exists():
            continue
        ann = json.loads(t["annotation_path"].read_text(encoding="utf-8"))
        f0 = np.load(t["f0_path"]).astype(np.float32)
        for r in ann["annotation"]["sigimsage_regions"]:
            for typ in r["sigimsage_types"]:
                if typ in counts:
                    counts[typ] += 1
            typ = r["sigimsage_types"][0]
            if len(r["sigimsage_types"]) == 1 and typ in cands:
                cands[typ].append({
                    "track": p_id, "id": r["sigimsage_id"],
                    "start": r["start_sec"], "end": r["end_sec"],
                    "types": r["sigimsage_types"],
                    "voiced_ratio": voiced_ratio(f0, r["start_sec"], r["end_sec"]),
                })
    selected = select_exemplars(cands, per_type=2, overrides=GALLERY_OVERRIDES)
    gallery = {}
    for kr, picks in selected.items():
        gallery[kr] = {"count": counts[kr], "snippets": []}
        for i, c in enumerate(picks):
            name = f"{kr.replace('(', '_').replace(')', '').replace(' ', '_')}_{i}"
            cut_snippet(index[c["track"]]["audio_path"], c["start"], c["end"],
                        out / f"{name}.mp3")
            f0 = np.load(index[c["track"]]["f0_path"]).astype(np.float32)
            s, e = max(0.0, c["start"] - 0.5), c["end"] + 0.5
            seg = f0[int(s / F0_HOP):int(e / F0_HOP)]
            gallery[kr]["snippets"].append({
                "file": f"{name}.mp3", "track": c["track"], "region": c["id"],
                "start_in_clip": c["start"] - s, "end_in_clip": c["end"] - s,
                "f0": [None if v <= 0 else round(float(v), 1) for v in seg],
                "f0_hop": F0_HOP,
            })
        print("gallery:", kr, counts[kr], "instances ->", len(picks), "snippets")
    (out / "gallery.json").write_text(json.dumps(gallery, ensure_ascii=False), encoding="utf-8")


def stage_ontology():
    """Ontology JSON + the notation-symbol / pitch-contour art from the paper's Figure 2."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    for kind in ("symbols", "contours"):
        src = PAPER_FIGURES / f"sigimsae_{kind}"
        dst = ASSETS / "sigimsae" / kind
        dst.mkdir(parents=True, exist_ok=True)
        missing = [t["slug"] for t in SIGIMSAE_TYPES.values() if not (src / f"{t['slug']}.png").exists()]
        if missing:
            raise SystemExit(f"ontology art missing from {src}: {', '.join(missing)}")
        for t in SIGIMSAE_TYPES.values():
            shutil.copy(src / f"{t['slug']}.png", dst / f"{t['slug']}.png")
        print(f"ontology: {len(SIGIMSAE_TYPES)} {kind}")
    (ASSETS / "ontology.json").write_text(json.dumps({
        "groups": GROUPS, "group_colors": GROUP_COLORS,
        "group_labels": GROUP_LABELS_EN, "types": SIGIMSAE_TYPES,
        "contour_h_cm": CONTOUR_H_CM, "cm_px": CM_PX,
    }, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "tracks", "cqt", "gallery", "ontology"])
    args = ap.parse_args()
    index = load_track_index(ROOT)
    validate_demo_tracks(index)
    if args.stage in ("all", "ontology"):
        stage_ontology()
    if args.stage in ("all", "tracks"):
        stage_tracks(index)
    if args.stage in ("all", "cqt"):
        stage_cqt(index)
    if args.stage in ("all", "gallery"):
        stage_gallery(index)
