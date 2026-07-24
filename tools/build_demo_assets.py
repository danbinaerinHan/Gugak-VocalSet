# tools/build_demo_assets.py
"""Generate all demo-page assets into docs/assets/. Run locally from repo root:

    python3 -m tools.build_demo_assets --stage all

Stages: tracks (demo JSONs + audio copy), cqt (viewer tiles), gallery, ontology.
"""
import argparse, filecmp, json, shutil
from pathlib import Path
import numpy as np
import librosa

from tools.demo.constants import SIGIMSAE_TYPES, GROUPS, GROUP_COLORS, GROUP_LABELS_EN
from tools.demo.tracks import load_track_index
from tools.demo.track_json import build_track_payload
from tools.demo.cqt_tiles import render_tiles
from tools.demo.gallery import select_exemplars, cut_snippet

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
DEMO_TRACKS = [   # the 5 public sample tracks
    "KC_TM_JC_GJ_P000074", "KC_TM_JC_PR_P000001", "KC_TM_MF_PS_P000285",
    "KC_TM_MF_MY_P000174", "KC_TM_MF_MY_P000103",
]
# Swap bad auto-picks here after listening: {"꺾어내기": [("KC_TM_..._P000123", "S042")]}
GALLERY_OVERRIDES = {}

F0_HOP = 0.01   # RMVPE hop (10 ms)


def voiced_ratio(f0_hz, start, end, hop=F0_HOP):
    seg = f0_hz[int(start / hop):max(int(start / hop) + 1, int(end / hop))]
    return float((seg > 0).mean()) if len(seg) else 0.0


def validate_demo_tracks(index):
    problems = []
    for p_id in DEMO_TRACKS:
        if p_id not in index:
            problems.append(f"{p_id}: missing from total_metadata.csv index")
            continue
        sample_mp3 = ROOT / "sample" / "audio" / f"{p_id}.mp3"
        if not sample_mp3.exists():
            problems.append(f"{p_id}: sample audio missing ({sample_mp3})")
        elif index[p_id]["audio_path"].exists() and not filecmp.cmp(sample_mp3, index[p_id]["audio_path"], shallow=False):
            problems.append(f"{p_id}: sample/audio and KVocSet audio differ — CQT/gallery would desync from played audio")
    if problems:
        raise SystemExit("demo track validation failed:\n  " + "\n  ".join(problems))


def stage_tracks(index):
    (ASSETS / "tracks").mkdir(parents=True, exist_ok=True)
    (ASSETS / "audio").mkdir(parents=True, exist_ok=True)
    for p_id in DEMO_TRACKS:
        t = index[p_id]
        ann = json.loads(t["annotation_path"].read_text(encoding="utf-8"))
        f0 = np.load(t["f0_path"]).astype(np.float32)
        payload = build_track_payload(p_id, t["row"], ann, f0)
        (ASSETS / "tracks" / f"{p_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        shutil.copy(ROOT / "sample" / "audio" / f"{p_id}.mp3", ASSETS / "audio" / f"{p_id}.mp3")
        print("tracks:", p_id, len(payload["sigimsae_regions"]), "regions")
    (ASSETS / "tracks" / "manifest.json").write_text(
        json.dumps(DEMO_TRACKS), encoding="utf-8")


def stage_cqt(index):
    out = ASSETS / "cqt"; out.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for p_id in DEMO_TRACKS:
        y, sr = librosa.load(index[p_id]["audio_path"], sr=22050, mono=True)
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
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "ontology.json").write_text(json.dumps({
        "groups": GROUPS, "group_colors": GROUP_COLORS,
        "group_labels": GROUP_LABELS_EN, "types": SIGIMSAE_TYPES,
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
