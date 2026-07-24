from tools.demo.gallery import select_exemplars


def region(track, rid, dur, voiced=1.0, types=("추성",)):
    return {"track": track, "id": rid, "start": 10.0, "end": 10.0 + dur,
            "types": list(types), "voiced_ratio": voiced}


def test_selects_two_from_distinct_tracks_near_median():
    cands = [region("A", "S001", 0.5), region("A", "S002", 0.5),
             region("B", "S001", 0.48), region("C", "S001", 5.0)]   # C: far from median
    sel = select_exemplars({"추성": cands}, per_type=2)
    assert [c["track"] for c in sel["추성"]] == ["A", "B"]


def test_filters_multilabel_and_unvoiced():
    cands = [region("A", "S001", 0.5, types=("추성", "퇴성")),
             region("B", "S001", 0.5, voiced=0.3),
             region("C", "S001", 0.5)]
    sel = select_exemplars({"추성": cands}, per_type=2)
    assert [c["track"] for c in sel["추성"]] == ["C"]   # 1 is fine for rare cases


def test_manual_override_wins():
    cands = [region("A", "S001", 0.5), region("B", "S001", 0.5)]
    sel = select_exemplars({"추성": cands}, per_type=2, overrides={"추성": [("B", "S001")]})
    assert [c["track"] for c in sel["추성"]] == ["B"]
