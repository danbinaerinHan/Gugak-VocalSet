from pathlib import Path

import pytest

from tools.demo.constants import SIGIMSAE_TYPES, GROUPS, GROUP_COLORS, CONTOUR_H_CM

def test_seventeen_types_and_group_sizes():
    assert len(SIGIMSAE_TYPES) == 17
    by_group = {}
    for t in SIGIMSAE_TYPES.values():
        by_group.setdefault(t["group"], []).append(t)
    assert {g: len(v) for g, v in by_group.items()} == {
        "onset": 4, "vibrato": 4, "mid": 2, "offset": 5, "accent": 1, "transition": 1,
    }

def test_every_type_fully_described():
    for kr, t in SIGIMSAE_TYPES.items():
        assert t["group"] in GROUPS
        assert t["roman"] and t["en"], kr

def test_groups_have_unique_colors():
    assert len(set(GROUP_COLORS.values())) == len(GROUPS)


def test_every_type_carries_the_papers_symbol_and_description():
    slugs = set()
    for kr, t in SIGIMSAE_TYPES.items():
        assert t["slug"] and t["desc"], kr
        assert 0 < t["sym_h_cm"] <= CONTOUR_H_CM, kr
        slugs.add(t["slug"])
    assert len(slugs) == 17          # one distinct art pair per type


def test_ontology_art_files_exist():
    figures = Path(__file__).resolve().parents[2] / "ismir2026paper" / "figures"
    if not figures.exists():
        pytest.skip("paper figures not checked out")
    for kr, t in SIGIMSAE_TYPES.items():
        for kind in ("symbols", "contours"):
            assert (figures / f"sigimsae_{kind}" / f"{t['slug']}.png").exists(), f"{kr}/{kind}"
