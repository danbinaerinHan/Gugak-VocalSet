from tools.demo.constants import SIGIMSAE_TYPES, GROUPS, GROUP_COLORS

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
