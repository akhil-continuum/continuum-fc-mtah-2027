"""The rule registry matches the rulebook ID for ID, and every entry is well formed."""
import copy

import pytest

from pipeline.rules import registry as reg


def test_registry_is_valid_and_complete():
    entries = reg.validate()
    assert len(entries) == len(reg.rulebook_ids())


def test_rulebook_ids_cover_every_section():
    ids = reg.rulebook_ids()
    for section in reg.SECTIONS:
        assert any(i.startswith(section) for i in ids), section
    assert "R0" in ids and "A24" in ids and "G10" in ids


def test_registry_ids_equal_rulebook_ids():
    book = reg.rulebook_ids()
    ids = [e["id"] for e in reg.load_registry()]
    assert sorted(ids) == sorted(book)
    assert len(ids) == len(set(ids))


def test_fails_when_rulebook_has_an_id_the_registry_lacks(tmp_path):
    text = open(reg.RULEBOOK, encoding="utf-8").read()
    extra = text.replace("| S11 |", "| S12 | A new rule. | CLUE | FEAT |\n| S11 |", 1)
    path = tmp_path / "book.md"
    path.write_text(extra, encoding="utf-8")
    with pytest.raises(reg.RegistryError, match="S12"):
        reg.validate(rulebook=str(path), check_functions=False)


def test_fails_when_registry_has_an_id_the_rulebook_lacks():
    entries = copy.deepcopy(reg.load_registry())
    ghost = dict(entries[0], id="D99")
    with pytest.raises(reg.RegistryError, match="D99"):
        reg.validate(entries + [ghost], check_functions=False)


def test_fails_on_bad_fields():
    entries = copy.deepcopy(reg.load_registry())
    entries[0]["status"] = "draftish"
    entries[1]["function"] = "pipeline.restarts.kickoff.no_such_function"
    entries[1]["status"] = "implemented"
    del entries[2]["params"]
    with pytest.raises(reg.RegistryError) as info:
        reg.validate(entries)
    msg = str(info.value)
    assert "draftish" in msg and "no_such_function" in msg and "params" in msg


def test_implemented_rules_resolve_to_code():
    for e in reg.load_registry():
        if e["status"] in ("implemented", "measured", "in_use"):
            assert callable(reg.resolve(e["function"])), e["id"]
