"""G10 agreement table: arithmetic on a synthetic root with known labels (no match data).

Layout (frames, 25 fps): graded [100, 3900); labels H [100, 600), D [600, 1500), A [1500, 2000),
H [2000, 3900). CORE_S = 2 s = 50 frames, so core frames are H [150, 550), A [1550, 1950),
H [2050, 3850). Ball seen on frames below 300 only.
"""
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

from pipeline import report
from pipeline.tagging import agreement
from tagging_synth import ALIASES, AWAY, FPS, HOME, NA, inst, make_root, write_xml

N = 4000
TAGS = [inst("BUILD UP", 200, 400, HOME),
        inst("PROGRESSION", 300, 500, AWAY),
        inst("DEFENDING IN MIDDLE THIRD", 1600, 1800, HOME),
        inst("BALL IN THE BOX", 560, 640, AWAY),
        inst("SET PIECES", 1480, 1520, AWAY, "Free Kick"),
        inst("SET PIECES", 3000, 3050, NA, "Throw-in"),
        inst("ATTACKING TRANSITION", 1510, 1560, AWAY),
        inst("DEFENSIVE TRANSITION", 2100, 2150, AWAY),
        inst("GOALS", 590, 610, HOME),
        inst("LONG BALL", 2500, 2600, HOME)]
RESTARTS = pd.DataFrame({
    "start_frame":    [600, 2900, 250, -1],
    "restart_frame":  [1505, 3060, 300, 1400],
    "start_kind":     ["slowdown", "exit", "exit", "goal"],
    "restart_type":   ["free_kick", "throw_in", "corner", "kick_off"],
    "p_home_restart": [0.2, 0.5, 0.9, 0.1]})


def labels_array():
    lab = np.array(["X"] * N, dtype=object)
    lab[100:600] = "H"
    lab[600:1500] = "D"
    lab[1500:2000] = "A"
    lab[2000:3900] = "H"
    return lab


def build_root(tmp_path, restarts=True, ball=True):
    root = make_root(tmp_path / "root", N, [[100, 3900]])
    os.makedirs(os.path.join(root, "labels"))
    pd.DataFrame({"frame": np.arange(N), "label": labels_array()}).to_csv(
        os.path.join(root, "labels", "vt.csv"), index=False)
    if ball:
        pd.DataFrame({"frame": np.arange(N, dtype=np.int32),
                      "seen": np.arange(N) < 300}).to_parquet(
            os.path.join(root, "ball.parquet"), index=False)
    if restarts:
        RESTARTS.to_parquet(os.path.join(root, "restarts.parquet"), index=False)
    xml = write_xml(tmp_path / "tags.xml", TAGS)
    return root, xml


@pytest.fixture
def table(tmp_path):
    root, xml = build_root(tmp_path)
    out = agreement.run(root, ref_version="vt", xml_path=xml, aliases=ALIASES)
    report.summary(_name="agreement_test", _reports_dir=str(tmp_path / "reports"), _quiet=True,
                   **out)     # everything is printable through summary()
    return out


def test_core_mask_keeps_frames_away_from_changes():
    owner = np.array([-1] * 3 + [0] * 10 + [2] * 4 + [1] * 6, dtype=np.int8)
    core = agreement.core_mask(owner, 2)
    assert np.flatnonzero(core).tolist() == list(range(5, 11)) + list(range(19, 23))
    assert agreement.core_mask(np.zeros(5, np.int8), 2).all()      # array edges are no change


def test_reference_summary(table):
    ref = table["reference"]
    assert ref["graded_frames"] == 3800
    assert ref["core_frames"] == 400 + 400 + 1800
    assert ref["noncore_live_frames"] == 3800 - 900 - 2600
    assert ref["share_graded_d"] == round(900 / 3800, 4)


def test_per_code_agreement(table):
    bu = table["agreement_build_up"]
    assert (bu["tags"], bu["core_covered"], bu["core_voting"], bu["agree_core"]) == \
        (1, 200, 200, 1.0)
    assert (bu["core_voting_seen"], bu["agree_core_seen"]) == (100, 1.0)
    assert (bu["core_voting_missing"], bu["agree_core_missing"]) == (100, 1.0)
    pr = table["agreement_progression"]
    assert (pr["core_voting"], pr["agree_core"]) == (200, 0.0)
    assert (pr["core_voting_seen"], pr["agree_core_seen"]) == (0, None)
    de = table["agreement_defending_in_middle_third"]
    assert (de["core_voting"], de["agree_core"]) == (200, 1.0)     # flipped vote agrees with A
    bb = table["agreement_ball_in_the_box"]
    assert (bb["core_voting"], bb["agree_core"]) == (0, None)
    assert (bb["noncore_voting"], bb["agree_noncore_uncertain"]) == (40, 0.0)
    assert bb["share_voting_ref_d"] == 0.5
    lb = table["agreement_long_ball"]
    assert (lb["frames_covered"], lb["voting_frames"], lb["core_voting"]) == (100, FPS, FPS)
    at = table["agreement_attacking_transition"]
    assert (at["core_voting"], at["noncore_voting"], at["agree_core"]) == (10, 40, 1.0)


def test_combined_vote_agreement(table):
    tv = table["agreement_tag_vote"]
    # +1 [200,300) agrees; [300,400) cancels; -1 [400,500) disagrees; the rest agree
    assert tv["core_voting"] == 100 + 100 + 200 + 10 + 50 + FPS
    assert tv["agree_core"] == round((100 + 200 + 10 + 50 + FPS) / tv["core_voting"], 4)
    assert tv["tags"] == 7            # voting codes only (no SET PIECES, no GOALS)


def test_set_pieces(table):
    sp = table["set_pieces"]
    assert sp["windows"] == 2
    assert sp["share_with_restart_inside"] == 0.5        # the throw-in restart is 0.4 s late
    assert sp["share_with_restart_within_tol"] == 1.0
    assert (sp["team_pairs"], sp["team_agree_p_home_restart"]) == (1, 1.0)
    assert (sp["type_pairs"], sp["type_agree"]) == (2, 1.0)
    assert (sp["detected_restarts"], sp["share_detected_restarts_in_a_window"]) == (4, 0.5)
    assert sp["share_with_ref_restart_within_tol"] == 0.5
    assert (sp["ref_team_pairs"], sp["ref_team_agree"]) == (1, 1.0)
    assert table["set_pieces_within_tol_by_type"] == {"free_kick": 1.0, "throw_in": 1.0}


def test_transitions(table):
    at = table["transition_attacking_transition"]
    assert at["tags_with_team"] == 1
    assert at["share_change_to_team_in_hint_window"] == 1.0
    assert at["share_change_to_team_within_reach"] == 1.0
    dt = table["transition_defensive_transition"]
    assert dt["share_change_to_team_in_hint_window"] == 0.0     # the change to H is 4 s early
    assert dt["share_change_to_team_within_reach"] == 1.0
    chance = table["transition_chance_rates"]
    assert (chance["ref_changes_to_home"], chance["ref_changes_to_away"]) == (1, 1)
    assert 0 < chance["chance_change_any_team_in_hint_window"] < \
        chance["chance_change_any_team_within_reach"] < 1


def test_goals(table):
    g = table["goals"]
    assert (g["tags"], g["followed_by_long_ref_d"], g["followed_by_long_stoppage"]) == (1, 1, 1)
    assert (g["followed_by_goal_kickoff"], g["kickoff_team_pairs"],
            g["kickoff_is_conceding_team"]) == (1, 1, 1)


def test_without_restarts_and_ball_files(tmp_path):
    root, xml = build_root(tmp_path, restarts=False, ball=False)
    out = agreement.run(root, ref_version="vt", xml_path=xml, aliases=ALIASES)
    assert out["set_pieces"]["restarts_file"] == "absent"
    assert out["goals"]["restarts_file"] == "absent"
    assert out["agreement_build_up"]["agree_core_seen"] is None
    assert out["agreement_build_up"]["agree_core"] == 1.0


def test_no_tags_on_a_bench_root(tmp_path):
    root = make_root(tmp_path / "bench", 500, [[0, 500]], source="bench", game_id="g")
    assert agreement.run(root, ref_version="vt") == {"tags": "no tags", "written": False}


def test_missing_version_labels_is_a_clear_error(tmp_path):
    root, xml = build_root(tmp_path)
    with pytest.raises(agreement.ReferenceUnavailable, match="labels/nope.csv"):
        agreement.run(root, ref_version="nope", xml_path=xml, aliases=ALIASES)


def test_pseudo_reference_is_pluggable(tmp_path):
    root, xml = build_root(tmp_path)
    pseudo = np.full(N, -1, dtype=np.int8)
    pseudo[200:300] = 0          # BUILD UP agrees here
    pseudo[400:450] = 1          # BUILD UP is silent, PROGRESSION agrees here
    pseudo[1600:1700] = 0        # DEFENDING (votes A) disagrees here
    pseudo[3000:3100] = 2
    out = agreement.run(root, ref="pseudo", xml_path=xml, aliases=ALIASES,
                        pseudo_fn=lambda r: pseudo)
    assert out["reference"]["name"] == "pseudo"
    assert out["reference"]["core_frames"] == 250 and out["reference"]["noncore_live_frames"] == 0
    assert (out["agreement_build_up"]["core_voting"], out["agreement_build_up"]["agree_core"]) \
        == (100, 1.0)
    assert (out["agreement_progression"]["core_voting"],
            out["agreement_progression"]["agree_core"]) == (50, 1.0)
    assert out["agreement_defending_in_middle_third"]["agree_core"] == 0.0
    assert out["agreement_build_up"]["noncore_voting"] == 0


def test_filled_owner_and_changes_bridge_pseudo_gaps():
    owner = np.array([-1, 0, -1, -1, 1, -1, 2, -1, 0, -1], dtype=np.int8)
    filled = agreement.filled_owner(owner, [(1, 9)])
    assert filled.tolist() == [-1, 0, 0, 0, 1, 1, 2, 2, 0, -1]
    assert np.flatnonzero(agreement.changes_to(filled, [(1, 9)], 1)).tolist() == [4]
    assert np.flatnonzero(agreement.changes_to(filled, [(1, 9)], 0)).tolist() == [8]


def test_pseudo_module_missing_or_incomplete(tmp_path, monkeypatch):
    root, xml = build_root(tmp_path)
    monkeypatch.setitem(sys.modules, "pipeline.models.pseudo", None)
    with pytest.raises(agreement.ReferenceUnavailable, match="T-37a"):
        agreement.run(root, ref="pseudo", xml_path=xml, aliases=ALIASES)
    monkeypatch.setitem(sys.modules, "pipeline.models.pseudo", types.ModuleType("stub"))
    with pytest.raises(agreement.ReferenceUnavailable, match="pseudo_labels"):
        agreement.run(root, ref="pseudo", xml_path=xml, aliases=ALIASES)
