"""Smart Tagging parser, per-frame votes and hints, and the tagging stage (rulebook G1 to G9,
R14, L5). Synthetic SportsCode XML only."""
import io
import json
import os
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

from pipeline import report
from pipeline.tagging import parse, stage, teams, votes
from tagging_synth import ALIASES, AWAY, FPS, HOME, NA, inst, make_root, write_xml


def table_of(tmp_path, instances, raw=(), aliases=ALIASES):
    path = write_xml(tmp_path / "tags.xml", instances, raw)
    return parse.load(path, FPS, aliases)


def arrays(tmp_path, instances, n=1000):
    table, _ = table_of(tmp_path, instances)
    return votes.frame_arrays(table, n, FPS)


# ------------------------------------------------------------------ parser

def test_parser_case_insensitive_codes_groups_and_values(tmp_path):
    xml = [("  build   up ", 2.0, 4.0, [("TEAM", "HOME fc"), ("half", "2ND HALF")]),
           ("Set Pieces", 10.0, 15.0, [("team", "away FC"), ("TYPE", "throw-IN")]),
           ("set pieces", 20.0, 25.0, [("Team", AWAY), ("Type", "Kick Off (After Goal)")])]
    table, dropped = table_of(tmp_path, xml)
    assert dropped == 0
    assert table["code"].tolist() == ["BUILD UP", "SET PIECES", "SET PIECES"]
    assert table["family"].tolist() == ["attacking", "set_piece", "set_piece"]
    assert table["team"].tolist() == [0, 1, 1]
    assert table["half"].tolist() == [2, 0, 0]
    assert table["set_type"].tolist() == ["", "throw_in", "kick_off_after_goal"]
    assert table["start_frame"].tolist() == [50, 250, 500]
    assert table["end_frame"].tolist() == [100, 375, 625]


def test_parser_missing_groups_and_na_team(tmp_path):
    xml = [("PROGRESSION", 1.0, 2.0, []),
           ("SET PIECES", 3.0, 8.0, [("Team", NA), ("Type", "Throw-in")]),
           ("BALL IN THE BOX", 9.0, 10.0, [("Side", "Left")])]
    table, _ = table_of(tmp_path, xml)
    assert table["team"].tolist() == [-1, -1, -1]
    assert table["team_status"].tolist() == [parse.TEAM_NONE] * 3
    assert table["half"].tolist() == [0, 0, 0]
    assert table["set_type"].tolist() == ["", "throw_in", ""]


def test_parser_duplicate_groups(tmp_path):
    xml = [("BUILD UP", 1.0, 2.0, [("Team", HOME), ("Team", "home fc")]),        # same twice
           ("BUILD UP", 3.0, 4.0, [("Team", HOME), ("Team", AWAY)]),            # both sides
           ("BUILD UP", 5.0, 6.0, [("Team", "Somebody Else")]),                 # unknown text
           ("SET PIECES", 7.0, 9.0, [("Type", "Corner Kick"), ("Type", "Goal Kick")])]
    table, _ = table_of(tmp_path, xml)
    assert table["team"].tolist() == [0, -1, -1, -1]
    assert table["team_status"].tolist() == [parse.TEAM_OK, parse.TEAM_CONFLICT,
                                              parse.TEAM_UNMAPPED, parse.TEAM_NONE]
    assert table["set_type"].iloc[3] == parse.OTHER_TYPE


def test_parser_drops_unreadable_intervals_only(tmp_path):
    raw = ["<instance><ID>90</ID><code>BUILD UP</code><end>3</end></instance>",
           "<instance><ID>91</ID><start>x</start><end>3</end><code>BUILD UP</code></instance>",
           "<instance><ID>92</ID><start>5</start><end>4</end><code>BUILD UP</code></instance>",
           "<instance><ID>93</ID><start>5</start><end>6</end></instance>"]
    table, dropped = table_of(tmp_path, [], raw=raw)
    assert dropped == 3
    assert len(table) == 1 and table["code"].iloc[0] == "" and table["family"].iloc[0] == "other"


def test_team_alias_table_and_tracking_check():
    aliases = teams.aliases_for("BAR-ATH")
    assert aliases["barcelona"] == 0 and aliases["atletic club"] == 1
    assert teams.aliases_for("no_such_game") == {}
    assert teams.check_against_tracking(aliases, "FC Barcelona", "Athletic Club") == teams.AGREE
    assert teams.check_against_tracking(aliases, "Athletic Club", "FC Barcelona") == \
        teams.DISAGREE
    assert teams.check_against_tracking(aliases, None, None) == teams.UNKNOWN


# ------------------------------------------------------------------ votes (G1, G2, G4, G9)

def test_attacking_code_votes_for_tagged_team(tmp_path):
    out = arrays(tmp_path, [inst("BUILD UP", 100, 200, HOME), inst("PROGRESSION", 300, 400, AWAY)])
    bu, pr = out["vote_build_up"].to_numpy(), out["vote_progression"].to_numpy()
    assert (bu[100:200] == 1).all() and bu[:100].sum() == 0 and bu[200:].sum() == 0
    assert (pr[300:400] == -1).all() and (pr[:300] == 0).all() and (pr[400:] == 0).all()
    assert out["vote_build_up"].dtype == np.int8


def test_defending_code_votes_for_the_other_team(tmp_path):
    """Trap: a DEFENDING tag names the team without the ball."""
    out = arrays(tmp_path, [inst("Defending in Middle Third", 100, 200, HOME),
                            inst("DEFENDING IN DEFENSIVE THIRD", 300, 400, AWAY)])
    assert (out["vote_defending_in_middle_third"].to_numpy()[100:200] == -1).all()
    assert (out["vote_defending_in_defensive_third"].to_numpy()[300:400] == 1).all()
    assert (out["tag_vote"].to_numpy()[100:200] == -1).all()


def test_transitions_vote(tmp_path):
    out = arrays(tmp_path, [inst("ATTACKING TRANSITION", 100, 150, AWAY),
                            inst("DEFENSIVE TRANSITION", 300, 350, AWAY)])
    assert (out["vote_attacking_transition"].to_numpy()[100:150] == -1).all()
    assert (out["vote_defensive_transition"].to_numpy()[300:350] == 1).all()


def test_long_ball_votes_in_its_first_second_only(tmp_path):
    """Trap (G9): the end of a long ball is a contest, so only the first second votes."""
    out = arrays(tmp_path, [inst("LONG BALL", 100, 200, HOME)])
    lb = out["vote_long_ball"].to_numpy()
    assert (lb[100:100 + FPS] == 1).all()
    assert (lb[100 + FPS:] == 0).all() and (lb[:100] == 0).all()


def test_overlapping_opposite_instances_of_one_code_cancel(tmp_path):
    out = arrays(tmp_path, [inst("BALL IN THE BOX", 100, 200, HOME),
                            inst("BALL IN THE BOX", 150, 250, AWAY)])
    v = out["vote_ball_in_the_box"].to_numpy()
    assert (v[100:150] == 1).all() and (v[150:200] == 0).all() and (v[200:250] == -1).all()


def test_tag_vote_is_unweighted_sign_and_no_team_no_vote(tmp_path):
    out = arrays(tmp_path, [inst("BUILD UP", 100, 200, HOME),
                            inst("PROGRESSION", 100, 200, HOME),
                            inst("BALL IN THE BOX", 100, 300, AWAY),
                            inst("PLAYERS IN THE BOX", 400, 500, NA),
                            inst("CREATING CHANCES", 600, 700)])
    tv = out["tag_vote"].to_numpy()
    assert (tv[100:200] == 1).all()          # 2 for H against 1 for A
    assert (tv[200:300] == -1).all()
    assert (tv[400:500] == 0).all() and (out["vote_players_in_the_box"].to_numpy() == 0).all()
    assert (out["vote_creating_chances"].to_numpy() == 0).all()


def test_no_tag_means_nothing(tmp_path):
    """G7: frames without a tag carry no evidence of any kind."""
    out = arrays(tmp_path, [])
    assert len(out) == 1000
    for column in votes.VOTE_COLUMNS + ("tag_vote",):
        assert (out[column] == 0).all()
    assert (out["turnover_hint_to"] == -1).all() and not out["restart_window"].any()
    assert (out["restart_team"] == -1).all() and (out["restart_type"] == "").all()
    assert not out["goal_hint"].any() and (out["goal_conceding_team"] == -1).all()


# ------------------------------------------------------------------ hints (G4, G5, G6)

def test_turnover_hint_window_ends_at_the_tag_start(tmp_path):
    out = arrays(tmp_path, [inst("ATTACKING TRANSITION", 300, 350, HOME),
                            inst("DEFENSIVE TRANSITION", 600, 650, HOME)])
    hint = out["turnover_hint_to"].to_numpy()
    w = int(votes.TURNOVER_HINT_S * FPS)
    assert (hint[300 - w:300] == 0).all() and hint[300] == -1 and hint[300 - w - 1] == -1
    assert (hint[600 - w:600] == 1).all() and hint[600] == -1
    assert hint.dtype == np.int8


def test_turnover_hints_to_both_teams_cancel_and_clip(tmp_path):
    out = arrays(tmp_path, [inst("ATTACKING TRANSITION", 300, 350, HOME),
                            inst("ATTACKING TRANSITION", 280, 330, AWAY),
                            inst("ATTACKING TRANSITION", 10, 40, AWAY),
                            inst("ATTACKING TRANSITION", 700, 750, NA)])
    hint = out["turnover_hint_to"].to_numpy()
    assert (hint[230:250] == 1).all() and (hint[250:280] == -1).all()
    assert (hint[280:300] == 0).all()
    assert (hint[0:10] == 1).all()                        # clipped at the video start
    assert (hint[650:700] == -1).all()                    # no team: no hint


def test_restart_windows_throw_in_has_no_team(tmp_path):
    out = arrays(tmp_path, [inst("SET PIECES", 100, 225, NA, "Throw-in"),
                            inst("SET PIECES", 400, 525, AWAY, "Corner Kick"),
                            inst("SET PIECES", 700, 825, HOME, "Something New")])
    w = out["restart_window"].to_numpy()
    assert w[100:225].all() and w[400:525].all() and not w[225:400].any()
    assert (out["restart_type"].iloc[100:225] == "throw_in").all()
    assert (out["restart_team"].to_numpy()[100:225] == -1).all()
    assert (out["restart_type"].iloc[400:525] == "corner").all()
    assert (out["restart_team"].to_numpy()[400:525] == 1).all()
    assert (out["restart_type"].iloc[700:825] == parse.OTHER_TYPE).all()
    assert (out["restart_team"].to_numpy()[700:825] == 0).all()


def test_overlapping_restart_windows_disagree_to_no_team(tmp_path):
    out = arrays(tmp_path, [inst("SET PIECES", 100, 225, NA, "Throw-in"),
                            inst("SET PIECES", 200, 300, AWAY, "Corner Kick")])
    team, kind = out["restart_team"].to_numpy(), out["restart_type"]
    assert (team[200:225] == -1).all() and (team[225:300] == 1).all()
    assert (kind.iloc[200:225] == parse.OTHER_TYPE).all()
    assert (kind.iloc[225:300] == "corner").all()


def test_goal_hint_exposes_the_conceding_team(tmp_path):
    out = arrays(tmp_path, [inst("GOALS", 400, 450, HOME), inst("GOALS", 800, 820, NA)])
    pad = int(votes.GOAL_PAD_S * FPS)
    g, c = out["goal_hint"].to_numpy(), out["goal_conceding_team"].to_numpy()
    assert g[400 - pad:450 + pad].all() and not g[400 - pad - 1] and not g[450 + pad]
    assert (c[400 - pad:450 + pad] == 1).all()
    assert g[800:820].all() and (c[800:820] == -1).all()
    # phase tags never vote on D (G3): a goal tag casts no team vote
    assert (out["tag_vote"] == 0).all()


def test_frame_arrays_columns_and_length(tmp_path):
    out = arrays(tmp_path, [inst("BUILD UP", 900, 1200, HOME)], n=1000)
    assert out["frame"].tolist() == list(range(1000))
    assert out["vote_build_up"].to_numpy()[900:].all()     # clipped at the video end
    expected = {"frame", "tag_vote", "turnover_hint_to", "restart_window", "restart_type",
                "restart_team", "goal_hint", "goal_conceding_team"} | set(votes.VOTE_COLUMNS)
    assert set(out.columns) == expected
    assert len(votes.VOTE_COLUMNS) == 12


# ------------------------------------------------------------------ stage

SCENE = [inst("BUILD UP", 200, 400, HOME, half="1st Half"),
         inst("DEFENDING IN MIDDLE THIRD", 300, 500, HOME, half="1st Half"),
         inst("LONG BALL", 600, 700, AWAY),
         inst("SET PIECES", 800, 925, NA, "Throw-in"),
         inst("SET PIECES", 1000, 1125, AWAY, "Corner Kick"),
         inst("ATTACKING TRANSITION", 1300, 1350, AWAY),
         inst("GOALS", 1500, 1550, HOME),
         inst("SET PIECES", 2600, 2725, AWAY, "Kick Off (After Goal)"),
         inst("WEIRD NEW CODE", 1700, 1750, HOME)]


def test_stage_writes_tags_and_returns_printable_aggregates(tmp_path):
    root = make_root(tmp_path / "root", 3000, [[100, 2900]])
    xml = write_xml(tmp_path / "t.xml", SCENE)
    stats = stage.run(root, xml_path=xml, aliases=ALIASES)
    assert stats["written"] is True
    frames = pd.read_parquet(os.path.join(root, stage.TAGS_FILE))
    assert len(frames) == 3000 and frames["restart_window"].sum() == 3 * 125
    assert stats["instances_total"] == len(SCENE)
    assert stats["instances_by_code"]["SET PIECES"] == 3
    assert stats["instances_by_code"]["OTHER"] == 1 and stats["unknown_codes_distinct"] == 1
    assert stats["instances_by_team"]["no_team"] == 1
    assert stats["share_half_tagged_instances_inside_that_half"] == 1.0
    assert stats["goal_tags_followed_by_kickoff_tag"] == 1
    assert stats["goal_kickoff_tag_is_conceding_team"] == 1
    assert stats["graded_frames_voting_by_code"]["LONG BALL"] == FPS
    assert stats["share_voting_frames_codes_conflict"] == round(100 / (200 + 100 + FPS + 50), 4)
    report.summary(_name="tagging_test", _reports_dir=str(tmp_path / "reports"), _quiet=True,
                   **stats)


def test_stage_says_no_tags_on_a_bench_root_and_writes_nothing(tmp_path):
    root = make_root(tmp_path / "bench_root", 500, [[0, 500]], source="bench", game_id="g")
    assert stage.run(root) == {"tags": stage.NO_TAGS, "written": False}
    assert not os.path.exists(os.path.join(root, stage.TAGS_FILE))


def _match_run(tmp_path, monkeypatch, argv, extra_files=None):
    """Run the tagging entry point on a synthetic 'match' root under a temporary artifacts dir."""
    import pipeline.ball.stage as shared
    from pipeline.tagging import __main__ as cli
    artifacts = tmp_path / "artifacts"
    root = make_root(artifacts / "match", 25000, [[1000, 24000]], source="match",
                     game_id="BAR-ATH")
    if extra_files:
        extra_files(root)
    distinct = [inst("BUILD UP", 20309, 20478, "Barcelona"),
                inst("SET PIECES", 21111, 21236, "Atletic Club", "Corner Kick"),
                inst("ATTACKING TRANSITION", 22223, 22281, "Atletic Club"),
                inst("GOALS", 23331, 23392, "Barcelona")]
    xml = write_xml(tmp_path / "match_tags.xml", distinct)
    monkeypatch.setattr(shared, "ARTIFACTS", str(artifacts))
    monkeypatch.setattr(stage, "tagging_path", lambda meta: xml)
    monkeypatch.setattr(stage, "tracking_team_names", lambda meta: ("FC Barcelona",
                                                                    "Athletic Club"))
    monkeypatch.setattr(report, "REPORTS_DIR", tmp_path / "reports")
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = cli.entry(argv)
    return code, buf.getvalue(), root


SECRETS = ("20309", "20478", "21111", "21236", "22223", "22281", "23331", "23392",
           "812.36", "819.12", "844.44", "933.24")


def _assert_aggregates_only(text):
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert lines
    for line in lines:
        head, _, value = line.partition(" = ")
        assert ": " in head and value
        parsed = json.loads(value)
        if isinstance(parsed, (list, dict)):
            assert len(parsed) <= report.MAX_ITEMS
    for secret in SECRETS:
        assert secret not in text


def test_match_entry_point_prints_no_instance_values(tmp_path, monkeypatch):
    code, text, root = _match_run(tmp_path, monkeypatch, ["match"])
    assert code == 0
    _assert_aggregates_only(text)
    assert "tagging: team_alias_check_against_tracking_meta = \"agree\"" in text
    assert os.path.isfile(os.path.join(root, stage.TAGS_FILE))


def test_match_agreement_entry_point_prints_no_instance_values(tmp_path, monkeypatch):
    def labels(root):
        os.makedirs(os.path.join(root, "labels"))
        lab = np.array(["X"] * 25000, dtype=object)
        lab[1000:24000] = "H"
        lab[21000:21300] = "D"
        lab[21300:22250] = "A"
        pd.DataFrame({"frame": np.arange(25000), "label": lab}).to_csv(
            os.path.join(root, "labels", "vtest.csv"), index=False)
    code, text, _ = _match_run(tmp_path, monkeypatch,
                               ["match", "--agreement", "--ref-version", "vtest"], labels)
    assert code == 0
    _assert_aggregates_only(text)
    assert "tagging_agreement_vtest: agreement_build_up" in text


def test_cli_refuses_bad_option_combinations(capsys):
    from pipeline.tagging import __main__ as cli
    assert cli.entry(["match", "--agreement"]) == 2
    assert cli.entry(["match", "--ref-version", "v6s"]) == 2
    assert cli.entry(["match", "--agreement", "--ref", "pseudo", "--ref-version", "v6s"]) == 2
    assert cli.entry(["match", "--agreement", "--ref-version", "../x"]) == 2
