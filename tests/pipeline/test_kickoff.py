"""Kick-off detection and the kick-off law (rulebook R6, R13, S4, S5, S6, L6) on scripted scenes.

Each rule has a scene where it should fire and a trap scene (rulebook A14, P23 and the
midfield-free-kick pattern seen on the bench) where it must not.
"""
import numpy as np
import pandas as pd
import pytest

from pipeline.restarts import kickoff
from pipeline.restarts.formation import kickoff_frame_measures
from restart_scenes import CX, CY, Scene, kickoff_shape

TOL = 25   # the rulebook's "within a second" check


def _run(scene, with_direction=True, ball=True):
    return kickoff.run(scene.players(), scene.meta(with_direction),
                       scene.ball_raw() if ball else None)


def _two_halves():
    s = Scene([(0, 3000), (3000, 6000)])
    s.kickoff(0, 200, 3000, kick_team=1)                 # away kicks off the first half
    s.walk_on(3000, 3500)
    s.kickoff(3500, 4000, 6000, kick_team=0)             # home kicks off the second half
    return s


def test_detects_both_half_start_kickoffs_with_timing():
    kos, _ = _run(_two_halves())
    starts = kos[kos["start_kind"] == "half_start"]
    assert list(starts["half"]) == [1, 2]
    assert abs(int(starts["restart_frame"].iloc[0]) - 200) <= TOL
    assert abs(int(starts["restart_frame"].iloc[1]) - 4000) <= TOL
    assert list(starts["start_frame"]) == [0, 3000]
    assert (kos["restart_type"] == "kick_off").all()
    assert len(kos) == 2


def test_kicking_team_from_formation_and_law_s6():
    kos, _ = _run(_two_halves())
    h1, h2 = kos.iloc[0], kos.iloc[1]
    assert h1["kick_team_llr"] < 0 < h2["kick_team_llr"]      # votes: away, then home
    assert h1["p_home_restart"] < 0.05
    assert h1["p_home_restart"] + h2["p_home_restart"] == pytest.approx(1.0)
    assert (kos.loc[kos["start_kind"] == "half_start", "law_kick_team_h1"] == 1).all()


def test_law_overrides_a_contradicting_second_half_vote():
    """S6: if both halves' formations vote for the same team, the stronger vote wins and the
    other half gets the other team; the two probabilities still sum to one."""
    rows = pd.DataFrame({"start_kind": ["half_start", "half_start"], "half": [1, 2],
                         "kick_team_llr": [3.0, 1.0], "p_home_restart": [0.95, 0.73]})
    out = kickoff.apply_kickoff_law(rows)
    assert out["p_home_restart"].iloc[0] > 0.5 > out["p_home_restart"].iloc[1]
    assert out["p_home_restart"].sum() == pytest.approx(1.0)
    assert (out["law_kick_team_h1"] == 0).all()


def test_pre_kickoff_mask_s4():
    s = _two_halves()
    kos, feats = _run(s)
    mask = feats["mask_pre_kickoff"].to_numpy()
    k1 = int(kos["restart_frame"].iloc[0])
    k2 = int(kos["restart_frame"].iloc[1])
    assert mask[:k1].all() and not mask[k1:3000].any()
    assert mask[3000:k2].all() and not mask[k2:].any()
    assert set(feats.columns) >= {"frame", "half", "form_kick_off", "mask_pre_kickoff",
                                  "p_restart_kick_off"}
    assert len(feats) == s.n_frames
    assert feats["form_kick_off"].between(0, 1).all()
    assert feats["p_restart_kick_off"].iloc[k2] > 0


def test_goal_kickoff_r13_taken_by_the_team_at_the_spot():
    s = Scene([(0, 6000), (6000, 7000)])
    s.kickoff(0, 100, 1500, kick_team=1)
    s.live(1500, 3000)
    s.kickoff(3000, 3700, 6000, kick_team=0)             # home conceded, home kicks off
    s.kickoff(6000, 6100, 7000, kick_team=0)
    kos, _ = _run(s)
    goals = kos[kos["start_kind"] == "goal"]
    assert len(goals) == 1
    g = goals.iloc[0]
    assert abs(int(g["restart_frame"]) - 3700) <= TOL
    assert g["p_home_restart"] > 0.9
    assert g["start_frame"] == -1                        # stoppage start comes from D4 later


def test_ball_unseen_throughout_still_detected():
    s = Scene([(0, 3000)])
    s.kickoff(0, 300, 3000, kick_team=0, ball_seen=False, ball_visible_after=False)
    kos, _ = _run(s)
    assert len(kos) == 1 and kos["evidence"].iloc[0] == "formation"
    assert abs(int(kos["restart_frame"].iloc[0]) - 300) <= TOL
    assert kos["kick_team_llr"].iloc[0] > 0


def test_taker_untracked_needs_the_ball_and_gets_team_from_the_law():
    s = Scene([(0, 3000), (3000, 6000)])
    s.kickoff(0, 300, 3000, kick_team=1, taker_tracked=False)
    s.kickoff(3000, 3400, 6000, kick_team=0)
    kos, _ = _run(s)
    h1 = kos.iloc[0]
    assert h1["start_kind"] == "half_start" and h1["evidence"] == "formation_ball"
    assert h1["kick_team_llr"] == 0.0                    # no taker seen: no vote of its own
    assert h1["p_home_restart"] < 0.05                   # S6 from the second half (home)
    # same formation without the ball: no taker, no ball -> not enough evidence
    kos2, _ = _run(s, ball=False)
    h1b = kos2[kos2["half"] == 1].iloc[0]
    assert h1b["evidence"] == "assumed_none_found"


def test_half_that_begins_at_the_kick():
    s = Scene([(0, 3000)])
    s.kickoff(0, 2, 3000, kick_team=0)
    kos, feats = _run(s)
    assert kos["start_kind"].iloc[0] == "half_start"
    assert not kos["evidence"].iloc[0].startswith("assumed")
    assert int(kos["restart_frame"].iloc[0]) <= TOL
    assert feats["mask_pre_kickoff"].sum() <= TOL


def test_live_play_before_first_formation_means_kickoff_was_not_seen():
    """Tracking starts after the half's kick-off (bench game 1): the first formation found is
    a goal kick-off, and the half-start kick-off is assumed at the half start (no mask)."""
    s = Scene([(0, 5000)])
    s.live(0, 2000)
    s.kickoff(2000, 2600, 5000, kick_team=1)
    kos, feats = _run(s)
    assert list(kos["start_kind"]) == ["half_start", "goal"]
    assert kos["evidence"].iloc[0] == "assumed_live_before" and kos["restart_frame"].iloc[0] == 0
    assert not feats["mask_pre_kickoff"].any()


def test_no_kickoff_found_in_second_half_is_assumed_and_not_masked():
    s = Scene([(0, 3000), (3000, 6000)])
    s.kickoff(0, 200, 3000, kick_team=1)
    s.live(3000, 6000)
    kos, feats = _run(s)
    h2 = kos[kos["half"] == 2].iloc[0]
    assert h2["evidence"] == "assumed_none_found" and h2["start_kind"] == "half_start"
    assert h2["p_home_restart"] > 0.9                    # S6: away kicked off the first half
    assert not feats["mask_pre_kickoff"].to_numpy()[3000:].any()


def test_trap_a14_midfielder_on_the_ball_in_live_play():
    """A14 / P23: ball still at the centre with a player on it, but an opponent presses inside
    the circle and players stand in the other half -> not a kick-off."""
    s = Scene([(0, 4000)])
    s.kickoff(0, 100, 1000, kick_team=0)
    home, away = kickoff_shape(0, home_dir=1)
    away[0] = (CX + 4.0, CY + 1.0)                       # presser inside the circle
    home[4:7, 0] = CX + 15.0                             # three home players in the away half
    s.frames(1000, 1500, home, away, np.array([CX, CY]))
    s.live(1500, 4000)
    kos, _ = _run(s)
    assert len(kos) == 1 and int(kos["restart_frame"].iloc[0]) < 200


def test_trap_midfield_free_kick_with_ball_unseen():
    """Bench false-positive pattern: a dead ball near midfield with the ball untracked, a player
    a few metres from the centre spot, everyone in his own half, opponents well back."""
    s = Scene([(0, 4000)])
    s.kickoff(0, 100, 1000, kick_team=0)
    home, away = kickoff_shape(0, home_dir=1, taker_offset=2.8, ring_m=13.0)
    home[1] = (CX - 6.0, CY + 6.0)                       # team-mate not at the ball
    s.frames(1000, 1400, home, away, np.array([np.nan, np.nan]))
    s.live(1400, 4000)
    kos, _ = _run(s)
    assert len(kos) == 1


def test_two_formations_in_one_stoppage_keep_the_later():
    """D4 spacing: formation, a defender steps into the circle for a few seconds, formation
    again, then the kick -> one kick-off, dated at the real kick."""
    s = Scene([(0, 6000)])
    s.kickoff(0, 100, 1500, kick_team=1)
    s.live(1500, 2500)
    home, away = kickoff_shape(0, home_dir=1)
    s.frames(2500, 2800, home, away, np.array([np.nan, np.nan]))
    away_in = away.copy()
    away_in[0] = (CX + 5.0, CY)
    s.frames(2800, 3000, home, away_in, np.array([np.nan, np.nan]))
    s.kickoff(3000, 3300, 6000, kick_team=0)
    kos, _ = _run(s)
    goals = kos[kos["start_kind"] == "goal"]
    assert len(goals) == 1
    assert abs(int(goals["restart_frame"].iloc[0]) - 3300) <= TOL


def test_attack_direction_fallback_s7():
    s = _two_halves()
    with_dir, _ = _run(s, with_direction=True)
    without, _ = _run(s, with_direction=False)
    pd.testing.assert_frame_equal(with_dir, without)


def test_measures_use_team_level_counts_only():
    s = _two_halves()
    m = kickoff_frame_measures(s.players(), s.meta(), s.ball_raw())
    assert len(m) == s.n_frames
    f = 100                                              # inside the first formation
    assert m["n_wrong_half"].iloc[f] == 0
    assert m["n_circle_0"].iloc[f] == 0 and m["n_ring_0"].iloc[f] >= 3
    assert m["d_spot_1"].iloc[f] < 1.0
    assert m["ball_d_spot"].iloc[f] == pytest.approx(0.0, abs=1e-6)


def test_restart_rows_follow_the_contract():
    kos, _ = _run(_two_halves())
    for col in ("start_frame", "restart_frame", "start_kind", "restart_type", "p_home_restart",
                "conf"):
        assert col in kos.columns
    assert kos["p_home_restart"].between(0, 1).all() and kos["conf"].between(0, 1).all()
    assert set(kos["start_kind"]) <= {"half_start", "goal"}


def test_published_anchor_replaces_an_undetected_half_start():
    """Match only (lead decision): no half-start formation detected -> the half's kick-off is
    the published anchor passed by the caller; formations before it are dropped; the mask
    runs from the half start to the anchor; the team stays undetermined (p = 0.5)."""
    s = Scene([(0, 3000), (3000, 8000)])
    s.live(0, 3000)                                      # no formation in half 1
    s.live(3000, 3800)                                   # mixed teams first (guard fires)
    home, away = kickoff_shape(1, home_dir=-1)
    s.frames(3800, 4000, home, away, np.array([np.nan, np.nan]))   # formation before the anchor
    s.live(4000, 8000)                                   # real kick-off not seen as a formation
    anchors = {1: 10, 2: 4500}
    kos, feats = kickoff.run(s.players(), s.meta(), s.ball_raw(), anchors=anchors)
    starts = kos[kos["start_kind"] == "half_start"]
    assert list(starts["evidence"]) == ["published_anchor", "published_anchor"]
    # FAQ convention: the dead span includes the anchor frame; live from the next frame
    assert list(starts["restart_frame"]) == [11, 4501]
    assert (kos[kos["half"] == 2]["restart_frame"] >= 4501).all()   # earlier formation dropped
    mask = feats["mask_pre_kickoff"].to_numpy()
    assert mask[:11].all() and not mask[11:3000].any()
    assert mask[3000:4501].all() and not mask[4501:].any()
    assert (starts["p_home_restart"] == 0.5).all()
    assert (starts["law_kick_team_h1"] == -1).all()


def test_detected_half_start_wins_over_the_anchor():
    """When the formation fires, the detection is kept (the anchor is only the CHECK)."""
    s = _two_halves()
    kos, _ = kickoff.run(s.players(), s.meta(), s.ball_raw(), anchors={1: 900, 2: 5000})
    starts = kos[kos["start_kind"] == "half_start"]
    assert "published_anchor" not in set(starts["evidence"])
    assert abs(int(starts["restart_frame"].iloc[1]) - 4000) <= TOL


def test_anchor_at_the_half_start_means_the_half_starts_at_its_kickoff():
    """S5 / FAQ: a half graded from the kick-off itself starts live at its first graded frame,
    even when the published anchor is a frame later: nothing is masked."""
    s = Scene([(0, 3000)])
    s.live(0, 3000)
    kos, feats = kickoff.run(s.players(), s.meta(), s.ball_raw(), anchors={1: 1})
    assert kos["evidence"].iloc[0] == "published_anchor"
    assert kos["restart_frame"].iloc[0] == 0
    assert not feats["mask_pre_kickoff"].any()
