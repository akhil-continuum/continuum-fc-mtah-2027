"""Stoppage candidates and restarts (D1-D3, D5, D9, D12, D13, R1-R4, R8-R12, L1, L2) and the traps
as negative rules (P22 keeper hold, P24 slow build-up, P26 ball moved during a stoppage, B5 short
gap in flight). Scenes use the shared builder pipeline.scenes. Home attacks +x.
"""
import numpy as np
import pytest

from pipeline.restarts import stage
from pipeline.scenes import Scene

L, W = 105.0, 68.0
TOL = 25


def crowd(sc, segments, n=8, seed=0):
    """n players per team; segments [(f0, f1, mode)] with mode 'jog' (3 m/s), 'walk' (1 m/s)
    or 'still'. Players oscillate along x around spread base points (team level only)."""
    rng = np.random.default_rng(seed)
    step = {"jog": 6.0, "walk": 2.0, "still": 0.0}
    for team in (0, 1):
        for i in range(n):
            x = rng.uniform(30, 75)
            y = rng.uniform(12, 56)
            keys = [(0, x, y)]
            sign = 1.0 if i % 2 else -1.0
            for f0, f1, mode in segments:
                f = f0
                while f < f1:
                    nxt = min(f1, f + 50)
                    x = x + sign * step[mode] * (nxt - f) / 50.0
                    sign = -sign
                    keys.append((nxt, x, y))
                    f = nxt
            sc.player(team, keys)


def run_scene(sc):
    ball, events = sc.run()
    inputs = {"players": sc.players_df(), "meta": sc.meta(), "ball_raw": sc.ball_raw_df(),
              "ball": ball, "events": events}
    rows, feats = stage.run(inputs)
    return rows[rows["start_kind"] != "half_start"].reset_index(drop=True), feats


def _kicker(sc, team, keys):
    return sc.player(team, keys)


# ----------------------------------------------------------------------------- sources

def test_throw_in_after_exit_restarts_to_the_other_team():
    sc = Scene(900)
    crowd(sc, [(0, 300, "jog"), (300, 560, "still"), (560, 900, "jog")])
    k = sc.player(0, [(0, 45, 10), (290, 50, 4)])                    # H carrier near y = 0
    t = sc.player(1, [(0, 60, 30), (500, 52, -0.5), (560, 52, -0.5), (600, 55, 6)])  # A thrower
    sc.ball_with(k, 0, 290)
    sc.ball([(290, None), (305, 51, -3.0)])                           # H kicks it out
    sc.hide_ball(312, 559)
    sc.ball([(560, 52, 0.5), (590, 62, 20)])                          # thrown in, travels
    rows, feats = run_scene(sc)
    assert len(rows) == 1
    r = rows.iloc[0]
    assert r["start_kind"] == "exit" and r["restart_type"] == "throw_in"
    assert abs(int(r["restart_frame"]) - 560) <= TOL
    assert r["p_home_restart"] < 0.5                                  # A throws (H touched last)
    assert feats["p_exit"].to_numpy()[int(r["start_frame"])] > 0
    assert feats["p_restart"].to_numpy()[int(r["restart_frame"])] > 0


@pytest.mark.parametrize("last_team,expect_type,expect_home", [(1, "corner", True),
                                                               (0, "goal_kick", False)])
def test_goal_line_exit_type_and_team_r10(last_team, expect_type, expect_home):
    """Over the goal line at x = L (away defends it): last touch by away -> corner to home;
    by home -> goal kick to away."""
    sc = Scene(900)
    crowd(sc, [(0, 300, "jog"), (300, 600, "still"), (600, 900, "jog")])
    k = sc.player(last_team, [(0, 85, 40), (290, 98, 40)])
    sc.ball_with(k, 0, 290)
    sc.ball([(290, None), (305, L + 3.0, 45)])
    sc.hide_ball(312, 599)
    if expect_type == "corner":
        sc.player(0, [(0, 70, 30), (590, L + 0.5, W - 0.5)])
        sc.ball([(600, L - 0.5, W - 0.5), (625, L - 12, 38)])
    else:
        sc.player(1, [(0, 70, 30), (590, L - 4, 34)])
        sc.ball([(600, L - 4, 34), (630, L - 30, 34)])
    rows, _ = run_scene(sc)
    assert len(rows) == 1
    r = rows.iloc[0]
    assert r["restart_type"] == expect_type
    assert (r["p_home_restart"] > 0.5) == expect_home
    assert abs(int(r["restart_frame"]) - 600) <= TOL


def test_foul_with_ball_gap_is_a_gap_stoppage_and_free_kick():
    sc = Scene(1000)
    crowd(sc, [(0, 400, "jog"), (400, 700, "still"), (700, 1000, "jog")])
    k = sc.player(0, [(0, 40, 30), (400, 55, 30)])
    sc.player(1, [(0, 70, 34), (400, 56, 31)])
    sc.ball_with(k, 0, 400)
    sc.hide_ball(401, 699)
    sc.ball([(700, 55, 31), (730, 70, 25)])
    rows, _ = run_scene(sc)
    assert len(rows) == 1
    r = rows.iloc[0]
    assert r["start_kind"] in ("gap", "slowdown") and r["restart_type"] == "free_kick"
    assert abs(int(r["start_frame"]) - 400) <= 50
    assert abs(int(r["restart_frame"]) - 700) <= TOL


def test_whistle_slowdown_with_ball_seen_still():
    """D5 without a gap: everyone stops, the ball lies still at the spot, then is kicked."""
    sc = Scene(1100)
    crowd(sc, [(0, 400, "jog"), (400, 750, "still"), (750, 1100, "jog")])
    k = sc.player(0, [(0, 40, 30), (400, 50, 30)])
    sc.ball_with(k, 0, 400)
    sc.ball([(400, None), (420, 51, 30), (749, 51, 30)])              # rolls 1 m and stops
    sc.ball([(750, 51, 30), (780, 70, 22)])
    rows, feats = run_scene(sc)
    assert len(rows) == 1
    r = rows.iloc[0]
    assert r["start_kind"] == "slowdown"
    assert abs(int(r["restart_frame"]) - 750) <= TOL
    assert abs(int(r["start_frame"]) - 400) <= 75
    assert feats["p_slowdown"].max() > 0


# ----------------------------------------------------------------------------- traps

def test_trap_p22_keeper_holds_the_ball():
    """The away keeper catches the ball in his box and holds it 6 s while play stays live
    (opponents jog away), then kicks long: no stoppage."""
    sc = Scene(900)
    crowd(sc, [(0, 900, "jog")])
    gk = sc.player(1, [(0, L - 3, 34)])
    k = sc.player(0, [(0, 70, 30), (300, 85, 32)])
    sc.ball_with(k, 0, 300)
    sc.ball([(300, None), (320, L - 3.4, 34)])
    sc.ball_with(gk, 320, 470, ahead=0.3)
    sc.hide_ball(330, 470)                                            # in his hands, unseen
    sc.ball([(471, L - 3.5, 34), (510, 60, 34)])
    rows, feats = run_scene(sc)
    assert len(rows) == 0
    assert feats["keeper_hold"].to_numpy()[320:330].any()


def test_trap_p24_slow_build_up_is_not_a_stoppage():
    """Everyone walking for 24 s while two home centre-backs pass sideways: no drop, no gap."""
    sc = Scene(800)
    crowd(sc, [(0, 800, "walk")])
    a = sc.player(0, [(0, 25, 20)])
    b = sc.player(0, [(0, 25, 48)])
    f = 0
    while f + 100 <= 800:
        sc.ball_with(a, f, f + 40)
        sc.ball([(f + 40, None), (f + 50, 25.5, 48)])
        sc.ball_with(b, f + 50, min(799, f + 90))
        if f + 100 < 800:
            sc.ball([(f + 90, None), (f + 100, 25.5, 20)])
        f += 100
    rows, _ = run_scene(sc)
    assert len(rows) == 0


def test_trap_p26_ball_moved_during_a_stoppage_does_not_restart():
    """D9 / P26: during a foul stoppage the ball is rolled 12 m to the spot while every player
    stands still; the restart is the later kick."""
    sc = Scene(1200)
    crowd(sc, [(0, 400, "jog"), (400, 850, "still"), (850, 1200, "jog")])
    k = sc.player(0, [(0, 40, 30), (400, 50, 30)])
    sc.ball_with(k, 0, 400)
    sc.hide_ball(401, 549)
    sc.ball([(550, 40, 30), (625, 52, 30), (849, 52, 30)])           # rolled 12 m in 3 s
    sc.ball([(850, 52, 30), (880, 70, 22)])
    rows, _ = run_scene(sc)
    assert len(rows) == 1
    assert abs(int(rows["restart_frame"].iloc[0]) - 850) <= TOL


@pytest.mark.parametrize("gap", [30, 60])
def test_trap_b5_short_gap_in_flight_during_live_play(gap):
    """A pass with the ball unseen in flight while everyone keeps running: no stoppage."""
    sc = Scene(600)
    crowd(sc, [(0, 600, "jog")])
    a = sc.player(0, [(0, 30, 30), (200, 40, 30)])
    b = sc.player(0, [(0, 75, 40)])
    sc.ball_with(a, 0, 200)
    sc.ball([(200, None), (200 + gap + 10, 74.5, 40)])
    sc.hide_ball(203, 203 + gap)
    sc.ball_with(b, 200 + gap + 10, 599)
    rows, _ = run_scene(sc)
    assert len(rows) == 0


def test_features_dead_columns_present():
    sc = Scene(900)
    crowd(sc, [(0, 300, "jog"), (300, 560, "still"), (560, 900, "jog")])
    k = sc.player(0, [(0, 45, 10), (290, 50, 4)])
    sc.ball_with(k, 0, 290)
    sc.ball([(290, None), (305, 51, -3.0)])
    sc.hide_ball(312, 559)
    sc.ball([(560, 52, 0.5), (590, 62, 20)])
    _, feats = run_scene(sc)
    for col in ("frame", "half", "form_kick_off", "mask_pre_kickoff", "p_restart_kick_off",
                "p_exit", "p_slowdown", "p_gap_stop", "p_restart", "p_dead_candidate",
                "keeper_hold", "form_throw_in", "form_corner", "form_goal_kick", "form_free_kick"):
        assert col in feats.columns, col
    assert len(feats) == 900
    assert feats["p_dead_candidate"].to_numpy()[400] > 0
