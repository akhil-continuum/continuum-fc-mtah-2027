"""Ball stage (B1-B3) and events touch/flight/exit (B4, B5, D1-D3) on synthetic scenes.

Each rule has a scene where it should fire and one for its nearest trap. Every scene is also run
with the bench noise model's jitter (ball 0.25 m, players 0.15 m) where the rule should survive
it.
"""
import numpy as np
import pandas as pd
import pytest

from pipeline.ball import clean
from pipeline.scenes import Scene
from pipeline.scenes import library as lib

TOL = 3


def _events(sc, kind, team=None):
    _, ev = sc.run()
    ev = ev[ev["kind"] == kind]
    if team is not None:
        ev = ev[ev["team"] == team]
    return ev


def _near(ev, frame, tol=TOL):
    return ev[(ev["frame"] - frame).abs() <= tol]


# Scene builder -------------------------------------------------------------------------------

def test_scene_writes_contract_files(tmp_path):
    sc = lib.pass_scene()
    sc.write(tmp_path)
    players = pd.read_parquet(tmp_path / "players.parquet")
    ball = pd.read_parquet(tmp_path / "ball_raw.parquet")
    assert list(players.columns) == ["frame", "team", "x_m", "y_m", "vx", "vy"]
    assert list(ball.columns) == ["frame", "x_m", "y_m", "seen"]
    assert len(ball) == sc.n_frames and ball["seen"].all()
    assert players["frame"].is_monotonic_increasing
    # Ball at the receiver's feet after the pass, 0.5 m ahead of him.
    rx, ry = sc.player_pos(1, 100)
    assert np.hypot(ball["x_m"][100] - rx, ball["y_m"][100] - ry) == pytest.approx(0.5, abs=1e-3)


def test_scene_hide_and_expectations():
    sc = lib.flight_gap_scene(8)
    raw = sc.ball_raw_df()
    a, b = sc.gap
    assert not raw["seen"][a:b + 1].any() and raw["seen"][a - 1] and raw["seen"][b + 1]
    assert any(e.kind == "pass" for e in sc.expected)


# Ball stage ----------------------------------------------------------------------------------

def test_b1_single_frame_jump_is_rejected():
    ball, _ = lib.teleport_scene().run(events=False)
    assert not ball["seen"][50] and ball["seen"][49] and ball["seen"][51]


def test_b1_trap_fast_kick_is_kept():
    sc = Scene(60)
    sc.player(0, [(0, 30, 30)])
    sc.ball([(0, 30.5, 30), (20, 30.5, 30), (40, 50.5, 30), (59, 52, 30)])   # 25 m/s shot
    ball, _ = sc.run(events=False)
    assert ball["seen"].all()


def test_b2_still_ball_on_penalty_mark_is_suspect():
    ball, _ = lib.false_marking_scene().run(events=False)
    assert ball["conf"][100] <= clean.CONF_SUSPECT + 1e-6
    assert ball["seen"][100]          # kept, only flagged


def test_b2_trap_penalty_with_taker_is_not_suspect():
    sc = Scene(150)
    sc.player(0, [(0, 9.5, 34)])      # taker next to the ball
    sc.ball([(0, 11.0, 34.0), (149, 11.0, 34.0)])
    ball, _ = sc.run(events=False)
    assert (ball["conf"] == 1.0).all()


@pytest.mark.parametrize("jitter", [False, True])
def test_b3_short_gap_in_straight_flight_is_filled(jitter):
    sc = lib.flight_gap_scene(8)
    sc2 = sc.jittered(seed=3) if jitter else sc
    ball, _ = sc2.run(events=False)
    a, b = sc.gap
    assert ball["filled"][a:b + 1].all() and not ball["seen"][a:b + 1].any()
    assert (ball["conf"][a:b + 1] < 1.0).all()
    true = sc.ball_xy()   # unseen in the scene, so take the path from an unhidden copy
    sc_full = lib.pass_scene()
    err = np.hypot(ball["x_m"][a:b + 1] - sc_full.ball_xy()[a:b + 1, 0],
                   ball["y_m"][a:b + 1] - sc_full.ball_xy()[a:b + 1, 1])
    assert err.max() < (1.0 if jitter else 0.05)
    del true


def test_b3_long_gap_stays_missing():
    sc = lib.flight_gap_scene(20)
    ball, _ = sc.run(events=False)
    a, b = sc.gap
    assert not ball["filled"][a:b + 1].any()
    assert ball["x_m"][a:b + 1].isna().all()


def test_b3_trap_gap_across_a_touch_is_not_filled():
    sc = lib.pass_scene()
    sc.hide_ball(60, 70)               # gap spans the reception at 65: not one straight flight
    ball, _ = sc.run(events=False)
    assert not ball["filled"][60:71].any()


def test_velocity_is_centred_and_accurate():
    sc = Scene(100)
    sc.player(0, [(0, 10, 10)])
    sc.ball([(0, 20, 30), (99, 59.6, 30)])      # 10 m/s along x
    ball, _ = sc.run(events=False)
    assert np.allclose(ball["vx"], 10.0, atol=1e-3) and np.allclose(ball["vy"], 0.0, atol=1e-3)


# Touch (B4) and flight (B5) ------------------------------------------------------------------

@pytest.mark.parametrize("jitter", [False, True])
def test_b4_pass_touches_and_b5_flight(jitter):
    sc = lib.pass_scene()
    sc = sc.jittered(seed=1) if jitter else sc
    t0 = _events(sc, "touch", team=0)
    assert len(_near(t0, 40)) == 1 and len(_near(t0, 65)) == 1
    assert len(_events(sc, "touch", team=1)) == 0
    fl = _events(sc, "flight")
    assert len(fl) == 1 and fl["team"].iloc[0] == 0
    assert 40 <= fl["frame"].iloc[0] <= 45 and 58 <= fl["end_frame"].iloc[0] <= 64


@pytest.mark.parametrize("jitter", [False, True])
def test_b4_deflection_touch_is_the_defenders(jitter):
    sc = lib.deflection_scene()
    sc = sc.jittered(seed=2) if jitter else sc
    assert len(_near(_events(sc, "touch", team=1), 33)) == 1


@pytest.mark.parametrize("jitter", [False, True])
def test_b4_trap_ball_passing_a_still_defender_is_not_his_touch(jitter):
    """P13: the ball rolls past a defender who never touches it."""
    sc = lib.dribble_push_scene()
    sc = sc.jittered(seed=4) if jitter else sc
    assert len(_events(sc, "touch", team=1)) == 0


def test_b4_trap_bounce_without_a_player_is_not_a_touch():
    sc = Scene(100)
    sc.player(0, [(0, 10, 10)])
    sc.player(1, [(0, 90, 60)])
    sc.ball([(0, 40, 30), (50, 60, 30), (99, 70, 40)])   # sharp turn far from everyone
    assert len(_events(sc, "touch")) == 0


@pytest.mark.parametrize("name", ["header_duel", "tackle_recovery", "interception"])
def test_b4_contested_touch_found(name):
    sc = lib.ALL[name]()
    exp = [e for e in sc.expected if e.kind == "touch" and not e.absent][0]
    assert len(_near(_events(sc, "touch", team=exp.team), exp.frame)) == 1


# Exits (D1-D3) -------------------------------------------------------------------------------

@pytest.mark.parametrize("line", ["touch_y0", "touch_y1", "goal_x0", "goal_x1"])
@pytest.mark.parametrize("jitter", [False, True])
def test_d1_d2_exit_over_each_line(line, jitter):
    sc = lib.exit_scene(line)
    exp = sc.expected[0]
    sc = sc.jittered(seed=5) if jitter else sc
    ex = _events(sc, "exit")
    assert len(ex) == 1
    r = ex.iloc[0]
    assert abs(r["frame"] - exp.frame) <= TOL
    assert r["line"] == line
    assert r["last_touch_team"] == exp.extra["last_touch_team"]
    assert bool(r["goal_mouth"]) == exp.extra["goal_mouth"]


@pytest.mark.parametrize("jitter", [False, True])
def test_d3_vanish_heading_out_is_an_extrapolated_exit(jitter):
    sc = lib.vanish_exit_scene()
    exp = sc.expected[0]
    sc = sc.jittered(seed=6) if jitter else sc
    ex = _events(sc, "exit")
    assert len(ex) == 1
    assert ex["detail"].iloc[0] == "extrapolated" and ex["line"].iloc[0] == "touch_y0"
    assert abs(ex["frame"].iloc[0] - exp.frame) <= TOL
    assert ex["last_touch_team"].iloc[0] == 0


@pytest.mark.parametrize("jitter", [False, True])
def test_d2_trap_projection_artefact_is_not_an_exit(jitter):
    sc = lib.projection_trap_scene()
    sc = sc.jittered(seed=7) if jitter else sc
    assert len(_events(sc, "exit")) == 0


@pytest.mark.parametrize("y", [-0.6, -1.5])
def test_d2_trap_player_running_with_the_ball_on_the_line(y):
    """Ball drawn a little outside while a player dribbles along the touchline: not an exit."""
    sc = Scene(150)
    p = sc.player(0, [(0, 40, y), (149, 58, y)])
    sc.player(1, [(0, 45, 6)])
    sc.ball([(0, 39, 3), (10, 40.5, y)])
    sc.ball_with(p, 11, 149, ahead=0.4)
    assert len(_events(sc, "exit")) == 0


def test_d3_trap_ball_lost_in_midfield_is_not_an_exit():
    sc = lib.pass_scene()
    sc.hide_ball(50, 149)
    assert len(_events(sc, "exit")) == 0


def test_no_events_without_the_ball():
    sc = lib.pass_scene()
    sc.hide_ball(0, 149)
    _, ev = sc.run()
    assert len(ev) == 0
