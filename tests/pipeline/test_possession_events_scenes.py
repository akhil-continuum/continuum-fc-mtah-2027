"""carry (P1), pass (P2), deflection (P7), control (P4, P5) on the skill's synthetic scenes.

Scenes (pipeline/scenes/library.py): a clean interception and carry; a toe-poke deflection; a
dribbler pushing the ball ahead past a defender (P13); a header duel then a loose ball (A8); a
tackle and recovery (A9). Each is also run with the bench noise model's jitter.
"""
import pytest

from pipeline.scenes import library as lib

TOL = 3


def _ev(name, jitter):
    sc = lib.ALL[name]()
    if jitter:
        sc = sc.jittered(seed=11)
    _, ev = sc.run()
    return sc, ev


def _kind(ev, kind, team=None):
    out = ev[ev["kind"] == kind]
    return out if team is None else out[out["team"] == team]


@pytest.mark.parametrize("jitter", [False, True])
def test_p1_carry_found(jitter):
    _, ev = _ev("carry", jitter)
    c = _kind(ev, "carry", 0)
    assert len(c) >= 1
    assert (c["end_frame"] - c["frame"]).max() >= 50
    assert len(_kind(ev, "carry", 1)) == 0 and len(_kind(ev, "control", 1)) == 0


@pytest.mark.parametrize("jitter", [False, True])
def test_p2_pass_between_team_mates(jitter):
    _, ev = _ev("pass", jitter)
    p = _kind(ev, "pass", 0)
    assert len(p) == 1
    assert abs(p["frame"].iloc[0] - 40) <= TOL and abs(p["end_frame"].iloc[0] - 65) <= TOL
    assert len(_kind(ev, "control", 1)) == 0


@pytest.mark.parametrize("jitter", [False, True])
def test_p4_interception_then_carry_is_control_dated_at_first_touch(jitter):
    _, ev = _ev("interception", jitter)
    c = _kind(ev, "control", 1)
    assert len(c) == 1
    r = c.iloc[0]
    assert abs(r["first_touch_frame"] - 45) <= TOL and r["frame"] == r["first_touch_frame"]
    assert r["confirmed_frame"] > r["first_touch_frame"]
    assert len(_kind(ev, "carry", 1)) >= 1


@pytest.mark.parametrize("jitter", [False, True])
def test_p7_toe_poke_is_a_deflection_not_control(jitter):
    _, ev = _ev("deflection", jitter)
    d = _kind(ev, "deflection", 1)
    assert len(d) == 1 and abs(d["frame"].iloc[0] - 33) <= TOL
    assert len(_kind(ev, "control", 1)) == 0


@pytest.mark.parametrize("jitter", [False, True])
def test_p13_dribble_push_past_defender_gives_him_nothing(jitter):
    _, ev = _ev("dribble_push", jitter)
    assert len(_kind(ev, "touch", 1)) == 0
    assert len(_kind(ev, "control", 1)) == 0
    assert len(_kind(ev, "pass", 0)) == 0          # pushing the ball ahead is not a pass
    assert len(_kind(ev, "carry", 0)) >= 1


@pytest.mark.parametrize("jitter", [False, True])
def test_a8_header_duel_then_loose_ball_is_not_control(jitter):
    _, ev = _ev("header_duel", jitter)
    assert len(_kind(ev, "touch", 1)) == 1
    assert len(_kind(ev, "control", 1)) == 0


@pytest.mark.parametrize("jitter", [False, True])
def test_a9_tackle_and_recovery_is_not_control(jitter):
    _, ev = _ev("tackle_recovery", jitter)
    assert len(_kind(ev, "touch", 1)) == 1
    assert len(_kind(ev, "control", 1)) == 0


def test_a10_tackle_poke_to_team_mate_is_a_real_short_possession():
    """A tackles and pokes the ball to a team-mate, who receives and loses it at once to an H
    presser. No A carry: only the completed pass (P2) confirms A's short possession."""
    from pipeline.scenes import Scene
    sc = Scene(175)
    c = sc.player(0, [(0, 40, 34), (40, 46.4, 34)])                 # H carrier
    sc.player(1, [(0, 50, 35), (40, 47.0, 34.2)])                   # A tackler
    m = sc.player(1, [(0, 36, 40.2)])                               # A team-mate
    h = sc.player(0, [(0, 30, 44), (64, 35.8, 40.6), (174, 45, 40)])  # H presser
    sc.ball_with(c, 0, 40)
    sc.ball([(40, None), (62, 36.5, 40.0)])                         # poke: 12 m in 0.9 s
    sc.ball_with(h, 66, 174)
    sc.ball([(62, 36.5, 40.0), (66, None)])                         # m's touch, ball to presser
    _, ev = sc.run()
    assert len(_kind(ev, "pass", 1)) == 1
    ctrl = _kind(ev, "control", 1)
    assert len(ctrl) == 1 and abs(ctrl["first_touch_frame"].iloc[0] - 41) <= TOL
    assert ctrl["detail"].iloc[0] == "pass"
    assert len(_kind(ev, "control", 0)) >= 1        # H's original and H's regain
    del m


def test_control_never_crosses_an_exit():
    sc = lib.exit_scene("touch_y0")
    _, ev = sc.run()
    ex = _kind(ev, "exit")["frame"].iloc[0]
    for _, r in _kind(ev, "control").iterrows():
        assert not (r["first_touch_frame"] < ex < r["confirmed_frame"])
