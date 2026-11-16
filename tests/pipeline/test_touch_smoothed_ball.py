"""B4 on a low-pass ball (T-34, P-17b): the "w15s" touch parameter set.

The competition match's ball track is smoothed (T-31): a kick is spread over about a second. These
scenes take a scripted ball and apply the same centred Gaussian (sigma 7.5 frames, the v3 bench's
match level) inside each seen run, then run the events with the "w15s" set. Each rule has a scene
where it should fire and its nearest trap. The default set must stay "w7"; the sharp scenes in
test_ball_events_scenes.py and test_possession_events_scenes.py cover it.

Resolution limit: on a ball smoothed over a second, two contacts closer than about 15 frames merge
into one change (the library's toe-poke comes 13 frames after the kick, its tackle is a contact
shared by two players). The smoothed scenes therefore space contacts about a second apart;
tackle_recovery is not used here because its contact cannot be attributed on a smeared ball.
"""
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

from pipeline.ball.clean import runs
from pipeline.events import detect, touch
from pipeline.scenes import Scene
from pipeline.scenes import library as lib

SIGMA = 7.5
TOL = 6          # a smeared change peaks a few frames from the scripted contact


def smoothed(sc, sigma=SIGMA, jitter_seed=None):
    """Copy of ``sc`` with its ball low-passed inside each seen run (optionally jittered first)."""
    out = sc.jittered(seed=jitter_seed) if jitter_seed is not None else \
        sc.jittered(ball_sigma=0.0, player_sigma=0.0)
    b = out._ball.copy()
    ok = ~out._hidden & np.isfinite(b[:, 0])
    for a, z in zip(*runs(ok)):
        for c in (0, 1):
            b[a:z, c] = gaussian_filter1d(b[a:z, c], sigma, mode="nearest")
    out._ball = b
    return out


def events(sc, touch_set="w15s"):
    ball, _ = sc.run(events=False)
    return detect(ball, sc.players_df(), sc.meta(), touch_set=touch_set)


def _kind(ev, kind, team=None):
    ev = ev[ev["kind"] == kind]
    return ev if team is None else ev[ev["team"] == team]


def _near(ev, frame, tol=TOL):
    return ev[(ev["frame"] - frame).abs() <= tol]


def test_default_set_is_the_legacy_one():
    assert touch.DEFAULT_SET == "w7"
    p = touch.touch_params()
    assert (p["w"], p["min_pts"], p["noise_k"], p["pwin"]) == (touch.W, touch.MIN_PTS,
                                                               touch.NOISE_K, 1)
    with pytest.raises(ValueError):
        touch.touch_params("nope")


def test_smoothing_spreads_the_kick():
    """The scene really is smeared: the ball's acceleration at the kick is a fraction of the sharp one."""
    sc = lib.pass_scene()
    acc = lambda s: np.abs(np.diff(s.ball_xy()[:, 0], 2)).max()   # noqa: E731
    assert acc(smoothed(sc)) < 0.2 * acc(sc)


@pytest.mark.parametrize("seed", [None, 1, 5])
def test_b4_smoothed_pass_touches_found(seed):
    ev = events(smoothed(lib.pass_scene(), jitter_seed=seed))
    t0 = _kind(ev, "touch", 0)
    assert len(_near(t0, 40)) == 1 and len(_near(t0, 65)) == 1
    assert len(_kind(ev, "touch", 1)) == 0
    p = _kind(ev, "pass", 0)
    assert len(p) == 1


def deflection_long_scene():
    """H passes, an A defender clips it 1.2 s later (heading turns ~20 degrees, ball keeps 11 m/s),
    an H team-mate receives 1.2 s after that."""
    sc = Scene(175)
    p = sc.player(0, [(0, 28, 34), (20, 29.6, 34)])
    sc.player(1, [(0, 47.5, 35.0)])                     # defender on the line of the pass
    r = sc.player(0, [(0, 60.5, 41.5), (80, 60.5, 41.5), (174, 66, 41.5)])
    sc.player(1, [(0, 85, 15)])
    sc.player(0, [(0, 20, 20)])
    sc.ball_with(p, 0, 20)
    sc.ball([(20, None), (50, 47.0, 34.0)])            # 17 m in 1.2 s
    sc.ball([(50, 47.0, 34.0), (80, 60.0, 41.0)])      # on at ~12 m/s, turned ~28 degrees
    sc.ball_with(r, 81, 174)
    return sc


@pytest.mark.parametrize("seed", [None, 2])
def test_b4_smoothed_deflection_touch_is_the_defenders(seed):
    ev = events(smoothed(deflection_long_scene(), jitter_seed=seed))
    assert len(_near(_kind(ev, "touch", 1), 50)) == 1
    assert len(_kind(ev, "control", 1)) == 0
    assert len(_near(_kind(ev, "deflection", 1), 50)) == 1


@pytest.mark.parametrize("name", ["interception", "header_duel"])
def test_b4_smoothed_contested_touch_found(name):
    sc = lib.ALL[name]()
    exp = [e for e in sc.expected if e.kind == "touch" and not e.absent][0]
    ev = events(smoothed(sc))
    assert len(_near(_kind(ev, "touch", exp.team), exp.frame)) == 1


@pytest.mark.parametrize("seed", [None, 4])
def test_b4_trap_smoothed_ball_rolling_past_a_still_defender(seed):
    """P13: the wider player window must not hand the dribbler's push to the defender it passes."""
    ev = events(smoothed(lib.dribble_push_scene(), jitter_seed=seed))
    assert len(_kind(ev, "touch", 1)) == 0
    assert len(_kind(ev, "control", 1)) == 0


def test_b4_trap_smoothed_bounce_without_a_player():
    sc = Scene(150)
    sc.player(0, [(0, 10, 10)])
    sc.player(1, [(0, 90, 60)])
    sc.ball([(0, 40, 30), (75, 60, 30), (149, 70, 40)])    # sharp turn far from everyone
    assert len(_kind(events(smoothed(sc)), "touch")) == 0


def test_b4_trap_smoothed_turn_near_a_player_who_is_beyond_the_window():
    """A player reaches the ball's path only 1 s after the turn: not his touch."""
    sc = Scene(150)
    sc.player(0, [(0, 10, 10)])
    sc.player(1, [(0, 75, 45), (100, 75, 45), (125, 66, 36)])
    sc.ball([(0, 40, 30), (75, 60, 30), (149, 70, 40)])
    ev = events(smoothed(sc))
    assert len(_near(_kind(ev, "touch"), 75, tol=15)) == 0
