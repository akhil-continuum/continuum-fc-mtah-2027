"""Label-convention truth (bench/truth_labels.py) on scripted scenes, one per convention C1-C8."""

import numpy as np
import pandas as pd

from bench.metrica_io import Game
from bench.truth import derive_labels
from bench.truth_labels import (T_HORIZON_S, aerial_floor, classify, derive_labels_conv)

N = 1000
FAR = 40.0


def ev(team, typ, sub, f0, f1=None, period=1):
    return {"team": team, "type": typ, "subtype": sub, "period": period,
            "start_frame": f0, "end_frame": f0 if f1 is None else f1}


def make_game(events, players=None, ball=None, missing=()):
    """Players 0,1 = H, 2,3 = A. Default: ball still at the centre, everyone FAR away."""
    if ball is None:
        ball = np.zeros((N, 2))
    if players is None:
        players = np.zeros((N, 4, 2))
        players[:, :, 0] = [-FAR, -FAR - 5, FAR, FAR + 5]
    ball = ball.copy()
    for a, b in missing:
        ball[a:b] = np.nan
    e = pd.DataFrame(events)
    e["order"] = np.arange(len(e))
    return Game("synthetic", np.ones(N, int), ball, players, np.array([0, 0, 1, 1], np.int8),
                ["h1", "h2", "a1", "a2"], e)


def lab(game):
    return derive_labels_conv(game)[0]


def contested_scene():
    """One H and one A player both standing on the ball: every loose ball is contested."""
    players = np.zeros((N, 4, 2))
    players[:, :, 0] = [0.3, -FAR, -0.3, FAR]
    return players


KO = [ev("H", "SET PIECE", "KICK OFF", 0), ev("H", "PASS", "", 0, 50)]


# -- C1: deliberate plays give possession, however brief ------------------------------------

def test_c1_header_won_counts_even_when_lost_at_once():
    g = make_game(KO + [
        ev("H", "PASS", "", 60, 100),
        ev("A", "CHALLENGE", "AERIAL-WON", 100), ev("A", "BALL LOST", "HEAD", 100, 110),
        ev("H", "CHALLENGE", "AERIAL-LOST", 100),
        ev("H", "RECOVERY", "", 110), ev("H", "PASS", "", 110, 990),
    ], players=contested_scene())
    lb = lab(g)
    old = derive_labels(g, "first_touch")[0]
    assert "A" not in set(old[:990])                 # the old convention ignored it
    assert lb[100] == "A" and lb[110] == "H"
    assert lb[99] == "A" or lb[95] == "H"            # switch shortly before the header only


def test_c1_blocked_clearance_counts_for_clearing_team():
    g = make_game(KO + [
        ev("H", "PASS", "", 60, 100), ev("A", "RECOVERY", "INTERCEPTION", 100),
        ev("A", "BALL LOST", "CLEARANCE", 105, 112),
        ev("H", "RECOVERY", "BLOCKED", 112), ev("H", "PASS", "", 112, 990),
    ], players=contested_scene())
    lb = lab(g)
    assert (lb[100:112] == "A").all() and lb[112] == "H"


def test_c1_dislodging_challenge_alone_does_not_switch():
    g = make_game(KO + [
        ev("H", "PASS", "", 60, 100),
        ev("A", "CHALLENGE", "TACKLE-WON", 200), ev("H", "CHALLENGE", "TACKLE-LOST", 200),
        ev("H", "RECOVERY", "", 215), ev("H", "PASS", "", 215, 990),
    ], players=contested_scene())
    assert set(lab(g)[:990]) == {"H"}


def test_c1_loser_record_after_interception_is_not_a_play():
    g = make_game(KO + [
        ev("H", "PASS", "", 60, 100), ev("A", "RECOVERY", "INTERCEPTION", 100),
        ev("A", "PASS", "", 100, 150), ev("H", "BALL LOST", "INTERCEPTION", 101, 140),
        ev("A", "PASS", "", 150, 990),
    ], players=contested_scene())
    lb = lab(g)
    assert lb[99] == "H" and (lb[100:990] == "A").all()


def test_event_map_examples():
    r = ev("A", "BALL LOST", "THEFT", 10, 10)
    assert classify(r, [r])[0] is None
    r = ev("A", "CHALLENGE", "GROUND-WON", 10)
    assert classify(r, [r])[:2] == ("anchor", "contact")
    r = ev("A", "CHALLENGE", "AERIAL-WON", 10)
    assert classify(r, [r])[:2] == ("claim", "aerial")
    r = ev("A", "CHALLENGE", "TACKLE-FAULT-LOST", 10)
    assert classify(r, [r]) == ("claim", "foul", "H")
    rec, out = ev("H", "RECOVERY", "SAVED", 10), ev("H", "BALL OUT", "", 10, 20)
    assert classify(rec, [rec, out])[0] == "touch"     # parried straight out: no possession


# -- C3: interceptions and tackles are dated at the contact ---------------------------------

def test_c3_tackle_towards_teammate_switches_at_contact():
    g = make_game(KO + [
        ev("H", "PASS", "", 60, 100),
        ev("A", "CHALLENGE", "TACKLE-WON", 200), ev("H", "CHALLENGE", "TACKLE-LOST", 200),
        ev("A", "RECOVERY", "", 230), ev("A", "PASS", "", 230, 990),
    ], players=contested_scene())
    lb = lab(g)
    assert lb[199] == "H" and (lb[200:990] == "A").all()


def test_c3_interception_dated_at_contact_even_when_clear_earlier():
    # The ball rolls straight to a lone A player: physically clear long before, but a logged
    # interception is dated at the contact.
    ball = np.zeros((N, 2))
    ball[100:150, 0] = np.linspace(0, 20, 50)
    ball[150:, 0] = 20
    players = np.zeros((N, 4, 2))
    players[:, :, 0] = [-30, -35, 20, 45]
    g = make_game(KO + [
        ev("H", "BALL LOST", "INTERCEPTION", 100, 150), ev("A", "RECOVERY", "INTERCEPTION", 150),
        ev("A", "PASS", "", 150, 990),
    ], players=players, ball=ball)
    lb = lab(g)
    assert lb[149] == "H" and lb[150] == "A"


# -- C2, C4: loose balls switch when the receiver is clearly going to get there first ---------

def loose_ball_scene(opponent_x):
    ball = np.zeros((N, 2))
    ball[100:150, 0] = np.linspace(0, 20, 50)        # rolls 20 m in 2 s after H's loss
    ball[150:, 0] = 20
    players = np.zeros((N, 4, 2))
    players[:, :, 0] = [opponent_x, -35, 20, 45]     # H player 0 chases, A player 2 waits at 20 m
    return make_game(KO + [
        ev("H", "BALL LOST", "", 100, 150), ev("A", "RECOVERY", "", 150),
        ev("A", "PASS", "", 150, 990),
    ], players=players, ball=ball)


def test_c2_clear_loose_ball_switches_before_first_touch():
    lb = lab(loose_ball_scene(opponent_x=-30.0))
    first = int(np.flatnonzero(lb[:990] == "A")[0])
    assert first < 150                               # before the receiver's touch
    assert 150 - first <= T_HORIZON_S * 25           # never earlier than the horizon
    assert first > 100                               # never before the old owner's last touch


def test_c2_contested_loose_ball_switches_at_touch():
    lb = lab(loose_ball_scene(opponent_x=19.0))        # an H player stands next to the receiver
    assert lb[149] == "H" and lb[150] == "A"


# -- C5: a long ball in the air stays with the kicker until just before the header -----------

def test_c5_aerial_floor_from_ball_flight():
    assert aerial_floor(0, 20) == 0                   # 0.8 s flight never rises above reach
    fl = aerial_floor(0, 75)                          # 3 s flight
    assert 70 <= fl < 75


def test_c5_long_ball_switch_only_just_before_header():
    players = np.zeros((N, 4, 2))
    players[:, :, 0] = [-30, -35, 0.5, 45]            # an A defender alone under the ball
    g = make_game(KO + [
        ev("H", "PASS", "LONG", 100, 175),
        ev("A", "CHALLENGE", "AERIAL-WON", 175), ev("A", "PASS", "HEAD", 175, 990),
    ], players=players)
    lb = lab(g)
    first = int(np.flatnonzero(lb[:990] == "A")[0])
    assert first == aerial_floor(100, 175) and 170 <= first < 175


# -- C6: a keeper gathering in open play is a live switch ------------------------------------

def test_c6_keeper_save_is_live_switch():
    g = make_game(KO + [
        ev("H", "SHOT", "ON TARGET-SAVED", 100, 120), ev("A", "RECOVERY", "SAVED", 120),
        ev("A", "PASS", "", 160, 990),
    ], players=contested_scene())
    lb = lab(g)
    assert "D" not in set(lb[:990]) and lb[120] == "A" and lb[119] == "H"


# -- C7, C8: stoppages unchanged, fouled team keeps the ball until a delayed whistle ----------

def test_c7_delayed_whistle_fouled_team_keeps_until_stop():
    # H has the ball; an H player fouls an A player at 300 (e.g. a challenge for a loose ball);
    # the whistle (FAULT RECEIVED, closing the event stoppage cluster) comes 1.6 s later.
    g = make_game(KO + [
        ev("H", "PASS", "", 100, 140),
        ev("H", "CHALLENGE", "TACKLE-FAULT-LOST", 300),
        ev("A", "FAULT RECEIVED", "", 340),
        ev("A", "SET PIECE", "FREE KICK", 600), ev("A", "PASS", "", 600, 990),
    ], players=contested_scene(), missing=[(341, 600)])
    lb, old = lab(g), derive_labels(g, "first_touch")[0]
    assert (old[300:341] == "H").all()               # old convention: the fouler's team
    assert lb[299] == "H" and (lb[300:341] == "A").all()
    assert (lb[341:600] == "D").all() and lb[600] == "A"


def test_c7_immediate_whistle_does_not_switch():
    g = make_game(KO + [
        ev("H", "PASS", "", 60, 100),
        ev("A", "CHALLENGE", "AERIAL-FAULT-WON", 300), ev("H", "CHALLENGE", "AERIAL-FAULT-LOST", 300),
        ev("A", "SET PIECE", "FREE KICK", 600), ev("A", "PASS", "", 600, 990),
    ], players=contested_scene(), missing=[(301, 600)])
    lb = lab(g)
    assert (lb[:301] == "H").all() and (lb[301:600] == "D").all() and lb[600] == "A"


def test_c8_dead_spans_and_restarts_identical_to_first_touch():
    g = make_game(KO + [
        ev("H", "PASS", "", 100, 150), ev("H", "BALL OUT", "", 150, 200),
        ev("A", "SET PIECE", "THROW IN", 400), ev("A", "PASS", "", 400, 430),
        ev("H", "RECOVERY", "INTERCEPTION", 500), ev("H", "BALL LOST", "HEAD", 500, 520),
        ev("A", "RECOVERY", "", 520), ev("A", "PASS", "", 520, 990),
    ], players=contested_scene(), missing=[(201, 400)])
    new, old = lab(g), derive_labels(g, "first_touch")[0]
    assert ((new == "D") == (old == "D")).all() and ((new == "X") == (old == "X")).all()
    assert new[400] == "A" and new[500] == "H" and new[520] == "A"
