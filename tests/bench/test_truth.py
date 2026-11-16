"""Truth derivation on small synthetic event sequences (bench/truth.py, bench/events.py)."""

import numpy as np
import pandas as pd
import pytest

from bench.events import confirm_stoppages, event_stoppages, is_control
from bench.metrica_io import Game
from bench.truth import derive_labels, label_stats


def ev(team, typ, sub, f0, f1=None, period=1):
    return {"team": team, "type": typ, "subtype": sub, "period": period,
            "start_frame": f0, "end_frame": f0 if f1 is None else f1}


def make_game(events, n=1000, period_split=None, missing=()):
    period = np.ones(n, int)
    if period_split:
        period[period_split:] = 2
    ball = np.zeros((n, 2))
    for a, b in missing:
        ball[a:b] = np.nan
    e = pd.DataFrame(events)
    e["order"] = np.arange(len(e))
    return Game("synthetic", period, ball, np.zeros((n, 2, 2)), np.array([0, 1], np.int8),
                ["h", "a"], e)


def labels_of(game, convention="first_touch"):
    return derive_labels(game, convention)[0]


def test_restart_law_after_ball_out():
    g = make_game([
        ev("H", "SET PIECE", "KICK OFF", 0), ev("H", "PASS", "", 0, 50),
        ev("H", "PASS", "", 100, 150), ev("H", "BALL OUT", "", 150, 200),
        ev("A", "SET PIECE", "THROW IN", 400), ev("A", "PASS", "", 400, 430),
        ev("A", "PASS", "", 500, 990),
    ], missing=[(201, 400)])
    lab = labels_of(g)
    assert (lab[:201] == "H").all()
    assert (lab[201:400] == "D").all()
    assert lab[400] == "A"              # R0: first live frame belongs to the restart team
    assert (lab[400:990] == "A").all()


def test_header_and_won_aerial_do_not_switch():
    g = make_game([
        ev("H", "SET PIECE", "KICK OFF", 0), ev("H", "PASS", "", 0, 100),
        ev("A", "CHALLENGE", "AERIAL-WON", 100), ev("A", "BALL LOST", "HEAD", 100, 140),
        ev("H", "RECOVERY", "", 140), ev("H", "PASS", "", 140, 990),
    ])
    lab = labels_of(g)
    assert set(lab[:990]) == {"H"}      # P9: a won header without control is not possession


def test_tackle_double_record_keeps_winner():
    g = make_game([
        ev("H", "SET PIECE", "KICK OFF", 0), ev("H", "PASS", "", 0, 100),
        ev("A", "CHALLENGE", "TACKLE-WON", 200), ev("A", "RECOVERY", "THEFT", 200),
        ev("A", "PASS", "", 200, 240),
        ev("H", "CHALLENGE", "TACKLE-LOST", 203), ev("H", "BALL LOST", "THEFT", 203),
        ev("A", "PASS", "", 240, 990),
    ])
    lab = labels_of(g)
    assert (lab[:200] == "H").all()
    assert (lab[200:990] == "A").all()  # the loser's THEFT record is not a new control


def test_instant_ball_lost_record_is_not_control():
    assert not is_control(ev("H", "BALL LOST", "INTERCEPTION", 10, 10))
    assert is_control(ev("H", "BALL LOST", "INTERCEPTION", 10, 40))
    assert not is_control(ev("H", "BALL LOST", "HEAD", 10, 40))


def test_switch_dating_conventions():
    g = make_game([
        ev("H", "SET PIECE", "KICK OFF", 0), ev("H", "PASS", "", 0, 90),
        ev("A", "RECOVERY", "INTERCEPTION", 100), ev("A", "PASS", "", 130, 990),
    ])
    ft, cf = labels_of(g, "first_touch"), labels_of(g, "confirmed")
    assert ft[99] == "H" and ft[100] == "A"      # dated at the first touch
    assert cf[129] == "H" and cf[130] == "A"     # dated at the first controlled action


def test_recovery_without_following_action_does_not_switch():
    g = make_game([
        ev("H", "SET PIECE", "KICK OFF", 0), ev("H", "PASS", "", 0, 90),
        ev("A", "RECOVERY", "INTERCEPTION", 100),
        ev("H", "RECOVERY", "", 110), ev("H", "PASS", "", 110, 990),
    ])
    assert set(labels_of(g)[:990]) == {"H"}


def test_clearance_out_gives_no_possession():
    g = make_game([
        ev("H", "SET PIECE", "KICK OFF", 0), ev("H", "PASS", "", 0, 200),
        ev("A", "RECOVERY", "", 200), ev("A", "BALL OUT", "CLEARANCE", 200, 220),
        ev("A", "CARRY", "", 200, 201),          # game 3 logs a 1-frame carry like this
        ev("H", "SET PIECE", "THROW IN", 400), ev("H", "PASS", "", 400, 990),
    ], missing=[(221, 400)])
    lab = labels_of(g)
    assert "A" not in set(lab)                   # P10: clearing team never had it
    assert (lab[221:400] == "D").all() and lab[400] == "H"


def test_unlogged_restart_needs_missing_ball_and_two_seconds():
    base = [ev("H", "SET PIECE", "KICK OFF", 0), ev("H", "PASS", "", 0, 100),
            ev("H", "BALL OUT", "", 100, 150), ev("A", "RECOVERY", "", 400),
            ev("A", "PASS", "", 400, 990)]
    stops = event_stoppages(pd.DataFrame(base).assign(order=range(5)), [(0, 1000)])
    unl = stops[stops.restart_type == "UNLOGGED"]
    assert len(unl) == 1 and unl.iloc[0].start == 151 and unl.iloc[0].restart == 400
    assert unl.iloc[0].restart_team == "A"
    nan = np.zeros(1000, bool)
    assert len(confirm_stoppages(stops, nan).query("restart_type == 'UNLOGGED'")) == 0
    nan[151:400] = True
    assert len(confirm_stoppages(stops, nan).query("restart_type == 'UNLOGGED'")) == 1


def test_advantage_does_not_stop_play():
    g = make_game([
        ev("H", "SET PIECE", "KICK OFF", 0), ev("H", "PASS", "", 0, 100),
        ev("A", "CHALLENGE", "TACKLE-ADVANTAGE-LOST", 120), ev("H", "PASS", "", 130, 990),
    ])
    assert "D" not in set(labels_of(g)[:990])


def test_foul_stoppage_and_free_kick_team():
    g = make_game([
        ev("H", "SET PIECE", "KICK OFF", 0), ev("H", "PASS", "", 0, 100),
        ev("A", "CHALLENGE", "GROUND-FAULT-LOST", 300), ev("H", "CHALLENGE", "GROUND-FAULT-WON", 302),
        ev("H", "FAULT RECEIVED", "", 305),
        ev("H", "SET PIECE", "FREE KICK", 600), ev("H", "PASS", "", 600, 990),
    ], missing=[(306, 600)])
    lab = labels_of(g)
    assert (lab[306:600] == "D").all() and lab[305] == "H" and lab[600] == "H"


def test_halves_separated_and_second_half_d_until_kickoff():
    g = make_game([
        ev("H", "SET PIECE", "KICK OFF", 10), ev("H", "PASS", "", 10, 490),
        ev("A", "SET PIECE", "KICK OFF", 560, period=2), ev("A", "PASS", "", 560, 990, period=2),
    ], period_split=500)
    lab = labels_of(g)
    assert (lab[:10] == "X").all()               # first half graded from its kick-off (S5)
    assert lab[499] == "X"                       # at least one X frame between halves (S3)
    assert (lab[500:560] == "D").all()           # second half D until its kick-off (S4)
    assert lab[560] == "A"


def test_label_stats_counts():
    lab = np.array(list("HHHHDDDDAAAAAAXXHH"))
    st = label_stats(lab, fps=1)
    assert st["n_possessions"] == 3 and st["n_stoppages"] == 1
    assert st["share_poss_under_4s"] == pytest.approx(1 / 3, abs=1e-3)
