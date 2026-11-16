"""Features stage (FEATURES.md agrees with the code; D12 missing context), the per-game
ball-player alignment search, and survival of frames with few or no tracked players."""
import os
import re

import numpy as np
import pandas as pd

from pipeline import features
from pipeline.events import base
from pipeline.scenes import Scene
from pipeline.scenes import library as lib

FEATURES_MD = os.path.join(os.path.dirname(features.__file__), "FEATURES.md")


def _build(sc):
    ball, ev = sc.run()
    return features.build(ball, sc.players_df(), ev, sc.meta()), ball, ev


def _documented():
    text = open(FEATURES_MD).read()
    names = set()
    for cell in re.findall(r"^\| `([^|]+)` *\|", text, flags=re.M):
        names.update(c.strip("` ") for c in cell.split(","))
    return text, names


def test_every_feature_column_is_documented():
    df, _, _ = _build(lib.pass_scene())
    text, names = _documented()
    for col in df.columns:
        if col == "frame":
            continue
        assert col in names or "`{}`".format(col) in text or _pattern_documented(col, text), col


def _pattern_documented(col, text):
    """Columns documented with a team or window wildcard, e.g. near_n1_* or near_d_*_c05."""
    for pat in re.findall(r"`([a-z0-9_*]+)`", text):
        if "*" in pat and re.fullmatch(pat.replace("*", "[a-z0-9]+"), col):
            return True
    if col.endswith("_a") and (col[:-2] + "_h") in text:
        return True
    if col.endswith("_a_s") and (col[:-4] + "_h_s") in text:
        return True
    return False


def test_d12_missing_context_in_a_gap():
    sc = lib.pass_scene()
    sc.hide_ball(80, 129)                       # 50 frames lost while the receiver carries
    df, ball, _ = _build(sc)
    assert df["ball_gap_len_s"][100] == 2.0
    assert abs(df["ball_t_since_seen_s"][100] - 21 / 25) < 1e-6
    assert abs(df["ball_t_until_seen_s"][100] - 30 / 25) < 1e-6
    assert abs(df["ball_last_seen_x"][100] - ball["x_m"][79]) < 1e-4
    assert abs(df["ball_next_seen_x"][100] - ball["x_m"][130]) < 1e-4
    assert df["ball_t_since_seen_s"][60] == 0 and df["ball_gap_len_s"][60] == 0
    assert np.isnan(df["near_d_h"][100])        # no contact evidence without the ball


def test_control_timing_has_both_directions():
    df, _, ev = _build(lib.interception_scene())
    f = int(ev.loc[(ev["kind"] == "control") & (ev["team"] == 1), "first_touch_frame"].iloc[0])
    assert df["t_until_control_a_s"][f - 10] == 10 / 25
    assert df["t_since_control_a_s"][f + 10] == 10 / 25


def _long_scene(lag):
    """2000 frames of a player dribbling a loop with the ball; players recorded ``lag`` frames
    late (ball ahead of players)."""
    n = 2000
    sc = Scene(n)
    t = np.arange(n)
    keys = [(int(f), 52 + 25 * np.cos(f / 80.0), 34 + 20 * np.sin(f / 80.0)) for f in t[::5]]
    p = sc.player(0, keys)
    sc.player(1, [(0, 52, 34)])
    # Ball beside the runner, not ahead: a ball ahead of a moving carrier is itself worth a
    # frame or two of apparent lag (speed x offset), which the search would rightly pick up.
    sc.ball_with(p, 0, n - 1, ahead=0.0, side=0.4)
    if lag:
        path = sc._players[p][1]
        late = np.full_like(path, np.nan)
        late[lag:] = path[:-lag]
        late[:lag] = path[0]
        sc._players[p] = (0, late)
    return sc


def test_alignment_search_recovers_a_player_delay():
    sc = _long_scene(5)
    ball, _ = sc.run(events=False)
    inp = base.make_inputs(ball, sc.players_df(), sc.meta())
    assert inp.lag == 5
    assert inp.align["align_median_d_m"] < inp.align["align_median_d_lag0_m"]


def test_alignment_search_keeps_zero_on_aligned_tracks():
    sc = _long_scene(0)
    ball, _ = sc.run(events=False)
    inp = base.make_inputs(ball, sc.players_df(), sc.meta())
    assert inp.lag == 0 and inp.radius_scale == 1.0


def test_frames_without_players_are_survived():
    sc = lib.pass_scene()
    players = sc.players_df()
    players = players[~players["frame"].between(50, 90)]           # whole-scene player loss
    ball, _ = sc.run(events=False)
    from pipeline.events import detect
    ev = detect(ball, players, sc.meta())
    df = features.build(ball, players, ev, sc.meta())
    assert len(df) == sc.n_frames
    assert (df["n_players_h"][50:91] == 0).all()
    assert df["near_d_h"][50:91].isna().all()
    # No touch, carry or control is placed where nobody is tracked.
    inside = ev[(ev["frame"] >= 52) & (ev["frame"] <= 88)]
    assert not inside["kind"].isin(["touch", "carry", "control"]).any()
