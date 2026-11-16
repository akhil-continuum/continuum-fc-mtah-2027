"""Bench noise (bench/noise.py): v3, v2 and legacy models, contract shape, seeding, routed fixes."""

import json
import os

import numpy as np
import pandas as pd
import pytest

from bench import noise
from bench.events import confirm_stoppages
from bench.metrica_io import Game


def synthetic_game(n=3000, split=1500, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)[:, None]
    base = np.stack([rng.uniform(-40, 40, 22), rng.uniform(-25, 25, 22)], -1)
    players = base[None] + np.stack([3 * np.sin(t / 50.0 + np.arange(22)),
                                     2 * np.cos(t / 70.0 + np.arange(22))], -1)
    # Home attacks +x in half 1: home players sit at lower x; teams swap ends at half-time.
    team = np.array([0] * 11 + [1] * 11, np.int8)
    players[:, team == 0, 0] -= 10
    players[:, team == 1, 0] += 10
    players[split:, :, 0] *= -1
    ball = players[:, 3].copy() + 0.5
    ball[700:760] = np.nan          # a dead-ball stretch
    ball[1000:1003] = np.nan        # a short clean gap (to be bridged)
    period = np.ones(n, int)
    period[split:] = 2
    return Game("syn", period, ball, players, team, ["p%d" % i for i in range(22)],
                pd.DataFrame(), pitch=(105.0, 68.0))


def write_truth(art, game):
    src = os.path.join(art, "syn")
    os.makedirs(src, exist_ok=True)
    lab = np.array(["H"] * game.n_frames, dtype=object)
    lab[700:760] = "D"
    lab[1499] = "X"
    pd.DataFrame({"frame": np.arange(game.n_frames), "label": lab}).to_csv(
        os.path.join(src, "truth.csv"), index=False)
    for name in ("truth_first_touch.csv", "truth_confirmed.csv"):
        pd.read_csv(os.path.join(src, "truth.csv")).to_csv(os.path.join(src, name), index=False)
    pd.DataFrame({"period": [1], "start": [700], "restart": [760], "restart_team": ["A"],
                  "restart_type": ["THROW IN"], "cause": ["ball_out"], "stop_frame": [690]}
                 ).to_csv(os.path.join(src, "stoppages.csv"), index=False)
    with open(os.path.join(src, "truth_meta.json"), "w") as fh:
        json.dump({"halves": [[0, 1500], [1500, 3000]], "graded_spans": [[0, 1499], [1500, 3000]]},
                  fh)


@pytest.fixture
def art(tmp_path, monkeypatch):
    monkeypatch.setattr(noise, "ART", str(tmp_path))
    return str(tmp_path)


@pytest.mark.parametrize("model", ["v3", "v2", "legacy"])
def test_contract_files_and_meta(art, model):
    game = synthetic_game()
    write_truth(art, game)
    meta = noise.build("syn", "match", game=game, model=model)
    root = os.path.join(art, noise.root_name("syn", "match", model))
    assert os.path.basename(root) == {"v3": "syn_match", "v2": "syn_legacy2match",
                                      "legacy": "syn_legacymatch"}[model]
    assert meta["noise"]["model"] == model
    players = pd.read_parquet(os.path.join(root, "players.parquet"))
    ball = pd.read_parquet(os.path.join(root, "ball_raw.parquet"))
    assert list(players.columns) == ["frame", "team", "x_m", "y_m", "vx", "vy"]
    # No speed channel on the ball: the match's raw_s is unusable, the bench offers none.
    assert list(ball.columns) == ["frame", "x_m", "y_m", "seen"]
    assert len(ball) == game.n_frames
    # (a) attack direction as the shared loader writes it; home attacks +x in half 1 here.
    assert meta["attack_direction"]["1"] == {"0": 1, "1": -1}
    assert meta["attack_direction"]["2"] == {"0": -1, "1": 1}
    # (b) velocities clipped like the loader.
    speed = np.hypot(players["vx"], players["vy"])
    assert np.nanmax(speed) <= noise.MAX_SPEED_MPS + 1e-6


def test_v2_is_seeded_and_deterministic(art):
    game = synthetic_game()
    write_truth(art, game)
    noise.build("syn", "match", game=game, model="v2")
    root = os.path.join(art, "syn_legacy2match")
    b1 = pd.read_parquet(os.path.join(root, "ball_raw.parquet"))
    p1 = pd.read_parquet(os.path.join(root, "players.parquet"))
    noise.build("syn", "match", game=game, model="v2")
    b2 = pd.read_parquet(os.path.join(root, "ball_raw.parquet"))
    p2 = pd.read_parquet(os.path.join(root, "players.parquet"))
    pd.testing.assert_frame_equal(b1, b2)
    pd.testing.assert_frame_equal(p1, p2)


def test_v2_ball_is_smooth_and_short_gaps_bridged():
    game = synthetic_game()
    b = noise._fill_short_gaps(game.ball, game.period, 50)
    assert np.isfinite(b[1000:1003]).all()           # 3-frame clean gap bridged
    assert np.isnan(b[700:760]).all()                # 60-frame gap kept
    e = noise._smooth_noise(np.random.default_rng(0), 20000, 1.2, 12)
    assert e.std() == pytest.approx(1.2, rel=1e-6)
    assert np.median(np.abs(np.diff(e, 2))) < 0.01   # P-3/P-6: millimetre second differences


def test_velocities_per_half_and_clipped():
    n = 100
    X = np.zeros((n, 1))
    X[50:] = 60.0                                    # positions jump at the half boundary
    Y = np.zeros((n, 1))
    VX, VY = noise._velocities(X, Y, [(0, 50), (50, 100)])
    assert np.nanmax(np.abs(VX)) == 0.0              # no difference spans the boundary
    X2 = np.arange(n, dtype=float)[:, None] * 2.0    # 50 m/s: above the loader's cap
    VX2, _ = noise._velocities(X2, Y, [(0, n)])
    assert np.isnan(VX2).all()


def test_half_start_kickoff_kept_even_when_empty():
    stops = pd.DataFrame({"period": [1, 1], "start": [0, 100], "restart": [0, 100],
                          "restart_team": ["A", "H"], "restart_type": ["KICK OFF", "THROW IN"],
                          "cause": ["half_start", "ball_out"], "stop_frame": [-1, 90]})
    out = confirm_stoppages(stops, np.zeros(200, bool))
    assert list(out["cause"]) == ["half_start"]


@pytest.mark.parametrize("order", ["before_gaps", "after_gaps"])
def test_v3_is_v2_plus_lowpass(art, order):
    """v3: same seed, seen mask and players as v2; only the ball positions are filtered; seeded."""
    game = synthetic_game()
    write_truth(art, game)
    noise.build("syn", "match", game=game, model="v2")
    noise.build("syn", "match", game=game, model="v3", lowpass_order=order)
    v2 = os.path.join(art, "syn_legacy2match")
    v3 = os.path.join(art, "syn_match")
    pd.testing.assert_frame_equal(pd.read_parquet(os.path.join(v2, "players.parquet")),
                                  pd.read_parquet(os.path.join(v3, "players.parquet")))
    b2 = pd.read_parquet(os.path.join(v2, "ball_raw.parquet"))
    b3 = pd.read_parquet(os.path.join(v3, "ball_raw.parquet"))
    assert (b2["seen"] == b3["seen"]).all()
    s = b3["seen"].to_numpy()
    assert not np.allclose(b2["x_m"].to_numpy()[s], b3["x_m"].to_numpy()[s])
    for name in ("truth.csv", "truth_first_touch.csv", "truth_confirmed.csv", "stoppages.csv"):
        assert os.path.isfile(os.path.join(v3, name))
    with open(os.path.join(v3, "meta.json")) as fh:
        meta = json.load(fh)
    assert meta["noise"]["params"]["ball_lowpass_order"] == order
    assert meta["noise"]["params"]["ball_lowpass_sigma_frames"] == noise.V3_LOWPASS["match"]
    noise.build("syn", "match", game=game, model="v3", lowpass_order=order)
    pd.testing.assert_frame_equal(b3, pd.read_parquet(os.path.join(v3, "ball_raw.parquet")))


def test_lowpass_stays_inside_runs_and_halves():
    n = 400
    b = np.zeros((n, 2))
    b[:, 0] = np.arange(n, dtype=float)
    b[100:110] = np.nan                       # a gap
    b[300:, 0] += 1000.0                      # a jump at the half boundary
    period = np.ones(n, int)
    period[300:] = 2
    out = noise.lowpass_in_runs(b, period, 8.0)
    assert np.isnan(out[100:110]).all() and np.isfinite(out[:100]).all()
    # Nothing leaks across the half boundary: half 1 never sees the +1000 jump.
    assert np.nanmax(out[:300, 0]) < 300 and out[300:, 0].min() >= 1300 - 1e-9
    # A linear track is unchanged away from run edges.
    assert np.allclose(out[145:265, 0], b[145:265, 0])   # 4 sigma from the run edges
    assert np.array_equal(noise.lowpass_in_runs(b, period, 0.0), b, equal_nan=True)


def test_lowpass_makes_second_difference_positively_correlated():
    """White jitter gives lag-1 autocorrelation of the second difference near -0.67; the match's
    filtered track gives about +0.99 (touch_funnel)."""
    rng = np.random.default_rng(1)
    b = np.cumsum(rng.normal(0, 0.1, (5000, 2)), axis=0) + rng.normal(0, 0.05, (5000, 2))
    def ac(x):
        d2 = np.diff(x[:, 0], 2)
        return np.corrcoef(d2[1:], d2[:-1])[0, 1]
    assert ac(b) < -0.3
    assert ac(noise.lowpass_in_runs(b, np.ones(5000, int), 7.5)) > 0.9
