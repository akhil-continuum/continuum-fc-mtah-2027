"""T-32 feature blocks: play context (pipeline/features/play_context.py) and the K nearest players
per team (pipeline/features/players_near_ball.py). Mirror symmetry, touch-count windows, duel."""
import numpy as np
import pandas as pd

from pipeline.features import play_context as PC
from pipeline.features import players_near_ball as PNB

L, W, FPS = 105.0, 68.0, 25.0


def _scene(n=400, seed=1, attack=(1, -1)):
    rng = np.random.default_rng(seed)
    rows = []
    for t in (0, 1):
        x = rng.uniform(10, 95, 11)
        y = rng.uniform(5, 63, 11)
        for f in range(n):
            vx = rng.normal(0, 1, 11)
            vy = rng.normal(0, 1, 11)
            x = np.clip(x + vx / FPS, 0, L)
            y = np.clip(y + vy / FPS, 0, W)
            for i in range(11):
                rows.append((f, t, x[i], y[i], vx[i], vy[i]))
    p = pd.DataFrame(rows, columns=["frame", "team", "x_m", "y_m", "vx", "vy"])
    seen = np.ones(n, bool)
    seen[150:230] = False
    bx = 30 + 40 * np.arange(n) / n + rng.normal(0, 0.3, n)
    by = 34 + 10 * np.sin(np.arange(n) / 40.0)
    ball = pd.DataFrame({"frame": np.arange(n), "x_m": np.where(seen, bx, np.nan),
                         "y_m": np.where(seen, by, np.nan), "vx": np.where(seen, 1.6, np.nan),
                         "vy": np.where(seen, 0.2, np.nan), "seen": seen,
                         "filled": np.zeros(n, bool)})
    ev = pd.DataFrame({"frame": [40, 60, 90, 250, 300, 310], "kind": ["touch"] * 6,
                       "team": [0, 0, 1, 1, 0, 1]})
    meta = {"n_frames": n, "fps": FPS, "pitch": [L, W], "halves": [[0, n]],
            "attack_direction": {"1": {"0": attack[0], "1": attack[1]}}}
    return ball, p, ev, meta


def _mirror(ball, p, ev, meta):
    q = p.copy()
    q["team"] = 1 - q["team"]
    q["x_m"], q["y_m"] = L - q["x_m"], W - q["y_m"]
    q["vx"], q["vy"] = -q["vx"], -q["vy"]
    q = q.sort_values(["frame", "team"], kind="mergesort").reset_index(drop=True)
    b = ball.copy()
    b["x_m"], b["y_m"] = L - b["x_m"], W - b["y_m"]
    b["vx"], b["vy"] = -b["vx"], -b["vy"]
    e = ev.copy()
    e["team"] = 1 - e["team"]
    m = dict(meta)
    m["attack_direction"] = {h: {"0": -d["1"], "1": -d["0"]}
                             for h, d in meta["attack_direction"].items()}
    return b, q, e, m


def _antisym(k):
    return "_diff" in k or "_rel_" in k


def test_play_context_mirror_symmetry():
    ball, p, ev, meta = _scene()
    a = PC.build(ball, p, ev, meta)
    b = PC.build(*_mirror(ball, p, ev, meta))
    assert set(a) == set(b)
    for k in a:
        x, y = np.asarray(a[k], float), np.asarray(b[k], float)
        assert np.array_equal(np.isnan(x), np.isnan(y)), k
        fin = np.isfinite(x)
        exp = -y[fin] if _antisym(k) else y[fin]
        np.testing.assert_allclose(x[fin], exp, atol=1e-6, err_msg=k)


def test_players_near_ball_mirror_swaps_blocks():
    ball, p, ev, meta = _scene()
    a = PNB.build(ball, p, meta)
    b = PNB.build(*_mirror(ball, p, ev, meta)[:2], _mirror(ball, p, ev, meta)[3])
    for k in a:
        if k.startswith("pnb_h") or k.startswith("pnb_a"):
            other = "pnb_" + ("a" if k[4] == "h" else "h") + k[5:]
            x, y = np.asarray(a[k]), np.asarray(b[other])
            sign = 1.0
        else:
            x, y = np.asarray(a[k]), np.asarray(b[k])
            sign = -1.0 if k in ("pnb_ball_xh", "pnb_ball_yh", "pnb_ball_vxh", "pnb_ball_vyh") \
                else 1.0
        assert np.array_equal(np.isnan(x), np.isnan(y)), k
        fin = np.isfinite(x)
        np.testing.assert_allclose(x[fin], sign * y[fin], atol=1e-6, err_msg=k)


def test_players_near_ball_ranks_and_hold():
    ball, p, ev, meta = _scene()
    c = PNB.build(ball, p, meta)
    d1, d2 = c["pnb_h1_d"], c["pnb_h2_d"]
    fin = np.isfinite(d1) & np.isfinite(d2)
    assert (d1[fin] <= d2[fin]).all()
    # ball last seen at frame 149, held for 0.5 s (12.5 frames), then no ball
    assert c["pnb_ball_ok"][149 + 12] == 1 and c["pnb_ball_ok"][149 + 13] == 0
    assert np.isnan(c["pnb_h1_d"][200])


def test_touch_count_windows():
    ev = pd.DataFrame({"frame": [100, 110, 130], "kind": ["touch"] * 3, "team": [0, 0, 1]})
    c = PC.touch_counts(ev, 400, FPS)
    assert c["pc_touch_diff_b1"][120] == 2          # [95, 120]: two home touches
    assert c["pc_touch_diff_a1"][120] == -1         # (120, 145]: one away touch
    assert c["pc_touch_diff_b1"][100] == 1          # the touch frame counts as "before"
    assert c["pc_touch_diff_a1"][99] == 2
    assert c["pc_touch_n_c1"][120] == 3


def test_duel_flag_and_lead():
    n = 50
    rows = []
    for f in range(n):
        rows += [(f, 0, 50.0, 34.0, 0.0, 0.0), (f, 1, 51.5, 34.0, -1.0, 0.0)]
        rows += [(f, 0, 10.0 + i, 5.0, 0, 0) for i in range(6)]
        rows += [(f, 1, 90.0 + i, 60.0, 0, 0) for i in range(6)]
    p = pd.DataFrame(rows, columns=["frame", "team", "x_m", "y_m", "vx", "vy"])
    ball = pd.DataFrame({"frame": np.arange(n), "x_m": 50.5, "y_m": 34.0, "vx": 0.0, "vy": 0.0,
                         "seen": True, "filled": False})
    meta = {"n_frames": n, "fps": FPS, "pitch": [L, W], "halves": [[0, n]],
            "attack_direction": {"1": {"0": 1, "1": -1}}}
    pls, counts = PC.LB.dense_players(p, n)
    c = PC.near_ball(ball, pls, counts, meta, n, FPS)
    assert c["pc_duel"][25] == 1
    assert c["pc_duel_lead_diff_c05"][25] > 0.4      # home 0.5 m, away 1.0 m from the ball
    assert c["pc_duel_close"][25] > 0.9              # away runs at home at 1 m/s


def _path(speed, heading):
    """Positions integrated from per-frame speed and heading; velocity by a +-2 frame central
    difference (as the loader computes it)."""
    x = np.cumsum(speed * np.cos(heading)) / FPS
    y = np.cumsum(speed * np.sin(heading)) / FPS
    vx, vy = np.full(len(x), np.nan), np.full(len(x), np.nan)
    vx[2:-2] = (x[4:] - x[:-4]) * FPS / 4.0
    vy[2:-2] = (y[4:] - y[:-4]) * FPS / 4.0
    return x, y, vx, vy


def test_brake_and_turn_of_nearest_player():
    """One home player runs at 6 m/s and brakes to a stop over frames 95-105 (15 m/s^2); another
    runs at 6 m/s and turns 90 degrees over frames 195-205. The ball is next to the one tested."""
    n = 300
    f = np.arange(n)
    sa = np.clip(6.0 * (105 - f) / 10.0, 0.0, 6.0)
    xa, ya, vxa, vya = _path(sa, np.zeros(n))
    hb = np.clip((f - 195) / 10.0, 0.0, 1.0) * np.pi / 2
    xb, yb, vxb, vyb = _path(np.full(n, 6.0), hb)
    rows = []
    for k in range(n):
        rows.append((k, 0, 10 + xa[k], 20 + ya[k], vxa[k], vya[k]))
        rows.append((k, 0, 10 + xb[k], 40 + yb[k], vxb[k], vyb[k]))
        rows.append((k, 1, 100.0, 60.0, 0.0, 0.0))
    p = pd.DataFrame(rows, columns=["frame", "team", "x_m", "y_m", "vx", "vy"])
    near_a = f < 150
    bx = np.where(near_a, 10 + xa, 10 + xb) + 0.3
    by = np.where(near_a, 20 + ya, 40 + yb)
    ball = pd.DataFrame({"frame": f, "x_m": bx, "y_m": by, "vx": 0.0, "vy": 0.0,
                         "seen": np.ones(n, bool), "filled": np.zeros(n, bool)})
    meta = {"n_frames": n, "fps": FPS, "pitch": [L, W], "halves": [[0, n]],
            "attack_direction": {"1": {"0": 1, "1": -1}}}
    c = PNB.build(ball, p, meta)
    brk, trn = c["pnb_h1_brk"], c["pnb_h1_trn"]
    assert 10.0 < brk[100] < 16.0 and abs(brk[50]) < 1e-6    # about 15 m/s^2
    assert abs(trn[50]) < 1e-6 and trn[100] < 1e-6            # braking in a line is no turn
    assert 1.3 < trn[200] <= np.pi / 2 + 1e-6 and abs(brk[200]) < 0.5  # edges smoothed
    assert np.isnan(brk[0]) and np.isnan(trn[n - 1])          # cannot follow outside the clip
