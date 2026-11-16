"""Team-shape features (pipeline/features/shape.py): mirror symmetry, validity flag under player
loss, direction of the forward-velocity feature, centred windows."""
import numpy as np
import pandas as pd

from pipeline.features import shape

L, W, FPS = 105.0, 68.0, 25.0


def _scene(n=400, seed=0, attack=(1, -1)):
    """Two teams of 11 random walkers; home runs forwards (its attack direction) in frames
    [200, 300); ball seen except a gap."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in (0, 1):
        x = rng.uniform(10, 95, 11)
        y = rng.uniform(5, 63, 11)
        for f in range(n):
            vx = rng.normal(0, 1, 11)
            vy = rng.normal(0, 1, 11)
            if t == 0 and 200 <= f < 300:
                vx = vx + 4.0 * attack[0]
            x = np.clip(x + vx / FPS, 0, L)
            y = np.clip(y + vy / FPS, 0, W)
            for i in range(11):
                rows.append((f, t, x[i], y[i], vx[i], vy[i]))
    p = pd.DataFrame(rows, columns=["frame", "team", "x_m", "y_m", "vx", "vy"])
    seen = np.ones(n, bool)
    seen[120:240] = False
    ball = pd.DataFrame({"frame": np.arange(n), "x_m": np.linspace(30, 70, n),
                         "y_m": np.full(n, 34.0), "seen": seen})
    meta = {"n_frames": n, "fps": FPS, "pitch": [L, W], "halves": [[0, n]],
            "attack_direction": {"1": {"0": attack[0], "1": attack[1]}}}
    return ball, p, meta


def _mirror(ball, p, meta):
    q = p.copy()
    q["team"] = 1 - q["team"]
    q["x_m"], q["y_m"] = L - q["x_m"], W - q["y_m"]
    q["vx"], q["vy"] = -q["vx"], -q["vy"]
    q = q.sort_values(["frame", "team"], kind="mergesort").reset_index(drop=True)
    b = ball.copy()
    b["x_m"], b["y_m"] = L - b["x_m"], W - b["y_m"]
    m = dict(meta)
    m["attack_direction"] = {h: {"0": -d["1"], "1": -d["0"]}
                             for h, d in meta["attack_direction"].items()}
    return b, q, m


def test_mirror_symmetry():
    ball, p, meta = _scene()
    a = shape.build(ball, p, meta)
    b = shape.build(*_mirror(ball, p, meta))
    assert set(a) == set(b)
    for k in a:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        assert np.array_equal(np.isnan(x), np.isnan(y)), k
        fin = np.isfinite(x)
        if "_diff_" in k or "_rel_" in k:
            np.testing.assert_allclose(x[fin], -y[fin], atol=1e-9, err_msg=k)
        else:
            np.testing.assert_allclose(x[fin], y[fin], atol=1e-9, err_msg=k)


def test_forward_velocity_follows_attack_direction():
    for attack in ((1, -1), (-1, 1)):
        ball, p, meta = _scene(attack=attack)
        c = shape.build(ball, p, meta)
        v = c["shape_fwd_vel_diff_c1"]
        assert np.nanmean(v[220:280]) > 3.0
        assert abs(np.nanmean(v[20:150])) < 1.0


def test_player_loss_gives_nan_and_flag_not_a_jump():
    ball, p, meta = _scene()
    drop = (p["frame"] >= 100) & (p["frame"] < 110) & (p["team"] == 1) & (p["x_m"] > -1)
    keep_two = p[drop].groupby("frame").head(2).index
    p2 = p[~drop | p.index.isin(keep_two)].reset_index(drop=True)
    c = shape.build(ball, p2, meta)
    assert c["shape_valid"][105] == 0 and c["shape_valid"][90] == 1
    # windowed values ignore the invalid frames rather than jumping
    full = shape.build(ball, p, meta)
    assert abs(c["shape_depth_diff_c1"][105] - full["shape_depth_diff_c1"][105]) < 2.0
    # a whole-window loss gives NaN
    p3 = p[(p["frame"] < 100) | (p["frame"] >= 200) | (p["team"] == 0)].reset_index(drop=True)
    c3 = shape.build(ball, p3, meta)
    assert np.isnan(c3["shape_fwd_vel_diff_c05"][150])


def test_pressure_needs_a_recent_ball():
    ball, p, meta = _scene()
    c = shape.build(ball, p, meta)
    assert np.isfinite(c["shape_press_diff_c05"][60])
    assert np.isnan(c["shape_press_diff_c05"][210])     # ball unseen for > 2 s, window inside gap
