"""Loose-ball features (pipeline/features/loose_ball.py, P30 / P-14): the team clearly about to
reach a rolling ball, the contested trap, mirror symmetry, unseen ball (short extrapolation, then
NaN), whole-scene player loss, centred windows."""
import numpy as np
import pandas as pd

from pipeline.features import loose_ball as LB

L, W, FPS = 105.0, 68.0, 25.0


def _scene(n=200, away_x=None, hide=None, drop=None):
    """Ball rolls from (30, 34) along +x at 8 m/s from frame 50. Home player waits at (60, 34)
    on the path; one away player at (away_x, 20) (default far); ten more per team far away."""
    rows = []
    hx = [60.0] + [5.0 + 2 * i for i in range(10)]
    hy = [34.0] + [5.0] * 10
    ax = [away_x if away_x is not None else 100.0] + [100.0 - 2 * i for i in range(10)]
    ay = [20.0 if away_x is not None else 64.0] + [64.0] * 10
    for f in range(n):
        if drop is not None and drop[0] <= f < drop[1]:
            continue
        for t, xs, ys in ((0, hx, hy), (1, ax, ay)):
            for i in range(11):
                rows.append((f, t, xs[i], ys[i], 0.0, 0.0))
    p = pd.DataFrame(rows, columns=["frame", "team", "x_m", "y_m", "vx", "vy"])
    fr = np.arange(n)
    bx = np.where(fr < 50, 30.0, 30.0 + 8.0 * (fr - 50) / FPS)
    vx = np.where(fr < 50, 0.0, 8.0)
    seen = np.ones(n, bool)
    if hide is not None:
        seen[hide[0]:hide[1]] = False
    ball = pd.DataFrame({"frame": fr, "x_m": np.where(seen, bx, np.nan),
                         "y_m": np.where(seen, 34.0, np.nan), "vx": np.where(seen, vx, np.nan),
                         "vy": np.where(seen, 0.0, np.nan), "seen": seen,
                         "filled": np.zeros(n, bool)})
    meta = {"n_frames": n, "fps": FPS, "pitch": [L, W], "halves": [[0, n]]}
    return ball, p, meta


def _mirror(ball, p, meta):
    q = p.copy()
    q["team"] = 1 - q["team"]
    q["x_m"], q["y_m"] = L - q["x_m"], W - q["y_m"]
    q["vx"], q["vy"] = -q["vx"], -q["vy"]
    q = q.sort_values(["frame", "team"], kind="mergesort").reset_index(drop=True)
    b = ball.copy()
    b["x_m"], b["y_m"] = L - b["x_m"], W - b["y_m"]
    b["vx"], b["vy"] = -b["vx"], -b["vy"]
    return b, q, meta


def test_ball_rolling_to_one_team_is_clear():
    c = LB.build(*_scene())
    # ball at ~51 m at frame 115, home player waiting at 60 m, nearest away player far
    assert c["lb_valid"][115] == 1
    assert c["lb_tti_diff_c0"][115] < -LB.MARGIN_S
    assert c["lb_clear_diff_c0"][115] == 1.0
    assert c["lb_pfirst_diff_c0"][115] > 0.9


def test_contested_ball_is_not_clear():
    # an away player standing next to the home player on the path: no clear team (trap)
    ball, p, meta = _scene()
    p.loc[(p.team == 1) & (p.x_m == 100.0) & (p.y_m == 64.0), ["x_m", "y_m"]] = [60.0, 35.0]
    c3 = LB.build(ball, p, meta)
    assert c3["lb_clear_diff_c0"][115] == 0.0
    assert abs(c3["lb_pfirst_diff_c0"][115]) < 0.5


def test_mirror_symmetry():
    a = LB.build(*_scene(away_x=70.0, hide=(120, 130)))
    b = LB.build(*_mirror(*_scene(away_x=70.0, hide=(120, 130))))
    for k in a:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        assert np.array_equal(np.isnan(x), np.isnan(y)), k
        fin = np.isfinite(x)
        if "_diff_" in k:
            np.testing.assert_allclose(x[fin], -y[fin], atol=1e-9, err_msg=k)
        elif k in ("lb_tti_h", "lb_tti_a"):
            other = "lb_tti_a" if k == "lb_tti_h" else "lb_tti_h"
            np.testing.assert_allclose(x[fin], np.asarray(b[other])[fin], atol=1e-9)
        else:
            np.testing.assert_allclose(x[fin], y[fin], atol=1e-9, err_msg=k)


def test_unseen_ball_extrapolates_briefly_then_nan():
    c = LB.build(*_scene(hide=(100, 160)))
    assert c["lb_valid"][105] == 1 and c["lb_extrap"][105] == 1          # 0.24 s after last seen
    assert c["lb_clear_diff_c0"][105] == 1.0
    assert c["lb_valid"][140] == 0 and np.isnan(c["lb_tti_diff_c0"][140])   # 1.6 s: too old
    assert np.isfinite(c["lb_tti_diff_c05"][113])     # centred window reaches valid frames


def test_whole_scene_player_loss_is_invalid():
    c = LB.build(*_scene(drop=(80, 90)))
    assert c["lb_valid"][85] == 0 and np.isnan(c["lb_tti_diff_c0"][85])
    assert c["lb_valid"][95] == 1
