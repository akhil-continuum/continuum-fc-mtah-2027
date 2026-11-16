"""The K nearest players of each team to the ball, one snapshot per frame (T-32 step 1), written
into features.parquet after the play-context columns.

Players of a team are ranked by distance to the ball on each frame; rank k is "the team's k-th
nearest player on this frame", not a person (no track slot is followed across frames). The ball
is the positioned ball (seen or filled by the ball stage), else its last position held for at most
BALL_HOLD_S; otherwise the frame has no ball (``pnb_ball_ok`` 0, every player value NaN).

Per team t (``h`` = home, ``a`` = away) and rank k = 1..K, in team t's own attack frame (x along
its attack direction, y rotated with it, so a 180-degree turn of the pitch leaves every value
unchanged):

- ``pnb_<t><k>_d``     distance to the ball, m
- ``pnb_<t><k>_dx``    player minus ball along the team's attack direction, m (+: ahead of it)
- ``pnb_<t><k>_dy``    player minus ball across, in the same rotated frame, m
- ``pnb_<t><k>_vto``   speed towards the ball, m/s (+: closing)
- ``pnb_<t><k>_vatt``  velocity along the team's attack direction, m/s (+: advancing)
- ``pnb_<t><k>_brk``   braking over +-0.2 s, m/s^2 (the player followed by position continuity)
- ``pnb_<t><k>_trn``   turn of the velocity over +-0.2 s, rad (0 when slow)

Ball, in the home team's attack frame (centred on the pitch centre):
``pnb_ball_ok`` (symmetric), ``pnb_ball_seen`` (symmetric), ``pnb_ball_speed`` (symmetric),
``pnb_ball_xh``, ``pnb_ball_yh``, ``pnb_ball_vxh``, ``pnb_ball_vyh``.

Mirror symmetry: swapping the teams and turning the pitch exchanges the ``h`` and ``a`` blocks
exactly and negates the four home-frame ball columns. The model builds both views from these
columns (pipeline/models/player_model.py); NaN means "no such player" or "no ball".
"""
import numpy as np

from pipeline.features import loose_ball as LB
from pipeline.features.play_context import team_sign

K = 4
BALL_HOLD_S = 0.5
QUANTS = ("d", "dx", "dy", "vto", "vatt")
# T-32 session 2 (variant S1g2): sudden brake / turn of each nearby player
EXTRA_QUANTS = ("brk", "trn")
BRAKE_DELTA = 5          # frames either side (0.4 s window at 25 fps)
FOLLOW_R_M = 1.5         # max distance from the predicted position to accept the same player
TURN_MIN_SPEED = 1.0     # m/s; below it a turn angle is meaningless (0)


def ball_position(ball, n, fps):
    x = np.full(n, np.nan)
    y = np.full(n, np.nan)
    vx = np.full(n, np.nan)
    vy = np.full(n, np.nan)
    f = ball["frame"].to_numpy().astype(np.int64)
    keep = (f >= 0) & (f < n)
    f = f[keep]
    x[f] = ball["x_m"].to_numpy(float)[keep]
    y[f] = ball["y_m"].to_numpy(float)[keep]
    vx[f] = ball["vx"].to_numpy(float)[keep]
    vy[f] = ball["vy"].to_numpy(float)[keep]
    seen = np.zeros(n, dtype=bool)
    seen[f] = ball["seen"].to_numpy(bool)[keep]
    have = np.isfinite(x) & np.isfinite(y)
    idx = np.maximum.accumulate(np.where(have, np.arange(n), -1))
    ok = (idx >= 0) & ((np.arange(n) - idx) <= BALL_HOLD_S * fps + 1e-9)
    j = np.clip(idx, 0, n - 1)
    bx = np.where(ok, x[j], np.nan)
    by = np.where(ok, y[j], np.nan)
    bvx = np.where(have, vx, np.where(ok, 0.0, np.nan))
    bvy = np.where(have, vy, np.where(ok, 0.0, np.nan))
    return bx, by, bvx, bvy, ok, seen & have


def follow_velocity(p, slots, delta, fps):
    """(n, 2) velocity, delta frames away, of the same-team player nearest to where the player in
    ``slots`` (one slot per frame) is predicted to be by its own velocity; NaN when no player is
    within FOLLOW_R_M or the frame is outside the array. Position continuity only: no slot id."""
    n = p.shape[0]
    rows = np.arange(n)
    x, y = p[rows, slots, 0], p[rows, slots, 1]
    vx, vy = np.nan_to_num(p[rows, slots, 2]), np.nan_to_num(p[rows, slots, 3])
    j = rows + delta
    ok = (j >= 0) & (j < n)
    cand = p[np.clip(j, 0, n - 1)]
    d = np.hypot(cand[:, :, 0] - (x + vx * delta / fps)[:, None],
                 cand[:, :, 1] - (y + vy * delta / fps)[:, None])
    d = np.where(np.isfinite(d), d, np.inf)
    b = np.argmin(d, axis=1)
    good = ok & (d[rows, b] < FOLLOW_R_M)
    v = cand[rows, b, 2:4].copy()
    v[~good] = np.nan
    return v


def brake_turn(p, order, fps):
    """(brk, trn), each (n, K): for the team's k-th nearest player to the ball, followed
    +-BRAKE_DELTA frames by position continuity: brk = (speed before - speed after) / time, m/s^2
    (+: braking); trn = angle between the velocities before and after, rad (0 when either speed is
    under TURN_MIN_SPEED). NaN when the player cannot be followed either side. Both are invariant
    under turning the pitch, so they are team-symmetric like the distances."""
    n, kk = order.shape
    brk = np.full((n, K), np.nan)
    trn = np.full((n, K), np.nan)
    dt = 2.0 * BRAKE_DELTA / fps
    for r in range(kk):
        vb = follow_velocity(p, order[:, r], -BRAKE_DELTA, fps)
        va = follow_velocity(p, order[:, r], BRAKE_DELTA, fps)
        sb, sa = np.hypot(vb[:, 0], vb[:, 1]), np.hypot(va[:, 0], va[:, 1])
        brk[:, r] = (sb - sa) / dt
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = (vb[:, 0] * va[:, 0] + vb[:, 1] * va[:, 1]) / np.maximum(sb * sa, 1e-9)
        t = np.where((sb > TURN_MIN_SPEED) & (sa > TURN_MIN_SPEED),
                     np.arccos(np.clip(cos, -1.0, 1.0)), 0.0)
        t[~(np.isfinite(sb) & np.isfinite(sa))] = np.nan
        trn[:, r] = t
    return brk, trn


def build(ball, players, meta):
    n = int(meta["n_frames"])
    fps = float(meta["fps"])
    L, W = meta["pitch"]
    bx, by, bvx, bvy, ok, seen = ball_position(ball, n, fps)
    sgn = team_sign(meta, n)
    pls, _counts = LB.dense_players(players, n)
    cols = {}
    for k, tag in ((0, "h"), (1, "a")):
        p = pls[k]
        s = sgn[:, k][:, None]
        dx = p[:, :, 0] - bx[:, None]
        dy = p[:, :, 1] - by[:, None]
        d = np.hypot(dx, dy)
        key = np.where(np.isfinite(d), d, np.inf)
        order = np.argsort(key, axis=1, kind="mergesort")[:, :K]
        kk = order.shape[1]
        dd = np.take_along_axis(d, order, 1)
        sdx = np.take_along_axis(dx, order, 1)
        sdy = np.take_along_axis(dy, order, 1)
        pvx = np.take_along_axis(p[:, :, 2], order, 1)
        pvy = np.take_along_axis(p[:, :, 3], order, 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            vto = -(pvx * sdx + pvy * sdy) / np.maximum(dd, 0.1)
        vals = {"d": dd, "dx": sdx * s, "dy": sdy * s, "vto": vto, "vatt": pvx * s}
        brk, trn = brake_turn(p, order, fps)
        vals["brk"], vals["trn"] = brk, trn
        for r in range(K):
            for q in QUANTS + EXTRA_QUANTS:
                col = vals[q][:, r] if r < kk else np.full(n, np.nan)
                col = np.where(np.isfinite(dd[:, r]) if r < kk else False, col, np.nan)
                cols["pnb_{}{}_{}".format(tag, r + 1, q)] = col
    sh = sgn[:, 0]
    cols["pnb_ball_ok"] = ok.astype(float)
    cols["pnb_ball_seen"] = seen.astype(float)
    cols["pnb_ball_speed"] = np.hypot(bvx, bvy)
    cols["pnb_ball_xh"] = (bx - L / 2.0) * sh
    cols["pnb_ball_yh"] = (by - W / 2.0) * sh
    cols["pnb_ball_vxh"] = bvx * sh
    cols["pnb_ball_vyh"] = bvy * sh
    assert all(len(v) == n for v in cols.values())
    return cols
