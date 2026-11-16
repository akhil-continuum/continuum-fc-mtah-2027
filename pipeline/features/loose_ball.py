"""Loose-ball evidence: which team can reach the ball first (rulebook P30, P3, P7; proposal P-14),
written into features.parquet after the team-shape columns.

Per frame t and per team, the time its best-placed tracked player needs to reach the ball's path
over the next PATH_S seconds, from the noisy inputs only (ball.parquet, players.parquet):

- Ball path. The ball position at t, t + STEP, ..., t + PATH_S (frames), taken from the tracked
  ball where it has a position (ball.parquet x_m / y_m; seen or filled by the ball stage). Where
  it has none, the position is extrapolated in a straight line from the last positioned frame
  with that frame's velocity (speed capped at BALL_V_CAP), for at most EXTRAP_S + PATH_S after
  it. The frame itself must be positioned or at most EXTRAP_S after the last positioned frame,
  otherwise every loose-ball value of the frame is NaN and ``lb_valid`` is 0. Using the future
  ball track is a centred (non-causal) read of the input, not hindsight truth.
- Player arrival (Spearman 2018, pitch control; the model bench.truth_labels runs on clean
  tracking, here with the reaction time set to 0 on the bench, see T_REACT_S): a player keeps his
  velocity for T_REACT_S, then runs straight at V_MAX_MPS; he plays the ball within R_CONTROL_M. Arrival at path point k is
  a_k = T_REACT_S + max(0, |b_k - (p + v T_REACT_S)| - R_CONTROL_M) / V_MAX_MPS, and his time to
  the ball is min_k max(s_k, a_k) (s_k = time of the ball at point k; the ball is taken as held
  at the last point, so a late arrival still counts). Missing velocity counts as standing still.
- Team value: the minimum over the team's tracked players on the frame, capped at TTI_CAP_S.
  Team level only: no player slot is followed across frames.
- Validity (T7, whole-scene player loss): the frame is invalid when either team has fewer than
  MIN_TEAM_PLAYERS tracked players.

Columns (FEATURES.md, section Loose ball). Every ``*_diff_*`` column is home minus away or its
antisymmetric transform, so swapping the teams (and flipping the pitch) negates it; distances do
not depend on the attack direction. Windows: ``_c0`` the frame itself, ``_c05`` / ``_c1`` centred
means over +-0.5 / 1 s ignoring invalid frames (NaN when the whole window is invalid).

- lb_tti_diff     home time minus away time (s): negative when home is first.
- lb_pfirst_diff  2 P(home first) - 1, with P from Spearman's logistic arrival uncertainty
                  (scale sqrt(3) sigma / pi, sigma TTI_SIGMA_S): in [-1, 1], positive for home.
- lb_clear_diff   +1 when home is first by at least MARGIN_S (P(first) >= P_CLEAR), -1 when away
                  is, else 0 ("clearly one team", P30).

No constant here refers to the competition match or the public window: the physical constants are
Spearman's; EXTRAP_S, PATH_S and the windows are chosen on physical grounds (a short, nearly
straight ball path; the reaction time of the model) and checked on the bench.
"""
import math

import numpy as np

# Reaction time: Spearman's 0.7 s was replaced by 0 on the bench. A reaction delay common to all
# players cancels in the home-minus-away difference; what it adds is the extrapolation of each
# player's velocity, and the noisy input velocities make that harmful. Bench, match-level roots,
# labels truth, live frames, sign accuracy of lb_pfirst_diff_c05 (mean of three games):
# reaction 0.7 s 0.751, 0.3 s 0.779, 0 s 0.794 (path 0.5 / 2 s and 7 m/s changed it < 0.01).
T_REACT_S = 0.0
V_MAX_MPS = 5.0             # Spearman (2018): straight run at this speed afterwards
R_CONTROL_M = 1.0           # a player plays the ball within about a stride
TTI_SIGMA_S = 0.45          # Spearman (2018): arrival-time uncertainty
P_CLEAR = 0.9               # "clearly first": P(first) >= 0.9 under that uncertainty
MARGIN_S = math.log(P_CLEAR / (1 - P_CLEAR)) * math.sqrt(3) * TTI_SIGMA_S / math.pi   # 0.545 s
PATH_S = 1.0                # look-ahead of the ball path
STEP_FRAMES = 2             # path sampled every 2 frames (0.08 s at 25 fps)
EXTRAP_S = 0.5              # unseen ball: extrapolate at most this long after the last position
BALL_V_CAP = 35.0           # m/s; above any kick, removes noisy velocity spikes
TTI_CAP_S = 3.0             # team time capped: beyond this the team is simply "far"
MIN_TEAM_PLAYERS = 6        # T7, as the team-shape features
WINDOWS_S = {0.0: "c0", 0.5: "c05", 1.0: "c1"}
CHUNK = 20000


def _centred_mean(a, half):
    a = np.asarray(a, dtype=float)
    if half <= 0:
        return a.copy()
    ok = np.isfinite(a)
    c = np.concatenate([[0.0], np.cumsum(np.where(ok, a, 0.0))])
    k = np.concatenate([[0], np.cumsum(ok)])
    n = len(a)
    i = np.arange(n)
    lo, hi = np.maximum(i - half, 0), np.minimum(i + half + 1, n)
    num, den = c[hi] - c[lo], k[hi] - k[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def ball_estimate(ball, n, fps):
    """(est_xy (n, 2), start_ok (n,), extrap (n,)): positioned ball, else the straight-line
    extrapolation from the last positioned frame (NaN beyond EXTRAP_S + PATH_S); start_ok marks
    frames positioned or at most EXTRAP_S after the last position."""
    x = np.full(n, np.nan)
    y = np.full(n, np.nan)
    vx = np.zeros(n)
    vy = np.zeros(n)
    f = ball["frame"].to_numpy().astype(np.int64)
    keep = (f >= 0) & (f < n)
    f = f[keep]
    x[f] = ball["x_m"].to_numpy(float)[keep]
    y[f] = ball["y_m"].to_numpy(float)[keep]
    if "vx" in ball:
        vx[f] = np.nan_to_num(ball["vx"].to_numpy(float)[keep])
        vy[f] = np.nan_to_num(ball["vy"].to_numpy(float)[keep])
    sp = np.hypot(vx, vy)
    scale = np.where(sp > BALL_V_CAP, BALL_V_CAP / np.maximum(sp, 1e-9), 1.0)
    vx, vy = vx * scale, vy * scale
    have = np.isfinite(x) & np.isfinite(y)
    idx = np.maximum.accumulate(np.where(have, np.arange(n), -1))
    j = np.clip(idx, 0, n - 1)
    dt = (np.arange(n) - idx) / fps
    reach = (idx >= 0) & (dt <= EXTRAP_S + PATH_S + 1e-9)
    ex = np.where(reach, x[j] + vx[j] * dt, np.nan)
    ey = np.where(reach, y[j] + vy[j] * dt, np.nan)
    est = np.stack([np.where(have, x, ex), np.where(have, y, ey)], axis=1)
    start_ok = have | ((idx >= 0) & (dt <= EXTRAP_S + 1e-9))
    return est, start_ok, (~have) & start_ok


def dense_players(players, n):
    """Per team (n, K, 4) arrays of x, y, vx, vy (NaN padding) and (n,) counts."""
    f = players["frame"].to_numpy().astype(np.int64)
    t = players["team"].to_numpy().astype(np.int64)
    keep = (f >= 0) & (f < n) & ((t == 0) | (t == 1))
    out, counts = [], []
    for team in (0, 1):
        s = keep & (t == team)
        ff = f[s]
        order = np.argsort(ff, kind="mergesort")
        ff = ff[order]
        vals = np.stack([players[c].to_numpy(float)[s][order] for c in ("x_m", "y_m", "vx", "vy")],
                        axis=1)
        cnt = np.bincount(ff, minlength=n)
        start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        slot = np.arange(len(ff)) - start[ff]
        K = int(cnt.max()) if len(cnt) and cnt.max() > 0 else 1
        arr = np.full((n, K, 4), np.nan)
        arr[ff, slot] = vals
        out.append(arr)
        counts.append(cnt)
    return out, counts


def team_times(est, start_ok, pl, fps):
    """(n,) time for the team's best-placed player to reach the ball path (s, uncapped; NaN where
    the frame has no ball start or no player)."""
    n = len(est)
    ks = np.arange(0, int(round(PATH_S * fps)) + 1, STEP_FRAMES)
    s_k = ks / fps
    out = np.full(n, np.nan)
    for a in range(0, n, CHUNK):
        b = min(n, a + CHUNK)
        fr = np.arange(a, b)
        idx = np.minimum(fr[:, None] + ks[None, :], n - 1)
        path = est[idx]                                          # (m, S, 2)
        beyond = (fr[:, None] + ks[None, :]) > n - 1
        path[beyond] = np.nan
        p = pl[a:b]                                              # (m, K, 4)
        v = np.nan_to_num(p[..., 2:4])
        q = p[..., 0:2] + v * T_REACT_S                          # (m, K, 2)
        d = np.linalg.norm(path[:, None, :, :] - q[:, :, None, :], axis=3)   # (m, K, S)
        arrive = T_REACT_S + np.maximum(0.0, d - R_CONTROL_M) / V_MAX_MPS
        tt = np.maximum(arrive, s_k[None, None, :])
        tt = np.where(np.isfinite(tt), tt, np.inf)
        best = tt.min(axis=(1, 2))
        best = np.where(np.isfinite(best) & start_ok[a:b], best, np.nan)
        out[a:b] = best
    return out


def build(ball, players, meta):
    """Loose-ball columns for features.parquet (FEATURES.md, section Loose ball)."""
    n = int(meta["n_frames"])
    fps = float(meta["fps"])
    est, start_ok, extrap = ball_estimate(ball, n, fps)
    pls, counts = dense_players(players, n)
    tti = [team_times(est, start_ok, pls[k], fps) for k in (0, 1)]
    valid = (counts[0] >= MIN_TEAM_PLAYERS) & (counts[1] >= MIN_TEAM_PLAYERS) & \
        np.isfinite(tti[0]) & np.isfinite(tti[1])
    th = np.where(valid, np.minimum(tti[0], TTI_CAP_S), np.nan)
    ta = np.where(valid, np.minimum(tti[1], TTI_CAP_S), np.nan)
    diff = th - ta
    scale = math.sqrt(3.0) * TTI_SIGMA_S / math.pi
    with np.errstate(invalid="ignore"):
        pfirst = np.tanh(-diff / (2.0 * scale))                 # = 2 / (1 + e^(diff/scale)) - 1
        clear = np.where(np.isfinite(diff),
                         np.where(diff <= -MARGIN_S, 1.0, np.where(diff >= MARGIN_S, -1.0, 0.0)),
                         np.nan)
    cols = {"lb_valid": valid.astype(float), "lb_extrap": (extrap & valid).astype(float),
            "lb_tti_h": th, "lb_tti_a": ta}
    for name, arr in (("tti", diff), ("pfirst", pfirst), ("clear", clear)):
        for w, suffix in WINDOWS_S.items():
            cols["lb_{}_diff_{}".format(name, suffix)] = _centred_mean(arr, int(round(w * fps)))
    assert all(len(v) == n for v in cols.values())
    return cols
