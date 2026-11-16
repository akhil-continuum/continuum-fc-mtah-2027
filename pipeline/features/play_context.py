"""Play context around the ball (T-32 step 0), written into features.parquet after the loose-ball
columns. Team level only: a "nearest player" is whoever of the team is nearest on that frame; no
track slot is followed across frames. Windows are centred or come as before / after twins.

Every ``*_diff*`` column is home minus away (or an antisymmetric transform of it), so swapping the
teams and flipping the pitch negates it; the other columns are symmetric (the same for both teams)
and are meant as context. Columns (FEATURES.md, section Play context):

- Wider team-shape windows (T1, T3, T4): shape.py's statistics with centred windows of +-6 and
  +-8 s (``shape_<stat>_<diff|rel>_c6``, ``_c8``).
- Longer loose-ball paths (P30): loose_ball.py's arrival model with a 2 s and a 3 s ball path
  (``lb2_*``, ``lb3_*``).
- Touch counts (P1, P15): touch events of each team in the last and next 1, 2, 4 s, home minus
  away (``pc_touch_diff_b<w>`` = frames [t - w, t], ``pc_touch_diff_a<w>`` = (t, t + w]), and all
  touches of both teams within +-1 s (``pc_touch_n_c1``, symmetric).
- Players near the ball (P15): players of each team within 2, 5 and 10 m of the positioned ball,
  home minus away, on the frame and as a +-1 s mean; both teams within 5 m (``pc_crowd5``).
- Duel (P13, P18): the nearest player of each team, both within DUEL_BALL_M of the ball and within
  DUEL_PAIR_M of each other. ``pc_duel`` 0/1 and ``pc_duel_close`` (closing speed of the pair,
  m/s) are symmetric; ``pc_duel_lead_diff`` is away distance minus home distance to the ball
  inside a duel (positive: home nearer), 0 elsewhere, +-0.5 s mean.
- Where the ball vanished and reappeared (D12, T5, T6): inside a ball gap, the nearest-player
  distance difference (away minus home, clipped at +-CLIP_M) at the last positioned frame weighted
  by the share of the gap still to come, and at the next positioned frame weighted by the share
  already elapsed; and the same for the counts within 5 m at reappearance. 0 outside gaps.
- Defensive line (T2): per team, the mean distance from its own goal line of its 2nd to 5th
  deepest players (the deepest is usually the keeper), minus its median over the half, home minus
  away; and the mean velocity of those players along the team's attack direction, home minus away.
- Pitch zone and crowded set pieces (S, R): ``pc_zone`` = distance of the ball from halfway as a
  share of half the pitch length (symmetric); ``pc_box_crowd`` = players of both teams inside the
  penalty area nearer the ball (symmetric).
"""
import math

import numpy as np

from pipeline.features import loose_ball as LB
from pipeline.features import shape as SH

WIDE_WINDOWS_S = {6.0: "c6", 8.0: "c8"}
LB_PATHS_S = {2.0: "lb2", 3.0: "lb3"}
LB_WINDOWS_S = {0.5: "c05", 1.0: "c1"}
TOUCH_WINDOWS_S = (1, 2, 4)
NEAR_RADII_M = (2, 5, 10)
DUEL_BALL_M = 3.0
DUEL_PAIR_M = 2.0
CLIP_M = 10.0
LINE_RANKS = (1, 5)          # 2nd to 5th deepest (0-based ranks 1..4)
BOX_DEPTH_M = 16.5
BOX_HALF_WIDTH_M = 20.16
MIN_TEAM_PLAYERS = 6


def _cm(a, half):
    return SH._centred_mean(a, half)


def _half_ids(meta, n):
    half = np.zeros(n, dtype=np.int16)
    for h, (a, b) in enumerate(meta["halves"], 1):
        half[max(0, int(a)):min(n, int(b))] = h
    return half


def team_sign(meta, n):
    """(n, 2) attack direction (+1 / -1) of each team on each frame, NaN outside the halves or
    when unknown."""
    out = np.full((n, 2), np.nan)
    ad = meta.get("attack_direction") or {}
    for h, (a, b) in enumerate(meta["halves"], 1):
        for t in (0, 1):
            d = ad.get(str(h), {}).get(str(t))
            if d is not None:
                out[max(0, int(a)):min(n, int(b)), t] = float(d)
    return out


# --- loose ball with a longer path --------------------------------------------------------------

def _ball_estimate(ball, n, fps, path_s):
    """loose_ball.ball_estimate with the path length as an argument."""
    x = np.full(n, np.nan)
    y = np.full(n, np.nan)
    vx = np.zeros(n)
    vy = np.zeros(n)
    f = ball["frame"].to_numpy().astype(np.int64)
    keep = (f >= 0) & (f < n)
    f = f[keep]
    x[f] = ball["x_m"].to_numpy(float)[keep]
    y[f] = ball["y_m"].to_numpy(float)[keep]
    vx[f] = np.nan_to_num(ball["vx"].to_numpy(float)[keep])
    vy[f] = np.nan_to_num(ball["vy"].to_numpy(float)[keep])
    sp = np.hypot(vx, vy)
    scale = np.where(sp > LB.BALL_V_CAP, LB.BALL_V_CAP / np.maximum(sp, 1e-9), 1.0)
    vx, vy = vx * scale, vy * scale
    have = np.isfinite(x) & np.isfinite(y)
    idx = np.maximum.accumulate(np.where(have, np.arange(n), -1))
    j = np.clip(idx, 0, n - 1)
    dt = (np.arange(n) - idx) / fps
    reach = (idx >= 0) & (dt <= LB.EXTRAP_S + path_s + 1e-9)
    ex = np.where(reach, x[j] + vx[j] * dt, np.nan)
    ey = np.where(reach, y[j] + vy[j] * dt, np.nan)
    est = np.stack([np.where(have, x, ex), np.where(have, y, ey)], axis=1)
    start_ok = have | ((idx >= 0) & (dt <= LB.EXTRAP_S + 1e-9))
    return est, start_ok


def _team_times(est, start_ok, pl, fps, path_s):
    """loose_ball.team_times with the path length as an argument."""
    n = len(est)
    ks = np.arange(0, int(round(path_s * fps)) + 1, LB.STEP_FRAMES)
    s_k = ks / fps
    out = np.full(n, np.nan)
    chunk = max(1000, LB.CHUNK // max(1, len(ks) // 13))
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        fr = np.arange(a, b)
        idx = np.minimum(fr[:, None] + ks[None, :], n - 1)
        path = est[idx]
        path[(fr[:, None] + ks[None, :]) > n - 1] = np.nan
        q = pl[a:b, :, 0:2]
        d = np.linalg.norm(path[:, None, :, :] - q[:, :, None, :], axis=3)
        tt = np.maximum(np.maximum(0.0, d - LB.R_CONTROL_M) / LB.V_MAX_MPS, s_k[None, None, :])
        tt = np.where(np.isfinite(tt), tt, np.inf)
        best = tt.min(axis=(1, 2))
        out[a:b] = np.where(np.isfinite(best) & start_ok[a:b], best, np.nan)
    return out


def loose_ball_long(ball, pls, counts, n, fps):
    cols = {}
    scale = math.sqrt(3.0) * LB.TTI_SIGMA_S / math.pi
    for path_s, tag in LB_PATHS_S.items():
        est, start_ok = _ball_estimate(ball, n, fps, path_s)
        tti = [_team_times(est, start_ok, pls[k], fps, path_s) for k in (0, 1)]
        valid = (counts[0] >= LB.MIN_TEAM_PLAYERS) & (counts[1] >= LB.MIN_TEAM_PLAYERS) & \
            np.isfinite(tti[0]) & np.isfinite(tti[1])
        cap = LB.TTI_CAP_S + path_s - LB.PATH_S
        diff = np.where(valid, np.minimum(tti[0], cap) - np.minimum(tti[1], cap), np.nan)
        with np.errstate(invalid="ignore"):
            pfirst = np.tanh(-diff / (2.0 * scale))
        for name, arr in (("tti", diff), ("pfirst", pfirst)):
            for w, suffix in LB_WINDOWS_S.items():
                cols["{}_{}_diff_{}".format(tag, name, suffix)] = _cm(arr, int(round(w * fps)))
    return cols


# --- shape with wider windows ---------------------------------------------------------------------

def wide_shape(ball, players, meta):
    n = int(meta["n_frames"])
    fps = float(meta["fps"])
    st = SH.team_stats(ball, players, meta)
    valid = (st["count"][:, 0] >= SH.MIN_TEAM_PLAYERS) & (st["count"][:, 1] >= SH.MIN_TEAM_PLAYERS)
    half = _half_ids(meta, n)
    cols = {}
    for s in SH.STATS:
        diff = np.where(valid, st[s][:, 0] - st[s][:, 1], np.nan)
        series = {"diff": diff} if s not in SH.RELATIVE else {"rel": SH._half_centred(diff, half)}
        for kind, arr in series.items():
            for w, suffix in WIDE_WINDOWS_S.items():
                cols["shape_{}_{}_{}".format(s, kind, suffix)] = _cm(arr, int(round(w * fps)))
    return cols


# --- touches -------------------------------------------------------------------------------------

def touch_counts(ev, n, fps):
    cols = {}
    per = np.zeros((2, n + 1))
    if ev is not None and len(ev):
        t = ev[ev["kind"] == "touch"]
        f = t["frame"].to_numpy().astype(np.int64)
        tm = t["team"].to_numpy().astype(np.int64)
        for k in (0, 1):
            s = (tm == k) & (f >= 0) & (f < n)
            np.add.at(per[k], f[s] + 1, 1.0)
    cum = np.cumsum(per, axis=1)            # cum[:, i] = touches in frames < i
    i = np.arange(n)
    for w in TOUCH_WINDOWS_S:
        h = int(round(w * fps))
        lo, hi = np.maximum(i - h, 0), np.minimum(i + h, n - 1)
        before = cum[:, i + 1] - cum[:, lo]             # [t - h, t]
        after = cum[:, hi + 1] - cum[:, i + 1]          # (t, t + h]
        cols["pc_touch_diff_b{}".format(w)] = before[0] - before[1]
        cols["pc_touch_diff_a{}".format(w)] = after[0] - after[1]
    h = int(round(fps))
    lo, hi = np.maximum(i - h, 0), np.minimum(i + h, n - 1)
    cols["pc_touch_n_c1"] = (cum[:, hi + 1] - cum[:, lo]).sum(axis=0)
    return cols


# --- players around the ball ----------------------------------------------------------------------

def near_ball(ball, pls, counts, meta, n, fps):
    L, W = meta["pitch"]
    bx = np.full(n, np.nan)
    by = np.full(n, np.nan)
    f = ball["frame"].to_numpy().astype(np.int64)
    keep = (f >= 0) & (f < n)
    bx[f[keep]] = ball["x_m"].to_numpy(float)[keep]
    by[f[keep]] = ball["y_m"].to_numpy(float)[keep]
    have = np.isfinite(bx) & np.isfinite(by)
    cols = {}
    d = []
    for k in (0, 1):
        p = pls[k]
        d.append(np.hypot(p[:, :, 0] - bx[:, None], p[:, :, 1] - by[:, None]))   # (n, K)
    h1 = int(round(fps))
    for r in NEAR_RADII_M:
        cnt = [np.where(have, np.sum(dk < r, axis=1), np.nan) for dk in d]
        diff = cnt[0] - cnt[1]
        cols["pc_near{}_diff".format(r)] = diff
        cols["pc_near{}_diff_c1".format(r)] = _cm(diff, h1)
        if r == 5:
            cols["pc_crowd5"] = cnt[0] + cnt[1]
    # nearest player of each team
    dmin, jmin = [], []
    for dk in d:
        dd = np.where(np.isfinite(dk), dk, np.inf)
        j = np.argmin(dd, axis=1) if dd.shape[1] else np.zeros(n, int)
        m = dd[np.arange(n), j]
        dmin.append(np.where(np.isfinite(m) & have, m, np.nan))
        jmin.append(j)
    ph = pls[0][np.arange(n), jmin[0]]
    pa = pls[1][np.arange(n), jmin[1]]
    rel = pa[:, 0:2] - ph[:, 0:2]
    pair = np.hypot(rel[:, 0], rel[:, 1])
    with np.errstate(invalid="ignore"):
        duel = (dmin[0] < DUEL_BALL_M) & (dmin[1] < DUEL_BALL_M) & (pair < DUEL_PAIR_M)
        rv = np.nan_to_num(pa[:, 2:4]) - np.nan_to_num(ph[:, 2:4])
        close = -(rv[:, 0] * rel[:, 0] + rv[:, 1] * rel[:, 1]) / np.maximum(pair, 0.1)
    cols["pc_duel"] = np.where(have, duel.astype(float), np.nan)
    cols["pc_duel_close"] = np.where(duel, close, np.nan)
    lead = np.where(duel, np.clip(dmin[1] - dmin[0], -DUEL_BALL_M, DUEL_BALL_M), 0.0)
    cols["pc_duel_lead_diff_c05"] = _cm(np.where(have, lead, np.nan), int(round(0.5 * fps)))
    # where the ball vanished and reappeared
    with np.errstate(invalid="ignore"):
        ddiff = np.clip(dmin[1] - dmin[0], -CLIP_M, CLIP_M)
    n5 = np.where(have, np.sum(d[0] < 5, axis=1) - np.sum(d[1] < 5, axis=1), np.nan)
    last = np.maximum.accumulate(np.where(have, np.arange(n), -1))
    nxt = np.minimum.accumulate(np.where(have, np.arange(n), n)[::-1])[::-1]
    gap = ~have & (last >= 0) & (nxt < n)
    li, ni = np.clip(last, 0, n - 1), np.clip(nxt, 0, n - 1)
    span = np.maximum(nxt - last, 1)
    el = (np.arange(n) - last) / span
    cols["pc_gap_last_near_diff"] = np.where(gap, np.nan_to_num(ddiff[li]) * (1 - el), 0.0)
    cols["pc_gap_next_near_diff"] = np.where(gap, np.nan_to_num(ddiff[ni]) * el, 0.0)
    cols["pc_gap_next_n5_diff"] = np.where(gap, np.nan_to_num(n5[ni]) * el, 0.0)
    # pitch zone and crowded penalty area
    cols["pc_zone"] = np.where(have, np.abs(bx - L / 2.0) / (L / 2.0), np.nan)
    gx = np.where(bx < L / 2.0, 0.0, L)
    inbox = 0
    for k in (0, 1):
        p = pls[k]
        with np.errstate(invalid="ignore"):
            inside = (np.abs(p[:, :, 0] - gx[:, None]) <= BOX_DEPTH_M) & \
                (np.abs(p[:, :, 1] - W / 2.0) <= BOX_HALF_WIDTH_M)
        inbox = inbox + inside.sum(axis=1)
    cols["pc_box_crowd"] = np.where(have, inbox, np.nan)
    return cols


# --- defensive line ------------------------------------------------------------------------------

def defensive_line(pls, counts, meta, n, fps):
    L = meta["pitch"][0]
    sgn = team_sign(meta, n)
    half = _half_ids(meta, n)
    lines, vels = [], []
    for k in (0, 1):
        p = pls[k]
        s = sgn[:, k][:, None]
        depth = np.where(s > 0, p[:, :, 0], L - p[:, :, 0])            # from own goal line
        key = np.where(np.isfinite(depth), depth, np.inf)
        order = np.argsort(key, axis=1, kind="mergesort")[:, LINE_RANKS[0]:LINE_RANKS[1]]
        dsel = np.take_along_axis(depth, order, axis=1)
        vsel = np.take_along_axis(p[:, :, 2], order, axis=1) * s
        ok = (counts[k] >= MIN_TEAM_PLAYERS) & np.isfinite(sgn[:, k])
        with np.errstate(invalid="ignore"):
            lines.append(np.where(ok, np.nanmean(np.where(np.isfinite(dsel), dsel, np.nan), 1)
                                  if dsel.shape[1] else np.nan, np.nan))
            vels.append(np.where(ok, np.nanmean(vsel, 1) if vsel.shape[1] else np.nan, np.nan))
    cols = {}
    rel = SH._half_centred(lines[0] - lines[1], half)
    vd = vels[0] - vels[1]
    for w, suf in ((1.0, "c1"), (4.0, "c4")):
        cols["pc_line_rel_{}".format(suf)] = _cm(rel, int(round(w * fps)))
    for w, suf in ((1.0, "c1"), (2.0, "c2")):
        cols["pc_linevel_diff_{}".format(suf)] = _cm(vd, int(round(w * fps)))
    return cols


def build(ball, players, ev, meta):
    """Play-context columns for features.parquet (FEATURES.md, section Play context)."""
    n = int(meta["n_frames"])
    fps = float(meta["fps"])
    pls, counts = LB.dense_players(players, n)
    cols = {}
    cols.update(wide_shape(ball, players, meta))
    cols.update(loose_ball_long(ball, pls, counts, n, fps))
    cols.update(touch_counts(ev, n, fps))
    with np.errstate(invalid="ignore", divide="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cols.update(near_ball(ball, pls, counts, meta, n, fps))
            cols.update(defensive_line(pls, counts, meta, n, fps))
    assert all(len(v) == n for v in cols.values())
    return cols
