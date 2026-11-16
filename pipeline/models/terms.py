"""Per-frame evidence terms for the rule scorer. Team level only; windows centred.

``dead_terms(...)`` and ``team_terms(...)`` return dicts of float arrays (n_frames,), one per
weight name in pipeline/models/rule_params.py. Dead terms lean towards D when positive; team
terms are + for home (team 0) and - for away (team 1). Every term is bounded (|term| <= 1) and is
0 where its evidence is absent, so a frame with no evidence gets the flat prior.

Inputs are the contract files only (CONTRACTS.md): ball.parquet, players.parquet,
events.parquet, features.parquet, meta.json and, when present, features_dead.parquet.
"""
import numpy as np
import pandas as pd

from pipeline.ball.contact import contact
from pipeline.events import base

SIGN = {0: 1.0, 1: -1.0}


# --- small helpers ---------------------------------------------------------------------------

def _window_sum(v, half):
    """Sum of v over the centred window [i - half, i + half], clipped at the ends."""
    n = len(v)
    c = np.concatenate([[0.0], np.cumsum(v)])
    i = np.arange(n)
    return c[np.minimum(i + half + 1, n)] - c[np.maximum(i - half, 0)]


def centred_mean(a, half):
    """Centred moving mean over +-half frames ignoring NaN; NaN where no finite value."""
    a = np.asarray(a, dtype=float)
    if half <= 0:
        return a.copy()
    ok = np.isfinite(a)
    num = _window_sum(np.where(ok, a, 0.0), half)
    den = _window_sum(ok.astype(float), half)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def centred_share(flag, half):
    """Share of frames with ``flag`` in the centred +-half window (window clipped at the ends)."""
    flag = np.asarray(flag, dtype=float)
    return _window_sum(flag, half) / _window_sum(np.ones_like(flag), half)


def ffill_index(mask):
    idx = np.where(mask, np.arange(len(mask)), -1)
    return np.maximum.accumulate(idx)


def bfill_index(mask):
    n = len(mask)
    idx = np.where(mask, np.arange(n), n)
    return np.minimum.accumulate(idx[::-1])[::-1]


def half_index(meta, n):
    half = np.zeros(n, dtype=np.int16)
    for h, (a, b) in enumerate(meta["halves"], 1):
        half[max(0, int(a)):min(n, int(b))] = h
    return half


def attack_home(meta, players, n):
    """+1 where home attacks towards increasing x, -1 otherwise, per frame (0 outside halves).

    From meta["attack_direction"] when present; otherwise (bench copies without it) from which
    team stands deeper towards x = 0 on average over the half (S7: ends swap at half-time)."""
    half = half_index(meta, n)
    out = np.zeros(n)
    ad = meta.get("attack_direction") or {}
    f = players["frame"].to_numpy()
    t = players["team"].to_numpy()
    x = players["x_m"].to_numpy(dtype=float)
    for h, (a, b) in enumerate(meta["halves"], 1):
        d = ad.get(str(h), {}).get("0")
        if d is None:
            sel = (f >= a) & (f < b)
            m0 = np.nanmean(x[sel & (t == 0)]) if np.any(sel & (t == 0)) else np.nan
            m1 = np.nanmean(x[sel & (t == 1)]) if np.any(sel & (t == 1)) else np.nan
            d = 0 if not (np.isfinite(m0) and np.isfinite(m1)) else (1 if m0 < m1 else -1)
        out[half == h] = float(d)
    return out


def _per_frame(n, frames, values, how="mean"):
    s = pd.Series(values).groupby(frames)
    agg = getattr(s, how)()
    out = np.full(n, np.nan)
    idx = agg.index.to_numpy()
    keep = (idx >= 0) & (idx < n)
    out[idx[keep]] = agg.to_numpy()[keep]
    return out


# --- player summaries ------------------------------------------------------------------------

def player_summaries(players, n):
    """Per frame: mean squared player speed, tracked count, per-team median position and spread."""
    f = players["frame"].to_numpy()
    t = players["team"].to_numpy().astype(int)
    x = players["x_m"].to_numpy(dtype=float)
    y = players["y_m"].to_numpy(dtype=float)
    s2 = players["vx"].to_numpy(dtype=float) ** 2 + players["vy"].to_numpy(dtype=float) ** 2
    out = {"ms": _per_frame(n, f, s2, "mean"),
           "n_players": np.bincount(f[(f >= 0) & (f < n)], minlength=n).astype(float)}
    for team in (0, 1):
        sel = t == team
        ft, xt, yt = f[sel], x[sel], y[sel]
        mx = _per_frame(n, ft, xt, "median")
        my = _per_frame(n, ft, yt, "median")
        cnt = np.bincount(ft[(ft >= 0) & (ft < n)], minlength=n)
        d = np.hypot(xt - mx[np.clip(ft, 0, n - 1)], yt - my[np.clip(ft, 0, n - 1)])
        spread = _per_frame(n, ft, d, "median")
        out["cx_{}".format(team)] = mx
        out["spread_{}".format(team)] = np.where(cnt >= 5, spread, np.nan)
    return out


def corrected_speed(ms, fps, half_s, floor_q):
    """Centred RMS player speed with a per-game noise floor removed (jitter adds a constant to
    the mean squared speed; the floor is a low percentile of the smoothed RMS, measured by code
    on each game)."""
    msc = centred_mean(ms, int(round(half_s * fps)))
    rms = np.sqrt(msc)
    fin = rms[np.isfinite(rms)]
    floor = float(np.percentile(fin, floor_q)) if len(fin) else 0.0
    corr = np.sqrt(np.clip(msc - floor ** 2, 0.0, None))
    return corr, floor


# --- dead versus live ------------------------------------------------------------------------

def outside(x, y, pitch, margin):
    L, W = pitch
    with np.errstate(invalid="ignore"):
        return (x < -margin) | (x > L + margin) | (y < -margin) | (y > W + margin)


def exit_span(ev, seen, out_flag, half, n):
    """D1-D3: max exit conf on frames from each exit to the ball's next seen frame inside the
    pitch (same half)."""
    term = np.zeros(n)
    ex = ev[ev["kind"] == "exit"]
    if not len(ex):
        return term
    inside_seen = seen & ~out_flag
    nxt = bfill_index(inside_seen)
    for e, c in zip(ex["frame"].to_numpy(), ex["conf"].to_numpy(dtype=float)):
        e = int(e)
        if e < 0 or e >= n:
            continue
        stop = int(nxt[e])
        h = half[e]
        if h > 0:
            last = int(np.flatnonzero(half == h)[-1]) + 1
            stop = min(stop, last)
        c = float(c) if np.isfinite(c) and c > 0 else 1.0
        term[e:stop] = np.maximum(term[e:stop], c)
    return term


def dead_terms(ball, players, ev, feat, meta, dead=None, shape=None, ps=None):
    from pipeline.models.rule_params import SHAPE
    sh = dict(SHAPE, **(shape or {}))
    n = int(meta["n_frames"])
    fps = float(meta["fps"])
    pitch = tuple(meta["pitch"])
    half = half_index(meta, n)
    seen = ball["seen"].to_numpy(bool)
    filled = ball["filled"].to_numpy(bool)
    x, y = ball["x_m"].to_numpy(float), ball["y_m"].to_numpy(float)
    bspeed = np.hypot(ball["vx"].to_numpy(float), ball["vy"].to_numpy(float))
    miss = (~seen & ~filled).astype(float)
    T = {}

    # D12 / D13: missingness in context.
    gap = np.nan_to_num(feat["ball_gap_len_s"].to_numpy(float), nan=0.0)
    c = sh["gap_len_clip"]
    with np.errstate(divide="ignore", invalid="ignore"):
        ln = np.clip(np.log2(np.maximum(gap, 1.0 / fps) / sh["gap_mid_s"]), -c, c) / c
    T["miss"] = miss
    T["miss_len"] = miss * ln
    dl = feat["ball_last_seen_dist_line_m"].to_numpy(float)
    osp = feat["ball_last_seen_out_speed_mps"].to_numpy(float)
    with np.errstate(invalid="ignore"):
        near_line = (dl <= sh["line_near_m"]).astype(float)
        heading = np.clip(np.nan_to_num(osp, nan=0.0) / sh["line_out_mps"], 0.0, 1.0)
    T["miss_line"] = miss * near_line * heading
    crowd = (np.nan_to_num(feat["ball_loss_players_3m_h"].to_numpy(float), nan=0.0)
             + np.nan_to_num(feat["ball_loss_players_3m_a"].to_numpy(float), nan=0.0))
    T["miss_crowd"] = miss * np.minimum(crowd, sh["crowd_full"]) / sh["crowd_full"]

    # D1 / D2: seen outside the lines, with persistence; exits until the ball is back inside.
    out_flag = seen & outside(x, y, pitch, sh["out_margin_m"])
    T["out"] = centred_share(out_flag, int(sh["out_half_frames"]))
    T["exit"] = exit_span(ev, seen, out_flag, half, n)

    # S4 / R6 from the restart stage.
    if dead is not None and len(dead) == n:
        T["pre_kickoff"] = dead["mask_pre_kickoff"].to_numpy(bool).astype(float)
        T["form_kickoff"] = np.clip(np.nan_to_num(dead["form_kick_off"].to_numpy(float)), 0, 1)
    else:
        T["pre_kickoff"] = np.zeros(n)
        T["form_kickoff"] = np.zeros(n)

    # D5 / D10 / T9: collective slowness and a dip against the surrounding seconds.
    ps = ps if ps is not None else player_summaries(players, n)
    speed, _floor = corrected_speed(ps["ms"], fps, sh["speed_half_s"], sh["speed_floor_q"])
    slow = np.clip((sh["speed_ref_mps"] - speed) / sh["speed_scale_mps"], -2.0, 2.0) / 2.0
    T["slow"] = np.nan_to_num(slow, nan=0.0)
    ctx = centred_mean(speed, int(round(sh["dip_ctx_s"] * fps)))
    dip = np.clip((ctx - speed) / sh["dip_scale_mps"], 0.0, 1.0)
    T["slow_dip"] = np.nan_to_num(dip, nan=0.0)

    # D9 / R2-R5 (trap P23): stationary seen ball; live: seen ball moving.
    with np.errstate(invalid="ignore"):
        still = seen & (bspeed < sh["still_mps"])
        moving = seen & (bspeed > sh["moving_mps"])
    T["still"] = centred_share(still, int(round(sh["still_half_s"] * fps)))
    T["moving"] = centred_share(moving, int(round(sh["moving_half_s"] * fps)))

    # D10: few tracked players.
    T["players_low"] = np.clip((sh["players_full"] - ps["n_players"]) / sh["players_span"],
                               0.0, 1.0)
    for k in T:
        T[k] = np.where(half > 0, T[k], 0.0).astype(float)
    return T


# --- which team ------------------------------------------------------------------------------

def _decay_last(frames, signs, n, half, tau_frames, forward=False):
    """sign of the last (or next) event in the same half times exp(-age / tau)."""
    mark = np.zeros(n, dtype=bool)
    val = np.zeros(n)
    frames = np.asarray(frames, dtype=np.int64)
    signs = np.asarray(signs, dtype=float)
    ok = (frames >= 0) & (frames < n)
    order = np.argsort(frames[ok], kind="mergesort")
    for f, s in zip(frames[ok][order], signs[ok][order]):
        mark[f] = True
        val[f] = s
    idx = bfill_index(mark) if forward else ffill_index(mark)
    have = (idx < n) if forward else (idx >= 0)
    j = np.clip(idx, 0, n - 1)
    same_half = have & (half[j] == half) & (half > 0)
    age = np.abs(np.arange(n) - j)
    return np.where(same_half, val[j] * np.exp(-age / tau_frames), 0.0)


def _spans(rows, n):
    term = np.zeros(n)
    for f0, f1, s in rows:
        f0, f1 = max(0, int(f0)), min(n - 1, int(f1))
        if f1 >= f0:
            term[f0:f1 + 1] += s
    return np.clip(term, -1.0, 1.0)


def team_terms(ball, players, ev, feat, meta, shape=None, ps=None):
    from pipeline.models.rule_params import SHAPE
    sh = dict(SHAPE, **(shape or {}))
    n = int(meta["n_frames"])
    fps = float(meta["fps"])
    L = float(meta["pitch"][0])
    half = half_index(meta, n)
    T = {}
    teamed = ev[ev["team"].isin([0, 1])]
    sign = teamed["team"].map(SIGN).to_numpy(dtype=float)

    # P1-P5: control ownership, last and next (first-touch dated).
    sel = (teamed["kind"] == "control").to_numpy()
    cf = teamed["first_touch_frame"].to_numpy()[sel]
    cf = np.where(cf >= 0, cf, teamed["frame"].to_numpy()[sel])
    T["ctrl_last"] = _decay_last(cf, sign[sel], n, half, sh["tau_ctrl_s"] * fps)
    T["ctrl_next"] = _decay_last(cf, sign[sel], n, half, sh["tau_ctrl_next_s"] * fps,
                                 forward=True)
    # P1 carry and P2 pass spans.
    for kind in ("carry", "pass"):
        s = (teamed["kind"] == kind).to_numpy()
        T[kind] = _spans(zip(teamed["frame"].to_numpy()[s], teamed["end_frame"].to_numpy()[s],
                             sign[s]), n)
    # B4 last team touch.
    s = (teamed["kind"] == "touch").to_numpy()
    T["touch_last"] = _decay_last(teamed["frame"].to_numpy()[s], sign[s], n, half,
                                  sh["tau_touch_s"] * fps)

    # P13 nearest-player team on the measured ball-player alignment, outside flights (B5).
    seen = ball["seen"].to_numpy(bool)
    x, y = ball["x_m"].to_numpy(float), ball["y_m"].to_numpy(float)
    lag, _d, _d0, _scale = base.estimate_alignment(players, x, y, seen, n)
    xs, ys = np.where(seen, x, np.nan), np.where(seen, y, np.nan)
    con = contact(base._shift(players, lag), xs, ys, n)
    dh, da = con.d[0], con.d[1]
    flight = np.nan_to_num(feat["in_flight"].to_numpy(float), nan=0.0) > 0
    with np.errstate(invalid="ignore"):
        ok = seen & ~flight & (np.fmin(dh, da) <= sh["near_max_m"]) & np.isfinite(dh) \
            & np.isfinite(da)
        raw = np.where(ok, np.tanh((da - dh) / sh["near_scale_m"]), np.nan)
    T["near"] = np.nan_to_num(centred_mean(raw, int(round(sh["near_half_s"] * fps))), nan=0.0)

    # T1 / T2 shape: territory and spread.
    ps = ps if ps is not None else player_summaries(players, n)
    dir_h = attack_home(meta, players, n)
    terr = ((ps["cx_0"] + ps["cx_1"]) / 2.0 - L / 2.0) / (L / 2.0) * dir_h
    w = int(round(sh["shape_half_s"] * fps))
    T["territory"] = np.clip(np.nan_to_num(centred_mean(terr, w), nan=0.0), -1.0, 1.0)
    spr = (ps["spread_0"] - ps["spread_1"]) / sh["spread_scale_m"]
    T["spread"] = np.clip(np.nan_to_num(centred_mean(spr, w), nan=0.0), -1.0, 1.0)
    for k in T:
        T[k] = np.where(half > 0, T[k], 0.0).astype(float)
    return T, {"align_lag_frames": int(lag)}
