"""Features stage: ``python -m pipeline.features match|bench`` -> <root>/features.parquet.

Reads ball.parquet, events.parquet, players.parquet, meta.json. One row per frame. Every column
is documented in pipeline/features/FEATURES.md (the two must agree; a test checks it). Windows
are centred; anything that measures time since something also has its time-until twin.

Implemented so far: ball state and ball-missing context (D12, B3), contact per team (P1, P13,
P15), flight and control timing (B5, P4), tracked-player counts (T7), and team shape (T1, T3,
T4, T7; pipeline/features/shape.py, appended columns), and loose-ball evidence (P30, P-14;
pipeline/features/loose_ball.py, appended after the shape columns). T2, T5, T6, T9 are not built
yet.
"""
import os

import numpy as np
import pandas as pd

from pipeline.ball.clean import runs
from pipeline.ball.contact import contact
from pipeline.ball.stage import read, read_meta

LOSS_RADIUS_M = 3.0
WINDOWS_S = {0.5: "c05", 1.0: "c1"}     # centred half-widths in seconds -> column suffix
TEAMS = (("h", 0), ("a", 1))
FLOAT = np.float32


def _centred_mean(a, half):
    """Centred moving mean over +/-half frames ignoring NaN; NaN where no finite value."""
    v = np.where(np.isfinite(a), a, 0.0)
    m = np.isfinite(a).astype(float)
    k = np.ones(2 * half + 1)
    num = np.convolve(v, k, mode="same")
    den = np.convolve(m, k, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _ffill_index(mask):
    """Index of the last True at or before each frame (-1 if none)."""
    idx = np.where(mask, np.arange(len(mask)), -1)
    return np.maximum.accumulate(idx)


def _bfill_index(mask):
    """Index of the first True at or after each frame (n if none)."""
    n = len(mask)
    idx = np.where(mask, np.arange(n), n)
    return np.minimum.accumulate(idx[::-1])[::-1]


def missing_context(ball, players, meta):
    """D12: ball-missing context per frame (see FEATURES.md)."""
    n = int(meta["n_frames"])
    fps = float(meta["fps"])
    L, W = meta["pitch"]
    seen = ball["seen"].to_numpy(bool)
    have = seen | ball["filled"].to_numpy(bool)
    x, y = ball["x_m"].to_numpy(float), ball["y_m"].to_numpy(float)
    vx, vy = ball["vx"].to_numpy(float), ball["vy"].to_numpy(float)
    out = {}
    gap_len = np.zeros(n, dtype=np.int32)
    for a, b in zip(*runs(~have)):
        gap_len[a:b] = b - a
    out["ball_gap_len_s"] = gap_len / fps
    last = _ffill_index(seen)
    nxt = _bfill_index(seen)
    frames = np.arange(n)
    out["ball_t_since_seen_s"] = np.where(last >= 0, (frames - last) / fps, np.nan)
    out["ball_t_until_seen_s"] = np.where(nxt < n, (nxt - frames) / fps, np.nan)
    li = np.clip(last, 0, n - 1)
    ok = last >= 0
    for name, arr in (("x", x), ("y", y), ("vx", vx), ("vy", vy)):
        out["ball_last_seen_{}".format(name)] = np.where(ok, arr[li], np.nan)
    ni = np.clip(nxt, 0, n - 1)
    okn = nxt < n
    out["ball_next_seen_x"] = np.where(okn, x[ni], np.nan)
    out["ball_next_seen_y"] = np.where(okn, y[ni], np.nan)
    lx, ly = out["ball_last_seen_x"], out["ball_last_seen_y"]
    lvx, lvy = out["ball_last_seen_vx"], out["ball_last_seen_vy"]
    dists = np.stack([lx, L - lx, ly, W - ly])
    outward = np.stack([-lvx, lvx, -lvy, lvy])
    with np.errstate(invalid="ignore"):
        k = np.argmin(np.where(np.isfinite(dists), dists, np.inf), axis=0)
    out["ball_last_seen_dist_line_m"] = np.take_along_axis(dists, k[None], 0)[0]
    out["ball_last_seen_out_speed_mps"] = np.take_along_axis(outward, k[None], 0)[0]
    # Players of each team within LOSS_RADIUS_M of the ball on the last seen frame.
    con = contact(players, x, y, n)
    for tag, t in TEAMS:
        c = con.count[LOSS_RADIUS_M][t].astype(float)
        out["ball_loss_players_3m_{}".format(tag)] = np.where(ok, c[li], np.nan)
    return out, con


def contact_features(ball, con, fps):
    """Contact per team (P1, P13, P15) and centred window means."""
    out = {}
    seen = ball["seen"].to_numpy(bool)
    vx, vy = ball["vx"].to_numpy(float), ball["vy"].to_numpy(float)
    for tag, t in TEAMS:
        d = np.where(seen, con.d[t], np.nan)
        out["near_d_{}".format(tag)] = d
        for r in (1.0, 2.0, 5.0):
            out["near_n{}_{}".format(int(r), tag)] = np.where(seen, con.count[r][t], np.nan)
        agree = np.hypot(vx - con.pvx[t], vy - con.pvy[t])
        out["near_vel_diff_{}".format(tag)] = np.where(seen, agree, np.nan)
        out["n_players_{}".format(tag)] = con.n_players[t].astype(float)
        for w, suffix in WINDOWS_S.items():
            half = int(round(w * fps))
            out["near_d_{}_{}".format(tag, suffix)] = _centred_mean(d, half)
            out["near_n2_{}_{}".format(tag, suffix)] = _centred_mean(
                out["near_n2_{}".format(tag)], half)
    out["crowd_3m"] = np.where(seen, con.count[3.0][0] + con.count[3.0][1], np.nan)
    out["near_d_diff"] = out["near_d_h"] - out["near_d_a"]
    return out


def event_features(ev, n, fps):
    """Flight flag and time since / until each team's control (both directions)."""
    out = {}
    flight = np.zeros(n, dtype=bool)
    for _, r in ev[ev["kind"] == "flight"].iterrows():
        flight[int(r["frame"]):int(r["end_frame"]) + 1] = True
    out["in_flight"] = flight.astype(float)
    frames = np.arange(n)
    ctrl = ev[ev["kind"] == "control"]
    for tag, t in TEAMS:
        mark = np.zeros(n, dtype=bool)
        f = ctrl.loc[ctrl["team"] == t, "first_touch_frame"].to_numpy()
        mark[f[(f >= 0) & (f < n)]] = True
        last, nxt = _ffill_index(mark), _bfill_index(mark)
        out["t_since_control_{}_s".format(tag)] = np.where(last >= 0, (frames - last) / fps,
                                                           np.nan)
        out["t_until_control_{}_s".format(tag)] = np.where(nxt < n, (nxt - frames) / fps,
                                                           np.nan)
    ex = np.zeros(n, dtype=bool)
    f = ev.loc[ev["kind"] == "exit", "frame"].to_numpy()
    ex[f[(f >= 0) & (f < n)]] = True
    last, nxt = _ffill_index(ex), _bfill_index(ex)
    out["t_since_exit_s"] = np.where(last >= 0, (frames - last) / fps, np.nan)
    out["t_until_exit_s"] = np.where(nxt < n, (nxt - frames) / fps, np.nan)
    return out


def build(ball, players, ev, meta):
    n = int(meta["n_frames"])
    fps = float(meta["fps"])
    cols = {"frame": np.arange(n, dtype=np.int32),
            "ball_seen": ball["seen"].to_numpy(bool).astype(float),
            "ball_filled": ball["filled"].to_numpy(bool).astype(float),
            "ball_conf": ball["conf"].to_numpy(float),
            "ball_speed": np.hypot(ball["vx"].to_numpy(float), ball["vy"].to_numpy(float))}
    miss, con = missing_context(ball, players, meta)
    cols.update(miss)
    cols.update(contact_features(ball, con, fps))
    cols.update(event_features(ev, n, fps))
    from pipeline.features import shape          # T1-T4, T7 team shape (appended columns)
    cols.update(shape.build(ball, players, meta))
    from pipeline.features import loose_ball     # P30 / P-14 loose-ball evidence (appended)
    cols.update(loose_ball.build(ball, players, meta))
    from pipeline.features import play_context   # T-32 step 0: play context (appended)
    cols.update(play_context.build(ball, players, ev, meta))
    from pipeline.features import players_near_ball  # T-32 step 1: K nearest per team
    cols.update(players_near_ball.build(ball, players, meta))
    df = pd.DataFrame(cols)
    for c in df.columns:
        if c != "frame":
            df[c] = df[c].astype(FLOAT)
    return df


def run(root):
    meta = read_meta(root)
    ball = read(root, "ball.parquet")
    players = read(root, "players.parquet")
    ev = read(root, "events.parquet")
    df = build(ball, players, ev, meta)
    df.to_parquet(os.path.join(root, "features.parquet"), index=False)
    nan_share = df.drop(columns="frame").isna().mean()
    return {"n_frames": int(len(df)), "n_features": int(df.shape[1] - 1),
            "nan_share_mean": round(float(nan_share.mean()), 4),
            "nan_share_max": round(float(nan_share.max()), 4),
            "in_flight_share": round(float(df["in_flight"].mean()), 4),
            "gap_len_s_median_of_missing": round(float(np.nanmedian(
                df.loc[df["ball_gap_len_s"] > 0, "ball_gap_len_s"])), 2)
            if (df["ball_gap_len_s"] > 0).any() else None}
