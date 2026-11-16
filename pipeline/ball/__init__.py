"""Ball stage: ``python -m pipeline.ball match|bench`` (see pipeline/ball/clean.py)."""
import os

import numpy as np

from pipeline.ball import clean, contact as contact_mod
from pipeline.ball.stage import read, read_meta

SPEED_CHECK_MIN_MPS = 3.0     # frames used to settle the unit of a raw speed channel
SPEED_CHECK_MAX_MPS = 30.0


def speed_channel_ratio(raw_speed, vx, vy, seen):
    """Median of raw speed / position speed on seen frames with clear motion (aggregate).

    About 3.6 means the channel is kph, about 1 means m/s.
    """
    sp = np.hypot(vx, vy)
    ok = seen & np.isfinite(raw_speed) & (sp >= SPEED_CHECK_MIN_MPS) & (sp <= SPEED_CHECK_MAX_MPS)
    if ok.sum() < 50:
        return None, int(ok.sum())
    return float(np.median(raw_speed[ok] / sp[ok])), int(ok.sum())


def speed_channel_diagnostics(raw_speed, x, y, seen, fps):
    """Aggregates settling the unit of a provider speed channel against position differences.

    Position speed: centred difference of the raw positions over +/-1 frame. Reports coverage,
    the share of zeros, quantiles of both, their correlation, and the median ratio at lags of
    -2..+2 frames (a lagged channel shows its best agreement off zero). Aggregates only.
    """
    n = len(x)
    ps = np.full(n, np.nan)
    ps[1:-1] = np.hypot(x[2:] - x[:-2], y[2:] - y[:-2]) * fps / 2.0
    fin = np.isfinite(raw_speed)
    out = {"raw_s_finite_share_of_seen": float(fin[seen].mean()) if seen.any() else None,
           "raw_s_finite_share_of_unseen": float(fin[~seen].mean()) if (~seen).any() else None,
           "raw_s_zero_share_of_finite": float((raw_speed[fin] == 0).mean()) if fin.any()
           else None}
    q = [5, 25, 50, 75, 95]
    both = fin & np.isfinite(ps) & seen
    if both.sum() >= 50:
        out["raw_s_quantiles_5_25_50_75_95"] = [round(float(v), 3)
                                                for v in np.percentile(raw_speed[both], q)]
        out["pos_speed_mps_quantiles_5_25_50_75_95"] = [round(float(v), 3)
                                                        for v in np.percentile(ps[both], q)]
        out["raw_s_pos_speed_pearson"] = round(float(np.corrcoef(raw_speed[both],
                                                                  ps[both])[0, 1]), 4)
        lag_ratio = {}
        for lag in (-2, -1, 0, 1, 2):
            sh = np.full(n, np.nan)
            if lag >= 0:
                sh[:n - lag] = ps[lag:]
            else:
                sh[-lag:] = ps[:n + lag]
            ok = fin & np.isfinite(sh) & (sh >= SPEED_CHECK_MIN_MPS) & (sh <= SPEED_CHECK_MAX_MPS)
            lag_ratio[str(lag)] = (round(float(np.median(raw_speed[ok] / sh[ok])), 4)
                                   if ok.sum() >= 50 else None)
        out["raw_s_over_pos_speed_median_by_lag"] = lag_ratio
        moving = both & (ps >= SPEED_CHECK_MIN_MPS)
        out["raw_s_frames_pos_moving"] = int(moving.sum())
    return out


def run(root):
    meta = read_meta(root)
    raw = read(root, "ball_raw.parquet")
    players = read(root, "players.parquet")
    n = int(meta["n_frames"])
    bx = np.full(n, np.nan)
    by = np.full(n, np.nan)
    f = raw["frame"].to_numpy()
    bx[f] = raw["x_m"].to_numpy(dtype=float)
    by[f] = raw["y_m"].to_numpy(dtype=float)
    near = contact_mod.contact(players, bx, by, n).d_any
    ball, stats = clean.clean_ball(raw, meta, near_player_m=near)
    ball.to_parquet(os.path.join(root, "ball.parquet"), index=False)

    in_half = np.zeros(n, dtype=bool)
    for a, b in meta["halves"]:
        in_half[max(0, a):min(n, b)] = True
    have = (ball["seen"] | ball["filled"]).to_numpy()
    edges, counts = clean.gap_histogram(have | ~in_half)
    stats["gap_hist_edges_frames"] = edges
    stats["gap_hist_counts"] = counts
    stats["share_seen_in_halves"] = float(ball["seen"].to_numpy()[in_half].mean())
    stats["share_filled_in_halves"] = float(ball["filled"].to_numpy()[in_half].mean())
    stats["share_missing_in_halves"] = float((~have[in_half]).mean())
    sp = np.hypot(ball["vx"].to_numpy(dtype=float), ball["vy"].to_numpy(dtype=float))
    sp_edges = [0, 1, 2, 4, 8, 12, 16, 20, 25, 30, 40]
    stats["speed_hist_edges_mps"] = sp_edges
    stats["speed_hist_counts"] = [int(v) for v in
                                  np.histogram(sp[np.isfinite(sp)], bins=sp_edges + [1e9])[0]]
    if "raw_s" in raw.columns:
        rs = np.full(n, np.nan)
        rs[f] = raw["raw_s"].to_numpy(dtype=float)
        ratio, used = speed_channel_ratio(rs, ball["vx"].to_numpy(dtype=float),
                                          ball["vy"].to_numpy(dtype=float),
                                          ball["seen"].to_numpy())
        stats["raw_s_over_mps_median"] = ratio
        stats["raw_s_check_frames"] = used
        stats.update(speed_channel_diagnostics(rs, bx, by, np.isfinite(bx) & np.isfinite(by),
                                               float(meta["fps"])))
    return stats
