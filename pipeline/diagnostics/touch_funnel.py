"""Touch funnel: where ball-velocity changes are lost on their way to touch events (P-17).

    uv run python -m pipeline.diagnostics.touch_funnel match
    uv run python -m pipeline.diagnostics.touch_funnel bench --root artifacts/bench/metrica_g3_match ...

Runs the same code on a bench root and on the match. Reads ``ball.parquet`` (ball stage),
``ball_raw.parquet``, ``players.parquet`` and ``meta.json``; never labels, truth or scores.
Writes reports/touch_funnel.match.json or reports/touch_funnel.bench.<root id>.json through
``pipeline.report.summary``. Everything printed is a whole-game rate, share, count, median or a
histogram of at most 12 bins; no minimum, maximum or tail quantile, nothing that locates an event.

The funnel mirrors ``pipeline.events.touch.touches`` step by step (same Inputs, same alignment
and radius scale as the events stage) and checks that it reproduces the detector's touch count:

  F0 graded frames -> seen -> velocity change valid (>= MIN_PTS seen frames each side, same half)
  F1 sharp: dv above the game's threshold max(DV_MPS, NOISE_K * noise scale), or heading change
     above ANGLE_DEG while both sides move fast enough
  F2 one peak per +-max(PEAK_HALF, W) frames (non-maximum suppression)
  F3 player test: nearest player (window min over +-1 frame, players shifted by the game's lag)
     within RADIUS_M * radius_scale -> touch, team of the nearest player

Beside the funnel: candidate peaks on a fixed physical ladder of dv thresholds; the dv and
acceleration distributions (is the track low-pass filtered?); the share of candidates with a
player within fixed radii at lag 0, at the game's lag and at the share-maximising lag in -10..10;
seen-run structure around candidates; velocity changes across short gaps (touches hidden in a
gap); per-team rates.

Rates are per minute of graded time and per minute of graded seen-ball time. Graded frames are
``meta["graded_spans"]`` when present, else ``meta["halves"]`` (the match, and clean bench roots).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from pipeline.ball.contact import contact
from pipeline.events import base, touch
from pipeline.io import sources
from pipeline.io.roots import require_bench_roots
from pipeline.report import match_entry, quiet_warnings, summary

# Params (diagnostic only; nothing here feeds a detector)
LADDER_MPS = (2.0, 3.0, 4.0, 6.0, 8.0)          # fixed physical dv thresholds, m/s over W frames
RADII_M = (0.5, 1.0, 1.5, 2.0, 3.0)
LAGS = tuple(range(-10, 11))                    # players at f + L against the ball at f
BEST_LAG_RADIUS_M = 1.0                         # the lag search maximises the share within this
SHORT_W = 3                                     # short-window dv for the kink sharpness ratio
SHORT_MIN_PTS = 3
SHARP_REF_MPS = 4.0                             # peaks used for sharpness and width statistics
GAP_EDGE_NEAR = 5                               # frames
GAP_KINK_MAX = 25                               # longest gap examined for an across-gap kink
GAP_FIT = 7                                     # frames each side of a gap for its velocities
GAP_FIT_MIN = 4
DV_EDGES = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 20, 1e9]             # m/s, 11 bins
PEAK_EDGES = [2, 3, 4, 5, 6, 8, 10, 13, 17, 25, 1e9]               # m/s, 10 bins
ACC_EDGES = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1e9]           # m/s^2, 10 bins
WIDTH_EDGES = [0, 3, 5, 7, 9, 11, 13, 16, 20, 1e9]                 # frames, 9 bins
EDGE_EDGES = [1, 2, 4, 6, 11, 26, 1e9]                             # frames to a gap, 6 bins


# ---------------------------------------------------------------- loading

def _graded_mask(meta, n):
    spans = meta.get("graded_spans") or meta["halves"]
    g = np.zeros(n, dtype=bool)
    for a, b in spans:
        g[max(0, int(a)):min(n, int(b))] = True
    return g


def load(root):
    with open(os.path.join(root, "meta.json")) as fh:
        meta = json.load(fh)
    ball = pd.read_parquet(os.path.join(root, "ball.parquet"))
    raw = pd.read_parquet(os.path.join(root, "ball_raw.parquet"))
    players = pd.read_parquet(os.path.join(root, "players.parquet"),
                              columns=["frame", "team", "x_m", "y_m", "vx", "vy"])
    return meta, ball, raw, players


# ---------------------------------------------------------------- helpers

def _peaks(score, sharp, half):
    """Greedy non-maximum suppression exactly as touch.touches does it."""
    idx = np.flatnonzero(sharp)
    taken = np.zeros(len(score), dtype=bool)
    out = []
    for i in idx[np.argsort(-score[idx], kind="stable")]:
        lo, hi = max(0, i - half), min(len(score), i + half + 1)
        if taken[lo:hi].any():
            continue
        taken[i] = True
        out.append(i)
    return np.array(sorted(out), dtype=np.int64)


def _hist(values, edges):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return [int(c) for c in np.histogram(values, edges)[0]]


def _share_hist(values, edges):
    c = np.array(_hist(values, edges), dtype=float)
    return [round(float(v), 4) for v in c / max(c.sum(), 1.0)]


def _med(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return round(float(np.median(values)), 4) if len(values) >= 20 else None


def _share(mask):
    mask = np.asarray(mask)
    return round(float(mask.mean()), 4) if len(mask) else None


def _per_min(count, minutes):
    return round(float(count) / minutes, 3) if minutes > 0 else None


def _shift(players, lag):
    if lag == 0:
        return players
    p = players.copy()
    p["frame"] = p["frame"] - lag
    return p


def distances_by_lag(players, x, y, frames, n, lags=LAGS):
    """{lag: (d_team0, d_team1)} at ``frames``, each a window min over +-1 frame (as the detector)."""
    keep = np.zeros(n, dtype=bool)
    for k in (-1, 0, 1):
        f = frames + k
        keep[f[(f >= 0) & (f < n)]] = True
    xs, ys = np.where(keep, x, np.nan), np.where(keep, y, np.nan)
    # Only player rows that can meet a kept frame at some lag.
    pf = players["frame"].to_numpy()
    off = max(abs(l) for l in lags) + 2
    near = np.convolve(keep.astype(float), np.ones(2 * off + 1), mode="same") > 0
    inside = (pf >= 0) & (pf < n)
    sel = inside.copy()
    sel[inside] = near[pf[inside]]
    sub = players[sel]
    out = {}
    for lag in lags:
        con = contact(_shift(sub, lag), xs, ys, n)
        d = [touch._window_min(con.d[t], 1)[frames] for t in (0, 1)]
        out[lag] = (d[0], d[1])
    return out


def _gap_edge_distance(seen, half, frames):
    """Frames from each frame to the nearest unseen frame (or half edge) on either side."""
    n = len(seen)
    pos = np.arange(n)
    bad = ~seen | (half == 0)
    hb = np.r_[True, half[1:] != half[:-1]]
    # last bad-or-half-start at or before i; next bad-or-half-end at or after i
    mark_prev = np.where(bad | hb, pos, -10 ** 9)
    prev = np.maximum.accumulate(mark_prev)
    he = np.r_[half[1:] != half[:-1], True]
    mark_next = np.where(bad | he, pos, 10 ** 9)
    nxt = np.minimum.accumulate(mark_next[::-1])[::-1]
    return np.minimum(frames - prev[frames], nxt[frames] - frames)


def _fit_v(t, v):
    t = t - t.mean()
    den = (t * t).sum()
    return float((t * (v - v.mean())).sum() / den) if den > 0 else np.nan


# ---------------------------------------------------------------- computation

def compute(meta, ball, raw, players):
    n = int(meta["n_frames"])
    fps = float(meta["fps"])
    inp = base.make_inputs(ball, players, meta)
    graded = _graded_mask(meta, n)
    gmin = graded.sum() / fps / 60.0
    seen_g = inp.seen & np.isfinite(inp.x) & graded
    smin = seen_g.sum() / fps / 60.0
    out = {"graded_minutes": round(gmin, 2), "seen_minutes": round(smin, 2),
           "seen_share_graded": _share(seen_g[graded]),
           "align_lag_frames": inp.lag, "radius_scale": round(inp.radius_scale, 3)}

    # --- the detector's own run (for the replication check)
    det_rows = touch.touches(inp)
    det_frames = np.array([r["frame"] for r in det_rows], dtype=np.int64)
    det_teams = np.array([r["team"] for r in det_rows], dtype=np.int64)

    # --- F0 validity of the velocity change
    dv, ang, vb, va = touch.velocity_change(inp)
    scale = touch.noise_scale(dv)
    thr = max(touch.DV_MPS, touch.NOISE_K * scale)
    valid = np.isfinite(dv)
    w = touch.W
    m = inp.seen & np.isfinite(inp.x)
    _, nb = touch.window_slopes(inp.x, m, -w, 0)
    _, na = touch.window_slopes(inp.x, m, 0, w)
    few = m & ((nb < touch.MIN_PTS) | (na < touch.MIN_PTS))
    out["f0_graded_share"] = {
        "unseen": _share(~m[graded]),
        "seen_few_pts": _share(few[graded]),
        "seen_half_edge": _share((m & ~few & ~valid)[graded]),
        "dv_valid": _share(valid[graded])}
    out["dv_noise_scale_mps"] = round(scale, 3)
    out["dv_threshold_mps"] = round(thr, 3)

    # --- F1 sharp frames and F2 peaks, exactly as the detector
    sb, sa = np.hypot(*vb), np.hypot(*va)
    ang_eff = np.where(np.minimum(sb, sa) >= max(touch.ANGLE_MIN_MPS, touch.NOISE_K * scale), ang, 0.0)
    sharp_dv = dv > thr
    sharp_ang = ang_eff > touch.ANGLE_DEG
    sharp = sharp_dv | sharp_ang
    score = np.where(sharp, np.nan_to_num(dv, nan=0.0) + 0.1 * ang_eff, -np.inf)
    ph = max(touch.PEAK_HALF, w)
    peaks = _peaks(score, sharp, ph)
    radius = touch.RADIUS_M * inp.radius_scale
    d_team = [touch._window_min(inp.con.d[t], 1) for t in (0, 1)]
    d_any = np.fmin(d_team[0], d_team[1])
    acc = peaks[d_any[peaks] <= radius]
    out["replicates_detector"] = bool(len(acc) == len(det_frames) and np.array_equal(acc, det_frames))
    pk_g = peaks[graded[peaks]]
    acc_g = acc[graded[acc]]
    out["f1_sharp_share_of_valid"] = _share(sharp[valid & graded])
    out["f2_peaks_per_graded_min"] = _per_min(len(pk_g), gmin)
    out["f2_peaks_per_seen_min"] = _per_min(len(pk_g), smin)
    out["f2_peak_trigger_share"] = {
        "dv_only": _share(sharp_dv[pk_g] & ~sharp_ang[pk_g]),
        "angle_only": _share(~sharp_dv[pk_g] & sharp_ang[pk_g]),
        "both": _share(sharp_dv[pk_g] & sharp_ang[pk_g])}
    out["f3_player_pass_share"] = _share(d_any[pk_g] <= radius)
    out["f3_no_player_tracked_share"] = _share(~np.isfinite(d_any[pk_g]))
    out["touches_per_graded_min"] = _per_min(len(acc_g), gmin)
    out["touches_per_seen_min"] = _per_min(len(acc_g), smin)
    out["detector_radius_m"] = round(radius, 3)

    # --- 1. fixed ladder of dv thresholds (dv only, same peak suppression)
    ladder = {}
    ladder_peaks = {}
    for t in LADDER_MPS:
        s = dv > t
        p = _peaks(np.where(s, dv, -np.inf), s, ph)
        p = p[graded[p]]
        ladder_peaks[t] = p
        ladder["g{:g}".format(t)] = _per_min(len(p), gmin)
        ladder["s{:g}".format(t)] = _per_min(len(p), smin)
    out["ladder_peaks_per_min"] = ladder      # gT per graded min, sT per seen min

    # --- 2. distributions: dv, peak dv, acceleration, sharpness
    out["dv_hist_edges"] = DV_EDGES[:-1]
    out["dv_hist_share_valid"] = _share_hist(dv[valid & graded], DV_EDGES)
    out["dv_median_valid"] = _med(dv[valid & graded])
    p2 = ladder_peaks[LADDER_MPS[0]]
    out["peak_dv_hist_edges"] = PEAK_EDGES[:-1]
    out["peak_dv_hist_per_graded_min"] = [round(c / gmin, 3) for c in _hist(dv[p2], PEAK_EDGES)]
    x, y = inp.x, inp.y
    trip = m[1:-1] & m[:-2] & m[2:] & graded[1:-1] & (inp.half[:-2] == inp.half[2:])
    a_s = np.hypot(x[2:] - 2 * x[1:-1] + x[:-2], y[2:] - 2 * y[1:-1] + y[:-2]) * fps * fps
    out["acc_edges"] = ACC_EDGES[:-1]
    out["acc_smoothed_hist_share"] = _share_hist(a_s[trip], ACC_EDGES)
    out["acc_smoothed_median"] = _med(a_s[trip])
    rx = raw["x_m"].to_numpy(float)
    ry = raw["y_m"].to_numpy(float)
    rs = raw["seen"].to_numpy(bool) & np.isfinite(rx)
    rtrip = rs[1:-1] & rs[:-2] & rs[2:] & graded[1:-1] & (inp.half[:-2] == inp.half[2:])
    d2x = rx[2:] - 2 * rx[1:-1] + rx[:-2]
    d2y = ry[2:] - 2 * ry[1:-1] + ry[:-2]
    a_r = np.hypot(d2x, d2y) * fps * fps
    out["acc_raw_hist_share"] = _share_hist(a_r[rtrip], ACC_EDGES)
    out["acc_raw_median"] = _med(a_r[rtrip])
    # Lag-1 autocorrelation of the raw second difference: white jitter gives about -0.67,
    # a low-pass filtered track gives a positive value.
    both = rtrip[1:] & rtrip[:-1]
    if both.sum() >= 100:
        cs = []
        for d2 in (d2x, d2y):
            u, v = d2[1:][both], d2[:-1][both]
            cs.append(float(np.corrcoef(u, v)[0, 1]))
        out["raw_d2_lag1_autocorr"] = round(float(np.mean(cs)), 4)
    else:
        out["raw_d2_lag1_autocorr"] = None
    # Kink sharpness: dv over a short window against dv over W at the same peak.
    inp_short = inp
    dvs, _, _, _ = touch.velocity_change(inp_short, w=SHORT_W, min_pts=SHORT_MIN_PTS)
    pr = ladder_peaks[SHARP_REF_MPS]
    ratio = dvs[pr] / dv[pr]
    out["kink_short_over_long_dv_median"] = _med(ratio)
    out["kink_short_over_long_dv_hist"] = _share_hist(ratio, [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1e9])
    # Width of each peak: contiguous frames with dv >= half the peak value.
    widths = []
    for i in pr:
        h = dv[i] / 2.0
        lo = i
        while lo - 1 >= 0 and dv[lo - 1] >= h:
            lo -= 1
        hi = i
        while hi + 1 < n and dv[hi + 1] >= h:
            hi += 1
        widths.append(hi - lo + 1)
    out["kink_width_edges"] = WIDTH_EDGES[:-1]
    out["kink_width_hist_share"] = _share_hist(widths, WIDTH_EDGES)
    out["kink_width_median"] = _med(widths)

    # --- 3. player proximity at lag 0, the game's lag and the best lag
    sets = {"det": pk_g, "l3": ladder_peaks[3.0], "l6": ladder_peaks[6.0]}
    union = np.unique(np.concatenate([v for v in sets.values()] + [np.zeros(0, np.int64)]))
    if len(union):
        dl = distances_by_lag(players, x, y, union, n)
        pos = {f: k for k, f in enumerate(union)}
        for name, fr in sets.items():
            if len(fr) == 0:
                continue
            ix = np.array([pos[f] for f in fr])
            dany = {lag: np.fmin(a[ix], b[ix]) for lag, (a, b) in dl.items()}
            share1 = {lag: float(np.mean(v <= BEST_LAG_RADIUS_M)) for lag, v in dany.items()}
            best = max(share1, key=lambda k: (share1[k], -abs(k)))
            for tag, lag in (("lag0", 0), ("laggame", inp.lag), ("lagbest", best)):
                out["near_{}_{}".format(name, tag)] = {
                    "r{:g}".format(r): round(float(np.mean(dany[lag] <= r)), 4) for r in RADII_M}
            out["near_{}_best_lag".format(name)] = int(best)
            out["near_{}_share1m_by_lag".format(name)] = {
                str(l): round(share1[l], 4) for l in (-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10)}
            out["near_{}_median_d_laggame".format(name)] = _med(dany[inp.lag])

    # --- 4. rejection reasons in detector order, as shares of the graded peaks and per minute
    n_pk = max(len(pk_g), 1)
    lost_player = int((~(d_any[pk_g] <= radius)).sum())
    lost_far = int(((d_any[pk_g] > radius) & np.isfinite(d_any[pk_g])).sum())
    out["f4_funnel_per_graded_min"] = {
        "graded_frames_valid_dv": round(float((valid & graded).sum()) / max(graded.sum(), 1), 4),
        "sharp_frames": _per_min(int((sharp & valid & graded).sum()), gmin),
        "peaks": _per_min(len(pk_g), gmin),
        "lost_no_player_tracked": _per_min(lost_player - lost_far, gmin),
        "lost_player_beyond_radius": _per_min(lost_far, gmin),
        "touches": _per_min(len(acc_g), gmin)}
    out["f4_lost_share_of_peaks"] = {"player_test": round(lost_player / n_pk, 4)}
    # Peaks between the 3 m/s floor and the adaptive threshold (lost to the adaptive threshold).
    l3 = ladder_peaks[3.0]
    below = l3[(dv[l3] <= thr) & ~sharp_ang[l3]]
    out["lost_to_adaptive_thr_per_graded_min"] = _per_min(len(below), gmin)
    out["lost_to_adaptive_thr_with_player_per_graded_min"] = _per_min(
        int((d_any[below] <= radius).sum()), gmin)
    out["detector_share_conf_mag"] = _share_hist(np.minimum(1.0, dv[acc_g] / touch.CONF_DV_FULL),
                                                 [0, 0.5, 0.75, 1.0, 1e9])

    # --- 5. seen-run structure around candidates
    for name, fr in (("det", pk_g), ("touch", acc_g), ("l3", ladder_peaks[3.0])):
        if len(fr):
            e = _gap_edge_distance(m, inp.half, fr)
            out["gapedge_{}_hist_share".format(name)] = _share_hist(e, EDGE_EDGES)
            out["gapedge_{}_within5".format(name)] = _share(e <= GAP_EDGE_NEAR)
    vg = np.flatnonzero(valid & graded)
    if len(vg):
        e = _gap_edge_distance(m, inp.half, vg)
        out["gapedge_validframes_hist_share"] = _share_hist(e, EDGE_EDGES)
        out["gapedge_validframes_within5"] = _share(e <= GAP_EDGE_NEAR)
    out["gapedge_edges"] = EDGE_EDGES[:-1]
    # Velocity change across short gaps: a touch the tracker hid.
    starts, stops = touch.runs(~m & graded)
    nk_any = nk_thr = nk_thr_pl = n_gap = 0
    t_idx = np.arange(n)
    for a, b in zip(starts, stops):
        L = b - a
        if L > GAP_KINK_MAX or a - GAP_FIT < 0 or b + GAP_FIT > n:
            continue
        if inp.half[a - 1] != inp.half[b] or inp.half[a - 1] == 0:
            continue
        bi = np.arange(a - GAP_FIT, a)
        ai = np.arange(b, b + GAP_FIT)
        bi, ai = bi[m[bi]], ai[m[ai]]
        if len(bi) < GAP_FIT_MIN or len(ai) < GAP_FIT_MIN:
            continue
        n_gap += 1
        vbx, vby = _fit_v(t_idx[bi], x[bi]), _fit_v(t_idx[bi], y[bi])
        vax, vay = _fit_v(t_idx[ai], x[ai]), _fit_v(t_idx[ai], y[ai])
        g_dv = float(np.hypot(vax - vbx, vay - vby)) * fps
        if g_dv > 3.0:
            nk_any += 1
        if g_dv > thr:
            nk_thr += 1
            dd = np.fmin(d_any[a - 1], d_any[b])
            if dd <= radius:
                nk_thr_pl += 1
    out["gap_short_per_graded_min"] = _per_min(n_gap, gmin)
    out["gap_kink_per_graded_min"] = {"dv_gt3": _per_min(nk_any, gmin),
                                      "dv_gt_thr": _per_min(nk_thr, gmin),
                                      "dv_gt_thr_player": _per_min(nk_thr_pl, gmin)}

    # --- 6. per team (rates only)
    nt = np.full(len(pk_g), -1)
    if len(pk_g):
        d0, d1 = d_team[0][pk_g], d_team[1][pk_g]
        nt = np.where(np.isfinite(d0) & ~(d1 < d0), 0, np.where(np.isfinite(d1), 1, -1))
    acc_mask = d_any[pk_g] <= radius
    for t in (0, 1):
        out["team{}_per_graded_min".format(t)] = {
            "peaks_nearest": _per_min(int((nt == t).sum()), gmin),
            "touches": _per_min(int(((nt == t) & acc_mask).sum()), gmin),
            "player_pass_share": (round(float(acc_mask[nt == t].mean()), 4)
                                  if (nt == t).any() else None)}
    out["detector_touches_team_share0"] = (round(float((det_teams[graded[det_frames]] == 0).mean()), 4)
                                           if len(det_frames) else None)
    return out


# ---------------------------------------------------------------- entry points

@match_entry
def run_match():
    quiet_warnings()
    stats = compute(*load(sources.match_root()))
    summary(_name="touch_funnel.match", **stats)
    return 0


def run_bench(roots):
    roots = require_bench_roots(roots)
    for root in roots:
        stats = compute(*load(root))
        summary(_name="touch_funnel.bench.{}".format(os.path.basename(os.path.normpath(root))),
                _quiet=True, **stats)
        print("{}: peaks/min {} touches/min {} replicates {}".format(
            os.path.basename(root), stats["f2_peaks_per_graded_min"],
            stats["touches_per_graded_min"], stats["replicates_detector"]))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Touch funnel diagnostic (aggregates only)")
    ap.add_argument("target", choices=("match", "bench"))
    ap.add_argument("--root", nargs="*", help="bench roots")
    args = ap.parse_args(argv)
    if args.target == "match":
        if args.root:
            ap.error("the match root is fixed")
        return run_match()
    if not args.root:
        ap.error("bench needs --root")
    return run_bench(args.root)


if __name__ == "__main__":
    raise SystemExit(main())
