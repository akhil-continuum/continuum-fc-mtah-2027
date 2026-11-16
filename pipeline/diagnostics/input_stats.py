"""Input statistics of a load-stage root, the same on the match and on any bench root.

    uv run python -m pipeline.diagnostics.input_stats match
    uv run python -m pipeline.diagnostics.input_stats bench                  # every noise root
    uv run python -m pipeline.diagnostics.input_stats bench --game metrica_g3_match ...

Reads only the load outputs (``players.parquet``, ``ball_raw.parquet``, ``meta.json``) and prints
aggregates through ``pipeline.report.summary`` (reports/input_stats.match.json and
reports/input_stats.bench.<id>.json). Nothing here reads labels or truth.

Graded frames: ``meta["graded_spans"]`` when the root has them (bench noise roots), otherwise
``meta["halves"]`` (the match: its halves are exactly its graded frames, DATA.md).

Statistics (all on graded frames, runs cut at span edges):

- ball missing rate; gap and seen-run length histograms (log-spaced, frames), counts and
  frame-weighted shares;
- missing rate and gap-start hazard by players within 3 m of the last seen ball position
  (0, 1, 2, 3+), by an 6 x 3 pitch grid of the last seen position, and by touchline band
  (within 12 m of y = 0, middle, within 12 m of y = W); y = 0 is the top of the provider image,
  which is the far touchline on the Metrica bench;
- missing rate by the mean speed of the tracked players (a label-free live/dead proxy) and by
  tracked players below 18;
- ball speed when seen, frame to frame and over +-5 frames; jump rate (implied speed above
  ``JUMP_MPS``) between consecutive seen frames and across gaps; median absolute second
  difference of the ball position (a direct jitter measure);
- tracked players per frame and per team; rate below 18 and below 10;
- unit check of a delivered ball speed channel (``raw_s``), if present.

``window_features(root)`` returns the per-window aggregates used by ``realism_auc``; it never
prints.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from pipeline.io import sources
from pipeline.report import match_entry, quiet_warnings, summary
from pipeline.io.roots import require_bench_roots

# Params
CROWD_RADIUS_M = 3.0
JUMP_MPS = 40.0                 # above any shot speed; a jump is a detection error
BAND_M = 12.0                   # touchline band width
GRID = (6, 3)                   # x cells, y cells (18 cells)
RUN_EDGES = [1, 2, 3, 5, 8, 13, 25, 50, 100, 250, 500, 1000, 2500, 10 ** 9]   # frames
SPEED_EDGES = [0, 0.5, 1, 2, 3, 4, 6, 8, 10, 13, 16, 20, 25, 30, 40, 60, 10 ** 6]  # m/s
PSPEED_EDGES = [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 100]                          # m/s, players
TOTAL_EDGES = list(range(0, 32, 2))                                         # players per frame
SD2_EDGES = [0, 1e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1, 3, 10 ** 6]       # m
NEAR_EDGES = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 7, 10, 15, 25, 10 ** 6]        # m, ball to nearest
LAGS = [-12, -8, -5, -3, -2, -1, 0, 1, 2, 3, 5, 8, 12]                       # frames
NEAR_STRIDE = 5
PROW_EDGES =[0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 100]                   # m/s, one player
WINDOW_FRAMES = 750             # 30 s windows for the adversarial check
LEVELS = ("low", "match", "high")


# ---------------------------------------------------------------- loading

def load_root(root):
    with open(os.path.join(root, "meta.json")) as fh:
        meta = json.load(fh)
    ball = pd.read_parquet(os.path.join(root, "ball_raw.parquet"))
    players = pd.read_parquet(os.path.join(root, "players.parquet"),
                              columns=["frame", "team", "x_m", "y_m", "vx", "vy"])
    return meta, ball, players


def graded_spans(meta):
    spans = meta.get("graded_spans") or meta["halves"]
    return [(int(a), int(b)) for a, b in spans]


def runs(mask):
    """(start, length, value) of maximal runs of a boolean array."""
    if len(mask) == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, bool)
    change = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    starts = np.concatenate([[0], change])
    lengths = np.diff(np.concatenate([starts, [len(mask)]]))
    return starts, lengths, mask[starts]


# ---------------------------------------------------------------- per-frame arrays

class Frames:
    """Per-frame arrays for one root; everything downstream is computed from these."""

    def __init__(self, root, spans=None):
        meta, ball, players = load_root(root)
        if spans is not None:
            meta = dict(meta, graded_spans=spans)
        self.meta = meta
        n = int(meta["n_frames"])
        self.n = n
        self.fps = float(meta.get("fps", 25.0))
        self.L, self.W = (float(v) for v in meta["pitch"])
        self.spans = graded_spans(meta)
        span_id = np.full(n, -1, int)
        for i, (a, b) in enumerate(self.spans):
            span_id[a:b] = i
        self.span_id = span_id
        self.graded = span_id >= 0

        ball = ball.sort_values("frame")
        bx = np.full(n, np.nan)
        by = np.full(n, np.nan)
        f = ball["frame"].to_numpy()
        bx[f] = ball["x_m"].to_numpy(float)
        by[f] = ball["y_m"].to_numpy(float)
        self.seen = np.isfinite(bx) & np.isfinite(by)
        self.bx, self.by = bx, by
        self.raw_s = None
        if "raw_s" in ball.columns:
            rs = np.full(n, np.nan)
            rs[f] = pd.to_numeric(ball["raw_s"], errors="coerce").to_numpy(float)
            self.raw_s = rs

        # Last seen ball position, forward-filled inside each graded span only.
        lx = pd.Series(np.where(self.seen, bx, np.nan))
        ly = pd.Series(np.where(self.seen, by, np.nan))
        grp = pd.Series(span_id)
        self.lx = np.array(lx.groupby(grp).ffill().to_numpy(), dtype=float)
        self.ly = np.array(ly.groupby(grp).ffill().to_numpy(), dtype=float)
        self.lx[~self.graded] = np.nan
        self.ly[~self.graded] = np.nan
        self.has_ref = np.isfinite(self.lx)

        pf = players["frame"].to_numpy()
        team = players["team"].to_numpy()
        self._pf, self._team = pf, team
        self._px = players["x_m"].to_numpy(float)
        self._py = players["y_m"].to_numpy(float)
        self.total = np.bincount(pf, minlength=n)[:n]
        self.per_team = [np.bincount(pf[team == t], minlength=n)[:n] for t in (0, 1)]
        sp = np.hypot(players["vx"].to_numpy(float), players["vy"].to_numpy(float))
        ok = np.isfinite(sp)
        rows_graded = self.graded[pf]
        self.prow_hist = _hist_share(sp[rows_graded & ok], PROW_EDGES)
        self.prow_nan = round(float((~ok[rows_graded]).mean()), 5) if rows_graded.any() else None
        ssum = np.bincount(pf[ok], weights=sp[ok], minlength=n)[:n]
        scnt = np.bincount(pf[ok], minlength=n)[:n]
        with np.errstate(invalid="ignore", divide="ignore"):
            self.pspeed = np.where(scnt > 0, ssum / np.maximum(scnt, 1), np.nan)
        d = np.hypot(players["x_m"].to_numpy(float) - self.lx[pf],
                     players["y_m"].to_numpy(float) - self.ly[pf])
        near = np.isfinite(d) & (d < CROWD_RADIUS_M)
        self.crowd = np.bincount(pf[near], minlength=n)[:n]
        # Ball speed, frame to frame (both frames seen, same span) and over +-5 frames.
        same = np.zeros(n, bool)
        same[1:] = (span_id[1:] == span_id[:-1]) & self.graded[1:]
        pair = same & self.seen & np.roll(self.seen, 1)
        pair[0] = False
        v1 = np.full(n, np.nan)
        v1[1:] = np.hypot(np.diff(bx), np.diff(by)) * self.fps
        self.v1 = np.where(pair, v1, np.nan)
        k = 5
        v10 = np.full(n, np.nan)
        v10[k:-k] = np.hypot(bx[2 * k:] - bx[:-2 * k], by[2 * k:] - by[:-2 * k]) * self.fps / (2 * k)
        ok10 = np.zeros(n, bool)
        ok10[k:-k] = (span_id[2 * k:] == span_id[:-2 * k]) & self.graded[k:-k]
        self.v10 = np.where(ok10, v10, np.nan)
        # Second difference of x (jitter); needs t-1, t, t+1 seen in one span.
        sd = np.full(n, np.nan)
        sd[1:-1] = np.abs(bx[2:] - 2 * bx[1:-1] + bx[:-2])
        ok2 = np.zeros(n, bool)
        ok2[1:-1] = (span_id[2:] == span_id[:-2]) & self.graded[1:-1]
        self.sd2 = np.where(ok2, sd, np.nan)

    def nearest_stats(self):
        """Distance from the seen ball to the nearest tracked player, at time offsets LAGS
        (players at t + k), on every NEAR_STRIDE-th seen graded frame; and the rate at which the
        nearest player's team changes between consecutive seen frames."""
        pf, team = self._pf, self._team
        px, py = self._px, self._py
        order = np.argsort(pf, kind="stable")
        pf, team, px, py = pf[order], team[order], px[order], py[order]
        starts = np.searchsorted(pf, np.arange(self.n + 1))
        out = {}
        base = np.flatnonzero(self.graded & self.seen)
        sub = base[::NEAR_STRIDE]
        med = {}
        for k in LAGS:
            f2 = sub + k
            ok = (f2 >= 0) & (f2 < self.n)
            fs, f2 = sub[ok], f2[ok]
            ok = self.span_id[f2] == self.span_id[fs]
            fs, f2 = fs[ok], f2[ok]
            cnt = starts[f2 + 1] - starts[f2]
            rep = np.repeat(np.arange(len(f2)), cnt)
            idx = np.concatenate([np.arange(starts[f], starts[f + 1]) for f in f2]) \
                if len(f2) else np.zeros(0, int)
            d = np.hypot(px[idx] - self.bx[fs][rep], py[idx] - self.by[fs][rep])
            dmin = np.full(len(f2), np.inf)
            np.minimum.at(dmin, rep, d)
            dmin = dmin[np.isfinite(dmin)]
            med[str(k)] = float(np.median(dmin)) if len(dmin) else None
            if k == 0:
                out["nearest_dist_edges_m"] = NEAR_EDGES[:-1]
                out["nearest_dist_hist"] = _hist_share(dmin, NEAR_EDGES)
                out["nearest_dist_median_m"] = med["0"]
                out["nearest_dist_over_5m_share"] = round(float((dmin > 5).mean()), 5) \
                    if len(dmin) else None
        out["nearest_dist_median_by_lag"] = med
        good = {k: v for k, v in med.items() if v is not None}
        out["nearest_dist_best_lag_frames"] = int(min(good, key=good.get)) if good else None
        # Nearest team on every seen graded frame, then flips between consecutive seen frames.
        cnt = starts[base + 1] - starts[base]
        rep = np.repeat(np.arange(len(base)), cnt)
        idx = np.concatenate([np.arange(starts[f], starts[f + 1]) for f in base]) \
            if len(base) else np.zeros(0, int)
        d = np.hypot(px[idx] - self.bx[base][rep], py[idx] - self.by[base][rep])
        best = pd.DataFrame({"r": rep, "d": d, "t": team[idx]}).sort_values(["r", "d"]) \
            .drop_duplicates("r")
        nt = np.full(len(base), -1)
        nt[best["r"].to_numpy()] = best["t"].to_numpy()
        consec = (np.diff(base) == 1) & (nt[1:] >= 0) & (nt[:-1] >= 0)
        flips = int((consec & (nt[1:] != nt[:-1])).sum())
        seen_min = consec.sum() / self.fps / 60.0
        out["nearest_team_flips_per_seen_min"] = round(flips / seen_min, 3) if seen_min else None
        return out

    def span_runs(self):
        """Gap and seen runs inside graded spans: lists of (start, length, seen_value)."""
        out = []
        for a, b in self.spans:
            s, l, v = runs(self.seen[a:b])
            out.append((s + a, l, v))
        return out


# ---------------------------------------------------------------- statistics

def _rate(mask_missing, where):
    w = where.sum()
    return float(mask_missing[where].mean()) if w else None


def _hist_share(values, edges):
    values = values[np.isfinite(values)]
    h, _ = np.histogram(values, bins=edges)
    tot = h.sum()
    return [round(float(c) / tot, 5) if tot else 0.0 for c in h]


def compute(fr):
    g = fr.graded
    miss = ~fr.seen
    out = {}
    out["n_graded_frames"] = int(g.sum())
    out["graded_minutes"] = round(g.sum() / fr.fps / 60.0, 2)
    out["ball_missing_rate"] = _rate(miss, g)

    # Runs.
    gap_len, seen_len, gap_before, gap_after = [], [], [], []
    teleport, teleport_n = 0, 0
    for s, l, v in fr.span_runs():
        gap_len.append(l[~v])
        seen_len.append(l[v])
        for st, ln in zip(s[~v], l[~v]):
            a, b = st - 1, st + ln
            if a >= 0 and b < fr.n and fr.seen[a] and fr.seen[b] and fr.span_id[a] == fr.span_id[b]:
                dist = np.hypot(fr.bx[b] - fr.bx[a], fr.by[b] - fr.by[a])
                teleport_n += 1
                teleport += dist * fr.fps / (ln + 1) > JUMP_MPS
    gap_len = np.concatenate(gap_len) if gap_len else np.zeros(0)
    seen_len = np.concatenate(seen_len) if seen_len else np.zeros(0)
    edges = RUN_EDGES
    hg, _ = np.histogram(gap_len, bins=edges)
    hgw, _ = np.histogram(gap_len, bins=edges, weights=gap_len)
    hs, _ = np.histogram(seen_len, bins=edges)
    hsw, _ = np.histogram(seen_len, bins=edges, weights=seen_len)
    out["run_bin_edges_frames"] = edges[:-1]
    out["gap_count_hist"] = [int(c) for c in hg]
    out["gap_frame_share_hist"] = [round(float(c) / max(1, hgw.sum()), 5) for c in hgw]
    out["seen_count_hist"] = [int(c) for c in hs]
    out["seen_frame_share_hist"] = [round(float(c) / max(1, hsw.sum()), 5) for c in hsw]
    minutes = g.sum() / fr.fps / 60.0
    out["gaps_per_min"] = round(len(gap_len) / minutes, 3) if minutes else None
    out["gap_median_frames"] = float(np.median(gap_len)) if len(gap_len) else None
    out["gap_mean_frames"] = float(np.mean(gap_len)) if len(gap_len) else None
    out["seen_run_median_frames"] = float(np.median(seen_len)) if len(seen_len) else None
    out["missing_share_in_gaps_over_10s"] = (
        round(float(gap_len[gap_len > 10 * fr.fps].sum() / max(1, gap_len.sum())), 5))
    out["missing_share_in_gaps_under_1s"] = (
        round(float(gap_len[gap_len < fr.fps].sum() / max(1, gap_len.sum())), 5))
    out["reappear_jump_rate"] = round(teleport / teleport_n, 5) if teleport_n else None

    # Context: crowd around the last seen ball position.
    ref = g & fr.has_ref
    cb = np.minimum(fr.crowd, 3)
    out["missing_by_crowd"] = {str(k) if k < 3 else "3+": _rate(miss, ref & (cb == k))
                               for k in range(4)}
    out["frame_share_by_crowd"] = {str(k) if k < 3 else "3+": round(float((ref & (cb == k)).sum()
                                   / max(1, ref.sum())), 5) for k in range(4)}
    # Gap-start hazard: P(missing at t+1 | seen at t), by crowd at t (around the ball at t).
    nxt = np.zeros(fr.n, bool)
    nxt[:-1] = (fr.span_id[1:] == fr.span_id[:-1]) & g[:-1]
    start_base = nxt & fr.seen
    gap_next = np.zeros(fr.n, bool)
    gap_next[:-1] = miss[1:]
    out["gap_hazard_by_crowd"] = {str(k) if k < 3 else "3+": _rate(gap_next, start_base & (cb == k))
                                  for k in range(4)}
    out["gap_hazard_overall"] = _rate(gap_next, start_base)

    # Pitch zones of the last seen position.
    cx = np.clip((fr.lx / fr.L * GRID[0]).astype(float), 0, GRID[0] - 1e-9)
    cy = np.clip((fr.ly / fr.W * GRID[1]).astype(float), 0, GRID[1] - 1e-9)
    zones, zshare = {}, {}
    for i in range(GRID[0]):
        for j in range(GRID[1]):
            m = ref & (np.floor(cx) == i) & (np.floor(cy) == j)
            zones["x{}y{}".format(i, j)] = _rate(miss, m)
            zshare["x{}y{}".format(i, j)] = round(float(m.sum() / max(1, ref.sum())), 5)
    out["missing_by_zone"] = zones
    out["frame_share_by_zone"] = zshare
    band = np.where(fr.ly < BAND_M, 0, np.where(fr.ly > fr.W - BAND_M, 2, 1))
    names = ("y_low_band", "middle", "y_high_band")
    out["missing_by_band"] = {names[k]: _rate(miss, ref & (band == k)) for k in range(3)}
    out["gap_hazard_by_band"] = {names[k]: _rate(gap_next, start_base & (band == k))
                                 for k in range(3)}

    # Label-free live/dead proxy: mean speed of the tracked players.
    pb = np.digitize(fr.pspeed, PSPEED_EDGES[1:-1])
    okp = g & np.isfinite(fr.pspeed)
    keys = ["{}-{}".format(PSPEED_EDGES[i], PSPEED_EDGES[i + 1]) for i in range(len(PSPEED_EDGES) - 1)]
    out["pspeed_bin_edges"] = PSPEED_EDGES
    out["missing_by_player_speed"] = {keys[i]: _rate(miss, okp & (pb == i)) for i in range(len(keys))}
    out["frame_share_by_player_speed"] = {keys[i]: round(float((okp & (pb == i)).sum()
                                          / max(1, okp.sum())), 5) for i in range(len(keys))}

    # Ball speed when seen.
    out["speed_bin_edges_mps"] = SPEED_EDGES[:-1]
    out["ball_speed_1f_hist"] = _hist_share(fr.v1[g], SPEED_EDGES)
    out["ball_speed_10f_hist"] = _hist_share(fr.v10[g], SPEED_EDGES)
    v1 = fr.v1[g & np.isfinite(fr.v1)]
    v10 = fr.v10[g & np.isfinite(fr.v10)]
    out["ball_speed_1f_median"] = float(np.median(v1)) if len(v1) else None
    out["ball_speed_1f_p90"] = float(np.percentile(v1, 90)) if len(v1) else None
    out["ball_speed_10f_median"] = float(np.median(v10)) if len(v10) else None
    out["ball_speed_10f_p90"] = float(np.percentile(v10, 90)) if len(v10) else None
    out["ball_jump_rate"] = round(float((v1 > JUMP_MPS).mean()), 6) if len(v1) else None
    sd = fr.sd2[g & np.isfinite(fr.sd2)]
    out["ball_second_diff_median_m"] = float(np.median(sd)) if len(sd) else None
    out["ball_static_share_10f"] = round(float((v10 < 0.3).mean()), 5) if len(v10) else None
    out["second_diff_edges_m"] = SD2_EDGES[:-1]
    out["ball_second_diff_hist"] = _hist_share(sd, SD2_EDGES)
    # Near-linear ball segments (second difference under 2 mm): interpolation signature.
    out["ball_linear_share"] = round(float((sd < 0.002).mean()), 5) if len(sd) else None

    # Players.
    tot = fr.total[g]
    out["players_total_edges"] = TOTAL_EDGES
    out["players_total_hist"] = _hist_share(tot.astype(float), TOTAL_EDGES + [10 ** 6])
    out["players_mean"] = float(tot.mean())
    out["players_rate_below_18"] = round(float((tot < 18).mean()), 5)
    out["players_rate_below_10"] = round(float((tot < 10).mean()), 5)
    out["players_rate_above_22"] = round(float((tot > 22).mean()), 5)
    team_counts = np.concatenate([fr.per_team[0][g], fr.per_team[1][g]])
    out["team_count_hist_0_to_15plus"] = _hist_share(np.minimum(team_counts, 15).astype(float),
                                                     list(range(17)))
    diff = np.abs(fr.per_team[0][g] - fr.per_team[1][g])
    out["team_count_absdiff_mean"] = float(diff.mean())
    out["missing_when_players_below_18"] = _rate(miss, g & (fr.total < 18))
    out["missing_when_players_18plus"] = _rate(miss, g & (fr.total >= 18))
    out["player_speed_mean"] = float(np.nanmean(fr.pspeed[g]))
    out["player_row_speed_edges"] = PROW_EDGES[:-1]
    out["player_row_speed_hist"] = fr.prow_hist
    out["player_row_speed_nan_share"] = fr.prow_nan

    # Ball-to-player consistency: nearest-player distance, its best time offset, nearest-team flips.
    out.update(fr.nearest_stats())

    # Delivered ball-speed channel, if any: which unit makes it agree with positions?
    if fr.raw_s is not None:
        rs = fr.raw_s
        ok = g & fr.seen & np.isfinite(rs) & np.isfinite(fr.v10) & (fr.v10 > 3)
        out["raw_s_finite_when_seen"] = _rate(np.isfinite(rs), g & fr.seen)
        out["raw_s_finite_when_missing"] = _rate(np.isfinite(rs), g & ~fr.seen)
        if ok.sum():
            out["raw_s_over_speed10f_median"] = float(np.median(rs[ok] / fr.v10[ok]))
            out["raw_s_speed_corr"] = float(np.corrcoef(rs[ok], fr.v10[ok])[0, 1])
    return out


# ---------------------------------------------------------------- windows (adversarial check)

FEATURES = [
    "missing_rate", "gap_starts", "longest_gap", "mean_seen_run", "ball_speed_1f_med",
    "ball_speed_10f_p90", "jump_rate", "second_diff_med", "static_share", "players_mean",
    "players_min", "players_below_18", "team_absdiff", "player_speed", "crowd3_share",
    "missing_crowd3", "missing_crowd0", "missing_far_band",
]


def window_features(fr, window=WINDOW_FRAMES):
    """One row per non-overlapping window inside a graded span. Never printed."""
    rows = []
    miss = ~fr.seen
    for a, b in fr.spans:
        for s in range(a, b - window + 1, window):
            e = s + window
            sl = slice(s, e)
            m = miss[sl]
            st, ln, v = runs(fr.seen[sl])
            gaps = ln[~v]
            seen_runs = ln[v]
            v1 = fr.v1[sl]
            v1 = v1[np.isfinite(v1)]
            v10 = fr.v10[sl]
            v10 = v10[np.isfinite(v10)]
            sd = fr.sd2[sl]
            sd = sd[np.isfinite(sd)]
            ref = fr.has_ref[sl]
            cr = fr.crowd[sl]
            ly = fr.ly[sl]
            tot = fr.total[sl]

            def r(mask):
                return float(m[mask].mean()) if mask.sum() else np.nan

            rows.append([
                float(m.mean()), float(len(gaps)), float(gaps.max()) if len(gaps) else 0.0,
                float(seen_runs.mean()) if len(seen_runs) else 0.0,
                float(np.median(v1)) if len(v1) else np.nan,
                float(np.percentile(v10, 90)) if len(v10) else np.nan,
                float((v1 > JUMP_MPS).mean()) if len(v1) else np.nan,
                float(np.median(sd)) if len(sd) else np.nan,
                float((v10 < 0.3).mean()) if len(v10) else np.nan,
                float(tot.mean()), float(tot.min()), float((tot < 18).mean()),
                float(np.abs(fr.per_team[0][sl] - fr.per_team[1][sl]).mean()),
                float(np.nanmean(fr.pspeed[sl])) if np.isfinite(fr.pspeed[sl]).any() else np.nan,
                float((ref & (cr >= 3)).mean()), r(ref & (cr >= 3)), r(ref & (cr == 0)),
                r(ref & (ly < BAND_M)),
            ])
    return pd.DataFrame(rows, columns=FEATURES)


# ---------------------------------------------------------------- entry points

def bench_roots(only=None):
    base = os.path.join(sources.ARTIFACTS, "bench")
    if only:
        return [os.path.join(base, g) for g in only]
    found = []
    for path in sorted(glob.glob(os.path.join(base, "*"))):
        name = os.path.basename(path)
        if name.rsplit("_", 1)[-1] in LEVELS and os.path.isfile(os.path.join(path, "meta.json")):
            found.append(path)
    return found


@match_entry
def run_match():
    quiet_warnings()
    stats = compute(Frames(sources.match_root()))
    summary(_name="input_stats.match", **stats)
    return 0


def run_bench(only=None, spans_from=None):
    spans = None
    if spans_from:
        with open(os.path.join(sources.ARTIFACTS, "bench", spans_from, "meta.json")) as fh:
            spans = json.load(fh)["graded_spans"]
    found = bench_roots(only)
    require_bench_roots(found)
    for root in found:
        stats = compute(Frames(root, spans))
        summary(_name="input_stats.bench.{}".format(os.path.basename(root)), _quiet=True, **stats)
        print("{}: missing {:.3f}  players {:.2f}".format(
            os.path.basename(root), stats["ball_missing_rate"], stats["players_mean"]))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Input statistics of a load-stage root")
    ap.add_argument("target", choices=("match", "bench"))
    ap.add_argument("--game", nargs="*", help="bench root ids under artifacts/bench/")
    ap.add_argument("--spans-from", help="bench root whose graded_spans to use (e.g. for a clean "
                    "load root without them)")
    args = ap.parse_args(argv)
    if args.target == "match":
        if args.game:
            ap.error("the match takes no --game")
        return run_match()
    return run_bench(args.game, args.spans_from)


if __name__ == "__main__":
    raise SystemExit(main())
