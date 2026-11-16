"""Match-like noisy input for a bench game (bench-truth skill, "Noise").

    uv run python -m bench.noise [--games ...] [--levels low match high] [--model v3|v2|legacy]

Three models, all seeded the same way (seed = CRC32 of "game|level"):

- "v3" (default, T-33, proposal P-17a) writes artifacts/bench/<game>_<level>/: v2 plus a centred
  Gaussian low-pass on the ball inside each seen run, because the match's ball track is low-pass
  filtered. Sigma per level in V3_LOWPASS, fitted with bench/smoothness.py to the match's
  ball-shape aggregates only. Players, ball gaps and the RNG stream are identical to v2's.
- "v2" (T-6, proposals P-3 to P-7) writes artifacts/bench/<game>_legacy2<level>/ and is fitted to
  the match's input aggregates (reports/input_stats.match.json).
- "legacy" is the T-2 model (COMMON, PARAMS, noisy_ball, noisy_players below, unchanged) and
  writes artifacts/bench/<game>_legacy<level>/.

Older models are kept so earlier numbers can be reproduced (bench.score --level legacy2match).

Turns a corrected Metrica game (identities, clean ball) into what the match gives us: automatic
tracking with team membership only, players dropping in and out, and a ball that goes missing in
context-dependent runs. Writes, per game and level, the load-stage contract files to the root:

    players.parquet   frame, team, x_m, y_m, vx, vy   (no identities; duplicates merged)
    ball_raw.parquet  frame, x_m, y_m, seen           (as observed; no cleaning)
    meta.json         same keys the shared load stage writes, plus a "noise" block
    truth*.csv, stoppages.csv                          copied from artifacts/bench/<game_id>/

Coordinates follow pipeline/io/load.py: metres, provider orientation (origin at the top-left
corner of the Metrica image, x along the length 0..105, y across 0..68 growing downwards).

Which numbers are measured and which are guesses is listed in the parameter blocks below.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import zlib

import numpy as np
import pandas as pd

from bench import metrica_io
from pipeline.io.load import (MAX_SPEED_MPS, attack_directions,  # shared, read-only
                              centred_velocity, merge_duplicates)


def _velocities(X, Y, halves):
    """Centred-difference velocities as the shared loader computes them, set to NaN above its
    MAX_SPEED_MPS (a jump, not running). Computed per half, so no difference spans the half
    boundary (Metrica halves are contiguous in frame numbers)."""
    VX = np.full(X.shape, np.nan)
    VY = np.full(Y.shape, np.nan)
    for a, b in halves:
        VX[a:b] = centred_velocity(X[a:b], FPS)
        VY[a:b] = centred_velocity(Y[a:b], FPS)
    too_fast = np.hypot(VX, VY) > MAX_SPEED_MPS
    VX[too_fast] = np.nan
    VY[too_fast] = np.nan
    return VX, VY


def _attack_meta(players, n, halves):
    """attack_direction and its agreement, from the noisy players, with the loader's function."""
    cx, cnt = {}, {}
    for t in (0, 1):
        sub = players[players["team"] == t]
        s = np.bincount(sub["frame"].to_numpy(), weights=sub["x_m"].to_numpy(float), minlength=n)
        c = np.bincount(sub["frame"].to_numpy(), minlength=n)
        cnt[t] = c
        with np.errstate(invalid="ignore", divide="ignore"):
            cx[t] = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    return attack_directions(cx[0], cx[1], cnt[0], cnt[1], halves)

ART = os.path.join(metrica_io.ROOT, "artifacts", "bench")
FPS = metrica_io.FPS

# ---- legacy model (T-2). Every value is a guess unless marked "target" (a match aggregate from DATA.md). The PDS
# bench-versus-match check is expected to move them.
PARAMS = {
    "low": dict(live_missing=0.08, players_mean=21.5, restart_visible_p=0.3, exit_visible_p=0.3,
                dup_rate=0.003),
    "match": dict(live_missing=0.18, players_mean=20.5, restart_visible_p=0.5,
                  exit_visible_p=0.5, dup_rate=0.01),
    "high": dict(live_missing=0.30, players_mean=19.0, restart_visible_p=0.7, exit_visible_p=0.7,
                 dup_rate=0.02),
}
COMMON = dict(
    crowd_radius_m=3.0, crowd_min=4, crowd_mult=4.0,      # skill: 4+ players within 3 m
    fast_speed_mps=12.0, fast_mult=3.0,                   # long fast kicks (aerial proxy)
    far_band_m=12.0, far_mult=2.0,                        # near the far touchline from camera
    run_xmin_frames=3, run_alpha=1.3, run_cap_frames=250,  # heavy-tailed run lengths (Pareto)
    ball_sigma_m=0.25, player_sigma_m=0.15,               # position jitter
    player_gap_s=(0.5, 5.0),                              # skill: tracklets break at 0.5 to 5 s
    camera_half_width_m=40.0, edge_mult=5.0,              # notional camera follows the ball
    dup_offset_m=0.3, dup_len_s=(0.2, 2.0),
    restart_visible_s=(0.5, 3.0), exit_visible_s=(0.2, 1.5), exit_decay=0.95,
)


def _seed(game_id, level):
    return zlib.crc32("{}|{}".format(game_id, level).encode()) & 0x7FFFFFFF


def _runs_from_starts(starts, lengths, n):
    """Union of [s, s + lengths[s]) over start frames."""
    mask = np.zeros(n + 1, np.int32)
    idx = np.flatnonzero(starts)
    np.add.at(mask, idx, 1)
    np.add.at(mask, np.minimum(idx + lengths[idx], n), -1)
    return np.cumsum(mask)[:n] > 0


def _calibrate(u, mult, lengths, eligible, target, n):
    """Base hazard so that the share of `eligible` frames covered by runs is `target`."""
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        cover = _runs_from_starts((u < mid * mult) & eligible, lengths, n)
        share = cover[eligible].mean() if eligible.any() else 0.0
        lo, hi = (mid, hi) if share < target else (lo, mid)
    base = 0.5 * (lo + hi)
    return base, _runs_from_starts((u < base * mult) & eligible, lengths, n)


def _pareto_lengths(rng, n, xmin, alpha, cap):
    u = rng.random(n)
    return np.minimum(np.ceil(xmin * (1.0 - u) ** (-1.0 / alpha)).astype(int), cap)


def ball_context(game):
    """Per-frame context from the clean game: crowd count, ball speed, far-side flag."""
    ball, players = game.ball, game.players
    d = np.linalg.norm(players - ball[:, None, :], axis=2)
    crowd = np.sum(d < COMMON["crowd_radius_m"], axis=1)
    v = np.full(len(ball), np.nan)
    v[1:-1] = np.linalg.norm(ball[2:] - ball[:-2], axis=1) * FPS / 2.0
    far = ball[:, 1] > (game.pitch[1] / 2.0 - COMMON["far_band_m"])   # top touchline (y up)
    return crowd, np.nan_to_num(v, nan=0.0), far


def noisy_ball(game, labels, stops, level, rng):
    p = {**COMMON, **PARAMS[level]}
    n = game.n_frames
    ball = game.ball.copy()
    seen_clean = np.isfinite(ball[:, 0])
    graded = labels != "X"
    live = graded & (labels != "D")
    crowd, speed, far = ball_context(game)

    # 1. Live-play dropouts: runs started with a context-dependent hazard.
    mult = np.ones(n)
    mult *= np.where(crowd >= p["crowd_min"], p["crowd_mult"], 1.0)
    mult *= np.where(speed > p["fast_speed_mps"], p["fast_mult"], 1.0)
    mult *= np.where(far, p["far_mult"], 1.0)
    u = rng.random(n)
    lengths = _pareto_lengths(rng, n, p["run_xmin_frames"], p["run_alpha"], p["run_cap_frames"])
    # Calibrate on the frames still seen in live play, so the clean gaps count towards the target.
    clean_live_missing = float((~seen_clean[live]).mean()) if live.any() else 0.0
    extra_target = max(0.0, (p["live_missing"] - clean_live_missing) / max(1e-9, 1 - clean_live_missing))
    eligible = live & seen_clean
    base, drop = _calibrate(u, mult, lengths, eligible, extra_target, n)
    ball[drop] = np.nan

    # 2. Dead ball: missing as in the corrected data (86 to 99 percent of dead frames), except
    #    (a) the ball placed at the restart spot is seen for the last moments before some restarts;
    #    (b) after some exits the ball stays visible outside the line for a moment.
    n_restart_vis, n_exit_vis = 0, 0
    for s in stops.itertuples():
        if s.restart - s.start < 25:
            continue
        after = np.flatnonzero(np.isfinite(game.ball[s.restart:s.restart + 50, 0]))
        if len(after) and rng.random() < p["restart_visible_p"]:
            spot = game.ball[s.restart + after[0]]
            k = int(rng.uniform(*p["restart_visible_s"]) * FPS)
            a = max(s.start, s.restart - k)
            ball[a:s.restart] = spot + rng.normal(0, 0.05, size=(s.restart - a, 2))
            n_restart_vis += 1
        if s.cause == "ball_out" and rng.random() < p["exit_visible_p"]:
            prev = np.flatnonzero(np.isfinite(game.ball[max(0, s.start - 10):s.start, 0]))
            if len(prev) >= 2:
                i1 = max(0, s.start - 10) + prev[-1]
                i0 = max(0, s.start - 10) + prev[-2]
                vel = (game.ball[i1] - game.ball[i0]) / max(1, i1 - i0)
                k = int(rng.uniform(*p["exit_visible_s"]) * FPS)
                k = min(k, s.restart - s.start)
                steps = np.arange(1, k + 1)
                travelled = np.cumsum(p["exit_decay"] ** steps)
                ball[s.start:s.start + k] = game.ball[i1] + travelled[:, None] * vel
                n_exit_vis += 1

    # 3. Jitter on every seen ball position.
    seen = np.isfinite(ball[:, 0])
    ball[seen] += rng.normal(0, p["ball_sigma_m"], size=(int(seen.sum()), 2))

    info = {"live_missing_target": p["live_missing"], "base_hazard": base,
            "clean_live_missing": clean_live_missing,
            "live_missing": float((~seen[live]).mean()),
            "dead_missing": float((~seen[graded & ~live]).mean()),
            "graded_missing": float((~seen[graded]).mean()),
            "restart_visible_stoppages": n_restart_vis, "exit_visible_stoppages": n_exit_vis}
    return ball, info


def noisy_players(game, labels, ball_noisy, level, rng):
    p = {**COMMON, **PARAMS[level]}
    n, P = game.n_frames, game.players.shape[1]
    pos = game.players.copy()
    on = np.isfinite(pos[:, :, 0])
    graded = labels != "X"

    # Notional camera centre: ball x where seen (forward-filled), smoothed over 2 s.
    bx = pd.Series(ball_noisy[:, 0]).ffill().bfill().fillna(0.0)
    cam = bx.rolling(51, center=True, min_periods=1).mean().to_numpy()
    edge = np.abs(pos[:, :, 0] - cam[:, None]) > p["camera_half_width_m"]

    lo, hi = (int(s * FPS) for s in p["player_gap_s"])
    u = rng.random((n, P))
    lengths = rng.integers(lo, hi + 1, size=(n, P))
    mult = np.where(edge, p["edge_mult"], 1.0)
    target_present = p["players_mean"] / 22.0

    def drop_for(base):
        out = np.zeros((n, P), bool)
        for j in range(P):
            out[:, j] = _runs_from_starts((u[:, j] < base * mult[:, j]) & on[:, j],
                                          lengths[:, j], n)
        return out

    lo_b, hi_b = 0.0, 0.05
    for _ in range(20):
        mid = 0.5 * (lo_b + hi_b)
        present = (on & ~drop_for(mid))[graded].sum() / max(1, on[graded].sum())
        lo_b, hi_b = (mid, hi_b) if present > target_present else (lo_b, mid)
    base = 0.5 * (lo_b + hi_b)
    drop = drop_for(base)
    pos[drop] = np.nan
    pos += rng.normal(0, p["player_sigma_m"], size=pos.shape)

    # Duplicates: a second copy within dup_offset_m for short runs.
    du = rng.random((n, P))
    dlen = rng.integers(int(p["dup_len_s"][0] * FPS), int(p["dup_len_s"][1] * FPS) + 1, (n, P))
    dups = np.zeros((n, P), bool)
    for j in range(P):
        dups[:, j] = _runs_from_starts(du[:, j] < p["dup_rate"] / (dlen[:, j].mean()), dlen[:, j], n)
    ang = rng.uniform(0, 2 * np.pi, size=(n, P))
    rad = rng.uniform(0, p["dup_offset_m"], size=(n, P))
    dup_pos = pos + np.stack([rad * np.cos(ang), rad * np.sin(ang)], -1)
    dup_pos[~dups] = np.nan

    # Slots: shuffle order per game (identities are dropped from the output anyway).
    all_pos = np.concatenate([pos, dup_pos], axis=1)
    team = np.concatenate([game.player_team, game.player_team])
    order = rng.permutation(all_pos.shape[1])
    all_pos, team = all_pos[:, order], team[order]

    # To provider orientation (pipeline/io/load.py): x 0..L, y 0..W growing downwards.
    L, W = game.pitch
    X = all_pos[:, :, 0] + L / 2.0
    Y = W / 2.0 - all_pos[:, :, 1]
    VX, VY = _velocities(X, Y, game.halves())
    parts = []
    tracked = np.zeros(n, int)
    for t in (0, 1):
        cols = np.flatnonzero(team == t)
        mx, my, mvx, mvy, alive, _ = merge_duplicates(X[:, cols], Y[:, cols], VX[:, cols], VY[:, cols])
        tracked += alive.sum(axis=1)
        f, s = np.nonzero(alive)
        parts.append(pd.DataFrame({"frame": f.astype(np.int32), "team": np.full(len(f), t, np.int8),
                                   "x_m": mx[f, s].astype(np.float32), "y_m": my[f, s].astype(np.float32),
                                   "vx": mvx[f, s].astype(np.float32), "vy": mvy[f, s].astype(np.float32),
                                   "_slot": s}))
    players = pd.concat(parts).sort_values(["frame", "team", "_slot"], kind="mergesort")
    players = players.drop(columns="_slot").reset_index(drop=True)
    info = {"players_mean_target": p["players_mean"], "player_base_hazard": base,
            "players_per_frame_graded_mean": float(tracked[graded].mean()),
            "players_per_frame_graded_max": int(tracked[graded].max()),
            "duplicate_player_frames": int(dups.sum())}
    return players, info


# ======================================================================================
# Model "v2" (T-6; proposals P-3 to P-7). Fitted only to match INPUT aggregates in
# reports/input_stats.match.json (no labels, no scores).
# ======================================================================================

V2_COMMON = dict(
    # P-3: no white jitter on players; a tiny one on the ball.
    ball_sigma_m=0.002, player_sigma_m=0.0,
    # Short gaps in the corrected ball (mostly < 0.2 s in live play) are bridged, as the match
    # tracker evidently does (the match has no gaps between 1 and 4 s).
    fill_max_frames=50,
    # P-4: long-gap live dropouts; context multipliers as measured on the match (crowd 3+ x3,
    # y_low band x2).
    crowd_radius_m=3.0, crowd_min=3, crowd_mult=3.0, fast_speed_mps=12.0, fast_mult=2.0,
    far_band_m=12.0, far_mult=2.0,
    # xmin swept 75 / 125 (P-4 range): 125 brings gaps/min and the median gap nearer the match.
    run_xmin_frames=125, run_alpha=1.5, run_cap_frames=1500,
    # P-6: smooth ball offset and ball lead over players.
    ball_offset_tau_frames=12, ball_lead_frames=3,
    # P-7: dead-ball visibility, tracker holds, stale snaps, false reappearances.
    restart_visible_p=0.4, restart_visible_s=(1.0, 4.0),
    exit_visible_p=0.5, exit_visible_s=(0.2, 1.5), exit_decay=0.95,
    ball_hold_p=0.3, ball_hold_s=(0.2, 2.0),
    snap_per_min=1.0, snap_s=(0.2, 1.0),
    # Match short gaps are mostly 2 frames and end with a jump (reappear-jump rate 7.5%).
    false_per_min=0.2, false_s=(0.5, 2.0), false_gap_frames=(1, 3), false_sigma_m=1.0,
    # P-5: players.
    player_gap_s=(0.5, 5.0), camera_half_width_m=40.0, edge_mult=5.0,
    mass_rate_per_min=1.5, mass_xmin_frames=25, mass_alpha=1.2, mass_cap_frames=750,
    mass_keep_p=0.5, mass_missing_mult=12.0,
    dup_offset_m=(0.55, 1.5), dup_len_s=(0.2, 2.0), dup_rate=0.002,
    team_flip_rate=0.008, team_flip_s=(0.5, 5.0),
)
# Tuning record (T-6), against reports/input_stats.match.json only, three passes at the match level.
# Distance with the proposal defaults: g1 0.139, g3 0.158. Final: g1 0.124, g2 0.139, g3 0.097.
# Levels bracket the weakly identified parameters (P-4 live loss, P-6 offset size).
V2_PARAMS = {
    "low": dict(live_missing=0.06, ball_offset_sigma_m=0.9, players_mean_pre=21.5),
    "match": dict(live_missing=0.10, ball_offset_sigma_m=1.2, players_mean_pre=21.3),
    "high": dict(live_missing=0.15, ball_offset_sigma_m=1.5, players_mean_pre=21.0),
}

# ======================================================================================
# Model "v3" (T-33; proposal P-17a). v2 plus a centred Gaussian low-pass on the ball inside each
# seen run (never across a gap or a half). The competition match's ball track is low-pass
# filtered (touch_funnel: second-difference lag-1 autocorrelation +0.99 against -0.45 on v2;
# kink width 24 frames against 7). Sigma fitted only to the match's ball-shape aggregates
# (bench/smoothness.py), never to touch rates, labels or scores. The RNG stream, the seen mask
# and the players are identical to v2's: the filter consumes no randomness and the players are
# built from the unfiltered v2 ball.
# ======================================================================================
V3_COMMON = dict(
    # "before_gaps": filter after offset, lead and jitter, before dropouts and the P-7 artefacts
    # (restart spot, holds, snaps, false reappearances stay sharp). "after_gaps": filter the final
    # ball inside its seen runs.
    # Chosen: after_gaps. before_gaps leaves v2's in-run snap jumps and the restart-spot jitter
    # sharp, so the second-difference autocorrelation stays negative (smoothness distance
    # 0.32-0.40 against 0.10); the match's is +0.99.
    ball_lowpass_order="after_gaps",
)
# Sigma in frames. T-33 fit (bench.smoothness fit, pooled smoothness distance over g1-g3, after_gaps):
# optimum low 7 (0.113), match 7 (0.103; 7.5 0.105), high 7.5 (0.092; 8 0.099). Each level sits
# within 0.01 of its optimum, ordered so the levels bracket the filter width (the match's kink width
# of 24 frames asks for more smoothing than its autocorrelation and peak ladder do).
V3_LOWPASS = {"low": 7.0, "match": 7.5, "high": 8.0}
MODELS = ("v3", "v2", "legacy")
ROOT_TAG = {"v3": "", "v2": "legacy2", "legacy": "legacy"}


def lowpass_in_runs(b, period, sigma):
    """Centred Gaussian (sigma frames, edges 'nearest') on x and y inside each run of finite
    positions that lies in one period. NaN stays NaN; nothing crosses a gap or a half."""
    from scipy.ndimage import gaussian_filter1d
    out = np.array(b, dtype=float, copy=True)
    if sigma <= 0:
        return out
    ok = np.isfinite(out[:, 0]) & np.isfinite(out[:, 1])
    brk = np.zeros(len(out) + 1, bool)
    brk[1:-1] = np.asarray(period[1:]) != np.asarray(period[:-1])
    edges = np.diff(np.concatenate(([0], ok.astype(np.int8), [0])))
    for a, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        cuts = [a] + [int(i) for i in np.flatnonzero(brk[a + 1:e]) + a + 1] + [e]
        for s, t in zip(cuts[:-1], cuts[1:]):
            for ax in (0, 1):
                out[s:t, ax] = gaussian_filter1d(out[s:t, ax], sigma, mode="nearest")
    return out


def v3_ball(game, labels, stops, level, rng, sigma=None, order=None):
    """(v3 ball, v2 ball, info). Same RNG consumption as v2_ball."""
    sigma = V3_LOWPASS[level] if sigma is None else sigma
    order = order or V3_COMMON["ball_lowpass_order"]
    state = rng.bit_generator.state
    b2, info = v2_ball(game, labels, stops, level, rng)
    if order == "after_gaps":
        b3 = lowpass_in_runs(b2, game.period, sigma)
    elif order == "before_gaps":
        end = rng.bit_generator.state
        rng.bit_generator.state = state
        hook = (lambda b: lowpass_in_runs(b, game.period, sigma))
        b3, _ = v2_ball(game, labels, stops, level, rng, _after_jitter=hook)
        assert rng.bit_generator.state == end
    else:
        raise ValueError(order)
    if not np.array_equal(np.isfinite(b2[:, 0]), np.isfinite(b3[:, 0])):
        raise AssertionError("the low-pass changed the seen mask")
    info = {**info, "model": "v3", "lowpass_sigma_frames": float(sigma), "lowpass_order": order}
    return b3, b2, info


def _fill_short_gaps(ball, period, max_len):
    """Linear interpolation over NaN runs of at most max_len frames with seen ends in one half."""
    out = ball.copy()
    miss = ~np.isfinite(out[:, 0])
    n = len(out)
    edges = np.diff(np.concatenate(([0], miss.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if b - a > max_len or a == 0 or b >= n or period[a - 1] != period[b]:
            continue
        w = np.arange(1, b - a + 1)[:, None] / (b - a + 1.0)
        out[a:b] = out[a - 1] + w * (out[b] - out[a - 1])
    return out


def _smooth_noise(rng, n, sigma, tau):
    from scipy.ndimage import gaussian_filter1d
    e = gaussian_filter1d(rng.normal(0, 1, n), tau)
    return e * (sigma / max(e.std(), 1e-12))


def _gap_runs(seen, graded):
    m = (~seen) & graded
    edges = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def v2_ball(game, labels, stops, level, rng, _after_jitter=None):
    """_after_jitter: optional deterministic map applied to the ball after offset, lead and jitter
    (v3 uses it for the low-pass; it must not change which frames are finite)."""
    p = {**V2_COMMON, **V2_PARAMS[level]}
    n = game.n_frames
    graded = labels != "X"
    live = graded & (labels != "D")
    crowd, speed, far = ball_context(game)

    b = _fill_short_gaps(game.ball, game.period, p["fill_max_frames"])
    k = p["ball_lead_frames"]
    if k:
        b = np.concatenate([b[k:], np.full((k, 2), np.nan)])
    for ax in (0, 1):
        b[:, ax] += _smooth_noise(rng, n, p["ball_offset_sigma_m"], p["ball_offset_tau_frames"])
    b += rng.normal(0, p["ball_sigma_m"], size=b.shape)
    if _after_jitter is not None:
        b = _after_jitter(b)

    # P-4: long live dropouts.
    seen0 = np.isfinite(b[:, 0])
    d = np.linalg.norm(game.players - game.ball[:, None, :], axis=2)
    crowd = np.sum(d < p["crowd_radius_m"], axis=1)
    mult = np.ones(n)
    mult *= np.where(crowd >= p["crowd_min"], p["crowd_mult"], 1.0)
    mult *= np.where(speed > p["fast_speed_mps"], p["fast_mult"], 1.0)
    mult *= np.where(far, p["far_mult"], 1.0)
    u = rng.random(n)
    lengths = _pareto_lengths(rng, n, p["run_xmin_frames"], p["run_alpha"], p["run_cap_frames"])
    clean_live_missing = float((~seen0[live]).mean()) if live.any() else 0.0
    extra = max(0.0, (p["live_missing"] - clean_live_missing) / max(1e-9, 1 - clean_live_missing))
    base, drop = _calibrate(u, mult, lengths, live & seen0, extra, n)
    b[drop] = np.nan

    # P-7 (a): ball placed at the restart spot, still; (b) ball visible past the line after exits.
    n_rv = n_ev = 0
    for s in stops.itertuples():
        if s.restart - s.start < 25:
            continue
        after = np.flatnonzero(np.isfinite(b[s.restart:s.restart + 50, 0]))
        if len(after) and rng.random() < p["restart_visible_p"]:
            spot = b[s.restart + after[0]].copy()
            kk = int(rng.uniform(*p["restart_visible_s"]) * FPS)
            a = max(s.start, s.restart - kk)
            b[a:s.restart] = spot + rng.normal(0, 0.01, size=(s.restart - a, 2))
            n_rv += 1
        if s.cause == "ball_out" and rng.random() < p["exit_visible_p"]:
            prev = np.flatnonzero(np.isfinite(game.ball[max(0, s.start - 10):s.start, 0]))
            if len(prev) >= 2:
                i1 = max(0, s.start - 10) + prev[-1]
                i0 = max(0, s.start - 10) + prev[-2]
                vel = (game.ball[i1] - game.ball[i0]) / max(1, i1 - i0)
                kk = min(int(rng.uniform(*p["exit_visible_s"]) * FPS), s.restart - s.start)
                trav = np.cumsum(p["exit_decay"] ** np.arange(1, kk + 1))
                b[s.start:s.start + kk] = game.ball[i1] + trav[:, None] * vel
                n_ev += 1

    # P-7 (c): the tracker holds its last value at the start of some gaps (still ball).
    seen = np.isfinite(b[:, 0])
    n_hold = 0
    for a, e in _gap_runs(seen, graded):
        if a == 0 or not seen[a - 1] or rng.random() >= p["ball_hold_p"]:
            continue
        kk = min(int(rng.uniform(*p["ball_hold_s"]) * FPS), e - a - 1)
        if kk > 0:
            b[a:a + kk] = b[a - 1]
            n_hold += 1

    # P-7 (d): stale snaps in live play: the value freezes, then jumps to the true position.
    seen = np.isfinite(b[:, 0])
    minutes = graded.sum() / FPS / 60.0
    cand = np.flatnonzero(live & seen)
    n_snap = int(rng.poisson(p["snap_per_min"] * minutes)) if len(cand) else 0
    for f in rng.choice(cand, size=min(n_snap, len(cand)), replace=False):
        e = min(n, f + int(rng.uniform(*p["snap_s"]) * FPS))
        seg = b[f:e]
        ok = np.isfinite(seg[:, 0])
        seg[ok] = b[f]

    # P-7 (e): false reappearance: a short gap, a detection near a random player, a short gap.
    n_false = int(rng.poisson(p["false_per_min"] * minutes))
    cand = np.flatnonzero(graded)
    for f in rng.choice(cand, size=min(n_false, len(cand)), replace=False):
        on = np.flatnonzero(np.isfinite(game.players[f, :, 0]))
        if not len(on):
            continue
        g1 = int(rng.integers(p["false_gap_frames"][0], p["false_gap_frames"][1] + 1))
        g2 = int(rng.integers(p["false_gap_frames"][0], p["false_gap_frames"][1] + 1))
        kk = int(rng.uniform(*p["false_s"]) * FPS)
        pos = game.players[f, rng.choice(on)] + rng.normal(0, p["false_sigma_m"], 2)
        m1, m2 = min(n, f + g1), min(n, f + g1 + kk)
        e = min(n, m2 + g2)
        b[f:m1] = np.nan
        b[m1:m2] = pos
        b[m2:e] = np.nan

    seen = np.isfinite(b[:, 0])
    info = {"model": "v2", "live_missing_target": p["live_missing"], "base_hazard": base,
            "clean_live_missing_after_fill": clean_live_missing,
            "live_missing": float((~seen[live]).mean()),
            "dead_missing": float((~seen[graded & ~live]).mean()),
            "graded_missing": float((~seen[graded]).mean()),
            "restart_visible_stoppages": n_rv, "exit_visible_stoppages": n_ev,
            "holds": n_hold, "snaps": n_snap, "false_reappearances": n_false}
    return b, info


def v2_players(game, labels, ball_noisy, level, rng):
    p = {**V2_COMMON, **V2_PARAMS[level]}
    n, P = game.n_frames, game.players.shape[1]
    pos = game.players.copy()
    on = np.isfinite(pos[:, :, 0])
    graded = labels != "X"

    # Independent per-player dropouts (camera edge), calibrated before the whole-scene loss.
    bx = pd.Series(ball_noisy[:, 0]).ffill().bfill().fillna(0.0)
    cam = bx.rolling(51, center=True, min_periods=1).mean().to_numpy()
    edge = np.abs(pos[:, :, 0] - cam[:, None]) > p["camera_half_width_m"]
    lo, hi = (int(s * FPS) for s in p["player_gap_s"])
    u = rng.random((n, P))
    lengths = rng.integers(lo, hi + 1, size=(n, P))
    mult = np.where(edge, p["edge_mult"], 1.0)

    def drop_for(base):
        out = np.zeros((n, P), bool)
        for j in range(P):
            out[:, j] = _runs_from_starts((u[:, j] < base * mult[:, j]) & on[:, j], lengths[:, j], n)
        return out

    target = p["players_mean_pre"] / 22.0
    lo_b, hi_b = 0.0, 0.05
    for _ in range(20):
        mid = 0.5 * (lo_b + hi_b)
        present = (on & ~drop_for(mid))[graded].sum() / max(1, on[graded].sum())
        lo_b, hi_b = (mid, hi_b) if present > target else (lo_b, mid)
    base = 0.5 * (lo_b + hi_b)
    pos[drop_for(base)] = np.nan

    # P-5: whole-scene loss. Events start more often while the ball is missing; during an event
    # each player is kept (for the whole event) with probability mass_keep_p.
    miss = ~np.isfinite(ball_noisy[:, 0])
    hz = np.where(miss, p["mass_missing_mult"], 1.0)
    hz_base = p["mass_rate_per_min"] / (FPS * 60.0) / hz.mean()
    starts = np.flatnonzero(rng.random(n) < hz_base * hz)
    mlen = _pareto_lengths(rng, n, p["mass_xmin_frames"], p["mass_alpha"], p["mass_cap_frames"])
    mass_frames = 0
    for s in starts:
        e = min(n, s + mlen[s])
        lost = rng.random(P) >= p["mass_keep_p"]
        pos[s:e, lost] = np.nan
        mass_frames += e - s

    if p["player_sigma_m"]:
        pos += rng.normal(0, p["player_sigma_m"], size=pos.shape)

    # Team-label flips (a player counted for the other team for a while).
    team_fr = np.repeat(game.player_team[None, :], n, axis=0).astype(np.int8)
    flen = rng.integers(int(p["team_flip_s"][0] * FPS), int(p["team_flip_s"][1] * FPS) + 1, (n, P))
    fu = rng.random((n, P))
    flips = np.zeros((n, P), bool)
    for j in range(P):
        flips[:, j] = _runs_from_starts(fu[:, j] < p["team_flip_rate"] / flen[:, j].mean(),
                                        flen[:, j], n)
    team_fr[flips] = 1 - team_fr[flips]

    # Duplicates far enough apart to survive the loader's 0.5 m merge.
    dlen = rng.integers(int(p["dup_len_s"][0] * FPS), int(p["dup_len_s"][1] * FPS) + 1, (n, P))
    du = rng.random((n, P))
    dups = np.zeros((n, P), bool)
    for j in range(P):
        dups[:, j] = _runs_from_starts(du[:, j] < p["dup_rate"] / dlen[:, j].mean(), dlen[:, j], n)
    ang = rng.uniform(0, 2 * np.pi, size=(n, P))
    rad = rng.uniform(*p["dup_offset_m"], size=(n, P))
    dup_pos = pos + np.stack([rad * np.cos(ang), rad * np.sin(ang)], -1)
    dup_pos[~dups] = np.nan

    all_pos = np.concatenate([pos, dup_pos], axis=1)
    team_all = np.concatenate([team_fr, team_fr], axis=1)
    order = rng.permutation(all_pos.shape[1])
    all_pos, team_all = all_pos[:, order], team_all[:, order]

    L, W = game.pitch
    X = all_pos[:, :, 0] + L / 2.0
    Y = W / 2.0 - all_pos[:, :, 1]
    VX, VY = _velocities(X, Y, game.halves())
    parts, tracked = [], np.zeros(n, int)
    for t in (0, 1):
        # Team membership is per frame (flips), so each slot is split by team before merging.
        Xt = np.where(team_all == t, X, np.nan)
        Yt = np.where(team_all == t, Y, np.nan)
        cols = np.flatnonzero(np.isfinite(Xt).any(axis=0))
        mx, my, mvx, mvy, alive, _ = merge_duplicates(Xt[:, cols], Yt[:, cols], VX[:, cols],
                                                      VY[:, cols])
        tracked += alive.sum(axis=1)
        f, s = np.nonzero(alive)
        parts.append(pd.DataFrame({"frame": f.astype(np.int32), "team": np.full(len(f), t, np.int8),
                                   "x_m": mx[f, s].astype(np.float32),
                                   "y_m": my[f, s].astype(np.float32),
                                   "vx": mvx[f, s].astype(np.float32),
                                   "vy": mvy[f, s].astype(np.float32), "_slot": s}))
    players = pd.concat(parts).sort_values(["frame", "team", "_slot"], kind="mergesort")
    players = players.drop(columns="_slot").reset_index(drop=True)
    tg = tracked[graded]
    info = {"model": "v2", "players_mean_pre_target": p["players_mean_pre"],
            "player_base_hazard": base, "mass_events": int(len(starts)),
            "mass_event_frames": int(mass_frames),
            "players_per_frame_graded_mean": float(tg.mean()),
            "players_per_frame_graded_max": int(tg.max()),
            "rate_below_18": float((tg < 18).mean()), "rate_below_10": float((tg < 10).mean()),
            "rate_above_22": float((tg > 22).mean()),
            "duplicate_player_frames": int(dups.sum()), "team_flip_player_frames": int(flips.sum())}
    return players, info


def root_name(game_id, level, model="v3"):
    """v3 -> <game>_<level>; v2 -> <game>_legacy2<level>; legacy (T-2) -> <game>_legacy<level>."""
    return "{}_{}{}".format(game_id, ROOT_TAG[model], level)


def build(game_id, level, game=None, model="v3", out_root=None, params_override=None,
          lowpass_sigma=None, lowpass_order=None, root=None):
    """Build one noisy root. model 'v3' -> <game>_<level>; 'v2' -> <game>_legacy2<level> (T-6
    noise); 'legacy' -> <game>_legacy<level> (the T-2 noise). Older models are kept so earlier
    numbers reproduce. params_override: {key: value} applied on top of the v2 level and common
    parameters (tuning only). lowpass_sigma / lowpass_order: v3 filter (tuning only; defaults
    V3_LOWPASS[level] and V3_COMMON). root: output folder name (tuning only)."""
    game = game or metrica_io.load_game(game_id)
    src = os.path.join(ART, game_id)
    labels = pd.read_csv(os.path.join(src, "truth.csv"))["label"].astype(str).to_numpy()
    stops = pd.read_csv(os.path.join(src, "stoppages.csv"))
    with open(os.path.join(src, "truth_meta.json")) as fh:
        tmeta = json.load(fh)
    rng = np.random.default_rng(_seed(game_id, level))
    if model == "legacy":
        ball, binfo = noisy_ball(game, labels, stops, level, rng)
        players, pinfo = noisy_players(game, labels, ball, level, rng)
        params = {**COMMON, **PARAMS[level]}
    else:
        saved_level, saved_common = dict(V2_PARAMS[level]), dict(V2_COMMON)
        try:
            if params_override:
                V2_PARAMS[level] = {**saved_level, **params_override}
            if model == "v3":
                sigma = V3_LOWPASS[level] if lowpass_sigma is None else lowpass_sigma
                order = lowpass_order or V3_COMMON["ball_lowpass_order"]
                ball, ball_v2, binfo = v3_ball(game, labels, stops, level, rng, sigma, order)
                # Players from the unfiltered ball: bit-identical to v2's players.
                players, pinfo = v2_players(game, labels, ball_v2, level, rng)
                params = {**V2_COMMON, **V2_PARAMS[level], **V3_COMMON,
                          "ball_lowpass_order": order, "ball_lowpass_sigma_frames": float(sigma)}
            else:
                ball, binfo = v2_ball(game, labels, stops, level, rng)
                players, pinfo = v2_players(game, labels, ball, level, rng)
                params = {**V2_COMMON, **V2_PARAMS[level]}
        finally:
            V2_PARAMS[level] = saved_level
            V2_COMMON.clear()
            V2_COMMON.update(saved_common)

    out = os.path.join(out_root or ART, root or root_name(game_id, level, model))
    os.makedirs(out, exist_ok=True)
    L, W = game.pitch
    bdf = pd.DataFrame({"frame": np.arange(game.n_frames, dtype=np.int32),
                        "x_m": (ball[:, 0] + L / 2.0).astype(np.float32),
                        "y_m": (W / 2.0 - ball[:, 1]).astype(np.float32)})
    bdf["seen"] = np.isfinite(bdf["x_m"].to_numpy())
    players.to_parquet(os.path.join(out, "players.parquet"), index=False)
    bdf.to_parquet(os.path.join(out, "ball_raw.parquet"), index=False)
    for name in ("truth.csv", "truth_first_touch.csv", "truth_confirmed.csv", "stoppages.csv"):
        shutil.copyfile(os.path.join(src, name), os.path.join(out, name))
    if os.path.exists(os.path.join(src, "truth_labels.csv")):    # bench.truth_labels (T-26)
        shutil.copyfile(os.path.join(src, "truth_labels.csv"), os.path.join(out, "truth_labels.csv"))
    attack, agreement = _attack_meta(players, game.n_frames, tmeta["halves"])
    meta = {"game_id": os.path.basename(out), "source": "bench", "n_frames": game.n_frames,
            "fps": float(FPS), "halves": tmeta["halves"], "graded_spans": tmeta["graded_spans"],
            "pitch": [L, W],
            "coordinates": "metres; x along the pitch length, y across; provider orientation",
            "frame_convention": "atd",
            # No delivered ball-speed channel: the match's raw_s is unusable (finite on 7.8% of
            # seen frames), so the bench offers none rather than a cleaner one.
            "ball_columns": ["x_m", "y_m", "seen"],
            "attack_direction": attack,
            "attack_direction_agreement": agreement,
            "attack_direction_note": "+1: team attacks towards increasing x_m; team 0 = H, 1 = A",
            "max_speed_mps": MAX_SPEED_MPS,
            "noise": {"model": model, "level": level, "seed": _seed(game_id, level),
                      "params": params, "ball": binfo, "players": pinfo}}
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True, default=float)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--games", nargs="*", default=None)
    ap.add_argument("--levels", nargs="*", default=["low", "match", "high"])
    ap.add_argument("--model", default="v3", choices=MODELS)
    args = ap.parse_args(argv)
    print("{:<26}{:>9}{:>9}{:>9}{:>9}{:>8}{:>8}{:>8}".format(
        "root", "liveMiss", "deadMiss", "gradMiss", "players", "<18", "<10", ">22"))
    for gid in args.games or metrica_io.available_games():
        game = metrica_io.load_game(gid)
        for level in args.levels:
            m = build(gid, level, game, model=args.model)
            b, pl = m["noise"]["ball"], m["noise"]["players"]

            def f(key):
                return "{:.3f}".format(pl[key]) if key in pl else "-"
            print("{:<26}{:>9.3f}{:>9.3f}{:>9.3f}{:>9.2f}{:>8}{:>8}{:>8}".format(
                m["game_id"], b["live_missing"], b["dead_missing"], b["graded_missing"],
                pl["players_per_frame_graded_mean"], f("rate_below_18"), f("rate_below_10"),
                f("rate_above_22")))


if __name__ == "__main__":
    main()
