"""Per-frame team-level formation measures for restart detection (rulebook R6, S4 to S6, L6).

Inputs are the load-stage contract files only: players.parquet (frame, team, x_m, y_m, vx, vy),
meta.json (halves, attack_direction, pitch) and optionally ball_raw.parquet. Everything here is
computed per team from whoever is tracked on the frame; no track slot is ever followed (T7).

The pitch centre is the middle of the pitch in the provider orientation (pitch length / 2,
width / 2); nothing refers to a particular game.
"""
import numpy as np
import pandas as pd

# Params (rulebook pitch constants and starting values; set on the bench, never on the match)
CENTRE_CIRCLE_M = 9.15      # rulebook pitch constant
CIRCLE_MARGIN_M = 1.0       # tracking noise allowance at the circle edge
HALF_TOL_M = 1.0            # a player this far over the halfway line still counts as "own half"
SPOT_RADIUS_M = 4.0         # a kicking-team player at the ball is this close to the centre spot
JOG_SPEED_MPS = 2.0         # above this a player is "moving" (jogging or faster)
RING_M = 12.0               # defenders "lining the circle" stand within this of the centre spot


def frame_index(players, n_frames):
    """Row -> frame index array (int64) clipped to the frame range."""
    return players["frame"].to_numpy(dtype=np.int64)


def half_of_frames(halves, n_frames):
    """Array (n_frames,) of half numbers 1, 2, ... (0 outside every half)."""
    out = np.zeros(n_frames, dtype=np.int8)
    for k, (start, stop) in enumerate(halves, 1):
        out[max(0, start):min(n_frames, stop)] = k
    return out


def kickoff_frame_measures(players, meta, ball=None):
    """Per-frame measures of the kick-off formation (R6). Returns a DataFrame with one row per
    frame 0..n_frames-1:

    n_tracked            tracked players on the frame (both teams)
    n_wrong_half         players more than HALF_TOL_M inside the opponents' half
    n_circle_<t>         team t players inside the centre circle (minus the margin)
    n_ring_<t>           team t players within RING_M of the centre spot (circle included)
    d_spot_<t>           team t's nearest player distance to the centre spot (m, inf if none)
    share_moving         share of tracked players above JOG_SPEED_MPS (NaN velocities skipped)
    speed_median         median player speed (m/s)
    ball_d_spot          ball distance to the centre spot (NaN when unseen)
    """
    n_frames = int(meta["n_frames"])
    length, width = float(meta["pitch"][0]), float(meta["pitch"][1])
    cx, cy = length / 2.0, width / 2.0
    halves = [tuple(h) for h in meta["halves"]]
    half_frame = half_of_frames(halves, n_frames)

    f = frame_index(players, n_frames)
    team = players["team"].to_numpy(dtype=np.int64)
    x = players["x_m"].to_numpy(dtype=np.float64)
    y = players["y_m"].to_numpy(dtype=np.float64)
    vx = players["vx"].to_numpy(dtype=np.float64)
    vy = players["vy"].to_numpy(dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y) & (f >= 0) & (f < n_frames)
    f, team, x, y, vx, vy = f[ok], team[ok], x[ok], y[ok], vx[ok], vy[ok]

    # attack direction of each row's team in the row's half (0 outside halves)
    direction = np.zeros(len(f))
    attack = meta.get("attack_direction") or derive_attack_direction(f, team, x, half_frame)
    row_half = half_frame[f]
    for half_key, per_team in attack.items():
        h = int(half_key)
        for team_key, d in per_team.items():
            direction[(row_half == h) & (team == int(team_key))] = float(d)
    # a player attacking +x is in his own half when x <= centre (+ tolerance)
    over = (x - cx) * direction > HALF_TOL_M
    over &= direction != 0

    dist = np.hypot(x - cx, y - cy)
    in_circle = dist < (CENTRE_CIRCLE_M - CIRCLE_MARGIN_M)
    speed = np.hypot(vx, vy)
    has_speed = np.isfinite(speed)

    out = pd.DataFrame({"frame": np.arange(n_frames, dtype=np.int64),
                        "half": half_frame})
    out["n_tracked"] = np.bincount(f, minlength=n_frames)
    out["n_wrong_half"] = np.bincount(f, weights=over.astype(float), minlength=n_frames)
    for t in (0, 1):
        sel = team == t
        out["n_team_{}".format(t)] = np.bincount(f[sel], minlength=n_frames)
        out["n_circle_{}".format(t)] = np.bincount(f[sel], weights=in_circle[sel].astype(float),
                                                   minlength=n_frames)
        out["n_ring_{}".format(t)] = np.bincount(f[sel], weights=(dist[sel] < RING_M).astype(float),
                                                 minlength=n_frames)
        dmin = np.full(n_frames, np.inf)
        np.minimum.at(dmin, f[sel], dist[sel])
        out["d_spot_{}".format(t)] = dmin
    n_speed = np.bincount(f[has_speed], minlength=n_frames).astype(float)
    n_moving = np.bincount(f[has_speed], weights=(speed[has_speed] > JOG_SPEED_MPS).astype(float),
                           minlength=n_frames)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["share_moving"] = np.where(n_speed > 0, n_moving / n_speed, np.nan)
    out["speed_median"] = _group_median(f[has_speed], speed[has_speed], n_frames)

    if ball is not None and len(ball):
        bf = ball["frame"].to_numpy(dtype=np.int64)
        keep = (bf >= 0) & (bf < n_frames)
        bd = np.full(n_frames, np.nan)
        bx = ball["x_m"].to_numpy(dtype=np.float64)[keep]
        by = ball["y_m"].to_numpy(dtype=np.float64)[keep]
        bd[bf[keep]] = np.hypot(bx - cx, by - cy)
        out["ball_d_spot"] = bd
    else:
        out["ball_d_spot"] = np.nan
    return out


def derive_attack_direction(f, team, x, half_frame):
    """S7 fallback when meta.json has no attack_direction: per half, the team whose centroid
    sits at lower x on most frames defends the low-x goal and attacks +x. Returns the meta.json
    form {"<half>": {"0": +-1, "1": +-1}}."""
    n_frames = len(half_frame)
    out = {}
    sums, counts = [], []
    for t in (0, 1):
        sel = team == t
        sums.append(np.bincount(f[sel], weights=x[sel], minlength=n_frames))
        counts.append(np.bincount(f[sel], minlength=n_frames).astype(float))
    both = (counts[0] >= 5) & (counts[1] >= 5)
    with np.errstate(invalid="ignore", divide="ignore"):
        lower = (sums[0] / counts[0]) < (sums[1] / counts[1])
    for h in sorted(set(half_frame.tolist()) - {0}):
        sel = both & (half_frame == h)
        if not sel.any():
            continue
        home_low = np.mean(lower[sel]) >= 0.5
        out[str(h)] = {"0": 1 if home_low else -1, "1": -1 if home_low else 1}
    return out


def _group_median(groups, values, n_groups):
    """Median of values per integer group (NaN for empty groups)."""
    out = np.full(n_groups, np.nan)
    if len(values) == 0:
        return out
    order = np.lexsort((values, groups))
    g, v = groups[order], values[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    ends = np.r_[starts[1:], len(g)]
    mid_lo = v[starts + (ends - starts - 1) // 2]
    mid_hi = v[starts + (ends - starts) // 2]
    out[g[starts]] = 0.5 * (mid_lo + mid_hi)
    return out
