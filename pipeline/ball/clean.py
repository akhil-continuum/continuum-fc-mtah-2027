"""Ball stage: ball_raw.parquet -> ball.parquet (rules B1, B2, B3; CONTRACTS.md).

Steps, each a function below so tests and later stages can call them:

1. ``reject_jumps`` (B1, P28): split the observed track wherever two consecutive observations
   imply more than MAX_SPEED_MPS (plus a jitter allowance). Short pieces cut off by such jumps
   are dropped (set unseen); they are false detections or a second ball.
2. ``static_marking_suspects`` (B2): a ball still for a long time on the centre spot or a
   penalty mark with no player near it is flagged; the point is kept with low confidence.
3. ``fill_gaps`` (B3): gaps of at most MAX_FILL frames are filled by linear interpolation only
   when the ball before and after lies on one straight flight (or is almost still on both
   sides). Filled frames have ``filled`` True and ``seen`` False. Longer gaps stay missing.
4. ``smooth``: positions by a centred local quadratic fit and velocities by a centred local
   linear fit, each inside contiguous runs of seen or filled frames only (never across a gap).

Missing stays missing: the ball-missing context (time since and until seen, last-seen position
and velocity, players near at loss) is in ``pipeline.ball.context`` and goes to features.
Everything is centred in time; nothing is causal.
"""
import numpy as np
import pandas as pd

# Params (start values from the rulebook; chosen on bench games only)
MAX_SPEED_MPS = 40.0        # B1
JUMP_SLACK_M = 1.0          # jitter allowance on top of MAX_SPEED_MPS * dt
MIN_PIECE_FRAMES = 5        # pieces shorter than this, cut off by a jump, are dropped
MAX_FILL = 12               # B3: longest gap filled, frames (~0.5 s)
ALWAYS_FILL = 2             # gaps this short are filled without the straightness test
FILL_FIT_FRAMES = 4         # observations each side used to estimate the flight
FILL_MAX_ANGLE_DEG = 25.0   # heading before/after vs the bridge
FILL_SPEED_RATIO = 2.0      # speeds before/after within this factor of the bridge speed
FILL_SLOW_MPS = 2.0         # both sides slower than this and a short bridge: fill
STILL_MPS = 0.5             # B2: "still"
MARKING_RADIUS_M = 1.0      # B2: within this of a mark
MARKING_STILL_S = 2.0       # B2: still at least this long
MARKING_CLEAR_M = 3.0       # B2: no player within this during the still run
SMOOTH_HALF = 2             # positions: quadratic fit over +/-2 frames
VEL_HALF = 3                # velocities: linear fit over +/-3 frames
CONF_SEEN, CONF_SUSPECT = 1.0, 0.2
CONF_FILL_MAX, CONF_FILL_MIN, CONF_FILL_STEP = 0.8, 0.4, 0.03
CONF_ISOLATED = 0.5         # seen runs shorter than ISOLATED_FRAMES between gaps
ISOLATED_FRAMES = 3


def runs(mask):
    """(starts, stops) of True runs in a boolean array, stop exclusive."""
    m = np.asarray(mask, dtype=np.int8)
    d = np.diff(np.r_[0, m, 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def half_breaks(n_frames, halves):
    """Boolean array, True on the first frame of each half (runs never cross half-time)."""
    br = np.zeros(n_frames, dtype=bool)
    for start, _stop in halves:
        if 0 <= start < n_frames:
            br[start] = True
    return br


def reject_jumps(x, y, seen, fps, breaks=None, max_speed=MAX_SPEED_MPS, slack=JUMP_SLACK_M,
                 min_piece=MIN_PIECE_FRAMES):
    """B1: (seen_after, n_rejected). Pieces shorter than min_piece bounded by a jump go."""
    idx = np.flatnonzero(seen)
    keep = seen.copy()
    if len(idx) < 2:
        return keep, 0
    dt = np.diff(idx) / fps
    step = np.hypot(np.diff(x[idx]), np.diff(y[idx]))
    jump = step > max_speed * dt + slack
    if breaks is not None:          # a new half starts a new piece but is not a jump
        newhalf = np.array([breaks[a + 1:b + 1].any() for a, b in zip(idx[:-1], idx[1:])])
    else:
        newhalf = np.zeros(len(jump), dtype=bool)
    piece = np.r_[0, np.cumsum(jump | newhalf)]
    sizes = np.bincount(piece)
    # Was the piece cut by a jump on either side?
    cut_before = np.r_[False, jump]
    cut_after = np.r_[jump, False]
    cut = np.zeros(len(sizes), dtype=bool)
    np.logical_or.at(cut, piece, cut_before | cut_after)
    drop_piece = (sizes < min_piece) & cut
    # Never drop every piece: if all are short, keep the longest.
    if drop_piece.all():
        drop_piece[np.argmax(sizes)] = False
    drop = drop_piece[piece]
    keep[idx[drop]] = False
    return keep, int(drop.sum())


def _fit_velocity(t, v):
    """Least-squares slope of v against t (at least 2 points)."""
    t = t - t.mean()
    den = (t * t).sum()
    return float((t * (v - v.mean())).sum() / den) if den > 0 else 0.0


def _angle_deg(a, b):
    na, nb = np.hypot(*a), np.hypot(*b)
    if na == 0 or nb == 0:
        return 0.0
    c = np.clip((a[0] * b[0] + a[1] * b[1]) / (na * nb), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def fill_gaps(x, y, seen, fps, breaks=None, max_fill=MAX_FILL):
    """B3: (x, y, filled, gap_len_of_fill). Linear fill of short straight-flight gaps."""
    x, y = x.copy(), y.copy()
    n = len(x)
    filled = np.zeros(n, dtype=bool)
    fill_len = np.zeros(n, dtype=np.int32)
    starts, stops = runs(~seen)
    for a, b in zip(starts, stops):
        length = b - a
        if a == 0 or b >= n or length > max_fill:
            continue
        if breaks is not None and breaks[a:b + 1].any():
            continue
        p0 = np.array([x[a - 1], y[a - 1]])
        p1 = np.array([x[b], y[b]])
        dt = (b - (a - 1)) / fps
        bridge = (p1 - p0) / dt
        ok = length <= ALWAYS_FILL
        if not ok:
            before = np.flatnonzero(seen[max(0, a - FILL_FIT_FRAMES - 2):a])
            before = before + max(0, a - FILL_FIT_FRAMES - 2)
            before = before[-FILL_FIT_FRAMES:]
            after = np.flatnonzero(seen[b:b + FILL_FIT_FRAMES + 2]) + b
            after = after[:FILL_FIT_FRAMES]
            if len(before) >= 2 and len(after) >= 2:
                tb, ta = before / fps, after / fps
                v0 = np.array([_fit_velocity(tb, x[before]), _fit_velocity(tb, y[before])])
                v1 = np.array([_fit_velocity(ta, x[after]), _fit_velocity(ta, y[after])])
                s0, s1, sb = np.hypot(*v0), np.hypot(*v1), np.hypot(*bridge)
                if max(s0, s1, sb) < FILL_SLOW_MPS:
                    ok = True
                elif min(s0, s1, sb) > 0:
                    ok = (_angle_deg(v0, bridge) <= FILL_MAX_ANGLE_DEG
                          and _angle_deg(v1, bridge) <= FILL_MAX_ANGLE_DEG
                          and max(s0, s1) / sb <= FILL_SPEED_RATIO
                          and sb / min(s0, s1) <= FILL_SPEED_RATIO)
        if not ok:
            continue
        w = (np.arange(a, b) - (a - 1)) / float(b - (a - 1))
        x[a:b] = p0[0] + w * (p1[0] - p0[0])
        y[a:b] = p0[1] + w * (p1[1] - p0[1])
        filled[a:b] = True
        fill_len[a:b] = length
    return x, y, filled, fill_len


def static_marking_suspects(x, y, seen, fps, pitch, near_player_m=None):
    """B2: boolean per frame, True on long still runs on a mark with no player near.

    ``near_player_m``: per-frame distance from the ball to the nearest player (any team), or None
    to skip the player test.
    """
    L, W = pitch
    marks = np.array([[L / 2.0, W / 2.0], [11.0, W / 2.0], [L - 11.0, W / 2.0]])
    n = len(x)
    out = np.zeros(n, dtype=bool)
    with np.errstate(invalid="ignore"):
        d_mark = np.min(np.hypot(x[:, None] - marks[None, :, 0], y[:, None] - marks[None, :, 1]),
                        axis=1)
        step = np.r_[np.inf, np.hypot(np.diff(x), np.diff(y)) * fps]
    still = seen & (d_mark <= MARKING_RADIUS_M) & (step <= STILL_MPS * 4)   # per-frame jitter
    min_len = int(round(MARKING_STILL_S * fps))
    for a, b in zip(*runs(still)):
        if b - a < min_len:
            continue
        # Net displacement over the run must also be small (truly still, not drifting).
        if np.hypot(x[b - 1] - x[a], y[b - 1] - y[a]) > MARKING_RADIUS_M:
            continue
        if near_player_m is not None:
            d = near_player_m[a:b]
            if np.any(np.isfinite(d) & (d <= MARKING_CLEAR_M)):
                continue
        out[a:b] = True
    return out


def _local_poly_fast(values, mask, half, order, deriv, fps):
    """Centred least-squares polynomial fit (value or first derivative) inside runs of ``mask``.

    Interior frames use a Savitzky-Golay kernel; frames within ``half`` of a run edge use the
    frames available on the run (window clipped, never crossing a gap). NaN outside runs.
    """
    from scipy.signal import savgol_coeffs
    n = len(values)
    out = _local_poly_edges(values, mask, half, order, deriv, fps)
    win = 2 * half + 1
    coeffs = savgol_coeffs(win, order, deriv=deriv, delta=1.0 / fps, use="dot")
    v = np.where(mask, values, 0.0)
    conv = np.convolve(v, coeffs[::-1], mode="same")
    # Interior frames: all of [i-half, i+half] inside the mask.
    full = np.convolve(mask.astype(np.int32), np.ones(win, dtype=np.int32), mode="same") == win
    out[full] = conv[full]
    return out


def _local_poly_edges(values, mask, half, order, deriv, fps):
    """Clipped-window fits on run edges only (frames whose window is clipped); NaN elsewhere."""
    n = len(values)
    out = np.full(n, np.nan)
    for a, b in zip(*runs(mask)):
        m = b - a
        seg = values[a:b]
        if m == 1:
            out[a] = seg[0] if deriv == 0 else np.nan
            continue
        edge = [i for i in range(m) if i < half or i >= m - half]
        for i in edge:
            lo, hi = max(0, i - half), min(m, i + half + 1)
            t = np.arange(lo, hi) - i
            k = min(order, len(t) - 1)
            coef = np.polyfit(t, seg[lo:hi], k)
            out[a + i] = coef[-1] if deriv == 0 else (coef[-2] * fps if k >= 1 else np.nan)
    return out


def smooth(x, y, mask, fps, pos_half=SMOOTH_HALF, vel_half=VEL_HALF):
    """(xs, ys, vx, vy): centred local fits inside runs of ``mask`` only."""
    xs = _local_poly_fast(x, mask, pos_half, 2, 0, fps)
    ys = _local_poly_fast(y, mask, pos_half, 2, 0, fps)
    vx = _local_poly_fast(x, mask, vel_half, 1, 1, fps)
    vy = _local_poly_fast(y, mask, vel_half, 1, 1, fps)
    return xs, ys, vx, vy


def clean_ball(ball_raw, meta, near_player_m=None):
    """ball_raw DataFrame + meta dict -> (ball DataFrame per CONTRACTS.md, stats dict)."""
    fps = float(meta["fps"])
    n = int(meta["n_frames"])
    pitch = tuple(meta["pitch"])
    br = half_breaks(n, meta["halves"])
    raw = ball_raw.sort_values("frame")
    x = np.full(n, np.nan)
    y = np.full(n, np.nan)
    f = raw["frame"].to_numpy()
    x[f] = raw["x_m"].to_numpy(dtype=float)
    y[f] = raw["y_m"].to_numpy(dtype=float)
    seen_raw = np.isfinite(x) & np.isfinite(y)

    seen, n_rej = reject_jumps(x, y, seen_raw, fps, br)
    x[~seen] = np.nan
    y[~seen] = np.nan
    suspect = static_marking_suspects(x, y, seen, fps, pitch, near_player_m)
    xf, yf, filled, fill_len = fill_gaps(x, y, seen, fps, br)
    have = seen | filled
    xs, ys, vx, vy = (np.full(n, np.nan) for _ in range(4))
    for start, stop in meta["halves"]:      # never smooth across half-time
        sl = slice(max(0, start), min(n, stop))
        xs[sl], ys[sl], vx[sl], vy[sl] = smooth(xf[sl], yf[sl], have[sl], fps)

    conf = np.zeros(n)
    conf[seen] = CONF_SEEN
    s_starts, s_stops = runs(seen)
    for a, b in zip(s_starts, s_stops):
        if b - a < ISOLATED_FRAMES:
            conf[a:b] = CONF_ISOLATED
    conf[filled] = np.clip(CONF_FILL_MAX - CONF_FILL_STEP * fill_len[filled], CONF_FILL_MIN,
                           CONF_FILL_MAX)
    conf[suspect] = CONF_SUSPECT

    out = pd.DataFrame({
        "frame": np.arange(n, dtype=np.int32),
        "x_m": xs.astype(np.float32), "y_m": ys.astype(np.float32),
        "vx": vx.astype(np.float32), "vy": vy.astype(np.float32),
        "seen": seen, "filled": filled, "conf": conf.astype(np.float32),
    })
    stats = {
        "frames": n,
        "share_seen_raw": float(seen_raw.mean()) if n else None,
        "share_seen": float(seen.mean()) if n else None,
        "share_filled": float(filled.mean()) if n else None,
        "share_missing": float((~have).mean()) if n else None,
        "rejected_points_b1": n_rej,
        "suspect_points_b2": int(suspect.sum()),
        "jitter_sigma_m_raw": jitter_sigma(x, y, seen),
    }
    return out, stats


def jitter_sigma(x, y, seen):
    """Robust per-axis position noise from second differences on consecutive seen frames (m).

    For white noise of scale s on a smooth path, the second difference has scale s * sqrt(6);
    the median absolute value (times 1.4826) estimates it. Real accelerations inflate it a little.
    """
    ok = seen[:-2] & seen[1:-1] & seen[2:]
    if ok.sum() < 50:
        return None
    d2 = np.r_[(x[2:] - 2 * x[1:-1] + x[:-2])[ok], (y[2:] - 2 * y[1:-1] + y[:-2])[ok]]
    return float(1.4826 * np.median(np.abs(d2)) / np.sqrt(6.0))


def gap_histogram(seen_or_filled, edges=(1, 2, 3, 5, 8, 13, 25, 50, 125, 250, 750, 1500)):
    """Counts of missing runs by length in frames; bins [e_i, e_i+1), last bin open."""
    starts, stops = runs(~np.asarray(seen_or_filled, dtype=bool))
    lengths = stops - starts
    bins = list(edges) + [10 ** 9]
    return list(edges), [int(v) for v in np.histogram(lengths, bins=bins)[0]]
