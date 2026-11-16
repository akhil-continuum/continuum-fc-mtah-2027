"""B4 touch and B5 flight.

touch (B4, STRONG): the ball's velocity changes sharply between a centred "before" window
[t - W, t] and "after" window [t, t + W] (vector change above DV_MPS, or a heading change above
ANGLE_DEG while moving), with a player within RADIUS_M around t. One event per local peak of the
change. Team = team of the nearest player at the peak. W, the threshold multiple and the player
window come from a named parameter set (PARAM_SETS below; "w7" by default, "w15s" for a low-pass
ball track). Without a player near it is a bounce or
an error and nothing is emitted. Velocities are least-squares slopes over seen frames only;
filled frames are interpolations and carry no touch evidence.

flight (B5, STRONG): a run of at least MIN_FLIGHT frames with ball speed above FLIGHT_MPS and no
player within RADIUS_M. One event per run, frame = first frame, end_frame = last frame,
team = team of the last touch before it (the passing team), -1 if none within FLIGHT_TOUCH_S.
"""
import numpy as np

from pipeline.ball.clean import runs

# Params (start values from the skill; chosen on bench games)
W = 7                    # frames in each one-sided window (bench: SG3 clean + noisy)
MIN_PTS = 4              # seen frames needed in each window
DV_MPS = 3.0             # floor of the velocity-change threshold
NOISE_K = 4.0            # threshold = max(DV_MPS, NOISE_K * noise scale of the change)
RAYLEIGH_MEDIAN = 1.1774 # median of a unit Rayleigh variable (2-D Gaussian noise magnitude)
ANGLE_DEG = 25.0
ANGLE_MIN_MPS = 3.0      # heading test only when both sides move at least this fast
RADIUS_M = 2.0
PEAK_HALF = 4            # one touch per +/-max(PEAK_HALF, W) frames
CONF_DV_FULL = 8.0       # velocity change giving full magnitude confidence
FLIGHT_MPS = 8.0
MIN_FLIGHT = 5
FLIGHT_TOUCH_S = 1.0

# Named parameter sets of the touch detector (B4). The module constants above are the "w7" set.
#   w7    sharp-kick ball (Metrica as published; the legacy and legacy2 bench roots). Default, so
#         every existing version and root reproduces byte for byte.
#   w15s  low-pass ball (a kick spread over about a second, like the competition match's and the
#         v3 bench roots'). Wider velocity windows measure the whole smeared change; a lower noise
#         multiple because the median change is now set by slow wobble, not jitter; the player
#         test over +-pwin frames because the smeared change can peak a few frames from the
#         contact; team at the peak frame and a peak score of dv alone (the heading term pulled
#         peaks late). Chosen on the v3 bench roots only (3 games x 3 levels, T-34 / P-17b,
#         a parameter sweep): the centre of a plateau W 13-16, k 1.5-2, pwin 4-8 where touch
#         F1 varies by under 0.01.
# A set is chosen explicitly (events stage ``--touch-set``); nothing switches automatically.
# pwin: the player test uses each team's nearest distance minimised over +-pwin frames.
# team_at: "min" = team of the nearest player over that window, "peak" = at the peak (+-1 frame).
# angle_w: weight of the heading change (degrees) in the peak score dv + angle_w * angle.
PARAM_SETS = {
    "w7": {"w": W, "min_pts": MIN_PTS, "noise_k": NOISE_K, "dv_floor": DV_MPS,
           "angle_deg": ANGLE_DEG, "radius": RADIUS_M, "pwin": 1, "team_at": "min",
           "angle_w": 0.1},
    "w15s": {"w": 15, "min_pts": 6, "noise_k": 1.75, "dv_floor": DV_MPS,
             "angle_deg": ANGLE_DEG, "radius": RADIUS_M, "pwin": 6, "team_at": "peak",
             "angle_w": 0.0},
}
DEFAULT_SET = "w7"


def touch_params(touch_set=None, **overrides):
    """The parameter dict of a named set, with any non-None override applied."""
    name = touch_set or DEFAULT_SET
    if name not in PARAM_SETS:
        raise ValueError("unknown touch parameter set {!r}; known: {}".format(
            name, ", ".join(sorted(PARAM_SETS))))
    p = dict(PARAM_SETS[name])
    for k, v in overrides.items():
        if v is None:
            continue
        if k not in p:
            raise ValueError("unknown touch parameter {!r}".format(k))
        p[k] = v
    return p


def window_slopes(v, m, lo, hi):
    """Per-frame least-squares slope (units per frame) of v over offsets lo..hi, mask m.

    Returns (slope, count). Slope NaN where fewer than 2 points.
    """
    n = len(v)
    s0 = np.zeros(n)
    s1 = np.zeros(n)
    s2 = np.zeros(n)
    sv = np.zeros(n)
    skv = np.zeros(n)
    vv = np.where(m, v, 0.0)
    mm = m.astype(float)
    for k in range(lo, hi + 1):
        sh_m = np.zeros(n)
        sh_v = np.zeros(n)
        if k >= 0:
            sh_m[:n - k] = mm[k:]
            sh_v[:n - k] = vv[k:]
        else:
            sh_m[-k:] = mm[:n + k]
            sh_v[-k:] = vv[:n + k]
        s0 += sh_m
        s1 += k * sh_m
        s2 += k * k * sh_m
        sv += sh_v
        skv += k * sh_v
    den = s0 * s2 - s1 * s1
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = np.where((s0 >= 2) & (den > 0), (s0 * skv - s1 * sv) / den, np.nan)
    return slope, s0


def velocity_change(inp, w=W, min_pts=MIN_PTS):
    """(dv, angle_deg, v_before, v_after) per frame from seen positions only."""
    m = inp.seen & np.isfinite(inp.x)
    bx, nb = window_slopes(inp.x, m, -w, 0)
    by, _ = window_slopes(inp.y, m, -w, 0)
    ax, na = window_slopes(inp.x, m, 0, w)
    ay, _ = window_slopes(inp.y, m, 0, w)
    ok = (nb >= min_pts) & (na >= min_pts) & m
    # Same half on both sides.
    h = inp.half
    same = np.ones(inp.n, dtype=bool)
    for k in range(1, w + 1):
        same[k:] &= h[k:] == h[:-k]
        same[:-k] &= h[:-k] == h[k:]
    ok &= same
    fps = inp.fps
    vb = np.stack([bx, by]) * fps
    va = np.stack([ax, ay]) * fps
    dv = np.where(ok, np.hypot(va[0] - vb[0], va[1] - vb[1]), np.nan)
    sb, sa = np.hypot(*vb), np.hypot(*va)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosang = (vb[0] * va[0] + vb[1] * va[1]) / (sb * sa)
    ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    ang = np.where(ok & (np.minimum(sb, sa) >= ANGLE_MIN_MPS), ang, 0.0)
    return dv, ang, vb, va


def _window_min(a, half):
    out = a.copy()
    for k in range(1, half + 1):
        out[k:] = np.fmin(out[k:], a[:-k])
        out[:-k] = np.fmin(out[:-k], a[k:])
    return out


def noise_scale(dv):
    """Per-component noise scale of the velocity change, from its median over the game.

    Most frames hold no touch, so the median of |dv| is set by tracking jitter; for 2-D Gaussian
    noise it is RAYLEIGH_MEDIAN times the per-component scale. One number per game.
    """
    ok = np.isfinite(dv)
    return float(np.median(dv[ok]) / RAYLEIGH_MEDIAN) if ok.sum() >= 50 else 0.0


LAST_STATS = {}


def touches(inp, touch_set=None, dv_thr=None, angle_thr=None, radius=None, w=None,
            noise_k=None, min_pts=None, pwin=None, team_at=None, angle_w=None, **extra):
    """List of touch event dicts. Aggregates of the last call are in LAST_STATS.

    ``touch_set`` names a set in PARAM_SETS (default DEFAULT_SET); keyword arguments override it
    (``extra``: other keys of the set; unknown keys raise).
    """
    p = touch_params(touch_set, dv_floor=dv_thr, angle_deg=angle_thr, radius=radius, w=w,
                     noise_k=noise_k, min_pts=min_pts, pwin=pwin, team_at=team_at,
                     angle_w=angle_w, **extra)
    w, noise_k, angle_thr = p["w"], p["noise_k"], p["angle_deg"]
    con = inp.con
    dv, ang, vb, va = velocity_change(inp, w=w, min_pts=p["min_pts"])
    scale = noise_scale(dv)
    dv_thr = max(p["dv_floor"], noise_k * scale)
    LAST_STATS.update(touch_param_set=touch_set or DEFAULT_SET,
                      touch_noise_scale_mps=round(scale, 3), touch_dv_threshold_mps=round(dv_thr, 3),
                      touch_dv_valid_share=round(float(np.isfinite(dv).mean()), 4))
    sb, sa = np.hypot(*vb), np.hypot(*va)
    ang = np.where(np.minimum(sb, sa) >= max(ANGLE_MIN_MPS, noise_k * scale), ang, 0.0)
    if p["team_at"] not in ("min", "peak"):
        raise ValueError("team_at must be 'min' or 'peak'")
    d_win = [_window_min(con.d[t], int(p["pwin"])) for t in (0, 1)]
    d_any = np.fmin(d_win[0], d_win[1])
    d_team = d_win
    if p["team_at"] == "peak" and int(p["pwin"]) != 1:
        d_team = [_window_min(con.d[t], 1) for t in (0, 1)]
    sharp = (dv > dv_thr) | (ang > angle_thr)
    score = np.where(sharp, np.nan_to_num(dv, nan=0.0) + p["angle_w"] * ang, -np.inf)
    rows = []
    idx = np.flatnonzero(sharp)
    taken = np.zeros(inp.n, dtype=bool)
    ph = max(PEAK_HALF, w)
    # Peaks of the change first (greedy non-maximum suppression over every sharp frame), then
    # the player test on the peak: a change whose peak has nobody near is a bounce or an error.
    peaks = []
    for i in idx[np.argsort(-score[idx], kind="stable")]:
        lo, hi = max(0, i - ph), min(inp.n, i + ph + 1)
        if taken[lo:hi].any():
            continue
        taken[i] = True
        peaks.append(i)
    radius = p["radius"] * getattr(inp, "radius_scale", 1.0)
    for i in peaks:
        if not d_any[i] <= radius:
            continue
        d0, d1 = d_team[0][i], d_team[1][i]
        if not (np.isfinite(d0) or np.isfinite(d1)):
            d0, d1 = d_win[0][i], d_win[1][i]     # team_at "peak", nobody tracked at the peak
        team = 0 if (np.isfinite(d0) and not d1 < d0) else 1
        own, other = (d0, d1) if team == 0 else (d1, d0)
        margin = (other - own) if np.isfinite(other) else 5.0
        mag = min(1.0, float(dv[i]) / CONF_DV_FULL) if np.isfinite(dv[i]) else 0.5
        sep = float(np.clip(0.5 + margin / 4.0, 0.3, 1.0))
        rows.append({"frame": int(i), "kind": "touch", "team": team,
                     "conf": round(mag * sep, 4), "x_m": inp.x[i], "y_m": inp.y[i],
                     "detail": "dv={:.1f}".format(float(dv[i]) if np.isfinite(dv[i]) else -1)})
    rows.sort(key=lambda r: r["frame"])
    return rows


def last_touch_before(touch_rows, frame, max_back):
    """The last touch strictly before ``frame`` and no earlier than frame - max_back, or None."""
    best = None
    for r in touch_rows:
        if r["frame"] >= frame:
            break
        if r["frame"] >= frame - max_back:
            best = r
    return best


def flights(inp, touch_rows, speed_thr=FLIGHT_MPS, radius=RADIUS_M, min_len=MIN_FLIGHT):
    radius = radius * getattr(inp, "radius_scale", 1.0)
    d_any = inp.con.d_any
    far = ~(d_any <= radius)          # NaN distance (no players) counts as far
    mask = inp.have & (inp.speed > speed_thr) & far
    rows = []
    for a, b in zip(*runs(mask)):
        if b - a < min_len or inp.half[a] != inp.half[b - 1]:
            continue
        t = last_touch_before(touch_rows, a + 1, int(FLIGHT_TOUCH_S * inp.fps))
        seen_share = float(inp.seen[a:b].mean())
        rows.append({"frame": int(a), "kind": "flight", "end_frame": int(b - 1),
                     "team": t["team"] if t else -1,
                     "conf": round(0.5 + 0.5 * seen_share, 4),
                     "x_m": inp.x[a], "y_m": inp.y[a]})
    return rows
