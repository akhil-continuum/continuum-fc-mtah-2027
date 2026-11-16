"""Team-shape features from players (rulebook T1 to T4, T7), written into features.parquet.

Every column is a home-minus-away difference of a per-team statistic, so it is mirror-symmetric
by construction: swapping the team labels and flipping the pitch (x -> L - x, y -> W - y,
velocities negated, attack directions swapped and negated) negates every ``*_diff_*`` column and
leaves ``shape_valid*`` unchanged. Per-team statistics are means, medians, shares and trimmed
ranges, never raw counts (T7), so a team with 8 tracked players gives about the same value as
one with 11. A frame where either team has fewer than MIN_TEAM_PLAYERS tracked players is
invalid: its statistics are NaN and ``shape_valid`` is 0. Windows are centred means over
+-0.5, 1, 2 and 4 s that ignore invalid frames (NaN when the whole window is invalid).

Per-team statistics (team t, its tracked players on the frame):
- fwd_vel   (T3): mean velocity along the team's own attack direction (meta.json
                  attack_direction; NaN when the direction is not known), m/s.
- speed     (T3, T9): mean speed, m/s.
- depth     (T1): trimmed x range, 90th minus 10th percentile, m.
- width     (T1): trimmed y range, 90th minus 10th percentile, m.
  depth and width also come as ``*_rel_*``: the difference minus its median over the half's valid
  frames, so a team that is always wider does not become a constant lean (T1 is about change).
- press     (T4): share of the team's tracked players within PRESS_RADIUS_M of the ball's last
                  seen position, while the ball was seen within BALL_RECENT_S (else NaN).
- approach  (T4): mean speed towards that ball position of those players (NaN when none).

The ball-free statistics (fwd_vel, speed, depth, width) carry the load when the ball is unseen.
"""
import numpy as np
import pandas as pd

MIN_TEAM_PLAYERS = 6        # T7: fewer tracked players than this in either team -> invalid frame
PRESS_RADIUS_M = 10.0       # T4: radius around the ball (insensitive to a ~2 m ball offset)
BALL_RECENT_S = 2.0         # T4: pressure only while the ball was seen this recently
TRIM_Q = (0.10, 0.90)       # T1/T7: trimmed range
WINDOWS_S = {0.5: "c05", 1.0: "c1", 2.0: "c2", 4.0: "c4"}
STATS = ("fwd_vel", "speed", "depth", "width", "press", "approach")
RELATIVE = ("depth", "width")   # also emitted centred on their half median ("rel")


def _centred_mean(a, half):
    """Centred moving mean over +-half frames ignoring NaN; NaN where no finite value."""
    a = np.asarray(a, dtype=float)
    ok = np.isfinite(a)
    c = np.concatenate([[0.0], np.cumsum(np.where(ok, a, 0.0))])
    k = np.concatenate([[0], np.cumsum(ok)])
    n = len(a)
    i = np.arange(n)
    lo, hi = np.maximum(i - half, 0), np.minimum(i + half + 1, n)
    num, den = c[hi] - c[lo], k[hi] - k[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _half_centred(diff, half):
    """diff minus its median over the valid frames of the same half (NaN outside the halves):
    removes a team's structural shape (one side always wider) and keeps in-game variation."""
    out = np.full(len(diff), np.nan)
    for h in np.unique(half[half > 0]):
        sel = half == h
        fin = diff[sel][np.isfinite(diff[sel])]
        if len(fin):
            out[sel] = diff[sel] - float(np.median(fin))
    return out


def _attack_sign(meta, frames, teams, n):
    """+1 / -1 attack direction of each player row's team in its half; NaN when unknown."""
    half = np.zeros(n, dtype=np.int16)
    for h, (a, b) in enumerate(meta["halves"], 1):
        half[max(0, int(a)):min(n, int(b))] = h
    ad = meta.get("attack_direction") or {}
    out = np.full(len(frames), np.nan)
    hf = half[np.clip(frames, 0, n - 1)]
    for h in range(1, len(meta["halves"]) + 1):
        for t in (0, 1):
            d = ad.get(str(h), {}).get(str(t))
            if d is not None:
                out[(hf == h) & (teams == t)] = float(d)
    return out


def _last_seen_ball(ball, fps):
    """Last seen ball position per frame, NaN when the ball was last seen over BALL_RECENT_S ago."""
    seen = ball["seen"].to_numpy(bool)
    n = len(seen)
    idx = np.maximum.accumulate(np.where(seen, np.arange(n), -1))
    recent = (idx >= 0) & ((np.arange(n) - idx) <= BALL_RECENT_S * fps)
    j = np.clip(idx, 0, n - 1)
    bx = np.where(recent, ball["x_m"].to_numpy(float)[j], np.nan)
    by = np.where(recent, ball["y_m"].to_numpy(float)[j], np.nan)
    return bx, by


def team_stats(ball, players, meta):
    """Per-team, per-frame statistics: {stat: (n, 2) array}, plus 'count' (n, 2)."""
    n = int(meta["n_frames"])
    fps = float(meta["fps"])
    f = players["frame"].to_numpy().astype(np.int64)
    t = players["team"].to_numpy().astype(np.int64)
    keep = (f >= 0) & (f < n) & ((t == 0) | (t == 1))
    f, t = f[keep], t[keep]
    x = players["x_m"].to_numpy(float)[keep]
    y = players["y_m"].to_numpy(float)[keep]
    vx = players["vx"].to_numpy(float)[keep]
    vy = players["vy"].to_numpy(float)[keep]
    sgn = _attack_sign(meta, f, t, n)
    bx, by = _last_seen_ball(ball, fps)
    dx, dy = bx[f] - x, by[f] - y
    dist = np.hypot(dx, dy)
    with np.errstate(invalid="ignore", divide="ignore"):
        near = np.where(np.isfinite(dist), (dist < PRESS_RADIUS_M).astype(float), np.nan)
        approach = np.where(near == 1.0, (vx * dx + vy * dy) / np.maximum(dist, 0.1), np.nan)
    key = f * 2 + t
    df = pd.DataFrame({"k": key, "x": x, "y": y, "fwd": vx * sgn, "spd": np.hypot(vx, vy),
                       "near": near, "app": approach})
    g = df.groupby("k", sort=True)
    out = {"count": np.zeros((n, 2))}
    cnt = g.size()
    out["count"][cnt.index.to_numpy() // 2, cnt.index.to_numpy() % 2] = cnt.to_numpy()

    def put(series):
        arr = np.full((n, 2), np.nan)
        idx = series.index.to_numpy()
        arr[idx // 2, idx % 2] = series.to_numpy(float)
        return arr

    out["fwd_vel"] = put(g["fwd"].mean())
    out["speed"] = put(g["spd"].mean())
    qx = g["x"].quantile(list(TRIM_Q)).unstack()
    qy = g["y"].quantile(list(TRIM_Q)).unstack()
    out["depth"] = put(qx[TRIM_Q[1]] - qx[TRIM_Q[0]])
    out["width"] = put(qy[TRIM_Q[1]] - qy[TRIM_Q[0]])
    out["press"] = put(g["near"].mean())
    out["approach"] = put(g["app"].mean())
    return out


def build(ball, players, meta):
    """Shape columns for features.parquet (names in FEATURES.md, section Team shape)."""
    n = int(meta["n_frames"])
    fps = float(meta["fps"])
    st = team_stats(ball, players, meta)
    valid = (st["count"][:, 0] >= MIN_TEAM_PLAYERS) & (st["count"][:, 1] >= MIN_TEAM_PLAYERS)
    cols = {"shape_valid": valid.astype(float),
            "shape_valid_c1": _centred_mean(valid.astype(float), int(round(fps)))}
    half = np.zeros(n, dtype=np.int16)
    for h, (a, b) in enumerate(meta["halves"], 1):
        half[max(0, int(a)):min(n, int(b))] = h
    for s in STATS:
        diff = st[s][:, 0] - st[s][:, 1]
        diff = np.where(valid, diff, np.nan)
        series = {"diff": diff}
        if s in RELATIVE:
            series["rel"] = _half_centred(diff, half)
        for kind, arr in series.items():
            for w, suffix in WINDOWS_S.items():
                cols["shape_{}_{}_{}".format(s, kind, suffix)] = _centred_mean(
                    arr, int(round(w * fps)))
    assert all(len(v) == n for v in cols.values())
    return cols
