"""Ball-to-player geometry per frame, team level only (no slot is ever followed).

``contact(players, bx, by, n_frames)`` returns per frame and per team: the distance to the
nearest player, that player's position and velocity, and the counts of players within each
radius in RADII. Frames where the ball has no position get NaN distances and zero counts.
"""
import numpy as np

RADII = (1.0, 2.0, 3.0, 5.0)


class Contact:
    """Arrays of length n_frames; per-team arrays are indexed [team] with team 0 = H, 1 = A."""

    def __init__(self, n):
        self.d = np.full((2, n), np.nan)          # nearest distance per team
        self.px = np.full((2, n), np.nan)         # nearest player's position and velocity
        self.py = np.full((2, n), np.nan)
        self.pvx = np.full((2, n), np.nan)
        self.pvy = np.full((2, n), np.nan)
        self.count = {r: np.zeros((2, n), dtype=np.int16) for r in RADII}
        self.n_players = np.zeros((2, n), dtype=np.int16)

    @property
    def d_any(self):
        return np.fmin(self.d[0], self.d[1])

    @property
    def nearest_team(self):
        """0 or 1 for the team of the nearest player, -1 where unknown."""
        d0, d1 = self.d[0], self.d[1]
        out = np.full(d0.shape, -1, dtype=np.int8)
        out[np.isfinite(d0) & ~(d1 < d0)] = 0
        out[np.isfinite(d1) & ~(d0 <= d1)] = 1
        return out


def contact(players, bx, by, n_frames):
    """Contact arrays for ball positions bx, by (length n_frames, NaN where unknown)."""
    out = Contact(n_frames)
    f = players["frame"].to_numpy()
    team = players["team"].to_numpy().astype(np.int64)
    ok = (f >= 0) & (f < n_frames)
    f, team = f[ok], team[ok]
    x = players["x_m"].to_numpy(dtype=float)[ok]
    y = players["y_m"].to_numpy(dtype=float)[ok]
    vx = players["vx"].to_numpy(dtype=float)[ok]
    vy = players["vy"].to_numpy(dtype=float)[ok]
    np.add.at(out.n_players, (team, f), 1)
    d = np.hypot(x - bx[f], y - by[f])
    have = np.isfinite(d)
    f, team, x, y, vx, vy, d = (a[have] for a in (f, team, x, y, vx, vy, d))
    for r in RADII:
        np.add.at(out.count[r], (team, f), (d <= r).astype(np.int16))
    if len(d) == 0:
        return out
    # Nearest per (team, frame): sort by key then distance, take the first of each key.
    key = team * n_frames + f
    order = np.lexsort((d, key))
    key_s = key[order]
    first = np.r_[True, key_s[1:] != key_s[:-1]]
    sel = order[first]
    t_sel, f_sel = team[sel], f[sel]
    out.d[t_sel, f_sel] = d[sel]
    out.px[t_sel, f_sel] = x[sel]
    out.py[t_sel, f_sel] = y[sel]
    out.pvx[t_sel, f_sel] = vx[sel]
    out.pvy[t_sel, f_sel] = vy[sel]
    return out
