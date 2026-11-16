"""Small scripted scenes for the restart tests (kick-off formation, live play, traps).

Temporary home: these should move to the shared synthetic scene builder (signal-engineer,
pipeline/scenes/) once it lands. Everything is deterministic (fixed seeds).

A scene is a list of per-frame "phases"; each phase function returns (home_xy, away_xy, ball_xy)
for a frame. Output matches the load-stage contract: players DataFrame (frame, team, x_m, y_m,
vx, vy), ball DataFrame (frame, x_m, y_m, seen) and a meta dict.
"""
import numpy as np
import pandas as pd

L, W = 105.0, 68.0
CX, CY = L / 2.0, W / 2.0
FPS = 25
REACT_DEF_S = 0.4    # defenders start running in this long after the kick
REACT_FWD_S = 0.5    # kicking forwards start crossing halfway this long after the kick


def kickoff_shape(kick_team, home_dir, taker_offset=0.3, taker_tracked=True, ring_m=10.0):
    """22 players in a legal kick-off formation. home_dir=+1: home defends low x (own half is
    x < CX). Returns (home (11, 2), away (11, 2))."""
    rng = np.random.default_rng(1)

    def own_half_points(direction, n):
        # spread over own half, at least 12 m from the centre spot
        pts = []
        while len(pts) < n:
            x = CX - direction * rng.uniform(5.0, 45.0)
            y = rng.uniform(4.0, W - 4.0)
            if np.hypot(x - CX, y - CY) > 12.0:
                pts.append((x, y))
        return np.array(pts)

    teams = {}
    for team, direction in ((0, home_dir), (1, -home_dir)):
        if team == kick_team:
            centre = [(CX - direction * taker_offset, CY), (CX - direction * 0.5, CY + 2.0)]
            if not taker_tracked:
                centre = [(np.nan, np.nan), (np.nan, np.nan)]
            forwards = [(CX - direction * 0.5, CY + 10.5), (CX - direction * 0.5, CY - 10.5)]
            rest = own_half_points(direction, 7)
            pts = np.vstack([centre, forwards, rest])
        else:
            ring = [(CX - direction * ring_m * np.cos(a), CY + ring_m * np.sin(a))
                    for a in (0.0, 0.5, -0.5)]
            pts = np.vstack([ring, own_half_points(direction, 8)])
        teams[team] = pts
    return teams[0], teams[1]


def live_shape(frame, seed=7, home_dir=1):
    """Open play: both teams spread over most of the pitch (many players in the other half),
    each team's centroid on its defensive side as in real play; moving slowly; ball with a
    player. home_dir=+1: home defends low x."""
    rng = np.random.default_rng(seed)
    lo, hi = (5.0, 80.0) if home_dir == 1 else (25.0, 100.0)
    base_h = np.column_stack([rng.uniform(lo, hi, 11), rng.uniform(4, W - 4, 11)])
    base_a = np.column_stack([L - rng.uniform(lo, hi, 11), rng.uniform(4, W - 4, 11)])
    t = frame / FPS
    drift = np.array([3.0 * np.sin(0.3 * t), 2.0 * np.cos(0.2 * t)])
    home = np.clip(base_h + drift, 1, [L - 1, W - 1])
    away = np.clip(base_a - drift, 1, [L - 1, W - 1])
    return home, away, home[3] + 0.5


class Scene:
    """Build frames phase by phase, then export contract DataFrames."""

    def __init__(self, halves, home_dir_by_half=None):
        self.halves = [tuple(h) for h in halves]
        self.n_frames = self.halves[-1][1]
        self.home_dir = home_dir_by_half or {h: (1 if h % 2 else -1)
                                             for h in range(1, len(self.halves) + 1)}
        self.home = np.full((self.n_frames, 11, 2), np.nan)
        self.away = np.full((self.n_frames, 11, 2), np.nan)
        self.ball = np.full((self.n_frames, 2), np.nan)

    def half_of(self, frame):
        for k, (a, b) in enumerate(self.halves, 1):
            if a <= frame < b:
                return k
        return 0

    def live(self, start, stop, seed=7):
        for f in range(start, stop):
            h, a, b = live_shape(f, seed, self.home_dir[self.half_of(f)])
            self.home[f], self.away[f], self.ball[f] = h, a, b
        return self

    def kickoff(self, start, kick, stop, kick_team, ball_seen=True, taker_tracked=True,
                ball_visible_after=True):
        """Formation from start to kick (ball on the spot), then the kick: ball rolls back into
        the kicking team's half at 8 m/s, the defending ring runs in at 5 m/s, two kicking
        forwards cross halfway; from kick + 2 s to stop the scene is open play."""
        half = self.half_of(start)
        d = self.home_dir[half]
        home0, away0 = kickoff_shape(kick_team, d, taker_tracked=taker_tracked)
        kdir = d if kick_team == 0 else -d       # kicking team's attack direction
        for f in range(start, stop):
            t = (f - kick) / FPS
            home, away = home0.copy(), away0.copy()
            if f < kick:
                ball = np.array([CX, CY]) if ball_seen else np.array([np.nan, np.nan])
            elif t < 2.0:
                ball = np.array([CX - kdir * 8.0 * t, CY])
                if not ball_visible_after:
                    ball = np.array([np.nan, np.nan])
                kick_side, def_side = (home, away) if kick_team == 0 else (away, home)
                # reaction times like the bench (formation break about 17 frames after the kick)
                t_def, t_fwd = max(0.0, t - REACT_DEF_S), max(0.0, t - REACT_FWD_S)
                # defending ring runs towards the ball
                for i in range(3):
                    v = ball - def_side[i]
                    n = np.linalg.norm(v)
                    if n > 0.5:
                        def_side[i] = def_side[i] + v / n * min(n, 5.0 * t_def)
                # two kicking forwards cross halfway
                for i in (2, 3):
                    kick_side[i, 0] = CX + kdir * (-0.5 + 6.0 * t_fwd)
            else:
                home, away, ball = live_shape(f, home_dir=d)
            self.home[f], self.away[f], self.ball[f] = home, away, ball
        return self

    def walk_on(self, start, stop):
        """Before a half's kick-off: players standing in their own halves, away from the centre,
        no ball (no formation: nobody lines the circle, no taker)."""
        half = self.half_of(start)
        d = self.home_dir[half]
        rng = np.random.default_rng(3)
        home = np.column_stack([CX - d * rng.uniform(15, 45, 11), rng.uniform(4, W - 4, 11)])
        away = np.column_stack([CX + d * rng.uniform(15, 45, 11), rng.uniform(4, W - 4, 11)])
        for f in range(start, stop):
            self.home[f], self.away[f] = home, away
            self.ball[f] = np.nan
        return self

    def frames(self, start, stop, home, away, ball):
        for f in range(start, stop):
            self.home[f], self.away[f], self.ball[f] = home, away, ball
        return self

    def shared(self):
        """The same scene as a pipeline.scenes.Scene (the shared builder writes the contract
        files and computes velocities exactly as for every other test). Each of the 22 player
        tracks becomes one path; NaN frames are untracked."""
        from pipeline.scenes import Scene as SharedScene
        attack = {str(h): {"0": d, "1": -d} for h, d in self.home_dir.items()}
        sc = SharedScene(self.n_frames, pitch=(L, W), fps=FPS,
                         halves=[list(h) for h in self.halves], attack=attack)
        for team, arr in ((0, self.home), (1, self.away)):
            for i in range(arr.shape[1]):
                sc._players.append((team, arr[:, i, :].copy()))
        sc._ball = self.ball.copy()
        return sc

    def players(self):
        return self.shared().players_df()

    def ball_raw(self):
        return self.shared().ball_raw_df()

    def meta(self, with_direction=True):
        m = {"n_frames": self.n_frames, "halves": [list(h) for h in self.halves],
             "pitch": [L, W], "fps": FPS}
        if with_direction:
            m["attack_direction"] = {str(h): {"0": d, "1": -d} for h, d in self.home_dir.items()}
        return m
