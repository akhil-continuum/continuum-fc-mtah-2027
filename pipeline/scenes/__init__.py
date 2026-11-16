"""Synthetic scenes: small scripted games whose answers are known, in the stage contract format.

Shared by the signal, restart and decoder tests. A scene is players and a ball moving along
piecewise-linear paths; it writes the same three files the load stage writes (players.parquet,
ball_raw.parquet, meta.json, see CONTRACTS.md), so any later stage runs on it unchanged.

    from pipeline.scenes import Scene

    sc = Scene(n_frames=150)                       # 6 s at 25 fps, pitch 105 x 68
    a = sc.player(0, [(0, 40, 34), (150, 52, 34)]) # home player walking forwards
    b = sc.player(0, [(0, 60, 40)])                # home team-mate standing still
    d = sc.player(1, [(0, 50, 20)])                # away defender
    sc.ball_with(a, 0, 50)                         # ball at a's feet, 0.5 m ahead, frames 0..50
    sc.ball([(50, None), (75, 60, 39.5)])          # pass: from where it is at 50 to b by 75
    sc.ball_with(b, 75, 150)
    sc.hide_ball(60, 66)                           # ball unseen for 6 frames in flight
    sc.expect("touch", 50, team=0)                 # the known answers, for the tests
    sc.write(root)                                 # or sc.players_df(), sc.ball_raw_df(), sc.meta()

Paths
-----
``keys`` are ``(frame, x, y)`` tuples in metres in the contract orientation (x along the length
0..L, y across 0..W). Positions are linear between keys. A player stands at his first key before
it and at his last key after it, and is tracked on every frame unless ``frames=(f0, f1)`` limits
him to f0..f1 (inclusive).
A ball key may give ``None`` for x to mean "wherever the ball already is at that frame".
Later ball calls overwrite earlier ones on the frames they cover.

Ready-made scenes with their answers are in ``pipeline.scenes.library``.
"""
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FLOAT = np.float32
VEL_HALF_WIDTH = 2      # same centred difference as the load stage


def _interp_keys(keys, n_frames, hold=False):
    """Piecewise-linear (n_frames, 2) path from (frame, x, y) keys.

    Outside the keys: NaN, or the first/last key's position with ``hold``.
    """
    keys = sorted(keys, key=lambda k: k[0])
    out = np.full((n_frames, 2), np.nan)
    if len(keys) == 1:
        out[:] = keys[0][1:3]
        return out
    f = np.array([k[0] for k in keys], dtype=float)
    frames = np.arange(n_frames, dtype=float)
    inside = np.ones(n_frames, dtype=bool) if hold else (frames >= f[0]) & (frames <= f[-1])
    for axis in (0, 1):
        v = np.array([k[1 + axis] for k in keys], dtype=float)
        out[inside, axis] = np.interp(frames[inside], f, v)
    return out


def _centred_velocity(pos, fps):
    """Centred difference over +/-2 frames (falling back to +/-1), NaN at gaps; as the loader."""
    out = np.full(pos.shape, np.nan)
    for k in range(VEL_HALF_WIDTH, 0, -1):
        fwd = np.full(pos.shape, np.nan)
        back = np.full(pos.shape, np.nan)
        fwd[:-k] = pos[k:]
        back[k:] = pos[:-k]
        v = (fwd - back) * fps / (2.0 * k)
        fill = np.isnan(out) & np.isfinite(v)
        out[fill] = v[fill]
    out[np.isnan(pos)] = np.nan
    return out


@dataclass
class Expected:
    """One known answer: an event the detectors should (or, with absent=True, should not) emit."""
    kind: str
    frame: int
    team: int = None
    absent: bool = False
    extra: dict = field(default_factory=dict)


class Scene:
    """A scripted scene. See the module docstring."""

    def __init__(self, n_frames, pitch=(105.0, 68.0), fps=25.0, halves=None, attack=None):
        self.n_frames = int(n_frames)
        self.pitch = (float(pitch[0]), float(pitch[1]))
        self.fps = float(fps)
        self.halves = halves or [[0, self.n_frames]]
        # +1: the team attacks towards increasing x. Default: home attacks +x in every half.
        self.attack = attack or {str(h): {"0": 1, "1": -1}
                                 for h in range(1, len(self.halves) + 1)}
        self._players = []          # (team, (n_frames, 2) path)
        self._ball = np.full((self.n_frames, 2), np.nan)
        self._hidden = np.zeros(self.n_frames, dtype=bool)
        self.expected = []

    # Building -------------------------------------------------------------------------------

    def player(self, team, keys, frames=None):
        """Add a player of team 0 (H) or 1 (A) on a path of (frame, x, y) keys. Returns its id.

        ``frames=(f0, f1)``: tracked only on f0..f1 inclusive (default: every frame).
        """
        path = _interp_keys(keys, self.n_frames, hold=True)
        if frames is not None:
            keep = np.zeros(self.n_frames, dtype=bool)
            keep[frames[0]:frames[1] + 1] = True
            path[~keep] = np.nan
        self._players.append((int(team), path))
        return len(self._players) - 1

    def player_pos(self, pid, frame):
        """(x, y) of player ``pid`` at ``frame``."""
        return tuple(self._players[pid][1][frame])

    def ball(self, keys):
        """Set the ball on frames first key .. last key along (frame, x, y) keys.

        ``(frame, None)`` or ``(frame, None, None)`` takes the ball's current position there.
        """
        resolved = []
        for k in keys:
            if len(k) < 3 or k[1] is None:
                pos = self._ball[k[0]]
                if not np.all(np.isfinite(pos)):
                    raise ValueError("ball position unknown at frame {}".format(k[0]))
                resolved.append((k[0], float(pos[0]), float(pos[1])))
            else:
                resolved.append((k[0], float(k[1]), float(k[2])))
        path = _interp_keys(resolved, self.n_frames)
        lo, hi = min(k[0] for k in resolved), max(k[0] for k in resolved)
        self._ball[lo:hi + 1] = path[lo:hi + 1]

    def ball_with(self, pid, f0, f1, ahead=0.5, side=0.0):
        """Ball travels with player ``pid`` on frames f0..f1, ``ahead`` metres along his motion.

        A still player keeps the ball ``ahead`` metres towards the side his team attacks.
        """
        team, path = self._players[pid]
        vel = pd.DataFrame(_centred_velocity(path, self.fps)).ffill().bfill().to_numpy()
        sign = self.attack["1"][str(team)]
        for f in range(f0, f1 + 1):
            v = vel[f]
            speed = np.hypot(v[0], v[1]) if np.all(np.isfinite(v)) else 0.0
            u = v / speed if speed > 0.3 else np.array([float(sign), 0.0])
            normal = np.array([-u[1], u[0]])
            self._ball[f] = path[f] + ahead * u + side * normal

    def hide_ball(self, f0, f1):
        """Ball unobserved on frames f0..f1 inclusive."""
        self._hidden[f0:f1 + 1] = True

    def expect(self, kind, frame, team=None, absent=False, **extra):
        """Record a known answer. absent=True means the detector must NOT emit it near frame."""
        self.expected.append(Expected(kind, int(frame), team, absent, extra))

    def jittered(self, ball_sigma=0.25, player_sigma=0.15, seed=0):
        """A copy with Gaussian position noise (the bench noise model's jitter, by default)."""
        rng = np.random.default_rng(seed)
        out = Scene(self.n_frames, self.pitch, self.fps, self.halves, self.attack)
        out._players = [(t, p + rng.normal(0, player_sigma, p.shape)) for t, p in self._players]
        out._ball = self._ball + rng.normal(0, ball_sigma, self._ball.shape)
        out._hidden = self._hidden.copy()
        out.expected = list(self.expected)
        return out

    # Contract outputs -----------------------------------------------------------------------

    def ball_xy(self):
        """(n_frames, 2) observed ball, NaN where hidden or unset."""
        b = self._ball.copy()
        b[self._hidden] = np.nan
        return b

    def ball_raw_df(self):
        b = self.ball_xy()
        seen = np.isfinite(b[:, 0]) & np.isfinite(b[:, 1])
        return pd.DataFrame({"frame": np.arange(self.n_frames, dtype=np.int32),
                             "x_m": b[:, 0].astype(FLOAT), "y_m": b[:, 1].astype(FLOAT),
                             "seen": seen})

    def players_df(self):
        parts = []
        for slot, (team, path) in enumerate(self._players):
            vel = _centred_velocity(path, self.fps)
            ok = np.isfinite(path[:, 0])
            f = np.flatnonzero(ok)
            parts.append(pd.DataFrame({
                "frame": f.astype(np.int32), "team": np.full(len(f), team, dtype=np.int8),
                "x_m": path[f, 0].astype(FLOAT), "y_m": path[f, 1].astype(FLOAT),
                "vx": vel[f, 0].astype(FLOAT), "vy": vel[f, 1].astype(FLOAT),
                "_slot": np.full(len(f), slot)}))
        if not parts:
            return pd.DataFrame({"frame": np.array([], np.int32), "team": np.array([], np.int8),
                                 **{c: np.array([], FLOAT) for c in ("x_m", "y_m", "vx", "vy")}})
        df = pd.concat(parts, ignore_index=True).sort_values(["frame", "team", "_slot"],
                                                             kind="mergesort")
        return df.drop(columns="_slot").reset_index(drop=True)

    def meta(self):
        return {"game_id": "scene", "source": "scene", "n_frames": self.n_frames,
                "fps": self.fps, "halves": self.halves, "attack_direction": self.attack,
                "pitch": list(self.pitch), "ball_columns": ["x_m", "y_m", "seen"]}

    def run(self, events=True):
        """Run the ball stage (and the events stage) in memory. Returns (ball, events or None)."""
        from pipeline.ball import clean
        from pipeline.ball.contact import contact
        raw = self.ball_raw_df()
        players = self.players_df()
        b = self.ball_xy()
        near = contact(players, b[:, 0], b[:, 1], self.n_frames).d_any
        ball, _ = clean.clean_ball(raw, self.meta(), near_player_m=near)
        if not events:
            return ball, None
        from pipeline.events import detect
        return ball, detect(ball, players, self.meta())

    def write(self, root):
        """Write players.parquet, ball_raw.parquet and meta.json to ``root``. Returns root."""
        os.makedirs(root, exist_ok=True)
        self.players_df().to_parquet(os.path.join(root, "players.parquet"), index=False)
        self.ball_raw_df().to_parquet(os.path.join(root, "ball_raw.parquet"), index=False)
        with open(os.path.join(root, "meta.json"), "w") as handle:
            json.dump(self.meta(), handle, indent=1, sort_keys=True)
        return root
