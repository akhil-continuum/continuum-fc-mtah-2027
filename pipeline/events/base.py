"""Shared inputs and the event table layout for the events stage.

events.parquet columns (CONTRACTS.md plus the extras marked +):

    frame             int32   the event's frame (exit: first frame the ball is out)
    kind              str     touch, flight, exit, carry, pass, deflection, control, keeper_gather
    team              int8    0 = H, 1 = A, -1 = none or unknown
    conf              float32 0..1
    end_frame       + int32   last frame of a span event (flight, carry, pass), else -1
    first_touch_frame int32   control only, else -1
    confirmed_frame   int32   control only, else -1
    line              str     exit only: touch_y0, touch_y1, goal_x0, goal_x1 ("" otherwise);
                              y0 is the touchline at y = 0, x0 the goal line at x = 0
    x_m, y_m          float32 exit: where the ball crossed the line; other kinds: ball position
    last_touch_team   int8    exit only: team that touched the ball last, -1 unknown
    last_touch_frame+ int32   exit only, frame of that touch, -1 unknown
    goal_mouth      + bool    exit only: crossed a goal line between the posts
    detail          + str     how it was found (exit: persist, vanish, extrapolated)

Integer columns use -1 for "not applicable"; there are no nullable columns.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline.ball.contact import contact

COLUMNS = ["frame", "kind", "team", "conf", "end_frame", "first_touch_frame", "confirmed_frame",
           "line", "x_m", "y_m", "last_touch_team", "last_touch_frame", "goal_mouth", "detail"]
INT_COLS = {"frame": np.int32, "team": np.int8, "end_frame": np.int32,
            "first_touch_frame": np.int32, "confirmed_frame": np.int32,
            "last_touch_team": np.int8, "last_touch_frame": np.int32}
DEFAULTS = {"team": -1, "conf": 0.0, "end_frame": -1, "first_touch_frame": -1,
            "confirmed_frame": -1, "line": "", "x_m": np.nan, "y_m": np.nan,
            "last_touch_team": -1, "last_touch_frame": -1, "goal_mouth": False, "detail": ""}


def event_frame(rows):
    """List of dicts -> events DataFrame with every column, sorted by frame then kind."""
    full = [{**DEFAULTS, **r} for r in rows]
    df = pd.DataFrame(full, columns=COLUMNS) if full else pd.DataFrame(
        {c: pd.Series(dtype="object") for c in COLUMNS})
    for c, t in INT_COLS.items():
        df[c] = df[c].astype(t)
    df["conf"] = df["conf"].astype(np.float32)
    df["x_m"] = df["x_m"].astype(np.float32)
    df["y_m"] = df["y_m"].astype(np.float32)
    df["goal_mouth"] = df["goal_mouth"].astype(bool)
    df["kind"] = df["kind"].astype(str)
    df["line"] = df["line"].astype(str)
    df["detail"] = df["detail"].astype(str)
    return df.sort_values(["frame", "kind"], kind="mergesort").reset_index(drop=True)


# Ball-player alignment, measured per game by code (never a constant tied to a game).
LAG_SEARCH = range(-10, 11)     # players shifted by L frames: player at f + L against ball at f
LAG_MIN_GAIN = 0.03             # apply a lag only if it lowers the median distance by 3 percent
                                # (bench SG3 clean and noisy: best non-zero lag gains under 2 percent)
BENCH_MEDIAN_D = 1.35           # median ball-to-nearest-player distance on bench SG3 (1.27 clean,
                                # 1.34 noisy), the scale the radii were set on
RADIUS_SCALE_MAX = 1.75
ALIGN_SAMPLE = 20000            # seen frames sampled (evenly) for the search


def _shift(players, lag):
    if lag == 0:
        return players
    p = players.copy()
    p["frame"] = p["frame"] - lag
    return p


def estimate_alignment(players, x, y, seen, n):
    """(lag, median distance at lag, median distance at 0, radius scale). Aggregates only."""
    idx = np.flatnonzero(seen & np.isfinite(x))
    if len(idx) < 200:
        return 0, None, None, 1.0
    step = max(1, len(idx) // ALIGN_SAMPLE)
    keep = np.zeros(n, dtype=bool)
    keep[idx[::step]] = True
    xs, ys = np.where(keep, x, np.nan), np.where(keep, y, np.nan)
    med = {}
    for lag in LAG_SEARCH:
        d = contact(_shift(players, lag), xs, ys, n).d_any[keep]
        d = d[np.isfinite(d)]
        med[lag] = float(np.median(d)) if len(d) else np.inf
    best = min(med, key=lambda k: (med[k], abs(k)))
    lag = best if med[best] < (1.0 - LAG_MIN_GAIN) * med[0] else 0
    scale = float(np.clip(med[lag] / BENCH_MEDIAN_D, 1.0, RADIUS_SCALE_MAX))
    return lag, med[lag], med[0], scale


@dataclass
class Inputs:
    """Everything a detector reads, as per-frame arrays (team level)."""
    n: int
    fps: float
    pitch: tuple
    halves: list
    attack: dict
    x: np.ndarray        # ball, from ball.parquet (smoothed; NaN where missing)
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    seen: np.ndarray
    filled: np.ndarray
    conf: np.ndarray
    con: object          # pipeline.ball.contact.Contact on the ball positions
    half: np.ndarray     # 1-based half index per frame, 0 outside halves
    lag: int = 0         # players shifted by this many frames to align with the ball
    radius_scale: float = 1.0   # contact radii multiplied by this (>= 1)
    align: dict = None   # aggregates of the alignment search
    players: object = None   # players.parquet shifted by ``lag`` (contact on another ball track)

    @property
    def have(self):
        return self.seen | self.filled

    @property
    def speed(self):
        return np.hypot(self.vx, self.vy)


def make_inputs(ball, players, meta, align=True):
    n = int(meta["n_frames"])
    get = lambda c: ball[c].to_numpy(dtype=float)          # noqa: E731
    x, y = get("x_m"), get("y_m")
    seen = ball["seen"].to_numpy(dtype=bool)
    lag, d_lag, d0, scale = (estimate_alignment(players, x, y, seen, n) if align
                             else (0, None, None, 1.0))
    half = np.zeros(n, dtype=np.int8)
    for h, (a, b) in enumerate(meta["halves"], 1):
        half[max(0, a):min(n, b)] = h
    shifted = _shift(players, lag)
    return Inputs(n=n, fps=float(meta["fps"]), pitch=tuple(meta["pitch"]),
                  halves=meta["halves"], attack=meta.get("attack_direction", {}),
                  x=x, y=y, vx=get("vx"), vy=get("vy"),
                  seen=ball["seen"].to_numpy(dtype=bool),
                  filled=ball["filled"].to_numpy(dtype=bool), conf=get("conf"),
                  con=contact(shifted, x, y, n), half=half, lag=int(lag), players=shifted,
                  radius_scale=scale,
                  align={"align_lag_frames": int(lag), "align_median_d_m": d_lag,
                         "align_median_d_lag0_m": d0, "radius_scale": round(scale, 3)})
