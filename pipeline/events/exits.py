"""D1 to D3: the ball leaving the pitch, with line, crossing point and last touch team.

D1 (LAW): out when the whole ball is over a touchline or goal line. A seen ball whose centre is
more than MARGIN_M beyond a line is "out" on that frame (ball radius plus tracking jitter).

D2 (STRONG): an out-run (consecutive seen frames out; unseen frames inside the run are allowed
up to BRIDGE_FRAMES) is an exit only if
  - it lasts PERSIST_FRAMES seen frames, or the ball then vanishes for at least VANISH_FRAMES
    (the ball going missing while out), and
  - the ball was seen inside the pitch within ENTRY_FRAMES before the run (a ball that reappears
    outside after a long gap is a dead ball being fetched, not an exit), and
  - a persistent (not vanishing) run reaches at least PERSIST_DEPTH_M beyond the line and is
    not "at a player's feet" (ball within FEET_M of a player on at least FEET_SHARE of the first
    FEET_FRAMES out frames): a player running with the ball on the line, drawn a little outside,
    is still playing (measured on the bench: such runs are shallow and have a player on the
    ball, real exits leave the players behind), and
  - it is not the projection trap (A15): a short run (under TRAP_MAX_FRAMES) after which the
    ball is seen back inside within TRAP_RETURN_FRAMES still travelling above TRAP_SPEED_MPS is
    a high ball drawn outside by the camera, not an exit.

D2 variant "reappear": the ball seen inside, then missing for more than ENTRY_FRAMES but at most
REAPPEAR_MAX_FRAMES, then seen outside in a run that passes the persistent test above (long,
deep, no player at it) went out during the gap. Dated at the crossing interpolated between the
two sightings; dropped if an exit was already found after the last inside sighting.

D3 (STRONG): the ball last seen inside within D3_MAX_DIST_M of a line, moving outwards at
D3_MIN_OUT_MPS or more so that it reaches the line within D3_MAX_TIME_S, then missing for at
least D3_MIN_GAP_FRAMES, has probably gone out. The exit is dated at the extrapolated crossing.

Line, crossing point: from the segment between the last inside and first outside position (or
the extrapolated path for D3), the first line it crosses. ``goal_mouth`` marks a goal-line
crossing between the posts (D4 / R9).

last_touch_team (R8): the team of the last touch event before the exit within LAST_TOUCH_S
(LAST_TOUCH_CONTACT=True would prefer a later frame with a player within CONTACT_M of the ball;
measured on the bench it is much worse, 0.66 against 0.93 agreement, so it is off); with no
touch, the team of the nearest player on the last frame within LAST_TOUCH_S before the
exit when any player was within LAST_NEAR_M of the ball; else -1.
"""
import numpy as np

from pipeline.ball.clean import runs
from pipeline.events.touch import last_touch_before

# Params (start values from the rulebook)
MARGIN_M = 0.3
PERSIST_FRAMES = 8
PERSIST_DEPTH_M = 1.0
FEET_M = 0.75
FEET_SHARE = 0.75
FEET_FRAMES = 8
VANISH_FRAMES = 25
BRIDGE_FRAMES = 5
ENTRY_FRAMES = 12
REAPPEAR_MAX_FRAMES = 75
CONF_REAPPEAR = 0.75
CONTACT_M = 1.0
LAST_TOUCH_CONTACT = False    # bench: worse, see docstring
TRAP_MAX_FRAMES = 25
TRAP_RETURN_FRAMES = 12
TRAP_SPEED_MPS = 8.0
D3_MAX_DIST_M = 5.0
D3_MIN_OUT_MPS = 2.0
D3_MAX_TIME_S = 1.0
D3_MIN_GAP_FRAMES = 50
MERGE_FRAMES = 50          # exits closer than this are one exit (the first)
LAST_TOUCH_S = 10.0
LAST_NEAR_M = 2.0
GOAL_HALF_WIDTH_M = 3.66 + 0.11
CONF_PERSIST, CONF_VANISH, CONF_D3 = 0.95, 0.85, 0.6


def out_side(x, y, pitch, margin=MARGIN_M):
    """Boolean per frame: centre beyond a line by more than margin (NaN -> False)."""
    L, W = pitch
    with np.errstate(invalid="ignore"):
        return (x < -margin) | (x > L + margin) | (y < -margin) | (y > W + margin)


def crossing(p0, p1, pitch):
    """First line crossed going from p0 (inside) to p1. Returns (line, point, t in [0, 1])."""
    L, W = pitch
    d = (p1[0] - p0[0], p1[1] - p0[1])
    best = None
    for line, axis, value in (("goal_x0", 0, 0.0), ("goal_x1", 0, L),
                              ("touch_y0", 1, 0.0), ("touch_y1", 1, W)):
        if d[axis] == 0:
            continue
        t = (value - p0[axis]) / d[axis]
        if 0.0 <= t <= 1.0 + 1e-9 and (best is None or t < best[2]):
            best = (line, (p0[0] + t * d[0], p0[1] + t * d[1]), t)
    if best is None:   # p0 already outside or degenerate: the line p1 is furthest beyond
        over = {"goal_x0": -p1[0], "goal_x1": p1[0] - L, "touch_y0": -p1[1],
                "touch_y1": p1[1] - W}
        line = max(over, key=over.get)
        pt = (min(max(p1[0], 0.0), L), min(max(p1[1], 0.0), W))
        best = (line, pt, 1.0)
    return best


def _goal_mouth(line, pt, pitch):
    return line.startswith("goal") and abs(pt[1] - pitch[1] / 2.0) <= GOAL_HALF_WIDTH_M


def _last_touch(inp, touch_rows, frame):
    back = int(LAST_TOUCH_S * inp.fps)
    t = last_touch_before(touch_rows, frame, back)
    lo = max(0, frame - back)
    if t is not None:
        if LAST_TOUCH_CONTACT:
            d = inp.con.d_any[t["frame"] + 1:frame]
            near = np.flatnonzero(d <= CONTACT_M)
            if len(near):
                f = t["frame"] + 1 + near[-1]
                return int(inp.con.nearest_team[f]), int(f), "contact"
        return t["team"], t["frame"], "touch"
    d = inp.con.d_any[lo:frame]
    near = np.flatnonzero(d <= LAST_NEAR_M)
    if len(near):
        f = lo + near[-1]
        return int(inp.con.nearest_team[f]), int(f), "near"
    return -1, -1, "none"


def _depth(x, y, pitch):
    L, W = pitch
    return np.nanmax(np.r_[-x, x - L, -y, y - W])


def _deep_and_free(inp, frames, pitch):
    """D2 guard for persistent runs: deep enough and not at a player's feet."""
    if _depth(inp.x[frames], inp.y[frames], pitch) < PERSIST_DEPTH_M:
        return False
    d = inp.con.d_any[frames[:FEET_FRAMES]]
    return not (np.mean(d <= FEET_M) >= FEET_SHARE)


def _out_runs(seen_idx, out_at_seen, bridge):
    """Group seen frames that are out into runs; returns list of (first_idx, last_idx) into
    seen_idx, allowing up to ``bridge`` unseen frames between consecutive out frames."""
    groups = []
    start = None
    for k in range(len(seen_idx)):
        if out_at_seen[k]:
            if start is None:
                start = k
            elif seen_idx[k] - seen_idx[k - 1] - 1 > bridge:
                groups.append((start, k - 1))
                start = k
        elif start is not None:
            groups.append((start, k - 1))
            start = None
    if start is not None:
        groups.append((start, len(seen_idx) - 1))
    return groups


def exits(inp, touch_rows):
    pitch = inp.pitch
    seen_idx = np.flatnonzero(inp.seen & np.isfinite(inp.x))
    if len(seen_idx) == 0:
        return []
    out = out_side(inp.x, inp.y, pitch)
    out_at = out[seen_idx]
    rows = []
    for k0, k1 in _out_runs(seen_idx, out_at, BRIDGE_FRAMES):
        a, b = seen_idx[k0], seen_idx[k1]
        n_out = k1 - k0 + 1
        if k0 == 0:
            continue
        f_in = seen_idx[k0 - 1]
        if out_at[k0 - 1] or inp.half[f_in] != inp.half[a] or a - f_in > REAPPEAR_MAX_FRAMES:
            continue
        reappear = a - f_in > ENTRY_FRAMES
        nxt = seen_idx[k1 + 1] if k1 + 1 < len(seen_idx) else None
        half_end = inp.halves[inp.half[a] - 1][1] if inp.half[a] > 0 else inp.n
        gap_after = (min(nxt, half_end) if nxt is not None else half_end) - b - 1
        vanish = gap_after >= VANISH_FRAMES
        persist = n_out >= PERSIST_FRAMES and _deep_and_free(inp, seen_idx[k0:k1 + 1], pitch)
        if not (persist or vanish) or (reappear and not persist):
            continue
        if (nxt is not None and n_out < TRAP_MAX_FRAMES and nxt - b <= TRAP_RETURN_FRAMES
                and not out[nxt] and np.nanmax(inp.speed[a:nxt + 1]) > TRAP_SPEED_MPS
                and inp.speed[nxt] > TRAP_SPEED_MPS):
            continue
        p0 = (inp.x[f_in], inp.y[f_in])
        p1 = (inp.x[a], inp.y[a])
        line, pt, t = crossing(p0, p1, pitch)
        frame = int(f_in + max(1, int(np.ceil(t * (a - f_in)))))
        team, tframe, _how = _last_touch(inp, touch_rows, frame)
        rows.append({"frame": frame, "kind": "exit", "line": line, "x_m": pt[0], "y_m": pt[1],
                     "goal_mouth": _goal_mouth(line, pt, pitch), "last_touch_team": team,
                     "last_touch_frame": tframe, "team": -1,
                     "conf": CONF_REAPPEAR if reappear else (CONF_PERSIST if persist
                                                             else CONF_VANISH),
                     "detail": "reappear" if reappear else ("persist" if persist else "vanish"),
                     "_after": int(f_in) if reappear else frame - 1})
    rows += _extrapolated(inp, touch_rows, out)
    rows.sort(key=lambda r: r["frame"])
    merged = []
    for r in rows:
        after = r.pop("_after", r["frame"] - 1)
        if merged and (r["frame"] - merged[-1]["frame"] < MERGE_FRAMES
                       or merged[-1]["frame"] > after):
            continue
        merged.append(r)
    return merged


def _extrapolated(inp, touch_rows, out):
    """D3: ball vanishes heading out."""
    L, W = inp.pitch
    rows = []
    gap = ~inp.have
    for a, b in zip(*runs(gap)):
        if a == 0 or b - a < D3_MIN_GAP_FRAMES:
            continue
        g = a - 1
        if not inp.seen[g] or out[g] or inp.half[g] == 0:
            continue
        if b >= inp.halves[inp.half[g] - 1][1]:      # missing to the end of the half
            continue
        x, y, vx, vy = inp.x[g], inp.y[g], inp.vx[g], inp.vy[g]
        if not np.all(np.isfinite([x, y, vx, vy])):
            continue
        best = None
        for line, dist, v_out in (("goal_x0", x, -vx), ("goal_x1", L - x, vx),
                                  ("touch_y0", y, -vy), ("touch_y1", W - y, vy)):
            if v_out < D3_MIN_OUT_MPS or dist > D3_MAX_DIST_M or dist < -MARGIN_M:
                continue
            tt = max(dist, 0.0) / v_out
            if tt <= D3_MAX_TIME_S and (best is None or tt < best[1]):
                best = (line, tt)
        if best is None:
            continue
        line, tt = best
        frame = int(g + max(1, int(np.ceil(tt * inp.fps))))
        pt = (min(max(x + vx * tt, 0.0), L), min(max(y + vy * tt, 0.0), W))
        team, tframe, _how = _last_touch(inp, touch_rows, g + 1)
        rows.append({"frame": frame, "kind": "exit", "line": line, "x_m": pt[0], "y_m": pt[1],
                     "goal_mouth": _goal_mouth(line, pt, inp.pitch), "last_touch_team": team,
                     "last_touch_frame": tframe, "team": -1, "conf": CONF_D3,
                     "detail": "extrapolated"})
    return rows
