"""Stoppages beyond kick-offs and their restarts (rulebook D1-D3, D5, D7, D9, D12, D13, R1-R4,
R8-R12, L1-L4, traps P22-P24, P26, B5).

Inputs (contract files only): players.parquet, meta.json, ball.parquet (cleaned: seen, filled,
x_m, y_m, vx, vy), events.parquet (exit rows with line, last_touch_team; touch rows).
Team level only: per-frame counts, shares and nearest-player-of-a-team; no slot is followed.
All windows are centred or look both ways.

Stoppage candidates come from three sources:

  exit      an `exit` event (D1-D3). Start = exit frame. Type from the line (R9, R10).
  gap       a ball gap (seen and filled both false) of at least GAP_MIN_FRAMES read in context
            (D12, D13): accepted when the players are collectively slow during it (mean share
            moving at most GAP_MOVING_MAX), or when the ball was last seen near a line heading
            out (D3). Traps: a gap that starts with the keeper holding the ball (P22) and lasts
            less than KEEPER_MAX_FRAMES is live; a short gap with collective movement is a ball
            in flight (B5).
  slowdown  a collective slow-down without a gap (D5, D7): the share of players moving drops
            from at least SLOW_BEFORE_MIN to at most SLOW_AFTER_MAX and stays low, and the ball
            stops or vanishes. Low speed without a drop is slow build-up, not a stoppage (P24).
            Frames with the keeper holding are not slow-downs (P22). Lower confidence.

Restart frame (L1, L2, L4): the first frame after the minimum stoppage (S10) where the ball is
seen moving faster than RESTART_BALL_MPS and the players start moving again (share moving over
the next RESUME_FRAMES at least RESUME_SHARE). Ball movement while the players stay still is
the ball being fetched (D9, P26) and does not end the stoppage. With no such frame, the restart
is where the collective movement resumes (formation break, L4). Throw-ins from a gap are dated
L2_LEAD_FRAMES before the ball's first seen frame (L2).

Restart type: from the exit line (touchline: throw-in; goal line: corner if the defending team
touched it last, goal kick if the attacking team did; between the posts: goal, kick-off follows),
otherwise from where the ball is at the restart (touchline, corner, goal area), free kick as the
fallback (R4).

Restart team (R8, R10, R12), as log-odds favouring home:
  last touch before an exit (R8)           +-LLR_LAST_TOUCH
  end of the pitch for corners / goal kicks +-LLR_END (corner: the team attacking that goal;
                                             goal kick: the team defending it)
  team of the nearest player at the restart ball, within NEAR_RESTART_M (R12) +-LLR_NEAR
  none of these: 0 (p_home_restart = 0.5)
"""
import numpy as np
import pandas as pd

from pipeline.restarts.formation import half_of_frames

# Params (start values; set on the bench, see registry.yaml)
JOG_MPS = 2.0                # a player above this is moving
STILL_MPS = 1.0              # a player below this is still
SHARE_WINDOW = 25            # centred mean of the per-frame share moving (1 s)
GAP_MIN_FRAMES = 50          # shortest ball gap read as a stoppage (S10: 2 s)
GAP_MOVING_MAX = 0.30        # mean share moving inside a dead gap (bench: dead median 0.07, live 0.49)
OUT_DIST_M = 5.0             # D3 context: last seen this close to a line ...
OUT_SPEED_MPS = 2.0          # ... moving outwards at least this fast
KEEPER_MAX_FRAMES = 250      # P22: a keeper holds the ball up to about 10 s
SLOW_BEFORE_FRAMES = 50      # slow-down: mean share moving over the 2 s before ...
SLOW_BEFORE_MIN = 0.35
SLOW_AFTER_FROM = 25         # ... and over [t + 1 s, t + 4 s)
SLOW_AFTER_TO = 100
SLOW_AFTER_MAX = 0.15
SLOW_BALL_MPS = 1.0          # the ball stops (or vanishes) after the slow-down
SLOW_LAG_FRAMES = 25         # whistle about 1 s before the detected drop (bench)
MIN_DEAD_FRAMES = 50         # S10: no restart within 2 s of the start
RESTART_BALL_MPS = 2.5
RESUME_FRAMES = 50
RESUME_SHARE = 0.08          # weak floor: everyone standing while the ball moves is P26
RESTART_TRAVEL_M = 6.0       # the restarted ball travels this far within RESUME_FRAMES (bench p10 7.9 m)
GAP_DIP_SHARE = 0.03         # a gap of GAP_DIP_MIN_FRAMES+ whose centred share dips this low is dead (bench: true dips median 0.00, false 0.08)
GAP_DIP_MIN_FRAMES = 100
SLOW_MIN_STOP_FRAMES = 125   # a slow-down-only stoppage lasts at least 5 s
EXIT_MIN_STOP_FRAMES = 75    # an exit followed by play within 3 s ...
EXIT_SEEN_MAX = 0.8          # ... with the ball seen on this share of it was not an exit (D2)
BREAK_SHARE = 0.30           # no ball: formation break when share moving rises to this
MAX_STOP_FRAMES = 6000       # 4 min: longer candidates are dropped (no restart found)
MERGE_GAP_FRAMES = 25        # candidates closer than this are one stoppage
L2_LEAD_FRAMES = 0           # rulebook start 5; bench (rebuilt match noise): 0 is better (-3 vs -8 frames)
TOUCH_ZONE_M = 1.5           # restart position this close to a touchline: throw-in
CORNER_ZONE_M = 3.0
GOAL_AREA_DEPTH_M, GOAL_AREA_HALF_W = 5.5, 9.16
BOX_DEPTH_M, BOX_HALF_W = 16.5, 20.16
NEAR_RESTART_M = 3.0
LLR_LAST_TOUCH = 1.7         # logit(0.85): R8 last-touch agreement 0.915 clean, 0.785 noisy
LLR_END = 2.2                # logit(0.9)
LLR_NEAR = 0.85              # logit(0.7): CLUE
CONF = {"exit": 0.9, "gap": 0.8, "gap_context": 0.6, "slowdown": 0.5}
KEEPER_BALL_M = 1.5
KEEPER_BALL_MPS = 1.5
FREE_KICK_CLEAR_M = 7.0      # free-kick formation: the other team at least this far (9.15 rule)
FREE_KICK_TAKER_M = 2.0

STOP_COLUMNS = ["start_frame", "restart_frame", "start_kind", "restart_type", "p_home_restart",
                "conf", "half", "formation_start", "formation_end", "kick_team_llr", "evidence",
                "team_llr", "type_source", "exit_line", "last_touch_team"]


# ----------------------------------------------------------------------------- per-frame

def _centred_mean(x, w):
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    k = np.ones(int(w) | 1)
    num = np.convolve(np.where(ok, x, 0.0), k, mode="same")
    den = np.convolve(ok.astype(float), k, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _window_mean(x, lo, hi):
    """mean of x over [t+lo, t+hi) for every t (NaN-aware, clipped at the edges)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    ok = np.isfinite(x)
    cs = np.r_[0.0, np.cumsum(np.where(ok, x, 0.0))]
    cn = np.r_[0.0, np.cumsum(ok)]
    t = np.arange(n)
    a = np.clip(t + lo, 0, n)
    b = np.clip(t + hi, 0, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cn[b] - cn[a] > 0, (cs[b] - cs[a]) / (cn[b] - cn[a]), np.nan)


def _runs(flag):
    d = np.diff(np.r_[0, np.asarray(flag, dtype=np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def direction_table(meta, n_frames):
    """(n_frames, 2) attack direction of home and away per frame (0 outside halves)."""
    half = half_of_frames([tuple(h) for h in meta["halves"]], n_frames)
    out = np.zeros((n_frames, 2))
    for hk, per in (meta.get("attack_direction") or {}).items():
        sel = half == int(hk)
        out[sel, 0] = float(per["0"])
        out[sel, 1] = float(per["1"])
    return out, half


def frame_measures(players, meta, ball):
    """Per-frame team-level measures used by every rule in this module."""
    n = int(meta["n_frames"])
    L, W = float(meta["pitch"][0]), float(meta["pitch"][1])
    cy = W / 2.0
    if not meta.get("attack_direction"):
        from pipeline.restarts.formation import derive_attack_direction
        half_tmp = half_of_frames([tuple(h) for h in meta["halves"]], n)
        meta = dict(meta, attack_direction=derive_attack_direction(
            players["frame"].to_numpy(np.int64), players["team"].to_numpy(np.int64),
            players["x_m"].to_numpy(float), half_tmp))
    dirs, half = direction_table(meta, n)

    f = players["frame"].to_numpy(np.int64)
    team = players["team"].to_numpy(np.int64)
    x = players["x_m"].to_numpy(float)
    y = players["y_m"].to_numpy(float)
    sp = np.hypot(players["vx"].to_numpy(float), players["vy"].to_numpy(float))
    ok = np.isfinite(x) & np.isfinite(y) & (f >= 0) & (f < n)
    f, team, x, y, sp = f[ok], team[ok], x[ok], y[ok], sp[ok]
    has_sp = np.isfinite(sp)

    m = pd.DataFrame({"frame": np.arange(n), "half": half})
    cnt = np.bincount(f[has_sp], minlength=n).astype(float)
    mov = np.bincount(f[has_sp], weights=(sp[has_sp] > JOG_MPS).astype(float), minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(cnt > 0, mov / cnt, np.nan)
    m["share_moving"] = share
    m["share_moving_c"] = _centred_mean(share, SHARE_WINDOW)
    m["n_tracked"] = np.bincount(f, minlength=n)

    # ball
    b = ball.set_index("frame").reindex(np.arange(n))
    seen = (b["seen"].fillna(False).astype(bool) | b["filled"].fillna(False).astype(bool)).to_numpy()
    bx = np.where(seen, b["x_m"].to_numpy(float), np.nan)
    by = np.where(seen, b["y_m"].to_numpy(float), np.nan)
    bsp = np.hypot(b["vx"].to_numpy(float), b["vy"].to_numpy(float))
    bsp = np.where(seen, bsp, np.nan)
    m["ball_seen"] = seen
    m["ball_x"], m["ball_y"], m["ball_speed"] = bx, by, bsp
    m["ball_vx"] = np.where(seen, b["vx"].to_numpy(float), np.nan)
    m["ball_vy"] = np.where(seen, b["vy"].to_numpy(float), np.nan)

    # nearest player of each team to the ball
    rbx, rby = bx[f], by[f]
    dball = np.hypot(x - rbx, y - rby)
    for t in (0, 1):
        sel = (team == t) & np.isfinite(dball)
        d = np.full(n, np.inf)
        np.minimum.at(d, f[sel], dball[sel])
        m["near_d_{}".format(t)] = d

    # depth from own goal line; keeper = deepest player of his team on the frame
    rdir = dirs[f, team]
    depth = np.where(rdir > 0, x, L - x)
    depth = np.where(rdir == 0, np.nan, depth)
    dmin = np.full((n, 2), np.inf)
    sel = np.isfinite(depth)
    np.minimum.at(dmin, (f[sel], team[sel]), depth[sel])
    is_keeper = sel & (depth <= dmin[f, team] + 1e-6)
    in_own_box = (depth <= BOX_DEPTH_M) & (np.abs(y - cy) <= BOX_HALF_W)
    hold_row = is_keeper & in_own_box & (dball <= KEEPER_BALL_M)
    hold = np.zeros(n, dtype=bool)
    hold[f[hold_row]] = True
    hold &= ~(bsp > KEEPER_BALL_MPS)
    m["keeper_hold"] = hold

    # formation scores (0/1 per frame, smoothed later)
    still = has_sp & (sp < STILL_MPS)
    dist_touch = np.minimum(np.abs(y), np.abs(y - W))
    on_touch = still & (dist_touch <= TOUCH_ZONE_M) & (x >= 0) & (x <= L)
    throw = np.zeros(n, dtype=bool)
    throw[f[on_touch]] = True
    ball_ok = ~seen | (np.minimum(np.abs(by), np.abs(by - W)) <= 3.0)
    m["form_throw_in"] = throw & ball_ok

    corner = np.zeros(n, dtype=bool)
    for cx_ in (0.0, L):
        for cyy in (0.0, W):
            near_c = still & (np.hypot(x - cx_, y - cyy) <= CORNER_ZONE_M)
            in_box = (np.abs(x - cx_) <= BOX_DEPTH_M) & (np.abs(y - cy) <= BOX_HALF_W)
            nbox = np.bincount(f[in_box], minlength=n)
            fl = np.zeros(n, dtype=bool)
            fl[f[near_c]] = True
            corner |= fl & (nbox >= 6)
    m["form_corner"] = corner

    goal_kick = np.zeros(n, dtype=bool)
    for end_x, defend_dir in ((0.0, 1.0), (L, -1.0)):
        in_box = (np.abs(x - end_x) <= BOX_DEPTH_M) & (np.abs(y - cy) <= BOX_HALF_W)
        att = in_box & (rdir == -defend_dir)
        n_att = np.bincount(f[att], minlength=n)
        dfd_ga = still & (rdir == defend_dir) & (np.abs(x - end_x) <= GOAL_AREA_DEPTH_M) \
            & (np.abs(y - cy) <= GOAL_AREA_HALF_W)
        has_dfd = np.zeros(n, dtype=bool)
        has_dfd[f[dfd_ga]] = True
        ball_ga = seen & (np.abs(bx - end_x) <= GOAL_AREA_DEPTH_M) & (np.abs(by - cy) <= GOAL_AREA_HALF_W)
        goal_kick |= (n_att == 0) & ((ball_ga & ~(bsp > 1.0)) | (~seen & has_dfd))
    m["form_goal_kick"] = goal_kick

    d0, d1 = m["near_d_0"].to_numpy(), m["near_d_1"].to_numpy()
    still_ball = seen & (bsp < 0.5) & (bx > 1) & (bx < L - 1) & (by > 1) & (by < W - 1)
    m["form_free_kick"] = still_ball & (((d0 <= FREE_KICK_TAKER_M) & (d1 >= FREE_KICK_CLEAR_M))
                                        | ((d1 <= FREE_KICK_TAKER_M) & (d0 >= FREE_KICK_CLEAR_M)))
    m.attrs["pitch"] = (L, W)
    m.attrs["dirs"] = dirs
    return m


# ----------------------------------------------------------------------------- candidates

def gap_candidates(m):
    """Ball gaps read in context. Returns list of dicts (start, kind, conf, detail)."""
    seen = m["ball_seen"].to_numpy()
    share = m["share_moving"].to_numpy()
    share_c = m["share_moving_c"].to_numpy()
    hold = m["keeper_hold"].to_numpy()
    L, W = m.attrs["pitch"]
    bx, by = m["ball_x"].to_numpy(), m["ball_y"].to_numpy()
    vx, vy = m["ball_vx"].to_numpy(), m["ball_vy"].to_numpy()
    half = m["half"].to_numpy()
    out = []
    for a, b in _runs(~seen & (half > 0)):
        if b - a < GAP_MIN_FRAMES:
            continue
        last = a - 1
        # P22: the keeper had it just before the gap and it is short -> live
        if last >= 0 and hold[max(0, last - 5):last + 1].any() and b - a < KEEPER_MAX_FRAMES:
            continue
        moving = np.nanmean(share[a:b]) if np.isfinite(share[a:b]).any() else np.nan
        dip = np.nanmin(share_c[a:b]) if np.isfinite(share_c[a:b]).any() else np.nan
        heading_out = False
        if last >= 0 and np.isfinite(bx[last]):
            dl = min(bx[last], L - bx[last], by[last], W - by[last])
            # outward component towards the nearest line
            comps = {bx[last]: -vx[last], L - bx[last]: vx[last], by[last]: -vy[last],
                     W - by[last]: vy[last]}
            out_v = comps[dl] if np.isfinite(comps[dl]) else 0.0
            heading_out = dl <= OUT_DIST_M and out_v >= OUT_SPEED_MPS
        if np.isfinite(moving) and moving <= GAP_MOVING_MAX:
            out.append({"start": a, "kind": "gap", "conf": CONF["gap"], "detail": "slow"})
        elif b - a >= GAP_DIP_MIN_FRAMES and np.isfinite(dip) and dip <= GAP_DIP_SHARE:
            out.append({"start": a, "kind": "gap", "conf": CONF["gap"], "detail": "dip"})
        elif heading_out:
            out.append({"start": a, "kind": "gap", "conf": CONF["gap_context"],
                        "detail": "heading_out"})
    return out


def slowdown_candidates(m):
    share = m["share_moving_c"].to_numpy()
    before = _window_mean(share, -SLOW_BEFORE_FRAMES, 0)
    after = _window_mean(share, SLOW_AFTER_FROM, SLOW_AFTER_TO)
    bsp = m["ball_speed"].to_numpy()
    seen = m["ball_seen"].to_numpy()
    ball_slow = np.where(seen, (bsp < SLOW_BALL_MPS).astype(float), 1.0)
    ball_after = _window_mean(ball_slow, SLOW_AFTER_FROM, SLOW_AFTER_TO)
    hold = m["keeper_hold"].to_numpy().astype(float)
    hold_after = _window_mean(hold, 0, SLOW_AFTER_TO)
    flag = (before >= SLOW_BEFORE_MIN) & (after <= SLOW_AFTER_MAX) & (ball_after >= 0.7) \
        & ~(hold_after >= 0.3) & (m["half"].to_numpy() > 0)
    out = []
    for a, b in _runs(flag):
        # the drop point: the frame in the run where before - after is largest
        k = a + int(np.nanargmax((before - after)[a:b]))
        out.append({"start": max(0, k - SLOW_LAG_FRAMES), "kind": "slowdown",
                    "conf": CONF["slowdown"], "detail": "drop"})
    return out


def exit_candidates(events):
    out = []
    if events is None or not len(events):
        return out
    ex = events[events["kind"] == "exit"]
    for _, r in ex.iterrows():
        out.append({"start": int(r["frame"]), "kind": "exit", "conf": float(r.get("conf", CONF["exit"])),
                    "detail": str(r.get("detail", "")), "line": str(r["line"]),
                    "last_touch_team": int(r["last_touch_team"]),
                    "goal_mouth": bool(r.get("goal_mouth", False)),
                    "x": float(r["x_m"]), "y": float(r["y_m"])})
    return out


# ----------------------------------------------------------------------------- restart

def find_restart(m, start, stop_limit):
    """Restart frame after a stoppage start (L1, L2, L4). Returns (frame, source) or (None, '')."""
    seen = m["ball_seen"].to_numpy()
    bsp = m["ball_speed"].to_numpy()
    share = m["share_moving"].to_numpy()
    lo = start + MIN_DEAD_FRAMES
    hi = min(stop_limit, start + MAX_STOP_FRAMES)
    if lo >= hi:
        return None, ""
    resume = _window_mean(share[lo:min(len(share), hi + RESUME_FRAMES)], 0, RESUME_FRAMES)[:hi - lo]
    moving_ball = seen[lo:hi] & (bsp[lo:hi] >= RESTART_BALL_MPS)
    travel = _travel(m, lo, hi)
    cand = moving_ball & (resume >= RESUME_SHARE) & (travel >= RESTART_TRAVEL_M)
    idx = np.flatnonzero(cand)
    if len(idx):
        r = lo + int(idx[0])
        # walk back to the first seen frame of this seen run (the kick / the release)
        k = r
        while k - 1 >= lo and seen[k - 1] and not (bsp[k - 1] < 0.5):
            k -= 1
        return k, "ball"
    sm = _centred_mean(share[lo:hi], SHARE_WINDOW)
    up = np.flatnonzero(sm >= BREAK_SHARE)
    if len(up):
        return lo + int(up[0]), "break"
    return None, ""


def _travel(m, lo, hi):
    """For t in [lo, hi): largest distance of the seen ball within RESUME_FRAMES from its
    position at t (NaN-aware; 0 where the ball is unseen at t)."""
    bx = m["ball_x"].to_numpy()
    by = m["ball_y"].to_numpy()
    n = len(bx)
    out = np.zeros(hi - lo)
    for k in range(0, RESUME_FRAMES, 2):
        j = np.clip(np.arange(lo, hi) + k, 0, n - 1)
        d = np.hypot(bx[j] - bx[lo:hi], by[j] - by[lo:hi])
        out = np.fmax(out, np.where(np.isfinite(d), d, 0.0))
    return out


# ----------------------------------------------------------------------------- type & team

def _restart_place(m, r):
    """Ball position at the restart (first seen within a few frames), or (nan, nan)."""
    bx, by = m["ball_x"].to_numpy(), m["ball_y"].to_numpy()
    for k in range(r, min(len(bx), r + 10)):
        if np.isfinite(bx[k]):
            return bx[k], by[k]
    return np.nan, np.nan


def _attacking_team_of_end(dirs_row, end_is_low_x):
    """Team attacking towards the goal at low x (direction -1) or high x (+1)."""
    want = -1.0 if end_is_low_x else 1.0
    for t in (0, 1):
        if dirs_row[t] == want:
            return t
    return -1


def type_and_team(m, cand, r):
    """(restart_type, type_source, team_llr) for one candidate with restart frame r."""
    L, W = m.attrs["pitch"]
    dirs = m.attrs["dirs"]
    llr = 0.0
    rtype, source = None, ""
    px, py = _restart_place(m, r)
    row = dirs[min(r, len(dirs) - 1)]
    if cand.get("kind") == "exit":
        line = cand["line"]
        ltt = cand["last_touch_team"]
        if cand.get("goal_mouth"):
            rtype, source = "kick_off", "exit_goal_mouth"
        elif line.startswith("touch"):
            rtype, source = "throw_in", "exit_line"
            if ltt in (0, 1):
                llr += LLR_LAST_TOUCH * (1 if ltt == 1 else -1)
        elif line.startswith("goal"):
            low = line.endswith("x0")
            att = _attacking_team_of_end(row, low)
            if ltt in (0, 1) and att in (0, 1):
                if ltt == att:
                    rtype, source = "goal_kick", "exit_line_last_touch"
                    winner = 1 - att
                else:
                    rtype, source = "corner", "exit_line_last_touch"
                    winner = att
                llr += LLR_END * (1 if winner == 0 else -1)
            else:
                # no last touch: type from where the ball is at the restart
                rtype, source, llr_end = _type_from_place(px, py, L, W, row)
                if rtype not in ("corner", "goal_kick"):
                    rtype, source = "goal_kick_or_corner", "exit_line"
                llr += llr_end
    if rtype is None:
        rtype, source, llr_end = _type_from_place(px, py, L, W, row)
        llr += llr_end
    # R12: team of the nearest player at the restart ball
    if np.isfinite(px) and rtype not in ("kick_off",):
        d0 = m["near_d_0"].to_numpy()[r]
        d1 = m["near_d_1"].to_numpy()[r]
        if min(d0, d1) <= NEAR_RESTART_M and d0 != d1:
            llr += LLR_NEAR * (1 if d0 < d1 else -1)
    return rtype, source, float(llr)


def _type_from_place(px, py, L, W, row):
    if not np.isfinite(px):
        return "free_kick", "fallback_no_ball", 0.0
    cy = W / 2.0
    for cx_ in (0.0, L):
        for cyy in (0.0, W):
            if np.hypot(px - cx_, py - cyy) <= CORNER_ZONE_M:
                att = _attacking_team_of_end(row, cx_ == 0.0)
                return "corner", "place", (LLR_END * (1 if att == 0 else -1) if att >= 0 else 0.0)
    if min(abs(py), abs(py - W)) <= TOUCH_ZONE_M or py < 0 or py > W:
        return "throw_in", "place", 0.0
    for end_x in (0.0, L):
        if abs(px - end_x) <= GOAL_AREA_DEPTH_M + 1.0 and abs(py - cy) <= GOAL_AREA_HALF_W + 1.0:
            att = _attacking_team_of_end(row, end_x == 0.0)
            return "goal_kick", "place", (LLR_END * (1 if att == 1 else -1) if att >= 0 else 0.0)
    return "free_kick", "place", 0.0


# ----------------------------------------------------------------------------- main

def detect_stoppages(m, events, halves, kickoffs=None, funnel=None):
    """Stoppage rows (STOP_COLUMNS). Kick-off rows are merged by the caller (see merge_kickoffs)."""
    cands = exit_candidates(events) + gap_candidates(m) + slowdown_candidates(m)
    count = {"exit": 0, "gap": 0, "slowdown": 0}
    for c in cands:
        count[c["kind"]] += 1
    half = m["half"].to_numpy()
    cands.sort(key=lambda c: c["start"])
    # merge candidates closer than MERGE_GAP_FRAMES or overlapping, priority exit > gap > slow
    prio = {"exit": 0, "gap": 1, "slowdown": 2}
    rows = []
    i = 0
    while i < len(cands):
        c = cands[i]
        h = int(half[min(c["start"], len(half) - 1)])
        if h == 0:
            i += 1
            continue
        stop_limit = int(halves[h - 1][1])
        r, rsrc = find_restart(m, c["start"], stop_limit)
        group = [c]
        j = i + 1
        while j < len(cands) and r is not None and cands[j]["start"] <= r + MERGE_GAP_FRAMES \
                and int(half[min(cands[j]["start"], len(half) - 1)]) == h:
            group.append(cands[j])
            if cands[j]["start"] > r - MERGE_GAP_FRAMES:
                # a candidate starting at/after the restart: the restart was premature only if
                # this one is an exit or gap; otherwise keep the restart
                if cands[j]["kind"] in ("exit", "gap") and cands[j]["start"] >= r:
                    group.pop()
                    break
            j += 1
        i = j
        if r is None:
            continue
        best = min(group, key=lambda g: prio[g["kind"]])
        if best["kind"] == "slowdown" and r - best["start"] < SLOW_MIN_STOP_FRAMES:
            continue
        if best["kind"] == "exit" and len(group) == 1 and r - best["start"] < EXIT_MIN_STOP_FRAMES \
                and m["ball_seen"].to_numpy()[best["start"]:r].mean() >= EXIT_SEEN_MAX:
            continue
        start = min(g["start"] for g in group if g["kind"] != "slowdown") \
            if any(g["kind"] != "slowdown" for g in group) else best["start"]
        rtype, tsrc, llr = type_and_team(m, best, r)
        if rtype == "throw_in" and rsrc == "ball":
            r = max(start + 1, r - L2_LEAD_FRAMES)
        conf = max(g["conf"] for g in group)
        rows.append({"start_frame": int(start), "restart_frame": int(r),
                     "start_kind": best["kind"], "restart_type": rtype,
                     "p_home_restart": float(1.0 / (1.0 + np.exp(-llr))) if llr != 0 else 0.5,
                     "conf": float(conf), "half": h, "formation_start": -1, "formation_end": -1,
                     "kick_team_llr": 0.0, "evidence": "+".join(sorted({g["kind"] for g in group})),
                     "team_llr": float(llr), "type_source": tsrc + "/" + rsrc,
                     "exit_line": best.get("line", ""),
                     "last_touch_team": int(best.get("last_touch_team", -1))})
    if funnel is not None:
        funnel.update(count)
    return pd.DataFrame(rows, columns=STOP_COLUMNS)


def merge_kickoffs(stops, kickoffs, halves):
    """Join stoppages with kick-off rows. A stoppage whose span contains a goal kick-off's
    restart (or ends within MERGE_GAP_FRAMES of it) becomes that kick-off's stoppage: the
    kick-off row takes its start frame and start_kind 'goal' and the stoppage row is dropped.
    Stoppages that end at or before a half's kick-off are dropped (pre-kick-off is masked)."""
    ko = kickoffs.copy()
    for col in STOP_COLUMNS:
        if col not in ko.columns:
            ko[col] = "" if col in ("type_source", "exit_line") else (
                -1 if col == "last_touch_team" else 0.0)
    ko["team_llr"] = ko["kick_team_llr"]
    ko["type_source"] = "kick_off_formation"
    keep = np.ones(len(stops), dtype=bool)
    st = stops.reset_index(drop=True)
    for idx, row in ko.iterrows():
        kr = int(row["restart_frame"])
        if row["start_kind"] == "half_start":
            keep &= ~((st["half"] == row["half"]) & (st["restart_frame"] <= kr + MERGE_GAP_FRAMES))
            continue
        hit = keep & (st["start_frame"] <= kr) & (st["restart_frame"] >= kr - MERGE_GAP_FRAMES) \
            & (st["half"] == row["half"])
        if hit.any():
            j = int(np.flatnonzero(hit.to_numpy())[0])
            ko.loc[idx, "start_frame"] = int(st.loc[j, "start_frame"])
            ko.loc[idx, "evidence"] = str(row["evidence"]) + "+" + str(st.loc[j, "evidence"])
            keep[j] = False
    out = pd.concat([ko, st[keep]], ignore_index=True)
    return out.sort_values("restart_frame", kind="stable").reset_index(drop=True)


def dead_features(m, rows, n_frames):
    """features_dead columns from this module (added to the kick-off ones)."""
    out = {}
    p_exit = np.zeros(n_frames, np.float32)
    p_slow = np.zeros(n_frames, np.float32)
    p_gap = np.zeros(n_frames, np.float32)
    p_restart = np.zeros(n_frames, np.float32)
    p_dead = np.zeros(n_frames, np.float32)
    for _, r in rows.iterrows():
        s, e = int(r["start_frame"]), int(r["restart_frame"])
        c = float(r["conf"])
        if 0 <= e < n_frames:
            p_restart[e] = max(p_restart[e], c)
        if s < 0:
            continue
        ev = str(r["evidence"])
        if "exit" in ev:
            p_exit[s] = max(p_exit[s], c)
        if "gap" in ev:
            p_gap[s] = max(p_gap[s], c)
        if "slowdown" in ev:
            p_slow[s] = max(p_slow[s], c)
        p_dead[s:e] = np.maximum(p_dead[s:e], c)
    out["p_exit"] = p_exit
    out["p_slowdown"] = p_slow
    out["p_gap_stop"] = p_gap
    out["p_restart"] = p_restart
    out["p_dead_candidate"] = p_dead
    out["keeper_hold"] = m["keeper_hold"].to_numpy()
    for k in ("throw_in", "corner", "goal_kick", "free_kick"):
        out["form_" + k] = _centred_mean(m["form_" + k].to_numpy().astype(float),
                                         SHARE_WINDOW).astype(np.float32)
    return out
