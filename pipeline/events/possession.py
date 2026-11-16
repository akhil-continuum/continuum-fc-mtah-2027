"""Phase 1 detectors: carry (P1), pass (P2), deflection (P7), control (P4, P5).

Team level only. A "player" here is the nearest player of a team on a frame, identified by
position; no slot is followed. Position continuity (the nearest player's position moving less
than JUMP_M between frames) stands in for "the same player".

carry (P1, STRONG): seen ball within CARRY_M of team X's nearest player, no opponent closer, the
ball's velocity within AGREE_MPS of that player's velocity (when his velocity is known; when it
is not, the ball must be within CARRY_TIGHT_M), with position continuity, for at least
CARRY_MIN_S (gaps of up to CARRY_GAP frames bridged). Row: frame = start, end_frame = end.

pass (P2, STRONG): two consecutive team contacts (touch or carry) by X, the second at least
PASS_DIFF_M from where the first contact's player would be by then running on at his velocity
(a different position cluster: a dribbler who pushes the ball ahead and runs onto it is not a
pass), PASS_MIN_FRAMES to PASS_MAX_S apart,
with no contact by the other team and no exit between, and the ball travelling at least
PASS_MIN_M. Row: frame = the first contact's last frame (the kick), end_frame = reception.

deflection (P7, LAW): a touch by Y between contacts of X (the previous contact within
DEFL_WINDOW_S is X; the next contact within DEFL_WINDOW_S is X, or an exit, or none), after which
the ball keeps at least DEFL_KEEP of its speed, and Y has no carry starting and no other touch
within DEFL_FOLLOW_S. Row: frame = the touch.

control (P4, P5; feeds the decoder's switch cost): contacts (touches that are not deflections,
and carries) are grouped into same-team sequences; a sequence ends at a contact of the other
team, an exit, or a pause longer than SEQ_GAP_S. A sequence that contains a carry or a pass is
a control: frame = first_touch_frame = its first contact, confirmed_frame = the frame the carry
reached CARRY_MIN_S or the pass was received. Sequences never cross an exit or half-time.
"""
import numpy as np

# Params (start values from the skill; chosen on bench games)
CARRY_M = 1.5
CARRY_TIGHT_M = 1.0
AGREE_MPS = 4.0          # bench (SG3 clean + noisy): best control F1
JUMP_M = 1.0
CARRY_MIN_S = 0.4        # rulebook start 0.6; bench: 0.4 better for short possessions
CARRY_GAP = 3
PASS_DIFF_M = 3.0
PASS_MIN_M = 3.0
PASS_MIN_FRAMES = 3
PASS_MAX_S = 5.0
DEFL_WINDOW_S = 2.0
DEFL_KEEP = 0.5
DEFL_FOLLOW_S = 1.0
SEQ_GAP_S = 3.0
SPEED_WIN = (2, 6)        # frames after/before a touch used for the speed comparison


def carry_frames(inp):
    """Per frame: team (0/1) satisfying the carry conditions, else -1; plus carrier position."""
    con = inp.con
    team = con.nearest_team.astype(np.int64)
    n = inp.n
    lab = np.full(n, -1, dtype=np.int8)
    px = np.full(n, np.nan)
    py = np.full(n, np.nan)
    ok_t = team >= 0
    idx = np.flatnonzero(ok_t & inp.seen)
    t = team[idx]
    d = con.d[t, idx]
    pvx, pvy = con.pvx[t, idx], con.pvy[t, idx]
    dv = np.hypot(inp.vx[idx] - pvx, inp.vy[idx] - pvy)
    known = np.isfinite(dv)
    k = inp.radius_scale
    good = np.where(known, (d <= CARRY_M * k) & (dv <= AGREE_MPS), d <= CARRY_TIGHT_M * k)
    good &= np.isfinite(d)
    lab[idx[good]] = t[good]
    px[idx] = con.px[t, idx]
    py[idx] = con.py[t, idx]
    return lab, px, py


def carries(inp):
    lab, px, py = carry_frames(inp)
    min_len = int(round(CARRY_MIN_S * inp.fps))
    rows = []
    n = inp.n
    f = 0
    while f < n:
        if lab[f] < 0:
            f += 1
            continue
        team, start, last = lab[f], f, f
        g = f + 1
        while g < n and g - last <= CARRY_GAP + 1 and inp.half[g] == inp.half[start]:
            if lab[g] == team:
                if np.hypot(px[g] - px[last], py[g] - py[last]) > JUMP_M * (g - last):
                    break
                last = g
            elif lab[g] >= 0:
                break
            g += 1
        if last - start + 1 >= min_len:
            frames = np.arange(start, last + 1)
            share = float(np.mean(lab[frames] == team))
            length_term = min(1.0, (last - start + 1) / (2.0 * min_len))
            rows.append({"frame": int(start), "end_frame": int(last), "kind": "carry",
                         "team": int(team), "conf": round(0.5 * share + 0.5 * length_term, 4),
                         "x_m": px[start], "y_m": py[start]})
        f = last + 1
    return rows


def _contacts(inp, touch_rows, carry_rows):
    """Time-ordered team contacts: dicts with team, start, end, start/end player positions."""
    out = []
    for r in touch_rows:
        f, t = r["frame"], r["team"]
        out.append({"type": "touch", "team": t, "start": f, "end": f,
                    "p0": (inp.con.px[t, f], inp.con.py[t, f]),
                    "p1": (inp.con.px[t, f], inp.con.py[t, f]), "row": r})
    for r in carry_rows:
        a, b, t = r["frame"], r["end_frame"], r["team"]
        out.append({"type": "carry", "team": t, "start": a, "end": b,
                    "p0": (inp.con.px[t, a], inp.con.py[t, a]),
                    "p1": (inp.con.px[t, b], inp.con.py[t, b]), "row": r})
    out.sort(key=lambda c: (c["start"], 0 if c["type"] == "touch" else 1))
    # A touch inside a carry of the same team is part of it.
    merged = []
    for c in out:
        if (c["type"] == "touch" and merged and merged[-1]["type"] == "carry"
                and merged[-1]["team"] == c["team"] and c["start"] <= merged[-1]["end"]):
            continue
        merged.append(c)
    return merged


def _dist(p, q):
    return float(np.hypot(p[0] - q[0], p[1] - q[1]))


def _speed(inp, lo, hi):
    lo, hi = max(0, lo), min(inp.n, hi)
    s = inp.speed[lo:hi][inp.seen[lo:hi]]
    s = s[np.isfinite(s)]
    return float(np.median(s)) if len(s) else np.nan


def _exit_between(exit_frames, a, b):
    i = np.searchsorted(exit_frames, a, side="right")
    return i < len(exit_frames) and exit_frames[i] <= b


RUN_FRAMES = 10            # frames used for a contact player's running velocity


def _run_velocity(inp, c):
    """Mean velocity of the contact's player over its last RUN_FRAMES frames (team nearest)."""
    t, b = c["team"], c["end"]
    lo = max(c["start"] if c["type"] == "carry" else b - RUN_FRAMES, b - RUN_FRAMES, 0)
    vx = inp.con.pvx[t, lo:b + 1]
    vy = inp.con.pvy[t, lo:b + 1]
    if c["type"] == "carry" and b - lo >= 3:
        px, py = inp.con.px[t, [lo, b]], inp.con.py[t, [lo, b]]
        if np.all(np.isfinite([px, py])):
            dt = (b - lo) / inp.fps
            return (px[1] - px[0]) / dt, (py[1] - py[0]) / dt
    ok = np.isfinite(vx) & np.isfinite(vy)
    return (float(np.median(vx[ok])), float(np.median(vy[ok]))) if ok.any() else (0.0, 0.0)


def passes(inp, contacts, exit_frames):
    rows = []
    max_gap = int(PASS_MAX_S * inp.fps)
    for c1, c2 in zip(contacts[:-1], contacts[1:]):
        if c1["team"] != c2["team"]:
            continue
        a, b = c1["end"], c2["start"]
        if not (PASS_MIN_FRAMES <= b - a <= max_gap) or inp.half[a] != inp.half[b]:
            continue
        if _exit_between(exit_frames, a, b):
            continue
        vx, vy = _run_velocity(inp, c1)
        dt = (b - a) / inp.fps
        p_run = (c1["p1"][0] + vx * dt, c1["p1"][1] + vy * dt)
        if _dist(c1["p1"], c2["p0"]) < PASS_DIFF_M or _dist(p_run, c2["p0"]) < PASS_DIFF_M:
            continue
        if not np.isfinite(inp.x[a]) or not np.isfinite(inp.x[b]):
            continue
        if np.hypot(inp.x[b] - inp.x[a], inp.y[b] - inp.y[a]) < PASS_MIN_M:
            continue
        rows.append({"frame": int(a), "end_frame": int(b), "kind": "pass", "team": c1["team"],
                     "conf": round(min(c1["row"]["conf"], c2["row"]["conf"]) * 0.5 + 0.5, 4),
                     "x_m": inp.x[a], "y_m": inp.y[a]})
    return rows


def deflections(inp, contacts, exit_frames):
    """Indices into ``contacts`` of touches that are deflections, and their rows."""
    win = int(DEFL_WINDOW_S * inp.fps)
    follow = int(DEFL_FOLLOW_S * inp.fps)
    rows, flagged = [], set()
    for k, c in enumerate(contacts):
        if c["type"] != "touch":
            continue
        t, y = c["start"], c["team"]
        prev = contacts[k - 1] if k > 0 else None
        if prev is None or prev["team"] == y or t - prev["end"] > win:
            continue
        nxt = contacts[k + 1] if k + 1 < len(contacts) else None
        exit_next = _exit_between(exit_frames, t, t + win)
        if nxt is not None and nxt["start"] - t <= follow and nxt["team"] == y:
            continue                     # Y follows up: carry or second touch
        if nxt is not None and nxt["start"] - t <= win and nxt["team"] == y:
            continue
        before = _speed(inp, t - SPEED_WIN[1], t - SPEED_WIN[0] + 1)
        after = _speed(inp, t + SPEED_WIN[0], t + SPEED_WIN[1] + 1)
        if not (np.isfinite(before) and np.isfinite(after)) or after < DEFL_KEEP * before:
            continue
        to_x = nxt is not None and nxt["team"] != y and nxt["start"] - t <= win
        conf = 0.9 if (to_x or exit_next) else 0.6
        flagged.add(k)
        rows.append({"frame": int(t), "kind": "deflection", "team": int(y), "conf": conf,
                     "x_m": inp.x[t], "y_m": inp.y[t]})
    return flagged, rows


def controls(inp, contacts, deflected, pass_rows, exit_frames):
    gap = int(SEQ_GAP_S * inp.fps)
    min_len = int(round(CARRY_MIN_S * inp.fps))
    pass_by_end = {(r["team"], r["end_frame"]): r for r in pass_rows}
    seqs, cur = [], []
    for k, c in enumerate(contacts):
        if k in deflected:
            continue
        if cur:
            last = cur[-1]
            if (c["team"] != last["team"] or c["start"] - last["end"] > gap
                    or inp.half[c["start"]] != inp.half[last["end"]]
                    or _exit_between(exit_frames, last["end"], c["start"])):
                seqs.append(cur)
                cur = []
        cur.append(c)
    if cur:
        seqs.append(cur)
    rows = []
    for seq in seqs:
        team = seq[0]["team"]
        confirmed, how = None, ""
        for c in seq:
            cand = None
            if c["type"] == "carry":
                cand = c["start"] + min_len - 1
                kind = "carry"
            elif (team, c["start"]) in pass_by_end:
                cand = c["start"]
                kind = "pass"
            if cand is not None and (confirmed is None or cand < confirmed):
                confirmed, how = cand, kind
            if (team, c["start"]) in pass_by_end and (confirmed is None or c["start"] < confirmed):
                confirmed, how = c["start"], "pass"
        if confirmed is None:
            continue
        first = seq[0]["start"]
        conf = float(np.clip(0.5 + 0.5 * seq[0]["row"]["conf"], 0.0, 1.0))
        rows.append({"frame": int(first), "kind": "control", "team": int(team),
                     "first_touch_frame": int(first), "confirmed_frame": int(confirmed),
                     "conf": round(conf, 4), "x_m": inp.x[first], "y_m": inp.y[first],
                     "detail": how})
    return rows


def detect(inp, touch_rows, rows):
    exit_frames = np.array(sorted(r["frame"] for r in rows if r["kind"] == "exit"), dtype=int)
    carry_rows = carries(inp)
    contacts = _contacts(inp, touch_rows, carry_rows)
    pass_rows = passes(inp, contacts, exit_frames)
    deflected, defl_rows = deflections(inp, contacts, exit_frames)
    ctrl_rows = controls(inp, contacts, deflected, pass_rows, exit_frames)
    return carry_rows + pass_rows + defl_rows + ctrl_rows
