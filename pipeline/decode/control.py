"""Control evidence for the switch cost (v3; rulebook P1 to P5, P7, P13, P15; Part 2 table).

Read from the events table only (CONTRACTS.md, Events): touch, deflection, carry, pass, control,
exit. Team level; nothing follows a player.

Every team contact (a touch that is not a deflection, or a carry) belongs to a same-team
contact sequence, built the same way as the events stage builds control (pipeline/events/
possession.py): a sequence ends at a contact of the other team, an exit, half-time or a pause
longer than SEQ_GAP_S. Each sequence is one *anchor* for a switch into its team, at its first
contact (first_touch_frame), with a grade:

    control  the sequence holds a carry or a completed pass (a ``control`` event; P1, P2, P4, P5)
             strength = 1 (the event's conf is not used: on the bench the share of control
             anchors followed by a true switch does not rise with conf, T-11)
    two      two or more contacts of the team in a row, no carry, no pass (weaker P2)
             strength = g_two
    single   one touch only (a touch is not control, P13 / A1)            strength = g_one
    deflection                                                            excluded (P7)

Grading by count (n_full set, the v3a choice): strength = clip((n_contacts - 1) / (n_full - 1),
0, 1). On the bench (match level, anchors inside the other team's true possession) the share
followed by a true switch within 1 s rises with the sequence's contacts: 1 -> 0.12, 2 -> 0.23,
3 -> 0.42, 4-5 -> 0.62, 6+ -> 0.77, while the control flag alone gives 0.35 (T-11).

The anchor's confirmed frame is the control event's confirmed_frame (control) or the second
contact (two), for the alternative dating convention (P16, OPEN).

The decoder turns the anchors into a per-frame, per-direction switch cost (decoders.switch_cost):

    cost_into_X(t) = c_none(t) - b_ctrl * max_anchor strength * kernel(t - anchor) , >= C_FLOOR
    c_none(t)      = c_lo + (c_base - c_lo) * seen_share(t)

seen_share is the share of frames with the ball seen within +-SEEN_HALF_WIN frames (centred).
Absence of contact evidence only counts against a switch where the ball was visible, so in long
ball gaps (and with no events table at all) the cost falls back to the v2s constant c_lo.
"""
import os

import numpy as np
import pandas as pd

SEQ_GAP_S = 3.0          # same as the events stage (pipeline/events/possession.py SEQ_GAP_S)
SEEN_HALF_WIN = 25       # +-1 s window for seen_share (centred)
C_FLOOR = 1.0            # a switch never costs less than this
FPS = 25.0

GRADE_CONTROL, GRADE_TWO, GRADE_SINGLE = 0, 1, 2


def _contacts(events):
    """Time-ordered (start, end, team) contacts: non-deflection touches and carries; a touch
    inside a same-team carry is part of it."""
    ev = events
    touch = ev[(ev["kind"] == "touch") & (ev["team"] >= 0)]
    defl = ev[(ev["kind"] == "deflection") & (ev["team"] >= 0)]
    dset = set(zip(defl["frame"].astype(int).tolist(), defl["team"].astype(int).tolist()))
    rows = []
    for f, t in zip(touch["frame"].astype(int).tolist(), touch["team"].astype(int).tolist()):
        if (f, t) not in dset:
            rows.append((f, f, t, 0))
    car = ev[(ev["kind"] == "carry") & (ev["team"] >= 0)]
    for f, e, t in zip(car["frame"].astype(int).tolist(), car["end_frame"].astype(int).tolist(),
                       car["team"].astype(int).tolist()):
        rows.append((f, max(f, e), t, 1))
    rows.sort(key=lambda r: (r[0], r[3]))
    merged = []
    for r in rows:
        if merged and r[3] == 0 and merged[-1][3] == 1 and merged[-1][2] == r[2] \
                and r[0] <= merged[-1][1]:
            continue
        merged.append(r)
    return merged


def anchors(events, halves, fps=FPS):
    """DataFrame of switch anchors: frame (first contact), team, grade, conf, confirmed_frame.

    events: the events table (or None). halves: [(start, stop)].
    """
    cols = ["frame", "team", "grade", "conf", "confirmed_frame", "n_contacts", "span", "n_carry"]
    if events is None or not len(events):
        return pd.DataFrame({c: pd.Series(dtype=float) for c in cols})
    exits = np.sort(events.loc[events["kind"] == "exit", "frame"].to_numpy(dtype=np.int64))
    half_of = lambda f: next((i for i, (a, b) in enumerate(halves) if a <= f < b), -1)  # noqa
    gap = int(round(SEQ_GAP_S * fps))
    seqs, cur = [], []
    for c in _contacts(events):
        if cur:
            last = cur[-1]
            i = np.searchsorted(exits, last[1], side="right")
            exit_between = i < len(exits) and exits[i] <= c[0]
            if (c[2] != last[2] or c[0] - last[1] > gap or exit_between
                    or half_of(c[0]) != half_of(last[1])):
                seqs.append(cur)
                cur = []
        cur.append(c)
    if cur:
        seqs.append(cur)
    ctrl = events[(events["kind"] == "control") & (events["team"] >= 0)]
    ctrl_at = {(int(f), int(t)): (float(cf), int(cfr)) for f, t, cf, cfr in zip(
        ctrl["first_touch_frame"], ctrl["team"], ctrl["conf"], ctrl["confirmed_frame"])}
    out = []
    for s in seqs:
        f0, team = s[0][0], s[0][2]
        hit = ctrl_at.get((f0, team))
        extra = (len(s), s[-1][1] - f0, sum(c[3] for c in s))
        if hit is not None:
            out.append((f0, team, GRADE_CONTROL, hit[0], hit[1]) + extra)
        elif len(s) >= 2:
            out.append((f0, team, GRADE_TWO, 1.0, s[1][0]) + extra)
        else:
            out.append((f0, team, GRADE_SINGLE, 1.0, f0) + extra)
    # control events whose first touch is not a sequence start here (should not happen; kept
    # so that no control event is lost if the two sequence builders ever disagree)
    starts = {(o[0], o[1]) for o in out}
    for (f, t), (cf, cfr) in ctrl_at.items():
        if (f, t) not in starts:
            out.append((f, t, GRADE_CONTROL, cf, cfr, 1, max(0, cfr - f), 0))
    df = pd.DataFrame(out, columns=cols)
    return df.sort_values("frame", kind="mergesort").reset_index(drop=True)


def seen_share(seen, half_win=SEEN_HALF_WIN):
    """Centred share of seen frames within +-half_win frames."""
    s = np.asarray(seen, dtype=np.float64)
    n = len(s)
    c = np.concatenate([[0.0], np.cumsum(s)])
    lo = np.clip(np.arange(n) - half_win, 0, n)
    hi = np.clip(np.arange(n) + half_win + 1, 0, n)
    return (c[hi] - c[lo]) / np.maximum(1, hi - lo)


def read_seen(root, n):
    """Ball seen flags from ball.parquet, or None when the root has no ball stage."""
    path = os.path.join(root, "ball.parquet")
    if not os.path.isfile(path):
        return None
    b = pd.read_parquet(path, columns=["frame", "seen"])
    seen = np.zeros(n, dtype=bool)
    fr = b["frame"].to_numpy(dtype=np.int64)
    ok = (fr >= 0) & (fr < n)
    seen[fr[ok]] = b["seen"].to_numpy(dtype=bool)[ok]
    return seen


def strength(n, anc, g_two, width, anchor="first", lead=None, g_one=0.0, shift=0, n_full=None):
    """(n, 2) max kernel-weighted evidence strength for a switch into H (col 0) and A (col 1).

    anchor: "first" (first_touch_frame) or "confirmed" (confirmed_frame); shift moves the
    kernel's peak that many frames after the anchor. g_one: strength of a single touch.
    Kernel: triangle of half-width ``width`` frames centred on the anchor; with ``lead`` set,
    the kernel reaches only ``lead`` frames before the anchor (asymmetric, v3b).
    n_full: when set, the strength is graded by the number of consecutive contacts of the
    sequence instead of its grade: clip((n_contacts - 1) / (n_full - 1), 0, 1), so a single
    touch is 0 and n_full or more contacts in a row are full control (g_two, g_one unused).
    """
    out = np.zeros((n, 2))
    if anc is None or not len(anc):
        return out
    g = anc["grade"].to_numpy(dtype=int)
    if n_full is not None and "n_contacts" in anc:
        nc = anc["n_contacts"].to_numpy(dtype=float)
        w = np.clip((nc - 1.0) / max(1.0, float(n_full) - 1.0), 0.0, 1.0)
    else:
        w = np.where(g == GRADE_CONTROL, 1.0,
                     np.where(g == GRADE_TWO, float(g_two), float(g_one)))
    keep = w > 0
    col = "confirmed_frame" if anchor == "confirmed" else "frame"
    fr = anc[col].to_numpy(dtype=np.int64)[keep] + int(shift)
    tm = anc["team"].to_numpy(dtype=np.int64)[keep]
    w = w[keep]
    width = max(1, int(width))
    before = width if lead is None else max(0, int(lead))
    offs = np.arange(-before, width)
    kern = np.where(offs < 0, 1.0 - np.abs(offs) / float(max(before, 1)) if before else 0.0,
                    1.0 - offs / float(width))
    kern = np.clip(kern, 0.0, 1.0)
    for team in (0, 1):
        sel = tm == team
        if not sel.any():
            continue
        idx = fr[sel][:, None] + offs[None, :]
        val = w[sel][:, None] * kern[None, :]
        ok = (idx >= 0) & (idx < n)
        np.maximum.at(out[:, team], idx[ok], val[ok])
    return out
