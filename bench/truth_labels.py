"""Bench truth under the organisers' label conventions (task T-26; nine general statements,
C1 to C9, read from the published labelling; summarised in docs/Continuum_FC.pdf, "Possession
rules").

    uv run python -m bench.truth_labels [--games metrica_g1 ...] [--no-copy]

Writes, per game, artifacts/bench/<game_id>/truth_labels.csv and truth_labels_meta.json, and copies
truth_labels.csv into every existing level root artifacts/bench/<game_id>_<level>/ (the old truth
files are never touched). The dead-ball spans, graded spans and restart law are exactly those of
bench.truth (same stoppages, same R0 code path); only the owner while live changes:

- C1: every deliberate play of the ball (pass, carry, shot, header won, clearance, interception,
  collection, a tackle followed by the tackler's team's collection) is a CLAIM: the acting team
  owns the ball from it, however briefly. A challenge won that only dislodges the ball is an
  ANCHOR: it dates the switch (C3, C4) only if the dislodging team makes the next claim.
- C3: a switch whose claim is an interception or theft (on either side's record), or that follows
  an anchor, is dated at the contact.
- C2, C4, C6: a switch whose claim collects a loose ball (including a keeper gathering, C6) is
  dated by `anticipation_frame`, a time-to-intercept rule on the clean tracking, bounded between
  the old owner's last touch and the claim.
- C5: a switch whose claim is a header is dated no earlier than `aerial_floor`, the moment the
  ball's flight descends to a height a jumping player can reach.
- C7, C8: stoppages and restarts are bench.events' (event-based), unchanged.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil

import numpy as np
import pandas as pd

from bench import metrica_io
from bench.events import (NON_PLAY, STOP_CLUSTER_FRAMES, confirm_stoppages, event_stoppages,
                          half_end_frames, is_stop)
from bench.truth import ART, graded_spans, label_stats, write_labels

FPS = metrica_io.FPS

# -- physical constants (none taken from the public window) ------------------------------------
# Time-to-intercept model of Spearman (2018, "Beyond Expected Goals", pitch control): a player
# keeps his current velocity for a reaction time, then runs straight at a maximum speed; the
# arrival-time uncertainty is logistic with scale sqrt(3) * sigma / pi, sigma = 0.45 s.
T_REACT_S = 0.7
V_MAX_MPS = 5.0
TTI_SIGMA_S = 0.45
# "Clearly going to reach it first": the receiver's probability of arriving first is at least
# 0.9 under that logistic, i.e. a lead of ln(9) * sqrt(3) * sigma / pi = 0.545 s.
P_CLEAR = 0.9
MARGIN_S = math.log(P_CLEAR / (1 - P_CLEAR)) * math.sqrt(3) * TTI_SIGMA_S / math.pi
# A player plays a ball within about a stride of his body centre.
R_CONTROL_M = 1.0
# After the receiver's touch the ball is held at the touch point for this long, so that an
# opponent arriving just after the receiver still counts as a contest (it bounds the search).
HOLD_AFTER_TOUCH_S = 2.0
# Imminence: the outcome is settled only once the receiver's touch is less than one reaction time
# away; before that an opponent can still react and change it, and the ball is still the
# kicker's play (C1, C5: a pass in flight stays with the kicker's team). The switch may precede
# the receiver's touch by at most this (hindsight time to the touch). Physical choice = the
# model's reaction time; sensitivity (1.5 s) reported in the note.
T_HORIZON_S = T_REACT_S
# Metrica logs the dispossessed player's BALL LOST (sometimes PASS) 1 to 5 frames after the
# winner's contact (bench measurement): such a record
# by the old owner, before any collection of its own, is not a new play.
LOSS_RECORD_FRAMES = 5
RECORD_TYPES = ("BALL LOST", "PASS", "CARRY", "SHOT")
# C7: a foul whose stoppage starts within one reaction time is an immediate whistle (play stops,
# nobody gains the ball); a later stoppage means the referee waited (advantage), and the fouled
# team has the ball from the foul until the whistle.
IMMEDIATE_WHISTLE_S = T_REACT_S
# Centred velocity: positions 2 frames either side (no lag).
VEL_HALF_FRAMES = 2
# Aerial balls (C5): kicked from the ground, headed at head height; above the reach of a jumping
# player nobody can play it.
G_MPS2 = 9.81
H_KICK_M = 0.0
H_HEADER_M = 2.0
H_REACH_M = 3.0

# -- event map (C1) -----------------------------------------------------------------------------
DELIBERATE_OUT = ("CLEARANCE", "HEAD", "CROSS", "DEEP BALL", "THROUGH BALL", "GOAL KICK")
TACKLE_DUELS = ("TACKLE", "GROUND", "DRIBBLE")


def _sub(row):
    return str(row["subtype"]) if row["subtype"] is not None and row["subtype"] == row["subtype"] else ""


def _other(team):
    return "A" if team == "H" else "H"


def classify(row, same_frame_rows):
    """(role, kind, team) of one live event.

    role: 'claim' (deliberate play: `team` owns the ball from here), 'anchor' (a won duel that
    only dislodges the ball: dates a switch if this team claims next), 'touch' (the team touched
    the ball but this record is not a claim of its own; counts as the owner's touch), or None.
    kind (claims only): 'contact' (dated at the event, C3), 'aerial' (C5), 'collect' (C2, C6),
    'foul' (C7: the fouled team has the ball from the foul until the whistle).
    `team` is the row's team except for fouls, where it is the fouled team.
    """
    t, sub, team = row["type"], _sub(row), row["team"]
    f0, f1 = int(row["start_frame"]), int(row["end_frame"])
    same_team = [r for r in same_frame_rows if r["team"] == team and r is not row]
    if t in NON_PLAY:
        return None, None, team
    if t == "FAULT RECEIVED":
        return "claim", "foul", team               # C7
    if t == "CHALLENGE":
        if "FAULT" in sub or "ADVANTAGE" in sub:
            return "claim", "foul", (team if "WON" in sub else _other(team))   # C7
        if "WON" not in sub:
            return None, None, team                # lost or undecided duel
        if sub.startswith("AERIAL"):
            return "claim", "aerial", team         # C1: a header won is possession
        return "anchor", "contact", team           # dislodge unless the team collects (C1, C3)
    if t == "BALL LOST":
        if "HAND BALL" in sub:
            return "claim", "foul", _other(team)   # C7: hand ball, the other team is fouled
        if "THEFT" in sub or "FORCED" in sub or f1 <= f0:
            return None, None, team                # dispossession record, not a play
        if "END HALF" in sub:
            return None, None, team
        return "claim", _kind_from_sub(sub), team
    if t == "BALL OUT":
        if any(tok in sub for tok in DELIBERATE_OUT):
            return "claim", _kind_from_sub(sub), team   # a clearance or header out is a play (C1)
        return "touch", None, team                 # last touch before out: may be a deflection
    if t == "RECOVERY":
        if "GOAL" in sub and "GOAL KICK" not in sub:
            return None, None, team                # picking the ball out of the net
        if any(is_stop(r) for r in same_team):
            return "touch", None, team             # one touch straight out / hand ball: no play
        if "INTERCEPTION" in sub or "THEFT" in sub or "BLOCKED" in sub:
            return "claim", "contact", team
        return "claim", "collect", team            # loose-ball collection, keeper gather (C6)
    if t == "CARRY":
        if f1 - f0 <= 1 and any(is_stop(r) for r in same_team):
            return "touch", None, team             # game 3's 1-frame carry on a touch out
        return "claim", "collect", team
    if t in ("PASS", "SHOT", "SET PIECE"):
        return "claim", _kind_from_sub(sub), team
    return None, None, team


def _kind_from_sub(sub):
    if "INTERCEPTION" in sub:
        return "contact"
    if "HEAD" in sub:
        return "aerial"
    return "collect"


# -- physics -----------------------------------------------------------------------------------

def centred_velocity(xy, half=VEL_HALF_FRAMES):
    """(n, ..., 2) positions -> velocities (m/s) by a centred difference; NaN where missing."""
    v = np.full_like(xy, np.nan)
    n = len(xy)
    if n > 2 * half:
        v[half:n - half] = (xy[2 * half:] - xy[:n - 2 * half]) * (FPS / (2.0 * half))
    return v


def _interp_ball(ball, a, b):
    seg = ball[a:b + 1].copy()
    ok = np.isfinite(seg[:, 0])
    if ok.sum() < max(2, 0.5 * len(seg)):
        return None
    idx = np.arange(len(seg))
    for k in range(2):
        seg[:, k] = np.interp(idx, idx[ok], seg[ok, k])
    return seg


def intercept_times(pos, vel, path, t_index):
    """Earliest time (s from frame t) each player can reach the ball path.

    pos, vel: (P, 2) at frame t; path: (S, 2) ball positions for frames t .. t+S-1. A player
    arrives at a point at T_REACT + max(0, |point - (pos + vel*T_REACT)| - R_CONTROL) / V_MAX; he
    intercepts at the first future frame whose ball position he can reach by then."""
    p = pos + vel * T_REACT_S
    d = np.linalg.norm(path[None, :, :] - p[:, None, :], axis=2)            # (P, S)
    arrive = T_REACT_S + np.maximum(0.0, d - R_CONTROL_M) / V_MAX_MPS
    s = np.arange(path.shape[0]) / FPS
    ok = arrive <= s[None, :]
    first = np.where(ok.any(axis=1), ok.argmax(axis=1), -1)
    out = np.where(first >= 0, s[np.maximum(first, 0)], np.inf)
    out[~np.isfinite(p).all(axis=1)] = np.nan
    return out


def anticipation_frame(game, vel, team_code, lo, claim):
    """C2/C4/C6: earliest frame in [lo, claim] from which, continuously until the claim, the
    receiver (the claiming team's player nearest the ball at the claim) is estimated to reach the
    ball path at least MARGIN_S before every opponent. Returns (frame, info)."""
    if claim <= lo:
        return claim, {"why": "empty_window"}
    hold = int(HOLD_AFTER_TOUCH_S * FPS)
    path_all = _interp_ball(game.ball, lo, claim)
    if path_all is None:
        return claim, {"why": "ball_missing"}
    path_all = np.vstack([path_all, np.repeat(path_all[-1:], hold, axis=0)])
    team = game.player_team
    pc = game.players[claim]
    dist = np.linalg.norm(pc - path_all[claim - lo], axis=1)
    dist[(team != team_code) | ~np.isfinite(dist)] = np.inf
    if not np.isfinite(dist).any():
        return claim, {"why": "no_receiver"}
    recv = int(np.argmin(dist))
    opp = team != team_code
    good = []
    for t in range(lo, claim + 1):
        pos, v = game.players[t], np.nan_to_num(vel[t])
        tt = intercept_times(pos, v, path_all[t - lo:], t)
        tr = tt[recv]
        to = tt[opp]
        to = to[np.isfinite(to) | np.isinf(to)]
        to_min = np.min(to) if len(to) else np.inf
        soon = (claim - t) / FPS <= T_HORIZON_S
        good.append(soon and np.isfinite(tr) and tr + MARGIN_S <= to_min)
    good = np.array(good, bool)
    if not good[-1]:
        return claim, {"why": "contested_at_touch", "receiver": recv}
    k = len(good) - 1
    while k > 0 and good[k - 1]:
        k -= 1
    return lo + k, {"why": "clear", "receiver": recv}


def aerial_floor(kick, contact):
    """C5: first frame at which a ball kicked from the ground at `kick` and headed at head height
    at `contact` (a parabola, no drag) is back below the reach of a jumping player. Equal to
    `kick` when the flight never rises above that reach."""
    T = (contact - kick) / FPS
    if T <= 0:
        return contact
    vz = (H_HEADER_M - H_KICK_M + 0.5 * G_MPS2 * T * T) / T
    disc = vz * vz - 2 * G_MPS2 * (H_REACH_M - H_KICK_M)
    if disc <= 0:
        return kick
    t_down = (vz + math.sqrt(disc)) / G_MPS2
    return min(contact, kick + int(math.ceil(t_down * FPS)))


# -- owner walk --------------------------------------------------------------------------------

def owner_switches_labels(game, events, stops, period, vel):
    """[(frame, team, info)] owner changes in one half under the label conventions."""
    st = stops[stops["period"] == period]
    dead = [(int(s.start), int(s.restart)) for s in st.itertuples()]
    restart_at = {int(s.restart): s.restart_team for s in st.itertuples()}

    def in_dead(f):
        return any(a <= f < b for a, b in dead)

    e = events[(events["period"] == period) & (~events["type"].isin(NON_PLAY))]
    e = e.sort_values(["start_frame", "order"]).to_dict("records")
    by_frame = {}
    for row in e:
        by_frame.setdefault(int(row["start_frame"]), []).append(row)

    switches = []
    owner = floor = last_touch = None
    last_owner_row = None
    anchor = None                     # (frame, team) first won duel by the non-owner
    last_switch = None                # (claim frame, new owner, old owner) of the last claim switch
    foul_lock = None                  # (fouled team, whistle frame) while a delayed whistle is pending
    pending = sorted(restart_at)
    for row in e:
        f0 = int(row["start_frame"])
        while pending and pending[0] <= f0:
            r = pending.pop(0)
            owner, floor, last_touch, anchor, last_owner_row = restart_at[r], r, r, None, None
            last_switch = foul_lock = None
            switches.append((r, owner, {"source": "restart"}))
        if in_dead(f0):
            continue
        role, kind, team = classify(row, by_frame[f0])
        if role is None:
            continue
        if role == "claim" and kind == "foul":
            nxt = [a for a, _ in dead if a > f0]
            if nxt and min(nxt) - f0 < IMMEDIATE_WHISTLE_S * FPS:
                continue                  # whistled at once: play stops, the owner is unchanged
            if (nxt and is_stop(row) and min(nxt) - f0 <= STOP_CLUSTER_FRAMES + 1
                    and (foul_lock is None or foul_lock[1] <= f0)):
                foul_lock = (team, min(nxt))  # a whistled foul: lock until its stoppage only
        if (role == "claim" and foul_lock is not None and f0 < foul_lock[1]
                and team != foul_lock[0]):
            continue                      # C7: until the whistle the fouled team keeps the ball
        if owner is None:
            if role == "claim":
                owner, floor, last_touch, last_owner_row = team, f0, f0, row
                switches.append((f0, team, {"source": "first_claim"}))
            continue
        if team == owner:
            last_touch = max(last_touch, f0)
            last_owner_row = row
            if role == "claim":
                anchor = None             # the owner played it again: a dislodge did not switch
            continue
        if role == "anchor":
            if anchor is None or anchor[1] != team:
                anchor = (f0, team)
            continue
        if role != "claim":
            continue
        if (row["type"] in RECORD_TYPES and last_switch is not None and last_switch[1] == owner
                and team == last_switch[2] and 0 <= f0 - last_switch[0] <= LOSS_RECORD_FRAMES):
            continue                      # the dispossessed player's record of the same duel
        lo = max(floor, last_touch + 1)
        old_sub = _sub(last_owner_row) if last_owner_row is not None else ""
        info = {"source": "claim", "type": row["type"], "subtype": _sub(row), "claim": f0,
                "kind": kind, "lo": lo}
        if kind == "foul":
            date, info["rule"] = f0, "C7_foul"
        elif anchor is not None and anchor[1] == team:
            date, info["rule"] = anchor[0], "C3_anchor"
        elif kind != "aerial" and (kind == "contact" or "INTERCEPTION" in old_sub):
            date, info["rule"] = f0, "C3_contact"
        else:
            tcode = 0 if team == "H" else 1
            date, why = anticipation_frame(game, vel, tcode, lo, f0)
            info["rule"], info["anticipation"] = "C2_anticipation", why.get("why")
            if kind == "aerial":
                fl = aerial_floor(last_touch, f0)
                info["aerial_floor"] = fl
                if fl > date:
                    date, info["rule"] = fl, "C5_aerial_floor"
        date = int(min(max(date, lo), f0))
        last_switch = (f0, team, owner)
        owner, floor, last_touch, last_owner_row, anchor = team, date, f0, row, None
        switches.append((date, team, info))
    while pending:
        r = pending.pop(0)
        switches.append((r, restart_at[r], {"source": "restart"}))
    switches.sort(key=lambda s: s[0])
    return switches


def derive_labels_conv(game, stops=None):
    """(labels, stops, spans, switches) under the label conventions; D and spans as bench.truth."""
    halves = game.halves()
    nan = np.isnan(game.ball[:, 0])
    if stops is None:
        stops = confirm_stoppages(event_stoppages(game.events, halves), nan)
    ends = half_end_frames(game.events, halves)
    spans = graded_spans(game, stops, ends)
    vel = centred_velocity(game.players)
    labels = np.full(game.n_frames, "X", dtype="<U1")
    all_sw = []
    for k, (a, b) in enumerate(spans):
        period = k + 1
        sw = owner_switches_labels(game, game.events, stops, period, vel)
        all_sw += [(f, t, dict(i, period=period)) for f, t, i in sw]
        seg = np.full(b - a, "D", dtype="<U1")
        for idx, (f, team, _) in enumerate(sw):
            stop = sw[idx + 1][0] if idx + 1 < len(sw) else b
            lo, hi = max(f, a), min(stop, b)
            if hi > lo:
                seg[lo - a:hi - a] = team
        for s in stops[stops["period"] == period].itertuples():
            lo, hi = max(s.start, a), min(s.restart, b)
            if hi > lo:
                seg[lo - a:hi - a] = "D"
        labels[a:b] = seg
    return labels, stops, spans, all_sw


# -- comparison with the first-touch truth -----------------------------------------------------

def possessions(labels):
    """[(start, stop, team)] maximal H/A runs (a non-graded or D frame ends a run)."""
    out, n, i = [], len(labels), 0
    while i < n:
        if labels[i] in ("H", "A"):
            j = i
            while j < n and labels[j] == labels[i]:
                j += 1
            out.append((i, j, labels[i]))
            i = j
        else:
            i += 1
    return out


def compare_to(old, new, max_shift=125):
    """Start shifts of possessions (new - old, frames) and possessions new to `new`.

    Each old possession is paired with the new possession of the same team whose start is
    nearest, among those overlapping it or starting within max_shift frames of it."""
    po, pn = possessions(old), possessions(new)
    starts_n = np.array([s for s, _, _ in pn])
    shifts, used = [], set()
    for s, e, t in po:
        best = None
        lo = np.searchsorted(starts_n, s - max_shift)
        hi = np.searchsorted(starts_n, max(e, s + max_shift))
        for j in range(lo, hi):
            a, b, tt = pn[j]
            if tt != t or j in used:
                continue
            if not (a < e and s < b) and abs(a - s) > max_shift:
                continue
            if best is None or abs(a - s) < abs(pn[best][0] - s):
                best = j
        if best is not None:
            used.add(best)
            shifts.append(pn[best][0] - s)
    new_only = [pn[j] for j in range(len(pn)) if j not in used]
    sh = np.array(shifts)
    moved = sh[sh != 0]
    q = lambda arr, p: float(np.percentile(arr, p)) if len(arr) else None
    return {
        "n_old": len(po), "n_new": len(pn), "n_paired": len(shifts),
        "n_moved": int(len(moved)), "n_earlier": int((moved < 0).sum()),
        "n_later": int((moved > 0).sum()),
        "abs_shift_median": q(np.abs(moved), 50), "abs_shift_p90": q(np.abs(moved), 90),
        "signed_shift_median": q(moved, 50), "signed_shift_p10": q(moved, 10),
        "n_new_only": len(new_only), "n_old_unpaired": len(po) - len(shifts),
        "new_only_len_s": {"median": q(np.array([(b - a) / FPS for a, b, _ in new_only]), 50)},
    }


def share_under(labels, sec):
    lens = np.array([(b - a) / FPS for a, b, _ in possessions(labels)])
    return round(float((lens < sec).mean()), 4) if len(lens) else None


def foul_check(labels, stops):
    """C7: share of foul stoppages whose last live frame is owned by the restart (fouled) team."""
    st = stops[stops["cause"] == "foul"]
    hit = tot = 0
    for s in st.itertuples():
        f = int(s.start) - 1
        if f < 0 or labels[f] not in ("H", "A"):
            continue
        tot += 1
        hit += int(labels[f] == s.restart_team)
    return {"n": tot, "owner_is_fouled_team": hit}


def build(game_id, copy=True):
    game = metrica_io.load_game(game_id)
    out_dir = os.path.join(ART, game_id)
    labels, stops, spans, sw = derive_labels_conv(game)
    old_stops = pd.read_csv(os.path.join(out_dir, "stoppages.csv"))
    same = (len(old_stops) == len(stops)
            and (old_stops[["start", "restart"]].to_numpy() == stops[["start", "restart"]].to_numpy()).all())
    if not same:
        raise RuntimeError("{}: stoppages differ from stoppages.csv".format(game_id))
    write_labels(os.path.join(out_dir, "truth_labels.csv"), labels)
    ft = pd.read_csv(os.path.join(out_dir, "truth_first_touch.csv"))["label"].astype(str).values
    rules = pd.Series([i.get("rule", i["source"]) for _, _, i in sw]).value_counts().to_dict()
    antic = pd.Series([i.get("anticipation") for _, _, i in sw if "anticipation" in i]).value_counts().to_dict()
    claim_types = pd.Series(["{} {}".format(i["type"], i["subtype"]).strip()
                             for _, _, i in sw if i["source"] == "claim"]).value_counts().to_dict()
    lead = np.array([i["claim"] - f for f, _, i in sw if i["source"] == "claim"])
    meta = {"game_id": game_id, "convention": "labels (C1-C9, T-26)",
            "params": {"T_REACT_S": T_REACT_S, "V_MAX_MPS": V_MAX_MPS, "TTI_SIGMA_S": TTI_SIGMA_S,
                       "P_CLEAR": P_CLEAR, "MARGIN_S": MARGIN_S, "R_CONTROL_M": R_CONTROL_M,
                       "HOLD_AFTER_TOUCH_S": HOLD_AFTER_TOUCH_S, "H_HEADER_M": H_HEADER_M,
                       "H_REACH_M": H_REACH_M},
            "stats": dict(label_stats(labels), share_poss_under_1s=share_under(labels, 1.0)),
            "stats_first_touch": dict(label_stats(ft), share_poss_under_1s=share_under(ft, 1.0)),
            "vs_first_touch": compare_to(ft, labels),
            "switch_rules": rules, "anticipation_outcomes": antic, "claim_types": claim_types,
            "switch_lead_frames": {"n": int(len(lead)), "n_pos": int((lead > 0).sum()),
                                   "median_pos": float(np.median(lead[lead > 0])) if (lead > 0).any() else None,
                                   "p90_pos": float(np.percentile(lead[lead > 0], 90)) if (lead > 0).any() else None},
            "foul_check": {"labels": foul_check(labels, stops), "first_touch": foul_check(ft, stops)},
            "restart_owner_ok": int(sum(1 for s in stops.itertuples()
                                        if s.restart < len(labels) and labels[s.restart] == s.restart_team)),
            "n_restarts_in_span": int(sum(1 for s in stops.itertuples()
                                          if s.restart < len(labels) and labels[s.restart] in ("H", "A")))}
    with open(os.path.join(out_dir, "truth_labels_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=float)
    copied = []
    if copy:
        for root in sorted(glob.glob(os.path.join(ART, game_id + "_*"))):
            if os.path.exists(os.path.join(root, "truth.csv")):
                shutil.copyfile(os.path.join(out_dir, "truth_labels.csv"),
                                os.path.join(root, "truth_labels.csv"))
                copied.append(os.path.basename(root))
    meta["copied_to"] = copied
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--games", nargs="*", default=None)
    ap.add_argument("--no-copy", action="store_true")
    args = ap.parse_args(argv)
    for gid in args.games or metrica_io.available_games():
        m = build(gid, copy=not args.no_copy)
        s, c = m["stats"], m["vs_first_touch"]
        print("{}: poss/min {:.2f} D {:.3f} stops {} <1s {:.3f} <1.5s {:.3f} <4s {:.3f} | moved {} "
              "(|d| med {} p90 {}) new-only {} | rules {}".format(
                  gid, s["possessions_per_min"], s["share_D"], s["n_stoppages"],
                  s["share_poss_under_1s"], s["share_poss_under_1_5s"], s["share_poss_under_4s"],
                  c["n_moved"], c["abs_shift_median"], c["abs_shift_p90"], c["n_new_only"],
                  m["switch_rules"]))


if __name__ == "__main__":
    main()
