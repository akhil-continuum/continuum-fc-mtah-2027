"""Event vocabulary for Metrica event data (bench only) and event-derived stoppages.

Everything here works on the standardised event table from bench.metrica_io: columns team (H/A),
type, subtype, period, start_frame, end_frame (0-based bench frames), order.

Stoppages from events (rulebook D1, D4, D5, D7; L1, L2): a stop event opens a stoppage, a restart
event closes it. This uses no tracking, so it can be compared with the ball channel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Events that never touch the ball.
NON_PLAY = {"CARD"}
# Controlled actions: the acting team had the ball under control (P1, P2, P4, P5, P6).
CONTROL_TYPES = {"PASS", "CARRY", "SHOT", "SET PIECE"}
# BALL LOST is a controlled action that failed, except when the 'loss' is a single header or
# clearance (P9, P10), which are one-touch contacts, or a dispossession record (THEFT, FORCED):
# Metrica logs a won tackle as the winner's RECOVERY THEFT and, 1 to 5 frames later, the loser's
# BALL LOST THEFT; that second record is not a new control by the loser.
UNCONTROLLED_LOSS_TOKENS = ("HEAD", "CLEARANCE", "THEFT", "FORCED")

# Events that may sit inside a stoppage without meaning play resumed: the keeper or a ball boy
# collecting the ball (RECOVERY), cards, further foul records.
DEAD_NOISE = {"RECOVERY", "CARD", "FAULT RECEIVED"}
# Further stop events within this many frames of the first one extend the stoppage start (foul
# records follow the challenge; a SHOT OUT is followed by its BALL OUT). Starting value.
STOP_CLUSTER_FRAMES = 50


def is_restart(row) -> bool:
    """Set pieces, plus goal kicks (Metrica games 1 and 2 log them as PASS/BALL LOST 'GOAL KICK',
    not as SET PIECE)."""
    if row["type"] == "SET PIECE":
        return True
    return "GOAL KICK" in str(row["subtype"]) and row["type"] in ("PASS", "BALL LOST", "BALL OUT")


def restart_type(row) -> str:
    if row["type"] == "SET PIECE":
        return str(row["subtype"]).split("-")[0] or "SET PIECE"
    return "GOAL KICK"


def is_loss_record(row) -> bool:
    """A BALL LOST that only records being dispossessed: THEFT/FORCED subtypes, or an
    instantaneous record (start == end; a real failed pass or shot travels for some frames)."""
    if row["type"] != "BALL LOST":
        return False
    sub = str(row["subtype"])
    if "THEFT" in sub or "FORCED" in sub:
        return True
    return int(row["end_frame"]) <= int(row["start_frame"])


def is_control(row) -> bool:
    t, sub = row["type"], str(row["subtype"])
    if t in CONTROL_TYPES:
        return True
    if t == "BALL LOST":
        if is_loss_record(row):
            return False
        return not any(tok in sub for tok in UNCONTROLLED_LOSS_TOKENS)
    return False


def is_touch(row) -> bool:
    """The acting team touched the ball (for first-touch dating), controlled or not."""
    t, sub = row["type"], str(row["subtype"])
    if t in NON_PLAY or t == "FAULT RECEIVED":
        return False
    if t == "CHALLENGE":
        return "WON" in sub and "FAULT" not in sub
    if is_loss_record(row):
        return False
    return True


def is_stop(row) -> bool:
    """An event that stops play: ball out, goal, foul whistled, offside, hand ball, end of half
    (D1, D4, D5, D7). Advantage is logged as ADVANTAGE, without FAULT, so it does not stop play
    (D6)."""
    t, sub = row["type"], str(row["subtype"])
    if t == "BALL OUT":
        return True
    if t == "SHOT" and ("OUT" in sub or ("GOAL" in sub and "GOAL KICK" not in sub)):
        return True
    if t == "CHALLENGE" and "FAULT" in sub:
        return True
    if t == "FAULT RECEIVED":
        return True
    if "OFFSIDE" in sub or "END HALF" in sub or "HAND BALL" in sub:
        return True
    return False


def stop_cause(rows) -> str:
    """Why play stopped, from the stop events of one stoppage (for reporting)."""
    if not len(rows):
        return "other"
    types = set(r["type"] for r in rows)
    subs = " ".join(str(r["subtype"]) for r in rows)
    if "GOAL" in subs.replace("GOAL KICK", "") and ("SHOT" in types or "BALL OUT" in types):
        return "goal"
    if "OFFSIDE" in subs:
        return "offside"
    if "END HALF" in subs:
        return "end_half"
    if "FAULT" in subs or "FAULT RECEIVED" in types or "HAND BALL" in subs:
        return "foul"
    if "BALL OUT" in types or "OUT" in subs:
        return "ball_out"
    return "other"


def event_stoppages(events: pd.DataFrame, halves):
    """One row per stoppage, from the event stream alone.

    A stoppage opens on the frame after a stop event ends (is_stop) and closes at the next restart
    event (SET PIECE, or a GOAL KICK-subtype action). If a live action arrives before any logged
    restart, the restart was not logged: the stoppage closes at that action, or at the RECOVERY
    by the same team immediately before it (restart_type UNLOGGED). A restart with no stop before
    it (kick-off at the start of a half, an unlogged whistle) opens its stoppage on the frame
    after the last action ended.

    Columns: period, start (first dead frame), restart (first live frame), restart_team (H/A),
    restart_type, cause, stop_frame (start of the first stop event, or -1).
    """
    out = []
    ev = events.sort_values(["start_frame", "order"]).reset_index(drop=True)
    cols = ["period", "start", "restart", "restart_team", "restart_type", "cause", "stop_frame"]
    for p_index, (h0, h1) in enumerate(halves):
        period = p_index + 1
        e = ev[(ev["period"] == period) & (~ev["type"].isin(NON_PLAY))]
        state = "pre"            # pre (before the half's kick-off), live, dead
        dead_start = h0
        stop_rows, noise = [], []
        last_live_end = h0 - 1
        last_restart = None
        for row in e.to_dict("records"):
            f0, f1 = int(row["start_frame"]), int(row["end_frame"])
            if is_restart(row):
                if last_restart is not None and f0 == last_restart:
                    continue     # second record of the same restart (RECOVERY + GOAL KICK)
                if state == "pre":
                    start, cause, stop_frame = h0, "half_start", -1
                elif state == "dead":
                    start, cause = dead_start, stop_cause(stop_rows)
                    stop_frame = int(stop_rows[0]["start_frame"])
                else:            # restart with no logged stop
                    start, cause, stop_frame = last_live_end + 1, "unlogged_stop", -1
                out.append({"period": period, "start": max(h0, min(start, f0)), "restart": f0,
                            "restart_team": row["team"], "restart_type": restart_type(row),
                            "cause": cause, "stop_frame": stop_frame})
                last_restart = f0
                state, stop_rows, noise = "live", [], []
                last_live_end = max(f0, f1)
                continue
            if state == "pre":
                state = "live"   # play logged before any kick-off: the half starts live here
            if state == "live":
                if is_stop(row):
                    state = "dead"
                    stop_rows, noise = [row], []
                    dead_start = max(f0, f1) + 1
                else:
                    last_live_end = max(last_live_end, f1)
                continue
            # state == "dead"
            first_stop = int(stop_rows[0]["start_frame"])
            if f0 < dead_start and not is_stop(row):
                # Started before the ball went dead: part of the action that stopped play (game 3
                # logs a 1-frame CARRY at the start frame of almost every action, BALL OUT too).
                continue
            if is_stop(row) and f0 - first_stop <= STOP_CLUSTER_FRAMES and not noise:
                stop_rows.append(row)
                dead_start = max(dead_start, max(f0, f1) + 1)
                continue
            if row["type"] in DEAD_NOISE or (is_stop(row) and row["type"] != "SHOT"):
                noise.append(row)
                continue
            # A live action with no logged restart: the restart was not logged.
            restart_row = row
            for nrow in reversed(noise):
                if nrow["type"] == "RECOVERY" and nrow["team"] == row["team"]:
                    restart_row = nrow
                else:
                    break
            r = int(restart_row["start_frame"])
            out.append({"period": period, "start": min(dead_start, r), "restart": r,
                        "restart_team": restart_row["team"], "restart_type": "UNLOGGED",
                        "cause": stop_cause(stop_rows), "stop_frame": first_stop})
            state, stop_rows, noise = "live", [], []
            last_live_end = max(f0, f1)
            last_restart = r
    return pd.DataFrame(out, columns=cols)


# A stoppage with an unlogged restart must last at least this long (S10 starting value, 2 s).
MIN_UNLOGGED_FRAMES = 50


def confirm_stoppages(stops: pd.DataFrame, ball_missing: np.ndarray, min_share=0.5,
                      min_unlogged_frames=MIN_UNLOGGED_FRAMES):
    """Keep logged-restart stoppages; keep UNLOGGED ones only if the corrected ball channel is
    missing on at least `min_share` of their frames and they last at least
    `min_unlogged_frames` (the challenge truth itself reads dead ball
    off the corrected tracking, and the dead-ball check shows the ball channel is empty on 86 to
    97 percent of event-dead frames). Zero-length stoppages are dropped, except half-start kick-offs.

    Adds column `confirmed_by` (event | ball_missing)."""
    keep, how = [], []
    for s in stops.itertuples():
        length = s.restart - s.start
        if s.cause == "half_start":
            # Kept even when empty (a half that starts on its kick-off frame): the kick-off is a
            # restart that downstream restart detection is measured against.
            keep.append(True)
            how.append("event")
            continue
        if length <= 0:
            keep.append(False)
            how.append("")
            continue
        if s.restart_type != "UNLOGGED":
            keep.append(True)
            how.append("event")
            continue
        share = float(ball_missing[s.start:s.restart].mean())
        keep.append(share >= min_share and length >= min_unlogged_frames)
        how.append("ball_missing")
    out = stops.assign(confirmed_by=how)[np.array(keep, bool)]
    return out.reset_index(drop=True)


def half_end_frames(events: pd.DataFrame, halves):
    """End of the graded span of each half: one past the end of the last ball event, or the first
    frame after an END HALF stop if that comes first."""
    out = []
    for p_index, (h0, h1) in enumerate(halves):
        e = events[(events["period"] == p_index + 1) & (~events["type"].isin(NON_PLAY))]
        out.append(min(h1, int(e["end_frame"].max()) + 1) if len(e) else h1)
    return out


def runs(mask: np.ndarray):
    """[(start, stop)] half-open runs of True."""
    m = np.asarray(mask, bool).astype(np.int8)
    edges = np.diff(np.concatenate(([0], m, [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))
