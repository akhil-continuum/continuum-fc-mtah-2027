"""Kick-off detection from tracking alone and the kick-off law (rulebook R6, R13, S4, S5, S6, L6).

A kick-off formation (R6) on a frame:
  - every tracked player is in his own half (at most KO_MAX_WRONG_HALF over the line, by more
    than HALF_TOL_M, to absorb tracking noise);
  - one team (the defending one) has nobody inside the centre circle;
  - that team lines the circle: at least DEF_RING_MIN of its players within RING_M of the spot;
  - the other team (the kicking one) has a player within SPOT_RADIUS_M of the centre spot;
  - the ball is unseen or within BALL_SPOT_M of the centre spot (B2: a still ball on the spot is
    a kick-off only when the formation holds, which is exactly this test).

The per-frame test is smoothed with a centred majority window, then frames where it holds are
joined into segments. A segment is a kick-off when it lasts long enough (players stationary in
the formation) and play breaks out after it (the formation stops holding). The restart instant
(L1/L6) is dated from the ball leaving the spot when the ball is seen, otherwise from the
formation break minus a lag measured on the bench.

Kicking team: the team with the player at the spot, voted frame by frame over the segment. The
kick-off law (S6) ties the two half-start kick-offs together: their evidence is combined into
one decision (first half team X, second half the other team), so p_home_restart of the two
half-start rows always sum to one.

Nothing here refers to a frame, time or place specific to any game; halves, pitch and attack
directions come from meta.json.
"""
import numpy as np
import pandas as pd

from pipeline.restarts.formation import CENTRE_CIRCLE_M, CIRCLE_MARGIN_M, kickoff_frame_measures

# Params (starting values; lags measured on the bench, see registry.yaml)
SPOT_RADIUS_M = 1.5          # kicking-team player at the ball (bench: true kick-offs <= 0.93 m)
DEF_RING_MIN = 2             # defending players within formation.RING_M of the spot (line the circle)
BALL_SPOT_M = 3.0            # a seen ball this close to the centre spot is "on the spot"
KO_MAX_WRONG_HALF = 1        # players allowed over the halfway line (noise, duplicates)
MIN_TRACKED = 12             # fewer tracked players: no formation decision on the frame
SMOOTH_FRAMES = 25           # centred majority window for the per-frame test
SMOOTH_SHARE = 0.6           # share of the window that must hold
MERGE_GAP_FRAMES = 50        # join holding stretches separated by less than this
MIN_HOLD_FRAMES = 25         # a kick-off formation holds at least 1 s ...
MIN_HOLD_AT_HALF_START = 10  # ... unless the segment starts at the half start (half begins at the kick)
HALF_START_SLACK = 25        # "starts at the half start": within this many frames
BREAK_CONFIRM_FRAMES = 125   # after the segment, the formation must stay broken ...
BREAK_CONFIRM_SHARE = 0.8    # ... on at least this share of these frames
BALL_LEAVE_M = 2.0           # ball has left the spot
BALL_SEARCH_BEFORE = 75      # frames before the segment end searched for the ball leaving
BALL_SEARCH_AFTER = 25       # frames after the segment end searched
INSTANT_FROM = "break"       # "break" or "ball": which evidence dates the kick (bench: break)
BALL_LAG_FRAMES = 7         # kick = first frame ball beyond BALL_LEAVE_M minus this (bench)
BREAK_LAG_FRAMES = 17        # kick = formation break minus this when the ball is unseen (bench:
                             # median break - truth over 56 kick-offs, 12 bench roots, no ball)
KICKER_MIN_SHARE = 0.1       # share of segment frames with a taker at the spot (no ball seen)
CONF_BALL = 0.95             # conf of a kick-off dated by the ball leaving the spot (bench precision)
CONF_BREAK = 0.6             # conf of a kick-off seen by the formation only (bench precision)
ANCHOR_LIVE_OFFSET = 1       # FAQ: the dead span before a kick-off includes the anchor frame
ANCHOR_AT_START_FRAMES = 2   # anchor this close to the half start: the half starts at its kick-off
CONF_ANCHOR = 0.99          # half-start kick-off at the published anchor (match only)
CONF_ASSUMED = 0.5           # conf of a half-start kick-off assumed at the half start
LIVE_MIXED_MIN = 4           # players in the opponents' half that mean "teams are mixed"
LIVE_MIXED_FRAMES = 750      # 30 s of mixed teams before a half's first kick-off = live play
LIVE_SPLIT_FRAMES = 125      # 5 s of mixed teams between two formations = play resumed between
MIN_KICKOFF_SPACING = 1500# 60 s: goal to kick-off takes 45 to 90 s (D4)
VOTE_LLR_MAX = 4.6          # clip of the kicking-team log-odds (p = 0.99)
VOTE_EFFECTIVE_FRAMES = 25   # consecutive frames are not independent: one vote per this many

RESTART_COLUMNS = ["start_frame", "restart_frame", "start_kind", "restart_type", "p_home_restart",
                   "conf", "half", "formation_start", "formation_end", "kick_team_llr",
                   "evidence"]


def formation_frames(measures, use_ball=True):
    """Per-frame kick-off formation test. Returns (holds bool array, kicking team array with
    0 = home, 1 = away, -1 = not determined on the frame) before smoothing.

    Two forms hold: (a) a player of one team at the spot, the other team outside the circle and
    lining it; (b) the taker not tracked (tracker gaps are common at dead balls): nobody inside
    the circle and both teams lining it. Form (b) holds the formation without voting a team.
    With use_ball=False the ball test is skipped (players only; used to date the break)."""
    m = measures
    in_half = m["half"].to_numpy() > 0
    enough = m["n_tracked"].to_numpy() >= MIN_TRACKED
    halves_ok = m["n_wrong_half"].to_numpy() <= KO_MAX_WRONG_HALF
    base = in_half & enough & halves_ok
    if use_ball:
        ball_d = m["ball_d_spot"].to_numpy()
        base &= ~np.isfinite(ball_d) | (ball_d <= BALL_SPOT_M)
    kick = np.full(len(m), -1, dtype=np.int8)
    for t in (0, 1):
        o = 1 - t
        at_spot = m["d_spot_{}".format(t)].to_numpy() <= SPOT_RADIUS_M
        others_out = m["n_circle_{}".format(o)].to_numpy() == 0
        others_line = m["n_ring_{}".format(o)].to_numpy() >= DEF_RING_MIN
        kick[base & at_spot & others_out & others_line & (kick == -1)] = t
    nobody_in = (m["n_circle_0"].to_numpy() == 0) & (m["n_circle_1"].to_numpy() == 0)
    both_line = ((m["n_ring_0"].to_numpy() >= DEF_RING_MIN)
                 & (m["n_ring_1"].to_numpy() >= DEF_RING_MIN))
    holds = (kick >= 0) | (base & nobody_in & both_line)
    return holds, kick


def _centred_share(flag, window):
    """Centred moving share of True values (window odd-ised), edges normalised."""
    w = int(window) | 1
    c = np.convolve(flag.astype(float), np.ones(w), mode="same")
    n = np.convolve(np.ones(len(flag)), np.ones(w), mode="same")
    return c / n


def _per_half_share(flag, halves, window):
    """_centred_share inside each half separately (no window crosses half-time)."""
    out = np.zeros(len(flag))
    for h0, h1 in halves:
        h0, h1 = int(h0), int(h1)
        if h1 > h0:
            out[h0:h1] = _centred_share(flag[h0:h1], window)
    return out


def _halves_from_measures(measures):
    half = measures["half"].to_numpy()
    return [(int(np.argmax(half == h)), int(len(half) - np.argmax(half[::-1] == h)))
            for h in sorted(set(half.tolist()) - {0})]


def _segments(flag):
    """[(start, stop)) runs of True."""
    if len(flag) == 0:
        return []
    d = np.diff(np.r_[0, flag.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def formation_score(measures):
    """Smoothed per-frame kick-off formation score in [0, 1] (the share of a centred window
    where the per-frame test holds). This is the `form_kick_off` dead-ball feature."""
    holds, _ = formation_frames(measures)
    return _per_half_share(holds, _halves_from_measures(measures), SMOOTH_FRAMES)


def _kick_instant(measures, seg_start, seg_end, half_start):
    """Restart frame for a kick-off segment [seg_start, seg_end). Returns (frame, source):
    evidence "formation_ball" when the ball is seen on the spot and then leaving it (that confirms the
    kick), else "formation". The frame comes from INSTANT_FROM: the formation break minus
    BREAK_LAG_FRAMES (default; more accurate on the bench), or the ball leaving the spot minus
    BALL_LAG_FRAMES when the ball confirms it."""
    ball_d = measures["ball_d_spot"].to_numpy()
    lo = max(seg_start, seg_end - BALL_SEARCH_BEFORE)
    hi = min(len(ball_d), seg_end + BALL_SEARCH_AFTER)
    window = ball_d[lo:hi]
    seen = np.isfinite(window)
    ball_frame = None
    # the ball must first be seen on the spot, then beyond BALL_LEAVE_M on two seen frames in a row
    on_spot = seen & (window <= BALL_LEAVE_M)
    if on_spot.any():
        beyond = seen & (window > BALL_LEAVE_M)
        beyond[:int(np.argmax(on_spot))] = False
        idx = np.flatnonzero(beyond[:-1] & beyond[1:])
        if len(idx):
            ball_frame = lo + int(idx[0]) - BALL_LAG_FRAMES
    source = "formation" if ball_frame is None else "formation_ball"
    if INSTANT_FROM == "ball" and ball_frame is not None:
        frame = ball_frame
    else:
        frame = seg_end - BREAK_LAG_FRAMES
    return int(max(frame, half_start, seg_start)), source


def _kick_team_llr(kick, seg_start, seg_end):
    """Log-odds that the home team (0) kicks off, from per-frame votes over the segment."""
    votes = kick[seg_start:seg_end]
    n_home = float(np.sum(votes == 0)) / VOTE_EFFECTIVE_FRAMES
    n_away = float(np.sum(votes == 1)) / VOTE_EFFECTIVE_FRAMES
    llr = np.log((n_home + 0.5) / (n_away + 0.5))
    return float(np.clip(llr, -VOTE_LLR_MAX, VOTE_LLR_MAX))


def detect_kickoffs(measures, halves, funnel=None, anchors=None):
    """Kick-off candidates. Returns a DataFrame with RESTART_COLUMNS (p_home_restart is the
    formation evidence only; apply_kickoff_law adds S6). If ``funnel`` is a dict it receives
    per-half counts of candidate segments surviving each filter (aggregates for diagnostics).
    ``anchors`` ({half: frame}, match only, from games.py through the caller) replaces the
    half-start kick-off when no formation is detected at the half start; bench games pass None."""
    holds, kick = formation_frames(measures)
    smooth = _per_half_share(holds, halves, SMOOTH_FRAMES) >= SMOOTH_SHARE
    holds_players, _ = formation_frames(measures, use_ball=False)
    smooth_players = _per_half_share(holds_players, halves, SMOOTH_FRAMES) >= SMOOTH_SHARE
    rows = []
    for half, (h0, h1) in enumerate(halves, 1):
        h0, h1 = int(h0), int(h1)
        flag = smooth[h0:h1].copy()
        segs = [(a + h0, b + h0) for a, b in _segments(flag)]
        merged = []
        for a, b in segs:
            if merged and a - merged[-1][1] < MERGE_GAP_FRAMES:
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        kept = []   # (a, b, vote_from): segments that pass, after the spacing rule
        count = {"segments": len(merged), "hold": 0, "break": 0, "evidence": 0}
        for a, b in merged:
            # the players hold the formation until their break; the ball leaving the spot can
            # end the ball-inclusive segment earlier (that is the kick, not a broken formation)
            pb = _player_break(smooth_players, b, h1)
            at_start = a - h0 <= HALF_START_SLACK
            if pb - a < (MIN_HOLD_AT_HALF_START if at_start else MIN_HOLD_FRAMES):
                continue
            count["hold"] += 1
            after = holds[b:min(h1, b + BREAK_CONFIRM_FRAMES)]
            if b >= h1 or len(after) == 0 or np.mean(~after) < BREAK_CONFIRM_SHARE:
                continue
            count["break"] += 1
            frame, source = _kick_instant(measures, a, pb, h0)
            taker_share = float(np.mean(kick[a:b] >= 0))
            # evidence that this is a kick-off and not another restart in midfield: a taker
            # at the spot on part of the segment, or the ball seen leaving the spot
            if source != "formation_ball" and taker_share < KICKER_MIN_SHARE:
                continue
            count["evidence"] += 1
            # D4: a kick-off cannot follow another within MIN_KICKOFF_SPACING. If the teams
            # stayed in their own halves between the two segments, both belong to one stoppage
            # and the later one is the kick-off (the formation holds until the kick; the earlier
            # segment's team votes are kept, same taker waiting at the spot). If play resumed in
            # between, the earlier one was the kick-off and the later one is something else.
            item = (a, b, a, frame, source, taker_share)
            if kept and a - kept[-1][1] < MIN_KICKOFF_SPACING:
                if _mixed_frames(measures, kept[-1][1], a) < LIVE_SPLIT_FRAMES:
                    kept[-1] = (a, b, kept[-1][2], frame, source, taker_share)
            else:
                kept.append(item)
        half_rows = []
        for a, b, vote_from, frame, source, taker_share in kept:
            llr = _kick_team_llr(kick, vote_from, b)
            half_rows.append({"start_frame": -1, "restart_frame": frame,
                              "start_kind": "goal", "restart_type": "kick_off",
                              "p_home_restart": float(1.0 / (1.0 + np.exp(-llr))),
                              "conf": CONF_BALL if source == "formation_ball" else CONF_BREAK,
                              "half": half, "formation_start": int(a), "formation_end": int(b),
                              "kick_team_llr": llr, "evidence": source})
        # S4/S5: the first kick-off of the half is the half-start one, unless live play came
        # before it (the half began at a kick-off the tracking did not see); then, and when
        # no kick-off was found at all, the half-start kick-off is assumed at the half start
        # with no team evidence of its own (S6 still gives it one through the other half).
        count["kept"] = len(half_rows)
        if funnel is not None:
            funnel[half] = count
        assumed = None
        if not half_rows:
            assumed = "assumed_none_found"
        elif _live_before(measures, h0, half_rows[0]["formation_start"]):
            assumed = "assumed_live_before"
        anchor = (anchors or {}).get(half)
        if assumed and anchor is not None and h0 <= int(anchor) < h1:
            # Lead decision 2026-11-13 (match only): with no half-start formation detected,
            # the half's kick-off frame is the organisers' published anchor (games.py), passed
            # in by the caller. Formations found before it cannot be kick-offs of this half
            # (S4: nothing is live before the half's kick-off) and are dropped.
            # Organisers' FAQ: the published pre-kick-off dead span includes the anchor frame
            # (live from the next frame), and a half graded "from the kickoff itself" starts
            # live at its first graded frame (S5). So: first live frame = anchor + 1, unless the
            # anchor lies within ANCHOR_AT_START_FRAMES of the half start (then the half start).
            anchor = int(anchor)
            first_live = h0 if anchor - h0 <= ANCHOR_AT_START_FRAMES else anchor + ANCHOR_LIVE_OFFSET
            n_before = sum(1 for r in half_rows if r["restart_frame"] < first_live)
            half_rows = [r for r in half_rows if r["restart_frame"] >= first_live]
            count["dropped_before_anchor"] = n_before
            half_rows.insert(0, {"start_frame": h0, "restart_frame": first_live,
                                 "start_kind": "half_start", "restart_type": "kick_off",
                                 "p_home_restart": 0.5, "conf": CONF_ANCHOR, "half": half,
                                 "formation_start": -1, "formation_end": -1,
                                 "kick_team_llr": 0.0, "evidence": "published_anchor"})
        elif assumed:
            half_rows.insert(0, {"start_frame": h0, "restart_frame": h0,
                                 "start_kind": "half_start", "restart_type": "kick_off",
                                 "p_home_restart": 0.5, "conf": CONF_ASSUMED, "half": half,
                                 "formation_start": -1, "formation_end": -1,
                                 "kick_team_llr": 0.0, "evidence": assumed})
        else:
            half_rows[0]["start_kind"] = "half_start"
            half_rows[0]["start_frame"] = h0
        rows += half_rows
    return pd.DataFrame(rows, columns=RESTART_COLUMNS)


def _player_break(smooth_players, seg_end, half_stop):
    """End of the players-only formation run that holds at seg_end - 1 (the ball test can end
    a segment earlier than the players do); seg_end when the players-only test does not hold
    there. Keeps the break lag independent of whether the ball is seen."""
    end = int(seg_end)
    if end - 1 < 0 or not smooth_players[end - 1]:
        return end
    while end < half_stop and smooth_players[end]:
        end += 1
    return end


def _mixed_frames(measures, start, stop):
    """Frames in [start, stop) with LIVE_MIXED_MIN or more players in the opponents' half."""
    return int(np.sum(measures["n_wrong_half"].to_numpy()[int(start):int(stop)] >= LIVE_MIXED_MIN))


def _live_before(measures, h0, frame):
    """True when the frames [h0, frame) show sustained live play: at least LIVE_MIXED_FRAMES
    mixed frames. Before a half's kick-off every player is in his own half; in open play the
    mixed count is reached on almost every frame."""
    return _mixed_frames(measures, h0, frame) >= LIVE_MIXED_FRAMES


def apply_kickoff_law(kickoffs):
    """S5/S6: the first kick-off of each of the first two halves is the half-start kick-off; the
    two are taken by different teams. Combine both formation votes into one decision:
    llr(home kicks off half 1) = llr_h1 - llr_h2. Later kick-offs (after goals, R13) keep their
    own formation evidence. Returns a copy with p_home_restart updated and the column
    `law_kick_team_h1` set on the half-start rows (0 home, 1 away)."""
    out = kickoffs.copy()
    out["law_kick_team_h1"] = -1
    starts = out[out["start_kind"] == "half_start"]
    h1 = starts[starts["half"] == 1]
    h2 = starts[starts["half"] == 2]
    llr = 0.0
    if len(h1):
        llr += float(h1["kick_team_llr"].iloc[0])
    if len(h2):
        llr -= float(h2["kick_team_llr"].iloc[0])
    p_h1 = float(1.0 / (1.0 + np.exp(-np.clip(llr, -2 * VOTE_LLR_MAX, 2 * VOTE_LLR_MAX))))
    team_h1 = 0 if llr > 0 else 1 if llr < 0 else -1      # -1: no evidence either way
    if len(h1):
        out.loc[h1.index[0], "p_home_restart"] = p_h1
    if len(h2):
        out.loc[h2.index[0], "p_home_restart"] = 1.0 - p_h1
    out.loc[starts.index, "law_kick_team_h1"] = team_h1
    return out


def pre_kickoff_mask(kickoffs, halves, n_frames):
    """S4 (and S5): bool (n_frames,) True on every frame of a half before that half's detected
    kick-off. A half with no detected kick-off is not masked (the CHECK reports it)."""
    mask = np.zeros(int(n_frames), dtype=bool)
    starts = kickoffs[kickoffs["start_kind"] == "half_start"]
    for half, (h0, _h1) in enumerate(halves, 1):
        row = starts[starts["half"] == half]
        if len(row):
            mask[int(h0):int(row["restart_frame"].iloc[0])] = True
    return mask


def run(players, meta, ball=None, funnel=None, anchors=None):
    """Everything this module produces for one game: (kickoffs DataFrame, features DataFrame
    with frame, half, form_kick_off, mask_pre_kickoff, p_restart_kick_off)."""
    halves = [tuple(h) for h in meta["halves"]]
    n_frames = int(meta["n_frames"])
    measures = kickoff_frame_measures(players, meta, ball)
    kickoffs = apply_kickoff_law(detect_kickoffs(measures, halves, funnel, anchors))
    feats = pd.DataFrame({"frame": measures["frame"].to_numpy(),
                          "half": measures["half"].to_numpy()})
    feats["form_kick_off"] = formation_score(measures).astype(np.float32)
    feats["mask_pre_kickoff"] = pre_kickoff_mask(kickoffs, halves, n_frames)
    p_restart = np.zeros(n_frames, dtype=np.float32)
    for frame, conf in zip(kickoffs["restart_frame"], kickoffs["conf"]):
        p_restart[int(frame)] = max(p_restart[int(frame)], float(conf))
    feats["p_restart_kick_off"] = p_restart
    return kickoffs, feats


__all__ = ["formation_frames", "formation_score", "detect_kickoffs", "apply_kickoff_law",
           "pre_kickoff_mask", "run", "CENTRE_CIRCLE_M", "CIRCLE_MARGIN_M"]
