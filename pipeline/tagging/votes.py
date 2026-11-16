"""Per-frame Smart Tagging evidence over the whole video (rulebook G1 to G9, R14, L5).

Everything here is evidence for the decoder and the team model: no array sets a label, none
moves a boundary (G8), and the absence of a tag is the absence of evidence (G7): a frame
without a tag holds 0, -1, False or '' and nothing else.

Team votes (int8 per code: +1 favours H, -1 favours A, 0 none), on H versus A only (G3):

- G1  attacking codes vote for the tagged team over their whole interval;
- G2  DEFENDING IN ... THIRD votes for the other team;
- G4  ATTACKING TRANSITION votes for the tagged team, DEFENSIVE TRANSITION for the other;
- G9  LONG BALL votes for the tagged team in its first LONG_BALL_VOTE_S seconds only;
- an instance without a team does not vote; where instances of one code with opposite votes
  overlap, the code's vote is 0.

``tag_vote`` is the sign of the unweighted sum of the per-code votes. It is a placeholder: the
weights come from the agreement table (G10, ``pipeline.tagging.agreement``).

Hints:

- ``turnover_hint_to`` (G4): the TURNOVER_HINT_S seconds ending at the start of each transition
  tag; the team that should gain the ball there (0 H, 1 A, -1 none or both);
- ``restart_window``, ``restart_type``, ``restart_team`` (G5, L5, R14): each SET PIECES
  interval; the team is -1 where the tag has none (every throw-in) or where windows disagree;
- ``goal_hint``, ``goal_conceding_team`` (G6): each GOALS interval widened by GOAL_PAD_S on
  both sides; the conceding team is the other team than the tagged one and kicks off (R13).
  Whether the goal is real (a kick-off formation follows, R6) is for the decoder side to check.
"""
import numpy as np
import pandas as pd

from pipeline.tagging.parse import (ATTACKING_CODES, ATTACKING_TRANSITION, DEFENDING_CODES,
                                    DEFENSIVE_TRANSITION, GOALS, LONG_BALL, OTHER_TYPE,
                                    RESTART_TYPE, SET_PIECES)

# Params (rulebook start values; tags have no bench equivalent, so none is bench-measured)
LONG_BALL_VOTE_S = 1.0      # G9: "at the start"
TURNOVER_HINT_S = 2.0       # G4: "within about 2 s before the tag start"
GOAL_PAD_S = 2.0            # G6 "near the tag": twice the edge accuracy of G8 (about 1 s)

TAGGED_TEAM_CODES = ATTACKING_CODES + (LONG_BALL, ATTACKING_TRANSITION)
OTHER_TEAM_CODES = DEFENDING_CODES + (DEFENSIVE_TRANSITION,)
VOTE_CODES = ATTACKING_CODES + (LONG_BALL,) + DEFENDING_CODES + (ATTACKING_TRANSITION,
                                                                 DEFENSIVE_TRANSITION)
RESTART_NAMES = ("throw_in", "corner", "goal_kick", "free_kick", "kick_off", OTHER_TYPE)
_UNSET = -9


def slug(code):
    return code.lower().replace(" ", "_")


def vote_column(code):
    return "vote_" + slug(code)


VOTE_COLUMNS = tuple(vote_column(code) for code in VOTE_CODES)


def vote_sign(code, team):
    """+1 when an instance of this code with this team favours H, -1 when A, 0 for no vote."""
    if team not in (0, 1):
        return 0
    own = 1 if team == 0 else -1
    if code in TAGGED_TEAM_CODES:
        return own
    if code in OTHER_TEAM_CODES:
        return -own
    return 0


def clip(start, end, n_frames):
    """The part of [start, end) inside the video, as (a, b) with a <= b."""
    a = min(max(int(start), 0), n_frames)
    b = min(max(int(end), a), n_frames)
    return a, b


def vote_span(code, start, end, fps):
    """The frames on which an instance votes: all of it, or the first second of a LONG BALL."""
    if code == LONG_BALL:
        end = min(end, start + int(round(LONG_BALL_VOTE_S * fps)))
    return start, end


def hinted_team(code, team):
    """The team a transition tag says has just won the ball (G4), or -1."""
    if team not in (0, 1):
        return -1
    if code == ATTACKING_TRANSITION:
        return team
    if code == DEFENSIVE_TRANSITION:
        return 1 - team
    return -1


def hint_span(start, fps):
    """The turnover window of a transition tag starting at `start`: [start - 2 s, start)."""
    return start - int(round(TURNOVER_HINT_S * fps)), start


def goal_span(start, end, fps):
    pad = int(round(GOAL_PAD_S * fps))
    return start - pad, end + pad


def _rows(table, code):
    part = table[table["code"] == code]
    return zip(part["start_frame"].tolist(), part["end_frame"].tolist(), part["team"].tolist(),
               part["set_type"].tolist())


def code_votes(table, n_frames, fps):
    """{vote column: int8 array (n_frames,)} for every voting code (G1, G2, G4, G9)."""
    out = {}
    for code in VOTE_CODES:
        home = np.zeros(n_frames, dtype=bool)
        away = np.zeros(n_frames, dtype=bool)
        for start, end, team, _ in _rows(table, code):
            sign = vote_sign(code, team)
            if sign == 0:
                continue
            a, b = clip(*vote_span(code, start, end, fps), n_frames)
            (home if sign > 0 else away)[a:b] = True
        out[vote_column(code)] = home.astype(np.int8) - away.astype(np.int8)
    return out


def _assign(arr, a, b, value, clash):
    """Write `value` on arr[a:b]; where another value is already there, write `clash`."""
    seg = arr[a:b]
    arr[a:b] = np.where(seg == _UNSET, value, np.where(seg == value, value, clash))


def turnover_hints(table, n_frames, fps):
    """int8 (n_frames,): 0 = a switch to H is hinted, 1 = to A, -1 = no hint or both (G4)."""
    to = np.full(n_frames, _UNSET, dtype=np.int8)
    for code in (ATTACKING_TRANSITION, DEFENSIVE_TRANSITION):
        for start, _, team, _ in _rows(table, code):
            team_to = hinted_team(code, team)
            if team_to < 0:
                continue
            a, b = clip(*hint_span(start, fps), n_frames)
            _assign(to, a, b, team_to, -1)
    to[to == _UNSET] = -1
    return to


def restart_windows(table, n_frames):
    """(window bool, restart_type str, restart_team int8) per frame (G5, L5, R14).

    Type is '' outside a window and OTHER_TYPE where the tag has no known type or two windows
    of different types overlap. Team is -1 unless every window on the frame names one team.
    """
    kind = np.full(n_frames, _UNSET, dtype=np.int8)
    team_arr = np.full(n_frames, _UNSET, dtype=np.int8)
    other = RESTART_NAMES.index(OTHER_TYPE)
    for start, end, team, set_type in _rows(table, SET_PIECES):
        a, b = clip(start, end, n_frames)
        name = RESTART_TYPE.get(set_type, OTHER_TYPE)
        _assign(kind, a, b, RESTART_NAMES.index(name), other)
        _assign(team_arr, a, b, team if team in (0, 1) else -1, -1)
    window = kind != _UNSET
    names = np.array(RESTART_NAMES + ("",), dtype=object)
    restart_type = names[np.where(window, kind, len(RESTART_NAMES))]
    team_arr[team_arr == _UNSET] = -1
    return window, restart_type, team_arr


def goal_hints(table, n_frames, fps):
    """(goal_hint bool, goal_conceding_team int8) per frame (G6, R13)."""
    conceding = np.full(n_frames, _UNSET, dtype=np.int8)
    for start, end, team, _ in _rows(table, GOALS):
        a, b = clip(*goal_span(start, end, fps), n_frames)
        _assign(conceding, a, b, 1 - team if team in (0, 1) else -1, -1)
    hint = conceding != _UNSET
    conceding[~hint] = -1
    return hint, conceding


def frame_arrays(table, n_frames, fps):
    """tags.parquet as a DataFrame: one row per video frame (CONTRACTS.md, Tags row)."""
    votes = code_votes(table, n_frames, fps)
    total = np.zeros(n_frames, dtype=np.int16)
    for column in VOTE_COLUMNS:
        total += votes[column]
    window, restart_type, restart_team = restart_windows(table, n_frames)
    goal_hint, conceding = goal_hints(table, n_frames, fps)
    out = {"frame": np.arange(n_frames, dtype=np.int32),
           "tag_vote": np.sign(total).astype(np.float32)}
    out.update(votes)
    out["turnover_hint_to"] = turnover_hints(table, n_frames, fps)
    out["restart_window"] = window
    out["restart_type"] = restart_type
    out["restart_team"] = restart_team
    out["goal_hint"] = goal_hint
    out["goal_conceding_team"] = conceding
    return pd.DataFrame(out)
