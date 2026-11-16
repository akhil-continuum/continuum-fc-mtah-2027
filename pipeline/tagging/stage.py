"""Tagging stage on one root: Smart Tagging XML -> ``<root>/tags.parquet`` plus aggregates.

``run(root)`` is what ``python -m pipeline.tagging match`` and ``... bench`` call (shared stage
command line, ``pipeline/ball/stage.py``). The tagging file is found through ``games.py`` for a
root whose ``meta.json`` says ``source: match``; every other root has no tags: the stage then
returns ``{"tags": "no tags"}`` and writes nothing.

The returned dict holds aggregates only (counts, shares, histograms of at most 20 bins, per-code
and per-family numbers). No instance, interval, time or frame number is ever returned or printed:
the caller prints the dict through ``pipeline.report.summary``.
"""
import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline.decode.inputs import graded_mask
from pipeline.io import epts, sources
from pipeline.tagging import parse, teams, votes

games = sources.games

TAGS_FILE = "tags.parquet"
NO_TAGS = "no tags"
DURATION_EDGES_S = (0, 1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 120)   # last bin is open-ended
GOAL_KICKOFF_MAX_S = 180.0    # a kick-off tag this soon after a GOALS tag counts as "follows"
MAX_OTHER_CODES = 5           # unknown codes are counted together, never listed


@dataclass
class Tags:
    meta: dict
    n_frames: int
    fps: float
    graded: np.ndarray        # bool (n_frames,)
    table: pd.DataFrame       # pipeline.tagging.parse.instance_table
    dropped: int              # instances with an unreadable interval
    alias_check: str          # agree / disagree / unknown / not checked


def read_meta(root):
    with open(os.path.join(root, "meta.json")) as handle:
        return json.load(handle)


def tagging_path(meta):
    """Path of the root's Smart Tagging file, or None when the root has none (bench games)."""
    if meta.get("source") != "match":
        return None
    return games.require(["smart_tagging"], game=meta["game_id"])[0]


def tracking_team_names(meta):
    """(home name, away name) from the match's tracking metadata, or (None, None)."""
    try:
        path = games.require(["atd_meta"], game=meta["game_id"])[0]
        info = epts.read_metadata(path)
    except (OSError, KeyError, ValueError, ET.ParseError):
        return None, None
    names = dict(info.teams)
    return names.get(info.home_team_id), names.get(info.away_team_id)


def load(root, xml_path=None, aliases=None):
    """Tags of a root, or None when it has no tagging file.

    ``xml_path`` and ``aliases`` are for tests and synthetic roots; by default the file comes
    from games.py (match roots only) and the team aliases from ``pipeline.tagging.teams``.
    """
    meta = read_meta(root)
    path = xml_path or tagging_path(meta)
    if path is None:
        return None
    if aliases is None:
        aliases = teams.aliases_for(meta.get("game_id"))
    n_frames = int(meta["n_frames"])
    fps = float(meta.get("fps", 25))
    table, dropped = parse.load(path, fps, aliases)
    check = "not checked"
    if meta.get("source") == "match":
        check = teams.check_against_tracking(aliases, *tracking_team_names(meta))
    return Tags(meta=meta, n_frames=n_frames, fps=fps, graded=graded_mask(meta, n_frames)[0],
                table=table, dropped=dropped, alias_check=check)


def covered(table, code, n_frames):
    """bool (n_frames,): frames inside any instance of the code, whatever its team."""
    mask = np.zeros(n_frames, dtype=bool)
    part = table[table["code"] == code]
    for start, end in zip(part["start_frame"].tolist(), part["end_frame"].tolist()):
        a, b = votes.clip(start, end, n_frames)
        mask[a:b] = True
    return mask


def _share(part, whole):
    return round(float(part) / float(whole), 4) if whole else None


def _counts_by_code(table):
    counts = {code: int((table["code"] == code).sum()) for code in parse.KNOWN_CODES}
    unknown = table[~table["code"].isin(parse.KNOWN_CODES)]
    counts["OTHER"] = int(len(unknown))
    return counts, int(unknown["code"].nunique())


def _duration_hists(table, fps):
    edges = list(DURATION_EDGES_S) + [np.inf]
    seconds = (table["end_frame"] - table["start_frame"]).to_numpy() / fps
    out = {}
    for family in parse.FAMILIES:
        part = seconds[(table["family"] == family).to_numpy()]
        if family == "other" and not len(part):
            continue
        out["duration_s_hist_" + family] = [int(v) for v in np.histogram(part, bins=edges)[0]]
    return out


def _half_check(tags):
    """Share of instances with a Half tag whose middle lies in that half's video block."""
    halves = [tuple(h) for h in tags.meta["halves"]]
    table = tags.table
    middle = ((table["start_frame"] + table["end_frame"]) // 2).to_numpy()
    half = table["half"].to_numpy()
    tagged = inside = 0
    for number, (first, stop) in enumerate(halves, start=1):
        rows = half == number
        tagged += int(rows.sum())
        inside += int(((middle[rows] >= first) & (middle[rows] < stop)).sum())
    return _share(inside, tagged)


def _goal_tag_check(table, fps):
    """GOALS tags, and how many are followed by a 'Kick Off (After Goal)' tag (in-file check)."""
    goals = table[table["code"] == parse.GOALS]
    kicks = table[(table["code"] == parse.SET_PIECES)
                  & (table["set_type"] == "kick_off_after_goal")]
    reach = int(round(GOAL_KICKOFF_MAX_S * fps))
    followed = pairs = agree = 0
    for start, team in zip(goals["start_frame"].tolist(), goals["team"].tolist()):
        near = kicks[(kicks["start_frame"] >= start) & (kicks["start_frame"] <= start + reach)]
        if not len(near):
            continue
        followed += 1
        kick_team = int(near.sort_values("start_frame")["team"].iloc[0])
        if team in (0, 1) and kick_team in (0, 1):
            pairs += 1
            agree += int(kick_team == 1 - team)
    return {"goal_tags": int(len(goals)),
            "goal_tags_with_team": int(goals["team"].isin([0, 1]).sum()),
            "goal_tags_followed_by_kickoff_tag": followed,
            "goal_kickoff_tag_team_pairs": pairs,
            "goal_kickoff_tag_is_conceding_team": agree}


def aggregates(tags, frames):
    """Everything the stage prints: aggregates of the instance table and the frame arrays."""
    table, n, graded, fps = tags.table, tags.n_frames, tags.graded, tags.fps
    n_graded = int(graded.sum())
    by_code, n_unknown = _counts_by_code(table)
    clipped = [votes.clip(a, b, n) for a, b in zip(table["start_frame"].tolist(),
                                                   table["end_frame"].tolist())]
    out = {
        "tags": "found",
        "instances_total": int(len(table)),
        "instances_dropped_unreadable_interval": int(tags.dropped),
        "instances_empty_or_outside_video": int(sum(1 for a, b in clipped if a == b)),
        "instances_by_code": by_code,
        "unknown_codes_distinct": n_unknown,
        "instances_by_team": {
            "home": int((table["team"] == 0).sum()),
            "away": int((table["team"] == 1).sum()),
            "no_team": int((table["team_status"] == parse.TEAM_NONE).sum()),
            "team_text_unmapped": int((table["team_status"] == parse.TEAM_UNMAPPED).sum()),
            "team_groups_conflict": int((table["team_status"] == parse.TEAM_CONFLICT).sum())},
        "instances_by_half_tag": {"1": int((table["half"] == 1).sum()),
                                  "2": int((table["half"] == 2).sum()),
                                  "none": int((table["half"] == 0).sum())},
        "share_half_tagged_instances_inside_that_half": _half_check(tags),
        "team_alias_check_against_tracking_meta": tags.alias_check,
        "n_frames": n,
        "graded_frames": n_graded,
    }

    out["graded_frames_covered_by_code"] = {
        code: int((covered(table, code, n) & graded).sum()) for code in parse.KNOWN_CODES}
    stack = np.stack([frames[c].to_numpy() for c in votes.VOTE_COLUMNS])
    out["graded_frames_voting_by_code"] = {
        code: int(((stack[i] != 0) & graded).sum()) for i, code in enumerate(votes.VOTE_CODES)}
    home = (stack > 0).any(axis=0)
    away = (stack < 0).any(axis=0)
    voting = (home | away) & graded
    out["share_graded_with_a_voting_tag"] = _share(voting.sum(), n_graded)
    out["share_voting_frames_codes_conflict"] = _share((home & away & graded).sum(),
                                                       voting.sum())
    tag_vote = frames["tag_vote"].to_numpy()
    out["share_graded_tag_vote_nonzero"] = _share(((tag_vote != 0) & graded).sum(), n_graded)
    out["share_tag_vote_home"] = _share(((tag_vote > 0) & graded).sum(),
                                        ((tag_vote != 0) & graded).sum())
    per_frame = (stack != 0).sum(axis=0)[graded]
    out["voting_codes_per_graded_frame_hist"] = [
        int(v) for v in np.bincount(per_frame, minlength=len(votes.VOTE_CODES) + 1)]

    sets = table[table["code"] == parse.SET_PIECES]
    kinds = sets["set_type"].replace("", "no_type")
    out["set_pieces_by_type"] = {str(k): int(v) for k, v in kinds.value_counts().items()}
    out["set_pieces_with_team_by_type"] = {
        str(k): int(v) for k, v in kinds[sets["team"].isin([0, 1])].value_counts().items()}
    window = frames["restart_window"].to_numpy()
    out["restart_window_frames_graded"] = int((window & graded).sum())
    out["share_graded_in_restart_window"] = _share((window & graded).sum(), n_graded)
    out["restart_window_frames_with_team"] = int(
        ((frames["restart_team"].to_numpy() >= 0) & graded).sum())

    hint = frames["turnover_hint_to"].to_numpy()
    trans = table[table["code"].isin([parse.ATTACKING_TRANSITION, parse.DEFENSIVE_TRANSITION])]
    hinted = [votes.hinted_team(c, t) for c, t in zip(trans["code"].tolist(),
                                                      trans["team"].tolist())]
    out["turnover_hints_by_team"] = {"to_home": hinted.count(0), "to_away": hinted.count(1),
                                     "no_team": hinted.count(-1)}
    out["turnover_hint_frames_graded"] = int(((hint >= 0) & graded).sum())

    out.update(_goal_tag_check(table, fps))
    out["goal_hint_frames_graded"] = int((frames["goal_hint"].to_numpy() & graded).sum())

    out["duration_s_hist_edges"] = list(DURATION_EDGES_S)
    out.update(_duration_hists(table, fps))
    return out


def run(root, xml_path=None, aliases=None):
    """Write ``<root>/tags.parquet`` and return the aggregates; "no tags" and nothing written
    when the root has no tagging file."""
    tags = load(root, xml_path=xml_path, aliases=aliases)
    if tags is None:
        return {"tags": NO_TAGS, "written": False}
    frames = votes.frame_arrays(tags.table, tags.n_frames, tags.fps)
    frames.to_parquet(os.path.join(root, TAGS_FILE), index=False)
    stats = aggregates(tags, frames)
    stats["written"] = True
    stats["columns_written"] = int(frames.shape[1])
    return stats
