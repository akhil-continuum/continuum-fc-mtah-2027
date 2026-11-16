"""Smart Tagging XML -> one table row per tagged instance (rulebook section G, DATA.md).

The file is a SportsCode XML: ``<instance>`` elements with ``<code>``, ``<start>`` and ``<end>``
in seconds from the first frame of the tactical video, and ``<label>`` children holding a
``<group>`` / ``<text>`` pair. ``frame = round(seconds * fps)``; intervals are half-open
``[start_frame, end_frame)`` like every other interval in this repository (the starter's
convention, ``starter/load_smart_tagging.py``).

Codes, group names and values are compared case-insensitively and with blanks collapsed.
An instance is kept whatever groups it lacks; only an unreadable or reversed time interval drops
it (counted, never printed). A group that appears twice with two different meanings is treated
as not given: an unclear team is no team, and no team means no vote.

The table lives in memory only. Nothing here prints; the stage prints aggregates.
"""
import math
import xml.etree.ElementTree as ET

import pandas as pd

from pipeline.tagging.teams import NO_TEAM_TEXTS, norm

ATTACKING_CODES = ("BUILD UP", "PROGRESSION", "CREATING CHANCES", "BALL IN FINAL THIRD",
                   "BALL IN THE BOX", "PLAYERS IN THE BOX")
LONG_BALL = "LONG BALL"
DEFENDING_CODES = ("DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
                   "DEFENDING IN DEFENSIVE THIRD")
ATTACKING_TRANSITION = "ATTACKING TRANSITION"
DEFENSIVE_TRANSITION = "DEFENSIVE TRANSITION"
SET_PIECES = "SET PIECES"
GOALS = "GOALS"

FAMILY = {code: "attacking" for code in ATTACKING_CODES}
FAMILY.update({code: "defending" for code in DEFENDING_CODES})
FAMILY.update({LONG_BALL: "long_ball", ATTACKING_TRANSITION: "transition",
               DEFENSIVE_TRANSITION: "transition", SET_PIECES: "set_piece", GOALS: "goal"})
FAMILIES = ("attacking", "defending", "long_ball", "transition", "set_piece", "goal", "other")
KNOWN_CODES = tuple(FAMILY)

# SET PIECES `Type` text (DATA.md), normalised with '-' read as a blank -> set_type
SET_TYPES = {
    "throw in": "throw_in",
    "corner kick": "corner",
    "goal kick": "goal_kick",
    "free kick": "free_kick",
    "kick off (after goal)": "kick_off_after_goal",
    "kick off (start)": "kick_off_start",
}
OTHER_TYPE = "other"          # a Type text that is not in the table above
# set_type -> restart_type in the vocabulary of restarts.parquet (CONTRACTS.md)
RESTART_TYPE = {"throw_in": "throw_in", "corner": "corner", "goal_kick": "goal_kick",
                "free_kick": "free_kick", "kick_off_after_goal": "kick_off",
                "kick_off_start": "kick_off", OTHER_TYPE: OTHER_TYPE}

TEAM_OK, TEAM_NONE, TEAM_UNMAPPED, TEAM_CONFLICT = "team", "none", "unmapped", "conflict"

COLUMNS = ("code", "family", "start_frame", "end_frame", "team", "team_status", "set_type",
           "half")


def norm_code(text):
    """Upper case with single blanks; None becomes ''."""
    return " ".join(str(text).split()).upper() if text is not None else ""


def read_instances(path):
    """([{code, start_s, end_s, labels: [(group, text), ...]}] in file order, n_dropped).

    Dropped: instances whose start or end is absent, not a finite number, or reversed.
    """
    out, dropped = [], 0
    for instance in ET.parse(path).iter("instance"):
        try:
            start_s = float(instance.findtext("start"))
            end_s = float(instance.findtext("end"))
        except (TypeError, ValueError):
            dropped += 1
            continue
        if not (math.isfinite(start_s) and math.isfinite(end_s)) or end_s < start_s:
            dropped += 1
            continue
        labels = [(label.findtext("group"), label.findtext("text"))
                  for label in instance.findall("label")]
        out.append({"code": instance.findtext("code"), "start_s": start_s, "end_s": end_s,
                    "labels": labels})
    return out, dropped


def group_values(labels, group):
    """Every text given for a group (group name compared case-insensitively), normalised."""
    want = norm(group)
    return [norm(text) for name, text in labels if norm(name) == want]


def team_of(labels, aliases):
    """(team, status): team is 0 (home), 1 (away) or -1; see the TEAM_* statuses.

    No Team group, or only "no team" texts: (-1, none). A text outside the alias table:
    (-1, unmapped). Texts naming both sides: (-1, conflict). Otherwise the one side named.
    """
    texts = [t for t in group_values(labels, "team") if t not in NO_TEAM_TEXTS]
    if not texts:
        return -1, TEAM_NONE
    if any(t not in aliases for t in texts):
        return -1, TEAM_UNMAPPED
    sides = {aliases[t] for t in texts}
    if len(sides) != 1:
        return -1, TEAM_CONFLICT
    return int(sides.pop()), TEAM_OK


def set_type_of(labels):
    """'' (no Type), a SET_TYPES value, or OTHER_TYPE (unknown text, or two different types)."""
    texts = {t.replace("-", " ").replace("_", " ") for t in group_values(labels, "type")}
    texts = {" ".join(t.split()) for t in texts if t}
    if not texts:
        return ""
    kinds = {SET_TYPES.get(t, OTHER_TYPE) for t in texts}
    return kinds.pop() if len(kinds) == 1 else OTHER_TYPE


def half_of(labels):
    """1 or 2 from the `Half` text ("1st Half", "2nd Half"); 0 when absent or unclear."""
    halves = {t[:1] for t in group_values(labels, "half") if t[:1] in ("1", "2")}
    return int(halves.pop()) if len(halves) == 1 else 0


def instance_table(instances, fps, aliases):
    """One row per instance: code, family, start_frame, end_frame (half-open), team (0 home,
    1 away, -1 none), team_status, set_type (SET PIECES only, else ''), half (0 = not given)."""
    rows = []
    for inst in instances:
        code = norm_code(inst["code"])
        team, status = team_of(inst["labels"], aliases)
        rows.append({
            "code": code,
            "family": FAMILY.get(code, "other"),
            "start_frame": int(round(inst["start_s"] * fps)),
            "end_frame": int(round(inst["end_s"] * fps)),
            "team": team,
            "team_status": status,
            "set_type": set_type_of(inst["labels"]) if code == SET_PIECES else "",
            "half": half_of(inst["labels"]),
        })
    table = pd.DataFrame(rows, columns=list(COLUMNS))
    for name in ("start_frame", "end_frame", "team", "half"):
        table[name] = table[name].astype("int64")
    return table


def load(path, fps, aliases):
    """(instance table, n_dropped) for a Smart Tagging file."""
    instances, dropped = read_instances(path)
    return instance_table(instances, fps, aliases), dropped
