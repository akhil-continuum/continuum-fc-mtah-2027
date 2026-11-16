"""Read a FIFA/EPTS tracking export (Metrica): the metadata XML and the raw text file.

The raw file has one line per frame. Its layout is not fixed: the metadata's
``DataFormatSpecifications`` describe, for each range of frames, which register comes where and
which separators split them. Metrica Sample Game 3 has eleven specifications because the
player-to-group order changes at substitutions, so reading groups in the order of the ``Player``
list (as the starter baseline does) is only safe when there is a single specification. This module
follows the specification for every frame range.

Nothing here prints. Errors carry counts and names, never values from the file.

Fast path: every separator of a specification is translated to a comma, so a line becomes a flat
row of fields in the order of the specification's leaves, and pandas' C parser reads the whole
file at once.
"""
import io
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Params
FIELD = ","  # every separator is translated to this before parsing
FORBIDDEN_SEPARATORS = set("0123456789.-+eENaInf \t\r\n")


def _local(tag):
    """Tag name without an XML namespace."""
    return tag.rsplit("}", 1)[-1]


def _child(node, name):
    for sub in node:
        if _local(sub.tag) == name:
            return sub
    return None


def _children(node, name):
    return [sub for sub in node if _local(sub.tag) == name]


def _iter(node, name):
    return [sub for sub in node.iter() if _local(sub.tag) == name]


def _text(node, name, default=None):
    sub = _child(node, name) if node is not None else None
    return sub.text.strip() if sub is not None and sub.text is not None else default


def channel_kind(channel_id, name=""):
    """Normalise a channel to x, y, z, s (speed) or its own id."""
    cid = (channel_id or "").strip().lower()
    nm = (name or "").strip().lower()
    if cid in ("x", "y", "z"):
        return cid
    for kind, pattern in (("x", r"(^|_)x$"), ("y", r"(^|_)y$"), ("z", r"(^|_)z$")):
        if re.search(pattern, nm) or re.search(pattern, cid):
            return kind
    if "height" in nm or "height" in cid:
        return "z"
    if cid in ("s", "speed") or "speed" in nm:
        return "s"
    return cid or nm


@dataclass
class Leaf:
    """One field of a raw line, in order."""
    role: str        # "frame", "player", "ball", "string" or "other"
    ref: str = ""    # player channel id, ball channel id or register name
    tag: str = ""    # XML tag of the reference, for the channel audit


@dataclass
class Spec:
    """One DataFormatSpecification: a frame range and its flat list of leaves."""
    start: int
    end: int
    separators: list
    leaves: list
    player_groups: list = field(default_factory=list)  # number of leaves in each player group
    ball_leaves: int = 0
    top_separator: str = ":"

    @property
    def n_fields(self):
        return len(self.leaves)


@dataclass
class EptsMeta:
    path: str
    fps: float
    field_size: tuple                 # (length, width) from FieldSize, metres
    teams: list                       # [(team_id, name)] in file order
    home_team_id: str
    away_team_id: str
    players: dict                     # player_id -> {"team_id", "name", "shirt"}
    channels: dict                    # channel_id -> {"name", "unit", "sensor", "kind"}
    player_channels: dict             # player channel id -> (player_id, channel_id)
    specs: list                       # [Spec]
    provider_params: dict             # global provider parameters, name -> text
    extra_channel_tags: dict          # unknown tags found anywhere -> count

    def player_order(self):
        """Player ids in metadata order: the slot order of the output."""
        return list(self.players)


def _flatten(node, leaves, separators, groups, depth=0):
    """Depth-first leaves of a register tree. Records player group sizes at depth 1."""
    for sub in node:
        tag = _local(sub.tag)
        if tag == "StringRegister":
            name = sub.get("name", "")
            role = "frame" if name.lower() in ("frame", "framecount", "frameid", "frame_id") \
                else "string"
            leaves.append(Leaf(role, name, tag))
        elif tag == "SplitRegister":
            sep = sub.get("separator")
            if sep is not None:
                separators.append(sep)
            before = len(leaves)
            _flatten(sub, leaves, separators, groups, depth + 1)
            inner = leaves[before:]
            if inner and all(leaf.role == "player" for leaf in inner) and not any(
                    _local(s.tag) == "SplitRegister" for s in sub):
                groups.append(len(inner))
        elif tag == "PlayerChannelRef":
            leaves.append(Leaf("player", sub.get("playerChannelId", ""), tag))
        elif tag == "BallChannelRef":
            leaves.append(Leaf("ball", sub.get("channelId", ""), tag))
        else:
            ref = sub.get("channelId") or sub.get("name") or sub.get("id") or ""
            leaves.append(Leaf("other", ref, tag))


def read_metadata(path):
    """Parse the EPTS metadata XML into an EptsMeta."""
    root = ET.parse(path).getroot()
    meta = _child(root, "Metadata")
    if meta is None:
        meta = root

    glob = _iter(meta, "GlobalConfig")
    fps = float(_text(glob[0], "FrameRate", "25")) if glob else 25.0
    provider = {}
    for gp in _iter(meta, "ProviderGlobalParameters"):
        for p in _children(gp, "ProviderParameter"):
            provider[_text(p, "Name", "")] = _text(p, "Value", None)

    size = _iter(meta, "FieldSize")
    length = float(_text(size[0], "Width", "105")) if size else 105.0
    width = float(_text(size[0], "Height", "68")) if size else 68.0

    teams = [(t.get("id"), _text(t, "Name", "")) for t in _iter(meta, "Team")]
    score = _iter(meta, "Score")
    home_id = score[0].get("idLocalTeam") if score else None
    away_id = score[0].get("idVisitingTeam") if score else None
    team_ids = [t[0] for t in teams]
    if home_id not in team_ids:
        # FIFA/EPTS convention, and the one kloppy follows: the first team listed is home.
        home_id = team_ids[0] if team_ids else None
    if away_id not in team_ids or away_id == home_id:
        others = [t for t in team_ids if t != home_id]
        away_id = others[0] if others else None

    players = {}
    for p in _iter(meta, "Player"):
        players[p.get("id")] = {"team_id": p.get("teamId"), "name": _text(p, "Name", ""),
                                "shirt": _text(p, "ShirtNumber", "")}

    channels = {}
    for sensor in _iter(meta, "Sensor"):
        for ch in _iter(sensor, "Channel"):
            name = _text(ch, "Name", "")
            channels[ch.get("id")] = {"name": name, "unit": _text(ch, "Unit", ""),
                                      "sensor": sensor.get("id"),
                                      "kind": channel_kind(ch.get("id"), name)}

    player_channels = {}
    for pc in _iter(meta, "PlayerChannel"):
        player_channels[pc.get("id")] = (pc.get("playerId"), pc.get("channelId"))

    specs = []
    for ds in _iter(root, "DataFormatSpecification"):
        leaves, seps, groups = [], [], []
        top_sep = ds.get("separator")
        if top_sep is not None:
            seps.append(top_sep)
        _flatten(ds, leaves, seps, groups)
        start = int(ds.get("startFrame")) if ds.get("startFrame") else None
        end = int(ds.get("endFrame")) if ds.get("endFrame") else None
        specs.append(Spec(start, end, sorted(set(seps)), leaves, groups,
                          sum(1 for leaf in leaves if leaf.role == "ball"), top_sep or ":"))

    known = {"Metadata", "GlobalConfig", "Sessions", "Teams", "Players", "Devices",
             "PlayerChannels", "DataFormatSpecifications"}
    extra = {}
    for sub in meta:
        tag = _local(sub.tag)
        if tag not in known:
            extra[tag] = extra.get(tag, 0) + 1
    for tag in ("BallChannel", "BallChannels", "TeamChannel", "TeamChannels", "SessionChannel",
                "SessionChannels", "BallState", "BallOwner", "BallOwnerChannel"):
        n = len(_iter(root, tag))
        if n:
            extra[tag] = extra.get(tag, 0) + n

    return EptsMeta(str(path), fps, (length, width), teams, home_id, away_id, players, channels,
                    player_channels, specs, provider, extra)


def _check_separators(spec):
    bad = [s for s in spec.separators if (len(s) != 1) or (s in FORBIDDEN_SEPARATORS)]
    if bad:
        raise ValueError("unsupported separator in a DataFormatSpecification ({} of them)".format(
            len(bad)))


def read_raw_table(raw_path, meta):
    """The raw file as a DataFrame of flat fields (columns 0..n-1), one row per line.

    Also returns the per-line field count, so callers can check each line against the
    specification that covers its frame.
    """
    for spec in meta.specs:
        _check_separators(spec)
    seps = sorted({s for spec in meta.specs for s in spec.separators} - {FIELD})
    with open(raw_path, "rb") as handle:
        blob = handle.read()
    if seps:
        blob = blob.translate(bytes.maketrans("".join(seps).encode(), FIELD.encode() * len(seps)))
    blob = blob.replace(b"\r", b"")
    if not blob.endswith(b"\n"):
        blob += b"\n"

    # Field count per non-empty line, from separator counts: fields = commas + 1.
    buf = np.frombuffer(blob, dtype=np.uint8)
    newline = np.flatnonzero(buf == ord("\n"))
    commas = np.concatenate(([0], np.cumsum(buf == ord(FIELD))))
    starts = np.concatenate(([0], newline[:-1] + 1))
    nonempty = newline > starts
    n_fields = (commas[newline] - commas[starts] + 1)[nonempty]

    width = int(n_fields.max()) if len(n_fields) else 0
    table = pd.read_csv(io.BytesIO(blob), header=None, names=list(range(width)),
                        na_values=["NaN", "nan", ""], keep_default_na=False,
                        low_memory=False, engine="c")
    return table, n_fields


def spec_for_frames(meta, frame_ids):
    """Index of the specification covering each frame id, -1 where none does."""
    out = np.full(len(frame_ids), -1, dtype=np.int64)
    if len(meta.specs) == 1 and meta.specs[0].start is None:
        out[:] = 0
        return out
    for i, spec in enumerate(meta.specs):
        lo = spec.start if spec.start is not None else -np.inf
        hi = spec.end if spec.end is not None else np.inf
        out[(frame_ids >= lo) & (frame_ids <= hi) & (out < 0)] = i
    return out


@dataclass
class RawTracking:
    """Frame-wide arrays straight from the raw file, before any cleaning.

    Arrays are indexed by position in ``frame_ids`` (one entry per line of the file).
    """
    frame_ids: np.ndarray          # int64, one per line
    player_ids: list               # slot order
    player_team: np.ndarray        # int8 per slot: 0 home, 1 away, -1 neither
    px: np.ndarray                 # (n_lines, n_slots) normalised 0-1, NaN when absent
    py: np.ndarray
    pextra: dict                   # kind -> (n_lines, n_slots) for other numeric player channels
    ball: dict                     # kind -> (n_lines,) ball channels, numeric or object
    n_bad_lines: int               # lines whose field count disagrees with their specification
    n_unspecified: int             # lines whose frame no specification covers


def read_tracking(meta, raw_path):
    """Read the raw file following the metadata's specifications."""
    table, n_fields = read_raw_table(raw_path, meta)
    if not meta.specs:
        raise ValueError("metadata has no DataFormatSpecification")

    frame_pos = [i for i, leaf in enumerate(meta.specs[0].leaves) if leaf.role == "frame"]
    if not frame_pos:
        raise ValueError("specification has no frame register")
    frame_ids = pd.to_numeric(table[frame_pos[0]], errors="coerce").to_numpy()
    if np.isnan(frame_ids).any():
        raise ValueError("{} lines without a numeric frame id".format(
            int(np.isnan(frame_ids).sum())))
    frame_ids = frame_ids.astype(np.int64)

    spec_idx = spec_for_frames(meta, frame_ids)
    order = meta.player_order()
    slot_of = {pid: i for i, pid in enumerate(order)}
    team_code = {meta.home_team_id: 0, meta.away_team_id: 1}
    player_team = np.array([team_code.get(meta.players[p]["team_id"], -1) for p in order],
                           dtype=np.int8)

    n = len(frame_ids)
    px = np.full((n, len(order)), np.nan)
    py = np.full((n, len(order)), np.nan)
    pextra, ball = {}, {}
    n_bad = 0

    for i, spec in enumerate(meta.specs):
        rows = np.flatnonzero(spec_idx == i)
        if not len(rows):
            continue
        bad = n_fields[rows] != spec.n_fields
        n_bad += int(bad.sum())
        rows = rows[~bad]
        for col, leaf in enumerate(spec.leaves):
            if leaf.role == "player":
                pid, cid = meta.player_channels.get(leaf.ref, (None, None))
                if pid not in slot_of:
                    continue
                kind = meta.channels.get(cid, {}).get("kind") or channel_kind(cid)
                values = pd.to_numeric(table[col].iloc[rows], errors="coerce").to_numpy()
                if kind == "x":
                    px[rows, slot_of[pid]] = values
                elif kind == "y":
                    py[rows, slot_of[pid]] = values
                else:
                    arr = pextra.setdefault(kind, np.full((n, len(order)), np.nan))
                    arr[rows, slot_of[pid]] = values
            elif leaf.role in ("ball", "other", "string"):
                kind = leaf.ref
                if leaf.role == "ball":
                    kind = meta.channels.get(leaf.ref, {}).get("kind") or channel_kind(leaf.ref)
                else:
                    kind = "{}_{}".format(leaf.role, leaf.ref or col)
                column = table[col].iloc[rows]
                numeric = pd.to_numeric(column, errors="coerce")
                if numeric.notna().sum() >= column.notna().sum():
                    arr = ball.setdefault(kind, np.full(n, np.nan))
                    arr[rows] = numeric.to_numpy()
                else:
                    arr = ball.setdefault(kind, np.full(n, None, dtype=object))
                    arr[rows] = column.astype(object).to_numpy()

    n_unspec = int((spec_idx < 0).sum())
    return RawTracking(frame_ids, order, player_team, px, py, pextra, ball, n_bad, n_unspec)
