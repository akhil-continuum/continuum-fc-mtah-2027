"""Minimal readers for the open Metrica sample games (bench data only).

TEMPORARY: the data-plumber owns the shared EPTS loader (pipeline/load). Once it exists, the
EPTS part of this module (Sample Game 3) should be swapped for it. The CSV reader for games 1 and 2
can stay here: those files exist only on the bench.

Output is one `Game` per sample game, in a single convention used by every bench module:

- frames are 0-based: bench frame = Metrica "Frame" - 1 (Metrica counts from 1, like the match
  tracking, and submissions count from 0);
- coordinates are metres with the origin at the centre spot, x along the pitch (-52.5..52.5),
  y across (-34..34), y positive towards the top of the Metrica image (Metrica y=0);
- team code 0 = home (H), 1 = away (A);
- missing observations are NaN, exactly as in the source files (nothing is filled).
"""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "bench_data", "raw", "metrica")

PITCH = (105.0, 68.0)  # Metrica sample-data pitch, metres
FPS = 25

GAME_IDS = ("metrica_g1", "metrica_g2", "metrica_g3")


@dataclass
class Game:
    game_id: str
    period: np.ndarray            # (n,) int, 1 or 2
    ball: np.ndarray              # (n, 2) metres, NaN when missing
    players: np.ndarray           # (n, P, 2) metres, NaN when not on the pitch / not tracked
    player_team: np.ndarray       # (P,) 0 = H, 1 = A
    player_ids: list              # (P,) source player names, e.g. "Player11"
    events: pd.DataFrame          # standardised, see _std_events
    pitch: tuple = PITCH
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return len(self.period)

    def halves(self):
        """[(start, stop)] half-open frame spans per period, in order."""
        out = []
        for p in sorted(set(self.period.tolist())):
            idx = np.flatnonzero(self.period == p)
            out.append((int(idx[0]), int(idx[-1]) + 1))
        return out


def to_metres(x, y, pitch=PITCH):
    """Metrica normalised (0..1, y down) to centred metres (y up)."""
    length, width = pitch
    return (np.asarray(x, float) - 0.5) * length, (0.5 - np.asarray(y, float)) * width


# -- games 1 and 2: CSV -----------------------------------------------------------------------

def _read_team_csv(path):
    """Return (frame, period, ball_xy_norm, {player: xy_norm})."""
    with open(path) as handle:
        handle.readline()
        handle.readline()
        header = handle.readline().strip().split(",")
    table = pd.read_csv(path, skiprows=3, header=None)
    names = []
    for i, name in enumerate(header):
        if name.startswith("Player") or name == "Ball":
            names.append((name, i))
    period = table[0].to_numpy(int)
    frame = table[1].to_numpy(int)
    data = {}
    for name, i in names:
        data[name] = table[[i, i + 1]].to_numpy(float)
    ball = data.pop("Ball")
    return frame, period, ball, data


def _std_events(rows, home_name, away_name):
    """Standardise event rows to one schema.

    Columns: team (H/A), type, subtype ('' when none), period, start_frame, end_frame (0-based
    bench frames), from_player, to_player, sx, sy, ex, ey (metres, NaN when not given).
    """
    ev = pd.DataFrame(rows)
    ev["team"] = ev["team"].map({home_name: "H", away_name: "A"})
    if ev["team"].isna().any():
        raise ValueError("event with unknown team")
    ev["subtype"] = ev["subtype"].fillna("").astype(str)
    for col in ("start_frame", "end_frame"):
        ev[col] = ev[col].astype(int) - 1
    # Metrica writes end frame 0 for instantaneous events in the CSV (e.g. KICK OFF rows).
    bad = ev["end_frame"] < ev["start_frame"]
    ev.loc[bad, "end_frame"] = ev.loc[bad, "start_frame"]
    ev["sx"], ev["sy"] = to_metres(ev["sx"], ev["sy"])
    ev["ex"], ev["ey"] = to_metres(ev["ex"], ev["ey"])
    ev = ev.reset_index(drop=True)
    ev["order"] = np.arange(len(ev))  # source order breaks ties at equal frames
    return ev


def load_csv_game(number: int, raw_dir: str = RAW_DIR) -> Game:
    folder = os.path.join(raw_dir, "Sample_Game_{}".format(number))
    stem = os.path.join(folder, "Sample_Game_{}_".format(number))
    f_h, p_h, ball_h, home = _read_team_csv(stem + "RawTrackingData_Home_Team.csv")
    f_a, p_a, ball_a, away = _read_team_csv(stem + "RawTrackingData_Away_Team.csv")
    if not (np.array_equal(f_h, f_a) and np.array_equal(p_h, p_a)):
        raise ValueError("home and away tracking files are not frame-aligned")
    if not np.array_equal(f_h, np.arange(1, len(f_h) + 1)):
        raise ValueError("tracking frames are not 1..n contiguous")
    ball_diff = np.nanmax(np.abs(ball_h - ball_a)) if np.isfinite(ball_h).any() else 0.0
    if ball_diff > 1e-6 or not np.array_equal(np.isnan(ball_h), np.isnan(ball_a)):
        raise ValueError("home and away ball columns disagree")

    ids, teams, xy = [], [], []
    for team_code, block in ((0, home), (1, away)):
        for name, arr in block.items():
            ids.append(name)
            teams.append(team_code)
            xy.append(np.stack(to_metres(arr[:, 0], arr[:, 1]), axis=1))
    players = np.stack(xy, axis=1)
    ball = np.stack(to_metres(ball_h[:, 0], ball_h[:, 1]), axis=1)

    raw = pd.read_csv(stem + "RawEventsData.csv")
    rows = pd.DataFrame({
        "team": raw["Team"], "type": raw["Type"], "subtype": raw["Subtype"],
        "period": raw["Period"], "start_frame": raw["Start Frame"],
        "end_frame": raw["End Frame"], "from_player": raw["From"], "to_player": raw["To"],
        "sx": raw["Start X"], "sy": raw["Start Y"], "ex": raw["End X"], "ey": raw["End Y"],
    })
    events = _std_events(rows, "Home", "Away")
    return Game("metrica_g{}".format(number), p_h, ball, players, np.array(teams, np.int8),
                ids, events, meta={"source": "metrica sample game {} (CSV)".format(number)})


# -- game 3: EPTS (txt + xml) and JSON events -------------------------------------------------

def _epts_layout(meta_path):
    """Teams, players, half spans and the per-frame-range player order of a data line.

    Metrica's EPTS export has one DataFormatSpecification per lineup (substitutions change who
    occupies the 22 player groups), each with startFrame/endFrame and an ordered list of
    PlayerChannelRefs (x then y per player).
    """
    root = ET.parse(meta_path).getroot()
    teams = [(t.get("id"), t.findtext("Name")) for t in root.iter("Team")]
    players = {}
    for p in root.iter("Player"):
        players[p.get("id")] = {"team": p.get("teamId"), "name": p.findtext("Name")}
    pchan = {pc.get("id"): pc.get("playerId") for pc in root.iter("PlayerChannel")}
    specs = []
    for spec in root.iter("DataFormatSpecification"):
        order = []
        for ref in spec.iter("PlayerChannelRef"):
            pid = pchan[ref.get("playerChannelId")]
            if pid not in order:
                order.append(pid)
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), order))
    params = {}
    for pp in root.find(".//ProviderGlobalParameters").iter("ProviderParameter"):
        params[pp.findtext("Name")] = pp.findtext("Value")
    halves = []
    for k in ("first_half", "second_half"):
        if params.get(k + "_start"):
            halves.append((int(params[k + "_start"]), int(params[k + "_end"])))
    return teams, players, specs, halves


def load_epts_game(raw_dir: str = RAW_DIR) -> Game:
    folder = os.path.join(raw_dir, "Sample_Game_3")
    meta_path = os.path.join(folder, "Sample_Game_3_metadata.xml")
    txt_path = os.path.join(folder, "Sample_Game_3_tracking.txt")
    ev_path = os.path.join(folder, "Sample_Game_3_events.json")

    teams, players, specs, half_spans = _epts_layout(meta_path)
    if len(teams) != 2:
        raise ValueError("expected two teams")
    home_id, away_id = teams[0][0], teams[1][0]
    all_players = []
    for _, _, order in specs:
        for pid in order:
            if pid not in all_players:
                all_players.append(pid)
    col = {pid: i for i, pid in enumerate(all_players)}

    frames, rows_p, rows_b = [], [], []
    spec_i = 0
    with open(txt_path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            fr, ps, bl = line.split(":")
            fr = int(fr)
            while not (specs[spec_i][0] <= fr <= specs[spec_i][1]):
                spec_i += 1
                if spec_i >= len(specs):
                    raise ValueError("frame {} outside every data spec".format(fr))
            order = specs[spec_i][2]
            row = np.full((len(all_players), 2), np.nan)
            for pid, group in zip(order, ps.split(";")):
                x, y = group.split(",")[:2]
                row[col[pid]] = (float(x), float(y))
            frames.append(fr)
            rows_p.append(row)
            bx_, by_ = bl.split(",")[:2]
            rows_b.append((float(bx_), float(by_)))
    frames = np.array(frames)
    pnorm, bnorm = np.array(rows_p), np.array(rows_b, float)
    if not np.array_equal(frames, np.arange(1, len(frames) + 1)):
        raise ValueError("EPTS frames are not 1..n contiguous")
    period = np.zeros(len(frames), int)
    for k, (a, b) in enumerate(half_spans):
        period[(frames >= a) & (frames <= b)] = k + 1
    keep = period > 0
    frames, pnorm, bnorm, period = frames[keep], pnorm[keep], bnorm[keep], period[keep]
    if frames[0] != 1:
        raise ValueError("first half does not start at frame 1")

    px, py = to_metres(pnorm[..., 0], pnorm[..., 1])
    bx, by = to_metres(bnorm[:, 0], bnorm[:, 1])
    team_code = np.array([0 if players[p]["team"] == home_id else 1 for p in all_players], np.int8)

    with open(ev_path) as handle:
        raw = json.load(handle)["data"]
    rows = []
    for e in raw:
        sub = e.get("subtypes")
        if isinstance(sub, list):
            sub = "-".join(s["name"] for s in sub)
        elif isinstance(sub, dict):
            sub = sub["name"]
        else:
            sub = ""
        start, end = e.get("start") or {}, e.get("end") or {}
        rows.append({
            "team": e["team"]["id"] if e.get("team") else None,
            "type": e["type"]["name"], "subtype": sub, "period": e["period"],
            "start_frame": start.get("frame"),
            "end_frame": end.get("frame") if end.get("frame") is not None else start.get("frame"),
            "from_player": (e.get("from") or {}).get("name"),
            "to_player": (e.get("to") or {}).get("name"),
            "sx": start.get("x"), "sy": start.get("y"), "ex": end.get("x"), "ey": end.get("y"),
        })
    rows = pd.DataFrame(rows)
    for c in ("sx", "sy", "ex", "ey"):
        rows[c] = pd.to_numeric(rows[c], errors="coerce")
    events = _std_events(rows, home_id, away_id)
    return Game("metrica_g3", period, np.stack([bx, by], 1), np.stack([px, py], -1), team_code,
                list(all_players), events,
                meta={"source": "metrica sample game 3 (EPTS + JSON)",
                      "home": teams[0][1], "away": teams[1][1]})


def load_game(game_id: str, raw_dir: str = RAW_DIR) -> Game:
    if game_id == "metrica_g1":
        return load_csv_game(1, raw_dir)
    if game_id == "metrica_g2":
        return load_csv_game(2, raw_dir)
    if game_id == "metrica_g3":
        return load_epts_game(raw_dir)
    raise KeyError(game_id)


def available_games(raw_dir: str = RAW_DIR):
    out = []
    for gid, rel in (("metrica_g1", "Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv"),
                     ("metrica_g2", "Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv"),
                     ("metrica_g3", "Sample_Game_3/Sample_Game_3_tracking.txt")):
        if os.path.exists(os.path.join(raw_dir, rel)):
            out.append(gid)
    return out
