"""Bench truth labels in submission format (the convention is described in the comments below).

    uv run python -m bench.truth [--games metrica_g1 ...]

Writes, per game, to artifacts/bench/<game_id>/:
    truth_first_touch.csv   switch dated at the first touch of the new team's controlling sequence
    truth_confirmed.csv     switch dated at the first controlled action of the new team
    truth.csv               = truth_first_touch.csv until the reviewer settles P16-P21 (OPEN)
    stoppages.csv           the stoppages used for D (with restart team and type)
    truth_meta.json         graded spans, halves, counts

Dead ball (D1-D5, D7, L1, L2): event stoppages closed by a logged restart, plus stoppages whose
restart was not logged when the corrected ball channel is missing on at least half their frames
(bench.events.confirm_stoppages). Owner (P1-P15): changes only on a controlled action of the
other team. Restart law (R0, P6): the restart team owns from the restart frame.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from bench import metrica_io
from bench.events import (NON_PLAY, confirm_stoppages, event_stoppages, half_end_frames,
                          is_control, is_restart, is_stop, is_touch)

ART = os.path.join(metrica_io.ROOT, "artifacts", "bench")
CONVENTIONS = ("first_touch", "confirmed")
DEFAULT_CONVENTION = "first_touch"   # provisional: P16 to P21 are OPEN (reviewer Q3 to Q5)


def _controls(row, same_frame_rows) -> bool:
    """is_control, except a CARRY that starts on the frame of a stop event of the same team (game
    3 logs a 1-frame CARRY after a touch that went straight out: a clearance out is not control,
    P10; a parried save out is not control, P11)."""
    if not is_control(row):
        return False
    if row["type"] == "CARRY":
        for other in same_frame_rows:
            if other["team"] == row["team"] and is_stop(other):
                return False
    return True


def graded_spans(game, stops, ends):
    """Half 1 graded from its kick-off (S5); half 2 from the start of its tracking, D until the
    kick-off (S4); each half ends one past its last ball event (D14)."""
    spans = []
    for k, ((h0, h1), end) in enumerate(zip(game.halves(), ends)):
        start = h0
        if k == 0:
            ko = stops[(stops["period"] == 1) & (stops["cause"] == "half_start")]
            if len(ko):
                start = int(ko["restart"].iloc[0])
        spans.append((start, end))
    # Halves are contiguous in Metrica files; keep at least one X frame between graded spans so
    # the scorer treats them as separate runs (S3).
    for k in range(len(spans) - 1):
        if spans[k][1] >= spans[k + 1][0]:
            spans[k] = (spans[k][0], spans[k + 1][0] - 1)
    return spans


def owner_switches(events: pd.DataFrame, stops: pd.DataFrame, period: int, convention: str):
    """[(frame, team)] owner changes in one half, restarts included, in frame order."""
    if convention not in CONVENTIONS:
        raise ValueError(convention)
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
    owner, floor = None, None   # floor: earliest frame a new switch may be dated at
    pending_restarts = sorted(restart_at)
    for i, row in enumerate(e):
        f0 = int(row["start_frame"])
        # Apply every restart at or before this event (R0), even with no event at that frame.
        while pending_restarts and pending_restarts[0] <= f0:
            r = pending_restarts.pop(0)
            owner, floor = restart_at[r], r
            switches.append((r, owner))
        if in_dead(f0):
            continue
        team = row["team"]
        if owner is None:
            if _controls(row, by_frame[f0]) or is_restart(row):
                owner, floor = team, f0
                switches.append((f0, team))
            continue
        if team == owner or not _controls(row, by_frame[f0]):
            continue
        # Team `team` takes control here. Date the switch.
        date = f0
        if convention == "first_touch":
            j = i - 1
            while j >= 0:
                prev = e[j]
                pf = int(prev["start_frame"])
                if pf < floor or in_dead(pf):
                    break
                if prev["team"] == owner and is_touch(prev):
                    break
                if prev["team"] == team and is_touch(prev):
                    date = pf
                j -= 1
        date = max(date, floor)
        owner, floor = team, date
        switches.append((date, team))
    while pending_restarts:
        r = pending_restarts.pop(0)
        switches.append((r, restart_at[r]))
    switches.sort(key=lambda s: s[0])
    return switches


def derive_labels(game, convention=DEFAULT_CONVENTION, stops=None):
    """(labels array of 'H'/'A'/'D'/'X', stops DataFrame, graded spans)."""
    halves = game.halves()
    nan = np.isnan(game.ball[:, 0])
    if stops is None:
        stops = confirm_stoppages(event_stoppages(game.events, halves), nan)
    ends = half_end_frames(game.events, halves)
    spans = graded_spans(game, stops, ends)
    labels = np.full(game.n_frames, "X", dtype="<U1")
    for k, (a, b) in enumerate(spans):
        period = k + 1
        sw = owner_switches(game.events, stops, period, convention)
        seg = np.full(b - a, "D", dtype="<U1")   # before the first owner: D (pre kick-off, S4)
        for idx, (f, team) in enumerate(sw):
            stop = sw[idx + 1][0] if idx + 1 < len(sw) else b
            lo, hi = max(f, a), min(stop, b)
            if hi > lo:
                seg[lo - a:hi - a] = team
        for s in stops[stops["period"] == period].itertuples():
            lo, hi = max(s.start, a), min(s.restart, b)
            if hi > lo:
                seg[lo - a:hi - a] = "D"
        labels[a:b] = seg
    return labels, stops, spans


def write_labels(path, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"frame": np.arange(len(labels)), "label": labels}).to_csv(path, index=False)


# -- summary statistics --------------------------------------------------------------------------

def label_stats(labels, fps=25):
    """Possessions per minute, share of D, stoppages, possession and stoppage lengths."""
    labels = np.asarray(labels)
    graded = labels != "X"
    runs_out = []
    # segments inside graded runs
    edges = np.diff(np.concatenate(([0], graded.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        w = labels[a:b]
        br = np.flatnonzero(w[1:] != w[:-1]) + 1
        bounds = np.concatenate(([0], br, [len(w)]))
        for i in range(len(bounds) - 1):
            runs_out.append((a + bounds[i], a + bounds[i + 1], w[bounds[i]]))
    poss = np.array([(b - s) / fps for s, b, l in runs_out if l in ("H", "A")])
    stops = np.array([(b - s) / fps for s, b, l in runs_out if l == "D"])
    minutes = graded.sum() / fps / 60.0
    q = lambda arr, p: float(np.percentile(arr, p)) if len(arr) else None
    return {
        "graded_minutes": round(float(minutes), 2),
        "n_possessions": int(len(poss)),
        "possessions_per_min": round(len(poss) / minutes, 3) if minutes else None,
        "share_D": round(float((labels[graded] == "D").mean()), 4) if graded.any() else None,
        "n_stoppages": int(len(stops)),
        "share_poss_under_1_5s": round(float((poss < 1.5).mean()), 4) if len(poss) else None,
        "share_poss_under_4s": round(float((poss < 4.0).mean()), 4) if len(poss) else None,
        "poss_len_s": {"min": q(poss, 0), "p10": q(poss, 10), "p25": q(poss, 25),
                       "median": q(poss, 50), "p75": q(poss, 75), "p90": q(poss, 90),
                       "max": q(poss, 100), "mean": float(poss.mean()) if len(poss) else None},
        "shortest_poss_s": sorted(poss.tolist())[:5],
        "poss_len_hist_s": {"edges": [0, 0.5, 1, 1.5, 2, 4, 8, 15, 30, 60, "inf"],
                            "counts": np.histogram(poss, [0, 0.5, 1, 1.5, 2, 4, 8, 15, 30, 60,
                                                          1e9])[0].tolist()},
        "stoppage_len_s": {"min": q(stops, 0), "p25": q(stops, 25), "median": q(stops, 50),
                           "p75": q(stops, 75), "max": q(stops, 100)},
        "shortest_stoppages_s": sorted(stops.tolist())[:5],
        "home_share_alive": round(float((labels == "H").sum() / max(1, np.isin(labels, ["H", "A"]).sum())), 4),
    }


def build(game_id):
    game = metrica_io.load_game(game_id)
    out_dir = os.path.join(ART, game_id)
    stats, stops, spans = {}, None, None
    for conv in CONVENTIONS:
        labels, stops, spans = derive_labels(game, conv, stops)
        write_labels(os.path.join(out_dir, "truth_{}.csv".format(conv)), labels)
        if conv == DEFAULT_CONVENTION:
            write_labels(os.path.join(out_dir, "truth.csv"), labels)
        stats[conv] = label_stats(labels)
    stops.to_csv(os.path.join(out_dir, "stoppages.csv"), index=False)
    meta = {"game_id": game_id, "n_frames": game.n_frames, "fps": metrica_io.FPS,
            "halves": [list(h) for h in game.halves()], "graded_spans": [list(s) for s in spans],
            "pitch": list(game.pitch), "truth_default_convention": DEFAULT_CONVENTION,
            "n_stoppages_used": int(len(stops)),
            "restart_types": stops["restart_type"].value_counts().to_dict(),
            "stoppage_confirmed_by": stops["confirmed_by"].value_counts().to_dict(),
            "stats": stats}
    with open(os.path.join(out_dir, "truth_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def public_gt_stats():
    """Same statistics on the public 5-minute window, as a convention check only."""
    path = os.path.join(metrica_io.ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    return label_stats(pd.read_csv(path)["label"].astype(str).values)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--games", nargs="*", default=None)
    args = ap.parse_args(argv)
    rows = []
    for gid in args.games or metrica_io.available_games():
        meta = build(gid)
        for conv, st in meta["stats"].items():
            rows.append((gid + ":" + conv, st))
    rows.append(("PUBLIC 5 min (check only)", public_gt_stats()))
    print("{:<32}{:>7}{:>8}{:>7}{:>7}{:>8}{:>8}{:>8}{:>9}  {}".format(
        "labels", "poss", "poss/m", "D", "stops", "<1.5s", "<4s", "medP s", "medStop", "shortest"))
    for name, st in rows:
        print("{:<32}{:>7}{:>8.2f}{:>7.3f}{:>7}{:>8.3f}{:>8.3f}{:>8.2f}{:>9.2f}  {}".format(
            name, st["n_possessions"], st["possessions_per_min"], st["share_D"],
            st["n_stoppages"], st["share_poss_under_1_5s"], st["share_poss_under_4s"],
            st["poss_len_s"]["median"], st["stoppage_len_s"]["median"],
            [round(x, 2) for x in st["shortest_poss_s"][:4]]))


if __name__ == "__main__":
    main()
