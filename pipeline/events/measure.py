"""Bench measurement of the events against a Metrica event file (bench games only).

    uv run python -m pipeline.events.measure --root artifacts/bench/dev_g3_clean \
        --events bench_data/raw/metrica/Sample_Game_3/Sample_Game_3_events.json

Reference sets (Metrica frames are 1-based: video frame = Metrica frame - 1):

- exits: every BALL OUT event, plus SHOT events whose subtypes include OUT or GOAL. Reference
  frame = the event's end frame (in the sample games: the last tracked frame, ball over the
  line). A detected exit matches a reference within EXIT_TOL frames (one to one, nearest first).
  The line is checked from the ball position at the reference frame.
- touches: PASS start (passer), PASS end of completed passes (receiver), SHOT start,
  RECOVERY start, BALL LOST start, SET PIECE start; deduplicated within 3 frames; only where
  the ball is seen on the reference frame. A detected touch matches within TOUCH_TOL frames.
  Precision is against the same set, so dribble touches inside a CARRY (real touches that the
  event data does not list) count as false positives: precision is a lower bound.

Refuses any root under artifacts/match: never run this on the match.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from pipeline.io.roots import require_bench_root

EXIT_TOL = 25
TOUCH_TOLS = (5, 10)
TEAM_IDS = None


def _team_index(ev):
    """Metrica team ids -> 0 (Home) / 1 (Away) from the metadata order in the event file."""
    names = []
    for e in ev:
        t = (e.get("team") or {}).get("id")
        if t and t not in names:
            names.append(t)
    return names


def load_reference(events_path, home_team_id=None):
    with open(events_path) as handle:
        data = json.load(handle)
    ev = data["data"] if isinstance(data, dict) else data
    teams = _team_index(ev)
    home = home_team_id or teams[0]
    team_of = lambda e: 0 if (e.get("team") or {}).get("id") == home else 1   # noqa: E731

    def subs(e):
        s = e.get("subtypes")
        if s is None:
            return set()
        if isinstance(s, dict):
            return {s["name"]}
        return {x["name"] for x in s}

    exits, touches = [], []
    for e in ev:
        kind = e["type"]["name"]
        st, en = e["start"]["frame"] - 1, e["end"]["frame"] - 1
        if kind == "BALL OUT" or (kind == "SHOT" and subs(e) & {"OUT", "GOAL"}):
            exits.append((en, team_of(e)))
        if kind in ("PASS", "SHOT", "RECOVERY", "BALL LOST", "SET PIECE"):
            touches.append((st, team_of(e), kind))
        if kind == "PASS" and e.get("to"):
            touches.append((en, team_of(e), "RECEIVE"))
    exits.sort()
    touches.sort()
    dedup = []
    for t in touches:
        if dedup and t[0] - dedup[-1][0] <= 3 and t[1] == dedup[-1][1]:
            continue
        dedup.append(t)
    return exits, dedup, home


def match_one_to_one(det, ref, tol):
    """Greedy nearest matching. Returns list of (i_det, i_ref)."""
    pairs = []
    for i, f in enumerate(det):
        for j, g in enumerate(ref):
            if abs(f - g) <= tol:
                pairs.append((abs(f - g), i, j))
    pairs.sort()
    used_d, used_r, out = set(), set(), []
    for _, i, j in pairs:
        if i in used_d or j in used_r:
            continue
        used_d.add(i)
        used_r.add(j)
        out.append((i, j))
    return out


def measure(root, events_path, home_team_id=None):
    require_bench_root(root)
    ev = pd.read_parquet(os.path.join(root, "events.parquet"))
    ball = pd.read_parquet(os.path.join(root, "ball.parquet"))
    seen = ball["seen"].to_numpy()
    bx, by = ball["x_m"].to_numpy(float), ball["y_m"].to_numpy(float)
    ref_exits, ref_touch, _home = load_reference(events_path, home_team_id)
    out = {}

    det = ev[ev["kind"] == "exit"].reset_index(drop=True)
    rf = [f for f, _ in ref_exits]
    pairs = match_one_to_one(det["frame"].tolist(), rf, EXIT_TOL)
    out["exit_ref"] = len(rf)
    out["exit_detected"] = len(det)
    out["exit_precision"] = round(len(pairs) / max(len(det), 1), 3)
    out["exit_recall"] = round(len(pairs) / max(len(rf), 1), 3)
    offs = [int(det["frame"][i] - rf[j]) for i, j in pairs]
    out["exit_offset_median_frames"] = float(np.median(offs)) if offs else None
    out["exit_by_detail_matched"] = {d: int(sum(det["detail"][i] == d for i, _ in pairs))
                                     for d in ("persist", "vanish", "reappear", "extrapolated")}
    out["exit_by_detail_all"] = {d: int((det["detail"] == d).sum())
                                 for d in ("persist", "vanish", "reappear", "extrapolated")}
    # Line agreement against the reference ball position (nearest line to the out point).
    L, W = 105.0, 68.0
    agree = 0
    lt_agree, lt_known = 0, 0
    for i, j in pairs:
        g = rf[j]
        if np.isfinite(bx[g]):
            dists = {"goal_x0": abs(bx[g]), "goal_x1": abs(bx[g] - L),
                     "touch_y0": abs(by[g]), "touch_y1": abs(by[g] - W)}
            agree += int(min(dists, key=dists.get) == det["line"][i])
        if det["last_touch_team"][i] >= 0:
            lt_known += 1
            lt_agree += int(det["last_touch_team"][i] == ref_exits[j][1])
    out["exit_line_agreement"] = round(agree / max(len(pairs), 1), 3)
    # Metrica's BALL OUT team is the team that put it out (last touch).
    out["exit_last_touch_known"] = lt_known
    out["exit_last_touch_agreement"] = round(lt_agree / max(lt_known, 1), 3)
    out["exit_missed_ref_frames"] = [rf[j] for j in range(len(rf))
                                     if j not in {jj for _, jj in pairs}][:20]
    out["exit_false_frames"] = [int(det["frame"][i]) for i in range(len(det))
                                if i not in {ii for ii, _ in pairs}][:20]

    tdet = ev[ev["kind"] == "touch"].reset_index(drop=True)
    refs = [(f, t, k) for f, t, k in ref_touch if 0 <= f < len(seen) and seen[f]]
    out["touch_ref_seen"] = len(refs)
    out["touch_detected"] = len(tdet)
    for tol in TOUCH_TOLS:
        pairs = match_one_to_one(tdet["frame"].tolist(), [r[0] for r in refs], tol)
        team_ok = sum(int(tdet["team"][i] == refs[j][1]) for i, j in pairs)
        out["touch_recall_tol{}".format(tol)] = round(len(pairs) / max(len(refs), 1), 3)
        out["touch_precision_tol{}".format(tol)] = round(len(pairs) / max(len(tdet), 1), 3)
        out["touch_team_agreement_tol{}".format(tol)] = round(team_ok / max(len(pairs), 1), 3)
        by_kind = {}
        for kind in ("PASS", "RECEIVE", "RECOVERY", "BALL LOST", "SHOT", "SET PIECE"):
            idx = [j for j, r in enumerate(refs) if r[2] == kind]
            hit = {j for _, j in pairs}
            by_kind[kind] = round(sum(j in hit for j in idx) / max(len(idx), 1), 3)
        out["touch_recall_by_kind_tol{}".format(tol)] = by_kind
    return out


CONTROL_TOL = 10
LENGTH_BINS_S = (1.5, 4.0)


def possession_starts(labels, fps):
    """[(frame, team, length_s, after)] for every H/A run start in a label array.

    ``after`` is "D" when the previous label was D or X (a restart), else "switch".
    """
    out = []
    lab = np.asarray(labels)
    n = len(lab)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and lab[j + 1] == lab[i]:
            j += 1
        if lab[i] in ("H", "A"):
            prev = lab[i - 1] if i > 0 else "X"
            out.append((i, 0 if lab[i] == "H" else 1, (j - i + 1) / fps,
                        "switch" if prev in ("H", "A") else "D"))
        i = j + 1
    return out


def switch_controls(ev, live=True):
    """Control events that start a new team's possession.

    live=True: the team differs from the previous control's and no exit lies between them (a
    live-ball switch). live=False: the first control after an exit (a restart's possession).
    """
    ctrl = ev[ev["kind"] == "control"].sort_values("frame")
    exits = np.sort(ev.loc[ev["kind"] == "exit", "frame"].to_numpy())
    out, prev_team, prev_frame = [], None, -1
    for _, r in ctrl.iterrows():
        exit_between = bool(np.any((exits > prev_frame) & (exits < r["frame"])))
        if live and prev_team is not None and r["team"] != prev_team and not exit_between:
            out.append((int(r["frame"]), int(r["team"])))
        if not live and exit_between:
            out.append((int(r["frame"]), int(r["team"])))
        prev_team, prev_frame = r["team"], r["frame"]
    return out


def measure_control(root, truth_csv, fps=25.0, tol=CONTROL_TOL, ev=None):
    """Precision and recall of live-switch controls against true live possession starts (H to A
    or A to H, team must agree, within tol frames), recall split by possession length. Starts
    after a stoppage (restarts, covered by the restart law) are reported apart."""
    ev = pd.read_parquet(os.path.join(root, "events.parquet")) if ev is None else ev
    truth = pd.read_csv(truth_csv)
    all_starts = possession_starts(truth["label"].to_numpy(), fps)
    starts = [s for s in all_starts if s[3] == "switch"]
    restarts = [s for s in all_starts if s[3] == "D"]
    det = switch_controls(ev)
    pairs = []
    for i, (f, t) in enumerate(det):
        for j, (g, tt, _l, _a) in enumerate(starts):
            if t == tt and abs(f - g) <= tol:
                pairs.append((abs(f - g), i, j))
    pairs.sort()
    ud, ur, matched = set(), set(), []
    for _, i, j in pairs:
        if i in ud or j in ur:
            continue
        ud.add(i)
        ur.add(j)
        matched.append((i, j))
    hit = {j for _, j in matched}
    out = {"control_all": int((ev["kind"] == "control").sum()),
           "control_switch_detected": len(det), "live_switch_starts_true": len(starts),
           "control_switch_precision": round(len(matched) / max(len(det), 1), 3),
           "control_switch_recall": round(len(matched) / max(len(starts), 1), 3)}
    edges = (0.0,) + LENGTH_BINS_S + (1e9,)
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = [j for j, s in enumerate(starts) if lo <= s[2] < hi]
        key = "recall_len_{}_{}s".format(lo, hi if hi < 1e9 else "inf")
        out[key] = round(sum(j in hit for j in idx) / max(len(idx), 1), 3)
        out[key.replace("recall", "n")] = len(idx)
    rdet = switch_controls(ev, live=False)
    out["restart_starts_true"] = len(restarts)
    out["restart_first_control_team_agreement"] = round(float(np.mean([
        any(t == s[1] and 0 <= f - s[0] <= 10 * fps for f, t in rdet) for s in restarts])), 3) \
        if restarts else None
    offs = [det[i][0] - starts[j][0] for i, j in matched]
    out["control_offset_median_frames"] = float(np.median(offs)) if offs else None
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--truth", default=None, help="bench truth.csv for the control measurement")
    ap.add_argument("--home-team-id", default=None)
    args = ap.parse_args(argv)
    res = measure(args.root, args.events, args.home_team_id)
    if args.truth:
        res.update(measure_control(args.root, args.truth))
    for k, v in res.items():
        print("{}: {}".format(k, json.dumps(v)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
