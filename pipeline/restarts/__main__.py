"""Restarts stage: ``python -m pipeline.restarts match|bench`` (CONTRACTS.md).

    uv run python -m pipeline.restarts match           # artifacts/match/, aggregates only
    make match STAGE=restarts                          # the same
    uv run python -m pipeline.restarts bench [--game dev_g3_restarts ...] [--write]

Reads players.parquet, meta.json and ball_raw.parquet from the root; writes
restarts.parquet (one row per kick-off found; other restart types come later) and
features_dead.parquet (frame, half, form_kick_off, mask_pre_kickoff, p_restart_kick_off).

Match checks (printed as aggregates through summary() only; rulebook "Output checks"):
- kick-offs per half, and how each half-start kick-off was dated (break / ball / assumed);
- S4 CHECK: |detected - published| second-half kick-off within KICKOFF_CHECK_TOL frames
  (pass/fail only; no frame offset is printed); the same for the first half;
- S5/S6: the first-half kicking team implied by the law agrees with the first graded label of
  the public ground truth (evaluation/gt, read for checking only, never as an input);
- goal kick-offs detected (compare the count with the four GOALS tags in DATA.md; the tag file
  itself is not read here).
The published kick-off frames in games.py (read through games.py) are used for the CHECK, and
also as the fallback: when no kick-off formation is detected at a half start, the published
kick-off frame is the restart there, and frames before it are masked as dead ball.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from pipeline.io.sources import ARTIFACTS, REPO, match_root
from pipeline.report import match_entry, quiet_warnings, summary
from pipeline.restarts import kickoff, stage

KICKOFF_CHECK_TOL = 25       # frames: the rulebook's "within a second" output check
GOAL_TAGS_IN_FILE = 4       # DATA.md: four GOALS tags in the Smart Tagging file
BENCH = os.path.join(ARTIFACTS, "bench")
PUBLIC_GT = os.path.join(REPO, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
FEATURE_FILE = "features_dead.parquet"
RESTART_FILE = "restarts.parquet"
OWN_ROOT_TAG = "restarts"    # bench roots this stage may write into (e.g. dev_g3_restarts)


def load_root(root):
    players = pd.read_parquet(os.path.join(root, "players.parquet"))
    with open(os.path.join(root, "meta.json")) as handle:
        meta = json.load(handle)
    ball_path = os.path.join(root, "ball_raw.parquet")
    ball = pd.read_parquet(ball_path, columns=["frame", "x_m", "y_m"]) \
        if os.path.isfile(ball_path) else None
    return players, meta, ball


def write_outputs(root, kickoffs, feats):
    kickoffs.to_parquet(os.path.join(root, RESTART_FILE), index=False)
    feats.to_parquet(os.path.join(root, FEATURE_FILE), index=False)


def stage_aggregates(kickoffs, n_halves):
    """Aggregates safe to print for any game: counts and categories, no frames."""
    ko_rows = kickoffs[kickoffs["restart_type"] == "kick_off"]
    per_half = {str(h): int((ko_rows["half"] == h).sum()) for h in range(1, n_halves + 1)}
    starts = kickoffs[kickoffs["start_kind"] == "half_start"]
    start_source = {str(int(h)): str(s) for h, s in zip(starts["half"], starts["evidence"])}
    goals = ko_rows[ko_rows["start_kind"] == "goal"]
    h1 = starts[starts["half"] == 1]
    h2 = starts[starts["half"] == 2]
    consistent = None
    if len(h1) and len(h2) and h1["kick_team_llr"].iloc[0] != 0 and h2["kick_team_llr"].iloc[0] != 0:
        # S6 consistency: the two half-start formation votes name different teams
        consistent = bool(np.sign(h1["kick_team_llr"].iloc[0]) != np.sign(h2["kick_team_llr"].iloc[0]))
    return {"kickoffs_per_half": per_half,
            "half_start_dated_by": start_source,
            "goal_kickoffs_detected": int(len(goals)),
            "goal_kickoffs_by_source": {str(k): int(v) for k, v in
                                        goals["evidence"].value_counts().items()},
            "half_start_votes_consistent_with_s6": consistent,
            "law_p_home_kicks_off_h1": float(h1["p_home_restart"].iloc[0]) if len(h1) else None}


def stoppage_aggregates(rows, feats, inputs):
    """Counts and shares only (no frames): stoppages per hour by source and type, share of graded
    frames inside a candidate, share of long ball gaps that coincide with a candidate, median
    stoppage length, share of candidates with p_home_restart = 0.5."""
    from pipeline.restarts.stoppages import GAP_MIN_FRAMES
    meta = inputs["meta"]
    n = int(meta["n_frames"])
    graded = np.zeros(n, dtype=bool)
    for a, b in meta["halves"]:
        graded[int(a):int(b)] = True
    hours = graded.sum() / 25.0 / 3600.0
    st = rows[rows["start_kind"] != "half_start"]
    st = st[st["start_frame"] >= 0]
    out = {"stoppages_total": int(len(st)),
           "stoppages_per_hour": round(len(st) / hours, 2) if hours else None,
           "stoppages_per_hour_by_source": {str(k): round(v / hours, 2) for k, v in
                                            st["start_kind"].value_counts().items()},
           "stoppages_per_hour_by_type": {str(k): round(v / hours, 2) for k, v in
                                          st["restart_type"].value_counts().items()}}
    inside = np.zeros(n, dtype=bool)
    for s0, e0 in zip(st["start_frame"], st["restart_frame"]):
        inside[int(s0):int(e0)] = True
    mask = feats["mask_pre_kickoff"].to_numpy() if "mask_pre_kickoff" in feats else np.zeros(n, bool)
    out["share_graded_in_stoppage_candidate"] = round(float(inside[graded].mean()), 4)
    out["share_graded_dead_incl_pre_kickoff_mask"] = round(float((inside | mask)[graded].mean()), 4)
    lengths = (st["restart_frame"] - st["start_frame"]).to_numpy() / 25.0
    out["median_stoppage_s"] = round(float(np.median(lengths)), 2) if len(lengths) else None
    out["stoppage_len_s_hist_edges"] = [0, 2, 5, 10, 20, 40, 60, 120, 240]
    out["stoppage_len_s_hist"] = np.histogram(lengths, bins=[0, 2, 5, 10, 20, 40, 60, 120, 240])[0].tolist()
    out["share_candidates_p_home_half"] = round(float(np.mean(st["p_home_restart"] == 0.5)), 4) \
        if len(st) else None
    ball = inputs.get("ball")
    if ball is not None:
        seen = (ball["seen"].astype(bool) | ball["filled"].astype(bool)).to_numpy()
        d = np.diff(np.r_[0, (~seen & graded).astype(np.int8), 0])
        gaps = [(a, b) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))
                if b - a >= GAP_MIN_FRAMES]
        hit = [bool(inside[a:b].any() or mask[a:b].any()) for a, b in gaps]
        out["long_ball_gaps"] = len(gaps)
        out["share_long_gaps_with_candidate"] = round(float(np.mean(hit)), 4) if gaps else None
    return out


def _first_public_label():
    """First graded label of the public ground truth ('H', 'A' or 'D'), or None."""
    if not os.path.isfile(PUBLIC_GT):
        return None
    gt = pd.read_csv(PUBLIC_GT, usecols=["label"])
    graded = gt["label"][gt["label"] != "X"]
    return str(graded.iloc[0]) if len(graded) else None


def match_checks(kickoffs, game):
    """Pass/fail checks against the published anchors and the public first label."""
    import games  # noqa: E402  (repo root is on sys.path through pipeline.io.sources)
    out = {}
    starts = kickoffs[kickoffs["start_kind"] == "half_start"]
    detected = kickoffs[(kickoffs["restart_type"] == "kick_off")
                        & kickoffs["evidence"].astype(str).str.startswith("formation")]
    for half, key in ((1, "kickoff"), (2, "kickoff_2nd")):
        tag = "h{}_".format(half)
        row = starts[starts["half"] == half]
        anchor = int(games.anchor_video_frame(games.GAMES[game][key]))
        # the half-start row; when it IS the published anchor the check does not apply
        if len(row) and row["evidence"].iloc[0] == "published_anchor":
            out[tag + "kickoff_anchor_pass"] = "not applicable: anchor used"
        elif len(row):
            err = abs(int(row["restart_frame"].iloc[0]) - anchor)
            out[tag + "kickoff_anchor_pass"] = bool(err <= KICKOFF_CHECK_TOL)
        else:
            out[tag + "kickoff_anchor_pass"] = False
        # any kick-off found from the formation (whatever it was labelled) near the anchor
        same = detected[detected["half"] == half]
        if len(same):
            near = int(np.min(np.abs(same["restart_frame"].to_numpy() - anchor)))
            out[tag + "any_detected_kickoff_within_tol"] = bool(near <= KICKOFF_CHECK_TOL)
        else:
            out[tag + "any_detected_kickoff_within_tol"] = False
    out["kickoff_anchor_tolerance_frames"] = KICKOFF_CHECK_TOL
    first = _first_public_label()
    law_team = int(starts["law_kick_team_h1"].iloc[0]) if len(starts) else -1
    implied = {0: "H", 1: "A"}.get(law_team)
    out["public_first_label_is_team"] = first in ("H", "A")
    out["h1_kick_team_determined"] = implied is not None
    out["h1_kick_team_implies_public_first_label"] = (implied == first) \
        if (first in ("H", "A") and implied is not None) else None
    out["goal_tags_in_file"] = GOAL_TAGS_IN_FILE
    return out


def match_anchors(game):
    """{half: kick-off video frame} published by the organisers, read through games.py only
    (lead decision 2026-11-13). Used by the detector only when no half-start formation is
    detected; otherwise only as the CHECK."""
    import games  # noqa: E402
    entry = games.GAMES[game]
    return {1: int(games.anchor_video_frame(entry["kickoff"])),
            2: int(games.anchor_video_frame(entry["kickoff_2nd"]))}


@match_entry
def run_match():
    quiet_warnings()
    import games  # noqa: E402
    root = match_root()
    inputs = stage.load_inputs(root)
    meta = inputs["meta"]
    funnel = {}
    kickoffs, feats = stage.run(inputs, anchors=match_anchors(games.GAME), funnel=funnel)
    write_outputs(root, kickoffs, feats)
    agg = stage_aggregates(kickoffs, len(meta["halves"]))
    # on the match, per-half counts are folded into whole-match totals (rules audit, 15 Nov):
    # kick-offs, the kick-off candidate funnel and the pre-kick-off mask length
    agg["kickoffs_total"] = int(sum(agg.pop("kickoffs_per_half").values()))
    funnel_total = {}
    for counts in funnel.get("kickoff", {}).values():
        for key, value in counts.items():
            funnel_total[str(key)] = funnel_total.get(str(key), 0) + int(value)
    agg["kickoff_candidate_funnel_total"] = funnel_total
    agg["stoppage_candidates_by_source_before_merge"] = funnel.get("stoppage_sources", {})
    agg.update(stoppage_aggregates(kickoffs, feats, inputs))
    agg["mask_pre_kickoff_frames_total"] = int(np.sum(feats["mask_pre_kickoff"].to_numpy()))
    agg.update(match_checks(kickoffs, games.GAME))
    summary(_name="restarts", **agg)
    return 0


def bench_roots(only=None):
    out = []
    if not os.path.isdir(BENCH):
        return out
    for name in sorted(os.listdir(BENCH)):
        root = os.path.join(BENCH, name)
        if only and name not in only:
            continue
        if os.path.isfile(os.path.join(root, "players.parquet")) and \
                os.path.isfile(os.path.join(root, "meta.json")):
            out.append((name, root))
    return out


def run_bench(only=None, write=False):
    roots = bench_roots(only)
    for name, root in roots:
        inputs = stage.load_inputs(root)
        meta = inputs["meta"]
        kickoffs, feats = stage.run(inputs)
        # Until the lead routes it, write only into roots this stage owns (name contains
        # OWN_ROOT_TAG); bench-builder's roots are read, never written.
        can_write = write and OWN_ROOT_TAG in name
        if can_write:
            write_outputs(root, kickoffs, feats)
        agg = stage_aggregates(kickoffs, len(meta["halves"]))
        agg.update(stoppage_aggregates(kickoffs, feats, inputs))
        summary(_name="restarts.bench.{}".format(name), written=bool(can_write), **agg)
    summary(_name="restarts.bench", bench_roots_found=len(roots))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Restarts stage (CONTRACTS.md)")
    parser.add_argument("target", choices=("match", "bench"))
    parser.add_argument("--game", nargs="*", help="bench roots under artifacts/bench/")
    parser.add_argument("--write", action="store_true",
                        help="bench: write restarts.parquet and features_dead.parquet into "
                             "each root (off by default: most bench roots belong to bench-builder)")
    args = parser.parse_args(argv)
    if args.target == "match":
        if args.game or args.write:
            parser.error("the match takes no options")
        return run_match()
    return run_bench(args.game, args.write)


if __name__ == "__main__":
    sys.exit(main())
