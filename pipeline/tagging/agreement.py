"""Per-code agreement of the Smart Tagging votes with a reference owner (rulebook G10).

    python -m pipeline.tagging match --agreement --ref-version v6s    # decoded labels as reference
    python -m pipeline.tagging match --agreement --ref pseudo         # pseudo-labels (T-37a)

The reference owner of a frame is -1 (none), 0 (H), 1 (A) or 2 (D).

- ``--ref-version V``: the decoded labels ``<root>/labels/V.csv``. Its **core** frames are H or A
  frames at least CORE_S seconds from any label change in that file; the other H or A frames are
  **non-core** (reported separately: that is agreement with our own uncertain labels).
- ``--ref pseudo``: ``pipeline.models.pseudo.pseudo_labels(root)`` (T-37a), imported only when
  asked for. Every pseudo-labelled H or A frame is core; there are no non-core frames.

Only graded frames count. What is computed, all as counts and shares, never per instance:

- per voting code (and for the unweighted ``tag_vote``): tags, frames covered, voting frames,
  core frames covered and voting, agreement on core frames, split by ball seen / missing
  (``ball.parquet``), agreement on non-core frames, share of voting frames the reference calls D;
- SET PIECES (G5, L5, R14): share of windows holding a restart frame of ``restarts.parquet``
  (strictly inside, and within SET_PIECE_TOL_S of the window, G8), agreement of the tag team with
  ``p_home_restart`` and of the tag type with ``restart_type``; share of windows holding a
  reference restart (D to H or A) and agreement of the tag team with the reference's restart team;
  share of detected restarts inside some window (G7: fewer than half are expected);
- the two TRANSITION codes (G4): share of hints whose 2 s window holds a reference change to the
  hinted team, the share with such a change within CHANGE_REACH_S, and the chance rates of both
  windows on graded frames (how often a window placed anywhere would hit);
- GOALS (G6): how many tags are followed by a long detected stoppage, by a long reference D run,
  and by a detected goal kick-off, and whether that kick-off's team is the conceding one (R13).

P-20 gate (T-12b), computed from the per-code rows, nothing applied to any output:

- mirror codes: a code whose per-frame vote equals an earlier code's (BUILD UP and DEFENDING IN
  ATTACKING THIRD of the other team, ...) is counted once, under the earlier (attacking) code;
- kept codes: core agreement A_k >= A_MIN (and the A_MIN_SENSITIVITY thresholds), weight
  w_k = min(W_CAP, LAMBDA * logit(A_k)); for the weighted vote sign(sum w_k vote_k): coverage C
  of the reference's live frames, core and non-core agreement A, gate value C * (A - GATE_BASE);
- edge lag of the kept codes at A_MIN: tag start edge minus the nearest reference change INTO the
  voting team within CHANGE_REACH_S, tag end edge minus the nearest change OUT of it (frames,
  positive = tag later); counts, and quartiles only from LAG_MIN_MATCHED matched edges up.
- P-20 rule 3 per code (T-12d): at each threshold, a kept code stays only if its own median
  start-edge lag is measurable (LAG_MIN_MATCHED matched edges) and at most LAG_GATE_FRAMES; the
  gate row (rule 2) is recomputed on the survivors (``lag_filtered_kept``, ``gate_after_lag``).
  The pooled lag rows above stay for comparability; the per-code rule is the one in force.

tags.parquet stays unweighted; T-12c applies weights only if the gate passes.
"""
import importlib
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline.tagging import parse, stage, votes

# Params (named constants; none refers to a frame, a time or an incident of any match)
CORE_S = 2.0              # core frames: at least this far from any label change of the reference
SET_PIECE_TOL_S = 1.0     # G8: tag edges are accurate to about a second
CHANGE_REACH_S = 5.0      # transition hint counted "near" if a change to the team is within +-5 s
LONG_STOPPAGE_S = 30.0    # D4: a goal stoppage usually lasts 45 to 90 s; 30 s leaves margin
GOAL_FOLLOW_S = 30.0      # the long stoppage must start or be under way within this after the tag
GOAL_KICKOFF_MAX_S = stage.GOAL_KICKOFF_MAX_S   # a goal kick-off this soon after the tag follows it
# P-20 gate (T-12b): set on the bench simulation, not on the match
A_MIN = 0.75                        # keep a code when its core agreement is at least this
A_MIN_SENSITIVITY = (0.80, 0.85)    # the same table at stricter thresholds
LAMBDA = 0.5                        # w_k = min(W_CAP, LAMBDA * logit(A_k))
W_CAP = 1.1
GATE_BASE = 0.70                    # gate value C * (A - GATE_BASE)
GATE_MIN = 0.10                     # wire in only when the gate value is at least this
LAG_GATE_FRAMES = 15                # median start-edge lag must be at most this
LAG_MIN_MATCHED = 5                 # fewer matched edges: counts only, no quantiles

RESTARTS_FILE = "restarts.parquet"
BALL_FILE = "ball.parquet"
LABEL_CODES = {"H": 0, "A": 1, "D": 2}
PSEUDO = "pseudo"
_VERSION_NAME = re.compile(r"^[A-Za-z0-9_.\-]{1,40}$")
TEAM_DEFINED = ("throw_in", "corner", "goal_kick", "free_kick", "kick_off")


class ReferenceUnavailable(RuntimeError):
    """The requested reference cannot be built (missing file or module). Message names no value."""


@dataclass
class Reference:
    name: str
    owner: np.ndarray       # int8 (n,): -1 none, 0 H, 1 A, 2 D
    core: np.ndarray        # bool (n,): confident H or A frames, graded only
    noncore: np.ndarray     # bool (n,): other H or A frames, graded only


# ---------------------------------------------------------------- references

def read_label_codes(path, n_frames):
    """int8 (n,) from a labels CSV (frame, label): H 0, A 1, D 2, X or anything else -1."""
    table = pd.read_csv(path, dtype={"label": str})
    if list(table.columns[:2]) != ["frame", "label"]:
        raise ValueError("labels file must have columns frame,label")
    frames = table["frame"].to_numpy()
    if len(frames) != n_frames or not np.array_equal(frames, np.arange(n_frames)):
        raise ValueError("labels file frames are not 0..n-1 for this root")
    codes = table["label"].fillna("").astype(str).str.strip().str.upper().map(LABEL_CODES)
    return codes.fillna(-1).to_numpy().astype(np.int8)


def core_mask(owner, k):
    """bool: H or A frames with no change of ``owner`` within k frames on either side.

    The edges of the array are not changes. k = 0 makes every H or A frame core.
    """
    n = len(owner)
    out = np.zeros(n, dtype=bool)
    if n == 0:
        return out
    cuts = np.flatnonzero(np.diff(owner) != 0) + 1
    starts = np.r_[0, cuts]
    stops = np.r_[cuts, n]
    for s, e in zip(starts.tolist(), stops.tolist()):
        if owner[s] not in (0, 1):
            continue
        a = s if s == 0 else s + k
        b = e if e == n else e - k
        if b > a:
            out[a:b] = True
    return out


def version_reference(root, version, n_frames, fps, graded):
    if not _VERSION_NAME.match(version or ""):
        raise ReferenceUnavailable("version name must be letters, digits, '_', '.', '-'")
    path = os.path.join(root, "labels", "{}.csv".format(version))
    if not os.path.isfile(path):
        raise ReferenceUnavailable("labels/{}.csv not found in the root".format(version))
    owner = read_label_codes(path, n_frames)
    live = np.isin(owner, (0, 1)) & graded
    core = core_mask(owner, int(round(CORE_S * fps))) & live
    return Reference(name=version, owner=owner, core=core, noncore=live & ~core)


def pseudo_reference(root, n_frames, graded, pseudo_fn=None):
    """Pseudo-labels of T-37a as the reference: every pseudo-labelled H or A frame is core."""
    if pseudo_fn is None:
        try:
            module = importlib.import_module("pipeline.models.pseudo")
        except ModuleNotFoundError as error:
            if error.name not in ("pipeline.models.pseudo", "pipeline.models"):
                raise
            raise ReferenceUnavailable(
                "--ref pseudo needs pipeline.models.pseudo (T-37a), not in this checkout")
        pseudo_fn = getattr(module, "pseudo_labels", None)
        if pseudo_fn is None:
            raise ReferenceUnavailable("pipeline.models.pseudo has no pseudo_labels(root)")
    owner = np.asarray(pseudo_fn(root))
    if owner.shape != (n_frames,):
        raise ValueError("pseudo_labels returned {} values, expected one per frame".format(
            owner.size))
    owner = np.where(np.isin(owner, (0, 1, 2)), owner, -1).astype(np.int8)
    core = np.isin(owner, (0, 1)) & graded
    return Reference(name=PSEUDO, owner=owner, core=core,
                     noncore=np.zeros(n_frames, dtype=bool))


# ---------------------------------------------------------------- helpers

def _share(part, whole):
    return round(float(part) / float(whole), 4) if whole else None


def agree_mask(vote, owner):
    """bool: the vote names the reference owner (+ with H, - with A)."""
    return ((vote > 0) & (owner == 0)) | ((vote < 0) & (owner == 1))


def _agreement(vote, owner, mask):
    m = mask & (vote != 0)
    return int(m.sum()), _share((agree_mask(vote, owner) & m).sum(), m.sum())


def filled_owner(owner, spans):
    """owner with -1 frames carrying the last H / A / D before them, inside each graded span only;
    -1 outside the spans. For decoded labels (no -1 inside a span) it is the labels themselves."""
    out = np.full(len(owner), -1, dtype=np.int8)
    for a, b in spans:
        seg = owner[a:b]
        known = seg >= 0
        idx = np.where(known, np.arange(len(seg)), -1)
        idx = np.maximum.accumulate(idx) if len(idx) else idx
        out[a:b] = np.where(idx >= 0, seg[np.maximum(idx, 0)], -1)
    return out


def changes_to(filled, spans, team):
    """bool: frames where the (filled) reference turns to `team` from another known state."""
    out = np.zeros(len(filled), dtype=bool)
    for a, b in spans:
        seg = filled[a:b]
        if len(seg) < 2:
            continue
        prev, cur = seg[:-1], seg[1:]
        out[a + 1:b] = (cur == team) & (prev >= 0) & (prev != team)
    return out


def _any_in(cum, a, b):
    """Is any flag set in [a, b) (clipped)? cum = r_[0, cumsum(flags)]."""
    n = len(cum) - 1
    a, b = max(0, a), min(n, b)
    return b > a and cum[b] - cum[a] > 0


def _cum(flags):
    return np.r_[0, np.cumsum(flags.astype(np.int64))]


def _window_chance(cum, graded, lo, hi):
    """Share of graded frames f for which [f + lo, f + hi) holds a flag."""
    n = len(cum) - 1
    f = np.flatnonzero(graded)
    if not len(f):
        return None
    a = np.clip(f + lo, 0, n)
    b = np.clip(f + hi, 0, n)
    return round(float(np.mean(cum[b] - cum[a] > 0)), 4)


def ball_seen(root, n_frames):
    """bool (n,) from ball.parquet's ``seen`` column, or None when the root has no ball file."""
    path = os.path.join(root, BALL_FILE)
    if not os.path.isfile(path):
        return None
    ball = pd.read_parquet(path, columns=["frame", "seen"])
    seen = np.zeros(n_frames, dtype=bool)
    frames = ball["frame"].to_numpy().astype(np.int64)
    ok = (frames >= 0) & (frames < n_frames)
    seen[frames[ok]] = ball["seen"].to_numpy().astype(bool)[ok]
    return seen


def read_restarts(root):
    path = os.path.join(root, RESTARTS_FILE)
    if not os.path.isfile(path):
        return None
    table = pd.read_parquet(path)
    return table[table["restart_frame"] >= 0].reset_index(drop=True)


# ---------------------------------------------------------------- per-code table

def code_row(vote, covered, n_tags, n_with_team, ref, graded, seen):
    """One row of the G10 table (a dict of at most 20 scalars)."""
    voting = (vote != 0) & graded
    core_voting, agree_core = _agreement(vote, ref.owner, ref.core)
    row = {"tags": int(n_tags), "tags_with_team": int(n_with_team),
           "frames_covered": int((covered & graded).sum()),
           "voting_frames": int(voting.sum()),
           "core_covered": int((covered & ref.core).sum()),
           "core_voting": core_voting, "agree_core": agree_core}
    for name, part in (("seen", seen), ("missing", None if seen is None else ~seen)):
        if part is None:
            row["core_voting_" + name], row["agree_core_" + name] = None, None
        else:
            row["core_voting_" + name], row["agree_core_" + name] = _agreement(
                vote, ref.owner, ref.core & part)
    row["noncore_voting"], row["agree_noncore_uncertain"] = _agreement(vote, ref.owner,
                                                                        ref.noncore)
    row["share_voting_ref_d"] = _share((voting & (ref.owner == 2)).sum(), voting.sum())
    row["share_voting_ref_none"] = _share((voting & (ref.owner < 0)).sum(), voting.sum())
    return row


def code_table(tags, frames, ref, seen):
    """{"agreement_<code slug>": row} for every voting code, plus "agreement_tag_vote"."""
    table, n, graded = tags.table, tags.n_frames, tags.graded
    out = {}
    any_cover = np.zeros(n, dtype=bool)
    for code in votes.VOTE_CODES:
        part = table[table["code"] == code]
        cover = stage.covered(table, code, n)
        any_cover |= cover
        vote = frames[votes.vote_column(code)].to_numpy()
        out["agreement_" + votes.slug(code)] = code_row(
            vote, cover, len(part), part["team"].isin([0, 1]).sum(), ref, graded, seen)
    voting_rows = table[table["code"].isin(votes.VOTE_CODES)]
    tag_vote = np.sign(frames["tag_vote"].to_numpy()).astype(np.int8)
    out["agreement_tag_vote"] = code_row(tag_vote, any_cover, len(voting_rows),
                                         voting_rows["team"].isin([0, 1]).sum(), ref, graded, seen)
    return out


# ---------------------------------------------------------------- set pieces

def _type_agrees(tag_type, detected):
    if detected == "goal_kick_or_corner":
        return tag_type in ("goal_kick", "corner")
    return tag_type == detected


def _nearest(frames_, a, b):
    """Index of the frame nearest to [a, b) (0 distance inside), or None when frames_ is empty."""
    if not len(frames_):
        return None
    dist = np.where(frames_ < a, a - frames_, np.where(frames_ >= b, frames_ - (b - 1), 0))
    return int(np.argmin(dist)), int(dist.min())


def set_piece_block(tags, ref, restarts, spans):
    table, n, fps = tags.table, tags.n_frames, tags.fps
    tol = int(round(SET_PIECE_TOL_S * fps))
    sets = table[table["code"] == parse.SET_PIECES]
    windows = []
    for start, end, team, set_type in zip(sets["start_frame"].tolist(),
                                          sets["end_frame"].tolist(), sets["team"].tolist(),
                                          sets["set_type"].tolist()):
        a, b = votes.clip(start, end, n)
        if b > a:
            windows.append((a, b, int(team), parse.RESTART_TYPE.get(set_type, parse.OTHER_TYPE)))

    out = {"windows": len(windows)}
    by_type_n, by_type_hit = {}, {}
    if restarts is None:
        out["restarts_file"] = "absent"
    else:
        rf = restarts["restart_frame"].to_numpy().astype(np.int64)
        p_home = restarts["p_home_restart"].to_numpy().astype(float)
        r_type = restarts["restart_type"].astype(str).to_numpy()
        inside = near = team_pairs = team_agree = type_pairs = type_agree = 0
        for a, b, team, kind in windows:
            inside += int(((rf >= a) & (rf < b)).any())
            hit = _nearest(rf, a, b)
            by_type_n[kind] = by_type_n.get(kind, 0) + 1
            if hit is None or hit[1] > tol:
                continue
            near += 1
            by_type_hit[kind] = by_type_hit.get(kind, 0) + 1
            i = hit[0]
            if team in (0, 1) and p_home[i] != 0.5:
                team_pairs += 1
                team_agree += int((p_home[i] > 0.5) == (team == 0))
            if kind in TEAM_DEFINED and r_type[i] in TEAM_DEFINED + ("goal_kick_or_corner",):
                type_pairs += 1
                type_agree += int(_type_agrees(kind, r_type[i]))
        out.update({"share_with_restart_inside": _share(inside, len(windows)),
                    "share_with_restart_within_tol": _share(near, len(windows)),
                    "team_pairs": team_pairs,
                    "team_agree_p_home_restart": _share(team_agree, team_pairs),
                    "type_pairs": type_pairs, "type_agree": _share(type_agree, type_pairs)})
        # G7 seen from the other side: the share of detected restarts inside some window
        in_window = np.zeros(len(rf), dtype=bool)
        for a, b, _, _ in windows:
            in_window |= (rf >= a - tol) & (rf < b + tol)
        out["detected_restarts"] = int(len(rf))
        out["share_detected_restarts_in_a_window"] = _share(in_window.sum(), len(rf))

    # the reference's own restarts: a change from D to H or A
    filled = filled_owner(ref.owner, spans)
    starts = np.zeros(n, dtype=bool)
    for a, b in spans:
        seg = filled[a:b]
        if len(seg) >= 2:
            starts[a + 1:b] = (seg[:-1] == 2) & np.isin(seg[1:], (0, 1))
    restart_frames = np.flatnonzero(starts)
    ref_hit = ref_pairs = ref_agree = 0
    for a, b, team, _ in windows:
        hit = _nearest(restart_frames, a, b)
        if hit is None or hit[1] > tol:
            continue
        ref_hit += 1
        if team in (0, 1):
            ref_pairs += 1
            ref_agree += int(filled[restart_frames[hit[0]]] == team)
    out.update({"share_with_ref_restart_within_tol": _share(ref_hit, len(windows)),
                "ref_team_pairs": ref_pairs, "ref_team_agree": _share(ref_agree, ref_pairs)})
    by_type = {kind: _share(by_type_hit.get(kind, 0), count)
               for kind, count in sorted(by_type_n.items())}
    return out, by_type


# ---------------------------------------------------------------- transitions

def transition_block(tags, ref, spans):
    table, n, fps, graded = tags.table, tags.n_frames, tags.fps, tags.graded
    filled = filled_owner(ref.owner, spans)
    cums = {team: _cum(changes_to(filled, spans, team)) for team in (0, 1)}
    hint = int(round(votes.TURNOVER_HINT_S * fps))
    reach = int(round(CHANGE_REACH_S * fps))
    out = {}
    for code in (parse.ATTACKING_TRANSITION, parse.DEFENSIVE_TRANSITION):
        part = table[table["code"] == code]
        n_team = hit_hint = hit_reach = 0
        for start, team in zip(part["start_frame"].tolist(), part["team"].tolist()):
            to = votes.hinted_team(code, team)
            if to < 0:
                continue
            n_team += 1
            a, b = votes.hint_span(start, fps)
            hit_hint += int(_any_in(cums[to], a, b + 1))   # a change dated at the tag start counts
            hit_reach += int(_any_in(cums[to], start - reach, start + reach + 1))
        out["transition_" + votes.slug(code)] = {
            "tags": int(len(part)), "tags_with_team": n_team,
            "share_change_to_team_in_hint_window": _share(hit_hint, n_team),
            "share_change_to_team_within_reach": _share(hit_reach, n_team)}
    both = _cum(changes_to(filled, spans, 0) | changes_to(filled, spans, 1))
    out["transition_chance_rates"] = {
        "hint_window_s": votes.TURNOVER_HINT_S, "reach_s": CHANGE_REACH_S,
        "chance_change_any_team_in_hint_window": _window_chance(both, graded, -hint, 1),
        "chance_change_any_team_within_reach": _window_chance(both, graded, -reach, reach + 1),
        "ref_changes_to_home": int(cums[0][-1]), "ref_changes_to_away": int(cums[1][-1])}
    return out


# ---------------------------------------------------------------- goals

def _runs(flags):
    d = np.diff(np.r_[0, flags.astype(np.int8), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def goal_block(tags, ref, restarts):
    table, n, fps = tags.table, tags.n_frames, tags.fps
    goals = table[table["code"] == parse.GOALS]
    pad = int(round(votes.GOAL_PAD_S * fps))
    follow = int(round(GOAL_FOLLOW_S * fps))
    long_len = int(round(LONG_STOPPAGE_S * fps))
    ko_reach = int(round(GOAL_KICKOFF_MAX_S * fps))
    d_start, d_stop = _runs(ref.owner == 2)
    keep = (d_stop - d_start) >= long_len
    d_start, d_stop = d_start[keep], d_stop[keep]
    out = {"tags": int(len(goals)), "tags_with_team": int(goals["team"].isin([0, 1]).sum()),
           "followed_by_long_ref_d": 0}
    if restarts is None:
        out["restarts_file"] = "absent"
    else:
        st = restarts[(restarts["start_kind"] != "half_start") & (restarts["start_frame"] >= 0)]
        s0 = st["start_frame"].to_numpy().astype(np.int64)
        s1 = st["restart_frame"].to_numpy().astype(np.int64)
        long_rows = (s1 - s0) >= long_len
        s0, s1 = s0[long_rows], s1[long_rows]
        ko = restarts[(restarts["restart_type"] == "kick_off")
                      & (restarts["start_kind"] == "goal")]
        ko_frames = ko["restart_frame"].to_numpy().astype(np.int64)
        ko_p = ko["p_home_restart"].to_numpy().astype(float)
        out.update({"followed_by_long_stoppage": 0, "followed_by_goal_kickoff": 0,
                    "kickoff_team_pairs": 0, "kickoff_is_conceding_team": 0})
    for start, team in zip(goals["start_frame"].tolist(), goals["team"].tolist()):
        a, b = start - pad, start + follow
        out["followed_by_long_ref_d"] += int(((d_start < b) & (d_stop > a)).any())
        if restarts is None:
            continue
        out["followed_by_long_stoppage"] += int(((s0 < b) & (s1 > a)).any())
        after = (ko_frames >= start) & (ko_frames <= start + ko_reach)
        if not after.any():
            continue
        out["followed_by_goal_kickoff"] += 1
        i = int(np.flatnonzero(after)[np.argmin(ko_frames[after])])
        if team in (0, 1) and ko_p[i] != 0.5:
            out["kickoff_team_pairs"] += 1
            out["kickoff_is_conceding_team"] += int((ko_p[i] > 0.5) == (team == 1))
    return out


# ---------------------------------------------------------------- kept codes and gate (P-20)

def mirror_codes(frames):
    """{dropped code: kept code} for codes whose per-frame vote equals an earlier code's.

    A mirror pair (BUILD UP of T and DEFENDING IN ATTACKING THIRD of the other team, ...) votes
    for the same team on the same frames, so the two vote arrays are equal. Detected from the
    arrays, not from a list of pairs; the earlier code in VOTE_CODES order (attacking first) is
    kept. Codes that never vote are not mirrors of anything.
    """
    out, kept = {}, []
    for code in votes.VOTE_CODES:
        vote = frames[votes.vote_column(code)].to_numpy()
        if not vote.any():
            continue
        twin = next((k for k, v in kept if np.array_equal(v, vote)), None)
        if twin is None:
            kept.append((code, vote))
        else:
            out[code] = twin
    return out


def code_weight(agree):
    """w_k = min(W_CAP, LAMBDA * logit(A_k)) nats per frame (P-20 rule 4); A_k = 1 gives W_CAP."""
    if agree >= 1.0:
        return W_CAP
    return round(min(W_CAP, LAMBDA * float(np.log(agree / (1.0 - agree)))), 4)


def kept_codes(rows, mirrors, a_min):
    """[(code, A_k, w_k)] of the non-mirror codes whose core agreement is at least a_min."""
    out = []
    for code in votes.VOTE_CODES:
        agree = rows["agreement_" + votes.slug(code)]["agree_core"]
        if code in mirrors or agree is None or agree < a_min:
            continue
        out.append((code, agree, code_weight(agree)))
    return out


def gate_row(frames, ref, graded, kept, a_min):
    """P-20 rules 1 and 2 for one threshold: coverage, agreement and C * (A - GATE_BASE)."""
    n = len(graded)
    total = np.zeros(n, dtype=np.float64)
    any_vote = np.zeros(n, dtype=bool)
    for code, _, weight in kept:
        vote = frames[votes.vote_column(code)].to_numpy()
        total += weight * vote
        any_vote |= vote != 0
    combined = np.sign(total).astype(np.int8)
    live = np.isin(ref.owner, (0, 1)) & graded
    coverage = _share(((combined != 0) & live).sum(), live.sum())
    core_voting, agree_core = _agreement(combined, ref.owner, ref.core)
    noncore_voting, agree_noncore = _agreement(combined, ref.owner, ref.noncore)
    gate = None
    if coverage is not None and agree_core is not None:
        gate = round(coverage * (agree_core - GATE_BASE), 4)
    return {"a_min": a_min, "kept_codes": len(kept), "live_frames": int(live.sum()),
            "coverage": coverage,
            "coverage_any_kept_vote": _share((any_vote & live).sum(), live.sum()),
            "core_voting": core_voting, "agree_core": agree_core,
            "noncore_voting": noncore_voting, "agree_noncore_uncertain": agree_noncore,
            "gate_value": gate, "gate_min": GATE_MIN,
            "gate_pass": None if gate is None else bool(gate >= GATE_MIN)}


def _threshold_tag(a_min):
    return "a{:03d}".format(int(round(a_min * 100)))


def gate_block(frames, ref, graded, rows, mirrors):
    """{"gate_<a>": row, "kept_agree_<a>": {code: A_k}, "kept_weight_<a>": {code: w_k}} for
    A_MIN and each sensitivity threshold, plus the mirror pairs found."""
    out = {"mirror_codes_dropped": {votes.slug(c): votes.slug(t) for c, t in mirrors.items()}}
    for a_min in (A_MIN,) + A_MIN_SENSITIVITY:
        kept = kept_codes(rows, mirrors, a_min)
        tag = _threshold_tag(a_min)
        out["gate_" + tag] = gate_row(frames, ref, graded, kept, a_min)
        out["kept_agree_" + tag] = {votes.slug(c): a for c, a, _ in kept}
        out["kept_weight_" + tag] = {votes.slug(c): w for c, _, w in kept}
    return out


def changes_out_of(filled, spans, team):
    """bool: frames where the (filled) reference leaves `team` for another known state."""
    out = np.zeros(len(filled), dtype=bool)
    for a, b in spans:
        seg = filled[a:b]
        if len(seg) < 2:
            continue
        out[a + 1:b] = (seg[:-1] == team) & (seg[1:] >= 0) & (seg[1:] != team)
    return out


def nearest_offsets(edges, change_frames, reach):
    """(offsets edge - nearest change within +-reach, n unmatched). Ties go to the earlier change."""
    offsets, unmatched = [], 0
    for edge in edges:
        if not len(change_frames):
            unmatched += 1
            continue
        d = edge - change_frames
        i = int(np.argmin(np.abs(d)))
        if abs(int(d[i])) <= reach:
            offsets.append(int(d[i]))
        else:
            unmatched += 1
    return offsets, unmatched


def lag_summary(offsets, unmatched):
    """Counts always; quantiles only with at least LAG_MIN_MATCHED offsets (never one instance)."""
    out = {"matched": len(offsets), "unmatched": int(unmatched),
           "q25_frames": None, "median_frames": None, "q75_frames": None}
    if len(offsets) >= LAG_MIN_MATCHED:
        q25, med, q75 = np.percentile(np.asarray(offsets, dtype=float), [25, 50, 75])
        out.update({"q25_frames": round(float(q25), 1), "median_frames": round(float(med), 1),
                    "q75_frames": round(float(q75), 1)})
    return out


def _lag_context(tags, ref, spans):
    """Reference changes INTO and OUT OF each team (frame indices) and the reach in frames."""
    filled = filled_owner(ref.owner, spans)
    into = {t: np.flatnonzero(changes_to(filled, spans, t)) for t in (0, 1)}
    out_of = {t: np.flatnonzero(changes_out_of(filled, spans, t)) for t in (0, 1)}
    return into, out_of, int(round(CHANGE_REACH_S * tags.fps))


def code_edge_offsets(tags, code, context):
    """{"start": (offsets, unmatched), "end": (offsets, unmatched)} for one code's instances.

    Start edge minus the nearest reference change INTO the code's voting team, end edge minus the
    nearest change OUT of it, within +-CHANGE_REACH_S (frames, positive = the tag is later).
    Instances that vote for nobody (no team) are skipped."""
    into, out_of, reach = context
    part = tags.table[tags.table["code"] == code]
    per = {"start": ([], 0), "end": ([], 0)}
    for start, end, team in zip(part["start_frame"].tolist(), part["end_frame"].tolist(),
                                part["team"].tolist()):
        sign = votes.vote_sign(code, team)
        if sign == 0:
            continue
        voted = 0 if sign > 0 else 1
        for edge_name, edge, changes in (("start", start, into[voted]),
                                         ("end", end, out_of[voted])):
            offs, miss = nearest_offsets([edge], changes, reach)
            per[edge_name] = (per[edge_name][0] + offs, per[edge_name][1] + miss)
    return per


def edge_lag_block(tags, ref, spans, kept, a_min=None):
    """P-20 rule 3 as first written (pooled): tag start edge minus the nearest reference change
    INTO the voting team, and tag end edge minus the nearest change OUT of it, within
    +-CHANGE_REACH_S; positive = the tag is later. Per kept code and pooled; aggregates only.

    ``a_min`` is the threshold the ``kept`` set was chosen at; it is printed as given (None when
    the caller does not say), never assumed to be A_MIN. The pooled gate here is kept for
    comparability with earlier reports; the rule in force is per code (``lag_filtered_kept``)."""
    context = _lag_context(tags, ref, spans)
    pooled = {"start": ([], 0), "end": ([], 0)}
    out = {}
    for code, _, _ in kept:
        per = code_edge_offsets(tags, code, context)
        for edge_name in ("start", "end"):
            pooled[edge_name] = (pooled[edge_name][0] + per[edge_name][0],
                                 pooled[edge_name][1] + per[edge_name][1])
            out["edge_lag_{}_{}".format(edge_name, votes.slug(code))] = lag_summary(
                *per[edge_name])
    for edge_name in ("start", "end"):
        out["edge_lag_{}_all_kept".format(edge_name)] = lag_summary(*pooled[edge_name])
    median = out["edge_lag_start_all_kept"]["median_frames"]
    out["edge_lag_gate"] = {"a_min": a_min, "reach_s": CHANGE_REACH_S,
                            "max_median_start_frames": LAG_GATE_FRAMES,
                            "min_matched_for_quantiles": LAG_MIN_MATCHED,
                            "median_start_frames": median,
                            "gate_pass": None if median is None else bool(median <= LAG_GATE_FRAMES)}
    return out


UNMEASURED = "unmeasured"


def lag_filtered_kept(tags, ref, spans, kept):
    """P-20 rule 3 per code: (kept_after, per_code_lag, dropped).

    For each kept code, its own start-edge lag (same matching as ``edge_lag_block``: start edge
    minus the nearest reference change into the code's voting team within +-CHANGE_REACH_S). A
    code stays only when its median is measurable (at least LAG_MIN_MATCHED matched start edges)
    and at most LAG_GATE_FRAMES. A late code votes for the old team after a turnover (T-12b), so
    early codes may not offset it, as they could in the pooled median.

    A code whose lag cannot be measured is DROPPED: this is the conservative choice (an unknown
    lag is treated as a failing one), and it also keeps a code with very few tags out.

    - kept_after: the surviving [(code, A_k, w_k)] in the input order, weights unchanged
      (w_k = min(W_CAP, LAMBDA * logit(A_k)) is not renormalised);
    - per_code_lag: {code: lag_summary of its start edges} for every input code;
    - dropped: {code: median start lag in frames, or UNMEASURED}.
    """
    context = _lag_context(tags, ref, spans)
    kept_after, per_code_lag, dropped = [], {}, {}
    for item in kept:
        code = item[0]
        summary = lag_summary(*code_edge_offsets(tags, code, context)["start"])
        per_code_lag[code] = summary
        median = summary["median_frames"]
        if median is None:
            dropped[code] = UNMEASURED
        elif median > LAG_GATE_FRAMES:
            dropped[code] = median
        else:
            kept_after.append(item)
    return kept_after, per_code_lag, dropped


def gate_after_lag(frames, tags, ref, spans, kept_after, dropped, a_min):
    """P-20 gate on the set that survives both filters (agreement at a_min, then per-code lag).

    Rule 2 (C * (A - GATE_BASE) >= GATE_MIN) is ``gate_row`` on ``kept_after``. Rule 3 is "every
    kept code passes its own lag test", true by construction after ``lag_filtered_kept``. The
    pooled start-edge lag of the survivors is printed for information only. gate_pass is True
    only when rule 2 passes on a non-empty surviving set (None from gate_row counts as False)."""
    row = gate_row(frames, ref, tags.graded, kept_after, a_min)
    pooled = edge_lag_block(tags, ref, spans, kept_after, a_min)["edge_lag_start_all_kept"]
    rule2 = row["gate_pass"] is True
    row.update({"lag_rule": "per_code", "lag_gate_frames": LAG_GATE_FRAMES,
                "lag_dropped_codes": len(dropped),
                "every_kept_code_passes_lag": True,
                "pooled_start_lag_matched_info": pooled["matched"],
                "pooled_start_lag_median_frames_info": pooled["median_frames"],
                "gate_pass": bool(rule2 and len(kept_after) > 0)})
    return row


def lag_gate_block(frames, tags, ref, spans, rows, mirrors):
    """For A_MIN and each sensitivity threshold: "lag_dropped_<a>" {code: median or UNMEASURED},
    "kept_after_lag_<a>" {code: A_k}, "kept_after_lag_weight_<a>" {code: w_k} (two flat dicts:
    report.summary refuses nested ones) and "gate_after_lag_<a>" (the gate row on the survivors)."""
    out = {}
    for a_min in (A_MIN,) + A_MIN_SENSITIVITY:
        tag = _threshold_tag(a_min)
        kept_after, _, dropped = lag_filtered_kept(tags, ref, spans,
                                                   kept_codes(rows, mirrors, a_min))
        out["lag_dropped_" + tag] = {votes.slug(c): v for c, v in dropped.items()}
        out["kept_after_lag_" + tag] = {votes.slug(c): a for c, a, _ in kept_after}
        out["kept_after_lag_weight_" + tag] = {votes.slug(c): w for c, _, w in kept_after}
        out["gate_after_lag_" + tag] = gate_after_lag(frames, tags, ref, spans, kept_after,
                                                      dropped, a_min)
    return out


# ---------------------------------------------------------------- entry

def reference_summary(ref, graded):
    n_graded = int(graded.sum())
    live = np.isin(ref.owner, (0, 1)) & graded
    return {"name": ref.name, "core_s": CORE_S if ref.name != PSEUDO else None,
            "graded_frames": n_graded,
            "core_frames": int(ref.core.sum()), "share_graded_core": _share(ref.core.sum(),
                                                                            n_graded),
            "noncore_live_frames": int(ref.noncore.sum()),
            "share_graded_live": _share(live.sum(), n_graded),
            "share_graded_d": _share(((ref.owner == 2) & graded).sum(), n_graded),
            "share_graded_none": _share(((ref.owner < 0) & graded).sum(), n_graded)}


def build_reference(root, tags, ref="version", ref_version=None, pseudo_fn=None):
    if ref == PSEUDO:
        return pseudo_reference(root, tags.n_frames, tags.graded, pseudo_fn=pseudo_fn)
    if ref != "version" or not ref_version:
        raise ReferenceUnavailable("give --ref-version NAME or --ref pseudo")
    return version_reference(root, ref_version, tags.n_frames, tags.fps, tags.graded)


def run(root, ref="version", ref_version=None, xml_path=None, aliases=None, pseudo_fn=None):
    """The G10 table for a root as a dict of aggregates; {"tags": "no tags"} when it has none.

    Writes nothing. The frame arrays are rebuilt from the XML with the stage's code, so the table
    never reads a stale tags.parquet.
    """
    tags = stage.load(root, xml_path=xml_path, aliases=aliases)
    if tags is None:
        return {"tags": stage.NO_TAGS, "written": False}
    reference = build_reference(root, tags, ref, ref_version, pseudo_fn)
    frames = votes.frame_arrays(tags.table, tags.n_frames, tags.fps)
    spans = stage.graded_mask(tags.meta, tags.n_frames)[1]
    seen = ball_seen(root, tags.n_frames)
    restarts = read_restarts(root)

    out = {"tags": "found", "reference": reference_summary(reference, tags.graded),
           "ball_seen_split": "absent" if seen is None else "ball.parquet seen"}
    rows = code_table(tags, frames, reference, seen)
    out.update(rows)
    mirrors = mirror_codes(frames)
    out.update(gate_block(frames, reference, tags.graded, rows, mirrors))
    out.update(edge_lag_block(tags, reference, spans, kept_codes(rows, mirrors, A_MIN), A_MIN))
    out.update(lag_gate_block(frames, tags, reference, spans, rows, mirrors))
    out["set_pieces"], out["set_pieces_within_tol_by_type"] = set_piece_block(
        tags, reference, restarts, spans)
    out.update(transition_block(tags, reference, spans))
    out["goals"] = goal_block(tags, reference, restarts)
    out["params"] = {"core_s": CORE_S, "set_piece_tol_s": SET_PIECE_TOL_S,
                     "change_reach_s": CHANGE_REACH_S, "long_stoppage_s": LONG_STOPPAGE_S,
                     "goal_follow_s": GOAL_FOLLOW_S, "goal_kickoff_max_s": GOAL_KICKOFF_MAX_S,
                     "turnover_hint_s": votes.TURNOVER_HINT_S,
                     "long_ball_vote_s": votes.LONG_BALL_VOTE_S, "goal_pad_s": votes.GOAL_PAD_S,
                     "a_min": A_MIN, "lambda": LAMBDA, "w_cap": W_CAP, "gate_base": GATE_BASE,
                     "gate_min": GATE_MIN, "lag_gate_frames": LAG_GATE_FRAMES}
    out["written"] = False
    return out
