"""Everything the decoders read, as per-frame arrays (contract files only, CONTRACTS.md).

- probabilities: ``<root>/probs/<model>.npy`` float32 (n_frames, 3) = P(H), P(A), P(D);
- graded frames: ``meta.json`` ``graded_spans`` when present (bench level roots), otherwise
  ``halves`` (the match: graded spans are the two half video blocks);
- S4 mask: ``features_dead.parquet`` ``mask_pre_kickoff`` (frames that must be D);
- restart-team evidence for the dead branch (R0, R8, S6): ``restarts.parquet`` p_home_restart
  where a row exists, otherwise the exit's last_touch_team from ``events.parquet`` (R8: the last
  team to touch loses the restart), otherwise nothing (0.5).

Every missing optional file is treated as "no evidence", never as a default team.

Restart evidence is a per-frame log-likelihood ratio ``restart_llr`` (+ favours DH) spread evenly
over a short window around the stoppage, so its total over the stoppage equals logit(p) whatever
frame the decoder starts the stoppage on. Evidence at or beyond HARD_P (the kick-off law, S6)
becomes a mask on the wrong dead branch: law, not a cost.
"""
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Params (start values; not swept: they only matter when restart evidence exists)
P_FLOOR = 1e-3              # probabilities are clipped to [P_FLOOR, 1] before -log
HARD_P = 0.99               # p_home_restart at or beyond this (or 1 - it) is treated as law
SOFT_P_CLIP = 0.98          # softer evidence is clipped to [1 - this, this]
RESTART_WINDOW = 50         # frames before restart_frame carrying a restarts.parquet row (2 s)
EXIT_WINDOW = 50            # frames from the exit frame carrying its last-touch evidence (2 s)
EXIT_RESTART_P = 0.80       # P(other team restarts | exit with a known last touch), R8; start
                            # value, to be replaced by the measured last-touch precision
EXIT_COVERED_BY_ROW = 90 * 25   # an exit within this many frames before a restarts row's restart
                                # frame belongs to that row's stoppage; the row wins


@dataclass
class DecodeInputs:
    n: int
    halves: list
    graded: np.ndarray                  # bool (n,)
    graded_spans: list                  # [(start, stop)]
    probs: np.ndarray                   # float64 (n, 3), clipped and renormalised
    mask_dead: np.ndarray               # bool (n,): must be D (S4)
    restart_llr: np.ndarray             # float64 (n,), + favours DH
    hard_home: np.ndarray               # bool (n,): DA forbidden here (law says home restarts)
    hard_away: np.ndarray               # bool (n,): DH forbidden here
    switch_cost: np.ndarray = None      # float64 (n,) or None -> constant c_switch (v3 hook)
    stop_cost: np.ndarray = None        # same for live -> dead (v4 hook)
    restart_cost: np.ndarray = None     # same for dead -> live (v4 hook)
    info: dict = field(default_factory=dict)
    # evidence per mode: {"all" | "kickoff": (restart_llr, hard_home, hard_away)}; the fields
    # above hold the "all" mode (T-5 behaviour). restart_points: (frames, conf) of the restart
    # rows, for the restart-frame bonus (v2s).
    evidence: dict = field(default_factory=dict)
    restart_points: tuple = None
    # v3: switch anchors from the events table (pipeline.decode.control.anchors; None when the
    # root has no events table) and ball seen flags (None when the root has no ball.parquet).
    anchors: object = None
    seen: np.ndarray = None
    # v4: (start frames, conf) of the stoppage candidates (restarts rows with a known start that
    # is not a half start), for the stop-cost discount.
    stop_points: tuple = None
    # T-42 (v12g): bool (n,) frames where the version's team probability differs from its
    # reference model's (the trees' busy gate); None unless the version names a "gate_ref".
    gate: np.ndarray = None


def read_meta(root):
    with open(os.path.join(root, "meta.json")) as handle:
        return json.load(handle)


def graded_mask(meta, n):
    spans = meta.get("graded_spans") or meta["halves"]
    graded = np.zeros(n, dtype=bool)
    out = []
    for a, b in spans:
        a, b = max(0, int(a)), min(n, int(b))
        if b > a:
            graded[a:b] = True
            out.append((a, b))
    return graded, out


def load_probs(root, model, n):
    path = os.path.join(root, "probs", "{}.npy".format(model))
    if not os.path.isfile(path):
        raise FileNotFoundError("probs/{}.npy not found in the root".format(model))
    p = np.load(path)
    if p.shape != (n, 3):
        raise ValueError("probs/{}.npy has shape {}, expected ({}, 3)".format(model, p.shape, n))
    return p


def clean_probs(p):
    p = np.asarray(p, dtype=np.float64).copy()
    bad = ~np.isfinite(p).all(axis=1)
    p[bad] = 1.0 / 3.0
    p = np.clip(p, P_FLOOR, 1.0)
    return p / p.sum(axis=1, keepdims=True), int(bad.sum())


def _read_optional(root, name, columns=None):
    path = os.path.join(root, name)
    if not os.path.isfile(path):
        return None
    return pd.read_parquet(path, columns=columns)


def pre_kickoff_mask(root, n):
    fd = _read_optional(root, "features_dead.parquet", ["frame", "mask_pre_kickoff"])
    mask = np.zeros(n, dtype=bool)
    if fd is None:
        return mask, False
    fr = fd["frame"].to_numpy()
    ok = (fr >= 0) & (fr < n)
    mask[fr[ok]] = fd["mask_pre_kickoff"].to_numpy(dtype=bool)[ok]
    return mask, True


def _add_evidence(llr, hard_home, hard_away, a, b, p, law=True):
    """Spread logit(p) over [a, b); with law, p beyond HARD_P becomes a mask on the wrong branch
    (only kick-off rows are law, S6/R13; every other row is soft evidence)."""
    n = len(llr)
    a, b = max(0, int(a)), min(n, int(b))
    if b <= a or not np.isfinite(p) or abs(p - 0.5) < 1e-9:
        return 0
    if law and p >= HARD_P:
        hard_home[a:b] = True
        return 1
    if law and p <= 1.0 - HARD_P:
        hard_away[a:b] = True
        return 1
    q = float(np.clip(p, 1.0 - SOFT_P_CLIP, SOFT_P_CLIP))
    llr[a:b] += np.log(q / (1.0 - q)) / (b - a)
    return 1


def restart_evidence(n, restarts=None, events=None):
    """(restart_llr, hard_home, hard_away, counts). Inputs are DataFrames or None."""
    llr = np.zeros(n, dtype=np.float64)
    hard_home = np.zeros(n, dtype=bool)
    hard_away = np.zeros(n, dtype=bool)
    counts = {"restart_rows_used": 0, "exit_evidence_used": 0, "exit_covered_by_row": 0}
    row_frames = np.zeros(0, dtype=np.int64)
    if restarts is not None and len(restarts):
        row_frames = restarts["restart_frame"].to_numpy(dtype=np.int64)
        starts = restarts["start_frame"].to_numpy(dtype=np.int64) \
            if "start_frame" in restarts else np.full(len(restarts), -1)
        law = (restarts["restart_type"].astype(str).to_numpy() == "kick_off") \
            if "restart_type" in restarts else np.ones(len(restarts), dtype=bool)
        for r, s, p, lw in zip(row_frames, starts,
                               restarts["p_home_restart"].to_numpy(dtype=float), law):
            a = max(int(s), int(r) - RESTART_WINDOW) if s >= 0 else int(r) - RESTART_WINDOW
            counts["restart_rows_used"] += _add_evidence(llr, hard_home, hard_away, a, r, p,
                                                         bool(lw))
    if events is not None and len(events):
        ex = events[(events["kind"] == "exit") & (events["last_touch_team"] >= 0)]
        for f, t in zip(ex["frame"].to_numpy(dtype=np.int64),
                        ex["last_touch_team"].to_numpy(dtype=np.int64)):
            if len(row_frames) and np.any((row_frames > f) &
                                          (row_frames - f <= EXIT_COVERED_BY_ROW)):
                counts["exit_covered_by_row"] += 1
                continue
            p_home = EXIT_RESTART_P if t == 1 else 1.0 - EXIT_RESTART_P
            counts["exit_evidence_used"] += _add_evidence(llr, hard_home, hard_away,
                                                          f, f + EXIT_WINDOW, p_home)
    return llr, hard_home, hard_away, counts


def kickoff_in_memory(root):
    """(restarts, features_dead) from the restart stage's kick-off code, run in memory for a
    bench root that has no restarts.parquet / features_dead.parquet (nothing is written; the
    rule scorer does the same)."""
    from pipeline.restarts import kickoff
    meta = read_meta(root)
    players = pd.read_parquet(os.path.join(root, "players.parquet"))
    raw = os.path.join(root, "ball_raw.parquet")
    ball = pd.read_parquet(raw, columns=["frame", "x_m", "y_m"]) if os.path.isfile(raw) else None
    return kickoff.run(players, meta, ball)


def restarts_in_memory(root):
    """(restarts, features_dead) from the whole restarts stage (kick-offs + T-7 stoppages), run in
    memory for a bench root without restarts.parquet; nothing is written (the stage only writes
    roots it owns)."""
    from pipeline.restarts import stage
    return stage.run(stage.load_inputs(root))


def kickoff_rows(restarts):
    if restarts is None or not len(restarts) or "restart_type" not in restarts:
        return restarts
    return restarts[restarts["restart_type"].astype(str) == "kick_off"]


def restart_points(restarts):
    """(restart frames, conf) of the rows with a known restart frame."""
    if restarts is None or not len(restarts):
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    r = restarts["restart_frame"].to_numpy(dtype=np.int64)
    c = restarts["conf"].to_numpy(dtype=float) if "conf" in restarts else np.ones(len(r))
    ok = r >= 0
    return r[ok], np.nan_to_num(c[ok])


def stop_points(restarts):
    """(start frames, conf) of the restarts rows with a known start that is not a half start."""
    if restarts is None or not len(restarts) or "start_frame" not in restarts:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    s = restarts["start_frame"].to_numpy(dtype=np.int64)
    c = restarts["conf"].to_numpy(dtype=float) if "conf" in restarts else np.ones(len(s))
    ok = s >= 0
    if "start_kind" in restarts:
        ok &= restarts["start_kind"].astype(str).to_numpy() != "half_start"
    return s[ok], np.nan_to_num(c[ok])


def build(root, probs, restarts=None, events=None, use_root_evidence=True,
          compute_kickoff=False, compute=None):
    """DecodeInputs for a root. probs is an (n, 3) array (already loaded or synthesised).

    restarts / events: DataFrames overriding the root's files (synthetic tests); with
    use_root_evidence the root's restarts.parquet / events.parquet are read when not given.
    compute (bench roots without restarts files): None, "kickoff" (kick-off code only; the T-5
    behaviour, same as compute_kickoff=True) or "full" (the whole restarts stage).
    """
    if compute_kickoff and compute is None:
        compute = "kickoff"
    meta = read_meta(root)
    n = int(meta["n_frames"])
    graded, spans = graded_mask(meta, n)
    p, n_bad = clean_probs(probs)
    if p.shape != (n, 3):
        raise ValueError("probabilities have shape {}, expected ({}, 3)".format(p.shape, n))
    mask, have_mask = pre_kickoff_mask(root, n)
    kick_source = "file" if have_mask else "none"
    if compute and not have_mask:
        k_rows, feats = kickoff_in_memory(root) if compute == "kickoff" \
            else restarts_in_memory(root)
        fr = feats["frame"].to_numpy()
        mask = np.zeros(n, dtype=bool)
        mask[fr] = feats["mask_pre_kickoff"].to_numpy(dtype=bool)
        have_mask, kick_source = True, "computed_" + compute
        if restarts is None and not os.path.isfile(os.path.join(root, "restarts.parquet")):
            restarts = k_rows
    if use_root_evidence:
        if restarts is None:
            restarts = _read_optional(root, "restarts.parquet")
        if events is None:
            events = _read_optional(root, "events.parquet",
                                    ["frame", "kind", "team", "conf", "last_touch_team",
                                     "end_frame", "first_touch_frame", "confirmed_frame"])
    llr, hh, ha, counts = restart_evidence(n, restarts, events)
    k_llr, k_hh, k_ha, k_counts = restart_evidence(n, kickoff_rows(restarts), events)
    counts["restart_rows_total"] = 0 if restarts is None else int(len(restarts))
    counts["kickoff_mode_rows_used"] = k_counts["restart_rows_used"]
    counts["kickoff_mode_exit_used"] = k_counts["exit_evidence_used"]
    from pipeline.decode import control
    anc = control.anchors(events, [tuple(h) for h in meta["halves"]], float(meta.get("fps", 25))) \
        if events is not None and {"end_frame", "first_touch_frame"} <= set(events.columns) \
        else None
    seen = control.read_seen(root, n)
    if anc is not None:
        counts["switch_anchors"] = int(len(anc))
        counts["switch_anchors_control"] = int((anc["grade"] == control.GRADE_CONTROL).sum())
    counts["have_ball_seen"] = seen is not None
    info = {"probs_nonfinite_rows": n_bad, "have_features_dead": have_mask,
            "kickoff_source": kick_source,
            "have_restarts": restarts is not None, "have_events": events is not None,
            "mask_pre_kickoff_frames": int(mask.sum()), **counts}
    return DecodeInputs(n=n, halves=[tuple(h) for h in meta["halves"]], graded=graded,
                        graded_spans=spans, probs=p, mask_dead=mask, restart_llr=llr,
                        hard_home=hh, hard_away=ha, info=info,
                        evidence={"all": (llr, hh, ha), "kickoff": (k_llr, k_hh, k_ha)},
                        restart_points=restart_points(restarts), anchors=anc, seen=seen,
                        stop_points=stop_points(restarts))


def from_arrays(probs, graded_spans=None, mask_dead=None, restart_llr=None, hard_home=None,
                hard_away=None, switch_cost=None):
    """DecodeInputs from bare arrays (tests and sweeps); one half, all graded by default."""
    p, _ = clean_probs(probs)
    n = len(p)
    spans = graded_spans or [(0, n)]
    graded = np.zeros(n, dtype=bool)
    for a, b in spans:
        graded[a:b] = True
    z = np.zeros(n, dtype=bool)
    return DecodeInputs(n=n, halves=[(spans[0][0], spans[-1][1])], graded=graded,
                        graded_spans=list(spans), probs=p,
                        mask_dead=z.copy() if mask_dead is None else np.asarray(mask_dead, bool),
                        restart_llr=np.zeros(n) if restart_llr is None else
                        np.asarray(restart_llr, float),
                        hard_home=z.copy() if hard_home is None else np.asarray(hard_home, bool),
                        hard_away=z.copy() if hard_away is None else np.asarray(hard_away, bool),
                        switch_cost=switch_cost)


def attach_gate(inp, root, ref_model):
    """Set inp.gate from <root>/probs/<ref_model>.npy (T-42). Both sides cleaned the same way,
    so frames where the two models agree compare exactly equal. A missing file is an error: the
    version must not silently fall back to a constant cost."""
    from pipeline.decode import evidence_cost
    ref, _ = clean_probs(load_probs(root, ref_model, inp.n))
    inp.gate = evidence_cost.gate_from_reference(inp.probs, ref)
    inp.info["gate_ref"] = ref_model
    inp.info["gate_share_graded"] = round(float(inp.gate[inp.graded].mean()), 4) \
        if inp.graded.any() else 0.0
    return inp


def decode_segments(inp):
    """[(start, stop)] spans to decode: each graded span, extended backwards over the S4-masked
    frames just before it (so the kick-off evidence on those frames reaches the first restart),
    never across a half boundary (S3)."""
    out = []
    for a, b in inp.graded_spans:
        half_start = max([h0 for h0, h1 in inp.halves if h0 <= a < h1], default=0)
        s = a
        while s - 1 >= half_start and inp.mask_dead[s - 1]:
            s -= 1
        out.append((s, b))
    return out
