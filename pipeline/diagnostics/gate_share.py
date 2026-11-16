"""Gate share of a gated per-player tree version (T-44): how much of the graded time the trees
are used, as whole-match / whole-root aggregates. Reads; writes nothing but the report.

    uv run python -m pipeline.diagnostics.gate_share match --version v11g
    uv run python -m pipeline.diagnostics.gate_share bench --version v11g [--levels low match high]
    make gate-share VERSION=v11g PART=match|bench

For the version's probability model (VERSIONS[v]["probs"], a "trees" model with a "gate", e.g.
rule_p1g with gate G4 and base rule_w2) three per-frame flags are computed exactly as
models.team_model.tree_probs computes them (same functions, same inputs, same weights choice):

- flagged: the raw busy gate (player_model.gate_flags[gate]);
- speak:   the smoothed gate weight is above zero (the trees contribute to q);
- differ:  the stored team probability differs from the base model's (probs/<model>.npy against
           probs/<base>.npy, decode.evidence_cost.gate_from_reference, as decode.inputs.attach_gate
           does for versions that declare "gate_ref"; T-42's ``gate_share_graded``).

Each is reported as a share of the graded frames and of the graded frames the version's own
labels call live (H or A), when labels/<version>.csv exists. On bench roots also as a share of
the truly live frames (labels truth), the denominator of the judge's bench reference
(T-41 bench measurement: rule_p1g flags .370/.382/.408 and speaks on .463/.477/.508 of live frames at
low / match / high), so the match number can be read against the bench.

Nothing changes for any version: no labels, probabilities or stage outputs are written.
"""
import argparse
import os
import sys

import numpy as np

from pipeline.io.roots import require_bench_root
from pipeline.io.sources import ARTIFACTS, match_root
from pipeline.report import match_entry, quiet_warnings, summary

GAMES = ("metrica_g1", "metrica_g2", "metrica_g3")
LEVELS = ("low", "match", "high")
H, A = 0, 1
ROUND = 4
FLAGS = ("flagged", "speak", "differ")


def tree_params(version):
    """(model name, params) of the version's gated tree model; ValueError otherwise."""
    from pipeline.decode import decoders
    from pipeline.models import team_model as TMOD
    model = decoders.VERSIONS[version].get("probs", "rule_v1")
    params = TMOD.load_model(model) if model in TMOD.MODELS and \
        os.path.isfile(TMOD.params_path(model)) else {}
    if params.get("kind") != "trees" or not params.get("gate"):
        raise ValueError("version {} does not read a gated tree model".format(version))
    return model, params


def gate_flags(root, params):
    """(flagged, speak) bool arrays for a root, as team_model.tree_probs computes them."""
    from pipeline.models import player_model as PM
    from pipeline.models import rule_scorer as RS
    from pipeline.models import team_model as TMOD
    data = RS.read_root(root, need_dead=False)
    meta = data["meta"]
    fps = float(meta["fps"])
    bparams = TMOD.load_model(params["base"])
    spec = TMOD.spec_for(bparams, root)
    _dead, team, _info = RS.compute_terms(data)
    X = TMOD.design(spec["columns"], team, data["features"], fps)
    z = X @ np.array([spec["weights"][c] for c in spec["columns"]])
    feats = data["features"]
    flags = PM.gate_flags(feats, data["events"], RS.sigmoid(z), fps)[params["gate"]]
    return np.asarray(flags, dtype=bool), PM.gate_weight(flags, fps) > 0


def differ_flags(root, model, base, n):
    """bool (n,): stored team probability of model differs from base's (inputs.attach_gate)."""
    from pipeline.decode import evidence_cost, inputs
    p, _ = inputs.clean_probs(inputs.load_probs(root, model, n))
    ref, _ = inputs.clean_probs(inputs.load_probs(root, base, n))
    return evidence_cost.gate_from_reference(p, ref)


def root_counts(root, version, truth=False):
    """Counts (not shares) for one root: frames in each denominator and flags inside them."""
    from pipeline.check_labels import read_codes
    from pipeline.decode import inputs
    model, params = tree_params(version)
    meta = inputs.read_meta(root)
    n = int(meta["n_frames"])
    graded, _spans = inputs.graded_mask(meta, n)
    flagged, speak = gate_flags(root, params)
    if len(flagged) != n:
        raise ValueError("gate length differs from the root's frame count")
    flags = {"flagged": flagged, "speak": speak, "differ": differ_flags(root, model,
                                                                       params["base"], n)}
    dens = {"graded": graded}
    lab = os.path.join(root, "labels", "{}.csv".format(version))
    if os.path.isfile(lab):
        codes = read_codes(lab)
        if len(codes) == n:
            dens["label_live"] = graded & np.isin(codes, (H, A))
    if truth:
        from pipeline.decode.__main__ import truth_codes
        gt = truth_codes(root, "truth_labels.csv")
        dens["truth_live"] = (gt == H) | (gt == A)
    out = {"differ_outside_speak": int(np.sum(flags["differ"] & ~flags["speak"] & graded))}
    for d, mask in dens.items():
        out["n_" + d] = int(mask.sum())
        for f, flag in flags.items():
            out["{}_{}".format(f, d)] = int(np.sum(flag & mask))
    return model, params, out


def shares(counts):
    """Shares from (pooled) counts: <flag>_share_<denominator>."""
    out = {}
    for key, value in counts.items():
        for f in FLAGS:
            if key.startswith(f + "_"):
                d = key[len(f) + 1:]
                if d.startswith("outside"):
                    continue
                den = counts.get("n_" + d, 0)
                out["{}_share_{}".format(f, d)] = round(value / den, ROUND) if den else None
    return out


def add(total, counts):
    for k, v in counts.items():
        total[k] = total.get(k, 0) + v
    return total


@match_entry
def run_match(version):
    quiet_warnings()
    model, params, counts = root_counts(match_root(), version, truth=False)
    summary(_name="gate_share", version=version, model=model, gate=params["gate"],
            base=params["base"], label_file_used="n_label_live" in counts,
            differ_outside_speak=counts["differ_outside_speak"], **shares(counts))
    return 0


def run_bench(version, levels, games=GAMES, roots=None):
    groups = [("roots", roots)] if roots else \
        [(lv, [os.path.join(ARTIFACTS, "bench", "{}_{}".format(g, lv)) for g in games])
         for lv in levels]
    for label, group in groups:
        total = {}
        for root in group:
            root = require_bench_root(root)
            has_truth = os.path.isfile(os.path.join(root, "truth_labels.csv"))
            model, params, counts = root_counts(root, version, truth=has_truth)
            add(total, counts)
            summary(_name="gate_share.bench.{}".format(os.path.basename(os.path.normpath(root))),
                    version=version, model=model, gate=params["gate"],
                    differ_outside_speak=counts["differ_outside_speak"], **shares(counts))
        summary(_name="gate_share.bench.{}".format(label), version=version, roots=len(group),
                differ_outside_speak=total.get("differ_outside_speak", 0), **shares(total))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="gate share of a gated tree version (aggregates)")
    ap.add_argument("target", choices=("match", "bench"))
    ap.add_argument("--version", required=True)
    ap.add_argument("--levels", nargs="+", default=list(LEVELS))
    ap.add_argument("--game", nargs="*", default=list(GAMES))
    ap.add_argument("--root", nargs="*", help="bench: explicit roots")
    args = ap.parse_args(argv)
    from pipeline.decode import decoders
    if args.version not in decoders.VERSIONS:
        ap.error("unknown version")
    if args.target == "match":
        if args.root:
            ap.error("the match takes only --version")
        return run_match(args.version)
    return run_bench(args.version, args.levels, args.game, args.root)


if __name__ == "__main__":
    sys.exit(main())
