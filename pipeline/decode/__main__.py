"""Decode stage: probabilities in, whole possessions out (CONTRACTS.md, Labels).

    uv run python -m pipeline.decode match --version v2 [--probs rule_v1]
    make match STAGE=decode VERSION=v2
    uv run python -m pipeline.decode bench --version v2 [--probs rule_v1] [--level match]
                                     [--game metrica_g1 ...] [--root DIR ...]
    make bench STAGE=decode VERSION=v2

Reads ``<root>/probs/<model>.npy`` plus meta.json, features_dead.parquet, restarts.parquet and
events.parquet when present; writes ``<root>/labels/<name>.csv`` (frame,label, every frame).
<name> is the version when no --probs is given (the version's own model, VERSIONS[v]["probs"],
default rule_v1); with an explicit --probs it is the version for rule_v1, else
``<version>_<model>``, so labels from stand-in evidence are never mistaken for the version.

Bench only: ``--probs synth_<profile>`` decodes stand-in probabilities made from the root's
truth (pipeline.decode.synth), with the synthetic restart hints instead of the root's evidence.

The output checks run on every output (pipeline.decode.checks). On the match a failing check
stops the stage before anything is written, and only aggregates are printed (summary()).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

from pipeline.decode import checks, decoders, inputs
from pipeline.io.roots import require_bench_roots
from pipeline.report import match_entry, quiet_warnings, summary

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH = os.path.join(REPO, "artifacts", "bench")
DEFAULT_MODEL = "rule_v1"
GAMES = ("metrica_g1", "metrica_g2", "metrica_g3")
LETTERS = np.array(["H", "A", "D", "X"])


def version_model(version):
    """The probabilities model a version is meant for (VERSIONS[v]["probs"], default rule_v1)."""
    return decoders.VERSIONS[version].get("probs", DEFAULT_MODEL)


def label_name(version, model, explicit=True):
    """Labels file stem. Without an explicit --probs the model is the version's own and the
    labels are <version>.csv; an explicit --probs keeps the old naming (<version> for rule_v1,
    else <version>_<model>)."""
    if not explicit or model == DEFAULT_MODEL:
        return version
    return "{}_{}".format(version, model)


def write_labels(root, name, codes):
    out = os.path.join(root, "labels")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "{}.csv".format(name))
    pd.DataFrame({"frame": np.arange(len(codes)), "label": LETTERS[codes]}).to_csv(
        path, index=False, lineterminator="\n")
    return path


def truth_codes(root, truth="truth.csv"):
    lab = pd.read_csv(os.path.join(root, truth), dtype={"label": str})["label"].to_numpy()
    return np.select([lab == c for c in "HADX"], [0, 1, 2, 3], -1).astype(np.int8)


def bench_inputs(root, model, compute="kickoff"):
    """DecodeInputs for a bench root, from the model's file or from synthetic evidence."""
    meta = inputs.read_meta(root)
    n = int(meta["n_frames"])
    if model.startswith("synth_"):
        from pipeline.decode import synth
        probs, hints = synth.for_root(root, truth_codes(root), model[len("synth_"):])
        return inputs.build(root, probs, restarts=hints, use_root_evidence=False)
    return inputs.build(root, inputs.load_probs(root, model, n), compute=compute)


def attach_gate(inp, root, version):
    """T-42: versions that name a "gate_ref" model read its probabilities for the gate."""
    ref = decoders.VERSIONS[version].get("gate_ref")
    if ref:
        inputs.attach_gate(inp, root, ref)
    return inp


def run(inp, version, overrides=None):
    codes, hidden = decoders.decode(version, inp, overrides)
    failures, warns, agg = checks.check(codes, inp, hidden)
    return codes, failures, warns, agg


@match_entry
def run_match(version, model, explicit=True):
    quiet_warnings()
    from pipeline.io.sources import match_root
    root = match_root()
    meta = inputs.read_meta(root)
    inp = inputs.build(root, inputs.load_probs(root, model, int(meta["n_frames"])))
    attach_gate(inp, root, version)
    codes, failures, warns, agg = run(inp, version)
    name = label_name(version, model, explicit)
    if not failures:
        write_labels(root, name, codes)
    # whole-match rates and counts only: no single-instance values (shortest possession or
    # stoppage) on the match (T-16 audit)
    keep = ("possessions_per_min", "share_d", "stoppages", "stoppages_per_90",
            "median_possession_s", "share_possessions_under_1_5s",
            "share_possessions_under_4s", "median_stoppage_s",
            "possession_length_hist", "hidden_path_violations", "law_restarts_checked",
            "law_restarts_wrong_team", "stoppages_under_s10_min", "graded_minutes")
    summary(_name="decode", version=version, probs=model, labels_written=not failures,
            checks_failed=len(failures), warnings=len(warns),
            hist_edges_s=list(checks.HIST_EDGES_S[:-1]),
            inputs={k: v for k, v in inp.info.items()},
            **{k: agg.get(k) for k in keep})
    if failures:
        # failure messages carry counts only
        summary(_name="decode", failure_1=failures[0][:80])
        return 1
    return 0


def run_bench(version, model, roots, explicit=True):
    name = label_name(version, model, explicit)
    for root in roots:
        inp = bench_inputs(root, model, decoders.VERSIONS[version].get("compute", "kickoff"))
        if not model.startswith("synth_"):
            attach_gate(inp, root, version)
        codes, failures, warns, agg = run(inp, version)
        write_labels(root, name, codes)
        tag = os.path.basename(os.path.normpath(root))
        summary(_name="decode.bench.{}".format(tag), version=name, checks_failed=len(failures),
                failures="; ".join(failures)[:80], warnings="; ".join(warns)[:80],
                possessions_per_min=agg.get("possessions_per_min"), share_d=agg.get("share_d"),
                stoppages=agg.get("stoppages"),
                median_possession_s=agg.get("median_possession_s"),
                shortest_possession_s=agg.get("shortest_possession_s"),
                law_restarts_wrong_team=agg.get("law_restarts_wrong_team"),
                hidden_path_violations=agg.get("hidden_path_violations"))
    summary(_name="decode.bench", version=name, roots=len(roots))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Decode stage (CONTRACTS.md, Labels)")
    ap.add_argument("target", choices=("match", "bench"))
    ap.add_argument("--version", required=True, choices=sorted(decoders.VERSIONS))
    ap.add_argument("--probs", default=None,
                    help="probability model under <root>/probs/ (bench also: synth_<profile>); "
                         "default: the version's own (VERSIONS[v]['probs'], else rule_v1)")
    ap.add_argument("--level", default="match", help="bench level (roots <game>_<level>)")
    ap.add_argument("--game", nargs="*", default=list(GAMES))
    ap.add_argument("--root", nargs="*", help="explicit bench roots")
    args = ap.parse_args(argv)
    explicit = args.probs is not None
    if not explicit:
        args.probs = version_model(args.version)
    if args.target == "match":
        if args.root or args.probs.startswith("synth_"):
            ap.error("the match takes only --version and a probability model")
        return run_match(args.version, args.probs, explicit)
    roots = args.root or [os.path.join(BENCH, "{}_{}".format(g, args.level)) for g in args.game]
    require_bench_roots(roots)
    return run_bench(args.version, args.probs, roots, explicit)


if __name__ == "__main__":
    sys.exit(main())
