"""Run a version end to end: every stage in order, on the match and on the bench level roots.

    make candidate VERSION=v2s [LEVELS="low match high"] [PART=all|match|bench]
    uv run python -m pipeline.candidate all   --version v2s
    uv run python -m pipeline.candidate match --version v2s
    uv run python -m pipeline.candidate bench --version v2s [--levels low match high]
    uv run python -m pipeline.candidate stages [--touch-set w15s] [--levels ...]
        (make bench-stages: the bench stage outputs for a named touch set; no decode, no scores)

Match chain (each stage its own process, the same commands as ``make match STAGE=<stage>``):
    load -> ball -> events -> features -> restarts -> probs (models.rule_scorer) ->
    probs2 (models.team_model: rule_v2 and every later team model with a params file) -> decode
probs2 always runs (cheap; it never touches rule_v1.npy, so versions reading rule_v1 are
unaffected, and each team model's file is computed on its own, so rule_v2.npy is unchanged);
the decode stage reads the version's own model, VERSIONS[v].get("probs", "rule_v1").
After the decode writes ``artifacts/match/labels/<version>.csv``, a provenance stamp
(pipeline.io.provenance) is written next to it; ``make submission`` refuses labels without a
current stamp. Every stage prints aggregates only, through summary().

Touch parameter set (T-36). The events stage runs with the version's own touch set,
``--touch-set <set>``, on the match and on the bench level roots alike. A version declares it in
its entry in pipeline/decode/decoders.py, next to "probs":

    VERSIONS["v9s"] = {..., "probs": "rule_v2", "touch_set": "w15s"}

Without the key the set is pipeline.events.touch.DEFAULT_SET ("w7"), which every version before
T-35 was built with, so their labels do not change. An unknown set name stops the run before any
stage. Everything downstream of events (features, restarts, probs, probs2, decode) is rebuilt in
the same run, in order, so no stage reuses outputs built from another set; the stamp records
``touch_set`` and ``make submission`` refuses a stamp whose set differs from the version's.
A run of a w15s version leaves w15s-built features and probs on the level roots (and on the
match root): decode another version only through its own ``make candidate`` run.

Bench chain, on the level roots ``artifacts/bench/metrica_g{1,2,3}_<level>`` (inputs from
``make bench-build``): ball -> events -> features -> probs on those roots, then decode and
bench.score (default and confirmed truth) for every level, then one table of pooled scores.
The restarts stage is not run on level roots: the decoder and the rule scorer compute it in
memory there (the stage writes only into its own roots).

A failing stage stops the run with its name and exit code.
"""
import argparse
import json
import os
import subprocess
import sys
import time

from pipeline.io import provenance
from pipeline.io.sources import ARTIFACTS, REPO, match_root
from pipeline.report import match_entry

GAMES = ("metrica_g1", "metrica_g2", "metrica_g3")
LEVELS = ("low", "match", "high")
TRUTHS = ("default", "confirmed")
BOOTSTRAP = 1000


class StageFailed(RuntimeError):
    pass


def probs_model(version):
    """The probabilities model the version's labels are decoded from (the decode stage default)."""
    from pipeline.decode import decoders
    return decoders.VERSIONS[version].get("probs", "rule_v1")


def touch_set(version):
    """The touch parameter set the version's events are built with: VERSIONS[v]["touch_set"],
    default pipeline.events.touch.DEFAULT_SET. An unknown name raises StageFailed."""
    from pipeline.decode import decoders
    from pipeline.events import touch
    name = decoders.VERSIONS[version].get("touch_set", touch.DEFAULT_SET)
    try:
        return check_touch_set(name)
    except StageFailed as error:
        raise StageFailed("version {}: {}".format(version, error))


def stamp_extra(version):
    """Version properties recorded in the provenance stamp; make submission checks them."""
    return {"probs": probs_model(version), "touch_set": touch_set(version)}


def team_models():
    """Team models the probs2 step writes: every one with a tracked params file (rule_v2 first;
    each model's file is computed independently, so adding models leaves rule_v2.npy unchanged)."""
    from pipeline.models import team_model
    return team_model.available_models()


def match_steps(version):
    return [("load", ["pipeline.load", "match"]),
            ("ball", ["pipeline.ball", "match"]),
            ("events", ["pipeline.events", "match", "--touch-set", touch_set(version)]),
            ("features", ["pipeline.features", "match"]),
            ("restarts", ["pipeline.restarts", "match"]),
            ("probs", ["pipeline.models.rule_scorer", "match"]),
            ("probs2", ["pipeline.models.team_model", "match", "--models"] + team_models()),
            ("decode", ["pipeline.decode", "match", "--version", version])]


def level_roots(levels, games=GAMES):
    return [os.path.join(ARTIFACTS, "bench", "{}_{}".format(g, lv)) for lv in levels for g in games]


def check_touch_set(name):
    from pipeline.events import touch
    if name not in touch.PARAM_SETS:
        raise StageFailed("unknown touch set {!r}; known: {}".format(
            name, ", ".join(sorted(touch.PARAM_SETS))))
    return name


def bench_stage_steps(levels, version=None, tset=None):
    """Stage steps on the level roots. The events step gets ``tset`` if given (make
    bench-stages), else the version's touch set, else touch.DEFAULT_SET."""
    roots = level_roots(levels)
    if tset is not None:
        tset = check_touch_set(tset)
    elif version is not None:
        tset = touch_set(version)
    else:
        from pipeline.events import touch
        tset = touch.DEFAULT_SET
    steps = [("bench " + name, [module, "bench", "--root"] + roots)
             for name, module in (("ball", "pipeline.ball"), ("events", "pipeline.events"),
                                  ("features", "pipeline.features"),
                                  ("probs", "pipeline.models.rule_scorer"),
                                  ("probs2", "pipeline.models.team_model"))]
    steps[1][1].extend(["--touch-set", tset])
    steps[-1][1].extend(["--models"] + team_models())
    return steps


def bench_decode_steps(version, levels, truths=TRUTHS, bootstrap=BOOTSTRAP):
    steps = []
    for lv in levels:
        steps.append(("bench decode " + lv,
                      ["pipeline.decode", "bench", "--version", version, "--level", lv]))
    for lv in levels:
        for truth in truths:
            steps.append(("bench score {} {}".format(lv, truth),
                          ["bench.score", "--version", version, "--level", lv, "--truth", truth,
                           "--bootstrap", str(bootstrap)]))
    return steps


def run_step(name, module_args, quiet=False):
    """One stage as its own process (python -m ...), from the repository root."""
    start = time.time()
    print("== candidate: {} ...".format(name), flush=True)
    result = subprocess.run([sys.executable, "-m"] + module_args, cwd=REPO,
                            stdout=subprocess.DEVNULL if quiet else None)
    if result.returncode != 0:
        raise StageFailed("stage '{}' failed with exit code {}".format(name, result.returncode))
    print("== candidate: {} done in {:.0f} s".format(name, time.time() - start), flush=True)


def run_match(version):
    print("== candidate: version {} touch set {}".format(version, touch_set(version)), flush=True)
    for name, args in match_steps(version):
        run_step(name, args)
    labels = os.path.join(match_root(), "labels", "{}.csv".format(version))
    stamp = provenance.write_stamp(labels, version, **stamp_extra(version))
    # hashes and names only; no content
    print("== candidate: match labels stamped (code files {}, code hash {}, touch set {})".format(
        stamp["code_files"], stamp["code_hash"][:12], stamp["touch_set"]), flush=True)


def bench_inputs_missing(levels):
    need = ("players.parquet", "ball_raw.parquet", "meta.json", "truth.csv")
    return [r for r in level_roots(levels)
            if not all(os.path.isfile(os.path.join(r, f)) for f in need)]


def score_table(version, levels, truths=TRUTHS):
    """Pooled bench scores from artifacts/bench/_scores/<v>_<level>_<truth>.json (bench only)."""
    cols = ("spell_f1_mean", "spell_bootstrap_low", "spell_bootstrap_high", "short_recall_4",
            "false_splits_per_hour", "missed_stoppages", "invented_stoppages",
            "restart_team_acc", "pred_poss_per_min")
    print("bench scores, version {} (pooled over {} games)".format(version, len(GAMES)))
    print("{:<8}{:<10}".format("level", "truth") + "".join("{:>12}".format(c[:11]) for c in cols))
    for lv in levels:
        for truth in truths:
            path = os.path.join(ARTIFACTS, "bench", "_scores",
                                "{}_{}_{}.json".format(version, lv, truth))
            if not os.path.isfile(path):
                print("{:<8}{:<10}  (no score file)".format(lv, truth))
                continue
            with open(path) as handle:
                pooled = json.load(handle).get("pooled", {})
            cells = []
            for c in cols:
                v = pooled.get(c)
                cells.append("{:>12}".format("-" if v is None else
                                             ("{:.3f}".format(v) if isinstance(v, float) else v)))
            print("{:<8}{:<10}".format(lv, truth) + "".join(cells))


def run_bench(version, levels):
    missing = bench_inputs_missing(levels)
    if missing:
        raise StageFailed("{} bench level roots lack their inputs: run make bench-build".format(
            len(missing)))
    print("== candidate: version {} touch set {}".format(version, touch_set(version)), flush=True)
    for name, args in bench_stage_steps(levels, version):
        run_step(name, args, quiet=True)
    for name, args in bench_decode_steps(version, levels):
        run_step(name, args, quiet=True)
    score_table(version, levels)


def run_stages(levels, tset):
    """make bench-stages: the bench stage steps for a named touch set, no decode, no scores."""
    missing = bench_inputs_missing(levels)
    if missing:
        raise StageFailed("{} bench level roots lack their inputs: run make bench-build".format(
            len(missing)))
    steps = bench_stage_steps(levels, tset=tset)
    print("== candidate: bench stages, touch set {}".format(steps[1][1][-1]), flush=True)
    for name, args in steps:
        run_step(name, args, quiet=True)


@match_entry
def main(argv=None):
    ap = argparse.ArgumentParser(description="run a version end to end")
    ap.add_argument("part", choices=("all", "match", "bench", "stages"),
                    help="stages: bench stage outputs only, for --touch-set (make bench-stages)")
    ap.add_argument("--version")
    ap.add_argument("--levels", nargs="+", default=list(LEVELS))
    ap.add_argument("--touch-set", default=None,
                    help="part stages only; a version's own set comes from its VERSIONS entry")
    args = ap.parse_args(argv)
    if args.part == "stages":
        if args.version:
            ap.error("part stages takes --touch-set, not --version")
        try:
            run_stages(args.levels, args.touch_set)
        except StageFailed as error:
            sys.stderr.write("candidate: {}\n".format(error))
            return 1
        return 0
    if args.touch_set:
        ap.error("--touch-set is for part stages; a version's set is declared in its VERSIONS "
                 "entry (pipeline/decode/decoders.py)")
    if not args.version:
        ap.error("--version is required")
    from pipeline.decode import decoders
    if args.version not in decoders.VERSIONS:
        ap.error("unknown version; the decoder knows {}".format(", ".join(sorted(decoders.VERSIONS))))
    try:
        touch_set(args.version)          # an unknown set stops the run before any stage
        if args.part in ("all", "bench"):
            run_bench(args.version, args.levels)
        if args.part in ("all", "match"):
            run_match(args.version)
    except StageFailed as error:
        sys.stderr.write("candidate: {}\n".format(error))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
