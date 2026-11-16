"""Rebuild the submission from a clean copy of the code and compare bytes (RULES.md rule 3).

    make rebuild-check VERSION=v2s TEAM=<name> [SOURCE=head|worktree] [BENCH=1] [REPEAT=1] [KEEP=1]
    uv run python -m pipeline.rebuild_check --version v2s --team <name> [--source head] ...

1. Clone this repository into a fresh temporary folder outside it (``git clone`` then a detached
   checkout of HEAD). ``--source head`` (default) uses the committed code only; ``--source
   worktree`` also copies the uncommitted edits and new files of this checkout (git's view:
   tracked and untracked, never ignored files) and says how many, for checking work before the
   lead commits it. The clone has its own, empty ``artifacts/`` and its own uv environment
   built from ``uv.lock``.
2. The match data is found through games.py: ``MTAH_DATA_DIR`` is set to the folder games.py
   resolves here, so the clone reads the same files. ``--bench`` links ``bench_data/`` (open
   Metrica data) and rebuilds the bench too (``make bench-build`` and the bench half of the
   candidate), then reports how many bench files came out identical.
3. In the clone: ``make candidate PART=match VERSION=<v>``, ``make submission VERSION=<v>
   TEAM=<name>``. Their output (aggregates only) goes to a log in the temporary folder, not here.
   The version's touch parameter set and probabilities model come from its VERSIONS entry in the
   cloned code (T-36), so the clone builds the events with the same set; the clone's submission
   step refuses a stamp whose set differs. ``--bench`` builds the default (w7) bench stages with
   ``make bench-build`` and then the version's own set with the bench half of the candidate.
4. Prints MATCH or MISMATCH with the sha256 of ``submissions/<name>.csv`` here and in the clone,
   then, per match stage file, ``same`` / ``differs`` / ``missing`` (names only), and when the
   version's labels differ the number of differing frames (one number). ``--repeat`` (automatic
   on a mismatch when both trees have the same code hash) runs the match chain a second time in
   the clone and names every stage file that changed between two runs of the same code: those
   stages are not deterministic.

Nothing from the match is printed except those hashes and counts. Exit code 0 on MATCH.
"""
import argparse
import filecmp
import os
import shutil
import subprocess
import sys
import tempfile
import time

from pipeline.io import provenance
from pipeline.io.sources import REPO
from pipeline.report import match_entry, quiet_warnings

MATCH_FILES = ("players.parquet", "ball_raw.parquet", "ball.parquet", "events.parquet",
               "features.parquet", "restarts.parquet", "features_dead.parquet",
               "probs/rule_v1.npy")
BENCH_FILES = ("ball.parquet", "events.parquet", "features.parquet", "probs/rule_v1.npy")
LEVELS = ("low", "match", "high")
GAMES = ("metrica_g1", "metrica_g2", "metrica_g3")


class RebuildFailed(RuntimeError):
    pass


def git(*args, cwd=REPO):
    return subprocess.run(["git", "-C", cwd] + list(args), check=True,
                          capture_output=True).stdout.decode()


def clone(dest, source):
    head = git("rev-parse", "HEAD").strip()
    subprocess.run(["git", "clone", "--quiet", "--no-checkout", REPO, dest], check=True,
                   capture_output=True)
    git("checkout", "--quiet", "--detach", head, cwd=dest)
    overlaid = removed = 0
    if source == "worktree":
        for rel in git("ls-files", "-co", "--exclude-standard", "-z").split("\0"):
            src = os.path.join(REPO, rel)
            if not rel or not os.path.isfile(src):
                continue
            dst = os.path.join(dest, rel)
            if os.path.isfile(dst) and filecmp.cmp(src, dst, shallow=False):
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            overlaid += 1
        for rel in git("ls-files", "-d", "-z").split("\0"):
            if rel and os.path.isfile(os.path.join(dest, rel)):
                os.remove(os.path.join(dest, rel))
                removed += 1
    return head, overlaid, removed


def clone_env():
    import games
    env = dict(os.environ)
    env["MTAH_DATA_DIR"] = os.path.realpath(games.DATA_DIR)
    for key in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "MAKEFLAGS", "MFLAGS", "MAKELEVEL"):
        env.pop(key, None)
    return env


def make(cwd, log, env, *args):
    start = time.time()
    label = " ".join(args)
    print("rebuild-check: make {} ...".format(label), flush=True)
    with open(log, "a") as handle:
        handle.write("\n### make {}\n".format(label))
        handle.flush()
        result = subprocess.run(["make", "--no-print-directory"] + list(args), cwd=cwd, env=env,
                                stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RebuildFailed("make {} failed in the clone (exit code {})".format(
            label, result.returncode))
    print("rebuild-check: make {} done in {:.0f} s".format(label, time.time() - start), flush=True)


def sha(path):
    return provenance.file_sha256(path) if os.path.isfile(path) else "missing"


def same(a, b):
    if not (os.path.isfile(a) and os.path.isfile(b)):
        return "missing"
    return "same" if filecmp.cmp(a, b, shallow=False) else "differs"


def differing_frames(a, b):
    """Number of frames whose label differs (one number), or None when lengths differ."""
    import pandas as pd
    la = pd.read_csv(a, dtype={"label": str})["label"].to_numpy()
    lb = pd.read_csv(b, dtype={"label": str})["label"].to_numpy()
    if len(la) != len(lb):
        return None
    return int((la != lb).sum())


def version_files(version):
    """The version's own files: its probabilities model (when not rule_v1) and its labels."""
    from pipeline import candidate
    model = candidate.probs_model(version)
    own = () if model == "rule_v1" else ("probs/{}.npy".format(model),)
    return own + ("labels/{}.csv".format(version),)


def compare_match(root_a, root_b, version, tag):
    files = MATCH_FILES + version_files(version)
    status = {f: same(os.path.join(root_a, f), os.path.join(root_b, f)) for f in files}
    print("rebuild-check: {}: ".format(tag) + ", ".join(
        "{} {}".format(f, s) for f, s in status.items()))
    return status


def compare_bench(clone_dir, version):
    rows = []
    for lv in LEVELS:
        for g in GAMES:
            name = "{}_{}".format(g, lv)
            for f in BENCH_FILES + version_files(version):
                a = os.path.join(REPO, "artifacts", "bench", name, f)
                b = os.path.join(clone_dir, "artifacts", "bench", name, f)
                rows.append((name, f, same(a, b)))
    for lv in LEVELS:
        for truth in ("default", "confirmed"):
            f = "{}_{}_{}.json".format(version, lv, truth)
            rows.append(("_scores", f, same(os.path.join(REPO, "artifacts", "bench", "_scores", f),
                                            os.path.join(clone_dir, "artifacts", "bench",
                                                         "_scores", f))))
    counts = {}
    for _, _, s in rows:
        counts[s] = counts.get(s, 0) + 1
    print("rebuild-check: bench files vs this checkout: {}".format(counts))
    for name, f, s in rows:
        if s != "same":
            print("rebuild-check:   bench {} {}: {}".format(name, f, s))
    return counts


@match_entry
def run(version, team, source="head", bench=False, repeat=False, keep=False, workdir=None):
    quiet_warnings()
    reference = os.path.join(REPO, "submissions", "{}.csv".format(team))
    if not os.path.isfile(reference):
        print("rebuild-check: no reference submission here (run make submission first); "
              "the rebuild and the stage comparison still run")
    base = tempfile.mkdtemp(prefix="mtah-rebuild-", dir=workdir)
    if os.path.realpath(base).startswith(os.path.realpath(REPO) + os.sep):
        raise RebuildFailed("the work folder must be outside the repository")
    dest = os.path.join(base, "repo")
    log = os.path.join(base, "rebuild.log")
    try:
        head, overlaid, removed = clone(dest, source)
        print("rebuild-check: clone of {} at {} (source {}; uncommitted files copied {}, "
              "deleted {})".format("this repository", head[:12], source, overlaid, removed))
        here_hash, clone_hash = provenance.code_hash(REPO)[0], provenance.code_hash(dest)[0]
        print("rebuild-check: code hash here {} clone {} ({})".format(
            here_hash[:12], clone_hash[:12], "same" if here_hash == clone_hash else "different"))
        from pipeline import candidate
        print("rebuild-check: version {} touch set {} probs {} (the clone's make candidate "
              "reads the same declaration)".format(version, candidate.touch_set(version),
                                                    candidate.probs_model(version)))
        env = clone_env()
        if bench:
            os.symlink(os.path.join(REPO, "bench_data"), os.path.join(dest, "bench_data"))
            make(dest, log, env, "bench-build")
            make(dest, log, env, "candidate", "PART=bench", "VERSION=" + version)
        make(dest, log, env, "candidate", "PART=match", "VERSION=" + version)
        make(dest, log, env, "submission", "VERSION=" + version, "TEAM=" + team)

        rebuilt = os.path.join(dest, "submissions", "{}.csv".format(team))
        ref_sha, new_sha = sha(reference), sha(rebuilt)
        verdict = "MATCH" if ref_sha == new_sha != "missing" else "MISMATCH"
        print("rebuild-check: {}".format(verdict))
        print("rebuild-check: reference sha256 {}".format(ref_sha))
        print("rebuild-check: rebuilt   sha256 {}".format(new_sha))

        here_root = os.path.join(REPO, "artifacts", "match")
        clone_root = os.path.join(dest, "artifacts", "match")
        status = compare_match(here_root, clone_root, version, "match stage files, here vs clone")
        lab = "labels/{}.csv".format(version)
        if status.get(lab) == "differs":
            n = differing_frames(os.path.join(here_root, lab), os.path.join(clone_root, lab))
            print("rebuild-check: labels {} differing frames: {}".format(
                version, "row counts differ" if n is None else n))
        if bench:
            compare_bench(dest, version)

        if repeat or (verdict == "MISMATCH" and here_hash == clone_hash):
            first = os.path.join(base, "run1_match")
            shutil.copytree(clone_root, first)
            make(dest, log, env, "candidate", "PART=match", "VERSION=" + version)
            status2 = compare_match(first, clone_root, version, "two runs of the same code")
            unstable = [f for f, s in status2.items() if s != "same"]
            print("rebuild-check: stages not deterministic: {}".format(
                ", ".join(unstable) if unstable else "none"))
        return 0 if verdict == "MATCH" else 1
    finally:
        if keep:
            print("rebuild-check: kept work folder (log inside): {}".format(base))
        else:
            shutil.rmtree(base, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="rebuild the submission from a clean clone")
    ap.add_argument("--version", required=True)
    ap.add_argument("--team", required=True)
    ap.add_argument("--source", choices=("head", "worktree"), default="head")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--repeat", action="store_true")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--workdir", help="parent of the temporary folder (outside the repository)")
    args = ap.parse_args(argv)
    return run(args.version, args.team, args.source, args.bench, args.repeat, args.keep,
               args.workdir or None)


if __name__ == "__main__":
    sys.exit(main())
