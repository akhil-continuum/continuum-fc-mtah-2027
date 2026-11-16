"""Build the submission file from a version's match labels.

    make submission VERSION=v2 [TEAM=<team_name>]
    uv run python -m pipeline.submit v2 --team <team_name>

Reads ``artifacts/match/labels/<version>.csv`` (submission format, written by the decoder), writes
``submissions/<team_name>.csv`` with exactly games.source_frames() rows, frames 0..n-1 in order,
labels H/A/D on graded frames and ``X`` on every frame outside games.half_video_blocks(). A graded
frame that is not H, A or D is an error: the team's rule is that every graded frame is decided.

Freshness gate: the labels must carry a current provenance stamp (pipeline.io.provenance), written
by ``make candidate`` after the full match chain ran on this code. Labels without a stamp, labels
rewritten after stamping, labels older than the code (code hash changed), and labels whose stamp
names another probabilities model or touch parameter set than the version declares (T-36:
``probs_differs_from_version``, ``touch_set_differs_from_version``) are refused, with the reason;
run ``make candidate VERSION=<v> PART=match`` and try again.

Prints through pipeline.report.summary only (row count, label counts on graded frames, the
submission's sha256). The validator is run afterwards by the make target, as its own command.
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
import games  # noqa: E402

from pipeline.io import frames, provenance  # noqa: E402
from pipeline.report import match_entry, quiet_warnings, summary  # noqa: E402

LABELS_DIR = os.path.join(REPO, "artifacts", "match", "labels")
OUT_DIR = os.path.join(REPO, "submissions")


class StaleLabels(RuntimeError):
    """The labels file was not made by the full chain on the current code."""


def version_expect(version):
    """The version's stamped properties (probs model, touch set) as the code declares them now;
    None for a name the decoder does not know (nothing to compare)."""
    from pipeline import candidate
    from pipeline.decode import decoders
    if version not in decoders.VERSIONS:
        return None
    return candidate.stamp_extra(version)


def require_fresh(labels_path, repo=REPO, version=None):
    expect = version_expect(version) if version is not None else None
    ok, reason = provenance.check_stamp(labels_path, repo, expect=expect)
    if not ok:
        raise StaleLabels("labels refused ({}); run: make candidate VERSION=<v> PART=match".format(
            reason))
    return reason


def build(version, team, labels_dir=LABELS_DIR, out_dir=OUT_DIR, game=games.GAME,
          check_fresh=True):
    import numpy as np
    import pandas as pd

    n = games.source_frames("tactical", game)
    labels_path = os.path.join(labels_dir, "{}.csv".format(version))
    if check_fresh:
        require_fresh(labels_path, version=version)
    table = pd.read_csv(labels_path, dtype={"label": str})
    if list(table.columns[:2]) != ["frame", "label"]:
        raise ValueError("labels file must have columns frame,label")
    if len(table) != n or not np.array_equal(table["frame"].to_numpy(), np.arange(n)):
        raise ValueError("labels file must have {} rows, frames 0..{} in order".format(n, n - 1))
    labels = table["label"].fillna("X").str.strip().str.upper().to_numpy().astype("<U1")
    graded = frames.half_index(n, frames.match_halves(game)) > 0
    undecided = int((~np.isin(labels[graded], ["H", "A", "D"])).sum())
    if undecided:
        raise ValueError("{} graded frames are not H, A or D".format(undecided))
    labels[~graded] = "X"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "{}.csv".format(team))
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    counts = {k: int(np.sum(labels[graded] == k)) for k in ("H", "A", "D")}
    return {"rows": int(n), "graded_frames": int(graded.sum()), "graded_counts": counts,
            "ungraded_x": int((~graded).sum()),
            "labels_provenance": "current" if check_fresh else "not checked",
            "submission_sha256": provenance.file_sha256(path)}


@match_entry
def main(argv=None):
    quiet_warnings()
    parser = argparse.ArgumentParser(description="build submissions/<team>.csv")
    parser.add_argument("version")
    parser.add_argument("--team", default="team")
    args = parser.parse_args(argv)
    summary(_name="submit", version=args.version, team=args.team,
            **build(args.version, args.team))
    return 0


if __name__ == "__main__":
    sys.exit(main())
