"""T-36: the touch parameter set is a property of a version in the build chain.

A version declares its set in its VERSIONS entry (pipeline/decode/decoders.py, key "touch_set",
default pipeline.events.touch.DEFAULT_SET); make candidate passes it to the events stage on the
match and the bench level roots, rebuilds every later stage in order, and stamps it.

Synthetic inputs only; nothing here reads or writes the match (stage runs and the stamp writer
are replaced by recorders).
"""
import json
import os
import subprocess

import numpy as np
import pandas as pd
import pytest

import games
from pipeline import candidate, rebuild_check, submit
from pipeline.decode import decoders
from pipeline.events import __main__ as events_main
from pipeline.events import touch
from pipeline.io import provenance
from pipeline.io.sources import REPO

# TEST-ONLY version entry: exists only inside these tests (monkeypatched into VERSIONS), so the
# chain can be checked with a w15s version before T-35 names the real ones.
EXAMPLE = "t36_example_w15s"


@pytest.fixture
def example_version(monkeypatch):
    spec = dict(decoders.VERSIONS["v6s"], touch_set="w15s")
    monkeypatch.setitem(decoders.VERSIONS, EXAMPLE, spec)
    return EXAMPLE


@pytest.fixture
def recorder(monkeypatch):
    """Replace stage runs, the bench input check, the score table and the stamp writer."""
    calls = {"steps": [], "stamps": []}
    monkeypatch.setattr(candidate, "run_step",
                        lambda name, args, quiet=False: calls["steps"].append((name, list(args))))
    monkeypatch.setattr(candidate, "bench_inputs_missing", lambda levels: [])
    monkeypatch.setattr(candidate, "score_table", lambda version, levels: None)

    def fake_stamp(labels, version, **extra):
        calls["stamps"].append((os.path.basename(labels), version, extra))
        return dict(extra, code_files=0, code_hash="0" * 64)
    monkeypatch.setattr(candidate.provenance, "write_stamp", fake_stamp)
    return calls


def _events_args(steps):
    found = [args for name, args in steps if name.endswith("events")]
    assert len(found) == 1
    return found[0]


# --- the mapping -----------------------------------------------------------------------------

def test_every_existing_version_is_w7():
    assert touch.DEFAULT_SET == "w7"
    # every version without a declaration (all versions before T-35) builds w7
    undeclared = [v for v, spec in decoders.VERSIONS.items() if "touch_set" not in spec]
    assert {"v2s", "v6s"} <= set(undeclared)
    for version in undeclared:
        assert candidate.touch_set(version) == "w7", version


def test_example_version_declares_w15s(example_version):
    assert candidate.touch_set(example_version) == "w15s"
    assert candidate.stamp_extra(example_version) == {"probs": "rule_v2", "touch_set": "w15s"}
    assert candidate.stamp_extra("v2s") == {"probs": "rule_v1", "touch_set": "w7"}


def test_unknown_set_is_refused(monkeypatch, recorder):
    monkeypatch.setitem(decoders.VERSIONS, "t36_bad", dict(decoders.VERSIONS["v2s"],
                                                           touch_set="w99"))
    with pytest.raises(candidate.StageFailed, match="w99"):
        candidate.touch_set("t36_bad")
    assert candidate.main(["all", "--version", "t36_bad"]) == 1
    assert recorder["steps"] == [] and recorder["stamps"] == []      # stopped before any stage


# --- the pass-through ------------------------------------------------------------------------

def test_match_steps_pass_the_set(example_version):
    for version, expected in (("v2s", "w7"), ("v6s", "w7"), (example_version, "w15s")):
        steps = candidate.match_steps(version)
        assert _events_args(steps) == ["pipeline.events", "match", "--touch-set", expected]
        names = [n for n, _ in steps]
        # everything after events is rebuilt in the same run, in order
        assert names[names.index("events"):] == ["events", "features", "restarts", "probs",
                                                 "probs2", "decode"]


def test_bench_steps_pass_the_set(example_version):
    roots = candidate.level_roots(["match"])
    for version, expected in ((None, "w7"), ("v6s", "w7"), (example_version, "w15s")):
        steps = candidate.bench_stage_steps(["match"], version)
        assert _events_args(steps) == ["pipeline.events", "bench", "--root"] + roots + [
            "--touch-set", expected]
        assert [n for n, _ in steps] == ["bench ball", "bench events", "bench features",
                                         "bench probs", "bench probs2"]
    steps = candidate.bench_stage_steps(["match"], tset="w15s")
    assert _events_args(steps)[-2:] == ["--touch-set", "w15s"]
    with pytest.raises(candidate.StageFailed):
        candidate.bench_stage_steps(["match"], tset="nope")


def test_candidate_all_runs_in_order_and_stamps_the_set(example_version, recorder):
    assert candidate.main(["all", "--version", example_version, "--levels", "match"]) == 0
    names = [n for n, _ in recorder["steps"]]
    assert names[:5] == ["bench ball", "bench events", "bench features", "bench probs",
                         "bench probs2"]
    assert names[5] == "bench decode match"
    assert names[-8:] == ["load", "ball", "events", "features", "restarts", "probs", "probs2",
                          "decode"]
    events = [a for n, a in recorder["steps"] if n in ("events", "bench events")]
    assert all(a[-2:] == ["--touch-set", "w15s"] for a in events) and len(events) == 2
    assert recorder["stamps"] == [("{}.csv".format(example_version), example_version,
                                   {"probs": "rule_v2", "touch_set": "w15s"})]


def test_candidate_existing_version_runs_w7(recorder):
    assert candidate.main(["match", "--version", "v6s"]) == 0
    events = [a for n, a in recorder["steps"] if n == "events"]
    assert events == [["pipeline.events", "match", "--touch-set", "w7"]]
    assert recorder["stamps"][0][2] == {"probs": "rule_v2", "touch_set": "w7"}


def test_stages_part_and_option_rules(recorder):
    assert candidate.main(["stages", "--touch-set", "w15s", "--levels", "low", "match"]) == 0
    assert _events_args(recorder["steps"])[-2:] == ["--touch-set", "w15s"]
    assert not any("decode" in n or "score" in n for n, _ in recorder["steps"])
    recorder["steps"].clear()
    assert candidate.main(["stages"]) == 0
    assert _events_args(recorder["steps"])[-2:] == ["--touch-set", "w7"]
    with pytest.raises(SystemExit):        # a version's set is not overridden on the command line
        candidate.main(["bench", "--version", "v6s", "--touch-set", "w15s"])
    with pytest.raises(SystemExit):
        candidate.main(["stages", "--version", "v6s"])


def test_events_cli_takes_the_set():
    assert events_main._split(["bench", "--root", "a", "b", "--touch-set", "w15s"]) == (
        "w15s", ["bench", "--root", "a", "b"])
    assert events_main._split(["match"]) == (None, ["match"])


# --- the stamp and the submission gate ---------------------------------------------------------

def test_check_stamp_compares_version_properties(tmp_path):
    repo = str(tmp_path / "repo")
    os.makedirs(os.path.join(repo, "pipeline"))
    subprocess.run(["git", "init", "-q", repo], check=True)
    with open(os.path.join(repo, "pipeline", "x.py"), "w") as handle:
        handle.write("A = 1\n")
    labels = tmp_path / "v.csv"
    labels.write_text("frame,label\n0,H\n")
    provenance.write_stamp(str(labels), "v", repo, probs="rule_v1", touch_set="w7")
    with open(provenance.stamp_path(str(labels))) as handle:
        assert json.load(handle)["touch_set"] == "w7"
    ok = {"probs": "rule_v1", "touch_set": "w7"}
    assert provenance.check_stamp(str(labels), repo, expect=ok) == (True, "current")
    assert provenance.check_stamp(str(labels), repo) == (True, "current")
    assert provenance.check_stamp(str(labels), repo, expect=dict(ok, touch_set="w15s")) == (
        False, "touch_set_differs_from_version")
    provenance.write_stamp(str(labels), "v", repo, probs="rule_v1")      # a pre-T-36 stamp
    assert provenance.check_stamp(str(labels), repo, expect=ok) == (
        False, "touch_set_differs_from_version")


def test_version_expect(example_version):
    assert submit.version_expect("v6s") == {"probs": "rule_v2", "touch_set": "w7"}
    assert submit.version_expect(example_version) == {"probs": "rule_v2", "touch_set": "w15s"}
    assert submit.version_expect("not_a_version") is None


def _synthetic_labels(path):
    """Synthetic labels of the right shape: D on graded frames, X elsewhere (no match data)."""
    from pipeline.io import frames
    n = games.source_frames("tactical", games.GAME)
    graded = frames.half_index(n, frames.match_halves(games.GAME)) > 0
    pd.DataFrame({"frame": np.arange(n), "label": np.where(graded, "D", "X")}).to_csv(
        path, index=False)


def test_submit_refuses_a_stamp_with_another_set(tmp_path, example_version):
    labels_dir, out_dir = tmp_path / "labels", tmp_path / "out"
    labels_dir.mkdir()
    path = labels_dir / "{}.csv".format(example_version)
    _synthetic_labels(path)
    provenance.write_stamp(str(path), example_version, probs="rule_v2", touch_set="w7")
    with pytest.raises(submit.StaleLabels, match="touch_set_differs_from_version"):
        submit.build(example_version, "t", labels_dir=str(labels_dir), out_dir=str(out_dir))
    assert not (out_dir / "t.csv").exists()
    provenance.write_stamp(str(path), example_version, **candidate.stamp_extra(example_version))
    stats = submit.build(example_version, "t", labels_dir=str(labels_dir), out_dir=str(out_dir))
    assert stats["labels_provenance"] == "current"


def test_rebuild_check_compares_the_versions_own_files():
    assert rebuild_check.version_files("v2s") == ("labels/v2s.csv",)
    assert rebuild_check.version_files("v6s") == ("probs/rule_v2.npy", "labels/v6s.csv")


# --- the Makefile ------------------------------------------------------------------------------

def _make(*args):
    env = {k: v for k, v in os.environ.items() if k not in ("MAKEFLAGS", "MFLAGS", "MAKELEVEL")}
    return subprocess.run(["make", "--no-print-directory"] + list(args), cwd=REPO, env=env,
                          capture_output=True, text=True)


def test_make_passes_touch_set_only_where_meant():
    out = _make("-n", "bench", "STAGE=events", "TOUCH_SET=w15s", "ROOT=artifacts/bench/x")
    assert out.returncode == 0
    assert "pipeline.events bench" in out.stdout and "--root artifacts/bench/x" in out.stdout
    assert "--touch-set w15s" in out.stdout
    out = _make("-n", "bench", "STAGE=events")
    assert "--touch-set" not in out.stdout                       # the stage default, w7
    out = _make("-n", "bench-stages", "TOUCH_SET=w15s", "LEVELS=match")
    assert "pipeline.candidate stages --levels match --touch-set w15s" in out.stdout
    out = _make("-n", "candidate", "VERSION=v6s", "TOUCH_SET=w15s")
    assert "pipeline.candidate all --version v6s" in out.stdout
    assert "--touch-set" not in out.stdout                       # the version decides
    out = _make("-n", "bench-build", "TOUCH_SET=w15s")
    assert "pipeline.events bench" in out.stdout and "--touch-set" not in out.stdout
    # TOUCH_SET with another stage stops before any python runs (real run, the guard fails first)
    out = _make("bench", "STAGE=features", "TOUCH_SET=w15s")
    assert out.returncode != 0 and "STAGE=events only" in out.stdout
