"""T-10: root guard, provenance stamps, submission freshness gate, check-only labels reader.

Synthetic inputs only; nothing here reads the match.
"""
import json
import os
import subprocess

import numpy as np
import pandas as pd
import pytest

import games
from pipeline import check_labels, submit
from pipeline.io import provenance, roots, sources


# --- root guard ------------------------------------------------------------------------------

def test_guard_refuses_match_root_and_inside():
    for path in (sources.match_root(), os.path.join(sources.match_root(), "labels"),
                 os.path.join(sources.match_root(), "..", "match")):
        assert roots.is_match_root(path)
        with pytest.raises(roots.NotABenchRoot) as err:
            roots.require_bench_root(path)
        assert "match" in str(err.value) and sources.REPO not in str(err.value)


def test_guard_refuses_data_dir_and_other_checkouts(tmp_path):
    assert roots.is_match_root(games.DATA_DIR)
    assert roots.is_match_root(os.path.join(games.DATA_DIR, "tactical"))
    other = tmp_path / "clone" / "artifacts" / "match"
    other.mkdir(parents=True)
    assert roots.is_match_root(str(other))
    assert roots.is_match_root(str(tmp_path / "MTAH Data" / "x"))


def test_guard_refuses_symlink_and_meta_source(tmp_path):
    link = tmp_path / "innocent"
    os.symlink(sources.match_root(), link)
    assert roots.is_match_root(str(link))
    fake = tmp_path / "copy"
    fake.mkdir()
    (fake / "meta.json").write_text(json.dumps({"source": "match", "game_id": "x"}))
    assert roots.is_match_root(str(fake))
    (fake / "meta.json").write_text(json.dumps({"source": "bench", "game_id": games.GAME}))
    assert roots.is_match_root(str(fake))


def test_guard_accepts_bench_roots(tmp_path):
    bench = tmp_path / "metrica_g1_match"
    bench.mkdir()
    (bench / "meta.json").write_text(json.dumps({"source": "bench", "game_id": "metrica_g1"}))
    assert not roots.is_match_root(str(bench))
    assert roots.require_bench_root(str(bench)) == os.path.realpath(str(bench))
    assert not roots.is_match_root(os.path.join(sources.ARTIFACTS, "bench", "metrica_g1_match"))


# --- provenance ------------------------------------------------------------------------------

def _git_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "pipeline").mkdir(parents=True)
    (repo / "pipeline" / "a.py").write_text("x = 1\n")
    (repo / "games.py").write_text("GAME = 'g'\n")
    (repo / "notes.md").write_text("not code\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    return str(repo)


def test_code_hash_content_not_dates_and_scope(tmp_path):
    repo = _git_repo(tmp_path)
    h1, n1 = provenance.code_hash(repo)
    assert n1 == 2                                   # pipeline/a.py and games.py, not notes.md
    os.utime(os.path.join(repo, "pipeline", "a.py"), (1, 1))
    assert provenance.code_hash(repo)[0] == h1       # dates do not matter
    with open(os.path.join(repo, "notes.md"), "a") as fh:
        fh.write("more\n")
    assert provenance.code_hash(repo)[0] == h1       # docs do not matter
    with open(os.path.join(repo, "pipeline", "a.py"), "a") as fh:
        fh.write("y = 2\n")
    assert provenance.code_hash(repo)[0] != h1       # code does


def test_stamp_reasons(tmp_path):
    repo = _git_repo(tmp_path)
    labels = tmp_path / "v.csv"
    assert provenance.check_stamp(str(labels), repo) == (False, "no_labels")
    labels.write_text("frame,label\n0,X\n")
    assert provenance.check_stamp(str(labels), repo) == (False, "no_stamp")
    provenance.write_stamp(str(labels), "v", repo)
    assert provenance.check_stamp(str(labels), repo) == (True, "current")
    with open(os.path.join(repo, "pipeline", "a.py"), "a") as fh:
        fh.write("z = 3\n")
    assert provenance.check_stamp(str(labels), repo) == (False, "code_changed_since_labels")
    provenance.write_stamp(str(labels), "v", repo)
    labels.write_text("frame,label\n0,D\n")
    assert provenance.check_stamp(str(labels), repo) == (False, "labels_changed_since_stamp")


# --- submission gate -------------------------------------------------------------------------

def _synthetic_labels(path):
    """Synthetic labels of the right shape: D on graded frames, X elsewhere (no match data)."""
    from pipeline.io import frames
    n = games.source_frames("tactical", games.GAME)
    graded = frames.half_index(n, frames.match_halves(games.GAME)) > 0
    lab = np.where(graded, "D", "X")
    pd.DataFrame({"frame": np.arange(n), "label": lab}).to_csv(path, index=False)
    return n, int(graded.sum())


def test_submit_refuses_unstamped_and_accepts_stamped(tmp_path):
    labels_dir, out_dir = tmp_path / "labels", tmp_path / "out"
    labels_dir.mkdir()
    path = labels_dir / "vt.csv"
    n, n_graded = _synthetic_labels(path)
    with pytest.raises(submit.StaleLabels, match="no_stamp"):
        submit.build("vt", "t", labels_dir=str(labels_dir), out_dir=str(out_dir))
    assert not (out_dir / "t.csv").exists()
    provenance.write_stamp(str(path), "vt")                # stamped against this checkout
    stats = submit.build("vt", "t", labels_dir=str(labels_dir), out_dir=str(out_dir))
    assert stats["rows"] == n and stats["graded_counts"]["D"] == n_graded
    assert stats["labels_provenance"] == "current"
    assert stats["submission_sha256"] == provenance.file_sha256(str(out_dir / "t.csv"))


# --- check-only reader -----------------------------------------------------------------------

def test_read_codes(tmp_path):
    p = tmp_path / "l.csv"
    p.write_text("frame,label\n0,h\n1, A \n2,D\n3,X\n4,Q\n")
    assert check_labels.read_codes(str(p)).tolist() == [0, 1, 2, 3, -1]
    p.write_text("frame,label\n0,H\n2,H\n")
    with pytest.raises(ValueError):
        check_labels.read_codes(str(p))


def test_check_labels_bench_refuses_match_root():
    with pytest.raises(roots.NotABenchRoot):
        check_labels.run_bench("v2s", [sources.match_root()])
