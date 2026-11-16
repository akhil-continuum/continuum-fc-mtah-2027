"""Which code made a labels file: a content hash of the code, stamped next to the labels.

``make candidate`` writes ``<labels>.provenance.json`` after a successful decode:

    {"version": "v2s", "code_hash": "<sha256>", "labels_sha256": "<sha256>", "code_files": 123,
     "probs": "rule_v1", "touch_set": "w7"}

(probs since T-24, touch_set since T-36: the version's probabilities model and touch parameter
set, from pipeline.candidate.stamp_extra.)

``pipeline.submit`` refuses to build a submission when the stamp is missing, when the labels file
no longer has the stamped hash (it was rewritten by something else), or when the code hash of the
checkout differs from the stamped one (the labels are older than the code).

The code hash covers the content (not the dates) of every file git sees in the checkout, tracked
or untracked-but-not-ignored, under CODE_PREFIXES, so a clean clone of the same commit gives the
same hash and an uncommitted edit changes it. Nothing here reads match data except the labels
file's bytes, and only hashes are returned.
"""
import hashlib
import json
import os
import subprocess

from pipeline.io.sources import REPO

# Everything that can change what the pipeline computes. Tests, notes and docs are left out.
CODE_PREFIXES = ("pipeline/", "bench/", "starter/", "evaluation/possession_metrics.py",
                 "evaluation/run_evaluation.py", "evaluation/validate_submission.py",
                 "games.py", "pyproject.toml", "uv.lock", "Makefile")
SKIP_PARTS = ("__pycache__",)


def file_sha256(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def code_files(repo=REPO):
    """Sorted repo-relative paths of the code files (git's view: tracked plus untracked, not
    ignored, and present on disk)."""
    out = subprocess.run(["git", "-C", repo, "ls-files", "-co", "--exclude-standard", "-z"],
                         check=True, capture_output=True).stdout.decode()
    files = set()
    for rel in out.split("\0"):
        if not rel or not rel.startswith(CODE_PREFIXES):
            continue
        if any(part in SKIP_PARTS for part in rel.split("/")):
            continue
        if os.path.isfile(os.path.join(repo, rel)):
            files.add(rel)
    return sorted(files)


def code_hash(repo=REPO):
    """(sha256 hex over path and content of every code file, number of files)."""
    digest = hashlib.sha256()
    files = code_files(repo)
    for rel in files:
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(file_sha256(os.path.join(repo, rel)).encode())
        digest.update(b"\n")
    return digest.hexdigest(), len(files)


def stamp_path(labels_path):
    base, _ = os.path.splitext(labels_path)
    return base + ".provenance.json"


def write_stamp(labels_path, version, repo=REPO, **extra):
    chash, n = code_hash(repo)
    stamp = {"version": version, "code_hash": chash, "code_files": n,
             "labels_sha256": file_sha256(labels_path)}
    stamp.update(extra)
    with open(stamp_path(labels_path), "w") as handle:
        json.dump(stamp, handle, indent=1, sort_keys=True)
        handle.write("\n")
    return stamp


def check_stamp(labels_path, repo=REPO, expect=None):
    """(ok, reason). reason is one of: current, no_labels, no_stamp, labels_changed_since_stamp,
    code_changed_since_labels (the labels are older than the code), or
    ``<key>_differs_from_version`` when ``expect`` (e.g. {"touch_set": "w7", "probs": "rule_v1"},
    the version's properties now) names a value the stamp does not carry. A stamp written before
    a key existed has no value for it and is refused the same way."""
    if not os.path.isfile(labels_path):
        return False, "no_labels"
    path = stamp_path(labels_path)
    if not os.path.isfile(path):
        return False, "no_stamp"
    with open(path) as handle:
        stamp = json.load(handle)
    if stamp.get("labels_sha256") != file_sha256(labels_path):
        return False, "labels_changed_since_stamp"
    for key, value in sorted((expect or {}).items()):
        if stamp.get(key) != value:
            return False, "{}_differs_from_version".format(key)
    if stamp.get("code_hash") != code_hash(repo)[0]:
        return False, "code_changed_since_labels"
    return True, "current"
