"""Rule registry: load, validate and cross-check pipeline/rules/registry.yaml against the rulebook.

    uv run python -m pipeline.rules.registry          # validate; prints counts by status

Every rule ID in docs/02-football-rulebook.md (sections S, R, D, L, P, T, B, G and the
ambiguous cases A) must have exactly one registry entry, and the registry may hold no ID the
rulebook lacks. ``rulebook_ids`` finds IDs as the first cell of a table row (``| S1 |``) plus
the R0 paragraph (``**R0 (LAW, STATE).**``).
"""
import importlib
import os
import re
import sys

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RULEBOOK = os.path.join(REPO, "docs", "02-football-rulebook.md")
REGISTRY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "registry.yaml")

SECTIONS = ("S", "R", "D", "L", "P", "T", "B", "G", "A")
FIELDS = ("id", "tier", "encoding", "statement", "function", "params", "status",
          "bench_precision", "bench_coverage")
OPTIONAL_FIELDS = ("notes",)
TIERS = ("LAW", "STRONG", "CLUE", "OPEN", "CASE", "METHOD")
ENCODINGS = ("MASK", "STATE", "EVENT", "PSEUDO", "COST", "FEAT", "SNAP", "CHECK", "TEST",
             "METHOD", "NONE")
STATUSES = ("planned", "implemented", "measured", "in_use", "dropped")

_ROW_ID = re.compile(r"^\|\s*([SRDLPTBGA]\d{1,2})\s*\|")
_PARA_ID = re.compile(r"\*\*([SRDLPTBGA]\d{1,2})\s*\(")


class RegistryError(ValueError):
    """The registry is malformed or out of step with the rulebook."""


def rulebook_ids(path=RULEBOOK):
    """Ordered list of every rule ID defined in the rulebook (Part 1 only)."""
    ids = []
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    part1 = text.split("# Part 2.")[0]
    for line in part1.splitlines():
        m = _ROW_ID.match(line.strip())
        if m:
            ids.append(m.group(1))
            continue
        for m in _PARA_ID.finditer(line):
            ids.append(m.group(1))
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def load_registry(path=REGISTRY):
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, list):
        raise RegistryError("registry.yaml must be a list of entries")
    return data


def _check_entry(entry):
    problems = []
    if not isinstance(entry, dict):
        return ["entry is not a mapping"]
    rid = entry.get("id", "?")
    missing = [f for f in FIELDS if f not in entry]
    extra = [f for f in entry if f not in FIELDS + OPTIONAL_FIELDS]
    if missing:
        problems.append("{}: missing fields {}".format(rid, missing))
    if extra:
        problems.append("{}: unknown fields {}".format(rid, extra))
    if entry.get("tier") not in TIERS:
        problems.append("{}: tier {!r} not in {}".format(rid, entry.get("tier"), TIERS))
    enc = entry.get("encoding") or ""
    for part in [p.strip() for p in str(enc).split(",") if p.strip()]:
        if part not in ENCODINGS:
            problems.append("{}: encoding {!r} not in {}".format(rid, part, ENCODINGS))
    status = entry.get("status")
    if status not in STATUSES:
        problems.append("{}: status {!r} not in {}".format(rid, status, STATUSES))
    if not isinstance(entry.get("statement"), str) or not entry.get("statement").strip() \
            or "\n" in entry.get("statement", ""):
        problems.append("{}: statement must be one non-empty line".format(rid))
    if not isinstance(entry.get("params"), dict):
        problems.append("{}: params must be a mapping (may be empty)".format(rid))
    func = entry.get("function") or ""
    if status in ("implemented", "measured", "in_use") and not func:
        problems.append("{}: status {} needs a function".format(rid, status))
    if status == "measured" and entry.get("bench_precision") is None \
            and entry.get("bench_coverage") is None:
        problems.append("{}: status measured needs bench_precision or bench_coverage".format(rid))
    for key in ("bench_precision", "bench_coverage"):
        v = entry.get(key)
        if v is not None and not (isinstance(v, (int, float)) and 0.0 <= v <= 1.0):
            problems.append("{}: {} must be null or in [0, 1]".format(rid, key))
    return problems


def resolve(dotted):
    """Import ``package.module.attr`` and return the attribute (RegistryError if absent)."""
    module_name, _, attr = dotted.rpartition(".")
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise RegistryError("cannot import {}: {}".format(module_name, error))
    if not hasattr(module, attr):
        raise RegistryError("{} has no attribute {}".format(module_name, attr))
    return getattr(module, attr)


def validate(registry=None, rulebook=RULEBOOK, check_functions=True):
    """Raise RegistryError listing every problem; return the registry when it is clean."""
    entries = load_registry() if registry is None else registry
    problems = []
    ids = [e.get("id") for e in entries if isinstance(e, dict)]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        problems.append("duplicate ids: {}".format(dupes))
    book = rulebook_ids(rulebook)
    lacking = [i for i in book if i not in ids]
    unknown = [i for i in ids if i not in book]
    if lacking:
        problems.append("rulebook ids missing from the registry: {}".format(lacking))
    if unknown:
        problems.append("registry ids not in the rulebook: {}".format(unknown))
    for entry in entries:
        problems += _check_entry(entry)
        if check_functions and isinstance(entry, dict) and entry.get("function"):
            try:
                resolve(entry["function"])
            except RegistryError as error:
                problems.append("{}: {}".format(entry.get("id"), error))
    if problems:
        raise RegistryError("; ".join(problems))
    return entries


def main():
    entries = validate()
    counts = {}
    for e in entries:
        counts[e["status"]] = counts.get(e["status"], 0) + 1
    print("registry ok: {} rules; by status {}".format(len(entries), counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
