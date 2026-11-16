"""ATD channel audit: what does the EPTS export carry besides x and y?

    uv run python -m pipeline.diagnostics.atd_channels match                  # tactical
    uv run python -m pipeline.diagnostics.atd_channels match --source broadcast
    uv run python -m pipeline.diagnostics.atd_channels bench --meta M.xml --raw T.txt --label sg3

Reports, through pipeline.report.summary only:

- every sensor channel declared in the metadata: id, name, unit, sensor;
- the kinds of player channels and how many there are;
- the ball channels each DataFormatSpecification references, in order, and any reference that is
  neither a player nor a ball channel;
- the number of comma-separated fields in the ball group and in each player group of the raw
  file, as histograms over lines (counts only, no values);
- flags for a ball-state, ball-owner or height channel anywhere in the metadata.

``--note`` rewrites this source's section of reports/atd-channels.md (gitignored) from the same
aggregates, so the note never holds anything summary() did not print.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

from pipeline.io import epts, sources
from pipeline.report import match_entry, quiet_warnings, summary

# Params
NOTE = os.path.join(sources.REPO, "reports", "atd-channels.md")   # gitignored
STATE = re.compile(r"state|alive|dead|status|in.?play|live", re.I)
OWNER = re.compile(r"owner|possess|team.?ball|control", re.I)
HEIGHT = re.compile(r"(^|[_\s])z$|height|altitude|elev", re.I)
MAX_ENTRIES = 20


def _chunks(prefix, mapping):
    """Split a dict into summary-sized dicts: {prefix: {...}} or {prefix_1: ..., prefix_2: ...}."""
    items = list(mapping.items())
    if len(items) <= MAX_ENTRIES:
        return {prefix: dict(items)}
    return {"{}_{}".format(prefix, i // MAX_ENTRIES + 1): dict(items[i:i + MAX_ENTRIES])
            for i in range(0, len(items), MAX_ENTRIES)}


def raw_field_counts(meta, raw_path):
    """Histograms of comma-field counts in the ball group and the player groups, over lines."""
    if not meta.specs:
        return {}, {}, Counter()
    top_sep = meta.specs[0].top_separator
    ball_hist, player_hist, parts_hist = Counter(), Counter(), Counter()
    with open(raw_path, "r") as handle:
        for line in handle:
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split(top_sep)
            parts_hist[len(parts)] += 1
            if len(parts) >= 3:
                ball_hist[len(parts[-1].split(","))] += 1
                for group in parts[1].split(";"):
                    player_hist[len(group.split(","))] += 1
    return dict(ball_hist), dict(player_hist), parts_hist


def audit(meta_path, raw_path):
    meta = epts.read_metadata(meta_path)
    out = {}
    channels = {cid: "{} | unit {} | sensor {}".format(c["name"], c["unit"] or "-", c["sensor"])
                for cid, c in meta.channels.items()}
    out.update(_chunks("sensor_channels", channels))
    out["n_sensor_channels"] = len(meta.channels)
    out["n_player_channels"] = len(meta.player_channels)
    out["n_players_in_metadata"] = len(meta.players)
    out["n_teams"] = len(meta.teams)
    kinds = Counter(meta.channels.get(cid, {}).get("kind", cid)
                    for _, cid in meta.player_channels.values())
    out["player_channel_kinds"] = dict(kinds)
    out["n_specs"] = len(meta.specs)

    ball_refs, other_refs, string_regs = Counter(), Counter(), Counter()
    group_sizes, spec_fields = Counter(), Counter()
    for spec in meta.specs:
        refs = tuple(leaf.ref for leaf in spec.leaves if leaf.role == "ball")
        ball_refs[" ".join(refs)] += 1
        for leaf in spec.leaves:
            if leaf.role == "other":
                other_refs["{}:{}".format(leaf.tag, leaf.ref)] += 1
            elif leaf.role in ("string", "frame"):
                string_regs["{}:{}".format(leaf.role, leaf.ref)] += 1
        for size in spec.player_groups:
            group_sizes[size] += 1
        spec_fields[spec.n_fields] += 1
    out["spec_ball_channel_refs"] = dict(ball_refs)
    out["spec_other_refs"] = dict(other_refs) or {"none": 0}
    out["spec_string_registers"] = dict(string_regs)
    out["spec_player_group_sizes"] = {str(k): v for k, v in sorted(group_sizes.items())}
    out["spec_fields_per_line"] = {str(k): v for k, v in sorted(spec_fields.items())}
    out["metadata_extra_tags"] = meta.extra_channel_tags or {"none": 0}
    out["provider_global_params"] = len(meta.provider_params)

    names = []
    for cid, c in meta.channels.items():
        names += [cid, c["name"]]
    for spec in meta.specs:
        names += [leaf.ref for leaf in spec.leaves if leaf.role in ("ball", "other", "string")]
        names += [leaf.tag for leaf in spec.leaves if leaf.role == "other"]
    names += list(meta.extra_channel_tags)
    names += list(meta.provider_params)
    names = [n for n in names if n]
    out["flag_ball_state_channel"] = any(STATE.search(n) for n in names)
    out["flag_ball_owner_channel"] = any(OWNER.search(n) for n in names)
    out["flag_height_channel"] = any(HEIGHT.search(n) for n in names)
    out["flag_speed_channel"] = any(c["kind"] == "s" for c in meta.channels.values())

    ball_hist, player_hist, parts_hist = raw_field_counts(meta, raw_path)
    out["raw_top_level_parts_per_line"] = {str(k): v for k, v in sorted(parts_hist.items())}
    out["raw_ball_group_fields"] = {str(k): v for k, v in sorted(ball_hist.items())[:MAX_ENTRIES]}
    out["raw_player_group_fields"] = {str(k): v
                                      for k, v in sorted(player_hist.items())[:MAX_ENTRIES]}
    return out


def write_note(label, title, result, path=NOTE):
    """Replace this label's section of the markdown note with the audit aggregates."""
    begin, end = "<!-- BEGIN {} -->".format(label), "<!-- END {} -->".format(label)
    lines = [begin, "## {}".format(title), ""]
    for key in sorted(result):
        lines.append("- `{}`: `{}`".format(key, json.dumps(result[key], sort_keys=True)))
    lines += ["", end]
    section = "\n".join(lines)
    text = open(path).read() if os.path.isfile(path) else "# ATD channel audit\n\n"
    if begin in text and end in text:
        head, rest = text.split(begin, 1)
        _, tail = rest.split(end, 1)
        text = head + section + tail
    else:
        text = text.rstrip("\n") + "\n\n" + section + "\n"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write(text)


@match_entry
def run_match(source, note):
    quiet_warnings()
    import games
    meta, raw = games.require(["atd_meta", "atd_raw"], source=source)
    result = audit(meta, raw)
    name = "atd_channels" if source == "tactical" else "atd_channels.{}".format(source)
    summary(_name=name, **result)
    if note:
        write_note("match-{}".format(source), "Match, {} source".format(source), result)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="ATD channel audit")
    parser.add_argument("target", choices=("match", "bench"))
    parser.add_argument("--source", default="tactical", choices=("tactical", "broadcast"))
    parser.add_argument("--meta")
    parser.add_argument("--raw")
    parser.add_argument("--label", default="bench")
    parser.add_argument("--note", action="store_true", help="update reports/atd-channels.md")
    args = parser.parse_args(argv)
    if args.target == "match":
        return run_match(args.source, args.note)
    if not (args.meta and args.raw):
        parser.error("bench needs --meta and --raw")
    result = audit(args.meta, args.raw)
    summary(_name="atd_channels.bench.{}".format(args.label), **result)
    if args.note:
        write_note("bench-{}".format(args.label), "Bench game {}".format(args.label), result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
