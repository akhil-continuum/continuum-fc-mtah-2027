"""``make data-check``: are the nine match files in place, and do their counts match DATA.md?

    uv run python -m pipeline.setup_data

Prints, through pipeline.report.summary only: presence yes/no and size in bytes for each asset of
each source (paths from games.asset_paths()), whether the size equals the team's reference
download, and record counts against the published numbers:

    tactical   ATD records 199,319   track slots 46   Smart Tagging instances 638
               homography rows 199,319
    broadcast  ATD records 164,488   track slots 51   homography rows 164,488

Counting reads the files but never prints from them: a line count, a number of <Player>
elements, a number of <instance> elements. Unpacking is pipeline/unpack_data.py (the lead's).
"""
import os
import sys
import xml.etree.ElementTree as ET

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
import games  # noqa: E402

from pipeline.report import match_entry, quiet_warnings, summary  # noqa: E402

# Params: published in DATA.md (frames come from games.py).
PUBLISHED_SLOTS = {"tactical": 46, "broadcast": 51}
PUBLISHED_TAGS = 638
SOURCES = ("tactical", "broadcast")
CHUNK = 1 << 24


def reference_bytes():
    """Sizes of the team's reference download, if the lead's unpack module provides them."""
    try:
        from pipeline.unpack_data import EXPECTED_BYTES
        return dict(EXPECTED_BYTES)
    except Exception:  # noqa: BLE001
        return {}


def count_lines(path):
    """Non-empty lines in a text file, counted in binary chunks."""
    n, last = 0, b"\n"
    with open(path, "rb") as handle:
        while True:
            block = handle.read(CHUNK)
            if not block:
                break
            n += block.count(b"\n")
            last = block[-1:]
    return n + (0 if last == b"\n" else 1)


def count_elements(path, name):
    n = 0
    for _, node in ET.iterparse(path, events=("end",)):
        if node.tag.rsplit("}", 1)[-1] == name:
            n += 1
        node.clear()
    return n


def check(data_dir=None):
    ref = reference_bytes()
    out = {}
    all_present = True
    for src in SOURCES:
        paths = games.asset_paths(data_dir=data_dir, source=src)
        present, size, size_ok = {}, {}, {}
        for asset, path in paths.items():
            present[asset] = os.path.isfile(path)
            size[asset] = os.path.getsize(path) if present[asset] else 0
            if (src, asset) in ref:
                size_ok[asset] = present[asset] and size[asset] == ref[(src, asset)]
        all_present = all_present and all(present.values())
        out[src + "_present"] = present
        out[src + "_bytes"] = size
        if size_ok:
            out[src + "_size_matches_reference"] = size_ok

        counts = {}
        expected_frames = games.source_frames(src)
        if present.get("atd_raw"):
            counts["atd_records"] = count_lines(paths["atd_raw"])
            counts["atd_records_ok"] = counts["atd_records"] == expected_frames
        if present.get("atd_meta"):
            counts["track_slots"] = count_elements(paths["atd_meta"], "Player")
            counts["track_slots_ok"] = counts["track_slots"] == PUBLISHED_SLOTS[src]
        if present.get("homography"):
            counts["homography_rows"] = count_lines(paths["homography"]) - 1  # header
            counts["homography_rows_ok"] = counts["homography_rows"] == expected_frames
        if src == "tactical" and present.get("smart_tagging"):
            counts["tag_instances"] = count_elements(paths["smart_tagging"], "instance")
            counts["tag_instances_ok"] = counts["tag_instances"] == PUBLISHED_TAGS
        if counts:
            out[src + "_counts"] = counts
    out["all_nine_present"] = all_present
    return out


@match_entry
def main():
    quiet_warnings()
    summary(_name="data_check", **check())
    return 0


if __name__ == "__main__":
    sys.exit(main())
