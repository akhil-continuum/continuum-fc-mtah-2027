"""Synthetic SportsCode XML and synthetic roots for the tagging tests (no match data)."""
import json
import os
from xml.sax.saxutils import escape

FPS = 25
ALIASES = {"home fc": 0, "away fc": 1}
HOME, AWAY, NA = "Home FC", "Away FC", "N/A"


def inst(code, start_frame, end_frame, team=None, set_type=None, half=None, extra=()):
    """One instance, times given in frames (written to the XML as seconds)."""
    labels = []
    if team is not None:
        labels.append(("Team", team))
    if set_type is not None:
        labels.append(("Type", set_type))
    if half is not None:
        labels.append(("Half", half))
    labels.extend(extra)
    return (code, start_frame / FPS, end_frame / FPS, labels)


def xml_text(instances, raw=()):
    parts = ["<?xml version=\"1.0\" encoding=\"utf-8\"?>", "<file><ALL_INSTANCES>"]
    for i, (code, start, end, labels) in enumerate(instances):
        parts.append("<instance><ID>{}</ID><start>{}</start><end>{}</end><code>{}</code>".format(
            i + 1, start, end, escape(code)))
        for group, text in labels:
            parts.append("<label><group>{}</group><text>{}</text></label>".format(
                escape(group), escape(text)))
        parts.append("</instance>")
    parts.extend(raw)
    parts.append("</ALL_INSTANCES><ROWS></ROWS></file>")
    return "\n".join(parts)


def write_xml(path, instances, raw=()):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(xml_text(instances, raw))
    return str(path)


def make_root(path, n_frames, halves, source="synthetic", game_id="synthetic_tags"):
    os.makedirs(path, exist_ok=True)
    meta = {"game_id": game_id, "source": source, "n_frames": n_frames, "fps": FPS,
            "halves": [list(h) for h in halves]}
    with open(os.path.join(path, "meta.json"), "w") as handle:
        json.dump(meta, handle)
    return str(path)
