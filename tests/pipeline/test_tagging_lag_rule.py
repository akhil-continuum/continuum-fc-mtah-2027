"""P-20 rule 3 per code (T-12d): a kept code stays only if its own median start-edge lag is
measurable and at most LAG_GATE_FRAMES; the gate row is recomputed on the survivors. Synthetic
roots only.

Reference labels (25 fps): segments of SEG frames from BASE, segment i is H when i is even and A
when it is odd, graded [BASE, BASE + NSEG * SEG). A change INTO the segment's team sits at the
start s of every segment but the first. Core (2 s = 50 frames) frames: [s + 50, s + SEG - 50).
Live frames: NSEG * SEG = 10400.
"""
import io
import json
import os
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import pytest

from pipeline import report
from pipeline.tagging import agreement, stage, votes
from tagging_synth import ALIASES, AWAY, HOME, inst, make_root, write_xml

SEG, BASE, NSEG = 400, 1013, 26
N = BASE + NSEG * SEG + 1000
LIVE = NSEG * SEG


def seg_start(i):
    return BASE + SEG * i


def along(code, segments, lo, hi, team):
    """One instance per segment i: [s + lo, s + hi), s the segment start."""
    return [inst(code, seg_start(i) + lo, seg_start(i) + hi, team) for i in segments]


H_SEGS = range(2, NSEG, 2)     # 12 H segments that start with a change into H
A_SEGS = range(1, NSEG, 2)     # 13 A segments, each starts with a change into A

# Mixed: two early codes, one late, one on time but with too few tags to measure.
MIXED = (along("BUILD UP", H_SEGS[:6], -20, 340, HOME)             # lag -20, A 1.0
         + along("CREATING CHANCES", A_SEGS[6:12], -30, 300, AWAY)  # lag -30, A 1.0
         + along("PROGRESSION", A_SEGS[:6], 40, 380, AWAY)          # lag +40, A 1.0
         + along("BALL IN FINAL THIRD", H_SEGS[6:9], 0, 100, HOME))  # 3 edges: unmeasured
# Passes the old gate only because of the late code: pooled lag median 0, coverage 4200 / 10400.
LEANS_ON_LATE = (along("BUILD UP", H_SEGS[:6], -20, 340, HOME)
                 + along("PROGRESSION", A_SEGS[:6], 40, 380, AWAY)
                 + along("BALL IN FINAL THIRD", H_SEGS[:3], 0, 100, HOME))
# Fails the old pooled lag only because of the late code (13 late edges against 12 early).
HELD_BACK_BY_LATE = (along("BUILD UP", H_SEGS, -20, 380, HOME)
                     + along("PROGRESSION", A_SEGS, 40, 80, AWAY))


def build(tmp_path, instances, home=HOME, away=AWAY, **root_kw):
    root = make_root(tmp_path / "root", N, [[BASE, BASE + LIVE]], **root_kw)
    lab = np.array(["X"] * N, dtype=object)
    for i in range(NSEG):
        lab[seg_start(i):seg_start(i + 1)] = "H" if i % 2 == 0 else "A"
    os.makedirs(os.path.join(root, "labels"))
    pd.DataFrame({"frame": np.arange(N), "label": lab}).to_csv(
        os.path.join(root, "labels", "vt.csv"), index=False)
    named = [(c, s, e, [(g, home if t == HOME else away if t == AWAY else t) for g, t in lab_])
             for c, s, e, lab_ in instances]
    return root, write_xml(tmp_path / "tags.xml", named)


def run(tmp_path, instances):
    root, xml = build(tmp_path, instances)
    out = agreement.run(root, ref_version="vt", xml_path=xml, aliases=ALIASES)
    report.summary(_name="lag_rule_test", _reports_dir=str(tmp_path / "reports"), _quiet=True,
                   **out)      # everything is printable through summary()
    return root, xml, out


def parts(root, xml):
    tags = stage.load(root, xml_path=xml, aliases=ALIASES)
    ref = agreement.version_reference(root, "vt", tags.n_frames, tags.fps, tags.graded)
    frames = votes.frame_arrays(tags.table, tags.n_frames, tags.fps)
    spans = stage.graded_mask(tags.meta, tags.n_frames)[1]
    rows = agreement.code_table(tags, frames, ref, None)
    return tags, ref, frames, spans, rows, agreement.mirror_codes(frames)


def test_late_code_dropped_early_codes_stay(tmp_path):
    root, xml, _ = run(tmp_path, MIXED)
    tags, ref, _, spans, rows, mirrors = parts(root, xml)
    kept = agreement.kept_codes(rows, mirrors, agreement.A_MIN)
    assert {c for c, _, _ in kept} == {"BUILD UP", "CREATING CHANCES", "PROGRESSION",
                                       "BALL IN FINAL THIRD"}
    after, per_code, dropped = agreement.lag_filtered_kept(tags, ref, spans, kept)
    assert [c for c, _, _ in after] == [c for c, _, _ in kept
                                        if c in ("BUILD UP", "CREATING CHANCES")]
    assert all(item in kept for item in after)            # A_k and w_k unchanged
    assert {c: w for c, _, w in after} == {"BUILD UP": agreement.W_CAP,
                                           "CREATING CHANCES": agreement.W_CAP}
    assert per_code["BUILD UP"]["median_frames"] == -20.0
    assert per_code["CREATING CHANCES"]["median_frames"] == -30.0
    assert per_code["PROGRESSION"]["median_frames"] == 40.0
    assert dropped == {"PROGRESSION": 40.0, "BALL IN FINAL THIRD": agreement.UNMEASURED}


def test_unmeasurable_code_is_dropped(tmp_path):
    _, _, out = run(tmp_path, MIXED)
    assert out["lag_dropped_a075"]["ball_in_final_third"] == "unmeasured"
    assert out["edge_lag_start_ball_in_final_third"]["matched"] == 3
    assert "ball_in_final_third" not in out["kept_after_lag_a075"]
    assert "ball_in_final_third" in out["kept_agree_a075"]     # the agreement filter kept it


def test_gate_recomputed_set_that_leaned_on_the_late_code_now_fails(tmp_path):
    _, _, out = run(tmp_path, LEANS_ON_LATE)
    old = out["gate_a075"]
    assert old["kept_codes"] == 3 and old["coverage"] == round(4200 / LIVE, 4)
    assert old["gate_pass"] is True
    assert out["edge_lag_gate"]["median_start_frames"] == 0.0     # early codes offset the late
    assert out["edge_lag_gate"]["gate_pass"] is True
    assert out["lag_dropped_a075"] == {"progression": 40.0, "ball_in_final_third": "unmeasured"}
    assert out["kept_after_lag_a075"] == {"build_up": 1.0}
    assert out["kept_after_lag_weight_a075"] == {"build_up": agreement.W_CAP}
    new = out["gate_after_lag_a075"]
    assert new["kept_codes"] == 1 and new["a_min"] == 0.75
    assert new["coverage"] == round(6 * 360 / LIVE, 4) and new["agree_core"] == 1.0
    assert new["gate_value"] == round(new["coverage"] * 0.30, 4)
    assert new["gate_pass"] is False
    assert new["lag_dropped_codes"] == 2 and new["every_kept_code_passes_lag"] is True
    assert new["pooled_start_lag_matched_info"] == 6
    assert new["pooled_start_lag_median_frames_info"] == -20.0


def test_gate_recomputed_set_held_back_by_the_late_code_now_passes(tmp_path):
    _, _, out = run(tmp_path, HELD_BACK_BY_LATE)
    assert out["gate_a075"]["gate_pass"] is True                   # rule 2 passed before
    assert out["edge_lag_gate"]["median_start_frames"] == 40.0     # pooled rule 3 failed
    assert out["edge_lag_gate"]["gate_pass"] is False
    assert out["lag_dropped_a075"] == {"progression": 40.0}
    new = out["gate_after_lag_a075"]
    assert new["kept_codes"] == 1 and new["coverage"] == round(12 * 400 / LIVE, 4)
    assert new["gate_pass"] is True
    assert new["pooled_start_lag_median_frames_info"] == -20.0
    for tag in ("a075", "a080", "a085"):                          # every A_k is 1.0 here
        assert out["gate_after_lag_" + tag]["a_min"] == int(tag[1:]) / 100
        assert out["kept_after_lag_" + tag] == {"build_up": 1.0}


def test_everything_dropped_is_a_failed_gate(tmp_path):
    late_only = along("PROGRESSION", A_SEGS, 40, 380, AWAY)
    _, _, out = run(tmp_path, late_only)
    assert out["kept_after_lag_a075"] == {}
    assert out["gate_after_lag_a075"]["kept_codes"] == 0
    assert out["gate_after_lag_a075"]["gate_pass"] is False


def test_lag_rows_print_the_a_min_they_were_computed_on(tmp_path):
    root, xml, out = run(tmp_path, MIXED)
    assert out["edge_lag_gate"]["a_min"] == agreement.A_MIN
    tags, ref, _, spans, rows, mirrors = parts(root, xml)
    kept80 = agreement.kept_codes(rows, mirrors, 0.80)
    block = agreement.edge_lag_block(tags, ref, spans, kept80, 0.80)
    assert block["edge_lag_gate"]["a_min"] == 0.80
    assert agreement.edge_lag_block(tags, ref, spans, kept80)["edge_lag_gate"]["a_min"] is None


def test_edge_lag_block_unchanged_by_the_refactor(tmp_path):
    _, _, out = run(tmp_path, MIXED)
    start = out["edge_lag_start_all_kept"]
    assert (start["matched"], start["unmatched"]) == (21, 0)
    assert start["median_frames"] == -20.0          # 6 x -30, 6 x -20, 3 x 0, 6 x +40
    assert out["edge_lag_start_progression"]["median_frames"] == 40.0


# ---------------------------------------------------------------- match entry point

def test_match_entry_prints_aggregates_only(tmp_path, monkeypatch):
    import pipeline.ball.stage as shared
    from pipeline.tagging import __main__ as cli
    artifacts = tmp_path / "artifacts"
    os.makedirs(artifacts)
    root, xml = build(artifacts, LEANS_ON_LATE, home="Barcelona", away="Atletic Club",
                      source="match", game_id="BAR-ATH")
    monkeypatch.setattr(shared, "ARTIFACTS", str(artifacts))
    os.rename(root, str(artifacts / "match"))
    monkeypatch.setattr(stage, "tagging_path", lambda meta: xml)
    monkeypatch.setattr(stage, "tracking_team_names", lambda meta: ("FC Barcelona",
                                                                    "Athletic Club"))
    monkeypatch.setattr(report, "REPORTS_DIR", tmp_path / "reports")
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = cli.entry(["match", "--agreement", "--ref-version", "vt"])
    assert code == 0
    text = buf.getvalue()
    edges = set()
    for _, start, end, _ in LEANS_ON_LATE:
        edges |= {round(start * 25), round(end * 25), start, end}
    printed = {}
    for line in (ln for ln in text.splitlines() if ln.strip()):
        head, _, value = line.partition(" = ")
        parsed = json.loads(value)
        printed[head.split(": ", 1)[1]] = parsed
        if isinstance(parsed, (list, dict)):
            assert len(parsed) <= report.MAX_ITEMS
            values = parsed.values() if isinstance(parsed, dict) else parsed
            assert not any(isinstance(v, (list, dict)) for v in values)
        else:
            values = [parsed]
        for v in values:                     # no instance edge (frame or second) is printed
            assert not (isinstance(v, (int, float)) and not isinstance(v, bool) and v in edges)
    assert printed["lag_dropped_a075"] == {"progression": 40.0,
                                           "ball_in_final_third": "unmeasured"}
    assert printed["gate_after_lag_a075"]["gate_pass"] is False
    assert {"kept_after_lag_a080", "kept_after_lag_weight_a085",
            "gate_after_lag_a085"} <= set(printed)


@pytest.mark.parametrize("tag", ["a075", "a080", "a085"])
def test_every_threshold_has_the_three_rows(tmp_path, tag):
    _, _, out = run(tmp_path, MIXED)
    for key in ("lag_dropped_", "kept_after_lag_", "kept_after_lag_weight_", "gate_after_lag_",
                "gate_", "kept_agree_", "kept_weight_"):
        assert key + tag in out
