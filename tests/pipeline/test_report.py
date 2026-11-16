import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from pipeline.report import SummaryRefused, match_entry, summary


@pytest.fixture
def rdir(tmp_path):
    return tmp_path / "reports"


@pytest.mark.parametrize("value", [
    np.arange(3),                       # any array, however short
    np.array([1.0]),
    np.zeros((2, 2)),
    pd.DataFrame({"a": [1]}),
    pd.Series([1, 2]),
    pd.Index([1]),
    "x" * 81,                           # long string
    list(range(21)),                    # more than 20 elements
    tuple(range(21)),
    {str(i): i for i in range(21)},
    [[1, 2]],                           # nested
    ["a", "b"],                         # lists hold numbers only
    {"a": [1, 2]},                      # dict values must be scalars
    {"a": np.arange(2)},
    {"a": "y" * 81},
    {1, 2},
    b"bytes",
    object(),
])
def test_refuses(value, rdir):
    with pytest.raises(SummaryRefused):
        summary(_name="t", _reports_dir=rdir, _quiet=True, bad=value)
    assert not (rdir / "t.json").exists()   # nothing written on refusal


def test_refusal_leaks_nothing_printed(rdir, capsys):
    with pytest.raises(SummaryRefused):
        summary(_name="t", _reports_dir=rdir, ok=1, bad=np.arange(5))
    assert capsys.readouterr().out == ""


def test_long_key_refused(rdir):
    with pytest.raises(SummaryRefused):
        summary(_name="t", _reports_dir=rdir, _quiet=True, **{"k" * 81: 1})


def test_bad_name_refused(rdir):
    with pytest.raises(SummaryRefused):
        summary(_name="a/b", _reports_dir=rdir, _quiet=True, x=1)


def test_accepts_aggregates_and_writes_json(rdir, capsys):
    out = summary(_name="t", _reports_dir=rdir, n=np.int64(3), rate=np.float32(0.5), flag=True,
                  missing=float("nan"), label="atd", hist=list(range(20)),
                  counts={"H": 1, "A": np.int32(2), 3: None}, nothing=None)
    assert out["n"] == 3 and out["rate"] == 0.5 and out["missing"] is None
    stored = json.loads((rdir / "t.json").read_text())
    assert stored["counts"] == {"H": 1, "A": 2, "3": None}
    assert len(stored["hist"]) == 20
    assert "t: n = 3" in capsys.readouterr().out


def test_calls_merge_within_process(rdir):
    summary(_name="m", _reports_dir=rdir, _quiet=True, a=1)
    summary(_name="m", _reports_dir=rdir, _quiet=True, b=2)
    assert json.loads((rdir / "m.json").read_text()) == {"a": 1, "b": 2}


def test_name_defaults_to_calling_module(rdir):
    summary(_reports_dir=rdir, _quiet=True, a=1)
    assert (rdir / "test_report.json").exists()


def test_match_entry_prints_type_and_message_only(capsys):
    @match_entry
    def boom():
        secret_rows = pd.DataFrame({"frame": [12345], "x": [0.123]})  # noqa: F841
        raise ValueError("2 lines without a frame id")

    with pytest.raises(SystemExit) as info:
        boom()
    assert info.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "error: ValueError: 2 lines without a frame id\n"


def test_match_entry_truncates_long_messages(capsys):
    @match_entry
    def boom():
        raise RuntimeError("z" * 1000)

    with pytest.raises(SystemExit):
        boom()
    assert len(capsys.readouterr().err) < 200


def test_match_entry_no_traceback_in_subprocess(tmp_path):
    script = tmp_path / "s.py"
    script.write_text(
        "from pipeline.report import match_entry\n"
        "@match_entry\n"
        "def main():\n"
        "    rows = [1, 2, 3]\n"
        "    raise KeyError('missing column')\n"
        "main()\n")
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                          cwd=str(tmp_path), env={"PYTHONPATH": str(__import__("pathlib").Path(
                              __file__).resolve().parents[2])})
    assert proc.returncode == 1
    assert "Traceback" not in proc.stderr
    assert proc.stderr.startswith("error: KeyError:")
