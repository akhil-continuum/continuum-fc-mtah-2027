"""The only printing route for code that runs on the match.

The competition terms forbid passing the match data, or any extract of it, to a hosted assistant,
and everything printed reaches one. So match code prints aggregates through ``summary()`` and
nothing else:

    from pipeline.report import summary, match_entry

    @match_entry
    def main():
        ...
        summary(possessions_per_min=2.1, share_d=0.24, label_counts={"H": 3, "A": 2, "D": 1})

``summary`` accepts

- scalars: int, float, bool, None, and strings of at most 80 characters;
- lists or tuples of at most 20 numbers (histogram counts, bin edges);
- dicts of at most 20 entries whose keys are short strings and whose values are scalars as above.

It raises ``SummaryRefused`` on anything else: numpy arrays (of any size), pandas objects, long
strings, nested containers, anything longer than 20 elements. The values are printed one per line
and written to ``reports/<name>.json``, where ``<name>`` is the calling module's last dotted
component unless ``_name`` is given. Several calls in one process merge into the same file; the
first call of a process replaces what an earlier run left there.

``match_entry`` wraps an entry point so that an exception prints its type and a shortened message
only, never a traceback (tracebacks can echo local values and DataFrame rows).
"""
import functools
import json
import math
import numbers
import os
import sys
from pathlib import Path

# Params
MAX_ITEMS = 20          # longest list or dict accepted
MAX_STR = 80            # longest string accepted, keys included
MAX_ERROR_CHARS = 160   # longest exception message match_entry will print
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

_written_this_process = set()


class SummaryRefused(TypeError):
    """Raised when a value is not an aggregate summary() is willing to print."""


def _is_number(value):
    return isinstance(value, numbers.Number) and not isinstance(value, bool) and not _is_array(value)


def _is_array(value):
    """numpy arrays and pandas objects, recognised without importing either."""
    module = type(value).__module__ or ""
    if module.startswith("pandas"):
        return True
    return hasattr(value, "shape") and hasattr(value, "dtype") and getattr(value, "ndim", 0) > 0


def _scalar(key, value):
    """A JSON-ready scalar, or SummaryRefused."""
    if value is None or isinstance(value, bool):
        return value
    if _is_array(value):
        raise SummaryRefused("{}: arrays and DataFrames are not aggregates ({})".format(
            key, type(value).__name__))
    # numpy scalars (np.int64, np.float32, np.bool_) are zero-dimensional: unwrap them.
    if hasattr(value, "item") and getattr(value, "ndim", None) == 0:
        value = value.item()
        if isinstance(value, bool):
            return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        value = float(value)
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, str):
        if len(value) > MAX_STR:
            raise SummaryRefused("{}: string longer than {} characters ({})".format(
                key, MAX_STR, len(value)))
        return value
    raise SummaryRefused("{}: type {} is not an aggregate".format(key, type(value).__name__))


def _check(key, value):
    """Validate one keyword argument of summary() and return its JSON form."""
    if not isinstance(key, str) or len(key) > MAX_STR:
        raise SummaryRefused("keys must be strings of at most {} characters".format(MAX_STR))
    if _is_array(value):
        raise SummaryRefused("{}: arrays and DataFrames are not aggregates ({})".format(
            key, type(value).__name__))
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_ITEMS:
            raise SummaryRefused("{}: {} elements, more than {}".format(
                key, len(value), MAX_ITEMS))
        out = []
        for item in value:
            if not (_is_number(item) or item is None):
                raise SummaryRefused("{}: lists may hold numbers only, not {}".format(
                    key, type(item).__name__))
            out.append(_scalar(key, item))
        return out
    if isinstance(value, dict):
        if len(value) > MAX_ITEMS:
            raise SummaryRefused("{}: {} entries, more than {}".format(
                key, len(value), MAX_ITEMS))
        out = {}
        for sub_key, item in value.items():
            if isinstance(sub_key, bool) or not isinstance(sub_key, (str, int)):
                raise SummaryRefused("{}: dict keys must be strings or ints".format(key))
            sub_key = str(sub_key)
            if len(sub_key) > MAX_STR:
                raise SummaryRefused("{}: dict key longer than {} characters".format(
                    key, MAX_STR))
            if isinstance(item, (list, tuple, dict, set)) or _is_array(item):
                raise SummaryRefused("{}: dict values must be scalars".format(key))
            out[sub_key] = _scalar("{}.{}".format(key, sub_key), item)
        return out
    if isinstance(value, (set, frozenset, bytes, bytearray)):
        raise SummaryRefused("{}: type {} is not an aggregate".format(key, type(value).__name__))
    return _scalar(key, value)


def _caller_name(depth=2):
    """Last dotted component of the calling module, resolving __main__ through __spec__."""
    frame = sys._getframe(depth)
    module = frame.f_globals.get("__name__", "report")
    if module == "__main__":
        spec = frame.f_globals.get("__spec__")
        module = getattr(spec, "name", None) or "main"
        if module.endswith(".__main__"):
            module = module[: -len(".__main__")]
    return module.rsplit(".", 1)[-1]


def summary(_name=None, _reports_dir=None, _quiet=False, **numbers_):
    """Print aggregates and write them to reports/<name>.json. Raises SummaryRefused first.

    Everything is validated before anything is printed or written, so a refused call leaks
    nothing.
    """
    checked = {key: _check(key, value) for key, value in numbers_.items()}
    name = _name or _caller_name()
    if not isinstance(name, str) or not name or len(name) > MAX_STR or "/" in name:
        raise SummaryRefused("report name must be a short string without '/'")

    folder = Path(_reports_dir) if _reports_dir else REPORTS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "{}.json".format(name)
    stored = {}
    tag = (str(folder), name)
    if tag in _written_this_process and path.is_file():
        with open(path) as handle:
            stored = json.load(handle)
    stored.update(checked)
    with open(path, "w") as handle:
        json.dump(stored, handle, indent=1, sort_keys=True, allow_nan=False)
        handle.write("\n")
    _written_this_process.add(tag)

    if not _quiet:
        for key, value in checked.items():
            print("{}: {} = {}".format(name, key, json.dumps(value, sort_keys=True)))
    return checked


def match_entry(function):
    """Wrap a match entry point: on any exception print type and short message, exit 1.

    No traceback is printed, because frames of a traceback can show local values. The message is
    cut to MAX_ERROR_CHARS; write error messages in match code with counts and names, never values.
    """
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except SystemExit:
            raise
        except BaseException as error:  # noqa: BLE001 - the point is to catch everything
            message = str(error).replace("\n", " ")
            if len(message) > MAX_ERROR_CHARS:
                message = message[:MAX_ERROR_CHARS] + "..."
            sys.stderr.write("error: {}: {}\n".format(type(error).__name__, message))
            sys.exit(1)
    return wrapper


def quiet_warnings():
    """Silence Python warnings in match runs: a warning can quote a value."""
    import warnings
    warnings.filterwarnings("ignore")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
