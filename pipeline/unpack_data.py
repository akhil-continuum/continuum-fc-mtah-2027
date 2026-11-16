"""Arrange the team's Drive download into the layout games.py expects.

Run as a whole command:  uv run python -m pipeline.unpack_data

Python port of the shell script in docs/"MTAH 2027 - Data Folder Setup.md". It reads the download
folder ("MTAH Data/" at the repository root), and fills <repo>/data/tactical and
<repo>/data/broadcast. It prints through pipeline.report.summary only: counts of zips and files
handled and, per published asset, present yes/no and size-matches-reference yes/no (no names,
no sizes); errors print their type and message only (match_entry).

Videos are moved (instant on the same disk); everything else is extracted from the zips, which
stay where they are. Safe to run again.
"""
import os
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import games  # noqa: E402

from pipeline.report import match_entry, quiet_warnings, summary  # noqa: E402

DOWNLOAD = REPO / "MTAH Data"
SOURCES = ("tactical", "broadcast")

# Sizes of the team's reference download, from the setup document. A mismatch means a truncated
# or different file, not necessarily an error.
EXPECTED_BYTES = {
    ("tactical", "video"): 5222307334,
    ("tactical", "atd_raw"): 195714285,
    ("tactical", "homography"): 23187043,
    ("tactical", "smart_tagging"): 261055,
    ("tactical", "atd_meta"): 59057,
    ("broadcast", "video"): 13962360320,
    ("broadcast", "atd_raw"): 141966465,
    ("broadcast", "homography"): 15401635,
    ("broadcast", "atd_meta"): 64956,
}


def wanted():
    """{file name: destination path} for the nine published files."""
    out = {}
    for src in SOURCES:
        for path in games.asset_paths(source=src).values():
            out[os.path.basename(path)] = Path(path)
    return out


def source_of(parts):
    for src in SOURCES:
        if src in parts:
            return src
    return None


@match_entry
def main():
    quiet_warnings()
    targets = wanted()
    data_dir = Path(games.DATA_DIR)
    for src in SOURCES:
        (data_dir / src).mkdir(parents=True, exist_ok=True)
    if not DOWNLOAD.is_dir():
        summary(_name="unpack_data", download_folder_found=False)
        return 1

    zips = sorted(p for p in DOWNLOAD.rglob("*.zip"))
    extracted = 0
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = os.path.basename(info.filename)
                dest = targets.get(name)
                if dest is None or source_of(Path(info.filename).parts) is None:
                    continue
                if dest.exists() and dest.stat().st_size == info.file_size:
                    continue
                with zf.open(info) as fin, open(dest, "wb") as fout:
                    shutil.copyfileobj(fin, fout, length=1 << 24)
                extracted += 1

    # Files already unzipped by hand inside the download folder, and loose videos that Drive
    # delivered outside the zips with a suffix such as -001.
    moved = 0
    for p in sorted(DOWNLOAD.rglob("*")):
        if not p.is_file() or p.suffix == ".zip":
            continue
        dest = targets.get(p.name)
        if dest is None and p.suffix == ".mp4":
            for src in SOURCES:
                if p.name.startswith(games.SOURCES[src]["stem"]):
                    dest = Path(games.asset_paths(source=src)["video"])
        if dest is None or dest.exists():
            continue
        shutil.move(str(p), str(dest))
        moved += 1
    present, size_ok = {}, {}
    for src in SOURCES:
        for asset, path in games.asset_paths(source=src).items():
            key = "{}.{}".format(src, asset)
            present[key] = os.path.isfile(path)
            size = os.path.getsize(path) if present[key] else 0
            size_ok[key] = size == EXPECTED_BYTES[(src, asset)]
    ok = all(present.values())
    summary(_name="unpack_data", download_folder_found=True, zip_files_found=len(zips),
            files_extracted_from_zips=extracted, files_moved_into_place=moved,
            present=present, size_matches_reference=size_ok, all_9_files_in_place=ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
