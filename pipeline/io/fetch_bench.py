"""Fetch the open Metrica sample games, pinned to one commit, and verify them by sha256.

    make bench-data
    uv run python -m pipeline.io.fetch_bench [--verify-only]

Source: github.com/metrica-sports/sample-data at commit COMMIT (the repository's HEAD on
13 Nov 2026, the day bench_data/ was first downloaded from its master branch). Files land in
``bench_data/raw/metrica/Sample_Game_<k>/`` (bench-builder's layout, read by bench.metrica_io).

A file already present with the expected sha256 is not downloaded again, so the command is
offline-safe once the bench data exists. A download whose sha256 differs is refused (left as
``<name>.part``), which also catches the pinned commit not matching the hashes below. The hashes
are those of the files the bench was built from.

Bench data only: nothing here touches the match.
"""
import argparse
import hashlib
import os
import sys
import urllib.request

from pipeline.io.sources import BENCH_DATA

COMMIT = "e706dd506b360d69d9d123d5b8026e7294b13996"
URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/{commit}/data/{path}"
RAW_DIR = os.path.join(BENCH_DATA, "raw", "metrica")

SHA256 = {
    "Sample_Game_1/Sample_Game_1_RawEventsData.csv":
        "097dda6345e17390dca095331458fd00e47960c480c76f4356fd4edf2575ef05",
    "Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv":
        "0a3df68a92af0290ede472e62de3b6489ddee29223da09bdf0cf76c8e0ab0618",
    "Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv":
        "0a3d0eff7785950379c587d3597b86a2d91c2492281b4652e1abb97c99b6b364",
    "Sample_Game_2/Sample_Game_2_RawEventsData.csv":
        "edf31a18599265b77a8baf150f2ce6d89456fb0324b62ea7a391229657a619ba",
    "Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv":
        "aa639f40762eade46c389020fe4458266b5ba14027b1e96e22e75db62803718b",
    "Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv":
        "3f9990ac204b07133e4202441d6fcbb386897c7c7f81e1bc3ce4d35bc8644694",
    "Sample_Game_3/Sample_Game_3_events.json":
        "28a53bfe9eaf32535362335553d517cdce0eabb588af6166009081753751f4e0",
    "Sample_Game_3/Sample_Game_3_metadata.xml":
        "009c7dcfcb66192014917ca1559c158f612353e390eafb85f8214b06a1c7dec1",
    "Sample_Game_3/Sample_Game_3_tracking.txt":
        "570e6ce82c2d93bc92723af331d93c31110bec1d6dc0be07c6d44ca3f981d658",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(rel, raw_dir=RAW_DIR, commit=COMMIT):
    dest = os.path.join(raw_dir, rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    with urllib.request.urlopen(URL.format(commit=commit, path=rel), timeout=120) as resp, \
            open(part, "wb") as out:
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            out.write(block)
    if sha256(part) != SHA256[rel]:
        raise ValueError("{}: sha256 differs from the pinned value; left as .part".format(rel))
    os.replace(part, dest)


def main(argv=None):
    ap = argparse.ArgumentParser(description="fetch and verify the Metrica sample games")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--raw-dir", default=RAW_DIR)
    args = ap.parse_args(argv)
    bad = 0
    for rel, want in sorted(SHA256.items()):
        path = os.path.join(args.raw_dir, rel)
        if os.path.isfile(path) and sha256(path) == want:
            print("ok        {}".format(rel))
            continue
        if args.verify_only:
            print("{:9} {}".format("missing" if not os.path.isfile(path) else "differs", rel))
            bad += 1
            continue
        print("fetching  {} @ {}".format(rel, COMMIT[:12]), flush=True)
        try:
            fetch(rel, args.raw_dir)
            print("ok        {}".format(rel))
        except (OSError, ValueError) as error:
            print("FAILED    {}: {}".format(rel, error))
            bad += 1
    print("metrica sample data @ {}: {} of {} files verified".format(
        COMMIT[:12], len(SHA256) - bad, len(SHA256)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
