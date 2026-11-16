# Bench sources

| Game id | Source | Format | Frame rate | Licence / terms | How the labels are made |
|---|---|---|---|---|---|
| metrica_g1 | Metrica Sports sample data, Sample Game 1 (github.com/metrica-sports/sample-data, `data/Sample_Game_1`) | Tracking CSV (home and away files, normalised 0–1, NaN for missing) + event CSV | 25 fps (Time column steps 0.04 s) | No licence file in the repository (the LICENSE URL returns 404). README "Legal stuff": "Please be responsible with the use of this data" and "If you use it for anything public, please acknowledge the source." Bench use only; not redistributed (`bench_data/` is gitignored) | `bench/truth.py`, from the events plus a ball-missing confirmation (convention in its docstring); the labels truth is `bench/truth_labels.py` |
| metrica_g2 | Same repository, Sample Game 2 | Same as game 1 | 25 fps | Same | Same |
| metrica_g3 | Same repository, Sample Game 3 | FIFA EPTS (metadata XML + raw txt; 6 data specifications for lineup changes) + event JSON | 25 fps (FrameRate in the metadata) | Same | Same. Game 3 is the same provider and format as the match |

Downloaded on 13 November 2026 to `bench_data/raw/metrica/Sample_Game_{1,2,3}/`, via raw GitHub
URLs. `make bench-data` fetches them at the pinned commit `e706dd5` and verifies each file by
sha256 (`pipeline/io/fetch_bench.py`).

Not used:

- The public 5 minutes (`evaluation/gt/`). It is never a bench game.
- Any match data.

Noise levels derived from these games are documented in `bench/noise.py`:

- `<game>_low`, `<game>_match`, `<game>_high`: the v3 model (T-33, P-17a): v2 plus a centred
  Gaussian low-pass on the ball inside each seen run, because the match's ball track is low-pass
  filtered. Sigma fitted only to the match's ball-shape aggregates (`bench/smoothness.py`);
- `<game>_legacy2{low,match,high}`: the v2 model (T-6), fitted to match input aggregates; the
  roots as they stood before T-33, with all their stage outputs, probs and labels;
- `<game>_legacy{low,match,high}`: the T-2 model.

All three keep the same seeds, so v3 and v2 roots have identical players and identical ball
gaps; only the ball positions differ. Truth files are the same in every root of a game.

Not yet added: SkillCorner open data, or any other automatic broadcast tracking. Its possession
field has not been checked (Tier 1).
