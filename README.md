# Continuum FC - Submission for MTAH 2027, Phase 1

> **Note from the team.** Hi MTAH team, thanks for the problem statement. We're football lovers
> and had a lot of fun with it. Reach out if you have questions about our approach, submission or
> experiments.
>
> Best, Continuum FC (akhil98swaminathan@gmail.com)

Code that turns the tracking of one football match (Barcelona v Athletic, the organisers' gated
data) into one label per video frame: home, away or dead ball (H, A, D). The ranking metric is
`spellF1`, which scores whole possessions.

## 1. Where to find the deliverables

| Deliverable | Where |
|---|---|
| Final submission file | `submissions/Continuum_FC.csv`, written by `make submission` (see 3.3). It is derived from the gated match data, so it is not committed; it is uploaded through the submission form |
| Technical document | [`docs/Continuum_FC.pdf`](docs/Continuum_FC.pdf) |
| Code | this repository |

## 2. Where to read what

| You want | Read |
|---|---|
| The method as submitted, its limitations and declarations | [`docs/Continuum_FC.pdf`](docs/Continuum_FC.pdf) |
| The football rules the code follows | `docs/02-football-rulebook.md`, `pipeline/rules/registry.yaml` |
| File formats between stages, and the make targets | `CONTRACTS.md` |
| Where the bench games come from | `bench/SOURCES.md` |
| The challenge, data and rules as published by the organisers | `CHALLENGE.md`, `DATA.md`, `EVALUATION.md`, `RULES.md`, `SUBMISSION.md`, `FAQ.md` |

## 3. How to run

The pipeline is a rule-based dead-ball scorer, a logistic team model, per-player gradient-boosted
trees used only in busy play, and a four-state Viterbi decoder in which the restarting team always
has the ball after a stoppage. Its name in the commands below is `v11g`. Every parameter was
fitted on three open Metrica Sports sample games (the bench), never on the competition match or
its public five minutes.

### 3.1 Set up

Requirements: macOS or Linux, `git`, `make`, [uv](https://docs.astral.sh/uv/) (it installs the
pinned Python 3.12 and every pinned package), about 20 GB of disk for the match data.

```bash
git clone https://github.com/akhil-continuum/continuum-fc-mtah-2027.git && cd continuum-fc-mtah-2027
uv sync --locked          # pinned environment from uv.lock
make test                 # unit tests, no data needed
```

Every `make` target runs through `uv run --locked`, with a fixed hash seed and one thread for the
numeric libraries. Building the submission only reads the committed parameter files
(`pipeline/models/*.json`) and never fits anything, so the match chain is deterministic given the
pinned environment (verified on one macOS machine).

### 3.2 Place the match data

The match data is gated (enrol on the More than a Hack 2027 page and accept the data terms; see
`DATA.md`). It is never committed. Put the nine files in a folder `data/` at the repository root
(ignored by git), or anywhere else and point `MTAH_DATA_DIR` at that folder:

```
<data folder>/
  tactical/   BAR-ATH_tactical.mp4, BAR-ATH_tactical_atd.xml, BAR-ATH_tactical_atd.txt,
              BAR-ATH_tactical_homography.csv, BAR-ATH_tactical_smart_tagging.xml
  broadcast/  BAR-ATH_broadcast.mp4, BAR-ATH_broadcast_atd.xml, BAR-ATH_broadcast_atd.txt,
              BAR-ATH_broadcast_homography.csv
```

```bash
export MTAH_DATA_DIR=/path/to/your/data/folder    # only if it is not ./data
make data-check                                    # files present, sizes and published counts
```

### 3.3 Build the submission

```bash
make candidate VERSION=v11g PART=match                 # every stage on the match, stamped labels
make submission VERSION=v11g TEAM=Continuum_FC         # submissions/Continuum_FC.csv + validator
make rebuild-check VERSION=v11g TEAM=Continuum_FC      # clean clone, rebuild, compare bytes
```

`make submission` refuses labels whose provenance stamp does not match the current code. Match
code prints aggregates only (counts, rates, short histograms), through `pipeline/report.py`.

Other targets: `make compare-labels A=v6w B=v11g PART=match|bench`, `make gate-share
VERSION=v11g PART=match|bench`, `make check-labels VERSION=v11g`, `make help`. File formats
between stages and every target are in `CONTRACTS.md`.

### 3.4 Build the bench (open data; needed to evaluate or refit, not to rebuild the submission)

```bash
make bench-build                          # fetch the Metrica sample games (pinned commit),
                                          # truth from their event data, noisy level roots, stages
uv run python -m bench.truth_labels       # the labels truth (organisers' conventions), copied
                                          # into every level root
make candidate VERSION=v11g PART=bench    # v11g on the nine bench roots, scored
uv run python -m bench.score --version v11g --level match --truth labels   # labels truth
```

The bench is three games at three noise levels (`low`, `match`, `high`) under
`artifacts/bench/`. Refit checks (bench only; they compare a fresh fit with the tracked parameter
file and print `BYTE-IDENTICAL` or `DIFFERS`; refitting the tree models is machine-dependent and
may not reproduce the committed JSON byte for byte on another machine, and the submission does not
depend on it):

```bash
uv run python -m pipeline.models.fit_t32 refit_check --variant S1g --gate G4 --model rule_p1g
```

`rule_w2` (and `rule_w1`) are refitted with
`uv run python -m pipeline.models.refit_w write --truth-w1 labels --truth-w2 default`, which
rewrites `pipeline/models/team_w1_params.json` and `team_w2_params.json`; `git diff` then shows
whether they changed. Both fitters need the bench roots in a `w15s` state (after a
`make candidate` run of v11g or v6w).

## 4. Data rules

1. The competition match data, and anything derived from it per frame, is never committed and
   never shown to a hosted assistant; match code prints aggregates only.
2. No hand labels on the match; every setting is chosen on the bench, never on the public five
   minutes.
3. Never commit `data/`, `artifacts/`, `bench_data/` or `submissions/`.
