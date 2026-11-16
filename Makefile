# Entry points (CONTRACTS.md). Every target runs the pinned environment through uv.
# Match targets print aggregates only, through pipeline.report.summary.
#
#   make data-check                 nine match files present? sizes and published counts
#   make match STAGE=load           one stage on the match  -> artifacts/match/
#                                   (STAGE=probs2 runs the team model rule_v2, after STAGE=probs)
#   make bench STAGE=load [GAME=id] [ROOT="dir ..."]  one stage on every bench game
#                                   -> artifacts/bench/<id>/, or on the given roots only
#                                   STAGE=events takes TOUCH_SET=w7|w15s (default w7)
#   make bench-stages [TOUCH_SET=w15s] [LEVELS=...]  ball, events (that set), features, probs,
#                                   probs2 on the level roots; no decode (default set w7)
#   make candidate VERSION=v [PART=all|match|bench] [LEVELS="low match high"]
#                                   every stage in order on the bench level roots and the match,
#                                   events with the version's own touch set (VERSIONS[v]
#                                   "touch_set", default w7); bench scores, match aggregates,
#                                   stamps the match labels (with the touch set)
#   make submission VERSION=v TEAM=name   submissions/<name>.csv from stamped labels + validator
#                                   (refuses labels older than the code: run make candidate)
#   make check-labels VERSION=v | TEAM=name   output checks on existing labels, no re-decode
#   make compare-labels A=v B=v PART=match|bench [LEVELS=...]   two versions' label files:
#                                   possessions and ratio, shares of H/A/D, agreement, stoppages
#   make gate-share VERSION=v PART=match|bench [LEVELS=...]   share of graded time the gated
#                                   per-player trees are used (gated tree versions, e.g. v11g)
#   make rebuild-check VERSION=v TEAM=name [SOURCE=head|worktree] [BENCH=1] [REPEAT=1] [KEEP=1]
#                                   rebuild the submission in a clean clone, compare bytes
#   make judge VERSION=v BASE=v [SCORE=0] [DRAWS=2000]   bench.score (three truths) and the
#                                   paired bootstrap of bench/paired.py, bench only
#   make bench-data                 fetch/verify the Metrica sample games (pinned commit)
#   make bench-build                bench truth, noise and stages on the level roots (w7), v0
#   make atd-audit                  ATD channel audit on the match (tactical and broadcast)
#   make baseline                   reproduce the organisers' baseline -> evaluation/results/repro_*
#   make score                      official scorer on everything in evaluation/results/
#   make test                       unit tests
#   make lock                       refresh uv.lock and requirements.lock.txt from pyproject.toml

UV      ?= uv
PY      := $(UV) run --locked python
STAGE   ?= load
TEAM    ?= team
VERSION ?=
BASE    ?=
GAME    ?=
PART    ?= all
LEVELS  ?= low match high
SOURCE  ?= head
DRAWS   ?= 2000
SCORE   ?= 1
MODULE_probs := models.rule_scorer
MODULE_probs2 := models.team_model
BENCH_GAMES  := metrica_g1 metrica_g2 metrica_g3
LEVEL_ROOTS   = $(foreach l,$(LEVELS),$(foreach g,$(BENCH_GAMES),artifacts/bench/$(g)_$(l)))
# Touch parameter set for the events stage (pipeline/events/touch.py PARAM_SETS); empty = the
# stage default (w7). make candidate ignores it: there the version's declaration decides.
TOUCH_SET ?=
ROOT      ?=
TOUCH_ARG  = $(if $(TOUCH_SET),--touch-set $(TOUCH_SET),)
TOUCH_ONLY_EVENTS = @test -z "$(TOUCH_SET)" -o "$(STAGE)" = events || \
	(echo "TOUCH_SET applies to STAGE=events only"; exit 2)

# Determinism: fixed hash seed and one thread everywhere thread count can change results.
export PYTHONHASHSEED       := 0
export OMP_NUM_THREADS      := 1
export OPENBLAS_NUM_THREADS := 1
export MKL_NUM_THREADS      := 1
export VECLIB_MAXIMUM_THREADS := 1
export NUMEXPR_NUM_THREADS  := 1

.PHONY: help data-check match bench atd-audit baseline score submission test lock \
        candidate judge rebuild-check check-labels bench-data bench-build bench-stages \
        compare-labels gate-share

help:
	@sed -n '2,/^$$/p' Makefile

data-check:
	$(PY) -m pipeline.setup_data

match:
	$(TOUCH_ONLY_EVENTS)
	$(PY) -m pipeline.$(or $(MODULE_$(STAGE)),$(STAGE)) match $(if $(VERSION),--version $(VERSION),) $(TOUCH_ARG)

bench:
	$(TOUCH_ONLY_EVENTS)
	$(PY) -m pipeline.$(or $(MODULE_$(STAGE)),$(STAGE)) bench $(if $(GAME),--game $(GAME),) $(if $(ROOT),--root $(ROOT),) $(if $(VERSION),--version $(VERSION),) $(TOUCH_ARG)

bench-stages:
	$(PY) -m pipeline.candidate stages --levels $(LEVELS) $(TOUCH_ARG)

atd-audit:
	$(PY) -m pipeline.diagnostics.atd_channels match --source tactical --note
	$(PY) -m pipeline.diagnostics.atd_channels match --source broadcast --note

baseline:
	$(PY) -m pipeline.baseline

score:
	$(PY) evaluation/run_evaluation.py

candidate:
	@test -n "$(VERSION)" || (echo "usage: make candidate VERSION=<v> [PART=all|match|bench]"; exit 2)
	$(PY) -m pipeline.candidate $(PART) --version $(VERSION) --levels $(LEVELS)

submission:
	@test -n "$(VERSION)" || (echo "usage: make submission VERSION=<v> TEAM=<team_name>"; exit 2)
	$(PY) -m pipeline.submit $(VERSION) --team $(TEAM)
	$(PY) evaluation/validate_submission.py submissions/$(TEAM).csv

check-labels:
	$(PY) -m pipeline.check_labels match $(if $(VERSION),--version $(VERSION),--team $(TEAM))

compare-labels:
	@test -n "$(A)" -a -n "$(B)" -a \( "$(PART)" = match -o "$(PART)" = bench \) || \
		(echo "usage: make compare-labels A=<v> B=<v> PART=match|bench [LEVELS=...]"; exit 2)
	$(PY) -m pipeline.compare_labels $(PART) --a $(A) --b $(B) $(if $(filter bench,$(PART)),--levels $(LEVELS),)

gate-share:
	@test -n "$(VERSION)" -a \( "$(PART)" = match -o "$(PART)" = bench \) || \
		(echo "usage: make gate-share VERSION=<v> PART=match|bench [LEVELS=...]"; exit 2)
	$(PY) -m pipeline.diagnostics.gate_share $(PART) --version $(VERSION) $(if $(filter bench,$(PART)),--levels $(LEVELS),)

rebuild-check:
	@test -n "$(VERSION)" || (echo "usage: make rebuild-check VERSION=<v> TEAM=<team_name>"; exit 2)
	$(PY) -m pipeline.rebuild_check --version $(VERSION) --team $(TEAM) --source $(SOURCE) \
		$(if $(BENCH),--bench,) $(if $(REPEAT),--repeat,) $(if $(KEEP),--keep,) \
		$(if $(REBUILD_DIR),--workdir $(REBUILD_DIR),)

judge:
	@test -n "$(VERSION)" -a -n "$(BASE)" || (echo "usage: make judge VERSION=<v> BASE=<v> [SCORE=0]"; exit 2)
	@if [ "$(SCORE)" = "1" ]; then for v in $(BASE) $(VERSION); do for l in $(LEVELS); do \
		for t in default confirmed labels; do \
		$(PY) -m bench.score --version $$v --level $$l --truth $$t --bootstrap 1000 || exit 1; \
		done; done; done; fi
	$(PY) -m bench.paired --versions $(BASE) $(VERSION) --levels $(LEVELS) \
		--truths default confirmed labels --pairs $(VERSION):$(BASE) --draws $(DRAWS) \
		--out artifacts/bench/_judge/$(VERSION)_vs_$(BASE).json

bench-data:
	$(PY) -m pipeline.io.fetch_bench

bench-build: bench-data
	$(PY) -m bench.deadball_check
	$(PY) -m bench.truth
	$(PY) -m bench.noise --levels $(LEVELS)
	$(PY) -m pipeline.ball bench --root $(LEVEL_ROOTS)
	$(PY) -m pipeline.events bench --root $(LEVEL_ROOTS)
	$(PY) -m pipeline.features bench --root $(LEVEL_ROOTS)
	$(PY) -m pipeline.models.rule_scorer bench --root $(LEVEL_ROOTS)
	@for l in $(LEVELS); do $(PY) -m bench.baseline --level $$l || exit 1; done

test:
	$(PY) -m pytest tests/pipeline

lock:
	$(UV) lock
	$(UV) export --frozen --no-hashes --no-header --no-emit-project --format requirements-txt \
		| grep -v '^ *#' | grep -v "sys_platform == 'win32'" | sed 's/ *;.*//' \
		| sort -f > requirements.lock.txt
