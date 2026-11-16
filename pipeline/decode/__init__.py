"""Decode stage: per-frame probabilities to H/A/D labels (rulebook Part 2).

v1 hysteresis yardstick, v2 four-state Viterbi (H, A, DH, DA) with the restart law as structure.
Entry point: ``python -m pipeline.decode match|bench --version <v>``.
"""
