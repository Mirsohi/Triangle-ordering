# Prospectively frozen held-out benchmark

This eight-graph suite was specified on 2026-09-10, before running DOVE or
VOLT on any graph in the suite. It is the confirmatory split. The earlier 13
public graphs are development data and the earlier 11-network expansion is
validation data.

Selection rules were fixed as follows: use public SNAP edge datasets; include
one dataset from each of eight application families; exclude every exact
dataset and temporal snapshot already used for development or validation; use
the common preprocessing rule (parse the first two integer columns, remove
self-loops, symmetrize, deduplicate, and densely remap non-isolated vertices);
and do not replace graphs based on results. The initially frozen method used
five seeded one-pass Neigh runs, selection by independently evaluated exact
candidate work, followed by 200 guarded DOVE steps with the development-tuned
hyperparameters. All five Neigh runtimes are charged.

## Audit-driven protocol amendment

After the one-pass run, a line-by-line comparison with the Neigh paper and
source showed that the published implementation's default is not one pass: it
iterates until the relative C+- improvement falls below 0.01. The final paper
therefore uses five seeded runs with `-n 0.01` and warm-starts DOVE from the
strongest exact-Lambda incumbent among those runs and the six built-ins. This
change strengthens the comparator and was made after seeing the one-pass
outputs, so the paper describes the held-out analysis as an audit-driven
evaluation rather than as a preregistered confirmatory test. The graphs, DOVE
hyperparameters, metrics, and no-replacement rule were unchanged.

The graph names and download URLs are defined in
`scripts/prepare_heldout_data.py`; `manifest.csv` is created by that script and
records SHA-256 hashes of the downloaded archives.
