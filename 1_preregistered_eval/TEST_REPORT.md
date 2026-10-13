# TEST_REPORT: pc_eval (researcher, 2026-09-28 UTC)

All tests ran in the Linux sandbox (2 vCPU Xeon, 7 GB). SNAP is blocked here, so no real SNAP file was used.
The four public case graphs (npz) are real, and every other download path was exercised against a local mock
server. Evidence files are in `tests/results/`.

## 0. Protocol

- `PREREGISTRATION.md` was written first, at 2026-09-28T05:58:39Z, before any package code existed. Its original
  hash was `90ee4040…d258`. It restates Appendix F verbatim, together with Section 6's protocol, Algorithm 1 and
  Appendix E.
- **Amendment 1** was added at 2026-09-28T06:28:45Z. No held-out or suite data had been downloaded; SNAP is not
  reachable from this sandbox. It changes one rule: the reserve for the final T pass is now
  r = min(0.5B, max(3·t1, t_T)) instead of min(0.5B, 3·t1). The reason is that on a synthetic graph with 4.8M
  edges the old rule let T+ILS exceed B by 6.0%, because the final pass needed 65 rounds. With the new rule the
  excess was 0.23% (`bench_heavytail_*.log`). The amendment also makes the `wsl:` VOLT launcher and the 4-hour
  Neigh timeout explicit.
- **Amendment 2** was added at 2026-09-28T06:51:18Z, on the coordinator's request and before any held-out download.
  It adds a bootstrap interval (95%, 10,000 resamples, seed 0) and an exact two-sided sign test of T+ILS versus
  DOVE. It also rewords `\HeldNeigh` and the `*TriCheck` macros to match the audited paper.
- The current hash `bdda241b6d7d9f13fe0a0b643f56ed2c0c229dc596ea283edc23f28c50a7ed4b` is stored in
  `PREREGISTRATION.sha256` and compiled into `run_eval.py`. The runner accepts it, and a one-byte change is
  refused. A mismatch caused only by CRLF line endings is tolerated and logged.

## 1. ILS port: numba with order-maintenance keys, compared to the C++ `insert_ils.cpp`

The C++ was compiled read-only from `work/orientation/insert_ils.cpp`. Both implementations consume the
identical SplitMix64 stream: the numba code reproduces the C++ RNG bit for bit. Test script:
`tests/test_ils_equivalence.py`; results in `ils_equivalence.json` and `.log`.

The test covers all 25 case graphs. Each graph starts from 2 orders (the stored reference and a random
permutation), with seeds 1–3 and p ∈ {1, 0, 0.5}. It compares:
- work after every single proposal (900,000 checks) and the full order at 761,760 of those points;
- work and order after 54,000 chunks of 256 and after 5,400 kicks;
- 1,350 Algorithm-1 runs with S ∈ {1, 3, 50}. Here the C++ primitives are driven by a Python implementation of
  lines 7–9 (reset = `ils_reset(incumbent)`), and numba runs `ils_chunks` in irregular batch sizes. The runs
  contain 71,136 kicks and 60,965 incumbent updates. After each run, the incumbent is restored by undo-log in
  numba and by reset in C++.

**Mismatches: 0.** The order-maintenance invariant (strictly increasing keys along the list) held in all 450 key
checks. The maintained work was also recomputed independently.

A relabel stress test (`ils_compressed_keys.json`) reset the keys to 1..n before every chunk, so that every
insertion must relabel. It ran on all 25 graphs, triggered 914 relabels, and had **0 mismatches**, including
Algorithm 1 with S = 1 and the undo.

Timing on identical proposal streams (same results in all cases):

| Graph | Proposals | C++ time | numba time |
|---|---|---|---|
| Bitcoin-Alpha | 500k | 1.92 s | 2.38 s |
| P2P-Gnutella05 | 500k | 0.52 s | 0.59 s |
| Loc-Brightkite | 500k | 7.22 s | 5.33 s |
| GitHub | 500k | 38.4 s | 38.5 s |
| Synthetic heavy-tailed (n = 1.63M, m = 4.78M, max degree 162k) | 20k | 118.7 s | 94.0 s |
| Synthetic road-like (n = 1.92M, m = 3.22M) | 20k | 15.46 s | 0.065 s (238× faster) |

On small graphs numba is as fast as C++. On graphs with millions of vertices it removes the O(n) cost of rank
shifting. On hub-heavy graphs, each move costs O(deg u + Σ deg of the flipped vertices) in both implementations,
so moves are slow: about 180 proposals/s at the 4.8M-edge heavy-tailed size.

## 2. Ports of T and smallest-last, the exact-work kernel and the loader (`tests/test_ports.py`)

Run on 85 graphs (the 25 case graphs plus 60 random graphs), with 7 start orders each for the case graphs.
Results in `ports_*.json`.

| Check | Reference implementation | Checks | Mismatches |
|---|---|---|---|
| smallest-last | `fixes/reference.py::smallest_last` | 85 | 0 |
| T (final order, work, number of rounds) | `algorithms.tangent_phase`, identifier ties, as re-exported by `lib.py` | 475 | 0 |
| single tangent round | `lib.tangent_round` | 475 | 0 |
| exact work | `dove.candidate_work` | 475 | 0 |
| lister probes = Λ; triangle count | `dove.exact_triangle_count` / networkx | 475 | 0 |
| vectorised preprocessing | `dove.graph_from_edges` (scrambled ids, duplicates, reversals, self-loops) | 85 | 0 |
| file parsing | `dove.load_edgelist` (comments, CSV header, tabs, 3 columns, bad lines) | 340 | 0 |
| GraphData adjacency | dove | 85 | 0 |

This was run twice: with the sandbox versions (numpy 2.4, numba 0.67) and with **the PC's exact versions**
(numpy 2.1.0, scipy 1.15.2, numba 0.61.0, networkx 3.4.2, in a venv). Both runs had 0 mismatches. Across the two
versions, all R* candidate works and the R*, T and DOVE orders were bit-identical on the four public graphs.
Only T+ILS differs, because it stops on a clock.

## 3. Part-B logic on the four public case graphs (`--local-npz`, full protocol, three seeds)

Logs are in `local_pub4_run_log.txt` and results in `local_pub4_results/`.

**Strengthened references reproduce ERRATA E1 exactly.** The stored Neigh runs come from
`ref/code_data/.../results/revision20260914/<g>.neigh.<s>.rank`, loaded as `argsort(rank)`, as in
`fixes/reference.py`.

| Graph | R* (this package) | Source | ERRATA E1 | Forward-only minimum |
|---|---|---|---|---|
| Bitcoin-Alpha | 74,394 | Neigh1-rev | 74,394 | 75,756 |
| GitHub | 1,997,990 | Neigh2-rev | 1,997,990 | 2,035,384 |
| Loc-Brightkite | 1,428,426 | Neigh3-rev | 1,428,426 | 1,437,697 |
| P2P-Gnutella05 | 54,862 | Neigh4 | 54,862 | 54,862 |

For Loc-Brightkite, `fixes/references.csv` also gives W_Neigh3-rev = 1,428,426, which ties the stored reference
reversed.

Method results (B is DOVE's time; "total s" is the T+ILS wall time per seed):

| Graph | T | DOVE | T+ILS seeds 1–3 | B | total s |
|---|---|---|---|---|---|
| Bitcoin-Alpha | 67,858 | 67,647 | 67,355 / 67,329 / 67,329 | 0.257 | 0.257–0.259 |
| GitHub | 1,872,360 | 1,915,067 | 1,869,946 / 1,869,911 / 1,869,938 | 6.81 | 6.78 |
| Loc-Brightkite | 1,306,040 | 1,305,250 | 1,293,046 / 1,293,369 / 1,293,273 | 4.55 | 4.46–4.47 |
| P2P-Gnutella05 | 53,517 | 53,308 | 52,217 / 52,163 / 52,218 | 0.501 | 0.50 |

All 24 saved orders were verified: each was reloaded from disk, and dove work = numba lister probes = the value
the method reported; the triangle counts were identical within each graph. `--quick` (1 seed) also passed
(`local_pub4_quick_run_log.txt`).

## 4. Full non-local code paths against a mock artifact and mock SNAP server (`tests/test_pipeline_mock.py`)

The mock artifact uses the real `src/dove`, with mock manifests, `per_graph.csv` and stored ranks. For
P2P-Mock, the real stored Neigh rank of revision20260914 is used. The mock server serves files at the exact
pre-registered SNAP paths. Results in `pipeline_mock.json` and `mock_run*.log`.

All of the following behaved as specified:
- **Part A:**
  - facebook tar extraction, gz files, and a manifest SHA-256 mismatch, which gives a warning and the graph is
    still used;
  - an n/m mismatch with `per_graph.csv`, which excludes the graph;
  - the stored-Neigh work check.
- **Part B:**
  - gz, zip member selection by basename, CSV headers, a 3-column temporal file, self-loops and duplicates;
  - a zip without the member, which marks the graph missing;
  - a 404 with 5 attempts, which marks it missing and is retried on resume;
  - a corrupt gzip, which marks it missing;
  - a bipartite graph, where Λ_R* = 0 and all methods tie.
- **Resume:** done graphs are skipped and the download failure is retried.
- **`--quick`:** runs only synthetic graphs in Part B; `make_tables.py` refuses both quick results and test-URL
  results unless `--allow-test` is given.

`tests/test_volt_plumbing.py` checks the `--volt` path with a fake VOLT and a fake `wsl.exe`, for both a native
path and `wsl:<path>`. In both cases the argument list equals the artifact's `run_neigh`, Neigh1–5 enter R* in both
directions, and a failing seed is recorded and excluded.

## 5. make_tables and paper compilation

`make_tables.py --allow-test` wrote the mock results into a scratch copy of `/home/claude/paper_new`. Files:
`tests/results/mock_generated/` and the compiled `mock_paper_compiled.pdf`.

- `macros.tex` starts with `\resultspendingfalse` and defines all 19 placeholder macros. It adds `\SuiteTIKvsT`,
  `\HeldTIKvsT` and `\SuiteCount`.
- The suite rows have 6 columns and the held-out rows 7, with "missing" rows spanning 6 columns and a Mean row.
- pdflatex, bibtex, pdflatex ×2 gave 0 errors and 0 undefined references. The red "Draft" banner and all `??`
  are gone; the original PDF had the banner once and `??` 27 times. The 3 overfull boxes are the same ones as in
  the original.
- I also compiled with worst-case held-out magnitudes (for example Λ = 123.456m). Table 5 then overflows its column
  by 4.8 pt, even after the package already sets "Facebook page--page" on two lines. With the long label on one
  line it overflowed by 24.8 pt.

### 5b. After Amendment 2 (audited paper)

- The sign test agrees with `scipy.stats.binomtest` on 5 cases. p-values are given to 2 significant digits, for
  example 0.0020 and 0.031, or as "<0.001".
- The check phrases report failures: with a simulated failed order plus a graph whose triangle counts differ
  between orders, the held-out phrase read "failed for 7 of 42 (see log)".
- I made a fresh full copy of `/home/claude/paper_new` (appendix.tex, figures/ and references.bib included) and
  wrote the mock results into it with `--allow-test`. `make_tables.py` defines all 25 macros of the audited
  placeholder, plus `\SuiteCount`, and refuses to run if any macro the paper expects is missing.
- pdflatex, bibtex, then pdflatex ×2 gave 0 errors and 0 undefined references. The "Label(s) may have changed"
  warning cleared on a third pdflatex pass.
- `pdftotext | grep "??"` finds nothing, and the "Draft" banner is absent. The sentences read correctly, for
  example "Neigh runs in the reference: none (VOLT not available). Exact-work and triangle-count checks: passed
  for all 42 saved orders."

## 6. Scale and runtime

Script: `tests/bench_large.py`; logs `bench_*.log` and `bench_*.json`.

Heavy-tailed synthetic graph (n = 1.63M, m = 4.78M), run with the PC's package versions on this 2.1 GHz Xeon:

| Step | Time |
|---|---|
| Parse and preprocess | 7 s |
| R* candidates | 41 s (Core 27 s, Check 7 s) |
| T | 8.5 s (59 rounds) |
| DOVE | 169 s |
| T+ILS | 169.0 s for B = 168.6 s |
| Verification | 4.6 s per order |

Peak memory was 1.6 GB.

Road-like synthetic graph (n = 1.92M, m = 3.22M): DOVE took 89 s. T+ILS made 54M proposals in 88.7 s against
B = 88.9 s and reached 1,069,224, compared with 1,236,414 for DOVE and 1,325,294 for T.

The PC is expected to be about 1.7× faster (the old artifact runs had Web-Stanford DOVE at 34 s). Estimated PC
time: Part A 25–45 min, Part B 30–60 min.

## 7. Informative: counter semantics (`tests/compare_counter_semantics.py`)

Algorithm 1 as printed resets c only when the incumbent improves, and the package implements it that way. The
historical tuning driver (`orient.py::ils_phase`) reset c whenever the best value since the last restart improved.
Pure ILS from the strengthened references, 3 seeds (1 s on synthetic graphs, 3 s on public graphs):

| Group | Paper rule, mean gain | Historical rule, mean gain | Paper better / tie / worse |
|---|---|---|---|
| 12 synthetic dev graphs | 16.85% | 16.82% | 4/4/4 |
| 4 public graphs | 6.46% | 6.54% | 2/0/2 |

The difference is at noise level.

## 8. Not tested here

- Real SNAP downloads: the URLs are from SNAP's naming scheme and were checked against the artifact's scripts
  where they overlap, but not fetched.
- Execution on Windows. Paths, temp files and encodings were written for it.
- A real VOLT build.
