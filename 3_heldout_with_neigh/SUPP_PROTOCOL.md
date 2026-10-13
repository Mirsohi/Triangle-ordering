# Supplementary held-out run with Neigh (review round 2) — declared before the run

**Status.** Supplementary and secondary. It is *not* part of the frozen pre-registration
(`pc_eval/PREREGISTRATION.md`, SHA-256 `bdda241b…`), whose analysis and numbers stay unchanged and remain the
headline. This file fixes, before any Neigh order of a held-out graph exists, what the supplementary run computes and
reports. Its SHA-256 is recorded in `SHA256SUMS.txt` and in every output environment record.

**Why.** Mock review 2, concern 1 and Q1: the pre-registered held-out reference had no Neigh orders (no VOLT binary on
the evaluation machine), so the 14.83% held-out mean is not measured against the strongest baseline of the 32-graph
suite. This run adds Neigh to the held-out reference and answers two questions:

* **Q1a (re-scoring).** How much of the frozen held-out gain remains against the Neigh-strengthened reference?
* **Q1b (restart).** Does the comparison of T, DOVE and T+ILS change when each starts from that reference?

## Executable

`volt/volt.exe` is VOLT (lecfab/volt) at the artifact's pinned commit `31ea4914…`, with the artifact's
`third_party/volt_reproducibility.patch`, cross-compiled for Windows (MinGW-w64 GCC 13, `-Ofast -fopenmp`, static).
`volt/volt_windows_portability.patch` replaces the C library `rand()`/`srand()` and `std::random_shuffle` by a
re-implementation of glibc's generator and libstdc++'s shuffle loop, because the Windows C library uses a different
generator (RAND_MAX = 32767). Tested before the run: on two graphs (n = 3,000 and 120,000) and seeds 1–5, the
Windows executable (under Wine) returns rank files byte-identical, apart from CRLF line ends, to the pinned Linux
build. No other change was made to VOLT.

## Part V: validation of the executable on the 32-graph suite (no optimisation)

For every suite graph with a stored artifact Neigh rank file (seed s), VOLT seed s is rerun on the graph exported
exactly as `pc_eval/methods.run_volt` exports it, and the new rank vector is compared with the stored one. Reported:
the number of exact matches, and for non-matches the exact work of both orders. Suite numbers are not otherwise
changed.

## Part N: held-out graphs (the 9 completed graphs of the frozen run; Facebook page–page stays missing)

1. **Reload check** exactly as in `pc_eval_posthoc`: n, m, dense-edge SHA-256 and the exact work of the saved
   R\*, T, DOVE and T+ILS orders against the frozen `result.json`. A failed check excludes the graph.
2. **Neigh.** `methods.run_volt` (unchanged pc_eval code): threshold 0.01, seeds 1–5, `-p 1`, exactly the
   pre-registered command line. Each rank is validated as a permutation; a failed seed is reported and left out.
3. **Strengthened reference R\*N** = the lowest exact Λ among the frozen R\* and the Neigh orders, each Neigh order in
   both directions. Ties keep the frozen R\*. Reported: which candidate wins, and the gap
   100 (Λ(R\*) − Λ(R\*N)) / Λ(R\*).
4. **Q1a.** Improvement of the frozen T, DOVE and T+ILS orders (T+ILS: mean over seeds 1–3) over R\*N, per graph;
   mean, median, improved/tied/regressed, bootstrap 95% interval of the mean (20,000 resamples, seed 2027).
5. **Q1b.** If R\*N ≠ R\*: rerun T, DOVE and T+ILS from R\*N with the pre-registered code and settings
   (`methods.run_T`, `methods.run_dove`, `methods.run_tils`, seeds 1–3; the T+ILS budget is the new DOVE time, as in
   the protocol). If R\*N = R\*, the frozen runs already start from R\*N and are reused (reported as reused).
   Reported: the same statistics as in 4, wins/ties/losses of T+ILS against DOVE and T with two-sided sign tests, and
   the deadline-capped T+ILS mean (defined below).
6. **Verification.** Every new order (10 Neigh orders per graph and every rerun order) is saved, reloaded and checked:
   permutation; exact work equal in pc_eval `core`, `dove.candidate_work`, the verification lister and the timing
   lister; triangle count equal to the frozen run.
7. **Kernel timing.** The numba lister of `pc_eval_posthoc` (`ph_kernels.Lister`) on R\*, T+ILS seed 1 (frozen),
   and, if different, R\*N and T+ILS seed 1 from R\*N. One untimed warm-up each, then 21 repetitions with the orders
   alternated (rotating start); each sample averages k listings, k fixed per graph so that an R\* sample takes at
   least 20 ms. Reported per graph: the ratio of medians and the geometric mean of the per-repetition paired ratios
   with a paired bootstrap interval over repetitions; across graphs: the geometric-mean ratio with a bootstrap
   interval over graphs.

**Deadline-capped T+ILS** (two rules, both reported, with the number of runs each changes).
*No-final-pass rule:* a run whose total time exceeds its budget B is scored with its ILS incumbent, i.e. the final
tangent pass is dropped (the ILS phase can end a few milliseconds after B because the last proposal batch is not
interrupted). *Strict rule:* only an order that existed at time B counts: the ILS incumbent if ILS ended by B,
otherwise the tangent-phase order if that phase ended by B, otherwise the start order R\*N. The strict rule is
conservative (it ignores improvements made before B inside the phase that crossed B) and is dominated by timer noise
on sub-second budgets. (Amended after the checker audit, before any held-out Neigh order existed: the second rule was
added and the start-order branch made explicit.)

Nothing in `pc_eval_results` or `pc_eval_posthoc_results` is written. Output: `pc_eval_neigh_supp_results`.

## Amendment A (execution environment), recorded 2026-09-29T02:02Z, before any held-out Neigh order existed

The PC run had not been started. To avoid any dependence on the Windows port, the run is executed on the Linux
machine used for the synthetic experiments (2 vCPUs, one thread per run), with the pinned Linux VOLT build from the
artifact's own route (`scripts/setup_volt.sh`: commit 31ea4914 + `volt_reproducibility.patch`; binary SHA-256
recorded in the environment file). This build reproduces all 20 stored artifact Neigh ranks of the four reused public
graphs exactly (tests/TEST_REPORT.md). Graphs are the npz caches written by the frozen PC run and the frozen orders
are the PC's saved orders (copied read-only; the dense-edge SHA-256 and all saved works are checked against
result.json). Everything else is unchanged. Consequences: (i) Q1a, the gap and the Neigh orders are
machine-independent; (ii) in Q1b the T+ILS budget is DOVE's time measured on this machine, so budgets are matched
within the machine as the protocol requires, but absolute times differ from the PC; (iii) kernel timings are measured
on this machine. Part V (validation of the Windows executable) is not needed for this build and is not run. The
Windows package remains available for an optional replication on the PC.
