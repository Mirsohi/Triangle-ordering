# pc_eval_neigh_supp: held-out run with Neigh (review round 2)

**Executed (Amendment A, see SUPP_PROTOCOL.md and PROTOCOL_HISTORY.md):** the run was done on the Linux machine with the
pinned Linux VOLT build (`volt/volt_linux`), `--parts N`, on copies of the frozen PC graph caches and orders. Results
are in `results_linux/`. The Windows route below (RUN_NEIGH_SUPP.cmd, `volt/volt.exe`) is kept for an optional
replication on the PC and was not used for the paper.

Supplementary and **not pre-registered**. The analyses were fixed in `SUPP_PROTOCOL.md` before any held-out Neigh
order existed. The frozen pre-registered results (`pc_eval_results`) and the post-hoc results
(`pc_eval_posthoc_results`) are only read. All macros start with `\NS`.

## Run

Double-click `RUN_NEIGH_SUPP.cmd` (about 1–1.5 h). It checks that `volt\volt.exe` starts, runs a smoke test on
Ego-0 and CollegeMsg, then the full run, then the tables. Output: `D:\Work\Triangle Ordering\pc_eval_neigh_supp_results`.

## What it computes

* **Part V (32 suite graphs).** For each stored artifact Neigh rank, it reruns VOLT with the same seed on the graph
  as pc_eval exports it, and checks that the rank vector is identical. This validates the Windows executable on the
  PC against the ranks the artifact produced under Linux.
* **Part N (9 held-out graphs).**
  1. Reload check against the frozen `result.json`.
  2. Neigh seeds 1–5 through pc_eval's unchanged `methods.run_volt`.
  3. R\*N = min(frozen R\*, Neigh orders in both directions).
  4. Q1a: the frozen T, DOVE and T+ILS orders are scored against R\*N.
  5. Q1b: T, DOVE and T+ILS are rerun from R\*N if it differs from R\*. Otherwise the frozen runs are reused.
  6. Every new order is verified.
  7. Kernel timing uses 21 alternating repetitions with paired intervals.

## VOLT for Windows

`volt/volt.exe` is lecfab/volt at the artifact's pinned commit `31ea4914…`, built with the artifact's
reproducibility patch. It is cross-compiled with MinGW-w64 GCC 13 (`-Ofast -fopenmp -static`) and needs no DLLs.
`volt/volt_windows_portability.patch` replaces the C library random generator with a copy of glibc's. The only
reason is that Windows' `rand()` differs (RAND_MAX 32767). The following checks were run in the sandbox, with the
Windows executable running under Wine:

* its rank files are byte-identical (apart from CRLF) to the pinned Linux build on two graphs, n = 3,000 and
  120,000, with seeds 1–5;
* it reproduces exactly the artifact's stored Neigh ranks for Bitcoin-Alpha and P2P-Gnutella05 (10 of 10);
* in a full mock run, its 20 held-out Neigh orders are identical to those of the Linux build.

## Tests (sandbox)

`tests/TEST_REPORT.md` covers:

* a mock pre-registered run without Neigh;
* the full supplementary run, with the Linux VOLT and with volt.exe under Wine;
* the failure path (a VOLT that always fails gives exit 2, and the failing graphs are named);
* resume and retry behaviour.
