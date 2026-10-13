# pc_eval_posthoc: post-hoc analysis for review round 1 (SECONDARY, NOT PRE-REGISTERED)

This package answers review concerns C2 and C3 on the PC. It reads the finished pre-registered run and never
changes it. Its numbers are **secondary and not pre-registered**. Report them separately from the frozen held-out
analysis (Table 5, `generated/macros.tex`). All its macros start with `\PH`.

## What it computes

For every graph whose pre-registered `result.json` has status `done` (the 32 suite graphs in Part A and the
held-out graphs in Part B):

1. **Reload check.** The graph is loaded exactly as `run_eval.py` loaded it: the npz cache
   `pc_eval_data\part<A|B>\cache\<graph>.npz` that the main run wrote, parsed by pc_eval's `graphs.load_cache` and
   `graphs.make_graphdata`. The script then checks:
   - n, m and the dense-edge SHA-256 against `result.json` (the SHA is mandatory in non-local mode: if the
     pre-registered `result.json` does not record it, the graph fails the check);
   - the exact work of the saved R*, T, DOVE and T+ILS orders against `result.json`.

   If any check fails, the graph is reported as `reload_check_failed` and nothing else is computed for it.
2. **Greedy-colouring orders (C3).** A greedy colouring is computed with vertices processed in one of two orders:
   - `SL`: reverse smallest-last, using pc_eval's `core.smallest_last`;
   - `DEG`: descending degree, identifier ties.

   The colour classes are then placed as blocks in ascending (`asc`) or descending (`desc`) colour order. Each of
   these four variants is also used reversed (`-rev`), which gives 8 orders. For each order the script reports the
   exact work, the construction time (neighbour CSR, process order, colouring and sort, timed separately for each
   variant) and the % change relative to R*. A proper colouring has no edge inside a class, so `X-desc` and
   `X-asc-rev` give identical work. The script checks this identity for every graph.
3. **ILS-T (C2).** Algorithm 1 without the initial T phase: ILS from R*, then the final tangent pass. It uses:
   - budget B = DOVE's measured time from `result.json`, and seeds 1–3;
   - pc_eval's ILS kernel, proposal stream, (p,k,S) = (1,10,50), chunk 256 and batching, all unchanged;
   - the same reserve r for the final pass as the T+ILS run with that seed (`result.json`).

   Both variants therefore stop ILS at B − r. By design, ILS-T receives t_T more ILS time than T+ILS, namely the
   time T+ILS spent on its initial T phase. That is the intended "no prefix, same B" contrast.
4. **C+ILS-T (C3).** ILS-T started from a colouring order, with that order's construction charged to B. Two starts
   are used:
   - `CILST_fixed`: the pre-declared fixed variant **DEG-desc**. It was chosen from the random-graph data before any
     PC result existed; see `PREDECLARED_VARIANT.md`. This is the deployable choice.
   - `CILST_best`: a per-graph selected start, the best of the 8 colouring orders by exact work. All 8
     constructions and evaluations are charged to B. It uses only information available before the search, so it
     is not an oracle.

   Unlike T+ILS, C+ILS-T is not guarded by R*, so it can end above R*. Its improved/tied/regressed counts relative
   to R* are reported.
5. **Verification of every new order.** This covers the 8 colouring orders and the 9 ILS-T/C+ILS-T orders. The
   ILS orders are saved to `orders\*.npy` and read back. For each order the script checks:
   - that it is a permutation;
   - that its work agrees across pc_eval `core`, `dove.candidate_work`, the probe count of pc_eval's verification
     lister, the probe count of the timing lister, and the reported value;
   - that its triangle count equals the one in `result.json`.
6. **Kernel timing with the final (two-direction) reference (C2).** A numba adaptive-forward lister with the
   semantics of the artifact's `cpp/exact_triangle_lister.cpp`:
   - forward and incoming CSR lists in (degree desc, id) local order, and a marker array for N+(u);
   - each edge handled once, scanning the smaller forward set;
   - probes = Λ.

   It lists the triangles of R*, T, DOVE, T+ILS seed 1 and ILS-T seed 1. Each order gets one untimed warm-up call,
   then 7 repetitions with the orders alternated (rotating start). Each repetition averages k consecutive listings,
   with k fixed per graph so that one R* sample takes at least 20 ms. Without this, sub-millisecond graphs are
   dominated by timer noise. The median of each order is reported, and the
   script aggregates the per-graph ratios to R* as a geometric mean. **This is a numba lister, not the C++ bitmap
   kernel of the old timing study.** Only the listing loop is timed; building the CSR is not. The ordering times
   (reference, T, DOVE, T+ILS) are copied from `result.json`.

**Overshoot caveat.** The final tangent pass after a colouring start can need more rounds than after R*. The reserve
r is sized for T+ILS, so C+ILS-T can overshoot B on the largest graphs. `\PH<G><M>OvershootPct` reports the largest
overshoot as % of B, and `\PH<G>TILSOvershootPct` gives T+ILS's own value for comparison. Do not call the comparison
equal-time if the overshoot exceeds about 2% of B.

**Failures are never silent.** Graphs are handled as follows:
- A graph that raises an error, fails its reload check, or fails verification is excluded from the aggregates and
  named in `\PH<G>CountText` / `\PH<G>NotAnalysed`.
- `\PH<G>Check` then reads "failed or not analysed on k graphs (see log)".
- `posthoc_eval.py` and `make_posthoc_tables.py` exit with code 2 after writing the tables, and `RUN_POSTHOC.cmd`
  then prints "FINISHED WITH FAILURES" instead of DONE.
- Graphs that the pre-registered run recorded as missing, skipped or excluded are not failures. They are named in
  the same macros, e.g. "9 of 10 (Facebook-page-page: missing, download failed)".
- Re-running retries graphs that raised an error. The old error file is renamed `posthoc_error_superseded_*.json`.

## How to run (Windows PC)

1. Wait until `RUN_EVAL.cmd` has printed **DONE**. The script refuses to start while any pre-registered graph lacks
   a final status. Running both at once would also disturb the main run's timings.
2. Copy this folder unchanged to `D:\Work\Triangle Ordering\pc_eval_posthoc`. It imports the unchanged
   `D:\Work\Triangle Ordering\pc_eval` modules; nothing is copied or modified. It also never writes `__pycache__` or
   a numba cache into `pc_eval`; its own cache is `pc_eval_posthoc\.numba_cache`.
3. Double-click `RUN_POSTHOC.cmd`. It runs:
   - a smoke test on `Ego-414`, written to `pc_eval_posthoc_results_smoke`;
   - the full run;
   - the table writer.

   Re-running resumes, because graphs that already have a `posthoc_result.json` are skipped.

Equivalent PowerShell commands (with the venv active, in `pc_eval_posthoc`):

```powershell
$R = "D:\Work\Triangle Ordering"
python posthoc_eval.py --pc-eval "$R\pc_eval" --artifact "$R\audit_workspace\DOVE_AISTATS_2027_code_and_data" `
  --results "$R\pc_eval_results" --data "$R\pc_eval_data" --out "$R\pc_eval_posthoc_results"
python make_posthoc_tables.py --pc-eval "$R\pc_eval" --results "$R\pc_eval_results" --out "$R\pc_eval_posthoc_results"
```

## Runtime estimate (Ryzen 5 3600, one thread)

Per graph, the package runs 9 ILS runs of length B = t_DOVE (3 methods × 3 seeds). The main run spent about 4B per
graph (DOVE plus 3 T+ILS seeds). On top of that come the colouring, verification of 17 orders and timing, which take
seconds on small graphs and about 1.5–2 min on the 4–5M-edge graphs (verification takes about 5 s per order there).

- The main run's estimate was 1–2 h, which implies ΣB ≈ 13–28 min.
- **Post-hoc total: about 2–4 h, roughly 2.2 times the main run's wall time.** Held-out part: about 45–60 min,
  dominated by Wiki-Talk, web-Google, com-Youtube and roadNet-CA at 8–13 min each.

The log prints the exact estimate, computed from the recorded B values, at the start. Peak memory is about 2 GB.

## Outputs (`pc_eval_posthoc_results`)

| File | Content |
|---|---|
| `part<A,B>\<graph>\posthoc_result.json` | everything per graph: reload checks, colouring works/times, every run with its info (proposals, reserve, overshoot, start), verification, kernel timing samples |
| `part<A,B>\<graph>\orders\{ILST,CILST_fixed,CILST_best}_seed{1,2,3}.npy` | the new orders |
| `posthoc_summary.json` | per-graph rows, aggregates and macro values |
| `posthoc_per_graph.csv` | one row per graph |
| `posthoc_macros.tex` | the `\PH…` macros below |
| `posthoc_{suite,heldout}_colour_rows.tex` | `graph & Λ_R* & SL-asc & SL-desc & DEG-asc & DEG-desc & best & t(DEG-desc) \\` (colouring columns: % reduction vs R*; negative = more work than R*; best = label of the best order; t in s) |
| `posthoc_{suite,heldout}_ils_rows.tex` | `graph & T+ILS & ILS-T & C+ILS-T fixed & C+ILS-T best \\` (% vs R*; the last three mean±SD over seeds; T+ILS is the pre-registered value) |
| `posthoc_{suite,heldout}_timing_rows.tex` | `graph & R* kernel ms & T/R* & DOVE/R* & T+ILS/R* & t_T & t_DOVE & t_T+ILS \\` (kernel-time ratios of medians; ordering times in s from result.json) |
| `posthoc_log.txt`, `posthoc_env_*.json` | log; environment, SHA-256 of all pc_eval and post-hoc files, the pre-declared variant and its declaration file hash |

## Macros (`posthoc_macros.tex`)

`<G>` is one of four graph groups:
- `Suite`: the 32-graph suite, Part A;
- `Held`: the held-out graphs, Part B;
- `Audit`: the 8 audit graphs of the suite, the graph set of the old timing study;
- `All`: Suite and Held together.

Percentages have 2 decimals, and a negative value means more work than R*. W/T/L compares seed-mean works.

| Macro | Meaning |
|---|---|
| `\PH<G>Count` | number of graphs analysed |
| `\PH<G>Planned` | number of graphs in the pre-registered plan for the group |
| `\PH<G>CountText` | e.g. "9 of 10 (Facebook-page-page: missing, download failed)"; "32 of 32" if all were analysed |
| `\PH<G>NotAnalysed` | "none", or the list of graphs not analysed with the reason |
| `\PH<G>ColSLascMean`, `…SLdescMean`, `…DEGascMean`, `…DEGdescMean` | mean % reduction vs R* of each colouring variant |
| `\PH<G>ColBestMean` / `\PH<G>ColFixedMean` | same for the best of the 8 orders / for DEG-desc |
| `\PH<G>ColBestBeatsRef` / `\PH<G>ColFixedBeatsRef` | "k of N" graphs where that colouring order has less work than R* |
| `\PH<G>ColTimeMax` | largest construction time of any variant (s) |
| `\PH<G>ILSTMean`, `\PH<G>ILSTMedian` | ILS-T mean / median % reduction vs R* (per-graph seed means) |
| `\PH<G>ILSTITR` | ILS-T improved/tied/regressed vs R* |
| `\PH<G>ILSTvsTILS` | W/T/L of ILS-T against the pre-registered T+ILS |
| `\PH<G>ILSTMinusTILS` | mean over graphs of (ILS-T − T+ILS) % reduction, in points |
| `\PH<G>ILSTvsDove` | W/T/L of ILS-T against DOVE |
| `\PH<G>ILSTOvershoot` | largest time over B of any ILS-T run (s) |
| `\PH<G>ILSTOvershootPct` | the same as % of B |
| `\PH<G>TILSOvershootPct` | largest overshoot of the pre-registered T+ILS runs, % of B (from result.json) |
| `\PH<G>CILSTFixed{Mean,Median,ITR,vsTILS,MinusTILS,vsDove,Overshoot,OvershootPct}` | the same for C+ILS-T from DEG-desc |
| `\PH<G>CILSTFixed{vsILST,MinusILST}` | C+ILS-T (fixed) against ILS-T |
| `\PH<G>CILSTBest{…}` | the same for C+ILS-T from the best of 8 |
| `\PH<G>TILSMean`, `\PH<G>DoveMean` | pre-registered T+ILS / DOVE means recomputed from result.json (consistency check; must equal `\SuiteTIKMean` etc.) |
| `\PH<G>TimeRatio{T,Dove,TILS,ILST}` | geometric mean over graphs of median kernel time / R*'s (4 decimals; numba lister) |
| `\PH<G>WorkRatio{T,Dove,TILS,ILST}` | geometric mean of work / Λ_R* for the same timed orders |
| `\PH<G>Faster{T,Dove,TILS,ILST}` | "k of N" graphs where that order's kernel time is below R*'s |
| `\PH<G>KernelMsMedian` | median over graphs of R*'s kernel time (ms) |
| `\PH<G>DoveSecMedian` | median DOVE (ordering) time (s) |
| `\PH<G>Check` | "passed for all N post-hoc orders", or "failed or not analysed on k graphs (see log)" when a post-hoc failure occurred (pre-registered missing graphs are not failures; see `CountText`) |
| `\PHFixedVariant` | `DEG--desc` |

Every macro is written as `\providecommand{\X}{}\renewcommand{\X}{…}`, so the file can be `\input` anywhere.

## Tests done in the sandbox (see `/home/claude/review1/NOTES.md`)

- `tests/test_kernels.py`: 65 graphs, 0 mismatches. It checks that:
  - the colouring equals the historical `kc.greedy_colors` / `kc.color_order`, and every colouring is proper;
  - each reversal pair has equal work;
  - the timing lister's probes equal Λ from pc_eval core and from dove, and its triangle count equals that of
    pc_eval's verification lister (715 order checks).
- End-to-end mock:
  - `run_eval.py` of pc_eval, unchanged, was run with `--local-npz` on the 4 public case graphs, Parts A and B;
  - `posthoc_eval.py` was run on those results; 136 orders were verified.
- `tests/test_cache_path.py`: the non-local loading path. It uses npz caches written by `graphs.save_cache` and the
  dense-edge SHA-256 in the provenance. A wrong SHA is refused, and so is an absent SHA. The run exits with code 2,
  and the Check macros report the failures.
- `tests/test_failures.py`: three scenarios, all passing:
  - a pre-registered missing held-out graph: exit 0, "4 of 5 (Facebook-page-page: missing, download failed)";
  - a raised exception in one graph: exit 2, the Check macro reports it;
  - a retry after the fix: exit 0, and the stale error file is renamed.
