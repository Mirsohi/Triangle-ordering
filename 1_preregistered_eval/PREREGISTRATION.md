# Pre-registration: held-out evaluation (Part B) and rebased 32-graph suite (Part A)

- Written (UTC): 2026-09-28T05:58:39Z
- Written before any evaluation code of this package existed and before any held-out graph was downloaded
  by this project. The SHA-256 of this file is recorded in `PREREGISTRATION.sha256`. `run_eval.py` refuses to
  start Part B if the hash of this file differs (see Section 3.13).
- Paper: "Ordering Vertices for Exact Triangle Listing: Concave Degree Geometry, Certified Rounding, and
  Random-Graph Limits" (AISTATS 2027 submission), sources `paper_content.tex` and `appendix.tex`.

## 1. Appendix F of the paper (verbatim LaTeX source)

```latex
\section{PRE-REGISTERED HELD-OUT EVALUATION}\label{app:prereg}
The protocol was frozen and hashed before any held-out graph was downloaded.
\paragraph{Graphs.} Email-Eu-core, CollegeMsg, LastFM-Asia, Deezer-Europe, Facebook page--page, Amazon0302, Wiki-Talk, com-Youtube, web-Google and roadNet-CA, all from SNAP \citep{leskovec2014snap}. None of them, nor any other snapshot of the same networks, was used earlier. The first two columns of each file are read, comments and headers are skipped, and the common preprocessing is applied: remove self-loops, symmetrize, deduplicate and remap non-isolated vertices densely. A graph that fails to download or parse is reported as missing and never replaced.
\paragraph{Methods and metrics.} The reference, T, \DOVE and \TILS are as above; \TILS receives \DOVE's measured time on each graph and uses seeds 1--3. Neigh enters the reference only if the VOLT binary is available on the evaluation machine, and this is reported. We report per-graph reduction relative to $R^\star$, the mean, median, improved/tied/regressed counts, the geometric-mean work ratio, and paired wins/ties/losses of \TILS against \DOVE and T. Exact work and triangle counts are recomputed for every saved order. Graphs with more than 20M edges would be skipped for optimization; none of the listed graphs is that large.
```

## 2. The method definitions that Appendix F refers to ("as above"), verbatim

Section 6 (Protocol):

```latex
All work values are recomputed independently from exported orders, and triangle counts are checked to be order-invariant. The \emph{reference} $R^\star$ for each graph is the lowest-$\Lam$ order among Natural, a seeded Random order, ascending Degree, core ordering, smallest-last degeneracy \citep{matula1983smallest}, Split, Check, and (public graphs) five Neigh runs of \citet{lecuyer2023tailored}, \emph{each in both directions}. [...] Every method starts from $R^\star$, keeps an exact incumbent, and is charged all of its time: \DOVE runs its default 200-step schedule, and \TILS receives the same wall-clock time on each graph. Improvement is $100(\Lam_{R^\star}-\Lam)/\Lam_{R^\star}$.
```

Algorithm 1 (T+ILS):

```latex
\Require graph $G$, reference order $\pi_0$, time budget $B$, $(p,k,S)$
\State $\pi^\star\gets\pi_0$ \Comment{exact incumbent, never worse than $\pi_0$}
\Repeat\ $x\gets\dout_{\pi^\star}$; $a\gets\partial F_0(x)$; $\pi\gets\mathrm{sort}(a)$ \Comment{Thm.~\ref{thm:rounding}}
\State \textbf{if} $\Lam(\pi)<\Lam(\pi^\star)$ \textbf{then} $\pi^\star\gets\pi$
\Until{no strict decrease, or 10\% of $B$ used}
\State $\pi\gets\pi^\star$; $c\gets0$
\While{time $<B$ minus the reserved final pass}
 \State draw 256 moves ($u$ next to a neighbor $v$); apply if $\Delta<0$, or if $\Delta=0$ w.p.\ $p$
 \State \textbf{if} $\Lam(\pi)<\Lam(\pi^\star)$ \textbf{then} $\pi^\star\gets\pi$; $c\gets0$ \textbf{else} $c\gets c+1$
 \State \textbf{if} $c=S$ \textbf{then} $\pi\gets\pi^\star$ with $k$ random moves; $c\gets0$
\EndWhile
\State final tangent pass on $\pi^\star$ (lines 2--4, run to convergence)
\State \Return $\pi^\star$
```

Appendix E (Methods and protocol), verbatim excerpts:

```latex
\paragraph{Reference orders.} Natural is the input order after dense remapping; Random uses seed 7; Degree is ascending by degree with identifier ties; core ordering sorts by core number; smallest-last repeatedly removes a vertex of minimum remaining degree \citep{matula1983smallest}. These last two differ, and smallest-last is the stronger baseline for $\Cpp$ and $k$-clique listing. Split and Check follow \citet{lecuyer2023tailored}. Neigh uses the published VOLT implementation with threshold 0.01 and seeds 1--5. Every candidate is evaluated in both directions, and $R^\star$ is the minimum exact $\Lam$.
\paragraph{\DOVE.} Scores start at $s_v=2r_v/(n-1)-1$ for the reference rank $r_v$. Adam (learning rate $0.08$, moments $(0.9,0.999)$) minimizes $J/m$ for 200 steps with $\tau_t=0.5(0.04/0.5)^{t/200}$ and $\epsilon_t=\max(0.01,\tau_t^2)$. Scores are standardized after each step, and the exact work of the sorted scores is checked every five steps to keep the best incumbent.
\paragraph{Tangent refinement and \TILS.} T iterates Theorem~\ref{thm:rounding} with the $F_0$ supergradient and identifier tie-breaking until work stops strictly decreasing, for at most 100 rounds. ILS proposes a uniformly random edge, one of its endpoints $u$, and a side (before or after the other endpoint $v$), and moves $u$ to that position. Only edges between $u$ and vertices it passes change direction, and the work change is computed exactly from edges incident to vertices whose forward degree changes. Proposals come in chunks of 256. Moves with negative delta are accepted, zero-delta moves with probability $p$, and after $S$ consecutive chunks without improvement the state resets to the incumbent and applies $k$ random moves. [...] $(1,10)$ was selected. \TILS spends at most 10\% of its budget on the initial T phase and ends with a charged T pass.
```

Section 4.2: "(p,k,S) = (1,10,50)". Graphs and timing (Appendix E): "The 32-graph and held-out evaluations ran on a
Windows desktop (AMD Ryzen 5 3600; Python 3.11.9, NumPy 2.1, SciPy 1.15, Numba 0.61)."

## 3. Concrete details fixed by this pre-registration

### 3.1 Held-out graphs (Part B): exactly these ten, in this processing order

| # | Paper name | URL | File read |
|---|---|---|---|
| 1 | Email-Eu-core | https://snap.stanford.edu/data/email-Eu-core.txt.gz | gzip stream |
| 2 | CollegeMsg | https://snap.stanford.edu/data/CollegeMsg.txt.gz | gzip stream (columns SRC DST TIMESTAMP; first two read) |
| 3 | LastFM-Asia | https://snap.stanford.edu/data/lastfm_asia.zip | zip member with basename `lastfm_asia_edges.csv` |
| 4 | Deezer-Europe | https://snap.stanford.edu/data/deezer_europe.zip | zip member with basename `deezer_europe_edges.csv` |
| 5 | Facebook page-page | https://snap.stanford.edu/data/facebook_large.zip | zip member with basename `musae_facebook_edges.csv` |
| 6 | Amazon0302 | https://snap.stanford.edu/data/amazon0302.txt.gz | gzip stream |
| 7 | Wiki-Talk | https://snap.stanford.edu/data/wiki-Talk.txt.gz | gzip stream |
| 8 | com-Youtube | https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz | gzip stream |
| 9 | web-Google | https://snap.stanford.edu/data/web-Google.txt.gz | gzip stream |
| 10 | roadNet-CA | https://snap.stanford.edu/data/roadNet-CA.txt.gz | gzip stream |

Table labels use the paper names above (`Facebook page-page` is typeset `Facebook page--page`).
None of these names occurs in the artifact's development, validation, audit or revision results; the runner
asserts that no stored Neigh rank file exists for any of them.

### 3.2 Download
- HTTP GET with User-Agent `pc_eval/1.0`, up to 5 attempts per file, waits of 5, 10, 20, 40 s between attempts,
  written through a temporary file and renamed atomically. A zip must contain exactly one member with the given
  basename (directories ignored). The SHA-256 and size of every downloaded file are recorded in the results
  (they cannot be pre-registered because the files have not been downloaded).
- If the run is resumed, a graph previously recorded as missing because of a download failure is attempted again
  (with the same URL); every failed attempt stays in the log. No other source or mirror is ever used.

### 3.3 Parsing and preprocessing (identical to the artifact's `dove.graph.load_edgelist`)
- Text is decoded as UTF-8 with replacement of invalid bytes. Each line is stripped; empty lines and lines starting
  with `#` or `%` are skipped; commas are replaced by spaces and the line is split on whitespace; lines with fewer
  than two fields are skipped; the first two fields are parsed with Python `int()`, and a line whose first two
  fields do not both parse (e.g. a CSV header) is skipped. Further columns are ignored.
- Preprocessing (as `dove.graph.graph_from_edges`): drop self-loops; replace each pair by (min, max); deduplicate;
  the vertex set is the set of ids occurring in at least one remaining edge, remapped to 0..n-1 in increasing
  original id; edges are stored sorted. The package uses a vectorized implementation whose output
  (edges, original ids, adjacency) is tested to be identical to `dove.graph.graph_from_edges`.

### 3.4 Exclusion rule
- A graph is **missing** if, after the retries above, the download fails, the archive member is absent or not
  unique, reading/decompression raises an error, or the loader finds no usable edge. A missing graph keeps its row
  in Table 3 with the word "missing", is excluded from all means/medians/counts, and is never replaced.
  `\HeldCount` states how many of the ten were completed.
- A graph with more than 20,000,000 undirected edges after preprocessing is skipped for optimization (reported with
  n, m and no method results). None is expected.
- A run that crashes inside a graph (e.g. out of memory) is resumed from the start of that graph; if it fails three
  times the graph is reported as missing with the error.

### 3.5 Reference R* (both parts)
- Candidates, in this fixed listing order: Natural (`arange(n)`), Random (`dove.orderings.random_order(g, 7)`),
  Degree (`dove.orderings.degree_order`, ascending, identifier ties), Core (`dove.orderings.degeneracy_order`, the
  artifact's core ordering), SmallestLast (Matula-Beck peeling: repeatedly remove the remaining vertex with the
  smallest (current remaining degree, id); order = removal order), Split (`dove.orderings.split_order`), Check
  (`dove.orderings.check_order`), then Neigh runs (below). Every candidate is also evaluated reversed; the
  forward version is listed immediately before its reversal.
- Neigh, Part B: only if `--volt <exe>` is given. Seeds 1-5, threshold 0.01, command line exactly as in the
  artifact's `experiments/run_volt_benchmark.py` (`<volt> <dense edge list> -o neigh -n 0.01 --seed s
  --rank-output <file> --order-only -p 1`), input = the dense preprocessed edge list; the rank file must be a
  permutation of 0..n-1; order = `argsort(rank, kind="stable")`. A seed whose run fails is left out and reported.
  `\HeldNeigh` states whether Neigh entered the reference ("with five" / "without" / partial).
- Neigh, Part A: the stored best-seed ranks `results/final_{development,validation,heldout}/ranks/<graph>.neigh.seed-<s>.rank`
  of the artifact (order = `argsort(rank)`), whose exact work must equal `neigh_best_work` in the artifact's
  `per_graph.csv` (this checks that vertex ids match). If `--volt` is given, seeds 1-5 are also rerun as in Part B
  and added as candidates.
- R* = the candidate with minimum exact Λ; ties broken by the listing order. Λ is computed with
  `dove.orderings.candidate_work`. Candidate construction time is not charged to any method (all methods receive
  R*).

### 3.6 T (tangent refinement)
From R*: repeat at most 100 rounds: x = forward degrees of the current order; a = twice the canonical F0
supergradient (2 to the endpoint with smaller x, 0 to the larger, 1 to each on ties); the candidate order sorts
vertices by ascending (a_v, v); if its Λ is strictly smaller, it becomes the current order, otherwise stop.
Output: the last strictly improving order (R* if none). Not budgeted; its time is recorded.

### 3.7 DOVE
`dove.relaxation.optimize_order(g, steps=200, learning_rate=0.08, initial_temperature=0.5,
final_temperature=0.04, eval_every=5, seed=7, initial_order=R*, initial_name="Rstar",
guard_with_builtins=False, smoothing_floor=0.01)` from the artifact's `src/dove` (the artifact's default
schedule). The measured time t_DOVE is the wall-clock (`time.perf_counter`) around this call. The runner refuses
to run if the artifact's `src/dove/*.py` differ from the SHA-256 values recorded in the package, unless explicitly
overridden (logged).

### 3.8 T+ILS
Budget B = t_DOVE on the same graph, measured in the same process immediately before. Seeds 1, 2, 3, run
sequentially, each with budget B. The clock starts before any T+ILS data structure is built, and everything
until the final order is returned is charged:
1. Initial T phase from R* (rules of 3.6). The first round always runs; a further round starts only if
   (elapsed + duration of the previous round) <= 0.1 B. Result: incumbent π*.
2. Reserved final-pass time r = min(0.5 B, 3 t1), where t1 is the measured duration of the first T round.
3. ILS from π = π*, c = 0, while elapsed < B - r: a chunk of 256 proposals. Each proposal draws a uniformly
   random edge e, a uniformly random endpoint u of e (v is the other endpoint) and a uniformly random side
   (immediately before or immediately after v), from the SplitMix64 stream of the tested `insert_ils.cpp`
   (state = seed * 0x9E3779B97F4A7C15 + 777; `below(k) = next() mod k`; uniform = (next() >> 11) * 2^-53; for a
   zero-delta proposal one uniform is drawn and the move is accepted if it is < p). Δ < 0: accept; Δ = 0: accept
   with probability p = 1; Δ > 0: reject. After each chunk: if Λ(π) < Λ(π*) then π* ← π and c ← 0, else
   c ← c + 1; if c = S = 50 then π ← π* followed by k = 10 random moves (drawn as proposals, applied regardless
   of Δ) and c ← 0. The time check happens between batches of whole chunks; a batch is sized to take at most
   about 1% of B (at least one chunk), so the chunk logic is independent of the batching.
4. Final T pass on π* (rules of 3.6, to convergence, at most 100 rounds), always executed; its time is charged.
   The total time and its excess over B are reported.
Implementation: move semantics identical to `insert_ils.cpp` ("u removed and reinserted immediately before/after
v"; exact Δ from edges incident to u and to the vertices whose forward degree changes), with order-maintenance
keys instead of integer-rank shifting, verified against the C++ code on identical proposal streams.

### 3.9 Execution
- One process, graphs sequential in the order of 3.1 (Part B) or manifest order (Part A), one thread
  (`OMP/OPENBLAS/MKL/NUMBA_NUM_THREADS=1`). Per graph: load → R* → T → DOVE → T+ILS seeds 1, 2, 3 → verification.
- Numba functions are compiled and warmed on a toy graph before any timed step; compile time is excluded.
- Part A runs before Part B when `--part both` is used.

### 3.10 Verification of every saved order
Saved orders per graph: R*, T, DOVE, T+ILS seeds 1-3. Each is written to disk, read back, checked to be a
permutation, and its Λ is recomputed (a) with `dove.orderings.candidate_work` and (b) with an independent
Numba adaptive-forward lister that performs the candidate membership tests (its probe count must equal Λ). Both
must equal the value reported by the method, and the lister's triangle count must be identical for all orders of
the graph. `\SuiteTriCheck` / `\HeldTriCheck` report the number of orders and graphs that passed.

### 3.11 Metrics
- Improvement of a method on a graph: 100 (Λ_R* − Λ)/Λ_R* (0 if Λ_R* = 0).
- T+ILS per graph: mean and sample SD (n−1) over seeds 1-3 of the improvement; graph-level value = seed mean;
  seed-mean work W̄ = mean of the three works.
- Over completed graphs: arithmetic mean and median of per-graph improvements; improved/tied/regressed of T+ILS
  = number of graphs with W̄ < Λ_R*, = Λ_R*, > Λ_R*; geometric-mean work ratio = exp(mean log(W̄/Λ_R*))
  (graphs with Λ_R* = 0 excluded); paired wins/ties/losses of T+ILS against DOVE and against T: W̄ < W, = W, > W
  with W the other method's work. Per-seed versions are stored in the JSON output.
- `\SuiteRefDrop`: mean over the suite graphs of 100 (Λ_fwd − Λ_R*)/Λ_fwd, where Λ_fwd is the minimum over the
  same candidate set evaluated in the forward (stored) direction only ("reversal alone").
- Table 2 rows: per split (Development 13, Validation 11, Audit 8) and All: number of completed graphs, mean
  improvement of DOVE, T, T+ILS, and T+ILS improved/tied/regressed.
- Table 3 rows: one per pre-registered graph: n, m, Λ_R*, T, DOVE, T+ILS mean±SD; missing graphs say "missing";
  a final Mean row over completed graphs.
- Formatting: percentages with 2 decimals; ratios with 4 decimals; n, m and work as integers below 1000,
  otherwise with k (one decimal) or m (two decimals for n and m, three for work).

### 3.12 Part A (rebased 32-graph suite)
The 13 development (`data/development_benchmark/manifest.csv`), 11 validation (`data/real_benchmark/manifest.csv`)
and 8 audit (`data/heldout_benchmark/manifest.csv`) graphs of the artifact, downloaded from the URLs used by the
artifact's `scripts/prepare_data.py`, `prepare_benchmark_data.py` and `prepare_heldout_data.py`, loaded with the
same parser and preprocessing, and checked against the artifact (compressed SHA-256 and decompressed size where the
manifest records them; n and m against `per_graph.csv`; stored Neigh work against `neigh_best_work`). A graph that
fails these checks is reported, not silently used. Methods, seeds, budgets, verification and metrics are those of
Part B.

### 3.13 Integrity
- `PREREGISTRATION.sha256` holds the SHA-256 of this file's bytes. Part B aborts before any download if the hash
  differs. If only line endings differ (CRLF introduced by copying), the hash of the LF-normalised content is
  compared, and the run proceeds only if it matches; this is logged.
- Quick/smoke runs (`--quick`) never download or touch held-out graphs and are written to a separate directory
  that `make_tables.py` refuses to use for the paper.
- No change to graphs, methods, parameters, budgets, seeds or metrics after this timestamp. Any amendment would be
  a new dated section below and would change the hash.

## 4. Amendment 1 — 2026-09-28T06:28:45Z (UTC), before any held-out graph was downloaded

Reason: a scale benchmark on a SYNTHETIC Chung-Lu graph of held-out size (n = 1.63M, m = 4.78M; no held-out or
suite data involved) showed that the final T pass after ILS can need many rounds (65 rounds, 9 s), while the reserve
of 3.2 (3 x first-round time, 0.4 s) assumed only a few; T+ILS then exceeded B by 6%. The amendment makes the
reserve a better estimate of a converged T pass, so that T+ILS stays closer to DOVE's time. Nothing else changes.

- 3.8 step 2 is replaced by: reserved final-pass time r = min(0.5 B, max(3 t1, t_T)), where t1 is the measured
  duration of the first T round and t_T the measured duration of the whole initial T phase (step 1). The final pass
  still always runs to convergence (at most 100 rounds); total time and excess over B remain reported per run, and
  the mean/maximum excess is reported with the results.
- Implementation details made explicit (no change of method): a VOLT binary built inside WSL (the artifact's
  supported build route) may be given as `--volt wsl:<linux path>`; it is then run through `wsl.exe -e` with the
  same arguments (file paths translated by `wslpath`). A Neigh run that does not finish within 4 hours counts as a
  failed seed.

## 5. Amendment 2 — 2026-09-28T06:51:18Z (UTC), before any held-out graph was downloaded

Requested by the coordinator to align with the final-audited paper. It adds two statistics and changes how some
macros are worded. Graphs, methods, budgets, seeds and all existing metrics are unchanged. Amendment 1 stands.

- **Bootstrap interval.** A 95% percentile bootstrap interval, over graphs, for the mean per-graph improvement of
  T+ILS (the per-graph value is the seed mean from 3.11). Graphs are resampled with replacement 10,000 times
  (`numpy.random.default_rng(0)`, `integers(0, G, size=(10000, G))`). The statistic is the mean of each
  resample, and the limits are `numpy.percentile(..., [2.5, 97.5])` with default linear interpolation. Graphs
  with Λ_R* = 0 count as 0%. It is reported as `[a, b]\%` with 2 decimals (`\SuiteTIKCI`, `\HeldTIKCI`).
- **Sign test.** A two-sided exact sign test on the per-graph paired comparison of T+ILS (seed-mean work) against
  DOVE, with ties excluded. With w wins and l losses and N = w + l, p = min(1, 2 Σ_{i ≤ min(w,l)} C(N,i) 2^−N),
  and p = 1 when N = 0. The value is given to 2 significant digits, or as "<0.001" (`\SuiteTIKvsDoveP`,
  `\HeldTIKvsDoveP`).
- **Scope.** Both statistics are computed for Part A (all completed suite graphs; 32 if none is missing) and for
  Part B (completed held-out graphs).
- **Macro wording (presentation only).**
  - `\HeldNeigh` is a noun phrase completing "Neigh runs in the reference: …": "five seeded runs, both
    directions", "none (VOLT not available)", or a stated partial count.
  - `\SuiteTriCheck` and `\HeldTriCheck` complete "Exact-work and triangle-count checks: …": "passed for all N
    saved orders", or "failed for k of N (see log)".
