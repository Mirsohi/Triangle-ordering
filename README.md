# Vertex Ordering for Exact Triangle Listing: Concave Geometry and Certified Rounding

Kourosh Mirsohi, Melika Gorgi, Erfan Hajihashemi (University of California, Irvine)

This repository contains the code, protocols and per-graph results behind the experiments. The proofs and all tables are in the appendix of the paper.

| Folder | Paper | Contents |
|---|---|---|
| `0_dove_baseline_and_kernels/` | §4, App F | DOVE baseline, reference orders, C++ exact lister, VOLT/Neigh setup (VOLT not redistributed) |
| `1_preregistered_eval/` | §6.2, App F, App G | Evaluation pipeline (`core.py`: work counter, tangent rounding, ILS; `methods.py`; `graphs.py`: SNAP download and preprocessing; `run_eval.py`; `make_tables.py`), the frozen pre-registration `PREREGISTRATION.md` with its SHA-256, and tests |
| `1_preregistered_eval_results/` | Tables 2, 5–7, 17 | Per-graph `result.json` for the 32 development graphs (`partA`) and the held-out graphs (`partB`), run plans, environment record, run log, table summary |
| `2_secondary_analyses/` + `_results/` | §6.5, App I | Secondary (not pre-registered) analyses: declared variant, kernels, scripts, per-graph CSV and summaries |
| `3_heldout_with_neigh/` | App J | Held-out reference including Neigh: protocol `SUPP_PROTOCOL.md` (hashes in `SHA256SUMS.txt`), protocol history, driver, results; VOLT build patch only (binary not included) |
| `4_exact_small_instances/` | §3, App A | Exact optima of 26 graphs with at most seven vertices by enumerating all orders, validation and geometry checks. The HiGHS study of §6.1 (49 instances) is not in this folder |

## Environment
Python 3.11, NumPy, SciPy, Numba, NetworkX, HiGHS (via SciPy). The exact versions for each run are recorded in its `env_*.json` file. Every run used one thread (`OMP/OPENBLAS/MKL/NUMBA_NUM_THREADS=1`).

## Data
All graphs are public SNAP datasets. They are not redistributed: `1_preregistered_eval/graphs.py` downloads and preprocesses them, and the preprocessing steps are in `PREREGISTRATION.md`. Saved orders (`.npy`) are not included because of their size. Rerunning the drivers regenerates them deterministically from the seeds stored in each `result.json`.

## Notes
- The hash-protected files (pre-registration, protocols and the code files listed in `SHA256SUMS.txt` or `env_*.json`) are shipped unmodified, so their recorded hashes still match. Absolute paths in logs and in some comments refer to the machines the runs were executed on.
- `*.cmd` files are the Windows launch scripts used for the runs.
