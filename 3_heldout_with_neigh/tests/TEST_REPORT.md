# Test report (sandbox, 2026-09-28)

Mock frozen run: pc_eval `run_eval.py --local-npz` on Bitcoin-Alpha and P2P-Gnutella05 with an artifact copy
without stored Neigh ranks (so R* has no Neigh), giving Part A and Part B results.

| Test | Result |
|---|---|
| Full run, Linux VOLT (pinned + reproducibility patch) | exit 0; Part V 10/10 stored artifact ranks identical; Part N R*N = Neigh on both graphs (gaps 2.26%, 10.02%); 30 orders verified |
| Full run, volt.exe under Wine (path conversion wrapper) | exit 0; Part V 10/10 identical; all 20 held-out Neigh orders identical to the Linux run |
| VOLT always fails (exit 3) | exit 2; `\NSCheck` and `\NSHeldCountText` name both graphs (neigh_failed); Part V volt_failed reported |
| Rerun of the failed output with a working VOLT | failed graphs retried (old record kept as supp_result_superseded_neigh_failed.json); exit 0 |
| Rerun of a finished output | all graphs skipped; exit 0 |
| volt.exe standalone vs Linux build | byte-identical ranks (CRLF stripped), n = 3,000 and 120,000, seeds 1-5 |

## After the checker audit (fixes applied, rerun)

| Test | Result |
|---|---|
| Full run, Linux VOLT, no-Neigh mock | exit 0; both capped rules reported (3 of 6 runs over budget: 1 ILS branch, 2 tangent branch); timing macros from paired ratios |
| VOLT failing seed 3 only | 2 WARNING lines; `\NSNeighSeeds` = 8 of 10 |
| Mock whose frozen R* already contains the stored Neigh ranks | fresh Neigh ties the stored ranks on all 4 graphs, R*N = R*, frozen runs reused and marked with a dagger; Part V 20/20 identical; exit 0 |
| Part N only (`--parts N`) | Part V graphs reported "not run" (exit 2), as intended |
| .cmd | VOLT check now requires `--rank-output` in the help text and a zero exit code; the smoke step requires exit code exactly 2; negative exit codes go to the error branch |
