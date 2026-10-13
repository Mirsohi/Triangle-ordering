# Protocol and code history of the supplementary Neigh run

| Time (UTC) | Event | SHA-256 |
|---|---|---|
| 2026-09-28 ~23:50 | SUPP_PROTOCOL.md, after checker audit, shipped to the PC (`pc_eval_neigh_supp`) | protocol `008abc0e…`, neigh_supp.py `5ff6a7b6…`, make_supp_tables.py `baf91cd7…` |
| (never run) | PC run (`RUN_NEIGH_SUPP.cmd`) not started | — |
| 2026-09-29 02:02:09 | Amendment A appended (execution on the Linux machine with the pinned Linux VOLT build; Part V not run). Code change: a `validation` flag so the tables can be written without Part V (no change to any computed quantity) | protocol `b2584ac2…` |
| 2026-09-29 02:02:25 | Run started (first held-out Neigh order created after this time) | environment record `supp_env_2026-09-29T020225Z.json` (protocol, code, VOLT and pc_eval hashes) |
| 2026-09-29 02:27:53 | Run finished: 9 of 10 held-out graphs, 45/45 Neigh seeds, 130 orders verified, exit 0 | — |
| 2026-09-29 | Tables rewritten after staging the frozen `result.json` of the missing graph (only its "missing" label changed) | — |

Linux VOLT binary (`volt/volt_linux`): built by the artifact's `scripts/setup_volt.sh` route (commit 31ea4914 +
`third_party/volt_reproducibility.patch`), SHA-256 `fa5ea73b68a3e05013ba263f342b0b819872cd13198ab498d8600f3d04c2110e`.

Command actually run:
```
python3 neigh_supp.py --pc-eval <pc_eval> --posthoc <pc_eval_posthoc> --artifact <artifact> \
  --results <copy of pc_eval_results> --data <copy of pc_eval_data> --out <results> --volt volt/volt_linux --parts N
```
