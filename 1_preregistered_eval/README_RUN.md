# pc_eval: how to run the pre-registered evaluation on the Windows PC

This package fills the last experimental tables of the paper (Table 4: 32-graph suite; Table 5: pre-registered
held-out graphs, and the numbers quoted in Section 6.2). The protocol is fixed in `PREREGISTRATION.md` (do not
edit it: `run_eval.py` checks its SHA-256 and refuses to start the held-out part if it changed).

Everything runs in the existing venv (Python 3.11.9, numpy 2.1.0, scipy 1.15.2, numba 0.61.0, networkx 3.4.2).
No native DLL is built; the only optional native program is VOLT (see the end).

## 0. Copy the package

Copy the whole `pc_eval` folder unchanged (as a zip or folder copy, not through an editor) to
`D:\Work\Triangle Ordering\pc_eval`.

## 1. Open PowerShell and activate the venv

```powershell
cd "D:\Work\Triangle Ordering"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass     # only if Activate.ps1 is blocked
.\.venv\Scripts\Activate.ps1
cd "D:\Work\Triangle Ordering\pc_eval"
python -c "import numpy, scipy, numba, networkx; print(numpy.__version__, scipy.__version__, numba.__version__, networkx.__version__)"
```

Expected output: `2.1.0 1.15.2 0.61.0 3.4.2`.

Before the long run: plug in the PC, select the High performance power plan, stop other heavy programs, and
prevent sleep (`powercfg /change standby-timeout-ac 0`). DOVE's time sets T+ILS's budget on each graph, and both
run in the same process, so background load affects both methods. Keeping the machine quiet keeps the timings clean.

To keep the commands short, set these variables once per PowerShell window:

```powershell
$ART  = "D:\Work\Triangle Ordering\audit_workspace\DOVE_AISTATS_2027_code_and_data"
$DATA = "D:\Work\Triangle Ordering\pc_eval_data"
$OUT  = "D:\Work\Triangle Ordering\pc_eval_results"
```

## 2. Smoke test (about 2-4 minutes, no held-out data)

```powershell
python run_eval.py --artifact $ART --data $DATA --out $OUT --part both --quick
```

This compiles the numba kernels (about 15-30 s, once; cached in `pc_eval\.numba_cache`). It then runs Part A on
the three smallest development graphs (it downloads SNAP's `facebook.tar.gz`, about 1 MB) and Part B on three
synthetic networkx graphs. It never downloads a held-out graph. The results go to `$OUT\quick` and are refused by
`make_tables.py`. The log should end with `run_eval finished`, and each graph should report `verification PASSED`.

## 3. Full run

Without VOLT:

```powershell
python run_eval.py --artifact $ART --data $DATA --out $OUT --part both
```

With VOLT built in WSL (optional, see section 6), replacing `<user>`:

```powershell
python run_eval.py --artifact $ART --data $DATA --out $OUT --part both --volt "wsl:/home/<user>/volt/volt"
```

- Part A (the 32-graph suite) runs first, then Part B (the ten held-out graphs). The pre-registration hash is
  checked before any held-out file is downloaded.
- Progress goes to the console and to `$OUT\run_log.txt`. Results are written after every graph
  (`$OUT\partA\<graph>\result.json`, `$OUT\partB\<graph>\result.json`, with the saved orders in `orders\*.npy`).
- If the run is interrupted (Ctrl+C, reboot), re-run the same command. Completed graphs are skipped, and an
  interrupted graph is restarted from its beginning. Do not change the options between resumptions.
- Downloads use up to 5 attempts per file. If a held-out graph still fails, it is reported as missing and
  never replaced (a re-run tries the download again). Do not substitute files by hand.
- Do not run two instances on the same `$OUT` at the same time.

## 4. Write the tables into the paper

```powershell
python make_tables.py --out $OUT --paper "<folder that contains paper_content.tex>"
```

This overwrites exactly three files: `<paper>\generated\macros.tex`, `suite_rows.tex` and `heldout_rows.tex`.
It also writes `$OUT\tables_summary.json`, which holds every number used (per graph and per seed, T+ILS versus T,
the bootstrap interval and sign test of Amendment 2, and T+ILS time overshoot). `\resultspendingfalse` is written only when both parts have a final status for every
graph. When it is written, the red draft banner disappears. PDF compilation happens later and is not part of this
package: pdflatex, bibtex, then pdflatex twice on `main_anonymous.tex`.

## 5. Estimated runtime and disk use (Ryzen 5 3600, one thread)

| Step | Time | Disk |
|---|---|---|
| Smoke test | 2-4 min | < 5 MB |
| Part A (32 graphs; largest m = 2.0M) | 25-45 min | about 0.5 GB (raw text + caches + orders) |
| Part B (10 graphs; largest m = 4.7M) | 30-60 min | about 0.8 GB |
| VOLT Neigh runs (if used) | + 5-20 min | + about 0.3 GB of edge lists in `$OUT\volt` |
| Total | about 1-2 h | about 1.3-1.6 GB |

On each graph, DOVE (200 steps) takes time t, and T+ILS takes about t for each of its three seeds. DOVE
therefore dominates: about 80-90 s on Wiki-Talk and web-Google, and about 50 s on com-Youtube and roadNet-CA.
Peak memory is about 2 GB (measured at 1.6 GB on a 4.8M-edge synthetic graph).

## 6. Optional: VOLT (Neigh) for the reference

The pre-registration adds Neigh (VOLT, threshold 0.01, seeds 1-5) to the held-out reference only if VOLT is
available. `\HeldNeigh` reports "with five" or "without". For the 32-graph suite, the artifact's stored best-seed
Neigh ranks are always used, and `--volt` adds the five seeds again. The paper says "five Neigh runs", so running
with VOLT is preferable. The artifact's `scripts/setup_volt.sh` needs bash, make and GNU OpenMP. The artifact
documents WSL as the supported build route, and this route has not been tested on your machine. It is optional:
if it fails, run without `--volt`.

```powershell
wsl --install -d Ubuntu        # only if WSL is not installed yet; needs a restart
wsl
```

Inside WSL:

```bash
sudo apt update && sudo apt install -y build-essential git dos2unix
mkdir -p ~/voltsrc && cp -r "/mnt/d/Work/Triangle Ordering/audit_workspace/DOVE_AISTATS_2027_code_and_data/scripts" \
   "/mnt/d/Work/Triangle Ordering/audit_workspace/DOVE_AISTATS_2027_code_and_data/third_party" ~/voltsrc/
dos2unix ~/voltsrc/scripts/setup_volt.sh ~/voltsrc/third_party/volt_reproducibility.patch
bash ~/voltsrc/scripts/setup_volt.sh ~/volt      # clones lecfab/volt at the pinned commit, patches, builds
~/volt/volt 2>&1 | head -5                        # should print VOLT's usage
exit
```

Then, back in PowerShell, check that Windows can call it:

```powershell
wsl.exe -e /home/<user>/volt/volt
```

Use `--volt "wsl:/home/<user>/volt/volt"` in section 3. File paths are translated with `wslpath`, and each Neigh
run is logged in `$OUT\volt\...\*.log`. A failed seed is recorded and left out of the reference.

## 7. What to send back

The whole `$OUT` folder except `$OUT\volt`. The `orders` folders are needed for the audit. `$DATA` holds the SNAP
downloads, which are not redistributed.

## 8. Paper integration

1. When the full run has finished, run
   `python make_tables.py --out $OUT --paper "<folder that contains paper_content.tex>"`.
   It writes the three generated files `macros.tex`, `suite_rows.tex` and `heldout_rows.tex` into that folder's
   `generated\` subfolder. It refuses to run if the paper's current `generated\macros.tex` defines a macro that
   make_tables does not produce.
2. If the paper lives somewhere else (for example on another machine), copy those three files into that paper
   folder's `generated\` subfolder unchanged.
3. Recompile: `pdflatex main_anonymous`, `bibtex main_anonymous`, then `pdflatex main_anonymous` twice (a third
   pdflatex pass may be needed if LaTeX reports "Label(s) may have changed"). Then check two things:
   - The red "Draft" banner is gone. `\resultspendingfalse` is written only when both parts have a final status
     for every graph.
   - `pdftotext main_anonymous.pdf - | findstr "??"` prints nothing.
4. The p-values use `$p=\SuiteTIKvsDoveP$`. When p < 0.001, the macro is "<0.001", so the text reads "p = <0.001".
   If you prefer "p < 0.001", edit that one sentence in the paper.
