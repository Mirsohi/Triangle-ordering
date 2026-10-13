@echo off
REM POST-HOC analysis for review round 1 (SECONDARY, NOT PRE-REGISTERED).
REM Run it ONLY AFTER RUN_EVAL.cmd has printed DONE. It reads the finished results and never changes them.
REM Double-click this file. It keeps its own window open; do not close it until it says DONE.
REM Re-running is safe: finished graphs are skipped; graphs that raised an error are retried.

setlocal
set ROOT=D:\Work\Triangle Ordering
set ART=%ROOT%\audit_workspace\DOVE_AISTATS_2027_code_and_data
set PCEVAL=%ROOT%\pc_eval
set DATA=%ROOT%\pc_eval_data
set RES=%ROOT%\pc_eval_results
set OUT=%ROOT%\pc_eval_posthoc_results
set LOG=%ROOT%\pc_eval_posthoc_console.log
REM keep numba/python caches inside this package, never inside pc_eval
set NUMBA_CACHE_DIR=%ROOT%\pc_eval_posthoc\.numba_cache
set PYTHONDONTWRITEBYTECODE=1

cd /d "%ROOT%" || goto :fail
call "%ROOT%\.venv\Scripts\activate.bat" || goto :fail
cd /d "%ROOT%\pc_eval_posthoc" || goto :fail

echo [%date% %time%] Checking packages...
python -c "import numpy, scipy, numba; print(numpy.__version__, scipy.__version__, numba.__version__)" || goto :fail

echo [%date% %time%] Step 1/3: smoke test on one small suite graph (1-2 min), written to %OUT%_smoke
REM The smoke run analyses only Ego-414, so its table step reports every other graph as "not run" and exits 2;
REM that is expected here. Only exit codes >= 3 or a missing PASSED line for Ego-414 count as a smoke failure.
python posthoc_eval.py --pc-eval "%PCEVAL%" --artifact "%ART%" --results "%RES%" --data "%DATA%" --out "%OUT%_smoke" --parts A --only Ego-414 >> "%LOG%" 2>&1
if errorlevel 3 goto :fail
findstr /C:"Ego-414: posthoc verification PASSED" "%OUT%_smoke\posthoc_log.txt" >nul || goto :fail
echo [%date% %time%] Smoke test passed.

echo [%date% %time%] Step 2/3: full post-hoc run over all completed graphs (about 2-4 h, roughly 2.2x the main run). Progress: %OUT%\posthoc_log.txt
python posthoc_eval.py --pc-eval "%PCEVAL%" --artifact "%ART%" --results "%RES%" --data "%DATA%" --out "%OUT%" >> "%LOG%" 2>&1
if errorlevel 2 goto :partial
if errorlevel 1 goto :fail

echo [%date% %time%] Step 3/3: rewriting the post-hoc tables (also done at the end of step 2)
python make_posthoc_tables.py --pc-eval "%PCEVAL%" --results "%RES%" --out "%OUT%" >> "%LOG%" 2>&1
if errorlevel 2 goto :partial
if errorlevel 1 goto :fail

echo.
echo [%date% %time%] DONE. Post-hoc results: %OUT%   Console log: %LOG%
echo Send back the whole %OUT% folder (and %LOG%). Tell Claude the post-hoc run finished.
pause
exit /b 0

:partial
echo.
echo [%date% %time%] FINISHED WITH FAILURES: at least one graph failed the post-hoc analysis. The tables were written
echo and name the failed graphs (see %OUT%\posthoc_log.txt, line "POST-HOC FAILURES"). Re-running retries graphs
echo that raised an error. Send back %OUT% and %LOG% and tell Claude.
pause
exit /b 2

:fail
echo.
echo [%date% %time%] STOPPED WITH AN ERROR. See %LOG% (and %OUT%\posthoc_log.txt). Re-running this file resumes.
pause
exit /b 1
